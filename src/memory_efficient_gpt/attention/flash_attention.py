"""
Implementation of FlashAttention for memory-efficient attention computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Check if flash_attn is available
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

class FlashAttention(nn.Module):
    """
    Attention module using FlashAttention algorithm.
    
    This implementation uses the FlashAttention library if available,
    or falls back to a PyTorch implementation with memory optimizations.
    
    Args:
        hidden_size (int): Size of hidden dimension
        num_heads (int): Number of attention heads
        head_dim (int, optional): Dimension of each attention head
        dropout (float, optional): Dropout probability
        causal (bool, optional): Whether to apply causal mask
    """
    
    def __init__(
        self,
        hidden_size,
        num_heads,
        head_dim=None,
        dropout=0.0,
        causal=True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.dropout = dropout
        self.causal = causal
        
        # Check if dimensions are compatible
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size {hidden_size} is not divisible by num_heads {num_heads}"
            )
        
        self.scaling = self.head_dim ** -0.5
        
        # Check if FlashAttention is available
        self.flash_attn_available = FLASH_ATTN_AVAILABLE
        
        # For PyTorch 2.0+, check if scaled_dot_product_attention is available
        self.sdpa_available = hasattr(F, "scaled_dot_product_attention")
    
    def forward(self, query, key, value, attention_mask=None):
        """
        Forward pass with FlashAttention.
        
        Args:
            query (torch.Tensor): Query tensor of shape [batch_size, seq_len, hidden_size]
            key (torch.Tensor): Key tensor of shape [batch_size, seq_len, hidden_size]
            value (torch.Tensor): Value tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask (torch.Tensor, optional): Attention mask of shape [batch_size, 1, 1, seq_len]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, q_len, _ = query.size()
        _, k_len, _ = key.size()
        
        # Reshape query, key, value to [batch_size, seq_len, num_heads, head_dim]
        query = query.view(batch_size, q_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, k_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, k_len, self.num_heads, self.head_dim)
        
        # Use FlashAttention if available
        if self.flash_attn_available:
            # FlashAttention expects inputs in [batch_size, seq_len, num_heads, head_dim]
            # Convert attention_mask if provided
            if attention_mask is not None:
                # FlashAttention uses a float mask where 0 = keep, -inf = mask
                # Convert from [batch_size, 1, 1, seq_len] to [batch_size, seq_len]
                mask = attention_mask.squeeze(1).squeeze(1)
                # Convert 0 = mask, 1 = keep to 0 = keep, -inf = mask
                mask = (1 - mask) * -10000.0
            else:
                mask = None
            
            # Call FlashAttention
            output = flash_attn_func(
                query,
                key,
                value,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal,
                mask=mask
            )
        
        # Use PyTorch 2.0+ scaled_dot_product_attention if available
        elif self.sdpa_available:
            # PyTorch expects inputs in [batch_size, num_heads, seq_len, head_dim]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            
            # Call scaled_dot_product_attention
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal and attention_mask is None
            )
            
            # Transpose back to [batch_size, seq_len, num_heads, head_dim]
            output = output.transpose(1, 2)
        
        # Fall back to manual implementation with memory optimizations
        else:
            # Transpose to [batch_size, num_heads, seq_len, head_dim]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            
            # Compute attention scores
            # Instead of computing the full attention matrix at once,
            # we can compute it in chunks to reduce memory usage
            chunk_size = 128  # Adjust based on available memory
            
            outputs = []
            for i in range(0, q_len, chunk_size):
                end_idx = min(i + chunk_size, q_len)
                q_chunk = query[:, :, i:end_idx, :]
                
                # Compute attention scores for this chunk
                attn_weights = torch.matmul(q_chunk, key.transpose(-1, -2)) * self.scaling
                
                # Apply causal mask if needed
                if self.causal:
                    causal_mask = torch.triu(
                        torch.ones(end_idx - i, k_len, dtype=torch.bool, device=query.device),
                        diagonal=1 + i
                    )
                    attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                
                # Apply attention mask if provided
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                
                # Apply softmax
                attn_weights = F.softmax(attn_weights, dim=-1)
                
                # Apply dropout
                if self.dropout > 0 and self.training:
                    attn_weights = F.dropout(attn_weights, p=self.dropout)
                
                # Compute output for this chunk
                chunk_output = torch.matmul(attn_weights, value)
                outputs.append(chunk_output)
            
            # Concatenate chunk outputs
            output = torch.cat(outputs, dim=2)
            
            # Transpose back to [batch_size, seq_len, num_heads, head_dim]
            output = output.transpose(1, 2)
        
        # Reshape to [batch_size, seq_len, hidden_size]
        output = output.reshape(batch_size, q_len, self.hidden_size)
        
        return output
    
    @classmethod
    def from_standard_attention(cls, attention_module):
        """
        Create a FlashAttention module from a standard attention module.
        
        Args:
            attention_module (nn.Module): Standard attention module
            
        Returns:
            FlashAttention: FlashAttention module
        """
        # Extract parameters from standard attention module
        hidden_size = attention_module.embed_dim if hasattr(attention_module, "embed_dim") else attention_module.hidden_size
        num_heads = attention_module.num_heads if hasattr(attention_module, "num_heads") else attention_module.num_attention_heads
        head_dim = attention_module.head_dim if hasattr(attention_module, "head_dim") else hidden_size // num_heads
        dropout = attention_module.dropout if hasattr(attention_module, "dropout") else 0.0
        causal = attention_module.is_causal if hasattr(attention_module, "is_causal") else True
        
        # Create FlashAttention module
        flash_attention = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            causal=causal
        )
        
        return flash_attention