"""
Implementation of Multi-Query Attention for memory-efficient attention computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention module for memory-efficient attention computation.
    
    This implementation uses a single key and value head for multiple query heads,
    reducing memory usage for KV cache.
    
    Args:
        hidden_size (int): Size of hidden dimension
        num_heads (int): Number of attention heads
        kv_heads (int, optional): Number of key/value heads (default: 1)
        head_dim (int, optional): Dimension of each attention head
        dropout (float, optional): Dropout probability
        causal (bool, optional): Whether to apply causal mask
    """
    
    def __init__(
        self,
        hidden_size,
        num_heads,
        kv_heads=1,
        head_dim=None,
        dropout=0.0,
        causal=True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.dropout = dropout
        self.causal = causal
        
        # Check if dimensions are compatible
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size {hidden_size} is not divisible by num_heads {num_heads}"
            )
        
        # Check if kv_heads is valid
        if self.num_heads % self.kv_heads != 0:
            raise ValueError(
                f"num_heads {num_heads} must be divisible by kv_heads {kv_heads}"
            )
        
        self.scaling = self.head_dim ** -0.5
        
        # Query projection
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        
        # Key and value projections (reduced number of heads)
        self.k_proj = nn.Linear(hidden_size, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_heads * self.head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
    
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False):
        """
        Forward pass with Multi-Query Attention.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask (torch.Tensor, optional): Attention mask of shape [batch_size, 1, 1, seq_len]
            past_key_value (tuple, optional): Cached key and value tensors
            use_cache (bool, optional): Whether to use cache for key and value
            
        Returns:
            tuple: (output, (key, value)) if use_cache=True, otherwise just output
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project query, key, value
        query = self.q_proj(hidden_states)  # [batch_size, seq_len, num_heads * head_dim]
        key = self.k_proj(hidden_states)    # [batch_size, seq_len, kv_heads * head_dim]
        value = self.v_proj(hidden_states)  # [batch_size, seq_len, kv_heads * head_dim]
        
        # Reshape query to [batch_size, seq_len, num_heads, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Reshape key and value to [batch_size, seq_len, kv_heads, head_dim]
        key = key.view(batch_size, seq_len, self.kv_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.kv_heads, self.head_dim)
        
        # Handle cached key and value
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=1)
            value = torch.cat([past_value, value], dim=1)
        
        # Save key and value for future use if needed
        if use_cache:
            present = (key, value)
        else:
            present = None
        
        # Transpose to [batch_size, num_heads/kv_heads, seq_len, head_dim]
        query = query.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        key = key.transpose(1, 2)      # [batch_size, kv_heads, seq_len, head_dim]
        value = value.transpose(1, 2)  # [batch_size, kv_heads, seq_len, head_dim]
        
        # Compute attention
        # For multi-query attention, we need to repeat key and value
        if self.kv_heads < self.num_heads:
            # Calculate repeat factor
            repeat_factor = self.num_heads // self.kv_heads
            
            # Repeat key and value
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)
        
        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
        
        # Apply causal mask if needed
        if self.causal:
            # Create causal mask
            # [seq_len, seq_len]
            causal_mask = torch.triu(
                torch.ones(seq_len, key.size(2), dtype=torch.bool, device=query.device),
                diagonal=1
            )
            
            # Apply causal mask
            attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply dropout
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Compute output
        output = torch.matmul(attn_weights, value)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Transpose and reshape
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, num_heads * head_dim]
        
        # Apply output projection
        output = self.o_proj(output)  # [batch_size, seq_len, hidden_size]
        
        if use_cache:
            return output, present
        else:
            return output
    
    @classmethod
    def from_standard_attention(cls, attention_module, kv_heads=1):
        """
        Create a MultiQueryAttention module from a standard attention module.
        
        Args:
            attention_module (nn.Module): Standard attention module
            kv_heads (int, optional): Number of key/value heads
            
        Returns:
            MultiQueryAttention: MultiQueryAttention module
        """
        # Extract parameters from standard attention module
        hidden_size = attention_module.embed_dim if hasattr(attention_module, "embed_dim") else attention_module.hidden_size
        num_heads = attention_module.num_heads if hasattr(attention_module, "num_heads") else attention_module.num_attention_heads
        head_dim = attention_module.head_dim if hasattr(attention_module, "head_dim") else hidden_size // num_heads
        dropout = attention_module.dropout if hasattr(attention_module, "dropout") else 0.0
        causal = attention_module.is_causal if hasattr(attention_module, "is_causal") else True
        
        # Create MultiQueryAttention module
        mqa = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dropout=dropout,
            causal=causal
        )
        
        # Copy weights if possible
        if hasattr(attention_module, "q_proj") and hasattr(attention_module, "k_proj") and hasattr(attention_module, "v_proj"):
            # Copy query projection weights
            mqa.q_proj.weight.data.copy_(attention_module.q_proj.weight.data)
            
            # For key and value, we need to average across heads
            if kv_heads < num_heads:
                # Reshape to [num_heads, head_dim, hidden_size]
                k_weight = attention_module.k_proj.weight.view(num_heads, head_dim, hidden_size)
                v_weight = attention_module.v_proj.weight.view(num_heads, head_dim, hidden_size)
                
                # Average across heads
                k_weight = k_weight.mean(dim=0, keepdim=True).repeat(kv_heads, 1, 1)
                v_weight = v_weight.mean(dim=0, keepdim=True).repeat(kv_heads, 1, 1)
                
                # Reshape back
                mqa.k_proj.weight.data.copy_(k_weight.reshape(kv_heads * head_dim, hidden_size))
                mqa.v_proj.weight.data.copy_(v_weight.reshape(kv_heads * head_dim, hidden_size))
            else:
                # Just copy the weights
                mqa.k_proj.weight.data.copy_(attention_module.k_proj.weight.data)
                mqa.v_proj.weight.data.copy_(attention_module.v_proj.weight.data)
            
            # Copy output projection weights
            mqa.o_proj.weight.data.copy_(attention_module.o_proj.weight.data)
        
        return mqa