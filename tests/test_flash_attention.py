"""
Tests for FlashAttention module.
"""

import unittest
import torch
import torch.nn as nn
from memory_efficient_gpt.attention import FlashAttention, convert_to_flash_attention

class TestFlashAttention(unittest.TestCase):
    """Test cases for FlashAttention module."""
    
    def test_flash_attention_forward(self):
        """Test FlashAttention forward pass."""
        # Create FlashAttention module
        hidden_size = 768
        num_heads = 12
        head_dim = hidden_size // num_heads
        
        flash_attn = FlashAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=0.1,
            causal=True
        )
        
        # Create input tensors
        batch_size = 2
        seq_len = 16
        
        query = torch.randn(batch_size, seq_len, hidden_size)
        key = torch.randn(batch_size, seq_len, hidden_size)
        value = torch.randn(batch_size, seq_len, hidden_size)
        
        # Test forward pass
        output = flash_attn(query, key, value)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
    
    def test_flash_attention_causal_mask(self):
        """Test FlashAttention with causal mask."""
        # Create FlashAttention modules with and without causal mask
        hidden_size = 768
        num_heads = 12
        head_dim = hidden_size // num_heads
        
        flash_attn_causal = FlashAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            causal=True
        )
        
        flash_attn_non_causal = FlashAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            causal=False
        )
        
        # Create input tensors
        batch_size = 2
        seq_len = 16
        
        query = torch.randn(batch_size, seq_len, hidden_size)
        key = torch.randn(batch_size, seq_len, hidden_size)
        value = torch.randn(batch_size, seq_len, hidden_size)
        
        # Test forward pass with and without causal mask
        output_causal = flash_attn_causal(query, key, value)
        output_non_causal = flash_attn_non_causal(query, key, value)
        
        # Check that outputs are different
        self.assertFalse(torch.allclose(output_causal, output_non_causal))
    
    def test_convert_to_flash_attention(self):
        """Test converting a model to use FlashAttention."""
        # Create a simple model with attention
        class SimpleAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_dim = 768
                self.num_heads = 12
                self.head_dim = self.embed_dim // self.num_heads
                self.dropout = 0.1
                self.is_causal = True
                
                self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
                self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
                self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
                self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
            
            def forward(self, hidden_states, attention_mask=None):
                batch_size, seq_len, _ = hidden_states.size()
                
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
                
                q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                
                # Compute attention scores
                attn_weights = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
                
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                
                if self.is_causal:
                    causal_mask = torch.triu(
                        torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
                        diagonal=1
                    )
                    attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                
                attn_weights = torch.softmax(attn_weights, dim=-1)
                attn_weights = torch.dropout(attn_weights, p=self.dropout, train=self.training)
                
                # Compute output
                attn_output = torch.matmul(attn_weights, v)
                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
                attn_output = self.out_proj(attn_output)
                
                return attn_output, (k, v)
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = SimpleAttention()
            
            def forward(self, hidden_states, attention_mask=None):
                # Extract query, key, value from hidden_states
                batch_size, seq_len, hidden_size = hidden_states.shape
                
                # Use the same tensor for query, key, value for simplicity
                query = key = value = hidden_states
                
                # Call attention
                attn_output = self.attention(query, key, value, attention_mask)
                return attn_output
        
        # Create model
        model = SimpleModel()
        
        # Convert to FlashAttention
        converted_model = convert_to_flash_attention(model)
        
        # Check that attention module was replaced
        self.assertIsInstance(converted_model.attention, FlashAttention)
        
        # Test forward pass
        batch_size = 2
        seq_len = 16
        hidden_size = 768
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Test that forward pass works
        output = converted_model(hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))

if __name__ == "__main__":
    unittest.main()