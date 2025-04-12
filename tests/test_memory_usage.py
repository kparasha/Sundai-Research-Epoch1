"""
Tests for memory usage of optimization techniques.
"""

import unittest
import torch
import gc
import time
from memory_efficient_gpt.quantization import QuantizedModel
from memory_efficient_gpt.attention import use_flash_attention_if_available
from memory_efficient_gpt.kv_cache import use_paged_kv_cache_if_available

def get_gpu_memory_usage():
    """Get GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

class TestMemoryUsage(unittest.TestCase):
    """Test cases for memory usage of optimization techniques."""
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_quantization_memory_savings(self):
        """Test memory savings from quantization."""
        # Skip if no GPU available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Load a small model for testing
        model_name = "gpt2"
        
        try:
            # Clear GPU memory
            gc.collect()
            torch.cuda.empty_cache()
            
            # Measure memory before loading model
            memory_before = get_gpu_memory_usage()
            
            # Load standard model
            from transformers import AutoModelForCausalLM
            standard_model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
            
            # Measure memory after loading standard model
            memory_standard = get_gpu_memory_usage() - memory_before
            
            # Clear GPU memory
            del standard_model
            gc.collect()
            torch.cuda.empty_cache()
            
            # Load quantized model (8-bit)
            quantized_model_8bit = QuantizedModel.from_pretrained(
                model_name,
                bits=8,
                device_map="auto"
            )
            
            # Measure memory after loading 8-bit model
            memory_8bit = get_gpu_memory_usage() - memory_before
            
            # Clear GPU memory
            del quantized_model_8bit
            gc.collect()
            torch.cuda.empty_cache()
            
            # Load quantized model (4-bit)
            quantized_model_4bit = QuantizedModel.from_pretrained(
                model_name,
                bits=4,
                device_map="auto"
            )
            
            # Measure memory after loading 4-bit model
            memory_4bit = get_gpu_memory_usage() - memory_before
            
            # Check memory savings
            self.assertLess(memory_8bit, memory_standard * 0.6)  # 8-bit should use less than 60% of standard memory
            self.assertLess(memory_4bit, memory_standard * 0.4)  # 4-bit should use less than 40% of standard memory
            
        except Exception as e:
            self.fail(f"Failed to test quantization memory savings: {e}")
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_flash_attention_memory_savings(self):
        """Test memory savings from FlashAttention."""
        # Skip if no GPU available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Create a simple model for testing
        class SimpleModel(torch.nn.Module):
            def __init__(self, use_flash_attention=False):
                super().__init__()
                self.embed_dim = 1024
                self.num_heads = 16
                self.seq_len = 1024
                self.use_flash_attention = use_flash_attention
                
                if use_flash_attention:
                    from memory_efficient_gpt.attention import FlashAttention
                    self.attention = FlashAttention(
                        hidden_size=self.embed_dim,
                        num_heads=self.num_heads,
                        head_dim=self.embed_dim // self.num_heads,
                        dropout=0.1,
                        causal=True
                    )
                else:
                    self.q_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
                    self.k_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
                    self.v_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
                    self.out_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
            
            def forward(self, x):
                batch_size, seq_len, _ = x.size()
                
                if self.use_flash_attention:
                    return self.attention(x, x, x)
                else:
                    q = self.q_proj(x)
                    k = self.k_proj(x)
                    v = self.v_proj(x)
                    
                    q = q.view(batch_size, seq_len, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
                    k = k.view(batch_size, seq_len, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
                    v = v.view(batch_size, seq_len, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
                    
                    # Compute attention scores
                    attn_weights = torch.matmul(q, k.transpose(-1, -2)) * (self.embed_dim // self.num_heads) ** -0.5
                    
                    # Apply causal mask
                    causal_mask = torch.triu(
                        torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
                        diagonal=1
                    )
                    attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                    
                    attn_weights = torch.softmax(attn_weights, dim=-1)
                    
                    # Compute output
                    attn_output = torch.matmul(attn_weights, v)
                    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
                    attn_output = self.out_proj(attn_output)
                    
                    return attn_output
        
        try:
            # Clear GPU memory
            gc.collect()
            torch.cuda.empty_cache()
            
            # Create input tensor
            batch_size = 2
            seq_len = 1024
            embed_dim = 1024
            
            x = torch.randn(batch_size, seq_len, embed_dim).cuda()
            
            # Measure memory with standard attention
            standard_model = SimpleModel(use_flash_attention=False).cuda()
            
            # Warm-up
            _ = standard_model(x)
            
            # Measure memory before forward pass
            memory_before = get_gpu_memory_usage()
            
            # Forward pass
            _ = standard_model(x)
            
            # Measure memory after forward pass
            memory_standard = get_gpu_memory_usage() - memory_before
            
            # Clear GPU memory
            del standard_model
            gc.collect()
            torch.cuda.empty_cache()
            
            # Measure memory with FlashAttention
            flash_model = SimpleModel(use_flash_attention=True).cuda()
            
            # Warm-up
            _ = flash_model(x)
            
            # Measure memory before forward pass
            memory_before = get_gpu_memory_usage()
            
            # Forward pass
            _ = flash_model(x)
            
            # Measure memory after forward pass
            memory_flash = get_gpu_memory_usage() - memory_before
            
            # Check memory savings
            self.assertLess(memory_flash, memory_standard * 0.8)  # FlashAttention should use less memory
            
        except Exception as e:
            self.fail(f"Failed to test FlashAttention memory savings: {e}")
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_paged_kv_cache_memory_efficiency(self):
        """Test memory efficiency of PagedKVCache."""
        # Skip if no GPU available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Create KV caches for testing
        from memory_efficient_gpt.kv_cache import KVCache, PagedKVCache
        
        num_layers = 12
        num_heads = 12
        head_dim = 64
        
        try:
            # Clear GPU memory
            gc.collect()
            torch.cuda.empty_cache()
            
            # Measure memory before allocation
            memory_before = get_gpu_memory_usage()
            
            # Allocate standard KV cache
            standard_cache = KVCache(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim
            )
            
            # Allocate cache for long sequence
            batch_size = 2
            max_seq_len = 16384  # Very long sequence
            
            standard_cache.allocate(batch_size, max_seq_len)
            
            # Measure memory after allocation
            memory_standard = get_gpu_memory_usage() - memory_before
            
            # Clear GPU memory
            del standard_cache
            gc.collect()
            torch.cuda.empty_cache()
            
            # Measure memory before allocation
            memory_before = get_gpu_memory_usage()
            
            # Allocate paged KV cache
            paged_cache = PagedKVCache(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                page_size=128
            )
            
            # Allocate cache for long sequence
            paged_cache.allocate(batch_size, max_seq_len)
            
            # Measure memory after allocation
            memory_paged = get_gpu_memory_usage() - memory_before
            
            # Check memory efficiency
            self.assertLess(memory_paged, memory_standard * 0.8)  # Paged cache should use less memory
            
            # Test memory usage with partial sequence
            # Create key and value tensors for a short sequence
            layer_idx = 0
            seq_len = 128
            
            key = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
            value = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
            
            # Update cache
            paged_cache.update(layer_idx, key, value, 0)
            
            # Measure memory after update
            memory_after_update = get_gpu_memory_usage() - memory_before
            
            # Check that memory usage is proportional to actual sequence length
            self.assertLess(memory_after_update, memory_paged * 0.2)  # Should use much less memory than full allocation
            
        except Exception as e:
            self.fail(f"Failed to test PagedKVCache memory efficiency: {e}")

if __name__ == "__main__":
    unittest.main()