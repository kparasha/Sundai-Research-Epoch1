"""
Tests for KV cache module.
"""

import unittest
import torch
from memory_efficient_gpt.kv_cache import KVCache, PagedKVCache, add_paged_kv_cache

class TestKVCache(unittest.TestCase):
    """Test cases for KV cache module."""
    
    def test_kv_cache_allocation(self):
        """Test KV cache allocation."""
        # Create KV cache
        num_layers = 12
        num_heads = 12
        head_dim = 64
        
        kv_cache = KVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim
        )
        
        # Allocate cache
        batch_size = 2
        max_seq_len = 128
        
        kv_cache.allocate(batch_size, max_seq_len)
        
        # Check cache shapes
        self.assertEqual(kv_cache.k_cache.shape, (batch_size, num_layers, num_heads, max_seq_len, head_dim))
        self.assertEqual(kv_cache.v_cache.shape, (batch_size, num_layers, num_heads, max_seq_len, head_dim))
    
    def test_kv_cache_update_get(self):
        """Test KV cache update and get operations."""
        # Create KV cache
        num_layers = 12
        num_heads = 12
        head_dim = 64
        
        kv_cache = KVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim
        )
        
        # Allocate cache
        batch_size = 2
        max_seq_len = 128
        
        kv_cache.allocate(batch_size, max_seq_len)
        
        # Create key and value tensors
        layer_idx = 0
        seq_len = 16
        
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # Update cache
        start_pos = 0
        kv_cache.update(layer_idx, key, value, start_pos)
        
        # Get from cache
        retrieved_key, retrieved_value = kv_cache.get(layer_idx, batch_size, start_pos, start_pos + seq_len)
        
        # Check that retrieved tensors match original tensors
        self.assertTrue(torch.allclose(key, retrieved_key))
        self.assertTrue(torch.allclose(value, retrieved_value))
    
    def test_kv_cache_free(self):
        """Test KV cache free operation."""
        # Create KV cache
        num_layers = 12
        num_heads = 12
        head_dim = 64
        
        kv_cache = KVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim
        )
        
        # Allocate cache
        batch_size = 2
        max_seq_len = 128
        
        kv_cache.allocate(batch_size, max_seq_len)
        
        # Free cache
        kv_cache.free()
        
        # Check that cache is freed
        self.assertIsNone(kv_cache.k_cache)
        self.assertIsNone(kv_cache.v_cache)
    
    def test_paged_kv_cache_allocation(self):
        """Test PagedKVCache allocation."""
        # Create PagedKVCache
        num_layers = 12
        num_heads = 12
        head_dim = 64
        page_size = 16
        
        paged_kv_cache = PagedKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            page_size=page_size
        )
        
        # Allocate cache
        batch_size = 2
        max_seq_len = 128
        
        paged_kv_cache.allocate(batch_size, max_seq_len)
        
        # Check that page tables are initialized
        for batch_idx in range(batch_size):
            for layer_idx in range(num_layers):
                self.assertIn((batch_idx, layer_idx), paged_kv_cache.page_tables)
                self.assertIn((batch_idx, layer_idx), paged_kv_cache.block_tables)
        
        # Check that physical pages are allocated
        num_pages_per_seq = (max_seq_len + page_size - 1) // page_size
        total_pages = batch_size * num_layers * num_pages_per_seq
        
        self.assertEqual(len(paged_kv_cache.key_pages), total_pages)
        self.assertEqual(len(paged_kv_cache.value_pages), total_pages)
        self.assertEqual(len(paged_kv_cache.free_pages), total_pages)
    
    def test_paged_kv_cache_update_get(self):
        """Test PagedKVCache update and get operations."""
        # Create PagedKVCache
        num_layers = 12
        num_heads = 12
        head_dim = 64
        page_size = 16
        
        paged_kv_cache = PagedKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            page_size=page_size
        )
        
        # Allocate cache
        batch_size = 2
        max_seq_len = 128
        
        paged_kv_cache.allocate(batch_size, max_seq_len)
        
        # Create key and value tensors
        layer_idx = 0
        seq_len = 16
        
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # Update cache
        start_pos = 0
        paged_kv_cache.update(layer_idx, key, value, start_pos)
        
        # Get from cache
        retrieved_key, retrieved_value = paged_kv_cache.get(layer_idx, list(range(batch_size)), start_pos, start_pos + seq_len)
        
        # Check that retrieved tensors match original tensors
        self.assertTrue(torch.allclose(key, retrieved_key))
        self.assertTrue(torch.allclose(value, retrieved_value))
    
    def test_paged_kv_cache_free(self):
        """Test PagedKVCache free operation."""
        # Create PagedKVCache
        num_layers = 12
        num_heads = 12
        head_dim = 64
        page_size = 16
        
        paged_kv_cache = PagedKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            page_size=page_size
        )
        
        # Allocate cache
        batch_size = 2
        max_seq_len = 128
        
        paged_kv_cache.allocate(batch_size, max_seq_len)
        
        # Update cache
        layer_idx = 0
        seq_len = 16
        
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        paged_kv_cache.update(layer_idx, key, value, 0)
        
        # Free cache
        paged_kv_cache.free()
        
        # Check that page tables are cleared
        self.assertEqual(len(paged_kv_cache.page_tables), 0)
        self.assertEqual(len(paged_kv_cache.block_tables), 0)
        
        # Check that all pages are freed
        num_pages_per_seq = (max_seq_len + page_size - 1) // page_size
        total_pages = batch_size * num_layers * num_pages_per_seq
        
        self.assertEqual(len(paged_kv_cache.free_pages), total_pages)

if __name__ == "__main__":
    unittest.main()