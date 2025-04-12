"""
Base implementation of KV cache.
"""

import torch
import torch.nn as nn

class KVCache:
    """
    Base class for KV cache implementations.
    
    Args:
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
    """
    
    def __init__(self, num_layers, num_heads, head_dim):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Initialize cache
        self.k_cache = None
        self.v_cache = None
    
    def allocate(self, batch_size, max_seq_len):
        """
        Allocate memory for KV cache.
        
        Args:
            batch_size (int): Batch size
            max_seq_len (int): Maximum sequence length
        """
        device = self._get_device()
        dtype = self._get_dtype()
        
        # Allocate key cache: [batch_size, num_layers, num_heads, max_seq_len, head_dim]
        self.k_cache = torch.zeros(
            batch_size, self.num_layers, self.num_heads, max_seq_len, self.head_dim,
            device=device, dtype=dtype
        )
        
        # Allocate value cache: [batch_size, num_layers, num_heads, max_seq_len, head_dim]
        self.v_cache = torch.zeros(
            batch_size, self.num_layers, self.num_heads, max_seq_len, self.head_dim,
            device=device, dtype=dtype
        )
    
    def update(self, layer_idx, key, value, start_pos):
        """
        Update KV cache with new key-value pairs.
        
        Args:
            layer_idx (int): Layer index
            key (torch.Tensor): Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
            value (torch.Tensor): Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
            start_pos (int): Starting position in the cache
        """
        batch_size, num_heads, seq_len, head_dim = key.size()
        
        # Update key cache
        self.k_cache[:batch_size, layer_idx, :num_heads, start_pos:start_pos+seq_len, :] = key
        
        # Update value cache
        self.v_cache[:batch_size, layer_idx, :num_heads, start_pos:start_pos+seq_len, :] = value
    
    def get(self, layer_idx, batch_size, start_pos, end_pos):
        """
        Get key-value pairs from cache.
        
        Args:
            layer_idx (int): Layer index
            batch_size (int): Batch size
            start_pos (int): Starting position in the cache
            end_pos (int): Ending position in the cache
            
        Returns:
            tuple: (key, value) tensors
        """
        # Get key from cache
        key = self.k_cache[:batch_size, layer_idx, :, start_pos:end_pos, :]
        
        # Get value from cache
        value = self.v_cache[:batch_size, layer_idx, :, start_pos:end_pos, :]
        
        return key, value
    
    def free(self, batch_indices=None):
        """
        Free memory for specified batches.
        
        Args:
            batch_indices (list, optional): Indices of batches to free
        """
        if batch_indices is None:
            # Free all memory
            self.k_cache = None
            self.v_cache = None
        else:
            # Free memory for specified batches
            if self.k_cache is not None:
                self.k_cache[batch_indices] = 0
            
            if self.v_cache is not None:
                self.v_cache[batch_indices] = 0
    
    def _get_device(self):
        """
        Get device for cache allocation.
        
        Returns:
            torch.device: Device for cache
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _get_dtype(self):
        """
        Get data type for cache allocation.
        
        Returns:
            torch.dtype: Data type for cache
        """
        return torch.float16 if torch.cuda.is_available() else torch.float32