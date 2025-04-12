"""
Implementation of PagedAttention for efficient KV cache management.
"""

import torch
import torch.nn as nn
import math

class PagedKVCache:
    """
    KV cache using paged memory management.
    
    This implementation is inspired by vLLM's PagedAttention, which partitions
    the KV cache into fixed-size pages to reduce memory fragmentation.
    
    Args:
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        page_size (int, optional): Number of tokens per page
    """
    
    def __init__(self, num_layers, num_heads, head_dim, page_size=16):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        
        # Initialize page tables and physical pages
        self.page_tables = {}  # Maps (batch_idx, layer_idx) to list of page indices
        self.key_pages = []    # List of physical pages for keys
        self.value_pages = []  # List of physical pages for values
        
        # Free list for page reuse
        self.free_pages = []
        
        # Block size tracking
        self.block_tables = {}  # Maps (batch_idx, layer_idx) to list of block sizes
        
        # Device and dtype
        self.device = None
        self.dtype = None
    
    def allocate(self, batch_size, max_seq_len):
        """
        Allocate memory for KV cache.
        
        Args:
            batch_size (int): Batch size
            max_seq_len (int): Maximum sequence length
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Calculate number of pages needed
        num_pages_per_seq = math.ceil(max_seq_len / self.page_size)
        total_pages = batch_size * self.num_layers * num_pages_per_seq
        
        # Allocate physical pages
        self.key_pages = [
            torch.zeros(
                self.num_heads, self.page_size, self.head_dim,
                device=self.device, dtype=self.dtype
            )
            for _ in range(total_pages)
        ]
        
        self.value_pages = [
            torch.zeros(
                self.num_heads, self.page_size, self.head_dim,
                device=self.device, dtype=self.dtype
            )
            for _ in range(total_pages)
        ]
        
        # Initialize page tables
        for batch_idx in range(batch_size):
            for layer_idx in range(self.num_layers):
                self.page_tables[(batch_idx, layer_idx)] = []
                self.block_tables[(batch_idx, layer_idx)] = []
        
        # Initialize free list
        self.free_pages = list(range(total_pages))
    
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
        
        for batch_idx in range(batch_size):
            # Get page table for this batch and layer
            page_table = self.page_tables.get((batch_idx, layer_idx), [])
            block_table = self.block_tables.get((batch_idx, layer_idx), [])
            
            # Calculate current position in the sequence
            curr_pos = start_pos
            
            # Calculate how many tokens we need to add
            remaining_tokens = seq_len
            
            # Check if we need to append to an existing page
            if page_table and block_table:
                last_page_idx = page_table[-1]
                last_block_size = block_table[-1]
                
                # If the last page is not full, append to it
                if last_block_size < self.page_size:
                    space_in_page = self.page_size - last_block_size
                    tokens_to_add = min(space_in_page, remaining_tokens)
                    
                    # Update key and value pages
                    page_offset = last_block_size
                    self.key_pages[last_page_idx][:, page_offset:page_offset+tokens_to_add, :] = \
                        key[batch_idx, :, :tokens_to_add, :]
                    
                    self.value_pages[last_page_idx][:, page_offset:page_offset+tokens_to_add, :] = \
                        value[batch_idx, :, :tokens_to_add, :]
                    
                    # Update block size
                    block_table[-1] += tokens_to_add
                    
                    # Update position and remaining tokens
                    curr_pos += tokens_to_add
                    remaining_tokens -= tokens_to_add
            
            # Allocate new pages for remaining tokens
            while remaining_tokens > 0:
                # Allocate a new page
                if not self.free_pages:
                    # No free pages available, create a new one
                    new_page_idx = len(self.key_pages)
                    self.key_pages.append(
                        torch.zeros(
                            self.num_heads, self.page_size, self.head_dim,
                            device=self.device, dtype=self.dtype
                        )
                    )
                    self.value_pages.append(
                        torch.zeros(
                            self.num_heads, self.page_size, self.head_dim,
                            device=self.device, dtype=self.dtype
                        )
                    )
                else:
                    # Use a free page
                    new_page_idx = self.free_pages.pop(0)
                
                # Add page to page table
                page_table.append(new_page_idx)
                
                # Calculate tokens to add to this page
                tokens_to_add = min(self.page_size, remaining_tokens)
                
                # Update key and value pages
                token_offset = curr_pos - start_pos
                self.key_pages[new_page_idx][:, :tokens_to_add, :] = \
                    key[batch_idx, :, token_offset:token_offset+tokens_to_add, :]
                
                self.value_pages[new_page_idx][:, :tokens_to_add, :] = \
                    value[batch_idx, :, token_offset:token_offset+tokens_to_add, :]
                
                # Update block table
                block_table.append(tokens_to_add)
                
                # Update position and remaining tokens
                curr_pos += tokens_to_add
                remaining_tokens -= tokens_to_add
            
            # Update page table and block table
            self.page_tables[(batch_idx, layer_idx)] = page_table
            self.block_tables[(batch_idx, layer_idx)] = block_table
    
    def get(self, layer_idx, batch_indices, start_pos, end_pos):
        """
        Get key-value pairs from cache.
        
        Args:
            layer_idx (int): Layer index
            batch_indices (list): Batch indices to retrieve
            start_pos (int): Starting position in the sequence
            end_pos (int): Ending position in the sequence
            
        Returns:
            tuple: (key, value) tensors of shape [len(batch_indices), num_heads, end_pos-start_pos, head_dim]
        """
        seq_len = end_pos - start_pos
        batch_size = len(batch_indices)
        
        # Allocate tensors for results
        key_result = torch.zeros(
            batch_size, self.num_heads, seq_len, self.head_dim,
            device=self.device, dtype=self.dtype
        )
        
        value_result = torch.zeros(
            batch_size, self.num_heads, seq_len, self.head_dim,
            device=self.device, dtype=self.dtype
        )
        
        for batch_idx, global_batch_idx in enumerate(batch_indices):
            # Get page table for this batch and layer
            page_table = self.page_tables.get((global_batch_idx, layer_idx), [])
            block_table = self.block_tables.get((global_batch_idx, layer_idx), [])
            
            if not page_table or not block_table:
                continue
            
            # Calculate which pages contain the requested tokens
            curr_pos = 0
            result_pos = 0
            
            for page_idx, block_size in zip(page_table, block_table):
                page_start = curr_pos
                page_end = curr_pos + block_size
                
                # Check if this page contains tokens we need
                if page_end > start_pos and page_start < end_pos:
                    # Calculate overlap
                    overlap_start = max(page_start, start_pos)
                    overlap_end = min(page_end, end_pos)
                    overlap_len = overlap_end - overlap_start
                    
                    # Calculate offsets
                    page_offset = overlap_start - page_start
                    result_offset = overlap_start - start_pos
                    
                    # Copy data from page to result
                    key_result[batch_idx, :, result_offset:result_offset+overlap_len, :] = \
                        self.key_pages[page_idx][:, page_offset:page_offset+overlap_len, :]
                    
                    value_result[batch_idx, :, result_offset:result_offset+overlap_len, :] = \
                        self.value_pages[page_idx][:, page_offset:page_offset+overlap_len, :]
                
                # Update position
                curr_pos += block_size
                
                # If we've gone past the end, we're done
                if curr_pos >= end_pos:
                    break
        
        return key_result, value_result
    
    def free(self, batch_indices=None):
        """
        Free memory for specified batches.
        
        Args:
            batch_indices (list, optional): Indices of batches to free
        """
        if batch_indices is None:
            # Free all memory
            for key in list(self.page_tables.keys()):
                batch_idx, layer_idx = key
                page_table = self.page_tables.pop(key, [])
                self.block_tables.pop(key, [])
                
                # Add pages to free list
                self.free_pages.extend(page_table)
                
                # Clear pages
                for page_idx in page_table:
                    self.key_pages[page_idx].zero_()
                    self.value_pages[page_idx].zero_()
        else:
            # Free memory for specified batches
            for batch_idx in batch_indices:
                for layer_idx in range(self.num_layers):
                    key = (batch_idx, layer_idx)
                    page_table = self.page_tables.pop(key, [])
                    self.block_tables.pop(key, [])
                    
                    # Add pages to free list
                    self.free_pages.extend(page_table)
                    
                    # Clear pages
                    for page_idx in page_table:
                        self.key_pages[page_idx].zero_()
                        self.value_pages[page_idx].zero_()
    
    def resize(self, batch_size, new_max_seq_len):
        """
        Resize the cache for a new maximum sequence length.
        
        Args:
            batch_size (int): Batch size
            new_max_seq_len (int): New maximum sequence length
        """
        # Calculate number of new pages needed
        old_num_pages = len(self.key_pages)
        new_num_pages_per_seq = math.ceil(new_max_seq_len / self.page_size)
        new_total_pages = batch_size * self.num_layers * new_num_pages_per_seq
        
        if new_total_pages > old_num_pages:
            # Allocate additional pages
            additional_pages = new_total_pages - old_num_pages
            
            for _ in range(additional_pages):
                self.key_pages.append(
                    torch.zeros(
                        self.num_heads, self.page_size, self.head_dim,
                        device=self.device, dtype=self.dtype
                    )
                )
                self.value_pages.append(
                    torch.zeros(
                        self.num_heads, self.page_size, self.head_dim,
                        device=self.device, dtype=self.dtype
                    )
                )
                
                # Add new page to free list
                self.free_pages.append(old_num_pages + _)