"""
Implementation of LSH (Locality-Sensitive Hashing) Attention for memory-efficient attention computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSHAttention(nn.Module):
    """
    Attention module using Locality-Sensitive Hashing (LSH) for efficient attention computation.
    
    This implementation is inspired by the Reformer paper (Kitaev et al., 2020).
    
    Args:
        hidden_size (int): Size of hidden dimension
        num_heads (int): Number of attention heads
        num_hashes (int, optional): Number of hash functions
        bucket_size (int, optional): Size of hash buckets
        head_dim (int, optional): Dimension of each attention head
        dropout (float, optional): Dropout probability
        causal (bool, optional): Whether to apply causal mask
    """
    
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_hashes=4,
        bucket_size=64,
        head_dim=None,
        dropout=0.0,
        causal=True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.num_hashes = num_hashes
        self.bucket_size = bucket_size
        self.dropout = dropout
        self.causal = causal
        
        # Check if dimensions are compatible
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size {hidden_size} is not divisible by num_heads {num_heads}"
            )
        
        self.scaling = self.head_dim ** -0.5
        
        # Random projection matrices for hashing
        self.projection_matrices = nn.Parameter(
            torch.randn(self.num_heads, self.num_hashes, self.head_dim),
            requires_grad=False
        )
    
    def forward(self, query, key, value, attention_mask=None):
        """
        Forward pass with LSH Attention.
        
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
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Hash-based chunking
        if q_len == k_len:  # Self-attention case
            # Compute hashes for query/key
            hashes = self._compute_hashes(query)  # [batch_size, num_heads, num_hashes, seq_len]
            
            # Sort by hash bucket
            sorted_hashes, hash_indices = self._sort_by_hash(hashes)
            
            # Chunk by hash bucket
            chunks = self._chunk_by_hash(sorted_hashes, hash_indices, query, key, value)
            
            # Compute attention within chunks
            chunk_outputs = self._compute_chunk_attention(chunks, attention_mask)
            
            # Unchunk and reorder
            output = self._unchunk_and_reorder(chunk_outputs, hash_indices)
        else:
            # For cross-attention, fall back to standard attention with memory optimizations
            output = self._compute_cross_attention(query, key, value, attention_mask)
        
        # Reshape to [batch_size, seq_len, hidden_size]
        output = output.transpose(1, 2).reshape(batch_size, q_len, self.hidden_size)
        
        return output
    
    def _compute_hashes(self, x):
        """
        Compute locality-sensitive hashes for the input.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            torch.Tensor: Hashes of shape [batch_size, num_heads, num_hashes, seq_len]
        """
        batch_size, num_heads, seq_len, head_dim = x.size()
        
        # Compute random projections
        # [batch_size, num_heads, seq_len, head_dim] x [num_heads, num_hashes, head_dim]
        # -> [batch_size, num_heads, seq_len, num_hashes]
        projections = torch.einsum("bnsd,nhd->bnsh", x, self.projection_matrices)
        
        # Convert to binary buckets
        # We use the sign of the projection as the hash
        hashes = torch.sign(projections)  # [batch_size, num_heads, seq_len, num_hashes]
        
        # Transpose to [batch_size, num_heads, num_hashes, seq_len]
        hashes = hashes.transpose(2, 3)
        
        return hashes
    
    def _sort_by_hash(self, hashes):
        """
        Sort sequences by hash bucket.
        
        Args:
            hashes (torch.Tensor): Hashes of shape [batch_size, num_heads, num_hashes, seq_len]
            
        Returns:
            tuple: (sorted_hashes, hash_indices)
        """
        batch_size, num_heads, num_hashes, seq_len = hashes.size()
        
        # Convert binary hash buckets to integers
        # We use a simple conversion: hash_bucket = sum(2^i * hash[i])
        bucket_range = torch.arange(num_hashes, device=hashes.device)
        bucket_range = 2 ** bucket_range.float()
        
        # [batch_size, num_heads, num_hashes, seq_len] -> [batch_size, num_heads, seq_len]
        buckets = torch.einsum("bnhs,h->bns", hashes.float(), bucket_range)
        
        # Sort by bucket
        _, hash_indices = torch.sort(buckets, dim=-1)  # [batch_size, num_heads, seq_len]
        
        # Expand hash_indices for gather operation
        hash_indices = hash_indices.unsqueeze(2).expand(-1, -1, num_hashes, -1)
        
        # Gather sorted hashes
        sorted_hashes = torch.gather(hashes, 3, hash_indices)
        
        return sorted_hashes, hash_indices
    
    def _chunk_by_hash(self, sorted_hashes, hash_indices, query, key, value):
        """
        Chunk sequences by hash bucket.
        
        Args:
            sorted_hashes (torch.Tensor): Sorted hashes
            hash_indices (torch.Tensor): Hash indices
            query (torch.Tensor): Query tensor
            key (torch.Tensor): Key tensor
            value (torch.Tensor): Value tensor
            
        Returns:
            tuple: (chunked_query, chunked_key, chunked_value)
        """
        batch_size, num_heads, num_hashes, seq_len = sorted_hashes.size()
        
        # Compute chunk boundaries
        # We want to find where the hash bucket changes
        # [batch_size, num_heads, num_hashes, seq_len] -> [batch_size, num_heads, num_hashes, seq_len-1]
        bucket_changes = (sorted_hashes[..., 1:] != sorted_hashes[..., :-1]).any(dim=2)
        
        # Add start and end boundaries
        # [batch_size, num_heads, seq_len-1] -> [batch_size, num_heads, seq_len+1]
        padding = torch.ones(batch_size, num_heads, 1, device=bucket_changes.device, dtype=torch.bool)
        bucket_changes = torch.cat([padding, bucket_changes, padding], dim=2)
        
        # Find chunk boundaries
        chunk_boundaries = torch.nonzero(bucket_changes, as_tuple=True)
        
        # Reshape for gather operation
        hash_indices = hash_indices.squeeze(2)  # [batch_size, num_heads, seq_len]
        
        # Gather query, key, value by hash indices
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        chunked_query = torch.gather(query, 2, hash_indices.unsqueeze(-1).expand(-1, -1, -1, query.size(-1)))
        chunked_key = torch.gather(key, 2, hash_indices.unsqueeze(-1).expand(-1, -1, -1, key.size(-1)))
        chunked_value = torch.gather(value, 2, hash_indices.unsqueeze(-1).expand(-1, -1, -1, value.size(-1)))
        
        return (chunked_query, chunked_key, chunked_value, chunk_boundaries)
    
    def _compute_chunk_attention(self, chunks, attention_mask=None):
        """
        Compute attention within chunks.
        
        Args:
            chunks (tuple): (chunked_query, chunked_key, chunked_value, chunk_boundaries)
            attention_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Chunk outputs
        """
        chunked_query, chunked_key, chunked_value, chunk_boundaries = chunks
        batch_size, num_heads, seq_len, head_dim = chunked_query.size()
        
        # Process chunks
        chunk_outputs = []
        
        # Group by batch and head
        for b in range(batch_size):
            for h in range(num_heads):
                # Get chunk boundaries for this batch and head
                b_h_boundaries = [(i, j) for i, j, k in zip(*chunk_boundaries) if i == b and j == h]
                
                # Process each chunk
                for i in range(len(b_h_boundaries) - 1):
                    start, end = b_h_boundaries[i][1], b_h_boundaries[i+1][1]
                    
                    # Extract chunk
                    q_chunk = chunked_query[b, h, start:end]
                    k_chunk = chunked_key[b, h, start:end]
                    v_chunk = chunked_value[b, h, start:end]
                    
                    # Compute attention scores
                    scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) * self.scaling
                    
                    # Apply causal mask if needed
                    if self.causal:
                        causal_mask = torch.triu(
                            torch.ones(end-start, end-start, dtype=torch.bool, device=scores.device),
                            diagonal=1
                        )
                        scores.masked_fill_(causal_mask, float("-inf"))
                    
                    # Apply attention mask if provided
                    if attention_mask is not None:
                        # Extract and apply mask for this chunk
                        # This is complex and depends on the specific format of attention_mask
                        pass
                    
                    # Apply softmax
                    attn_weights = F.softmax(scores, dim=-1)
                    
                    # Apply dropout
                    if self.dropout > 0 and self.training:
                        attn_weights = F.dropout(attn_weights, p=self.dropout)
                    
                    # Compute output
                    chunk_output = torch.matmul(attn_weights, v_chunk)
                    
                    # Store output
                    chunk_outputs.append((b, h, start, end, chunk_output))
        
        # Combine chunk outputs
        output = torch.zeros_like(chunked_query)
        
        for b, h, start, end, chunk_output in chunk_outputs:
            output[b, h, start:end] = chunk_output
        
        return output
    
    def _unchunk_and_reorder(self, chunk_outputs, hash_indices):
        """
        Unchunk and reorder outputs.
        
        Args:
            chunk_outputs (torch.Tensor): Chunk outputs
            hash_indices (torch.Tensor): Hash indices
            
        Returns:
            torch.Tensor: Reordered outputs
        """
        batch_size, num_heads = hash_indices.size()[:2]
        
        # Create reverse indices for reordering
        reverse_indices = torch.zeros_like(hash_indices)
        
        # For each batch and head
        for b in range(batch_size):
            for h in range(num_heads):
                # Create reverse mapping
                reverse_indices[b, h].scatter_(
                    0, hash_indices[b, h], torch.arange(hash_indices.size(2), device=hash_indices.device)
                )
        
        # Reshape for gather operation
        reverse_indices = reverse_indices.unsqueeze(-1).expand(-1, -1, -1, chunk_outputs.size(-1))
        
        # Gather outputs by reverse indices
        reordered_outputs = torch.gather(chunk_outputs, 2, reverse_indices)
        
        return reordered_outputs
    
    def _compute_cross_attention(self, query, key, value, attention_mask=None):
        """
        Compute cross-attention with memory optimizations.
        
        Args:
            query (torch.Tensor): Query tensor
            key (torch.Tensor): Key tensor
            value (torch.Tensor): Value tensor
            attention_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Output tensor
        """
        batch_size, num_heads, q_len, head_dim = query.size()
        _, _, k_len, _ = key.size()
        
        # Compute attention scores in chunks to reduce memory usage
        chunk_size = 128  # Adjust based on available memory
        
        outputs = []
        for i in range(0, q_len, chunk_size):
            end_idx = min(i + chunk_size, q_len)
            q_chunk = query[:, :, i:end_idx, :]
            
            # Compute attention scores for this chunk
            attn_weights = torch.matmul(q_chunk, key.transpose(-1, -2)) * self.scaling
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, i:end_idx]
            
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
        
        return output
    
    @classmethod
    def from_standard_attention(cls, attention_module):
        """
        Create an LSHAttention module from a standard attention module.
        
        Args:
            attention_module (nn.Module): Standard attention module
            
        Returns:
            LSHAttention: LSHAttention module
        """
        # Extract parameters from standard attention module
        hidden_size = attention_module.embed_dim if hasattr(attention_module, "embed_dim") else attention_module.hidden_size
        num_heads = attention_module.num_heads if hasattr(attention_module, "num_heads") else attention_module.num_attention_heads
        head_dim = attention_module.head_dim if hasattr(attention_module, "head_dim") else hidden_size // num_heads
        dropout = attention_module.dropout if hasattr(attention_module, "dropout") else 0.0
        causal = attention_module.is_causal if hasattr(attention_module, "is_causal") else True
        
        # Create LSHAttention module
        lsh_attention = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            causal=causal
        )
        
        return lsh_attention