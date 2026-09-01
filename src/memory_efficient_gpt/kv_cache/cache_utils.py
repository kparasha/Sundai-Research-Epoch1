"""
Utility functions for KV cache management.
"""

import torch
import torch.nn as nn
from .paged_kv_cache import PagedKVCache

def add_paged_kv_cache(model):
    """
    Add PagedKVCache to a model.
    
    This function modifies the model's forward pass to use PagedKVCache.
    
    Args:
        model (nn.Module): Model to modify
        
    Returns:
        nn.Module: Modified model
    """
    # Check if model has a generate method
    if not hasattr(model, "generate"):
        print("Model does not have a generate method. Cannot add PagedKVCache.")
        return model
    
    # Get model parameters
    config = getattr(model, "config", None)
    if config is None:
        print("Model does not have a config attribute. Cannot add PagedKVCache.")
        return model
    
    # Extract parameters from config
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "n_layer", None)
    
    num_heads = getattr(config, "num_attention_heads", None)
    if num_heads is None:
        num_heads = getattr(config, "n_head", None)
    
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "n_embd", None)
    
    # Calculate head dimension
    head_dim = hidden_size // num_heads if hidden_size is not None and num_heads is not None else None
    
    if num_layers is None or num_heads is None or head_dim is None:
        print("Could not extract model parameters. Cannot add PagedKVCache.")
        return model
    
    # Create PagedKVCache
    paged_kv_cache = PagedKVCache(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=16  # Default page size
    )
    
    # Store original generate method
    original_generate = model.generate
    
    # Define new generate method with PagedKVCache
    def generate_with_paged_kv_cache(input_ids, attention_mask=None, max_length=None, **kwargs):
        """
        Generate with PagedKVCache.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            max_length (int, optional): Maximum length of generated sequence
            **kwargs: Additional arguments for generation
            
        Returns:
            torch.Tensor: Generated token IDs
        """
        # Get batch size and sequence length
        batch_size, seq_len = input_ids.size()
        
        # Set default max_length if not provided
        if max_length is None:
            max_length = seq_len + 50  # Default: input length + 50 tokens
        
        # Allocate PagedKVCache
        paged_kv_cache.allocate(batch_size=batch_size, max_seq_len=max_length)
        
        # Store original forward methods of attention layers
        original_forwards = {}
        
        # Find attention layers
        attention_layers = []
        for name, module in model.named_modules():
            if "attention" in name.lower() and hasattr(module, "forward"):
                attention_layers.append((name, module))
        
        # Modify forward methods to use PagedKVCache
        for i, (name, layer) in enumerate(attention_layers):
            original_forwards[name] = layer.forward
            
            # Define new forward method with PagedKVCache
            def make_forward_with_cache(layer_idx, orig_forward):
                def forward_with_cache(hidden_states, attention_mask=None, **kwargs):
                    # Check if we're in generation mode
                    if kwargs.get("use_cache", False):
                        # Get position information
                        position_ids = kwargs.get("position_ids", None)
                        past_key_value = kwargs.get("past_key_value", None)
                        
                        if past_key_value is None:
                            # First forward pass
                            start_pos = 0
                        else:
                            # Subsequent forward pass
                            start_pos = past_key_value[0].size(-2)
                        
                        # Call original forward
                        outputs = orig_forward(
                            hidden_states,
                            attention_mask=attention_mask,
                            **kwargs
                        )
                        
                        # Extract key and value from outputs
                        if isinstance(outputs, tuple) and len(outputs) > 1:
                            # Standard case: outputs = (attn_output, ..., (key, value))
                            if isinstance(outputs[-1], tuple) and len(outputs[-1]) == 2:
                                key, value = outputs[-1]
                            else:
                                # Try to find key and value in the outputs
                                key = value = None
                                for item in outputs:
                                    if isinstance(item, tuple) and len(item) == 2:
                                        key, value = item
                                        break
                        
                        if key is not None and value is not None:
                            # Update PagedKVCache
                            paged_kv_cache.update(
                                layer_idx=layer_idx,
                                key=key,
                                value=value,
                                start_pos=start_pos
                            )
                            
                            # Return outputs with updated past_key_value
                            if isinstance(outputs, tuple):
                                # Find the index of past_key_value in outputs
                                for i, item in enumerate(outputs):
                                    if isinstance(item, tuple) and len(item) == 2:
                                        # Replace with a dummy tensor to indicate we're using PagedKVCache
                                        dummy = torch.tensor([layer_idx, start_pos], device=hidden_states.device)
                                        outputs_list = list(outputs)
                                        outputs_list[i] = (dummy, dummy)
                                        return tuple(outputs_list)
                        
                        return outputs
                    else:
                        # Not in generation mode, use original forward
                        return orig_forward(hidden_states, attention_mask=attention_mask, **kwargs)
                
                return forward_with_cache
            
            # Replace forward method
            layer.forward = make_forward_with_cache(i, original_forwards[name])
        
        # Call original generate method
        try:
            outputs = original_generate(input_ids, attention_mask=attention_mask, max_length=max_length, **kwargs)
        finally:
            # Restore original forward methods
            for name, layer in attention_layers:
                if name in original_forwards:
                    layer.forward = original_forwards[name]
            
            # Free PagedKVCache
            paged_kv_cache.free()
        
        return outputs
    
    # Replace generate method
    model.generate = generate_with_paged_kv_cache
    
    # Store PagedKVCache in model for reference
    model.paged_kv_cache = paged_kv_cache
    
    return model

def use_paged_kv_cache_if_available(model):
    """
    Use PagedKVCache if available, otherwise keep the original model.
    
    Args:
        model (nn.Module): Model to modify
        
    Returns:
        nn.Module: Modified model or original model
    """
    try:
        # Try to import vLLM
        import vllm
        print("vLLM is available. Using vLLM for PagedAttention.")
        
        # TODO: Add vLLM integration
        return model
    except ImportError:
        try:
            # Use our custom PagedKVCache implementation
            return add_paged_kv_cache(model)
        except Exception as e:
            print(f"Failed to add PagedKVCache: {e}")
            return model