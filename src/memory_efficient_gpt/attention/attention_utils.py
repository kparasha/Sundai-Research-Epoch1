"""
Utility functions for attention optimization.
"""

import torch
import torch.nn as nn
from .flash_attention import FlashAttention

def convert_to_flash_attention(model):
    """
    Convert a model to use FlashAttention.
    
    This function replaces standard attention modules with FlashAttention modules.
    
    Args:
        model (nn.Module): Model to convert
        
    Returns:
        nn.Module: Converted model
    """
    # Identify attention modules to replace
    attention_patterns = [
        "attention",
        "attn",
        "self_attn",
        "SelfAttention"
    ]
    
    # Track replaced modules
    replaced_modules = 0
    
    # Replace attention modules
    for name, module in list(model.named_modules()):
        # Skip if not an attention module
        if not any(pattern in name.lower() for pattern in attention_patterns):
            continue
        
        # Skip if already using FlashAttention
        if isinstance(module, FlashAttention):
            continue
        
        # Check if module has query, key, value projections
        has_qkv = hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj")
        
        # Check if module has query, key, value methods
        has_qkv_methods = hasattr(module, "get_query_layer") and hasattr(module, "get_key_layer") and hasattr(module, "get_value_layer")
        
        if has_qkv or has_qkv_methods:
            try:
                # Get parent module and child name
                parent_name, child_name = _get_parent_and_child_name(name)
                parent = _get_module_by_name(model, parent_name)
                
                # Create FlashAttention module
                flash_attn = FlashAttention.from_standard_attention(module)
                
                # Replace attention module
                setattr(parent, child_name, flash_attn)
                
                replaced_modules += 1
            except Exception as e:
                print(f"Failed to replace attention module {name}: {e}")
    
    print(f"Replaced {replaced_modules} attention modules with FlashAttention")
    
    return model

def _get_parent_and_child_name(name):
    """
    Split a module name into parent and child components.
    
    Args:
        name (str): Full module name
        
    Returns:
        tuple: (parent_name, child_name)
    """
    if '.' in name:
        parent_name, child_name = name.rsplit('.', 1)
        return parent_name, child_name
    else:
        return '', name

def _get_module_by_name(model, name):
    """
    Get a module by its name.
    
    Args:
        model (nn.Module): Model to search
        name (str): Module name
        
    Returns:
        nn.Module: Found module
    """
    if not name:
        return model
    
    for n in name.split('.'):
        model = getattr(model, n)
    
    return model

def use_flash_attention_if_available(model):
    """
    Use FlashAttention if available, otherwise keep the original model.
    
    Args:
        model (nn.Module): Model to convert
        
    Returns:
        nn.Module: Converted model or original model
    """
    try:
        from flash_attn import flash_attn_func
        return convert_to_flash_attention(model)
    except ImportError:
        try:
            # Check if PyTorch 2.0+ scaled_dot_product_attention is available
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                # Use BetterTransformer if available
                try:
                    from transformers import BetterTransformer
                    return BetterTransformer.transform(model)
                except ImportError:
                    pass
        except:
            pass
        
        print("FlashAttention not available. Using original model.")
        return model