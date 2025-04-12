"""
Implementation of model manager for inference server.
"""

import torch
import os
from ..quantization import QuantizedModel
from ..attention import use_flash_attention_if_available
from ..kv_cache import use_paged_kv_cache_if_available

class ModelManager:
    """
    Manager for loading and caching models.
    
    Args:
        device_map (str, optional): Device mapping for models
        cache_dir (str, optional): Directory for caching models
    """
    
    def __init__(self, device_map="auto", cache_dir=None):
        self.device_map = device_map
        self.cache_dir = cache_dir
        
        # Initialize model cache
        self.models = {}
    
    def load_model(self, model_name, quantization_bits=None, use_flash_attention=True, 
                  kv_cache_strategy="paged", offload_strategy=None):
        """
        Load a model with optimizations.
        
        Args:
            model_name (str): Name or path of the model
            quantization_bits (int, optional): Bits for quantization (4 or 8)
            use_flash_attention (bool, optional): Whether to use FlashAttention
            kv_cache_strategy (str, optional): Strategy for KV cache management
            offload_strategy (str, optional): Strategy for offloading
            
        Returns:
            nn.Module: Loaded model
        """
        # Check if model is already loaded
        if model_name in self.models:
            return self.models[model_name]
        
        # Load model with quantization if specified
        if quantization_bits in [4, 8]:
            model = QuantizedModel.from_pretrained(
                model_name,
                bits=quantization_bits,
                device_map=self.device_map,
                torch_dtype=torch.float16
            )
        else:
            # Load model without quantization
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device_map,
                torch_dtype=torch.float16
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Attach tokenizer to model for convenience
            model.tokenizer = tokenizer
        
        # Apply FlashAttention if specified
        if use_flash_attention:
            model = use_flash_attention_if_available(model)
        
        # Apply PagedAttention if specified
        if kv_cache_strategy == "paged":
            model = use_paged_kv_cache_if_available(model)
        
        # Apply offloading if specified
        if offload_strategy:
            model = self._apply_offloading(model, offload_strategy)
        
        # Cache model
        self.models[model_name] = model
        
        return model
    
    def unload_model(self, model_name):
        """
        Unload a model from memory.
        
        Args:
            model_name (str): Name of the model to unload
        """
        if model_name in self.models:
            # Remove model from cache
            del self.models[model_name]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_model(self, model_name):
        """
        Get a loaded model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            nn.Module: Loaded model
            
        Raises:
            ValueError: If model is not loaded
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} is not loaded")
        
        return self.models[model_name]
    
    def is_model_loaded(self, model_name):
        """
        Check if a model is loaded.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            bool: Whether the model is loaded
        """
        return model_name in self.models
    
    def get_loaded_models(self):
        """
        Get names of all loaded models.
        
        Returns:
            list: Names of loaded models
        """
        return list(self.models.keys())
    
    def _apply_offloading(self, model, offload_strategy):
        """
        Apply offloading to a model.
        
        Args:
            model (nn.Module): Model to offload
            offload_strategy (str): Offloading strategy
            
        Returns:
            nn.Module: Offloaded model
        """
        if offload_strategy == "cpu":
            try:
                # Try to use DeepSpeed for CPU offloading
                import deepspeed
                
                # Initialize DeepSpeed inference engine
                ds_engine = deepspeed.init_inference(
                    model=model,
                    mp_size=1,
                    dtype=torch.float16,
                    replace_with_kernel_inject=False,
                    replace_method="auto",
                    max_out_tokens=1024,
                    offload_device="cpu"
                )
                
                return ds_engine.module
            except ImportError:
                print("DeepSpeed not available. Using device_map for CPU offloading.")
                
                # Use device_map for CPU offloading
                from transformers.utils import infer_auto_device_map
                
                # Get model parameters
                if hasattr(model, "model"):
                    # For QuantizedModel
                    base_model = model.model
                else:
                    base_model = model
                
                # Infer device map
                device_map = infer_auto_device_map(
                    base_model,
                    max_memory={0: "4GiB", "cpu": "32GiB"},
                    no_split_module_classes=["GPTJBlock", "GPT2Block", "OPTDecoderLayer", "LlamaDecoderLayer"]
                )
                
                # Apply device map
                base_model.device_map = device_map
                
                return model
        
        elif offload_strategy == "nvme":
            try:
                # Try to use DeepSpeed for NVMe offloading
                import deepspeed
                
                # Initialize DeepSpeed inference engine
                ds_engine = deepspeed.init_inference(
                    model=model,
                    mp_size=1,
                    dtype=torch.float16,
                    replace_with_kernel_inject=False,
                    replace_method="auto",
                    max_out_tokens=1024,
                    offload_device="nvme",
                    offload_nvme_path="/tmp/nvme_offload"
                )
                
                return ds_engine.module
            except ImportError:
                print("DeepSpeed not available. NVMe offloading requires DeepSpeed.")
                return model
        
        else:
            # No offloading
            return model