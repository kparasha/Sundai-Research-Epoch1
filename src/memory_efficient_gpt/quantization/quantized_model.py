"""
Implementation of quantized models.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from .quantized_linear import QuantizedLinear
from .config import QuantizationConfig

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

class QuantizedModel:
    """
    Wrapper for quantized models.
    
    This class provides a unified interface for loading and using quantized models,
    supporting both custom quantization and bitsandbytes integration.
    
    Args:
        model_name (str): Name or path of the pre-trained model
        config (QuantizationConfig): Configuration for quantization
        device_map (str or dict): Device mapping for model placement
        torch_dtype (torch.dtype): Data type for non-quantized tensors
    """
    
    def __init__(
        self,
        model_name,
        config=None,
        device_map="auto",
        torch_dtype=torch.float16
    ):
        self.model_name = model_name
        self.config = config or QuantizationConfig()
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        
        self.model = None
        self.tokenizer = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """
        Load the model with quantization.
        """
        if self.config.use_bitsandbytes and BITSANDBYTES_AVAILABLE:
            # Use bitsandbytes for quantization
            self._load_with_bitsandbytes()
        else:
            # Use custom quantization
            self._load_with_custom_quantization()
    
    def _load_with_bitsandbytes(self):
        """
        Load the model using bitsandbytes quantization.
        """
        from transformers import AutoTokenizer, BitsAndBytesConfig
        
        # Convert our config to bitsandbytes config
        bnb_config = BitsAndBytesConfig(**self.config.to_bitsandbytes_config())
        
        # Load the model with bitsandbytes quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype
        )
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def _load_with_custom_quantization(self):
        """
        Load the model using custom quantization.
        """
        from transformers import AutoTokenizer
        
        # Load the model in full precision first
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype
        )
        
        # Quantize the model
        self._quantize_model(model)
        
        # Store the quantized model
        self.model = model
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def _quantize_model(self, model):
        """
        Apply custom quantization to the model.
        
        Args:
            model (nn.Module): Model to quantize
        """
        # Identify modules to quantize
        modules_to_quantize = []
        
        for name, module in model.named_modules():
            # Skip excluded modules
            if any(excluded in name for excluded in self.config.excluded_modules):
                continue
            
            # Check if module type is in target modules
            if any(target in module.__class__.__name__ for target in self.config.target_modules):
                modules_to_quantize.append((name, module))
        
        # Quantize identified modules
        for name, module in modules_to_quantize:
            if isinstance(module, nn.Linear):
                parent_name, child_name = self._get_parent_and_child_name(name)
                parent = self._get_module_by_name(model, parent_name)
                
                # Replace the linear module with a quantized version
                quantized_module = QuantizedLinear.from_float(
                    module,
                    bits=self.config.bits,
                    group_size=self.config.group_size,
                    symmetric=self.config.symmetric
                )
                
                setattr(parent, child_name, quantized_module)
    
    def _get_parent_and_child_name(self, name):
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
    
    def _get_module_by_name(self, model, name):
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
    
    def generate(self, prompt, **kwargs):
        """
        Generate text from a prompt.
        
        Args:
            prompt (str or list): Input prompt or list of prompts
            **kwargs: Additional arguments for generation
            
        Returns:
            str or list: Generated text
        """
        # Prepare inputs
        if isinstance(prompt, str):
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            outputs = self.model.generate(**inputs, **kwargs)
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text
        else:
            # Handle list of prompts
            inputs = self.tokenizer(prompt, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            outputs = self.model.generate(**inputs, **kwargs)
            
            # Decode
            generated_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            return generated_texts
    
    def __call__(self, *args, **kwargs):
        """
        Forward pass through the model.
        
        Args:
            *args: Positional arguments for the model
            **kwargs: Keyword arguments for the model
            
        Returns:
            Any: Model outputs
        """
        return self.model(*args, **kwargs)
    
    @classmethod
    def from_pretrained(cls, model_name, bits=8, **kwargs):
        """
        Load a pre-trained model with quantization.
        
        Args:
            model_name (str): Name or path of the pre-trained model
            bits (int): Bit precision for quantization (4 or 8)
            **kwargs: Additional arguments for model loading
            
        Returns:
            QuantizedModel: Quantized model
        """
        config = QuantizationConfig(bits=bits)
        return cls(model_name, config=config, **kwargs)