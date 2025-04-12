"""
Configuration classes for quantization.
"""

class QuantizationConfig:
    """
    Configuration for quantization parameters.
    
    Args:
        bits (int): Bit precision for quantization (4 or 8)
        group_size (int): Size of quantization groups
        symmetric (bool): Whether to use symmetric quantization
        use_bitsandbytes (bool): Whether to use bitsandbytes library
        target_modules (list): List of module types to quantize
        excluded_modules (list): List of module names to exclude from quantization
    """
    
    def __init__(
        self,
        bits=8,
        group_size=128,
        symmetric=True,
        use_bitsandbytes=True,
        target_modules=None,
        excluded_modules=None
    ):
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.use_bitsandbytes = use_bitsandbytes
        self.target_modules = target_modules or ["Linear"]
        self.excluded_modules = excluded_modules or []
    
    def __repr__(self):
        return (
            f"QuantizationConfig(bits={self.bits}, "
            f"group_size={self.group_size}, "
            f"symmetric={self.symmetric}, "
            f"use_bitsandbytes={self.use_bitsandbytes}, "
            f"target_modules={self.target_modules}, "
            f"excluded_modules={self.excluded_modules})"
        )
    
    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a QuantizationConfig from a dictionary.
        
        Args:
            config_dict (dict): Dictionary containing configuration parameters
            
        Returns:
            QuantizationConfig: Configuration object
        """
        return cls(**config_dict)
    
    def to_dict(self):
        """
        Convert the configuration to a dictionary.
        
        Returns:
            dict: Dictionary representation of the configuration
        """
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "use_bitsandbytes": self.use_bitsandbytes,
            "target_modules": self.target_modules,
            "excluded_modules": self.excluded_modules
        }
    
    def to_bitsandbytes_config(self):
        """
        Convert the configuration to a bitsandbytes configuration.
        
        Returns:
            dict: bitsandbytes configuration
        """
        if self.bits == 8:
            return {
                "load_in_8bit": True,
                "llm_int8_threshold": 6.0,
                "llm_int8_skip_modules": self.excluded_modules
            }
        elif self.bits == 4:
            return {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4" if self.symmetric else "fp4"
            }
        else:
            raise ValueError(f"Unsupported bit precision: {self.bits}. Use 4 or 8.")