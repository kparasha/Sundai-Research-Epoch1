"""
Quantization module for memory-efficient GPT models.
"""

from .quantized_linear import QuantizedLinear
from .quantized_model import QuantizedModel
from .config import QuantizationConfig