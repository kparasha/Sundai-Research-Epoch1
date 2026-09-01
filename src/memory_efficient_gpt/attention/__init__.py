"""
Attention optimization module for memory-efficient GPT models.
"""

from .flash_attention import FlashAttention
from .lsh_attention import LSHAttention
from .multi_query_attention import MultiQueryAttention
from .attention_utils import convert_to_flash_attention, use_flash_attention_if_available