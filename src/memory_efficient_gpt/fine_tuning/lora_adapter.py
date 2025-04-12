"""
Implementation of LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

class LoRAAdapter:
    """
    LoRA adapter for memory-efficient fine-tuning.
    
    Args:
        model (nn.Module): Model to adapt
        rank (int, optional): Rank of LoRA adaptation matrices
        alpha (int, optional): Alpha parameter for LoRA scaling
        dropout (float, optional): Dropout probability for LoRA layers
        target_modules (list, optional): List of module names to apply LoRA to
    """
    
    def __init__(
        self,
        model,
        rank=8,
        alpha=32,
        dropout=0.1,
        target_modules=None
    ):
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        
        # Create LoRA configuration
        self.lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
    
    def apply(self):
        """
        Apply LoRA adapter to the model.
        
        Returns:
            nn.Module: Model with LoRA adapter
        """
        # Apply LoRA adapter
        model_with_lora = get_peft_model(self.model, self.lora_config)
        
        return model_with_lora
    
    @staticmethod
    def from_pretrained(model, adapter_path):
        """
        Load a model with a pre-trained LoRA adapter.
        
        Args:
            model (nn.Module): Base model
            adapter_path (str): Path to LoRA adapter weights
            
        Returns:
            nn.Module: Model with loaded LoRA adapter
        """
        from peft import PeftModel
        
        # Load LoRA adapter
        model_with_lora = PeftModel.from_pretrained(model, adapter_path)
        
        return model_with_lora
    
    @staticmethod
    def merge_and_save(model, output_path):
        """
        Merge LoRA weights into the base model and save.
        
        Args:
            model (nn.Module): Model with LoRA adapter
            output_path (str): Path to save merged model
        """
        # Check if model has LoRA adapter
        if not hasattr(model, "merge_and_unload"):
            raise ValueError("Model does not have a LoRA adapter")
        
        # Merge LoRA weights into base model
        merged_model = model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_path)
        
        return merged_model