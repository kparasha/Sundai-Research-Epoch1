"""
Implementation of QLoRA (Quantized Low-Rank Adaptation) for memory-efficient fine-tuning.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments
from ..quantization import QuantizedModel

class QLoRATrainer:
    """
    QLoRA trainer for memory-efficient fine-tuning.
    
    Args:
        model (nn.Module): Model to fine-tune
        rank (int, optional): Rank of LoRA adaptation matrices
        alpha (int, optional): Alpha parameter for LoRA scaling
        dropout (float, optional): Dropout probability for LoRA layers
        target_modules (list, optional): List of module names to apply LoRA to
        quantization_bits (int, optional): Number of bits for quantization (4 or 8)
    """
    
    def __init__(
        self,
        model,
        rank=8,
        alpha=32,
        dropout=0.1,
        target_modules=None,
        quantization_bits=4
    ):
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.quantization_bits = quantization_bits
        
        # Prepare model for QLoRA
        self.prepared_model = self._prepare_model()
    
    def _prepare_model(self):
        """
        Prepare model for QLoRA training.
        
        Returns:
            nn.Module: Prepared model
        """
        # Quantize model
        if not isinstance(self.model, QuantizedModel):
            from ..quantization import QuantizationConfig
            config = QuantizationConfig(bits=self.quantization_bits)
            quantized_model = QuantizedModel(self.model, config)
        else:
            quantized_model = self.model
        
        # Prepare model for k-bit training
        prepared_model = prepare_model_for_kbit_training(
            quantized_model,
            use_gradient_checkpointing=True
        )
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA adapter
        model_with_lora = get_peft_model(prepared_model, lora_config)
        
        return model_with_lora
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        output_dir="./qlora_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        max_grad_norm=0.3,
        **kwargs
    ):
        """
        Train the model with QLoRA.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir (str): Output directory
            num_train_epochs (int): Number of training epochs
            per_device_train_batch_size (int): Batch size per device
            gradient_accumulation_steps (int): Gradient accumulation steps
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay
            max_grad_norm (float): Maximum gradient norm
            **kwargs: Additional arguments for TrainingArguments
            
        Returns:
            Trainer: Trained model
        """
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset is not None else False,
            **kwargs
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.prepared_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        # Train model
        trainer.train()
        
        return trainer
    
    def save(self, output_dir):
        """
        Save the trained model.
        
        Args:
            output_dir (str): Output directory
        """
        self.prepared_model.save_pretrained(output_dir)
    
    def merge_and_save(self, output_dir):
        """
        Merge LoRA weights into the base model and save.
        
        Args:
            output_dir (str): Output directory
        """
        # Merge LoRA weights into base model
        merged_model = self.prepared_model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_dir)