"""
Tests for fine-tuning module.
"""

import unittest
import torch
import os
import tempfile
import shutil
from memory_efficient_gpt.fine_tuning import FineTuningPipeline

class TestFineTuning(unittest.TestCase):
    """Test cases for fine-tuning module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fine_tuning_pipeline_initialization(self):
        """Test fine-tuning pipeline initialization."""
        # Skip if no GPU available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Create fine-tuning pipeline
        model_name = "gpt2"
        
        config = {
            "quantization_bits": 8,
            "use_lora": True,
            "lora_rank": 8,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "learning_rate": 1e-4,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 1
        }
        
        try:
            # Initialize pipeline
            pipeline = FineTuningPipeline(model_name, config)
            
            # Check that pipeline is initialized
            self.assertIsNotNone(pipeline)
            self.assertEqual(pipeline.model_name, model_name)
            self.assertEqual(pipeline.config["quantization_bits"], 8)
            self.assertEqual(pipeline.config["use_lora"], True)
            
        except Exception as e:
            self.fail(f"Failed to initialize fine-tuning pipeline: {e}")
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fine_tuning_pipeline_model_preparation(self):
        """Test fine-tuning pipeline model preparation."""
        # Skip if no GPU available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Create fine-tuning pipeline
        model_name = "gpt2"
        
        config = {
            "quantization_bits": 8,
            "use_lora": True,
            "lora_rank": 8,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "learning_rate": 1e-4,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 1
        }
        
        try:
            # Initialize pipeline
            pipeline = FineTuningPipeline(model_name, config)
            
            # Prepare model
            pipeline._prepare_model()
            
            # Check that model is prepared
            self.assertIsNotNone(pipeline.model)
            
            # Check that model is quantized
            self.assertTrue(hasattr(pipeline.model, "is_quantized") or 
                           any("quantized" in name for name, _ in pipeline.model.named_modules()))
            
            # Check that model has LoRA adapters
            self.assertTrue(hasattr(pipeline.model, "peft_config") or 
                           any("lora" in name.lower() for name, _ in pipeline.model.named_modules()))
            
        except Exception as e:
            self.fail(f"Failed to prepare model for fine-tuning: {e}")
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fine_tuning_pipeline_save_load(self):
        """Test fine-tuning pipeline save and load."""
        # Skip if no GPU available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Create fine-tuning pipeline
        model_name = "gpt2"
        
        config = {
            "quantization_bits": 8,
            "use_lora": True,
            "lora_rank": 8,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "learning_rate": 1e-4,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 1
        }
        
        try:
            # Initialize pipeline
            pipeline = FineTuningPipeline(model_name, config)
            
            # Prepare model
            pipeline._prepare_model()
            
            # Save model
            output_dir = os.path.join(self.temp_dir, "saved_model")
            pipeline.save_model(output_dir)
            
            # Check that model is saved
            self.assertTrue(os.path.exists(output_dir))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "fine_tuning_config.json")))
            
            # Load model
            new_pipeline = FineTuningPipeline(model_name)
            new_pipeline.load_model(output_dir)
            
            # Check that model is loaded
            self.assertIsNotNone(new_pipeline.model)
            
            # Check that config is loaded
            self.assertEqual(new_pipeline.config["quantization_bits"], 8)
            self.assertEqual(new_pipeline.config["use_lora"], True)
            
        except Exception as e:
            self.fail(f"Failed to save and load fine-tuned model: {e}")
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fine_tuning_pipeline_training(self):
        """Test fine-tuning pipeline training."""
        # Skip if no GPU available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Create fine-tuning pipeline
        model_name = "gpt2"
        
        config = {
            "quantization_bits": 8,
            "use_lora": True,
            "lora_rank": 8,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "learning_rate": 1e-4,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 1
        }
        
        try:
            # Initialize pipeline
            pipeline = FineTuningPipeline(model_name, config)
            
            # Create a simple dataset
            from torch.utils.data import Dataset
            
            class SimpleDataset(Dataset):
                def __init__(self, tokenizer):
                    self.tokenizer = tokenizer
                    self.texts = ["Hello, I am a language model.", "The weather today is sunny."]
                    
                def __len__(self):
                    return len(self.texts)
                
                def __getitem__(self, idx):
                    text = self.texts[idx]
                    encodings = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
                    return {
                        "input_ids": encodings["input_ids"].squeeze(),
                        "attention_mask": encodings["attention_mask"].squeeze(),
                        "labels": encodings["input_ids"].squeeze()
                    }
            
            # Create dataset
            dataset = SimpleDataset(pipeline.tokenizer)
            
            # Train model
            output_dir = os.path.join(self.temp_dir, "trained_model")
            pipeline.train(dataset, output_dir=output_dir, num_train_epochs=1, logging_steps=1)
            
            # Check that model is trained
            self.assertIsNotNone(pipeline.model)
            self.assertIsNotNone(pipeline.trainer)
            
            # Check that output directory exists
            self.assertTrue(os.path.exists(output_dir))
            
        except Exception as e:
            self.fail(f"Failed to train model: {e}")

if __name__ == "__main__":
    unittest.main()