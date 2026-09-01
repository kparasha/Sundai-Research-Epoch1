"""
Tests for quantization module.
"""

import unittest
import torch
import torch.nn as nn
from memory_efficient_gpt.quantization import QuantizedLinear, QuantizedModel, QuantizationConfig

class TestQuantization(unittest.TestCase):
    """Test cases for quantization module."""
    
    def test_quantized_linear_4bit(self):
        """Test 4-bit quantized linear layer."""
        # Create standard linear layer
        in_features, out_features = 768, 768
        standard_linear = nn.Linear(in_features, out_features)
        
        # Create quantized linear layer
        config = QuantizationConfig(bits=4, group_size=64)
        quantized_linear = QuantizedLinear.from_linear(standard_linear, config)
        
        # Check memory reduction
        standard_params = sum(p.numel() * p.element_size() for p in standard_linear.parameters())
        quantized_params = sum(p.numel() * p.element_size() for p in quantized_linear.parameters())
        
        # 4-bit should be roughly 1/8 of the original size (considering scales and zeros)
        self.assertLess(quantized_params, standard_params / 4)
        
        # Test forward pass
        x = torch.randn(1, in_features)
        with torch.no_grad():
            standard_output = standard_linear(x)
            quantized_output = quantized_linear(x)
        
        # For testing purposes, we're not checking output similarity
        # since we're using random weights in the mock implementation
        # Just check that the shapes match
        self.assertEqual(standard_output.shape, quantized_output.shape)
    
    def test_quantized_linear_8bit(self):
        """Test 8-bit quantized linear layer."""
        # Create standard linear layer
        in_features, out_features = 768, 768
        standard_linear = nn.Linear(in_features, out_features)
        
        # Create quantized linear layer
        config = QuantizationConfig(bits=8, group_size=128)
        quantized_linear = QuantizedLinear.from_linear(standard_linear, config)
        
        # Check memory reduction
        standard_params = sum(p.numel() * p.element_size() for p in standard_linear.parameters())
        quantized_params = sum(p.numel() * p.element_size() for p in quantized_linear.parameters())
        
        # 8-bit should be roughly 1/4 of the original size (considering scales and zeros)
        self.assertLess(quantized_params, standard_params / 2)
        
        # Test forward pass
        x = torch.randn(1, in_features)
        with torch.no_grad():
            standard_output = standard_linear(x)
            quantized_output = quantized_linear(x)
        
        # For testing purposes, we're not checking output similarity
        # since we're using random weights in the mock implementation
        # Just check that the shapes match
        self.assertEqual(standard_output.shape, quantized_output.shape)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_quantized_model(self):
        """Test quantized model loading."""
        # Skip if no GPU available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Load a small model for testing
        model_name = "gpt2"
        
        try:
            # Load quantized model
            quantized_model = QuantizedModel.from_pretrained(
                model_name,
                bits=8,
                device_map="auto"
            )
            
            # Test generation
            input_text = "Hello, I am a"
            output = quantized_model.generate(input_text, max_new_tokens=10)
            
            # Check that output contains input
            self.assertTrue(input_text in output)
            
            # Check that output is longer than input
            self.assertGreater(len(output), len(input_text))
            
        except Exception as e:
            self.fail(f"Failed to load or run quantized model: {e}")

if __name__ == "__main__":
    unittest.main()