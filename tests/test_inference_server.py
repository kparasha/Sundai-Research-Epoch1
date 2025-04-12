"""
Tests for inference server.
"""

import unittest
import torch
import threading
import time
import requests
from memory_efficient_gpt.server import InferenceServer, ModelManager

class TestInferenceServer(unittest.TestCase):
    """Test cases for inference server."""
    
    def setUp(self):
        """Set up test environment."""
        # Create model manager
        self.model_manager = ModelManager()
        
        # Create inference server
        self.server = InferenceServer(models_config={
            "gpt2": {
                "quantization_bits": 8,
                "use_flash_attention": True,
                "kv_cache_strategy": "paged"
            }
        })
        
        # Start server in a separate thread
        self.server_thread = threading.Thread(target=self._start_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
    
    def _start_server(self):
        """Start inference server."""
        self.server.start(host="0.0.0.0", port=8000)
    
    def test_server_endpoints(self):
        """Test server endpoints."""
        try:
            # Test models endpoint
            response = requests.get("http://localhost:8000/models")
            self.assertEqual(response.status_code, 200)
            
            # Test load model endpoint
            response = requests.post("http://localhost:8000/models/gpt2/load")
            self.assertEqual(response.status_code, 200)
            
            # Test generate endpoint
            response = requests.post("http://localhost:8000/generate", json={
                "input_text": "Hello, I am a",
                "model_name": "gpt2",
                "max_new_tokens": 10,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.0,
                "use_flash_attention": True,
                "kv_cache_strategy": "paged"
            })
            
            self.assertEqual(response.status_code, 200)
            self.assertIn("generated_text", response.json())
            self.assertIn("Hello, I am a", response.json()["generated_text"])
            
            # Test batch generate endpoint
            response = requests.post("http://localhost:8000/batch_generate", json={
                "requests": [
                    {
                        "input_text": "Hello, I am a",
                        "model_name": "gpt2",
                        "max_new_tokens": 10
                    },
                    {
                        "input_text": "The weather today is",
                        "model_name": "gpt2",
                        "max_new_tokens": 10
                    }
                ]
            })
            
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()["responses"]), 2)
            
            # Test unload model endpoint
            response = requests.post("http://localhost:8000/models/gpt2/unload")
            self.assertEqual(response.status_code, 200)
            
        except Exception as e:
            self.fail(f"Failed to test server endpoints: {e}")
    
    def test_server_performance(self):
        """Test server performance."""
        try:
            # Load model
            requests.post("http://localhost:8000/models/gpt2/load")
            
            # Warm up
            requests.post("http://localhost:8000/generate", json={
                "input_text": "Hello, I am a",
                "model_name": "gpt2",
                "max_new_tokens": 10
            })
            
            # Measure generation time
            start_time = time.time()
            
            response = requests.post("http://localhost:8000/generate", json={
                "input_text": "Hello, I am a",
                "model_name": "gpt2",
                "max_new_tokens": 50
            })
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Check that generation is reasonably fast
            self.assertLess(elapsed_time, 5.0)  # Should take less than 5 seconds
            
            # Check tokens per second
            tokens_per_second = response.json()["tokens_per_second"]
            self.assertGreater(tokens_per_second, 5.0)  # Should generate at least 5 tokens per second
            
        except Exception as e:
            self.fail(f"Failed to test server performance: {e}")
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop server
        self.server_thread.join(timeout=1)

if __name__ == "__main__":
    unittest.main()