# Testing Memory-Efficient GPT

This document outlines how to test the Memory-Efficient GPT implementation to verify it works according to specifications.

## Running Tests

The test suite includes unit tests, integration tests, and benchmarks to validate the functionality and performance of the memory optimization techniques.

### Prerequisites

Before running tests, ensure you have installed the package and its dependencies:

```bash
# Install the package in development mode
pip install -e .

# Install test dependencies
pip install -e ".[dev]"

# Install optional dependencies for comprehensive testing
pip install -e ".[flash,deepspeed,vllm]"
```

### Running Unit Tests

To run all unit tests:

```bash
# Run all tests
python tests/run_tests.py

# Run specific test modules
python -m unittest tests/test_quantization.py
python -m unittest tests/test_flash_attention.py
python -m unittest tests/test_kv_cache.py
```

### Running Benchmarks

To benchmark memory usage and performance:

```bash
# Run benchmarks with default model (gpt2)
python tests/benchmark.py

# Run benchmarks with a specific model
python tests/benchmark.py --model gpt2-medium

# Specify output directory for benchmark results
python tests/benchmark.py --output-dir benchmark_results
```

## Test Coverage

The test suite covers the following aspects:

### 1. Quantization Tests

- Validates 4-bit and 8-bit quantization of linear layers
- Measures memory reduction compared to full-precision models
- Verifies output quality with quantized weights
- Tests model loading and generation with quantized models

### 2. FlashAttention Tests

- Tests forward pass with FlashAttention implementation
- Verifies causal masking functionality
- Tests conversion of standard attention to FlashAttention
- Measures memory savings during attention computation

### 3. KV Cache Tests

- Tests standard KV cache allocation, update, and retrieval
- Tests paged KV cache with memory-efficient page management
- Verifies correct sequence handling with paged attention
- Measures memory efficiency for long sequences

### 4. Memory Usage Tests

- Comprehensive memory usage measurements for all optimization techniques
- Comparison of peak memory usage between standard and optimized implementations
- Verification of memory scaling with sequence length

### 5. Fine-Tuning Tests

- Tests QLoRA fine-tuning pipeline
- Verifies model saving and loading
- Tests training with small datasets
- Measures memory usage during fine-tuning

### 6. Inference Server Tests

- Tests server endpoints for model loading and generation
- Measures generation performance and throughput
- Tests batch processing capabilities
- Verifies memory management during concurrent requests

## Manual Testing

For manual testing and verification:

1. **Model Loading Test**:
   ```python
   from memory_efficient_gpt.quantization import QuantizedModel
   
   # Load a quantized model
   model = QuantizedModel.from_pretrained("gpt2", bits=4)
   
   # Generate text
   output = model.generate("Hello, I am a")
   print(output)
   ```

2. **Memory Profiling**:
   ```python
   import torch
   from memory_efficient_gpt.quantization import QuantizedModel
   
   # Track memory before loading
   torch.cuda.empty_cache()
   memory_before = torch.cuda.memory_allocated() / 1024 / 1024
   
   # Load model
   model = QuantizedModel.from_pretrained("gpt2", bits=4)
   
   # Track memory after loading
   memory_after = torch.cuda.memory_allocated() / 1024 / 1024
   
   print(f"Memory usage: {memory_after - memory_before:.2f} MB")
   ```

3. **Inference Server Test**:
   ```bash
   # Start the inference server
   python -m memory_efficient_gpt.server --port 8000 --models gpt2 --quantization 4
   ```

   Then in another terminal or using a tool like curl:
   ```bash
   curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"input_text": "Hello, I am a", "model_name": "gpt2", "max_new_tokens": 50}'
   ```

## Expected Results

When running the tests, you should expect:

1. **Quantization Tests**: Memory usage should be reduced by approximately:
   - 4-bit quantization: 75-85% reduction
   - 8-bit quantization: 50-60% reduction

2. **FlashAttention Tests**: Memory usage during attention computation should be reduced by 20-40% compared to standard attention.

3. **KV Cache Tests**: For long sequences, paged KV cache should use significantly less memory than standard KV cache (up to 80% reduction for very long sequences).

4. **Inference Performance**: Generation speed should be comparable to or faster than standard implementations, with tokens per second varying by model size.

5. **Fine-Tuning**: QLoRA fine-tuning should enable training of models that would otherwise not fit in GPU memory.

## Troubleshooting

If tests fail, check the following:

1. **CUDA Availability**: Many tests require a CUDA-capable GPU. Tests will be skipped if CUDA is not available.

2. **Dependencies**: Ensure all required dependencies are installed, including optional dependencies for specific features.

3. **Memory Issues**: If tests fail due to out-of-memory errors, try reducing batch sizes or model sizes in the test configurations.

4. **Library Versions**: Ensure compatible versions of PyTorch, transformers, and other dependencies.

## Reporting Issues

If you encounter issues with the tests or the implementation, please report them with:

1. Test name and error message
2. System configuration (GPU, CUDA version, PyTorch version)
3. Steps to reproduce the issue
4. Expected vs. actual behavior