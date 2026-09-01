# Memory-Efficient GPT

A comprehensive implementation of memory optimization techniques for GPT-style transformer models, enabling large language models to run efficiently on consumer-grade hardware.

## Overview

Memory-Efficient GPT is a project focused on reducing the memory footprint of GPT models during both inference and training/fine-tuning. By implementing various optimization techniques, from simple quantization to advanced architectural modifications, this project enables running large language models on consumer-grade GPUs and optimizes memory usage for Retrieval-Augmented Generation (RAG) and autonomous agent use cases.

## Documentation

For detailed documentation, please refer to the following:

- [Sprint Plan](docs/sprint_plan.md): 2-hour sprint plan for implementation
- [PRD](docs/prd.md): Product Requirements Document
- [Data Model](docs/data_model.md): Data structures and relationships
- [UML Diagram](docs/uml_diagram.md): UML diagrams for system architecture
- [TDD](docs/tdd.md): Test-Driven Development plan
- [FDD](docs/fdd.md): Feature-Driven Development plan
- [Technical Architecture](docs/technical_architecture.md): Detailed technical architecture
- [README](docs/readme.md): Detailed project README

## Key Features

- **Low-Bit Quantization**: 4-bit and 8-bit quantization for model weights
- **FlashAttention**: Memory-efficient attention computation
- **PagedAttention**: Efficient KV cache management via vLLM
- **QLoRA**: Memory-efficient fine-tuning with quantized models
- **DeepSpeed Offloading**: CPU/NVMe offloading for handling larger models
- **Reformer-Inspired Techniques**:
  - LSH Attention for long contexts
  - Reversible layers for memory-efficient backpropagation
  - Chunked feed-forward networks for reduced peak memory usage
- **Unified Inference Server**: Combining multiple optimization techniques
- **Fine-Tuning Pipeline**: Optimized for consumer GPUs
- **Integration Frameworks**: For RAG and autonomous agent use cases

## Installation

```bash
# Clone the repository
git clone https://github.com/sundai-research/memory-efficient-gpt.git
cd memory-efficient-gpt

# Install the package
pip install -e .

# Install optional dependencies
pip install -e ".[flash,deepspeed,vllm]"
```

## Quick Start

### Quantized Inference

```python
from memory_efficient_gpt import QuantizedModel

# Load a quantized model
model = QuantizedModel.from_pretrained("gpt2-large", bits=4)

# Generate text
output = model.generate("Hello, I am a")
print(output)
```

### Memory-Efficient Fine-Tuning

```python
from memory_efficient_gpt import FineTuningPipeline

# Configure fine-tuning
config = {
    "quantization_bits": 4,
    "use_lora": True,
    "lora_rank": 8,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "learning_rate": 1e-4
}

# Initialize pipeline
pipeline = FineTuningPipeline("gpt2-large", config)

# Fine-tune model
pipeline.train("path/to/dataset")

# Save trained model
pipeline.save_model("path/to/output")
```

### Inference Server

```bash
# Start the inference server
python -m memory_efficient_gpt.server --port 8000 --models gpt2-large,gpt2-xl --quantization 4
```

```python
# Client code
import requests

response = requests.post("http://localhost:8000/generate", json={
    "model": "gpt2-large",
    "prompt": "Hello, I am a",
    "max_tokens": 50,
    "temperature": 0.7
})

print(response.json()["text"])
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization support
- [FlashAttention](https://github.com/HazyResearch/flash-attention) for efficient attention computation
- [vLLM](https://github.com/vllm-project/vllm) for PagedAttention implementation
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) for offloading support
- [QLoRA](https://github.com/artidoro/qlora) for quantized fine-tuning techniques
- [Hugging Face](https://huggingface.co/) for the Transformers ecosystem
- [Reformer](https://github.com/lucidrains/reformer-pytorch) for architectural inspiration