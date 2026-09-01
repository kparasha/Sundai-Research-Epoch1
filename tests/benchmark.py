#!/usr/bin/env python
"""
Benchmark memory usage and performance of Memory-Efficient GPT.
"""

import argparse
import torch
import time
import gc
import numpy as np
import matplotlib.pyplot as plt
import os
from memory_efficient_gpt.quantization import QuantizedModel
from memory_efficient_gpt.attention import use_flash_attention_if_available
from memory_efficient_gpt.kv_cache import use_paged_kv_cache_if_available

def get_gpu_memory_usage():
    """Get GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def benchmark_model(model_name, optimizations, max_new_tokens=100, input_text="Hello, I am a language model. I can help you with"):
    """
    Benchmark a model with various optimizations.
    
    Args:
        model_name (str): Name or path of the model
        optimizations (dict): Dictionary of optimizations to apply
        max_new_tokens (int, optional): Maximum number of new tokens to generate
        input_text (str, optional): Input text for generation
        
    Returns:
        dict: Benchmark results
    """
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Load model
    start_time = time.time()
    
    if optimizations.get("quantization"):
        # Load quantized model
        model = QuantizedModel.from_pretrained(
            model_name,
            bits=optimizations["quantization"],
            device_map="auto"
        )
    else:
        # Load standard model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Attach tokenizer to model for convenience
        model.tokenizer = tokenizer
    
    # Apply FlashAttention if specified
    if optimizations.get("flash_attention"):
        model = use_flash_attention_if_available(model)
    
    # Apply PagedAttention if specified
    if optimizations.get("paged_kv_cache"):
        model = use_paged_kv_cache_if_available(model)
    
    # Measure loading time
    loading_time = time.time() - start_time
    
    # Measure memory usage
    memory_usage = get_gpu_memory_usage()
    
    # Warm up
    _ = model.generate(input_text, max_new_tokens=10)
    
    # Measure generation time
    start_time = time.time()
    
    output = model.generate(input_text, max_new_tokens=max_new_tokens)
    
    # Measure generation time
    generation_time = time.time() - start_time
    
    # Calculate tokens per second
    input_tokens = len(model.tokenizer.encode(input_text))
    output_tokens = len(model.tokenizer.encode(output))
    tokens_generated = output_tokens - input_tokens
    
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    # Return results
    return {
        "model_name": model_name,
        "optimizations": optimizations,
        "loading_time": loading_time,
        "memory_usage": memory_usage,
        "generation_time": generation_time,
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_per_second,
        "output": output
    }

def run_benchmarks(model_name, output_dir=None):
    """
    Run benchmarks for a model with various optimizations.
    
    Args:
        model_name (str): Name or path of the model
        output_dir (str, optional): Directory to save benchmark results
    """
    # Define optimizations to benchmark
    optimizations_list = [
        {"name": "Standard", "optimizations": {}},
        {"name": "8-bit Quantization", "optimizations": {"quantization": 8}},
        {"name": "4-bit Quantization", "optimizations": {"quantization": 4}},
        {"name": "FlashAttention", "optimizations": {"flash_attention": True}},
        {"name": "PagedKVCache", "optimizations": {"paged_kv_cache": True}},
        {"name": "8-bit + FlashAttention", "optimizations": {"quantization": 8, "flash_attention": True}},
        {"name": "8-bit + PagedKVCache", "optimizations": {"quantization": 8, "paged_kv_cache": True}},
        {"name": "4-bit + FlashAttention", "optimizations": {"quantization": 4, "flash_attention": True}},
        {"name": "4-bit + PagedKVCache", "optimizations": {"quantization": 4, "paged_kv_cache": True}},
        {"name": "All Optimizations", "optimizations": {"quantization": 4, "flash_attention": True, "paged_kv_cache": True}}
    ]
    
    # Run benchmarks
    results = []
    
    for opt in optimizations_list:
        try:
            print(f"Benchmarking {model_name} with {opt['name']}...")
            result = benchmark_model(model_name, opt["optimizations"])
            result["optimization_name"] = opt["name"]
            results.append(result)
            print(f"  Memory usage: {result['memory_usage']:.2f} MB")
            print(f"  Tokens per second: {result['tokens_per_second']:.2f}")
        except Exception as e:
            print(f"Failed to benchmark {opt['name']}: {e}")
    
    # Plot results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot memory usage
        plt.figure(figsize=(12, 6))
        plt.bar([r["optimization_name"] for r in results], [r["memory_usage"] for r in results])
        plt.title(f"Memory Usage for {model_name}")
        plt.xlabel("Optimization")
        plt.ylabel("Memory Usage (MB)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name.replace('/', '_')}_memory_usage.png"))
        
        # Plot tokens per second
        plt.figure(figsize=(12, 6))
        plt.bar([r["optimization_name"] for r in results], [r["tokens_per_second"] for r in results])
        plt.title(f"Generation Speed for {model_name}")
        plt.xlabel("Optimization")
        plt.ylabel("Tokens per Second")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name.replace('/', '_')}_tokens_per_second.png"))
        
        # Save results as CSV
        import csv
        with open(os.path.join(output_dir, f"{model_name.replace('/', '_')}_benchmark_results.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Optimization", "Memory Usage (MB)", "Loading Time (s)", "Generation Time (s)", "Tokens Generated", "Tokens per Second"])
            
            for r in results:
                writer.writerow([
                    r["optimization_name"],
                    r["memory_usage"],
                    r["loading_time"],
                    r["generation_time"],
                    r["tokens_generated"],
                    r["tokens_per_second"]
                ])
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark Memory-Efficient GPT")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Directory to save benchmark results")
    
    args = parser.parse_args()
    
    # Run benchmarks
    run_benchmarks(args.model, args.output_dir)

if __name__ == "__main__":
    main()