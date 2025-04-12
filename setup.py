from setuptools import setup, find_packages

setup(
    name="memory-efficient-gpt",
    version="0.1.0",
    description="Memory-efficient techniques for GPT models",
    author="Sundai Research",
    author_email="info@sundai-research.com",
    url="https://github.com/sundai-research/memory-efficient-gpt",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "bitsandbytes>=0.39.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "pydantic>=1.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "flash": [
            "flash-attn>=2.0.0",
        ],
        "deepspeed": [
            "deepspeed>=0.9.0",
        ],
        "vllm": [
            "vllm>=0.1.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)