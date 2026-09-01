"""
Implementation of inference server.
"""

import torch
import time
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from ..quantization import QuantizedModel
from ..attention import use_flash_attention_if_available
from ..kv_cache import use_paged_kv_cache_if_available
from .model_manager import ModelManager

class InferenceRequest(BaseModel):
    """
    Request model for inference.
    """
    input_text: Optional[str] = None
    input_tokens: Optional[List[int]] = None
    model_name: str
    max_new_tokens: int = 20
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    use_flash_attention: bool = True
    kv_cache_strategy: str = "paged"  # "paged", "standard", "none"
    offload_strategy: Optional[str] = None  # "cpu", "nvme", "none"
    request_id: Optional[str] = None

class InferenceResponse(BaseModel):
    """
    Response model for inference.
    """
    generated_text: str
    request_id: str
    model_name: str
    elapsed_time: float
    tokens_generated: int
    tokens_per_second: float

class BatchInferenceRequest(BaseModel):
    """
    Request model for batch inference.
    """
    requests: List[InferenceRequest]

class BatchInferenceResponse(BaseModel):
    """
    Response model for batch inference.
    """
    responses: List[InferenceResponse]

class InferenceServer:
    """
    Server for memory-efficient model inference.
    
    This server combines various memory optimization techniques for inference,
    including quantization, FlashAttention, and PagedAttention.
    
    Args:
        models_config (dict, optional): Configuration for models
        device_map (str, optional): Device mapping for models
    """
    
    def __init__(self, models_config=None, device_map="auto"):
        self.models_config = models_config or {}
        self.device_map = device_map
        
        # Initialize model manager
        self.model_manager = ModelManager(device_map=device_map)
        
        # Initialize FastAPI app
        self.app = FastAPI(title="Memory-Efficient GPT Inference Server")
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """
        Register API routes.
        """
        @self.app.post("/generate", response_model=InferenceResponse)
        async def generate(request: InferenceRequest):
            return self.handle_request(request)
        
        @self.app.post("/batch_generate", response_model=BatchInferenceResponse)
        async def batch_generate(request: BatchInferenceRequest):
            responses = self.handle_batch_requests(request.requests)
            return BatchInferenceResponse(responses=responses)
        
        @self.app.get("/models")
        async def get_models():
            return {"models": list(self.model_manager.get_loaded_models())}
        
        @self.app.post("/models/{model_name}/load")
        async def load_model(model_name: str):
            try:
                self.load_model(model_name)
                return {"status": "success", "message": f"Model {model_name} loaded successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/{model_name}/unload")
        async def unload_model(model_name: str):
            try:
                self.unload_model(model_name)
                return {"status": "success", "message": f"Model {model_name} unloaded successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def start(self, host="0.0.0.0", port=8000):
        """
        Start the inference server.
        
        Args:
            host (str, optional): Host to bind to
            port (int, optional): Port to bind to
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)
    
    def load_model(self, model_name):
        """
        Load a model.
        
        Args:
            model_name (str): Name or path of the model
            
        Returns:
            nn.Module: Loaded model
        """
        # Get model configuration
        model_config = self.models_config.get(model_name, {})
        
        # Load model with optimizations
        model = self.model_manager.load_model(
            model_name=model_name,
            quantization_bits=model_config.get("quantization_bits", None),
            use_flash_attention=model_config.get("use_flash_attention", True),
            kv_cache_strategy=model_config.get("kv_cache_strategy", "paged"),
            offload_strategy=model_config.get("offload_strategy", None)
        )
        
        return model
    
    def unload_model(self, model_name):
        """
        Unload a model.
        
        Args:
            model_name (str): Name of the model to unload
        """
        self.model_manager.unload_model(model_name)
    
    def handle_request(self, request):
        """
        Handle an inference request.
        
        Args:
            request (InferenceRequest): Inference request
            
        Returns:
            InferenceResponse: Inference response
        """
        # Generate request ID if not provided
        request_id = request.request_id or str(uuid.uuid4())
        
        # Load model if not already loaded
        if not self.model_manager.is_model_loaded(request.model_name):
            self.load_model(request.model_name)
        
        # Get model
        model = self.model_manager.get_model(request.model_name)
        
        # Prepare input
        if request.input_text is not None:
            input_text = request.input_text
        elif request.input_tokens is not None:
            # Decode tokens to text
            input_text = model.tokenizer.decode(request.input_tokens)
        else:
            raise ValueError("Either input_text or input_tokens must be provided")
        
        # Generate text
        start_time = time.time()
        
        generated_text = model.generate(
            input_text,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate tokens generated
        input_tokens = model.tokenizer.encode(input_text)
        output_tokens = model.tokenizer.encode(generated_text)
        tokens_generated = len(output_tokens) - len(input_tokens)
        
        # Calculate tokens per second
        tokens_per_second = tokens_generated / elapsed_time if elapsed_time > 0 else 0
        
        # Create response
        response = InferenceResponse(
            generated_text=generated_text,
            request_id=request_id,
            model_name=request.model_name,
            elapsed_time=elapsed_time,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second
        )
        
        return response
    
    def handle_batch_requests(self, requests):
        """
        Handle batch inference requests.
        
        Args:
            requests (List[InferenceRequest]): List of inference requests
            
        Returns:
            List[InferenceResponse]: List of inference responses
        """
        responses = []
        
        for request in requests:
            response = self.handle_request(request)
            responses.append(response)
        
        return responses