from abc import ABC, abstractmethod
from typing import Any
from model_config import ModelConfig, ModelFramework
import torch
import onnxruntime as ort
import uuid
import time
from core_component.base_model import BaseAIModel

class ONNXModel(BaseAIModel):
    """ONNX implementation with external config"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.framework != ModelFramework.ONNX:
            raise ValueError("Invalid framework for ONNX model")

    def load(self, model_path: str, **kwargs):
        # Configure providers based on device
        providers = ["CPUExecutionProvider"]
        if self.config.device == "cuda":
            providers = ["CUDAExecutionProvider"] + providers
            
        # Create inference session
        self._model = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=ort.SessionOptions()
        )
        
        # Extract metadata
        meta = self._model.get_modelmeta()
        self.update_config(
            version=meta.version or self.config.version,
            metadata={
                **self.config.metadata,
                "inputs": [i.name for i in self._model.get_inputs()],
                "outputs": [o.name for o in self._model.get_outputs()]
            }
        )
        self._loaded = True

    # In ONNXModel class
    def _handle_device_change(self):
        if not self._loaded:
            return
        
        # Recreate session with new providers
        providers = ["CPUExecutionProvider"]
        if self.config.device == "cuda":
            providers = ["CUDAExecutionProvider"] + providers
        
        # Preserve session options
        sess_options = self._model.get_session_options()
        self._model = ort.InferenceSession(
            self.config.model_path,
            sess_options=sess_options,
            providers=providers
        )

    def predict(self, input_data: Any, **kwargs) -> Any:
        if not self._loaded:
            raise RuntimeError("Model not loaded")
            
        input_name = self._model.get_inputs()[0].name
        return self._model.run(None, {input_name: input_data})[0]

