from abc import ABC, abstractmethod
from typing import Any
from lightserve.core_component.model_config import ModelConfig, ModelFramework
import torch
import onnxruntime as ort
import uuid
import time
from lightserve.models.base_model import BaseAIModel

"""
@author: Yixin Huang
@last update: 2025-03-09 11:41
@tested: True
"""


class PyTorchModel(BaseAIModel):
    """PyTorch implementation with external config"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.framework != ModelFramework.PYTORCH:
            raise ValueError("Invalid framework for PyTorch model")

    def load(self, model_path: str, **kwargs):
        # Update config with load parameters
        if "device" in kwargs:
            self.update_config(device=kwargs["device"])
            
        # Model loading logic
        self._model = torch.load(model_path)
        self._model = self._model.to(self.config.device)
        self._model.eval()
        
        # Update metadata
        self.config.metadata.update({
            "input_shape": kwargs.get("input_shape"),
            "classes": kwargs.get("classes")
        })
        self._loaded = True

    def _handle_device_change(self):
        if not self._loaded:
            return
        
        # Move model to new device
        self._model = self._model.to(self.config.device)
        
        # Optional memory cleanup
        if 'cuda' in self.config.device:
            torch.cuda.synchronize()
        elif self.prev_device.startswith('cuda'):
            torch.cuda.empty_cache()

    def predict(self, input_data: Any, **kwargs) -> Any:
        if not self._loaded:
            raise RuntimeError("Model not loaded")
            
        with torch.no_grad():
            input_tensor = torch.as_tensor(input_data).to(self.config.device)
            return self._model(input_tensor).cpu().numpy()
