from abc import ABC, abstractmethod
from typing import Any
import sys
sys.path.append('/home/yixin/study/')

from lightserve.core_component.model_config import ModelConfig, ModelFramework
import torch
import onnxruntime as ort
import uuid
import time

class BaseAIModel(ABC):
    """Base class with separate configuration"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._model: Any = None
        self._loaded: bool = False
        self.model_id = str(uuid.uuid4())
        self._creation_time: float = time.time()
        
    @property
    def instance_info(self) -> dict:
        """Get structured instance information"""
        return {
            "model_id": self.model_id,
            "framework": self.config.framework.value,
            "version": self.config.version,
            "device": self.config.device,
            "status": "loaded" if self._loaded else "unloaded",
            "created_at": self._creation_time,
            "model_path": self.config.model_path
        }

    @abstractmethod
    def load(self, model_path: str, **kwargs):
        """Load model with configuration"""
        pass

    @abstractmethod
    def predict(self, input_data: Any, **kwargs) -> Any:
        """Run inference with current config"""
        pass

    def update_config(self, **kwargs):
        """Safe configuration update"""
        self.config = self.config.model_copy(update=kwargs)

    @abstractmethod
    def _handle_device_change(self):
        """Implementation-specific device update logic"""
        pass

# Add to BaseAIModel class
    def set_device(self, new_device: str):
        """Safe device update with resource management"""
        if new_device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device {new_device}")
        
        if new_device == self.config.device:
            return
        
        # Store previous state for rollback
        prev_device = self.config.device
        self.update_config(device=new_device)
        
        try:
            self._handle_device_change()
        except Exception as e:
            # Rollback on failure
            self.update_config(device=prev_device)
            raise RuntimeError(f"Device change failed: {str(e)}")

