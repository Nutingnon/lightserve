from abc import ABC, abstractmethod
from typing import Any
from model_config import ModelConfig, ModelFramework
import torch
import onnxruntime as ort
import uuid
import time
from core_component.base_model import BaseAIModel


class YoloModel(BaseAIModel):
    """YOLOv11 load with external config"""
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.framework!= ModelFramework.YOLO:
            raise ValueError("Invalid framework for YOLO model")
        
    def load(self, model_path: str, **kwargs):
        # Update config with load parameters
        from ultralytics import YOLO

        if "device" in kwargs:
            self.update_config(device=kwargs["device"])
            
        # Model loading logic
        self._model = YOLO(model_path)  # load an official model
        self._model = self._model.to(self.config.device)
        self._model.eval()
        
        # Update metadata
        self.config.metadata.update({
            "input_shape": kwargs.get("input_shape", "Unknown"),
        })
        self._loaded = True

    def predict(self, input_data, **kwargs):
        result = self._model(input_data)
        """
        output looks like:
        [[{'name': 'car',
        'class': 2,
        'confidence': 0.98326,
        'box': {'x1': 0.09262, 'y1': 119.44924, 'x2': 979.95294, 'y2': 591.2923}}],
        [{'name': 'person',
        'class': 0,
        'confidence': 0.89378,
        'box': {'x1': 2.45407, 'y1': 1.56372, 'x2': 799.61499, 'y2': 733.8681}}]]
        """
        return [x.summary() for x in result]

    
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
    