from abc import ABC, abstractmethod
from typing import Any
import sys
sys.path.append('/home/yixin/study/')

from lightserve.core_component.model_config import ModelConfig, ModelFramework
import torch
import onnxruntime as ort
import uuid
import time
from lightserve.models import PyTorchModel, ONNXModel, YoloModel, TransformersResNetForImageClassification, BaseAIModel

"""
@author: Yixin Huang
@last update: 2025-03-09 15:06
@tested: True
"""


class ModelFactory:
    """Factory for model creation with config validation"""
    
    @staticmethod
    def create_model(config: ModelConfig, model_id: str = None) -> BaseAIModel:
        """Create model instance with config validation
            and custom model_id if provided
        """
        if config.framework == ModelFramework.PYTORCH:
            instance =  PyTorchModel(config)
        elif config.framework == ModelFramework.ONNX:
            instance = ONNXModel(config)
        elif config.framework == ModelFramework.YOLO:
            instance = YoloModel(config)
        else:
            raise ValueError(f"Unsupported framework: {config.framework}")
        
        if model_id:
            instance.model_id = model_id
        return instance

# Usage example
if __name__ == "__main__":
    # Create configuration
    config = ModelConfig(
        framework=ModelFramework.YOLO,
        device="cuda",
        version="11.3.0",
        metadata={"task": "segmentation"},
        max_batch_size=32,
        max_workers=4,
        model_path="/home/yixin/study/models/yolo_model/detection/yolo11x.pt"
    )
    
    # Create and load model
    model = ModelFactory.create_model(config)
    model.load(config.model_path, input_shape=(3, 224, 224))
    
    # Runtime config update
    model.update_config(max_batch_size=64)
    
    print(f"Current device: {model.config.device}")
    print(f"Model metadata: {model.config.metadata}")


    # Initialize model on CPU
    config = ModelConfig(
        framework=ModelFramework.YOLO,
        device="cpu",
        version="12.3.0",
        model_path="/home/yixin/study/models/yolo_model/detection/yolo12x.pt"
    )
    model = ModelFactory.create_model(config)
    model.load(config.model_path, input_shape=(3,224,224))

    # Migrate to GPU
    try:
        model.set_device("cuda")
        if model.config.device == "cuda":
            print("Device changed successfully")
    except RuntimeError as e:
        print(f"Failed to change device: {e}")
