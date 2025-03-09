from abc import ABC, abstractmethod
from typing import Any
from model_config import ModelConfig, ModelFramework
import torch
import onnxruntime as ort
import uuid
import time

"""
@author: Yixin Huang
@last update: 2025-03-06 14:36
@tested: True
"""

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
