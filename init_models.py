from lightserve.core_component.model_manager import ModelManager
from lightserve.core_component.model_config import ModelConfig, ModelFramework

"""
@author: Yixin Huang
@last update: 2025-03-07 15:05
@tested: False

"""

manager = ModelManager()

# Example YOLO model
yolo_config = ModelConfig(
    framework=ModelFramework.YOLO,
    model_path="yolov8n.pt",
    device="cuda"
)
yolo_id = manager.create_instance(yolo_config)

# Example ONNX model
onnx_config = ModelConfig(
    framework=ModelFramework.ONNX,
    model_path="model.onnx",
    device="cpu"
)
onnx_id = manager.create_instance(onnx_config)