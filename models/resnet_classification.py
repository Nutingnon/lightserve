from abc import ABC, abstractmethod
from typing import Any
from model_config import ModelConfig, ModelFramework
import torch
import onnxruntime as ort
import uuid
import time
from core_component.base_model import BaseAIModel

class TransformersResNetForImageClassification(BaseAIModel):
    """HuggingFace Transformers implementation with external config
        This is for ResNetForImageClassification
        please refer to
        - https://huggingface.co/microsoft/resnet-50
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.framework != ModelFramework.TRANSFORMERS:
            raise ValueError("Invalid framework for Transformers model")

    def load(self, model_path: str, **kwargs):
        from transformers import AutoModel, AutoTokenizer
        from transformers import AutoImageProcessor, ResNetForImageClassification
        # 设备配置
        if "device" in kwargs:
            self.update_config(device=kwargs["device"])
            
        # 加载模型
        self._model = AutoModel.from_pretrained(model_path)
        
        # 设备移动和评估模式
        self._model = self._model.to(self.config.device)
        self._model.eval()
        
        # 更新元数据
        self.config.metadata.update({
            "model_type": self._model.config.model_type,
            "supported_tasks": kwargs.get("tasks", ["text-classification"]),
            "max_length": kwargs.get("max_length", 512)
        })
        self._loaded = True

    def _handle_device_change(self):
        if not self._loaded:
            return
        
        # 移动模型到新设备
        self._model = self._model.to(self.config.device)
        
        # CUDA内存清理
        if 'cuda' in self.config.device:
            torch.cuda.synchronize()
        elif self.prev_device.startswith('cuda'):
            torch.cuda.empty_cache()

    def predict(self, input_data: torch.Tensor, **kwargs) -> Any:
        with torch.no_grad():
            logits = self._model(input_data).logits
            predicted_label = logits.argmax(-1).item()
            return predicted_label
