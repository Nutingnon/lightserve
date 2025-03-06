from pydantic import BaseModel as PydanticBase, Field, field_validator
from typing import Dict, Any, Optional
from enum import Enum

class ModelFramework(str, Enum):
    PYTORCH = "pytorch"
    YOLO = "yolo"
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"

class ModelConfig(PydanticBase):
    """Central configuration model for all AI models"""
    
    device: str = Field(
        default="cpu",
        description="Computation device (cpu/cuda)",
        pattern="^(cpu|cuda)$"
    )
    framework: ModelFramework = Field(
        ...,
        description="Model framework type"
    )
    version: str = Field(
        default="1.0.0",
        description="Model version",
        min_length=5
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model metadata dictionary"
    )
    max_batch_size: int = Field(
        default=32,
        gt=0,
        description="Maximum batch size for inference"
    )
    max_workers: int = Field(
        default=1,
        gt=0,
        description="Maximum number of workers for model loading"
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Optional model identifier"
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Optional model path"
    )

    @field_validator("device", mode="before")
    def validate_device(cls, v):
        """Auto-detect GPU availability if cuda requested"""
        if v == "cuda":
            import torch
            if not torch.cuda.is_available():
                print("CUDA is not available. Using CPU instead.")
                return "cpu"
        return v

    class Config:
        extra = "forbid"  # Prevent extra fields
        validate_assignment = True




