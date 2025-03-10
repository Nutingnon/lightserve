from lightserve.core_component.model_config import ModelConfig, GlobalConfig
from pydantic import BaseModel
import yaml
from typing import List, Optional
from lightserve.core_component.model_config import ModelConfig, ModelFramework
from lightserve.lifecycle_controller import ScalingPolicy

class DeploymentConfig(BaseModel):
    name: str
    framework: ModelFramework
    model_path: str
    scaling: ScalingPolicy
    resources: dict

class AppConfig(BaseModel):
    defaults: dict
    models: List[DeploymentConfig]
    monitoring: dict
    resources: dict

def load_config(path: str) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)