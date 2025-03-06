from model_manager import ModelManager
from resource_monitor import ResourceMonitor
import time
import logging
from typing import Dict
from pydantic import BaseModel

logger = logging.getLogger("LifecycleController")
"""
@author: yixin.huang
@last update: 2025-03-04 19:43:00
@tested: False
"""

class ScalingPolicy(BaseModel):
    min_instances: int = 1
    max_instances: int = 5
    target_gpu_util: float = 70.0
    target_memory_util: float = 80.0

class LifecycleController:
    def __init__(self, 
                model_manager: ModelManager,
                monitor: ResourceMonitor,
                policies: Dict[str, ScalingPolicy]):
        self.model_manager = model_manager
        self.monitor = monitor
        self.policies = policies
        self._running = False

    def start(self):
        """Start background control loop"""
        self._running = True
        while self._running:
            self._check_idle_instances()
            self._auto_scale()
            self._health_check()
            time.sleep(30)

    def _check_idle_instances(self):
        """Cleanup long-idle instances"""
        for instance in self.model_manager.list_instances():
            last_used = time.time() - instance['last_used']
            if last_used > self.policies['default'].max_idle_time:
                self.model_manager.destroy_instance(instance['model_id'])

    def _auto_scale(self):
        """Adjust instance count based on metrics"""
        stats = self.monitor.get_current_stats()
        
        for model_type, policy in self.policies.items():
            current_instances = self._count_instances(model_type)
            
            if current_instances < policy.min_instances:
                self._scale_up(model_type)
            
            if self._needs_scaling(stats, policy):
                if current_instances < policy.max_instances:
                    self._scale_up(model_type)
                else:
                    logger.warning(f"Max instances reached for {model_type}")

    def _health_check(self):
        """Restart unhealthy instances"""
        health = self.monitor.check_health()
        for alert in health['alerts']:
            if 'GPU_MEMORY' in alert.type:
                self._handle_gpu_pressure()

    def _scale_up(self, model_type: str):
        config = self._get_model_config(model_type)
        self.model_manager.create_instance(config)