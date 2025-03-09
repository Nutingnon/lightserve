from model_manager import ModelManager
from resource_monitor import ResourceMonitor
import time
import logging
from typing import Dict, Optional
from pydantic import BaseModel
import threading

"""
@author: yixin.huang
@last update: 2025-03-07 13:48:00
@tested: True

This lifecycle controller manages model instances by:
- Removing idle instances
- Auto-scaling based on resource utilization
- Performing health checks
- Enforcing scaling policies

"""

logger = logging.getLogger("LifecycleController")

class ScalingPolicy(BaseModel):
    min_instances: int = 1
    max_instances: int = 10
    target_gpu_util: float = 70.0
    target_mem_util: float = 80.0
    max_idle_time: int = 300  # Added missing field
    model_type: Optional[str] = None

class LifecycleController:
    def __init__(self, 
                model_manager: ModelManager,
                monitor: ResourceMonitor,
                policies: Dict[str, ScalingPolicy]):
        """
        Initialize the LifecycleController.

        Args:
            model_manager (ModelManager): The model manager instance used to manage model instances.
            monitor (ResourceMonitor): The resource monitor instance used to monitor system resources.
            policies (Dict[str, ScalingPolicy]): A dictionary of scaling policies, where the key is the policy name
                and the value is an instance of ScalingPolicy.
        """
        self.model_manager = model_manager
        self.monitor = monitor
        self.policies = policies
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Validate policies
        if 'default' not in self.policies:
            raise ValueError("Missing default scaling policy")

    def start(self):
        """Start background control loop in a daemon thread"""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(
                target=self._control_loop,
                daemon=True
            )
            self._thread.start()
            logger.info("Lifecycle controller started")

    def stop(self):
        """Stop the control loop"""
        self._running = False
        if self._thread:
            self._thread.join()
        logger.info("Lifecycle controller stopped")

    def _control_loop(self):
        """Main control loop with error handling"""
        while self._running:
            try:
                self._check_idle_instances()
                self._auto_scale()
                self._health_check()
            except Exception as e:
                logger.error(f"Control loop error: {str(e)}")
            time.sleep(30)

    def _check_idle_instances(self):
        """Cleanup instances exceeding max idle time"""
        policy = self.policies['default']
        for instance in self.model_manager.list_instances():
            idle_time = time.time() - instance['last_used']
            if idle_time > policy.max_idle_time:
                logger.info(f"Reclaiming idle instance {instance['model_id']}")
                self.model_manager.destroy_instance(instance['model_id'])

    def _auto_scale(self):
        """Adjust instance count based on resource utilization"""
        stats = self.monitor.get_current_stats()
        
        for policy_name, policy in self.policies.items():
            if policy_name == 'default':
                continue
                
            current = self._count_instances(policy.model_type)
            target = self._calculate_desired_count(policy, stats)
            
            if current < target:
                self._scale_up(policy)
            elif current > target:
                self._scale_down(policy, current - target)

    def _count_instances(self, model_type: str) -> int:
        """Count instances of specific model type"""
        return len([
            i for i in self.model_manager.list_instances()
            if i['framework'] == model_type
        ])

    def _calculate_desired_count(self, policy: ScalingPolicy, stats: dict) -> int:
        """Calculate required instances based on metrics"""
        gpu_util = stats['gpus'][0]['utilization']
        mem_util = stats['system']['memory_used'] / stats['system']['memory_total'] * 100
        
        if gpu_util > policy.target_gpu_util or mem_util > policy.target_mem_util:
            return min(policy.max_instances, self._count_instances(policy.model_type) + 1)
        else:
            return max(policy.min_instances, self._count_instances(policy.model_type) - 1)

    def _health_check(self):
        """Handle system health alerts"""
        health = self.monitor.check_health()
        for alert in health['alerts']:
            if alert['type'] == 'GPU_OVERHEAT':
                logger.warning(f"Handling GPU alert: {alert['message']}")
                self._reduce_gpu_load()

    def _reduce_gpu_load(self):
        """Mitigate GPU pressure"""
        gpu_instances = [
            i for i in self.model_manager.list_instances()
            if i['device'] == 'cuda'
        ]
        if gpu_instances:
            self.model_manager.destroy_instance(gpu_instances[0]['model_id'])

# Testing example
if __name__ == "__main__":
    from model_config import ModelConfig, ModelFramework
    from model_manager import ModelManager
    from resource_monitor import ResourceMonitor
    
    # Test setup
    config = ModelConfig(
        framework=ModelFramework.YOLO,
        model_path="/home/yixin/study/models/yolo_model/detection/yolo12x.pt",
        device="cuda"
    )
    
    manager = ModelManager()
    monitor = ResourceMonitor(manager)
    policies = {
        "default": ScalingPolicy(max_idle_time=5),
        "resnet": ScalingPolicy(
            model_type="pytorch",
            min_instances=1,
            max_instances=3
        ),
        "yolo": ScalingPolicy(
            model_type="yolo",
            min_instances=1,
            max_instances=10
        )
    }
    
    controller = LifecycleController(manager, monitor, policies)
    
    try:
        # Create initial instance
        
        for i in range(5):
            manager.create_instance(config)
        print("Initial instances:", len(manager.list_instances()))
        
        # Start controller
        controller.start()
        
        # Simulate idle timeout
        time.sleep(10)
        print("After idle cleanup:", len(manager.list_instances()))
        
        # Test scaling
        monitor._collect_gpu_stats = lambda: {'0': {'utilization': 90}}  # Mock high util
        time.sleep(35)
        print("After scaling:", len(manager.list_instances()))
        
    finally:
        controller.stop()