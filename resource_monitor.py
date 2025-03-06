import time
import threading
import psutil
from pydantic import BaseModel
from typing import Dict, List, Optional
from model_manager import ModelManager
from base_model import BaseAIModel
import logging
import torch
import asyncio
import functools
import os

try:
    import pynvml
except ImportError:
    # give user instruction to install nvidia-ml-py
    print("nvidia-ml-py not found. Please install it using the following command:"+"\npython3 -m pip install nvidia-ml-py")
    
    pynvml = None

# Ensure CUDA is initialized in the main thread before launching executor
if torch.cuda.is_available():
    torch.cuda.init()  # Force CUDA initialization in the main thread

os.environ["KINETO_DAEMON_INIT_DELAY_S"]="3"

"""
@author: Yixin Huang
@last update: 2025-03-06 16:26
@tested: True

"""

logger = logging.getLogger("ResourceMonitor")

"""
The resource_monitor.py file is designed to monitor system and GPU resources in a multi - model service environment.
"""

class GPUStats(BaseModel):
    utilization: float = 0.0
    memory_used: float = 0.0  # MB
    memory_total: float = 0.0  # MB
    temperature: float = 0.0  # °C

class SystemStats(BaseModel):
    cpu_percent: float = 0.0
    memory_used: float = 0.0  # MB
    memory_total: float = 0.0  # MB
    disk_usage: float = 0.0  # percent

class ModelInstanceStats(BaseModel):
    model_id: str
    framework: str
    device: str
    inference_count: int = 0
    avg_latency: float = 0.0  # ms
    memory_usage: float = 0.0  # MB

class ResourceMonitor:
    def __init__(self, model_manager: ModelManager, interval: float = 5.0):
        self.model_manager = model_manager
        self.interval = interval
        self._running = False
        self._monitor_thread = None
        
        # Hardware resources
        self.system_stats = SystemStats()
        self.gpu_stats: Dict[int, GPUStats] = {}
        
        # Model instance tracking
        self.instance_stats: Dict[str, ModelInstanceStats] = {}
        
        # Historical data (last 60 samples)
        self.history: List[dict] = []
        self.max_history = 60
        
        # Initialize NVML if available
        self.nvml_initialized = False
        if pynvml:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except Exception as e:
                logger.warning(f"NVML initialization failed: {str(e)}")

    async def start(self):
        """Start monitoring task"""
        if not self._running:
            self._running = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("Resource monitoring started")

    async def stop(self):
        """Stop monitoring task"""
        if self._running:
            self._running = False
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
            logger.info("Resource monitoring stopped")

    async def _monitor_loop(self):
        """Async monitoring loop"""
        while self._running:
            try:
                await self._collect_system_stats()
                await self._collect_gpu_stats()
                await self._collect_instance_stats()
                self._store_history()
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
    async def _collect_system_stats(self):
        """Collect system stats with async execution"""
        loop = asyncio.get_running_loop()
        mem = await loop.run_in_executor(None, psutil.virtual_memory)
        cpu_percent = await loop.run_in_executor(None, psutil.cpu_percent)
        disk = await loop.run_in_executor(None, psutil.disk_usage, '/')
        
        self.system_stats = SystemStats(
            cpu_percent=cpu_percent,
            memory_used=mem.used / (1024**2),
            memory_total=mem.total / (1024**2),
            disk_usage=disk.percent
        )

    async def _collect_gpu_stats(self):
        """Collect GPU stats with async execution"""
        if not self.nvml_initialized:
            return

        self.gpu_stats.clear()
        loop = asyncio.get_running_loop()
        
        def collect_gpu_stats_sync():
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )

                    self.gpu_stats[i] = GPUStats(
                        utilization=util.gpu,
                        memory_used=mem_info.used / (1024**2),
                        memory_total=mem_info.total / (1024**2),
                        temperature=temp
                    )
            except pynvml.NVMLError as e:
                logger.error(f"GPU monitoring error: {str(e)}")

        await loop.run_in_executor(None, collect_gpu_stats_sync)


    async def _collect_instance_stats(self):
        """Collect model stats with async execution"""
        loop = asyncio.get_running_loop()
        current_instances = await loop.run_in_executor(None, self.model_manager.list_instances)
        new_stats = {}
        
        for inst in current_instances:
            stats = self.instance_stats.get(inst["model_id"], 
                ModelInstanceStats(
                    model_id=inst["model_id"],
                    framework=inst["framework"],
                    device=inst["device"],
                    memory_usage=inst['memory_usage']
                )
            )
            get_instance_partial = functools.partial(self.model_manager.get_instance, only_model=False)
            # instance = await loop.run_in_executor(None, self.model_manager.get_instance, inst["model_id"])
            instance = await loop.run_in_executor(None, get_instance_partial, inst["model_id"])
            stats.inference_count = instance.use_count
            new_stats[inst["model_id"]] = stats
            
        self.instance_stats = new_stats



    def _store_history(self):
        """Store current state in history buffer"""
        snapshot = {
            "timestamp": time.time(),
            "system": self.system_stats.model_dump(),
            "gpus": {str(k): v.model_dump() for k, v in self.gpu_stats.items()},
            "instances": [v.model_dump() for v in self.instance_stats.values()]
        }
        
        # Limit history to last self.max_history samples
        self.history = (self.history + [snapshot])[-self.max_history:]


    def get_current_stats(self) -> dict:
        """Get current monitoring data"""
        return {
            "system": self.system_stats,
            "gpus": self.gpu_stats,
            "instances": self.instance_stats
        }

    def check_health(self) -> dict:
        """Perform health checks and return status"""
        alerts = []
        
        # GPU temperature check
        for gpu_id, stats in self.gpu_stats.items():
            if stats.temperature > 85:
                alerts.append({
                    "type": "GPU_OVERHEAT",
                    "message": f"GPU {gpu_id} temperature {stats.temperature}°C"
                })
        
        # Memory overload check
        if self.system_stats.memory_total > 0 and self.system_stats.memory_used / self.system_stats.memory_total > 0.9:
            alerts.append({
                "type": "HIGH_MEMORY_USAGE",
                "message": f"System memory usage {self.system_stats.memory_used:.1f}/{self.system_stats.memory_total:.1f} MB"
            })
        
        return {"status": "OK" if not alerts else "WARNING", "alerts": alerts}

# Usage example
if __name__ == "__main__":
    # from model_manager import ModelManager
    # from model_config import ModelConfig, ModelFramework
    
    # # Initialize dependencies
    # manager = ModelManager()
    # monitor = ResourceMonitor(manager, interval=1.0)
    
    # # Create test model
    # config = ModelConfig(
    #     framework=ModelFramework.PYTORCH,
    #     device="cuda",
    #     model_path="test.pth"
    # )
    # model_id = manager.create_instance(config)
    
    # # Start monitoring
    # monitor.start()
    
    # # Simulate some activity
    # try:
    #     for _ in range(3):
    #         print("Current stats:", monitor.get_current_stats())
    #         print("Health status:", monitor.check_health())
    #         time.sleep(2)
    # finally:
    #     monitor.stop()
    #     manager.destroy_instance(model_id)


    import asyncio
    from model_manager import ModelManager
    from model_config import ModelConfig, ModelFramework

    async def main():
        # Initialize dependencies
        manager = ModelManager()
        monitor = ResourceMonitor(manager, interval=1.0)

        # Create test model
        config = ModelConfig(
            framework=ModelFramework.YOLO,
            device="cuda",
            model_path="/home/yixin/study/models/yolo_model/detection/yolo12x.pt"
        )
        
        # Run blocking operations in executor
        loop = asyncio.get_running_loop()
        model_id = await loop.run_in_executor(None, manager.create_instance, config)

        # Start monitoring
        await monitor.start()

        # Simulate some activity
        try:
            for _ in range(3):
                print("Current stats:", monitor.get_current_stats())
                print("Health status:", monitor.check_health())
                await asyncio.sleep(2)
        finally:
            await monitor.stop()
            await loop.run_in_executor(None, manager.destroy_instance, model_id)

    asyncio.run(main())