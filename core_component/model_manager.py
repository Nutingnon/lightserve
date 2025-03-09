from typing import Dict, Optional, List
from model_config import ModelConfig, ModelFramework
from base_model import BaseAIModel, ModelFactory
import time
import logging
import torch
from pydantic import BaseModel
from enum import Enum, auto
import psutil


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelManager")

"""
@author: Yixin Huang
@last update: 2025-03-07 14:46
@tested: True
"""

class AllocationStrategy(Enum):
    SPREAD = auto()  # Distribute across GPUs
    PACK = auto()    # Consolidate on fewest GPUs
    BALANCED = auto() # Balance memory and utilization

class GPUResource(BaseModel):
    total_memory: float  # In MB
    used_memory: float = 0.0
    utilization: float = 0.0  # 0-100%

class ModelInstance:
    def __init__(self, model: BaseAIModel):
        self.model = model
        self.last_used: float = time.time()
        self.use_count: int = 0
        self.gpu_id: Optional[int] = None

    def update_usage(self):
        self.last_used = time.time()
        self.use_count += 1

class GPUAllocator:
    def __init__(self):
        self.gpus: Dict[int, GPUResource] = self._detect_gpus()
        
    def _detect_gpus(self) -> Dict[int, GPUResource]:
        """Initialize GPU resources"""
        gpus = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.get_device_properties(i).total_memory
                gpus[i] = GPUResource(total_memory=mem / (1024**2))
        return gpus

    def allocate_gpu(self, required_mem: float) -> Optional[int]:
        """Find suitable GPU using best-fit strategy"""
        for gpu_id, gpu in sorted(self.gpus.items(), 
                                key=lambda x: x[1].used_memory):
            available = gpu.total_memory - gpu.used_memory
            if available >= required_mem:
                gpu.used_memory += required_mem
                return gpu.total_memory, available, gpu_id
        return gpu.total_memory, available, None

    def release_gpu(self, gpu_id: int, allocated_mem: float):
        if gpu_id in self.gpus:
            self.gpus[gpu_id].used_memory -= allocated_mem

class ModelManager:
    def __init__(self, allocation_strategy: AllocationStrategy = AllocationStrategy.BALANCED):
        self.instances: Dict[str, ModelInstance] = {}
        self.allocator = GPUAllocator()
        self.strategy = allocation_strategy
        self.model_mem_requirements: Dict[str, float] = {}  # Model type to memory MB

    def create_instance(self, config: ModelConfig) -> str:
        """Create new model instance with resource allocation"""
        # Calculate memory requirements
        required_mem = self._estimate_memory(config)
        
        # Allocate resources
        gpu_id = None
        if config.device == "cuda":
            # memory is in MB
            total_memory, available, gpu_id = self.allocator.allocate_gpu(required_mem)
            logger.info(f"The required memory is {required_mem}, GPU ID: {gpu_id}, Available Memory: {available}\n")
            if gpu_id is None:  # Fallback to CPU if no GPU available
                config = config.model_copy(update={"device": "cpu"})
                logger.warning(f"Insufficient GPU memory (Total Memory: {total_memory}), falling back to CPU for {config.framework} The required memory is: {required_mem} MB, but {available} MB left\n")

        # Create model instance
        model = ModelFactory.create_model(config)
        model.load(config.model_path)
        
        # Store instance
        instance = ModelInstance(model)
        instance.gpu_id = gpu_id
        self.instances[model.model_id] = instance
        # self.model_mem_requirements[config.framework] = required_mem
        # change the key to model_id instead of framework
        self.model_mem_requirements[model.model_id] = required_mem
        
        logger.info(f"Created instance {model.model_id} on {config.device.upper()}")
        return model.model_id

    def destroy_instance(self, model_id: str):
        """Destroy instance and release resources"""
        if model_id not in self.instances:
            raise KeyError(f"Instance {model_id} not found")
        
        instance = self.instances.pop(model_id)
        if instance.gpu_id is not None:
            # required_mem = self.model_mem_requirements.get(
            #     instance.model.config.framework, 0
            # )
            # change the key to model_id instead of framework

            required_mem = self.model_mem_requirements.get(instance.model.model_id, 0)
            self.allocator.release_gpu(instance.gpu_id, required_mem)
        
        logger.info(f"Destroyed instance {model_id}")

    def get_instance(self, model_id: str, only_model=True) -> BaseAIModel:
        """Retrieve model instance with usage tracking"""
        instance = self.instances[model_id]
        instance.update_usage()
        if only_model:
            return instance.model
        else:
            return instance


    def cleanup_idle_instances(self, max_idle_seconds: float = 600):
        """Destroy instances exceeding idle time"""
        now = time.time()
        to_remove = [
            model_id for model_id, instance in self.instances.items()
            if (now - instance.last_used) > max_idle_seconds
        ]
        
        for model_id in to_remove:
            self.destroy_instance(model_id)

    # def _estimate_memory(self, config: ModelConfig) -> float:
    #     """Estimate GPU memory requirements (simplified example)"""
    #     # In real implementation, use model profiling
    #     framework_mem = {
    #         ModelFramework.PYTORCH: 1500,  # MB
    #         ModelFramework.ONNX: 1000,
    #         ModelFramework.YOLO: 1000
    #     }
    #     return framework_mem.get(config.framework, 2000)

    def _estimate_memory(self, config: ModelConfig) -> float:
        """Estimate GPU memory requirements using profiling"""
        if config.framework in [ModelFramework.PYTORCH, ModelFramework.YOLO]:
            # Create a dummy model instance
            model = ModelFactory.create_model(config)

            # Get the memory usage before the inference
            use_cuda = False
            if torch.cuda.is_available():
                use_cuda = True
                torch.cuda.reset_peak_memory_stats()
                before_memory = torch.cuda.memory_allocated()
            else:
                # Detect system memory (RAM) usage
                memory = psutil.virtual_memory()
                before_memory = memory.used
            model.load(config.model_path)
            # Create a dummy input
            dummy_input = torch.rand(1, 3, 640, 640).to(config.device)  # Adjust shape according to your model
            
            if torch.cuda.is_available():
                # Run the model with CUDA profiler
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                    profile_memory=True,
                    with_stack=True
                ) as prof:
                    model.predict(dummy_input)
                memory_usage = (prof.key_averages().total_average().device_memory_usage - before_memory) / (1024 ** 2)  # Convert to MB

                # release the model
                del model
                torch.cuda.empty_cache()

            else:
                # Run the model with CPU profiler
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    profile_memory=True,
                    with_stack=True

                ) as prof:
                    model(dummy_input)

                memory_usage = (prof.key_averages().total_average().cpu_memory_usage - before_memory) / (1024 ** 2)  # Convert to MB
                # release the model
                del model

            return memory_usage
        
        
        # Fallback to the previous method for other frameworks
        framework_mem = {
            ModelFramework.ONNX: 1000,
        }
        return framework_mem.get(config.framework, 2000)



    def list_instances(self) -> List[dict]:
        """Get status of all instances"""
        return [{
            "model_id": inst.model.model_id,
            "framework": inst.model.config.framework.value,
            "device": inst.model.config.device,
            "last_used": inst.last_used,
            "gpu_id": inst.gpu_id,
            "memory_usage": self.model_mem_requirements.get(inst.model.model_id, 0),  # Add memory usage if available
        } for inst in self.instances.values()]

# Testing
if __name__ == "__main__":
    # Test configuration
    config = ModelConfig(
        framework=ModelFramework.YOLO,
        device="cuda",
        model_path="/home/yixin/study/models/yolo_model/detection/yolo12x.pt",
        max_batch_size=32
    )

    # Initialize manager
    manager = ModelManager()
    
    # Test instance creation
    model_id1 = manager.create_instance(config)
    model_id2 = manager.create_instance(config)
    
    # Test instance retrieval
    try:
        model = manager.get_instance(model_id1)
        print(f"Prediction on {model.config.device}: {model.predict(torch.rand(1, 3, 640 ,640).to(config.device))}")
    except KeyError:
        print("Instance not found")
    
    # Test cleanup
    print("Before cleanup:", len(manager.list_instances()))
    time.sleep(10)
    manager.cleanup_idle_instances(max_idle_seconds=3)
    print("After cleanup:", len(manager.list_instances()))
    
    # Test resource release
    manager.destroy_instance(model_id2)
    print("Remaining instances:", [m.model_id for m in manager.instances.values()])