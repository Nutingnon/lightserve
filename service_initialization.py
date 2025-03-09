from lightserve import ModelManager, LifecycleController, ResourceMonitor

# Load configuration
config = load_config("lightserve/config.yaml")

# Initialize core components
manager = ModelManager(
    allocation_strategy=config.resources["gpu_allocation_strategy"]
)
monitor = ResourceMonitor(manager, interval=config.monitoring["health_check_interval"])
controller = LifecycleController(
    manager,
    monitor,
    policies={m.name: m.scaling for m in config.models}
)

# Deploy configured models
for model_cfg in config.models:
    model_config = ModelConfig(
        framework=model_cfg.framework,
        model_path=model_cfg.model_path,
        device=model_cfg.get("device", config.defaults["device"]),
        metadata=model_cfg.resources
    )
    manager.create_instance(model_config)