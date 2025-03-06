from fastapi import FastAPI, HTTPException
from model_manager import ModelManager
from base_model import BaseAIModel
from typing import Dict, Any
import logging

app = FastAPI()
logger = logging.getLogger("APIGateway")

"""
@author: yixin.huang
@time: 2025-03-04 19:43:00
@validation: No
"""

@app.on_event("startup")
async def startup():
    app.state.model_manager = ModelManager()
    app.state.instance_selector = RoundRobinSelector()

@app.post("/predict/{model_type}")
async def predict(model_type: str, data: Dict[str, Any]):
    try:
        instance_id = app.state.instance_selector.get_instance(model_type)
        model = app.state.model_manager.get_instance(instance_id)
        result = model.predict(data)
        return {"result": result}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(500, detail=str(e))

class RoundRobinSelector:
    def __init__(self):
        self.counters: Dict[str, int] = {}

    def get_instance(self, model_type: str) -> str:
        instances = [
            i for i in app.state.model_manager.list_instances()
            if i['framework'] == model_type
        ]
        
        if not instances:
            raise ValueError(f"No instances for {model_type}")
            
        idx = self.counters.get(model_type, 0) % len(instances)
        self.counters[model_type] = idx + 1
        return instances[idx]['model_id']