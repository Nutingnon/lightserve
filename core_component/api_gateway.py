from fastapi import FastAPI, HTTPException
from model_manager import ModelManager
from lightserve.core_component.model_factory import BaseAIModel
from typing import Dict, Any
import logging

app = FastAPI()
logger = logging.getLogger("APIGateway")

"""
@author: yixin.huang
@time: 2025-03-07 15:43:00
@tested: False
"""



async def lifespan(app: FastAPI):
    """
    Lifespan event handler for the FastAPI application.
    This function is called when the application starts up and shuts down.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    app.state.model_manager = ModelManager()
    app.state.instance_selector = RoundRobinSelector()
    yield

@app.post("/predict/{model_type}")
async def predict(model_type: str, data: Dict[str, Any]):
    """
    Endpoint to make predictions using a specific model type.

    Args:
        model_type (str): The type of the model to use for prediction.
        data (Dict[str, Any]): The input data for the prediction.

    Returns:
        Dict[str, Any]: A dictionary containing the prediction result.

    Raises:
        HTTPException: If an error occurs during prediction or no instances are available.
    """
    try:
        instance_id = app.state.instance_selector.get_instance(model_type)
        model = app.state.model_manager.get_instance(instance_id)
        result = model.predict(data)
        return {"result": result}
    except ValueError as e:
        # Handle the case where no instances are available
        logger.error(f"No instances available: {str(e)}")
        raise HTTPException(500, detail=str(e))
    except Exception as e:
        # Handle other prediction errors
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(500, detail=str(e))

class RoundRobinSelector:
    """
    A selector that uses a round-robin algorithm to select model instances.
    """
    def __init__(self):
        self.counters: Dict[str, int] = {}

    def get_instance(self, model_type: str) -> str:
        """
        Select a model instance using the round-robin algorithm.

        Args:
            model_type (str): The type of the model to select an instance for.

        Returns:
            str: The ID of the selected model instance.

        Raises:
            ValueError: If no instances are available for the specified model type.
        """
        instances = [
            i for i in app.state.model_manager.list_instances()
            if i['framework'] == model_type
        ]
        
        if not instances:
            raise ValueError(f"No instances for {model_type}")
            
        idx = self.counters.get(model_type, 0) % len(instances)
        self.counters[model_type] = idx + 1
        return instances[idx]['model_id']