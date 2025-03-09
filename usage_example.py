import uvicorn
import requests

# Assuming the API is running on localhost:8000
API_URL = "http://localhost:8000"

# Function to start the FastAPI application
def start_api():
    uvicorn.run("api_gateway:app", host="0.0.0.0", port=8000, reload=True)

# Function to make a prediction request
def make_prediction(model_type, data):
    url = f"{API_URL}/predict/{model_type}"
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.text}")
        return None

if __name__ == "__main__":
    import threading

    # Start the API in a separate thread
    api_thread = threading.Thread(target=start_api)
    api_thread.start()

    # Wait for the API to start
    import time
    time.sleep(5)

    # Example prediction data
    model_type = "example_model_type"
    data = {"input": "example_input"}

    # Make a prediction request
    result = make_prediction(model_type, data)
    if result:
        print(f"Prediction result: {result}")

    # Stop the API thread (this is a simple example, proper shutdown handling may be needed)
    api_thread.join()