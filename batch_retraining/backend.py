import requests
import mlflow
import threading
import mlflow.pyfunc
from middleware import preprocessing
import pandas as pd
import sys

MIDDLEWARE_URL = "http://localhost:8000"
model_name = "Decision Stump"
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

current_champion_id = "9972cbee493e4633acea801222ae81ea"
active_model = None  # Model currently used for inference
challenger_model = None  # Backup model for inference if a new champion is loading
next_model = None  # Stores the newly downloaded model for use in the next iteration

df = pd.read_csv('./dataset/dataset.csv')
X_train, X_test, y_train, y_test = preprocessing(df)

def inference(model):
    """Perform inference using the given model."""
    if model:
        print("Model Prediction:", model.predict(X_test))
    else:
        print("No model available for inference.")

def load_new_champion_model(new_champion_run_id):
    """Loads the new champion model asynchronously and updates it for the next iteration."""
    global next_model

    try:
        print(f"Fetching new champion model (ID: {new_champion_run_id}) in the background...")
        new_model = mlflow.pyfunc.load_model(f"runs:/{new_champion_run_id}/{model_name}")
        
        # Store it for use in the next iteration (do NOT activate it yet)
        next_model = new_model
        print(f"New Champion Model (ID: {new_champion_run_id}) downloaded and ready for next iteration.")

    except Exception as e:
        print(f"Error loading new champion model: {e}")

def request_training():
    """Send a synchronous request to the middleware for training."""
    global current_champion_id, active_model, challenger_model, next_model
    
    response = requests.post(f"{MIDDLEWARE_URL}/train")
    
    if response.status_code == 200:
        result = response.json()
        new_champion_run_id = result.get("champion_run_id")

        # First-time model load
        if current_champion_id is None:
            print("Loading initial champion model...")
            active_model = mlflow.pyfunc.load_model(f"runs:/{new_champion_run_id}/{model_name}")
            current_champion_id = new_champion_run_id

        # New Best Model Detected
        elif new_champion_run_id != current_champion_id:
            print(f"New Champion Detected! Current ID: {current_champion_id}, New ID: {new_champion_run_id}")
            sys.stdout.flush()
            
            # Set current champion as challenger (backup)
            challenger_model = active_model  
            
            # Fetch new champion in the background
            threading.Thread(target=load_new_champion_model, args=(new_champion_run_id,), daemon=True).start()

        # If the new model is downloaded, switch to it
        if next_model is not None:
            print(f"Switching to newly downloaded model for inference.")
            active_model = next_model
            current_champion_id = new_champion_run_id
            next_model = None  # Reset for next iteration

        print(f"Champion ID: {current_champion_id}, Active Model ID: {current_champion_id}")
        inference(active_model)  # Perform inference using the currently active model

    else:
        print("Error:", response.text)

if __name__ == "__main__":
    selection = 1
    while selection:
        selection = input("Enter 1 to send another request, 0 to finish: ")
        if selection == "1":
            request_training()
        else:
            print("Exiting...")
            break
