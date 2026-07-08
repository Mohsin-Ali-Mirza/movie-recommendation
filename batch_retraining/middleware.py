import asyncio
from fastapi import FastAPI, BackgroundTasks

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

import os
import random

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = FastAPI()

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
client = MlflowClient()
experiment_name = "Weak Classifier Experiment"
mlflow.set_experiment(experiment_name)

df = pd.read_csv("./dataset/dataset.csv")
MODEL_NAME = "Decision Stump"

get_latest_champion_id = None
get_latest_challenger_id = None

def user_change(df):
    new_row = [[random.randrange(1, 10), random.uniform(4.3, 7.9), random.uniform(2.0, 4.4), 
                random.uniform(1.0, 6.9), random.uniform(0.1, 2.5), "Iris-setosa"]]
    df = pd.concat([df, pd.DataFrame(new_row, columns=df.columns)], ignore_index=True)
    return df

def preprocessing(df):
    X = df.iloc[:, 1:5]
    y = df["Species"]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return train_test_split(X, y, test_size=0.3, random_state=42)

def find_tag_version(model_versions, tag_run_id):
    for mv in model_versions:
        if mv.run_id == tag_run_id:
            return mv.version
    return None

async def train_and_log_model():
    if mlflow.active_run():
        mlflow.end_run()  # End the active run

    global df, get_latest_champion_id, get_latest_challenger_id

    df = user_change(df)
    X_train, X_test, y_train, y_test = preprocessing(df)

    model = DecisionTreeClassifier(max_depth=1, random_state=42)

    run = await asyncio.to_thread(mlflow.start_run, run_name=MODEL_NAME)
    
    await asyncio.to_thread(model.fit, X_train, y_train)
    y_pred = await asyncio.to_thread(model.predict, X_test)
    signature = infer_signature(X_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Log Model & Metrics
    await asyncio.to_thread(mlflow.log_param, "max_depth", model.get_depth())
    await asyncio.to_thread(mlflow.log_metric, "accuracy", accuracy)
    await asyncio.to_thread(mlflow.sklearn.log_model, model, MODEL_NAME, signature=signature)

    # Register Model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    model_version = await asyncio.to_thread(mlflow.register_model, model_uri, MODEL_NAME)

    # Fetch top 2 models asynchronously
    best_models = await asyncio.to_thread(
        mlflow.search_runs, experiment_names=[experiment_name], order_by=["metrics.accuracy DESC"], max_results=2
    )

    if len(best_models) == 1:
        await asyncio.to_thread(client.set_registered_model_alias, MODEL_NAME, "Champion", model_version.version)
        await asyncio.to_thread(client.set_registered_model_alias, MODEL_NAME, "Challenger", model_version.version)
        return

    champion_run_id = best_models.iloc[0]["run_id"]
    challenger_run_id = best_models.iloc[1]["run_id"]

    model_versions = await asyncio.to_thread(client.search_model_versions, f"name='{MODEL_NAME}'")
    champion_version = find_tag_version(model_versions, champion_run_id)
    challenger_version = find_tag_version(model_versions, challenger_run_id)

    await asyncio.to_thread(client.set_registered_model_alias, MODEL_NAME, "Champion", champion_version)
    await asyncio.to_thread(client.set_registered_model_alias, MODEL_NAME, "Challenger", challenger_version)

    get_latest_champion_id, get_latest_challenger_id = champion_run_id, challenger_run_id


# Global Variables
training_counter = 0
training_lock = False
THRESHOLD = 10

async def train_model():
    global training_lock, training_counter
    training_lock = True
    print("Training started...")

    await train_and_log_model()

    print("Training completed. Updating the model...")

    training_counter = 0
    training_lock = False

@app.post("/train")
async def handle_train_request(background_tasks: BackgroundTasks):
    global training_counter, training_lock

    best_models = await asyncio.to_thread(
        mlflow.search_runs, experiment_names=[experiment_name], order_by=["metrics.accuracy DESC"], max_results=2
    )
    get_latest_champion_id = best_models.iloc[0]["run_id"]
    get_latest_challenger_id = best_models.iloc[1]["run_id"]

    if training_lock:
        return {"message": f"Training already in progress. Using old model. Current Counter: {training_counter}",
                "champion_run_id": get_latest_champion_id,
                "challenger_run_id": get_latest_challenger_id}

    training_counter += 1

    if training_counter >= THRESHOLD and not training_lock:
        training_counter = 0
        background_tasks.add_task(train_model)
        return {"message": "Training triggered.",
                "champion_run_id": get_latest_champion_id,
                "challenger_run_id": get_latest_challenger_id}

    return {"message": f"Request received. Current counter: {training_counter}",
            "champion_run_id": get_latest_champion_id,
            "challenger_run_id": get_latest_challenger_id}

@app.post("/inference")
async def inference():
    return {"message": "Inference using current model."}
