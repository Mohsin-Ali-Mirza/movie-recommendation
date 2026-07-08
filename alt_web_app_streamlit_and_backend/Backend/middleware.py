import asyncio
import os
import time
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from fastapi import FastAPI, BackgroundTasks

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

app = FastAPI()

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
client = MlflowClient()
experiment_name = "SVD Movie Recommender"
mlflow.set_experiment(experiment_name)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "svd_model.pkl")
NUM_FEATURES = 50

async def prepare_user_item_matrix(ratings):
    df = ratings.groupby(['userId', 'movieId'], as_index=False).agg({'rating': 'mean'})
    user_movie_ratings = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_means = user_movie_ratings.mean(axis=1)
    return user_movie_ratings, user_means

async def train_svd_model(ratings):
    start_time = time.time()
    user_movie_ratings, user_means = await prepare_user_item_matrix(ratings)
    movie_ids = user_movie_ratings.columns.tolist()
    normalized_ratings = user_movie_ratings.sub(user_means, axis=0)
    sparse_ratings = csr_matrix(normalized_ratings.values)
    U, sigma, Vt = svds(sparse_ratings, k=min(NUM_FEATURES, sparse_ratings.shape[1] - 1))
    sigma = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_means.values[:, np.newaxis]
    elapsed = time.time() - start_time

    model_data = {
        "U": U,
        "sigma": sigma,
        "Vt": Vt,
        "user_means": user_means,
        "movie_ids": movie_ids
    }
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"SVD training completed in {elapsed:.2f} seconds!")
    
    return predicted_ratings

async def train_and_log_svd_model():
    ratings = pd.read_csv("./dataset/ratings.csv")  # Replace with actual ratings data source
    predicted_ratings = await train_svd_model(ratings)
    
    run = await asyncio.to_thread(mlflow.start_run, run_name="SVD Model")
    signature = infer_signature(ratings, predicted_ratings)
    
    await asyncio.to_thread(mlflow.log_param, "num_features", NUM_FEATURES)
    await asyncio.to_thread(mlflow.sklearn.log_model, predicted_ratings, "SVD_Model", signature=signature)
    
    print("SVD Model training and logging completed.")

# Global Variables
training_lock = False
THRESHOLD = 10
training_counter = 0

async def train_model():
    global training_lock, training_counter
    training_lock = True
    print("Training SVD model started...")

    await train_and_log_svd_model()

    print("SVD model training completed.")
    training_counter = 0
    training_lock = False

@app.post("/train")
async def handle_train_request(background_tasks: BackgroundTasks):
    global training_counter, training_lock
    
    if training_lock:
        return {"message": "Training already in progress. Using old model."}

    training_counter += 1
    
    if training_counter >= THRESHOLD and not training_lock:
        training_counter = 0
        background_tasks.add_task(train_model)
        return {"message": "Training triggered."}

    return {"message": f"Request received. Current counter: {training_counter}"}

@app.post("/inference")
async def inference():
    return {"message": "Inference using current SVD model."}