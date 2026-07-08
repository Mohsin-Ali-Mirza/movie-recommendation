import pandas as pd
import numpy as np
import re
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import time
from logging import getLogger
from typing import Dict, Any
from fastapi import FastAPI
import joblib
import os
import pickle
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "svd_model.pkl")
# Setup logging
AppLogger = getLogger("SVDBasedRecommender")



def search_movie_by_title_svd(movie_title, df_movies):
    """Searches for a movie by its title (case-insensitive)."""
    movie_title_escaped = re.escape(movie_title)
    movie_entry = df_movies[df_movies["title"].str.contains(movie_title_escaped, case=False, na=False, regex=True)]
    if not movie_entry.empty:
        return movie_entry.iloc[0]
    else:
        print(f"Movie '{movie_title}' not found.")
        return None

def prepare_user_item_matrix(ratings):
    """Prepare the user-item matrix."""
    df = ratings.groupby(['userId', 'movieId'], as_index=False).agg({'rating': 'mean'})
    user_movie_ratings = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_means = user_movie_ratings.mean(axis=1)
    return user_movie_ratings, user_means

def train_svd(ratings, num_features=50):
    """Train the SVD model."""
    AppLogger.info("Training SVD model...")
    start_time = time.time()
    user_movie_ratings, user_means = prepare_user_item_matrix(ratings)
    movie_ids = user_movie_ratings.columns.tolist()
    normalized_ratings = user_movie_ratings.sub(user_means, axis=0)
    sparse_ratings = csr_matrix(normalized_ratings.values)
    U, sigma, Vt = svds(sparse_ratings, k=min(num_features, sparse_ratings.shape[1] - 1))
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

    with open(MODEL_PATH ,"wb") as f:
        pickle.dump(model_data, f)
    AppLogger.info(f"SVD training completed in {elapsed:.2f} seconds!")
    return predicted_ratings, user_movie_ratings, movie_ids, user_means

def recommend_movies_svd(user_ratings, predicted_ratings, user_movie_ratings, movie_ids, movies_metadata, n_recommendations=5, filter_rated=True):
    """Recommend movies based on the SVD model."""
    user_id = user_ratings['userid']
    if user_id not in user_movie_ratings.index:
        print(f"User ID {user_id} not found in training data. Unable to recommend.")
        return pd.DataFrame([])
    
    user_idx = user_movie_ratings.index.get_loc(user_id)
    predicted_scores = predicted_ratings[user_idx]
    recommendations = pd.DataFrame({'id': movie_ids, 'predicted_rating': predicted_scores})
    
    # Ensure column names match before merging
    recommendations.rename(columns={'id': 'movieId'}, inplace=True)
    recommendations = recommendations.merge(movies_metadata[['movieId', 'title', 'genres']], on='movieId', how='left')
    
    if filter_rated:
        user_rated = set(user_ratings['rated_movies']['movieId'].values)
        recommendations = recommendations[~recommendations['movieId'].isin(user_rated)]
        
    recommendations = recommendations.sort_values(by='predicted_rating', ascending=False)
    recs = recommendations.head(n_recommendations).copy()
    recs['modelName'] = 'svd'
    return recs


def load_or_train_svd(ratings, num_features=50):
    if os.path.exists(MODEL_PATH):
        AppLogger.info("Loading pre-trained SVD model...")
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        
        U = model_data["U"]
        sigma = model_data["sigma"]
        Vt = model_data["Vt"]
        user_means = model_data["user_means"]
        movie_ids = model_data["movie_ids"]
        
        predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_means.values[:, np.newaxis]
        user_movie_ratings, user_means = prepare_user_item_matrix(ratings)
        AppLogger.info("Model loaded successfully.")
        return predicted_ratings, user_movie_ratings, movie_ids, user_means
    else:
        return train_svd(ratings, num_features)