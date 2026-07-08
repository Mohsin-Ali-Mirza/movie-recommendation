
from typing import Union, Dict, Any, List
import pandas as pd
from fastapi import FastAPI, Body, Depends, HTTPException
import numpy as np
from content_bases_model import load_data, get_unique_genres, recommend_movies_content, create_user_profile
from svd_bases_recomendor import train_svd, recommend_movies_svd, search_movie_by_title_svd
from hybrid_based_recomendor import recommend_movies_hybrid
import requests
import streamlit as st

# In a real application, these might be loaded at startup or managed more dynamically.
movies_path = r"dataset\movies.csv"
ratings_path = r"dataset\ratings.csv"
tags_path = r"dataset\tags.csv"
df_movies, df_ratings, df_tags = load_data(movies_path, ratings_path, tags_path)
predicted_ratings, user_movie_ratings, movie_ids, user_means = train_svd(df_ratings)
all_unique_genres = get_unique_genres(df_movies)


def calculate_mapk_func(relevant_movie_ids: List[int], recommended_movie_ids: List[int], k: int = 5) -> float:
    """Calculates Mean Average Precision at k (MAP@K)."""
    apk_score = calculate_apk_func(recommended_movie_ids, relevant_movie_ids, k)
    return apk_score

def calculate_apk_func(recommended_ids, relevant_ids, k=5):
    ap = 0.0
    num_hits = 0.0
    for i, movie_id in enumerate(recommended_ids[:k], start=1):
        if movie_id in relevant_ids:
            num_hits += 1.0
            ap += num_hits / i
    if not relevant_ids:
        return 0.0
    return ap / min(len(relevant_ids), k)


def get_relevant_movie_ids(user_ratings: Dict[str, int], df_movies: pd.DataFrame) -> set[int]:
    """Extracts relevant movie IDs from user ratings (ratings >= 3)."""
    relevant_movies = set()
    for movie_name, rating in user_ratings.items():
        if rating >= 3:
            movie_entry = search_movie_by_title_svd(movie_name, df_movies)
            if movie_entry is not None:
                relevant_movies.add(movie_entry['movieId'])
    return relevant_movies

def evaluate_model_performance(user_ratings: Dict[str, int], model_type: str, k: int = 5) -> Dict[str, float]:
    """Evaluates the performance of a recommendation model."""
    relevant_movie_ids = get_relevant_movie_ids(user_ratings, df_movies)
    if model_type == "content-based":
        df_user_profile = create_user_profile(df_movies, user_ratings)
        recommendations_df = recommend_movies_content(df_movies, df_user_profile, all_unique_genres)
        if recommendations_df is None or recommendations_df.empty:
            mapk_score = 0.0
            recommendation_ids = []
        else:
            recommendation_ids = recommendations_df['movieId'].astype(int).tolist()
            mapk_score = calculate_mapk_func(relevant_movie_ids, recommendation_ids[:k], k)

    elif model_type == "svd":
        user_ratings_mapped = {}
        for movie_name, rating in user_ratings.items():
            movie_entry = search_movie_by_title_svd(movie_name, df_movies)
            if movie_entry is not None:
                user_ratings_mapped[movie_entry['movieId']] = rating
        user_ratings_df_for_svd = pd.DataFrame(list(user_ratings_mapped.items()), columns=['movieId', 'rating'])
        user_ratings_df_for_svd['movieId'] = user_ratings_df_for_svd['movieId'].astype(int)
        user_profile = {'userid': 1, 'rated_movies': user_ratings_df_for_svd} # Assuming user_id=1
        recommendations_df = recommend_movies_svd(user_profile, predicted_ratings, user_movie_ratings, movie_ids, df_movies, k)

        if recommendations_df is None or recommendations_df.empty:
            mapk_score = 0.0
            recommendation_ids = []
        else:
            recommendation_ids = recommendations_df['movieId'].astype(int).tolist()
            mapk_score = calculate_mapk_func(relevant_movie_ids, recommendation_ids[:k], k)

    elif model_type == "hybrid":
        df_user_profile = create_user_profile(df_movies, user_ratings)
        recommendations_df = recommend_movies_hybrid(user_ratings, n_recommendations=k) # Hybrid takes movie dict directly
        if recommendations_df is None or recommendations_df.empty:
            mapk_score = 0.0
            recommendation_ids = []
        else:
            recommendation_ids = [rec['movieId'] for rec in recommendations_df] # Extract movieIds from list of dict
            mapk_score = calculate_mapk_func(relevant_movie_ids, recommendation_ids[:k], k)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return {"mapk_score": mapk_score, "relevant_movie_count": len(relevant_movie_ids), "recommended_movie_count": len(recommendation_ids)}
