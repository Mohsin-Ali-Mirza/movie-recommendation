from typing import Union, Dict, Any ,List
import pandas as pd
from fastapi import FastAPI , Body , Depends , HTTPException
from movies import suggested_random_movies
from content_bases_model import load_data ,  get_unique_genres , recommend_movies_content ,create_user_profile  
from svd_bases_recomendor import train_svd, recommend_movies_svd  , search_movie_by_title_svd , load_or_train_svd  # Import functions
from hybrid_based_recomendor import recommend_movies_hybrid
from feedback import evaluate_model_performance
import pickle
import os


movies_path = r"dataset\movies.csv"
ratings_path = r"dataset\ratings.csv"
tags_path = r"dataset\tags.csv"
df_movies, df_ratings, df_tags = load_data(movies_path, ratings_path, tags_path)
all_unique_genres = get_unique_genres(df_movies)



app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/suggest_movies")
def suggest_movies():
    suggest_movies=suggested_random_movies()
    return suggest_movies


@app.post("/recommend_movies_by_content_based")
def recomend_moves(movies_dict:Dict[str, Any] ):
    df_user_profile = create_user_profile(df_movies, movies_dict)
    recomeded_df= recommend_movies_content(df_movies, df_user_profile, all_unique_genres) # Returns DataFrame now
    recomeded = recomeded_df[['title', 'genres', "ratings","movieId"]].to_dict(orient='records') # Convert DataFrame to list of dict
    print(recomeded)
    return recomeded




@app.post("/recommend_movies_by_svd")
def recommend_content_by_svd(user_ratings: Dict[str, Any], n_recommendations: int = 5,user_id:int=1):
    """Recommend movies using SVD based on user ratings given as movie names."""
    
    # Convert movie names to movie IDs
    user_ratings_mapped = {}
    for movie_name, rating in user_ratings.items():
        movie_entry = search_movie_by_title_svd(movie_name, df_movies)
        if movie_entry is not None:
            user_ratings_mapped[movie_entry['movieId']] = rating
    
    if not user_ratings_mapped:
        return {"error": "No valid movies found in the input."}
    
    # Convert the mapped ratings into the expected format
    user_ratings_df = pd.DataFrame(list(user_ratings_mapped.items()), columns=['movieId', 'rating'])
    user_ratings_df['movieId'] = user_ratings_df['movieId'].astype(int)
    
    user_profile = {
        'userid': user_id,  # Dummy user ID (can be adjusted as needed)
        'rated_movies': user_ratings_df
    }
    predicted_ratings, user_movie_ratings, movie_ids, user_means=load_or_train_svd(df_ratings)
    # predicted_ratings, user_movie_ratings, movie_ids, user_means = train_svd(df_ratings)
    
    # Get the top N recommended movies
    recommended_movies = recommend_movies_svd(user_profile, predicted_ratings, user_movie_ratings, movie_ids, df_movies, n_recommendations)
    print(recommended_movies)
    return recommended_movies.to_dict(orient='records')



@app.post("/recommend_movies_by_hybrid")
def recommend_movies_by_hybrid(movies_dict: Dict[str, Any], n_recommendations: int = 5,user_id:int=1):
    """API endpoint to recommend movies using hybrid recommendation."""
    
    
    # Create user profile based on input movie titles and ratings
    df_user_profile = create_user_profile(df_movies, movies_dict)
    
    # Generate content-based recommendations
    content_recs = recommend_movies_content(df_movies, df_user_profile, all_unique_genres)
    


    # Generate SVD-based recommendations
    user_ratings_mapped = {}
    for movie_name, rating in movies_dict.items():
        movie_entry = search_movie_by_title_svd(movie_name, df_movies)
        if movie_entry is not None:
            user_ratings_mapped[movie_entry['movieId']] = rating
    
    if not user_ratings_mapped:
        return {"error": "No valid movies found in the input."}
    
    # Convert the mapped ratings into the expected format
    user_ratings_df = pd.DataFrame(list(user_ratings_mapped.items()), columns=['movieId', 'rating'])
    user_ratings_df['movieId'] = user_ratings_df['movieId'].astype(int)
    
    user_profile = {
        'userid': user_id,  # Dummy user ID (can be adjusted as needed)
        'rated_movies': user_ratings_df
    }
    
    # Train the SVD model
    predicted_ratings, user_movie_ratings, movie_ids, user_means=load_or_train_svd(df_ratings)
    
    # Get the top N recommended movies
    svd_recs = recommend_movies_svd(user_profile, predicted_ratings, user_movie_ratings, movie_ids, df_movies, n_recommendations)
# Combine both recommendations (hybrid approach)
    combined_recs = pd.concat([content_recs, svd_recs]).drop_duplicates(subset=["movieId"])
    hybrid_recs = combined_recs.sample(n=n_recommendations, random_state=42)  # Random sampling to avoid bias
    hybrid_recs = hybrid_recs.replace([float('inf'), float('-inf')], float('nan'))  # Replace inf with NaN
    hybrid_recs = hybrid_recs.fillna(0)  # Replace NaN values with 0 or any other placeholder value you prefer
    print(hybrid_recs)
    # Now, return the clean DataFrame as a dictionary
    return hybrid_recs[['title', 'genres', 'ratings', 'movieId']].to_dict(orient='records')


@app.post("/evaluate_recommendations")
async def evaluate_recommendations_api(user_ratings: Dict[str, int], model_type: str):
    """API endpoint to evaluate recommendation model performance."""
    try:
        evaluation_metrics = evaluate_model_performance(user_ratings, model_type)
        return {"evaluation_metrics": evaluation_metrics, "model_type": model_type}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")


@app.post("/collect_feedback")
async def collect_feedback_api(feedback_data: List[Dict[str, Any]]):
    """API endpoint to collect user feedback and adjust the model."""
    if not feedback_data:
        raise HTTPException(status_code=400, detail="No feedback data provided.")
    # In real app, persist feedback data (e.g., database). For now, just printing
    print("Feedback received:", feedback_data)
    return {"message": "Feedback collected successfully"}
