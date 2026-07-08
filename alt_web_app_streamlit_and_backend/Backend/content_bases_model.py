import pandas as pd
import numpy as np
import regex as re
import streamlit as st

def load_data(movies_path, ratings_path, tags_path):
    """Loads movie, ratings, and tags data from CSV files."""
    df_movies = pd.read_csv(movies_path)
    df_ratings = pd.read_csv(ratings_path)
    df_tags = pd.read_csv(tags_path)
    st.write(df_movies,df_ratings,df_tags)
    return df_movies, df_ratings, df_tags

def search_movie_by_id(movie_id, df_movies):
    """Searches for a movie by its ID."""
    movie_entry = df_movies[df_movies["movieId"] == movie_id]
    if not movie_entry.empty:
        return movie_entry
    else:
        print(f"Movie with ID {movie_id} not found.")
        return None
    
def search_movie_by_title(movie_title, df_movies):
    """Searches for a movie by its title (case-insensitive)."""
    movie_title_escaped = re.escape(movie_title)
    movie_entry = df_movies[df_movies["title"].str.contains(movie_title_escaped, case=False, na=False, regex=True)]
    if not movie_entry.empty:
        return movie_entry
    else:
        print(f"Movie '{movie_title}' not found.")
        return None

def create_user_profile(df_movies, movies):
    """Creates a user profile by prompting for movie titles and ratings."""
    user_movies = []
    user_movie_ids = []
    user_ratings = []
    user_movie_genres = []

    for movie, rating in movies.items():
        movie_search_result = search_movie_by_title(movie, df_movies)

        if movie_search_result is not None:  # Check if the result is not None
            if not movie_search_result.empty:
                movie_id = movie_search_result.iloc[0]["movieId"]
                movie_title = movie_search_result.iloc[0]["title"]
                genre_title = movie_search_result.iloc[0]["genres"]

                user_movies.append(movie_title)
                user_movie_ids.append(movie_id)
                user_ratings.append(rating)
                user_movie_genres.append(genre_title)
        else:
            print(f"Movie '{movie}' Not found, skipping.")

    user_profile = {
        "movieId": user_movie_ids,
        "title": user_movies,
        "ratings": user_ratings,
        "genres": user_movie_genres,
    }

    return pd.DataFrame(user_profile)
    
def get_unique_genres(df_movies):
    """Extracts all unique genres from the movies DataFrame."""
    all_unique_genres = set()
    for genres in df_movies['genres']:
        genre_list = genres.split('|')
        all_unique_genres.update(genre_list)
    return all_unique_genres

def one_hot_encode_movies(df, unique_genres):
    """One-hot encodes the genres in a DataFrame."""
    unique_genres_list = list(unique_genres)
    one_hot_df = pd.DataFrame(0, index=df.index, columns=unique_genres_list)
    for index, row in df.iterrows():
        genres = row['genres'].split('|')
        for genre in genres:
            if genre in unique_genres_list:
                one_hot_df.at[index, genre] = 1
    df = pd.concat([df, one_hot_df], axis=1)
    return df

def recommend_movies_content(df_movies, df_user_profile, all_unique_genres):
    """Recommends movies based on user profile."""
    unrated_movies_df = df_movies[~df_movies["movieId"].isin(df_user_profile["movieId"])]
    unrated_movies_encoded = one_hot_encode_movies(unrated_movies_df, all_unique_genres)
    rated_movies_encoded = one_hot_encode_movies(df_user_profile, all_unique_genres)

    unique_genres = set()
    for genres in df_user_profile['genres']:
        genre_list = genres.split('|')
        unique_genres.update(genre_list)

    columns_of_interest = [genre for genre in unique_genres if genre in unrated_movies_encoded.columns]
    filtered_movies = unrated_movies_encoded[unrated_movies_encoded[columns_of_interest].any(axis=1)]

    df_movie_tags = rated_movies_encoded.iloc[:, 4:]
    unrated_movies_tags = filtered_movies.iloc[:, 3:]
    matrix_user_profile = df_user_profile["ratings"].to_numpy().reshape(-1, 1)
    matrix_resultant = matrix_user_profile.T @ df_movie_tags
    sum_matrix = np.sum(matrix_resultant, axis=1)
    matrix_resultant /= sum_matrix[0]
    recommendation_profile = matrix_resultant @ unrated_movies_tags.T

    df_recommendation_profile = pd.DataFrame(recommendation_profile.T)
    df_recommendation_profile["movieId"] = filtered_movies["movieId"].values
    df_recommendation_profile.columns = ["ratings", "movieId"]
    top_5_recommendations = df_recommendation_profile.sort_values(by='ratings', ascending=False).head(5)
    top_5_movies = pd.merge(top_5_recommendations, df_movies, left_on='movieId', right_on='movieId', how='inner')
    print(top_5_movies)
    return top_5_movies[['title', 'genres',"ratings","movieId"]]
