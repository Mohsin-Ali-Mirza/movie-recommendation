import pandas as pd


def load_movies_data(path):
    return pd.read_csv(path)
# Load data



def suggested_random_movies():
    movies_file = r"dataset\movies.csv"
    ratings_file = r"dataset\ratings.csv"
    movies_df = pd.read_csv(movies_file)
    ratings_df = pd.read_csv(ratings_file)
    # Calculate average ratings per movie
    average_ratings = ratings_df.groupby("movieId")["rating"].mean()

    # Filter movies with rating >= 3
    filtered_movies = movies_df[movies_df["movieId"].isin(average_ratings[average_ratings >= 3].index)]

    # Expand genres into separate rows
    filtered_movies["genres"] = filtered_movies["genres"].str.split("|")
    movies_exploded = filtered_movies.explode("genres")

    # Remove movies with '(no genres listed)'
    movies_exploded = movies_exploded[movies_exploded["genres"] != "(no genres listed)"]

    # Select 5 random movies from different genres
    random_movies = movies_exploded.groupby("genres").sample(n=1).sample(n=5, random_state=42)

    # Display the selected movies
    return random_movies["title"].tolist()