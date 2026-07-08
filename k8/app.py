from flask import Flask, render_template, request, jsonify
import os
import gdown
import pandas as pd
import numpy as np
import logging
from pymongo import MongoClient
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Global variables for storing data and state
movies = None
combined_features = None
ratings_data = None  # New global to store ratings from data ingestion
user_profile = []
disliked_movies = []

# -----------------------------
# Data Ingestion and Preprocessing
# -----------------------------

class DataIngestion:
    def __init__(self):
        self.MOVIES_FILE_ID = os.getenv("MOVIES_FILE_ID")
        self.RATINGS_FILE_ID = os.getenv("RATINGS_FILE_ID")
        self.TAGS_FILE_ID = os.getenv("TAGS_FILE_ID")
        self.LINKS_FILE_ID = os.getenv("LINKS_FILE_ID")

        if not all([self.MOVIES_FILE_ID, self.RATINGS_FILE_ID, self.TAGS_FILE_ID, self.LINKS_FILE_ID]):
            raise EnvironmentError("One or more dataset file IDs are missing in environment variables.")

        self.DATASET_DIR = "./dataset"
        os.makedirs(self.DATASET_DIR, exist_ok=True)
        logging.basicConfig(level=logging.DEBUG)

    def download_file_from_gdrive(self, file_id, dest_path):
        """Download a file from Google Drive using its file ID."""
        if not file_id:
            raise ValueError(f"Invalid file ID: {file_id}")

        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        logging.debug(f"Downloading {dest_path} from {url}")
        gdown.download(url, dest_path, quiet=False)

    def download_data(self):
        try:
            movies_path = os.path.join(self.DATASET_DIR, "movies.csv")
            ratings_path = os.path.join(self.DATASET_DIR, "ratings.csv")
            tags_path = os.path.join(self.DATASET_DIR, "tags.csv")
            links_path = os.path.join(self.DATASET_DIR, "links.csv")

            # Download datasets if they do not exist
            if not os.path.exists(movies_path):
                self.download_file_from_gdrive(self.MOVIES_FILE_ID, movies_path)

            if not os.path.exists(ratings_path):
                self.download_file_from_gdrive(self.RATINGS_FILE_ID, ratings_path)

            if not os.path.exists(tags_path):
                self.download_file_from_gdrive(self.TAGS_FILE_ID, tags_path)

            if not os.path.exists(links_path):
                self.download_file_from_gdrive(self.LINKS_FILE_ID, links_path)

            logging.debug("Data successfully downloaded.")

            # Load the datasets
            movies = pd.read_csv(movies_path)
            ratings = pd.read_csv(ratings_path)
            tags = pd.read_csv(tags_path)

            return movies, ratings, tags
        except Exception as e:
            logging.error(f"Error downloading or reading data: {e}")
            raise


class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def clean_data(self):
        movies, ratings, tags = self.data
        tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
        movies = pd.merge(movies, tags_grouped, on='movieId', how='left')
        movies['tag'] = movies['tag'].fillna('')
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
        movies['genres'] = movies['genres'].str.split('|')

        genre_dummies = movies['genres'].str.join('|').str.get_dummies()
        tag_dummies = movies['tag'].str.get_dummies(sep=' ')
        combined_features = pd.concat([genre_dummies, tag_dummies], axis=1)

        return movies, combined_features

class MongoDBHandler:
    def __init__(self, uri, db_name):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        print(f"this is mongo {uri}")

    def insert_data(self, collection_name, data):
        collection = self.db[collection_name]

        # Skip insertion if collection already has data
        if collection.estimated_document_count() > 0:
            logging.info(f"Collection '{collection_name}' already exists in the database. Skipping insertion.")
            return False

        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")
        collection.insert_many(data)
        logging.info(f"Data successfully inserted into collection '{collection_name}'.")
        return True

# -----------------------------
# Recommendation Algorithms
# -----------------------------

class SVDAlgo:
    """
    Recommendation using Singular Value Decomposition.
    This method builds a user-item matrix from ratings_data, appends a new user row with high ratings for liked movies,
    performs SVD, and then predicts ratings for unseen movies.
    """
    def __init__(self, ratings, k=20):
        self.ratings = ratings
        self.k = k

    def recommend_movies(self, movies, user_likes, disliked_movies=[], num_recommendations=5):
        try:
            # Create pivot table (users x movies)
            R = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
            # Append new user row with a high rating (e.g., 5) for liked movies
            new_user = pd.Series(0, index=R.columns)
            for movie in user_likes:
                if movie in new_user.index:
                    new_user[movie] = 5
            # Using pd.concat instead of deprecated append
            R = pd.concat([R, new_user.to_frame().T], ignore_index=True)
            new_user_index = R.index[-1]
            R_matrix = R.to_numpy()

            # Determine number of latent factors (k)
            k = min(self.k, min(R_matrix.shape) - 1)
            U, sigma, Vt = svds(R_matrix, k=k)
            sigma = np.diag(sigma)
            all_user_pred = np.dot(np.dot(U, sigma), Vt)
            preds = all_user_pred[new_user_index, :]
            preds_series = pd.Series(preds, index=R.columns)

            # Exclude already liked or disliked movies
            preds_series = preds_series.drop(labels=[m for m in user_likes if m in preds_series.index], errors='ignore')
            preds_series = preds_series.drop(labels=[m for m in disliked_movies if m in preds_series.index], errors='ignore')
            top_movie_ids = preds_series.sort_values(ascending=False).head(num_recommendations).index
            scores = preds_series.loc[top_movie_ids]
            recommendations = movies[movies['movieId'].isin(top_movie_ids)].copy()
            recommendations['score'] = recommendations['movieId'].map(scores.to_dict())
            recommendations = recommendations.sort_values(by='score', ascending=False)
            return recommendations
        except Exception as e:
            logging.error("SVDAlgo error: " + str(e))
            return pd.DataFrame()

class CollaborativeAlgo:
    """
    Item-based Collaborative Filtering using cosine similarity.
    Computes similarity between movies based on user ratings and aggregates scores from liked movies.
    """
    def __init__(self, ratings):
        self.ratings = ratings
        self.R = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.similarity_matrix = cosine_similarity(self.R.T)
        self.movie_ids = self.R.columns.tolist()

    def recommend_movies(self, movies, user_likes, disliked_movies=[], num_recommendations=5):
        try:
            scores = {movie: 0 for movie in self.movie_ids}
            count = {movie: 0 for movie in self.movie_ids}
            for movie in user_likes:
                if movie in self.movie_ids:
                    idx = self.movie_ids.index(movie)
                    sim_scores = self.similarity_matrix[idx]
                    for j, other_movie in enumerate(self.movie_ids):
                        if other_movie in user_likes or other_movie in disliked_movies:
                            continue
                        scores[other_movie] += sim_scores[j]
                        count[other_movie] += 1
            avg_scores = {movie: (scores[movie] / count[movie] if count[movie] > 0 else 0) for movie in scores}
            score_series = pd.Series(avg_scores)
            top_movie_ids = score_series.sort_values(ascending=False).head(num_recommendations).index
            top_scores = score_series.loc[top_movie_ids]
            recommendations = movies[movies['movieId'].isin(top_movie_ids)].copy()
            recommendations['score'] = recommendations['movieId'].map(top_scores.to_dict())
            recommendations = recommendations.sort_values(by='score', ascending=False)
            return recommendations
        except Exception as e:
            logging.error("CollaborativeAlgo error: " + str(e))
            return pd.DataFrame()

class HybridAlgo:
    """
    Combines SVD and Collaborative Filtering by averaging normalized scores from both methods.
    """
    def __init__(self, ratings, k=20):
        self.svd_algo = SVDAlgo(ratings, k)
        self.collab_algo = CollaborativeAlgo(ratings)

    def recommend_movies(self, movies, user_likes, disliked_movies=[], num_recommendations=5):
        try:
            # Get a larger set of recommendations from both methods
            svd_rec = self.svd_algo.recommend_movies(movies, user_likes, disliked_movies, num_recommendations=20)
            collab_rec = self.collab_algo.recommend_movies(movies, user_likes, disliked_movies, num_recommendations=20)
            
            if not svd_rec.empty:
                svd_min, svd_max = svd_rec['score'].min(), svd_rec['score'].max()
                svd_rec['norm_score'] = (svd_rec['score'] - svd_min) / (svd_max - svd_min + 1e-9)
            else:
                svd_rec['norm_score'] = 0
            if not collab_rec.empty:
                collab_min, collab_max = collab_rec['score'].min(), collab_rec['score'].max()
                collab_rec['norm_score'] = (collab_rec['score'] - collab_min) / (collab_max - collab_min + 1e-9)
            else:
                collab_rec['norm_score'] = 0

            merged = pd.merge(svd_rec[['movieId', 'norm_score']], 
                              collab_rec[['movieId', 'norm_score']], 
                              on='movieId', how='outer', suffixes=('_svd', '_collab'))
            merged.fillna(0, inplace=True)
            merged['hybrid_score'] = (merged['norm_score_svd'] + merged['norm_score_collab']) / 2.0
            top_movies = merged.sort_values(by='hybrid_score', ascending=False).head(num_recommendations)['movieId']
            recommendations = movies[movies['movieId'].isin(top_movies)].copy()
            recommendations['score'] = recommendations['movieId'].map(merged.set_index('movieId')['hybrid_score'].to_dict())
            recommendations = recommendations.sort_values(by='score', ascending=False)
            return recommendations
        except Exception as e:
            logging.error("HybridAlgo error: " + str(e))
            return pd.DataFrame()

# -----------------------------
# Flask Endpoints
# -----------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/load_data', methods=['POST'])
def load_data():
    global movies, combined_features, ratings_data
    try:
        data_ingestion = DataIngestion()
        movies, ratings, tags = data_ingestion.download_data()
        ratings_data = ratings  # Save ratings data for recommendation algorithms
        preprocessor = DataPreprocessor((movies, ratings, tags))
        movies, combined_features = preprocessor.clean_data()

        # MongoDB Integration
        MONGO_URI = os.getenv("MONGO_URI")
        mongo_handler = MongoDBHandler(uri=MONGO_URI,db_name="MovieRecommendationDB")
        mongo_movies_inserted = mongo_handler.insert_data("movies", movies)
        mongo_ratings_inserted = mongo_handler.insert_data("ratings", ratings)
        mongo_tags_inserted = mongo_handler.insert_data("tags", tags)

        message = "Data loaded successfully!"
        if not mongo_movies_inserted or not mongo_ratings_inserted or not mongo_tags_inserted:
            message += " Some collections were already present in MongoDB and were not overwritten."

        return jsonify({
            'message': message,
            'mongo_status': {
                'movies': mongo_movies_inserted,
                'ratings': mongo_ratings_inserted,
                'tags': mongo_tags_inserted
            },
            'movies': movies[['movieId', 'title']].to_dict(orient='records')
        }), 200
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return jsonify({'message': f"Error loading data: {str(e)}"}), 500

@app.route('/show_movies', methods=['GET'])
def show_movies():
    global movies
    if movies is None:
        return jsonify({'message': 'No movies loaded yet. Please load the data first.'}), 400

    page = int(request.args.get('page', 0))
    start = page * 10
    end = start + 10
    paginated_movies = movies.iloc[start:end]

    return jsonify({'movies': paginated_movies[['movieId', 'title', 'genres']].to_dict(orient='records')}), 200

@app.route('/select_movies', methods=['POST'])
def select_movies():
    global user_profile
    user_profile = request.json.get('user_likes', [])
    return jsonify({'message': 'Movies selected successfully!'}), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    global movies, user_profile, disliked_movies, ratings_data

    if ratings_data is None:
        return jsonify({'message': 'Ratings data not loaded yet.'}), 400

    # Choose algorithm based on request (default to 'hybrid')
    algorithm = request.json.get('algorithm', 'hybrid').lower()
    if algorithm not in ['svd', 'collaborative', 'hybrid']:
        return jsonify({'message': 'Invalid algorithm selected. Choose from svd, collaborative, hybrid.'}), 400

    if algorithm == 'svd':
        recommender = SVDAlgo(ratings_data)
    elif algorithm == 'collaborative':
        recommender = CollaborativeAlgo(ratings_data)
    elif algorithm == 'hybrid':
        recommender = HybridAlgo(ratings_data)

    recommendations = recommender.recommend_movies(movies, user_profile, disliked_movies)
    return jsonify({'recommendations': recommendations[['movieId', 'title', 'year']].to_dict(orient='records')}), 200

@app.route('/feedback', methods=['POST'])
def feedback():
    global user_profile, disliked_movies
    feedback_data = request.json.get('feedback', [])
    for item in feedback_data:
        if item['liked']:
            user_profile.append(item['movieId'])
        else:
            disliked_movies.append(item['movieId'])
    return jsonify({'message': 'Feedback recorded successfully!'}), 200

@app.route('/refresh_recommendations', methods=['POST'])
def refresh_recommendations():
    global movies, user_profile, disliked_movies, ratings_data

    if ratings_data is None:
        return jsonify({'message': 'Ratings data not loaded yet.'}), 400

    # Use algorithm parameter if provided; default to hybrid
    algorithm = (request.json.get('algorithm', 'hybrid').lower() if request.json else 'hybrid')
    if algorithm not in ['svd', 'collaborative', 'hybrid']:
        return jsonify({'message': 'Invalid algorithm selected. Choose from svd, collaborative, hybrid.'}), 400

    if algorithm == 'svd':
        recommender = SVDAlgo(ratings_data)
    elif algorithm == 'collaborative':
        recommender = CollaborativeAlgo(ratings_data)
    elif algorithm == 'hybrid':
        recommender = HybridAlgo(ratings_data)

    updated_recommendations = recommender.recommend_movies(movies, user_profile, disliked_movies)
    return jsonify({'recommendations': updated_recommendations[['movieId', 'title', 'year']].to_dict(orient='records')}), 200

if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0", port=5000)
