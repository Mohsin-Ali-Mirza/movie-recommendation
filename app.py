from flask import Flask, render_template, request, jsonify, session
from flask import make_response
import os
from datetime import timedelta
import pandas as pd
import numpy as np
import logging
import threading
from kaggle.api.kaggle_api_extended import KaggleApi
from pymongo import MongoClient
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from argon2 import PasswordHasher
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.secret_key = "your_secret_key"  # Secure in production
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # <-- Use 'Lax' for local dev
app.config['SESSION_COOKIE_SECURE'] = False    # <-- False for local dev (HTTP)


app.secret_key = "your_secret_key"  # Replace with a secure key in production

logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)

movies = None
combined_features = None
ratings_data = None
directors_data = None
user_profile = []
disliked_movies = []
director_progress = ""

# -----------------------------
# Data Ingestion and Preprocessing
# -----------------------------

class DataIngestion:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.api = KaggleApi()
        self.api.authenticate()

    def download_data(self):
        try:
            download_path = './dataset'
            logging.info(f"Downloading dataset from {self.dataset_path} to {download_path}")
            self.api.dataset_download_files(self.dataset_path, path=download_path, unzip=True)
            
            movies_path = os.path.join(download_path, 'movies.csv')
            ratings_path = os.path.join(download_path, 'ratings.csv')
            tags_path = os.path.join(download_path, 'tags.csv')
            links_path = os.path.join(download_path, 'links.csv')

            if not os.path.exists(movies_path):
                raise FileNotFoundError("Movies file not found.")
            if not os.path.exists(ratings_path):
                raise FileNotFoundError("Ratings file not found.")
            if not os.path.exists(tags_path):
                raise FileNotFoundError("Tags file not found.")
            if not os.path.exists(links_path):
                raise FileNotFoundError("Links file not found.")

            logging.info("Reading movies, ratings, tags, and links data...")
            movies = pd.read_csv(movies_path)
            ratings = pd.read_csv(ratings_path)
            tags = pd.read_csv(tags_path)
            links = pd.read_csv(links_path)

            logging.info("Data successfully loaded")
            return movies, ratings, tags, links
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
    def __init__(self, uri="mongodb://localhost:27017/", db_name="MovieRecommendationDB"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def insert_data(self, collection_name, data):
        collection = self.db[collection_name]
        if collection.estimated_document_count() > 0:
            logging.info(f"Collection '{collection_name}' already exists. Skipping insertion.")
            return False
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")
        collection.insert_many(data)
        logging.info(f"Data successfully inserted into collection '{collection_name}'.")
        return True

# -----------------------------
# Background Director Data Loading with Caching in MongoDB
# -----------------------------

def load_directors_data(movies_df):
    global directors_data, director_progress
    try:
        mongo_handler = MongoDBHandler()
        db = mongo_handler.db
        
        crew_collection = db["imdb_crew"]
        names_collection = db["imdb_names"]
        
        if crew_collection.estimated_document_count() > 0 and names_collection.estimated_document_count() > 0:
            director_progress = "IMDb datasets found in MongoDB. Loading from database..."
            logging.info(director_progress)
            crew_df = pd.DataFrame(list(crew_collection.find()))
            names_df = pd.DataFrame(list(names_collection.find()))
        else:
            director_progress = "Downloading IMDb crew dataset..."
            logging.info(director_progress)
            imdb_crew_url = "https://datasets.imdbws.com/title.crew.tsv.gz"
            crew_df = pd.read_csv(imdb_crew_url, compression='gzip', sep='\t', low_memory=False)
            crew_collection.insert_many(crew_df.to_dict("records"))
            director_progress = "IMDb crew dataset downloaded and stored."
            logging.info(director_progress)
            
            director_progress = "Downloading IMDb names dataset..."
            logging.info(director_progress)
            imdb_names_url = "https://datasets.imdbws.com/name.basics.tsv.gz"
            names_df = pd.read_csv(imdb_names_url, compression='gzip', sep='\t', low_memory=False)
            names_collection.insert_many(names_df.to_dict("records"))
            director_progress = "IMDb names dataset downloaded and stored."
            logging.info(director_progress)
        
        director_progress = "Processing crew dataset..."
        logging.info(director_progress)
        crew_df['directorId'] = crew_df['directors'].apply(lambda x: x.split(',')[0] if pd.notnull(x) and x != '\\N' else None)
        
        director_progress = "Processing names dataset..."
        logging.info(director_progress)
        names_df = names_df[['nconst', 'primaryName']]
        names_df.rename(columns={'nconst': 'directorId', 'primaryName': 'director'}, inplace=True)
        
        director_progress = "Merging crew with names..."
        logging.info(director_progress)
        crew_with_names = pd.merge(crew_df[['tconst', 'directorId']], names_df, on='directorId', how='left')
        
        director_progress = "Merging movies with director data..."
        logging.info(director_progress)
        merged = pd.merge(movies_df, crew_with_names[['tconst', 'director']], left_on='imdbId', right_on='tconst', how='left')
        merged.drop(columns=['tconst'], inplace=True)
        
        director_progress = "Extracting director data..."
        logging.info(director_progress)
        directors_data = merged[['movieId', 'director']]
        
        directors_collection = db["imdb_directors"]
        if directors_collection.estimated_document_count() == 0:
            directors_collection.insert_many(directors_data.to_dict("records"))
            director_progress = "Director data loading complete and stored in MongoDB."
        else:
            director_progress = "Director data loading complete (loaded from previous cache)."
        logging.info(director_progress)
    except Exception as e:
        director_progress = "Error loading director data: " + str(e)
        logging.error("Error in load_directors_data: " + str(e))

@app.route('/director_progress', methods=['GET'])
def get_director_progress():
    return jsonify({'progress': {'text': director_progress, 'progress': 0}}), 200

# -----------------------------
# Recommendation Algorithms
# -----------------------------

class SVDAlgo:
    def __init__(self, ratings, k=20):
        self.ratings = ratings
        self.k = k

    def recommend_movies(self, movies, user_likes, disliked_movies=[], num_recommendations=5):
        try:
            R = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
            new_user = pd.Series(0, index=R.columns)
            for movie in user_likes:
                if movie in new_user.index:
                    new_user[movie] = 5
            R = pd.concat([R, new_user.to_frame().T], ignore_index=True)
            new_user_index = R.index[-1]
            R_matrix = R.to_numpy()
            k = min(self.k, min(R_matrix.shape) - 1)
            U, sigma, Vt = svds(R_matrix, k=k)
            sigma = np.diag(sigma)
            all_user_pred = np.dot(np.dot(U, sigma), Vt)
            preds = all_user_pred[new_user_index, :]
            preds_series = pd.Series(preds, index=R.columns)
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

class ContentAlgo:
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
            logging.error("ContentAlgo error: " + str(e))
            return pd.DataFrame()

class HybridAlgo:
    def __init__(self, ratings, k=20):
        self.svd_algo = SVDAlgo(ratings, k)
        self.content_algo = ContentAlgo(ratings)

    def recommend_movies(self, movies, user_likes, disliked_movies=[], num_recommendations=5):
        try:
            svd_rec = self.svd_algo.recommend_movies(movies, user_likes, disliked_movies, num_recommendations=20)
            content_rec = self.content_algo.recommend_movies(movies, user_likes, disliked_movies, num_recommendations=20)
            
            if not svd_rec.empty:
                svd_min, svd_max = svd_rec['score'].min(), svd_rec['score'].max()
                svd_rec['norm_score'] = (svd_rec['score'] - svd_min) / (svd_max - svd_min + 1e-9)
            else:
                svd_rec['norm_score'] = 0
            if not content_rec.empty:
                content_min, content_max = content_rec['score'].min(), content_rec['score'].max()
                content_rec['norm_score'] = (content_rec['score'] - content_min) / (content_max - content_min + 1e-9)
            else:
                content_rec['norm_score'] = 0

            merged = pd.merge(svd_rec[['movieId', 'norm_score']], 
                              content_rec[['movieId', 'norm_score']], 
                              on='movieId', how='outer', suffixes=('_svd', '_content'))
            merged.fillna(0, inplace=True)
            merged['hybrid_score'] = (merged['norm_score_svd'] + merged['norm_score_content']) / 2.0
            top_movies = merged.sort_values(by='hybrid_score', ascending=False).head(num_recommendations)['movieId']
            recommendations = movies[movies['movieId'].isin(top_movies)].copy()
            recommendations['score'] = recommendations['movieId'].map(merged.set_index('movieId')['hybrid_score'].to_dict())
            recommendations = recommendations.sort_values(by='score', ascending=False)
            return recommendations
        except Exception as e:
            logging.error("HybridAlgo error: " + str(e))
            return pd.DataFrame()

class DirectorsAlgo:
    def __init__(self, directors_df):
        self.directors_df = directors_df.copy()

    def recommend_movies(self, movies, director_input, num_recommendations=5):
        try:
            matched = self.directors_df[self.directors_df['director'].str.contains(director_input, case=False, na=False)]
            if matched.empty:
                return pd.DataFrame(columns=['movieId', 'title', 'year'])
            recommendations = pd.merge(
                matched,
                movies[['movieId', 'title', 'year']],
                on='movieId',
                how='left',
                suffixes=('', '_movie')
            )
            if 'title' not in recommendations.columns and 'title_movie' in recommendations.columns:
                recommendations.rename(columns={'title_movie': 'title'}, inplace=True)
            for col in ['movieId', 'title', 'year']:
                if col not in recommendations.columns:
                    recommendations[col] = None
            return recommendations.head(num_recommendations)
        except Exception as e:
            logging.error("DirectorsAlgo error: " + str(e))
            return pd.DataFrame(columns=['movieId', 'title', 'year'])

# -----------------------------
# Flask Endpoints
# -----------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/load_data', methods=['POST'])
def load_data():
    global movies, combined_features, ratings_data, directors_data, director_progress
    try:
        data_ingestion = DataIngestion(dataset_path='shubhammehta21/movie-lens-small-latest-dataset')
        ml_movies, ratings, tags, links = data_ingestion.download_data()
        ratings_data = ratings
        preprocessor = DataPreprocessor((ml_movies, ratings, tags))
        movies, combined_features = preprocessor.clean_data()

        movies = pd.merge(movies, links[['movieId', 'imdbId']], on='movieId', how='left')
        movies['imdbId'] = movies['imdbId'].apply(lambda x: "tt" + str(x).zfill(7) if pd.notnull(x) else None)

        mongo_handler = MongoDBHandler()
        db = mongo_handler.db
        directors_collection = db["imdb_directors"]
        if directors_collection.estimated_document_count() > 0:
            director_progress = "Directors data already loaded from MongoDB."
            directors_data = pd.DataFrame(list(directors_collection.find()))
        else:
            director_progress = "Initializing director data loading..."
            threading.Thread(target=load_directors_data, args=(movies.copy(),), daemon=True).start()

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
                'tags': mongo_tags_inserted,
                'directors': director_progress
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

# -----------------------------
# New User Authentication Endpoints
# -----------------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    mongo_handler = MongoDBHandler()
    db = mongo_handler.db
    users_collection = db["users"]
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        name = data.get('name')
        age = data.get('age')
        country = data.get('country')
        password = data.get('password')
        
        if users_collection.find_one({"username": username}):
            return jsonify({"message": "Username already exists."}), 400
        
        ph = PasswordHasher()
        password_hash = ph.hash(password)
        user_doc = {
            "username": username,
            "name": name,
            "age": age,
            "country": country,
            "password_hash": password_hash,
            "liked_movies": [],
            "disliked_movies": []
        }
        users_collection.insert_one(user_doc)
        return jsonify({"message": "User registered successfully."}), 200
    else:
        return render_template('register.html')

@app.route('/login', methods=['POST'])
def login():
    mongo_handler = MongoDBHandler()
    users_collection = mongo_handler.db["users"]
    data = request.json
    username = data.get('username')
    password = data.get('password')
    user = users_collection.find_one({"username": username})

    if not user:
        return jsonify({"message": "Invalid username or password."}), 400

    ph = PasswordHasher()
    try:
        ph.verify(user['password_hash'], password)
    except Exception:
        return jsonify({"message": "Invalid username or password."}), 400

    session['username'] = username
    session.permanent = True  # recommended to maintain session clearly
    app.permanent_session_lifetime = timedelta(hours=5)  # example duration

    return jsonify({"message": "Logged in successfully."}), 200

@app.route('/logout', methods=['GET'])
def logout():
    session.clear()
    response = make_response(jsonify({"message": "Logged out successfully."}), 200)
    response.set_cookie('session', '', expires=0, samesite='None', secure=False)  # Adjust secure based on HTTPS
    return response



@app.route('/profile', methods=['GET'])
def profile():
    if 'username' not in session:
        return jsonify({"message": "User not logged in."}), 401
    mongo_handler = MongoDBHandler()
    db = mongo_handler.db
    user = db["users"].find_one({"username": session['username']}, {"password_hash": 0})
    if not user:
        return jsonify({"message": "User not found."}), 404
    user["_id"] = str(user["_id"])  # Convert ObjectId to string
    return jsonify({"profile": user}), 200

@app.route('/delete_profile', methods=['POST'])
def delete_profile():
    if 'username' not in session:
        return jsonify({"message": "User not logged in."}), 401
    mongo_handler = MongoDBHandler()
    db = mongo_handler.db
    result = db["users"].delete_one({"username": session['username']})
    session.pop('username', None)
    if result.deleted_count:
        return jsonify({"message": "Profile deleted successfully."}), 200
    else:
        return jsonify({"message": "Profile deletion failed."}), 400

@app.route('/manage_session', methods=['POST'])
def manage_session():
    if 'username' not in session:
        return jsonify({"message": "User not logged in."}), 401

    action = request.json.get('action')
    mongo_handler = MongoDBHandler()
    db = mongo_handler.db
    users_collection = db["users"]

    user = users_collection.find_one({"username": session['username']})
    if not user:
        return jsonify({"message": "User not found."}), 404

    global user_profile, disliked_movies

    if action == 'save':
        liked = list(set(user.get("liked_movies", []) + user_profile))
        disliked = list(set(user.get("disliked_movies", []) + disliked_movies))
        users_collection.update_one(
            {"username": session['username']},
            {"$set": {"liked_movies": liked, "disliked_movies": disliked}}
        )
        # Clear session after saving
        user_profile = []
        disliked_movies = []
        return jsonify({"message": "Session data will be saved to your profile successfully!"}), 200

    elif action == 'clear':
        user_profile = []
        disliked_movies = []
        return jsonify({"message": "Session history will be cleared at the end!"}), 200

    else:
        return jsonify({"message": "Invalid action."}), 400

# -----------------------------
# Feedback and Recommendation Endpoints
# -----------------------------

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_data = request.json.get('feedback', [])
    if 'username' in session:
        mongo_handler = MongoDBHandler()
        db = mongo_handler.db
        users_collection = db["users"]
        user = users_collection.find_one({"username": session['username']})
        if not user:
            return jsonify({"message": "User not found."}), 404
        liked = user.get("liked_movies", [])
        disliked = user.get("disliked_movies", [])
        for item in feedback_data:
            if item['liked']:
                if item['movieId'] not in liked:
                    liked.append(item['movieId'])
            else:
                if item['movieId'] not in disliked:
                    disliked.append(item['movieId'])
        users_collection.update_one(
            {"username": session['username']},
            {"$set": {"liked_movies": liked, "disliked_movies": disliked}}
        )
        return jsonify({"message": "Feedback recorded successfully!"}), 200
    else:
        for item in feedback_data:
            if item['liked']:
                user_profile.append(item['movieId'])
            else:
                disliked_movies.append(item['movieId'])
        return jsonify({"message": "Feedback recorded successfully!"}), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    global movies, ratings_data, directors_data
    if movies is None:
        return jsonify({'message': 'Movies data not loaded yet.'}), 400

    algorithm = request.json.get('algorithm', 'hybrid').lower()
    valid_algorithms = ['svd', 'content', 'hybrid', 'directors']
    if algorithm not in valid_algorithms:
        return jsonify({'message': 'Invalid algorithm selected. Choose from svd, content, hybrid, directors.'}), 400

    if 'username' in session:
        mongo_handler = MongoDBHandler()
        db = mongo_handler.db
        user = db["users"].find_one({"username": session['username']})
        user_likes = user.get("liked_movies", []) if user else []
        user_disliked = user.get("disliked_movies", []) if user else []
    else:
        user_likes = request.json.get('user_likes', [])
        user_disliked = disliked_movies

    if algorithm == 'svd':
        if ratings_data is None:
            return jsonify({'message': 'Ratings data not loaded yet.'}), 400
        recommender = SVDAlgo(ratings_data)
        recommendations = recommender.recommend_movies(movies, user_likes, user_disliked)
    elif algorithm == 'content':
        if ratings_data is None:
            return jsonify({'message': 'Ratings data not loaded yet.'}), 400
        recommender = ContentAlgo(ratings_data)
        recommendations = recommender.recommend_movies(movies, user_likes, user_disliked)
    elif algorithm == 'hybrid':
        if ratings_data is None:
            return jsonify({'message': 'Ratings data not loaded yet.'}), 400
        recommender = HybridAlgo(ratings_data)
        recommendations = recommender.recommend_movies(movies, user_likes, user_disliked)
    elif algorithm == 'directors':
        director_input = request.json.get('director', None)
        if not director_input:
            return jsonify({'message': 'Director name must be provided for the directors algorithm.'}), 400
        if directors_data is None:
            return jsonify({'message': 'Directors data not loaded yet. Check /director_progress for details.'}), 400
        recommender = DirectorsAlgo(directors_data)
        recommendations = recommender.recommend_movies(movies, director_input)

    return jsonify({'recommendations': recommendations[['movieId', 'title', 'year']].to_dict(orient='records')}), 200

@app.route('/refresh_recommendations', methods=['POST'])
def refresh_recommendations():
    global movies, ratings_data, directors_data
    if movies is None:
        return jsonify({'message': 'Movies data not loaded yet.'}), 400

    algorithm = request.json.get('algorithm', 'hybrid').lower() if request.json else 'hybrid'
    valid_algorithms = ['svd', 'content', 'hybrid', 'directors']
    if algorithm not in valid_algorithms:
        return jsonify({'message': 'Invalid algorithm selected. Choose from svd, content, hybrid, directors.'}), 400

    if 'username' in session:
        mongo_handler = MongoDBHandler()
        db = mongo_handler.db
        user = db["users"].find_one({"username": session['username']})
        user_likes = user.get("liked_movies", []) if user else []
        user_disliked = user.get("disliked_movies", []) if user else []
    else:
        user_likes = request.json.get('user_likes', [])
        user_disliked = disliked_movies

    if algorithm == 'svd':
        if ratings_data is None:
            return jsonify({'message': 'Ratings data not loaded yet.'}), 400
        recommender = SVDAlgo(ratings_data)
        updated_recommendations = recommender.recommend_movies(movies, user_likes, user_disliked)
    elif algorithm == 'content':
        if ratings_data is None:
            return jsonify({'message': 'Ratings data not loaded yet.'}), 400
        recommender = ContentAlgo(ratings_data)
        updated_recommendations = recommender.recommend_movies(movies, user_likes, user_disliked)
    elif algorithm == 'hybrid':
        if ratings_data is None:
            return jsonify({'message': 'Ratings data not loaded yet.'}), 400
        recommender = HybridAlgo(ratings_data)
        updated_recommendations = recommender.recommend_movies(movies, user_likes, user_disliked)
    elif algorithm == 'directors':
        director_input = request.json.get('director', None)
        if not director_input:
            return jsonify({'message': 'Director name must be provided for the directors algorithm.'}), 400
        if directors_data is None:
            return jsonify({'message': 'Directors data not loaded yet. Check /director_progress for details.'}), 400
        recommender = DirectorsAlgo(directors_data)
        updated_recommendations = recommender.recommend_movies(movies, director_input)

    return jsonify({'recommendations': updated_recommendations[['movieId', 'title', 'year']].to_dict(orient='records')}), 200

if __name__ == '__main__':
    app.run()
