Below is a generated README file for your movie recommendation system project based on the provided code. This README assumes the project is hosted on a platform like GitHub and provides an overview, setup instructions, usage details, and more.

---

# Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-orange.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A hybrid movie recommendation system built with Python, FastAPI, and Streamlit. This project integrates content-based filtering, SVD (Singular Value Decomposition)-based collaborative filtering, and a hybrid approach to recommend movies based on user preferences. It includes a RESTful API backend and an interactive frontend for users to rate movies and receive recommendations.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a movie recommendation system that leverages multiple techniques:
- *Content-Based Filtering*: Recommends movies based on genre similarity to user-rated movies.
- *SVD-Based Collaborative Filtering*: Uses matrix factorization to predict user preferences based on historical ratings.
- *Hybrid Approach*: Combines content-based and SVD-based methods for improved recommendation accuracy.

The system includes a FastAPI backend for serving recommendation models and a Streamlit frontend for user interaction. Users can rate movies, receive recommendations, and provide feedback to refine suggestions.

---

## Features
- Suggest random popular movies to new users.
- Generate recommendations using content-based, SVD-based, or hybrid models.
- Evaluate model performance with MAP@K (Mean Average Precision at K).
- Collect user feedback to improve recommendations.
- Interactive UI with login (optional) and guest mode.
- MongoDB integration for user authentication and feedback storage.

---

## Technologies
- *Python 3.8+*: Core programming language.
- *FastAPI*: Backend API framework.
- *Streamlit*: Frontend for user interaction.
- *Pandas & NumPy*: Data manipulation and computation.
- *SciPy*: Sparse matrix operations and SVD.
- *MongoDB*: Database for user authentication and feedback.
- *Pickle*: Model persistence.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- MongoDB installed and running locally
- Git (optional, for cloning the repository)

### Steps
1. *Clone the Repository*
   bash
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   

2. *Create a Virtual Environment*
   bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   

3. *Install Dependencies*
   bash
   pip install -r requirements.txt
   

4. *Download Dataset*
   - The system uses the MovieLens dataset (e.g., movies.csv, ratings.csv, tags.csv).
   - Place these files in the dataset/ directory. You can download them from [MovieLens](https://grouplens.org/datasets/movielens/).

5. *Set Up MongoDB*
   - Ensure MongoDB is running locally (mongod).
   - Create a database named movie_recommender and a collection named ratings.
   - Add a sample user for testing:
     json
     {
       "username": "123",
       "userId": 1
     }
     

6. *Run the Application*
   - Start the FastAPI backend:
     bash
     uvicorn main:app --reload
     
   - Start the Streamlit frontend:
     bash
     streamlit run streamlit_app.py
     

---

## Usage

1. *Access the Frontend*
   - Open your browser and go to http://localhost:8501 to access the Streamlit UI.

2. *Sign In or Use Guest Mode*
   - Choose "Sign In" and use a username (e.g., "123") or proceed as a guest.

3. *Rate Movies*
   - Enter movie names and assign ratings (1-5).

4. *Get Recommendations*
   - Select a model (Content-Based, SVD, or Hybrid) and click "Get Recommendations."

5. *Provide Feedback*
   - Rate recommended movies and click "Give Feedback and Get New Movies" to refine suggestions.

6. *Exit*
   - Click "Exit" to clear the session and start over.

---

## API Endpoints

The FastAPI backend provides the following endpoints:

| Endpoint                          | Method | Description                              | Request Body Example                          |
|-----------------------------------|--------|------------------------------------------|-----------------------------------------------|
| /suggest_movies                 | GET    | Suggest random movies                   | None                                          |
| /recommend_movies_by_content_based | POST   | Content-based recommendations          | {"movies_dict": {"Movie1": 4, "Movie2": 3}} |
| /recommend_movies_by_svd        | POST   | SVD-based recommendations              | {"user_rating": {"Movie1": 4}, "user_id": 1} |
| /recommend_movies_by_hybrid     | POST   | Hybrid recommendations                 | {"movies_dict": {"Movie1": 4}, "user_id": 1} |
| /evaluate_recommendations       | POST   | Evaluate model performance             | {"user_ratings": {"Movie1": 4}, "model_type": "svd"} |
| /collect_feedback               | POST   | Collect user feedback                  | [{"movieId": 1, "rating": 4, "title": "Movie1"}] |

Access the API documentation at http://127.0.0.1:8000/docs when the server is running.

---

## Project Structure


movie-recommendation-system/
├── dataset/                # MovieLens dataset files (movies.csv, ratings.csv, tags.csv)
├── models/                 # Pre-trained SVD model (svd_model.pkl)
├── content_bases_model.py  # Content-based filtering logic
├── svd_bases_recomendor.py # SVD-based recommendation logic
├── hybrid_based_recomendor.py # Hybrid recommendation logic
├── feedback.py             # Model evaluation functions
├── movies.py               # Random movie suggestion logic
├── main.py                 # FastAPI application
├── streamlit_app.py        # Streamlit frontend
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation


---

## Dataset

The system uses the MovieLens dataset:
- *movies.csv*: Movie titles, IDs, and genres.
- *ratings.csv*: User ratings for movies.
- *tags.csv*: User-assigned tags for movies.

Ensure these files are placed in the dataset/ directory.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (git checkout -b feature/your-feature).
3. Commit your changes (git commit -m "Add your feature").
4. Push to the branch (git push origin feature/your-feature).
5. Open a pull request.

Please include tests and documentation updates where applicable.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README provides a comprehensive guide to your project. Adjust the repository URL, dataset links, or any specific details (e.g., MongoDB setup) as needed for your actual deployment. Let me know if you'd like further refinements!