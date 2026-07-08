import streamlit as st
import requests

# Initialize the global URL dictionary
# BASE_URL = "http://127.0.0.1:8000"
BASE_URL = "http://backend-service.default.svc.cluster.local:8000"
urls = {
    "suggest_movies": f"{BASE_URL}/suggest_movies",
    "recommend_movies_by_content_based": f"{BASE_URL}/recommend_movies_by_content_based",
    "recommend_movies_by_svd": f"{BASE_URL}/recommend_movies_by_svd",
    "recommend_movies_by_hybrid": f"{BASE_URL}/recommend_movies_by_hybrid",
    "collect_feedback": f"{BASE_URL}/collect_feedback"
}

def create_UI():
    st.set_page_config("Movie Recommendation System", initial_sidebar_state="collapsed")

    st.markdown("""
        <style>
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateX(-20px); }
            100% { opacity: 1; transform: translateX(0); }
        }
        .welcome-message { font-size: 48px; text-align: center; color: black; animation: fadeIn 2.0s ease forwards; margin-top: 50px; }
        .stTextInput > div > input { height: 100%; width: 100%; font-size: 14px; }
        </style>
    """, unsafe_allow_html=True)

    st.title("Movie Rating and Recommendation System")

    # Add Sign In/Without Sign In selection
    auth_mode = st.radio("Choose an option:", ["Without Sign In", "Sign In"], index=0, horizontal=True)

    if auth_mode == "Without Sign In":
        st.write("You are using the system as a guest. No sign-in required.")
    # Note: Sign In logic will be implemented later; for now, proceed with Without Sign In

    st.header("Rate Movies")
    movies = ["Movie Name 1", "Movie Name 2", "Movie Name 3"]  # Replace with actual movie names

    if "ratings_dict" not in st.session_state:
        st.session_state.ratings_dict = {}

    for i, movie in enumerate(movies):
        col1, col2 = st.columns([2, 1])
        with col1:
            movie_name = st.text_input(f"Rate {movie}", "", key=f"movie_name_{i}")
        with col2:
            rating = st.slider(f"Rating {movie}", 1, 5, 3, key=f"rating_{i}", label_visibility="hidden")
        if movie_name.strip():
            st.session_state.ratings_dict[movie_name] = rating

    st.header("Suggested Movies")
    if "suggested_movies" not in st.session_state:
        response = requests.get(urls["suggest_movies"])
        st.session_state.suggested_movies = response.json() if response.status_code == 200 else ["Movie1", "Movie2", "Movie3"]

    display_movies = st.selectbox("Select a movie:", st.session_state.suggested_movies, label_visibility="hidden")

    st.header("Model Selection")
    model_selected = st.selectbox("Select a model:", ["Content Based", "Hybrid", "SVD Based"])
    st.session_state.model = model_selected

    if "recommended_movies" not in st.session_state:
        st.session_state.recommended_movies = []

    recomedation_button = st.button("Get Recommendations")

    if recomedation_button:
        with st.spinner("Fetching recommendations..."):
            if model_selected == "Content Based":
                response = requests.post(urls["recommend_movies_by_content_based"], json=st.session_state.ratings_dict)
            elif model_selected == "SVD Based":
                response = requests.post(urls["recommend_movies_by_svd"], json=st.session_state.ratings_dict)
            elif model_selected == "Hybrid":
                response = requests.post(urls["recommend_movies_by_hybrid"], json=st.session_state.ratings_dict)

            if response.status_code == 200:
                st.session_state.recommended_movies = response.json()
            else:
                st.error("Failed to fetch recommendations")

    # Display Recommended Movies and Allow Rating
    if st.session_state.recommended_movies:
        st.header(f"Recommended Movies ({model_selected})")
        for movie in st.session_state.recommended_movies:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(movie.get("title", "Unknown"))
            with col2:
                unique_key = f"rec_rating_{movie.get('movieId', 0)}"
                initial_rating = st.session_state.ratings_dict.get(movie.get("title", "Unknown"), 0)
                rating = st.slider(
                    "Rating", 0, 5, initial_rating,
                    key=unique_key, label_visibility="hidden"
                )
                st.session_state.ratings_dict[movie.get("title", "Unknown")] = rating

        # Button to Give Feedback and Get New Movies (for initial recommendations)
        if st.button("Give Feedback and Get New Movies"):
            with st.spinner("Processing feedback and fetching new recommendations..."):
                feedback_list = []
                for movie in st.session_state.recommended_movies:
                    movie_id = movie.get('movieId')
                    rating = st.session_state.ratings_dict.get(movie.get("title", "Unknown"), 0)
                    if rating is not None:
                        feedback_list.append({
                            "movieId": movie_id,
                            "rating": rating,
                            "title": movie.get("title", "Unknown Movie"),
                            "genres": movie.get("genres", "Unknown Genre"),
                        })

                if feedback_list:
                    feedback_response = requests.post(urls["collect_feedback"], json=feedback_list)
                    if feedback_response.status_code == 200:
                        api_response_data = feedback_response.json()
                        st.success(api_response_data.get("message", "Feedback submitted successfully!"))

                        user_ratings_for_next_recs = st.session_state.ratings_dict.copy()
                        if model_selected == "Content Based":
                            recs_response = requests.post(urls["recommend_movies_by_content_based"], json=user_ratings_for_next_recs)
                        elif model_selected == "SVD Based":
                            recs_response = requests.post(urls["recommend_movies_by_svd"], json=user_ratings_for_next_recs)
                        elif model_selected == "Hybrid":
                            recs_response = requests.post(urls["recommend_movies_by_hybrid"], json=user_ratings_for_next_recs)

                        if recs_response.status_code == 200:
                            st.session_state.recommended_movies = recs_response.json()
                            st.rerun()
                        else:
                            st.error(f"Failed to get new recommendations: {recs_response.status_code}")
                    else:
                        st.error(f"Failed to submit feedback: {feedback_response.status_code}")
                else:
                    st.warning("No ratings to submit for feedback.")

    # Content-Based Feedback Loop (Infinite)
    st.header("Content-Based Feedback Loop")
    if 'content_feedback_recommended_movies' not in st.session_state:
        st.session_state.content_feedback_recommended_movies = []
    if 'content_feedback_ratings' not in st.session_state:
        st.session_state.content_feedback_ratings = {}

    if st.button("Start Content-Based Feedback Loop"):
        user_ratings_for_feedback = st.session_state.ratings_dict.copy()
        response = requests.post(urls["recommend_movies_by_content_based"], json=user_ratings_for_feedback)
        if response.status_code == 200:
            st.session_state.content_feedback_recommended_movies = response.json()
            st.session_state.content_feedback_ratings = {}
        else:
            st.error(f"Failed to get recommendations for content-based feedback loop: {response.status_code}")

    if st.session_state.content_feedback_recommended_movies:
        st.subheader("Rate these Content-Based Recommendations for Feedback:")
        for movie in st.session_state.content_feedback_recommended_movies:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(movie.get("title", "Unknown Movie"))
            with col2:
                rating_key = f"content_feedback_rating_{movie.get('movieId', 0)}"
                rating = st.slider(
                    "Rating", 0, 5, 3, key=rating_key, label_visibility="hidden"
                )
                st.session_state.content_feedback_ratings[movie.get('movieId')] = rating

        if st.button("Submit Feedback and Get New Content-Based Recommendations"):
            feedback_list = []
            for movie in st.session_state.content_feedback_recommended_movies:
                movie_id = movie.get('movieId')
                feedback_rating = st.session_state.content_feedback_ratings.get(movie_id)
                if feedback_rating is not None:
                    feedback_list.append({
                        "movieId": movie_id,
                        "rating": feedback_rating,
                        "title": movie.get("title", "Unknown Movie"),
                        "genres": movie.get("genres", "Unknown Genre"),
                    })

            if feedback_list:
                feedback_response = requests.post(urls["collect_feedback"], json=feedback_list)
                if feedback_response.status_code == 200:
                    api_response_data = feedback_response.json()
                    st.success(api_response_data.get("message", "Feedback submitted successfully!"))

                    user_ratings_for_next_recs = st.session_state.ratings_dict.copy()
                    recs_response = requests.post(urls["recommend_movies_by_content_based"], json=user_ratings_for_next_recs)
                    if recs_response.status_code == 200:
                        st.session_state.content_feedback_recommended_movies = recs_response.json()
                        st.session_state.content_feedback_ratings = {}
                        st.rerun()
                    else:
                        st.error(f"Failed to get new content-based recommendations: {recs_response.status_code}")
                else:
                    st.error(f"Failed to submit feedback: {feedback_response.status_code}")

    # SVD-Based Feedback Loop (Infinite)
    st.header("SVD-Based Feedback Loop")
    if 'svd_feedback_recommended_movies' not in st.session_state:
        st.session_state.svd_feedback_recommended_movies = []
    if 'svd_feedback_ratings' not in st.session_state:
        st.session_state.svd_feedback_ratings = {}

    if st.button("Start SVD-Based Feedback Loop"):
        user_ratings_for_feedback = st.session_state.ratings_dict.copy()
        response = requests.post(urls["recommend_movies_by_svd"], json=user_ratings_for_feedback)
        if response.status_code == 200:
            st.session_state.svd_feedback_recommended_movies = response.json()
            st.session_state.svd_feedback_ratings = {}
        else:
            st.error(f"Failed to get recommendations for SVD-based feedback loop: {response.status_code}")

    if st.session_state.svd_feedback_recommended_movies:
        st.subheader("Rate these SVD-Based Recommendations for Feedback:")
        for movie in st.session_state.svd_feedback_recommended_movies:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(movie.get("title", "Unknown Movie"))
            with col2:
                rating_key = f"svd_feedback_rating_{movie.get('movieId', 0)}"
                rating = st.slider(
                    "Rating", 0, 5, 3, key=rating_key, label_visibility="hidden"
                )
                st.session_state.svd_feedback_ratings[movie.get('movieId')] = rating

        if st.button("Submit Feedback and Get New SVD-Based Recommendations"):
            feedback_list = []
            for movie in st.session_state.svd_feedback_recommended_movies:
                movie_id = movie.get('movieId')
                feedback_rating = st.session_state.svd_feedback_ratings.get(movie_id)
                if feedback_rating is not None:
                    feedback_list.append({
                        "movieId": movie_id,
                        "rating": feedback_rating,
                        "title": movie.get("title", "Unknown Movie"),
                        "genres": movie.get("genres", "Unknown Genre"),
                    })

            if feedback_list:
                feedback_response = requests.post(urls["collect_feedback"], json=feedback_list)
                if feedback_response.status_code == 200:
                    api_response_data = feedback_response.json()
                    st.success(api_response_data.get("message", "Feedback submitted successfully!"))

                    user_ratings_for_next_recs = st.session_state.ratings_dict.copy()
                    recs_response = requests.post(urls["recommend_movies_by_svd"], json=user_ratings_for_next_recs)
                    if recs_response.status_code == 200:
                        st.session_state.svd_feedback_recommended_movies = recs_response.json()
                        st.session_state.svd_feedback_ratings = {}
                        st.rerun()
                    else:
                        st.error(f"Failed to get new SVD-based recommendations: {recs_response.status_code}")
                else:
                    st.error(f"Failed to submit feedback: {feedback_response.status_code}")

    # Hybrid-Based Feedback Loop (Infinite)
    st.header("Hybrid-Based Feedback Loop")
    if 'hybrid_feedback_recommended_movies' not in st.session_state:
        st.session_state.hybrid_feedback_recommended_movies = []
    if 'hybrid_feedback_ratings' not in st.session_state:
        st.session_state.hybrid_feedback_ratings = {}

    if st.button("Start Hybrid-Based Feedback Loop"):
        user_ratings_for_feedback = st.session_state.ratings_dict.copy()
        response = requests.post(urls["recommend_movies_by_hybrid"], json=user_ratings_for_feedback)
        if response.status_code == 200:
            st.session_state.hybrid_feedback_recommended_movies = response.json()
            st.session_state.hybrid_feedback_ratings = {}
        else:
            st.error(f"Failed to get recommendations for hybrid-based feedback loop: {response.status_code}")

    if st.session_state.hybrid_feedback_recommended_movies:
        st.subheader("Rate these Hybrid-Based Recommendations for Feedback:")
        for movie in st.session_state.hybrid_feedback_recommended_movies:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(movie.get("title", "Unknown Movie"))
            with col2:
                rating_key = f"hybrid_feedback_rating_{movie.get('movieId', 0)}"
                rating = st.slider(
                    "Rating", 0, 5, 3, key=rating_key, label_visibility="hidden"
                )
                st.session_state.hybrid_feedback_ratings[movie.get('movieId')] = rating

        if st.button("Submit Feedback and Get New Hybrid-Based Recommendations"):
            feedback_list = []
            for movie in st.session_state.hybrid_feedback_recommended_movies:
                movie_id = movie.get('movieId')
                feedback_rating = st.session_state.hybrid_feedback_ratings.get(movie_id)
                if feedback_rating is not None:
                    feedback_list.append({
                        "movieId": movie_id,
                        "rating": feedback_rating,
                        "title": movie.get("title", "Unknown Movie"),
                        "genres": movie.get("genres", "Unknown Genre"),
                    })

            if feedback_list:
                feedback_response = requests.post(urls["collect_feedback"], json=feedback_list)
                if feedback_response.status_code == 200:
                    api_response_data = feedback_response.json()
                    st.success(api_response_data.get("message", "Feedback submitted successfully!"))

                    user_ratings_for_next_recs = st.session_state.ratings_dict.copy()
                    recs_response = requests.post(urls["recommend_movies_by_hybrid"], json=user_ratings_for_next_recs)
                    if recs_response.status_code == 200:
                        st.session_state.hybrid_feedback_recommended_movies = recs_response.json()
                        st.session_state.hybrid_feedback_ratings = {}
                        st.rerun()
                    else:
                        st.error(f"Failed to get new hybrid-based recommendations: {recs_response.status_code}")
                else:
                    st.error(f"Failed to submit feedback: {feedback_response.status_code}")

    if st.button("Exit"):
        st.session_state.clear()
        st.rerun()

if __name__ == "__main__":
    create_UI()