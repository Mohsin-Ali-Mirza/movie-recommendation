# hybrid_model.py
import pandas as pd
from content_bases_model import recommend_movies_content
from svd_bases_recomendor import recommend_movies_svd


def recommend_movies_hybrid(svd_model, df_movies, user_movies, all_unique_genres,n_recommendations=5):
    """Recommend movies using hybrid approach (SVD + Content-based)."""
    
    # Content-based recommendation
    content_recs = recommend_movies_content(df_movies, user_movies,all_unique_genres)
    print(content_recs)
    # SVD-based recommendation
    # svd_recs = recommend_movies_svd(svd_model, df_movies, user_movies, n_recommendations)

    # # Combine recommendations
    # combined_recs = pd.concat([svd_recs, content_recs]).drop_duplicates(subset=["movieId"])
    
    # # Randomly sample to avoid bias
    # hybrid_recs = combined_recs.sample(n=n_recommendations, random_state=42)

    # return hybrid_recs
    return 1
