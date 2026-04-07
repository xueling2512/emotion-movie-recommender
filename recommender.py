import pandas as pd
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# DATA LOADING
# ============================================
links = pd.read_csv("dataset/links.csv")
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")
data = pd.merge(ratings, movies, on="movieId")

# Extract year from movie title
movies['year'] = movies['title'].apply(lambda x: int(re.search(r'\((\d{4})\)', x).group(1)) if re.search(r'\((\d{4})\)', x) else 0)

# ============================================
# CONTENT-BASED SIMILARITY (TF-IDF + Cosine)
# ============================================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies["genres"])
cosine_sim_cb = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# ============================================
# EMOTION TO GENRE MAPPING
# ============================================
emotion_map = {
    "Happy": ["Comedy", "Animation"],
    "Sad": ["Drama"],
    "Stressed": ["Comedy", "Sci-Fi"],
    "Excited": ["Action", "Adventure"],
    "Romantic": ["Romance"]
}

# ============================================
# COLLABORATIVE FILTERING (Item-Item)
# Pre-calculate movie similarity matrix
# ============================================
user_movie_matrix = data.pivot_table(index="userId", columns="title", values="rating").fillna(0)
movie_similarity_df = pd.DataFrame(
    cosine_similarity(user_movie_matrix.T),
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)

# ============================================
# RECOMMENDATION FUNCTION 1: COLLABORATIVE FILTERING (IMPROVED)
# Modified with rating-weighted scoring
# 
# ============================================
def recommend_movies(movie_title, emotion, top_n=10):
    """
    Improved Item-Item Collaborative Filtering
    - Finds movies similar to the input movie using cosine similarity
    - Weighs each similar movie by its average rating (higher rated = better)
    - Filters by emotion/genre matching
    - Returns top N recommendations (N = top_n parameter)
    
    Args:
        movie_title (str): Title of the movie user likes
        emotion (str): User's current mood (Happy, Sad, etc.)
        top_n (int): Number of recommendations to return (from slider)
    
    Returns:
        list: Recommended movie titles
    """
    
    # Check if movie exists in our similarity matrix
    if movie_title not in movie_similarity_df.columns:
        return []
    
    # Get similarity scores for the input movie, sorted highest to lowest
    similar_scores = movie_similarity_df[movie_title].sort_values(ascending=False)
    
    # Remove the input movie itself from recommendations
    similar_scores = similar_scores.drop(movie_title, errors='ignore')
    
    # Get genres that match the selected emotion
    target_genres = emotion_map.get(emotion, [])
    
    # Calculate weighted scores for similar movies
    # Consider more movies than needed (2x) to have enough after filtering
    scored_movies = []
    consider_count = min(top_n * 3, len(similar_scores))  # Look at 3x the needed amount
    
    for movie, similarity in similar_scores.head(consider_count).items():
        # Get this movie's average rating from all users
        movie_rating_data = data[data["title"] == movie]["rating"]
        movie_avg_rating = movie_rating_data.mean() if not movie_rating_data.empty else 2.5
        
        # Calculate weighted score: similarity * (normalized rating)
        # This gives higher scores to similar movies that are also highly rated
        rating_weight = movie_avg_rating / 5.0  # Normalize rating to 0-1 range
        weighted_score = similarity * rating_weight
        
        # Check if this movie matches the user's emotion
        movie_genres = movies[movies["title"] == movie]["genres"].values
        emotion_match = False
        if len(movie_genres) > 0:
            emotion_match = any(g in movie_genres[0] for g in target_genres)
        
        scored_movies.append({
            'title': movie,
            'score': weighted_score,
            'similarity': similarity,
            'rating': movie_avg_rating,
            'emotion_match': emotion_match
        })
    
    # Sort: emotion match first, then by weighted score
    scored_movies.sort(key=lambda x: (not x['emotion_match'], -x['score']))
    
    # Extract just the titles for top N recommendations
    recommendations = [m['title'] for m in scored_movies[:top_n]]
    
    return recommendations


# ============================================
# RECOMMENDATION FUNCTION 2: CONTENT-BASED
# 
# ============================================
def recommend_movies_cb(movie_title, emotion, top_n=10):
    """
    Content-Based Filtering
    - Uses movie genres to find similar movies
    - Emotion acts as a boost multiplier
    - Returns top N recommendations (N = top_n parameter)
    """
    if movie_title not in indices:
        return []
    
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim_cb[idx]))
    
    target_genres = [g.lower() for g in emotion_map.get(emotion, [])]
    
    scored_movies = []
    for i, score in sim_scores:
        if i == idx:
            continue
        
        m_genres = movies.iloc[i]["genres"].lower()
        
        # Boost score if movie matches emotion genre
        bonus = 1.0
        if any(g in m_genres for g in target_genres):
            bonus = 5.0  # High multiplier for mood matching
        
        final_score = score * bonus
        scored_movies.append((i, final_score))
    
    scored_movies.sort(key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in scored_movies[:top_n]]  # Uses top_n here
    
    return movies.iloc[recommended_indices]["title"].tolist()


# ============================================
# RECOMMENDATION FUNCTION 3: HYBRID
# (Combines both approaches)
# ============================================
def recommend_movies_hybrid(movie_title, emotion, cb_weight=0.5, cf_weight=0.5, top_n=10):
    """
    Hybrid Recommender System
    - Combines Collaborative Filtering + Content-Based
    - Uses rank-based scoring to merge both lists
    - Returns top N recommendations (N = top_n parameter)
    """
    # Get recommendations from both algorithms (get more than needed for better merging)
    fetch_count = top_n * 2
    cb_recs = recommend_movies_cb(movie_title, emotion, top_n=fetch_count)
    cf_recs = recommend_movies(movie_title, emotion, top_n=fetch_count)
    
    hybrid_scores = {}
    
    # Score Content-Based recommendations (1st place = highest points)
    for rank, movie in enumerate(cb_recs):
        score = (len(cb_recs) - rank) * cb_weight
        hybrid_scores[movie] = hybrid_scores.get(movie, 0) + score
    
    # Score Collaborative Filtering recommendations
    for rank, movie in enumerate(cf_recs):
        score = (len(cf_recs) - rank) * cf_weight
        hybrid_scores[movie] = hybrid_scores.get(movie, 0) + score
    
    # Sort by combined score
    sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Extract titles for top N
    top_hybrid_movies = [movie for movie, score in sorted_hybrid][:top_n]
    
    return top_hybrid_movies


# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_popular_movies(n=10):
    """Get top N movies by average rating"""
    popular = data.groupby("title")["rating"].mean().sort_values(ascending=False).head(n)
    return popular


def get_movie_details(movie):
    """Get genres, rating, and TMDB ID for a movie"""
    movie_info = movies[movies["title"] == movie]
    if movie_info.empty:
        return {"genres": "N/A", "rating": "N/A", "tmdbId": None}
    
    # Get movieId to find tmdbId
    movie_id = movie_info["movieId"].values[0]
    tmdb_match = links[links["movieId"] == movie_id]
    tmdb_id = tmdb_match["tmdbId"].values[0] if not tmdb_match.empty else None
    
    # Get average rating
    avg_rating = data[data["title"] == movie]["rating"].mean()
    
    return {
        "genres": movie_info["genres"].values[0],
        "rating": round(avg_rating, 2) if not pd.isna(avg_rating) else "N/A",
        "tmdbId": tmdb_id
    }


def global_browse_movies(movies_df, ratings_df, selected_genres=None, min_rating=0.0, 
                         sort_by="Rating", sort_order="Descending", year_range=None):
    """
    Filter and sort movies for the Browse Library tab
    """
    df = movies_df.copy()
    
    # Attach average ratings
    avg_ratings = ratings_df.groupby("movieId")["rating"].mean().reset_index()
    df = df.merge(avg_ratings, on="movieId", how="left")
    df["rating"] = df["rating"].fillna(0)
    
    # Filter by Year Range
    if year_range:
        df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
    
    # Filter by Genre
    if selected_genres:
        genre_regex = "|".join(selected_genres)
        df = df[df["genres"].str.contains(genre_regex, case=False, na=False)]
    
    # Filter by Minimum Rating
    df = df[df["rating"] >= min_rating]
    
    # Sorting logic
    sort_map = {
        "Rating": "rating",
        "A-Z": "title",
        "Year": "year"
    }
    
    target_column = sort_map.get(sort_by, "rating")
    ascending_bool = True if sort_order == "Ascending" else False
    
    df = df.sort_values(by=target_column, ascending=ascending_bool)
    
    return df
