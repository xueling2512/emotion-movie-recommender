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
# EMOTION TO GENRE MAPPING (EXPANDED)
# ============================================
emotion_map = {
    "Happy": ["Comedy", "Animation", "Musical", "Children"],
    "Sad": ["Drama", "Romance", "Film-Noir"],
    "Stressed": ["Comedy", "Animation", "Musical", "Children"],
    "Excited": ["Action", "Adventure", "Thriller", "Sci-Fi"],
    "Romantic": ["Romance", "Drama", "Comedy"]
}

# ============================================
# NEW: PURE EMOTION-BASED RECOMMENDATION
# ============================================
def recommend_by_mood_only(emotion, top_n=10):
    """
    TRUE EMOTION-BASED RECOMMENDATION
    
    This function completely ignores the selected movie.
    It only recommends movies that match the user's emotional state.
    
    How it works:
    1. User selects a mood (e.g., "Happy")
    2. System finds all movies with Comedy, Animation, Musical, or Children genres
    3. Sorts those movies by average rating (highest first)
    4. Returns top N
    
    This is what makes the system TRULY emotion-based.
    """
    target_genres = emotion_map.get(emotion, [])
    
    if not target_genres:
        # Fallback: return popular movies
        popular = data.groupby("title")["rating"].mean().sort_values(ascending=False).head(top_n)
        return popular.index.tolist()
    
    # Find all movies matching the emotion's genres
    emotion_movies = []
    for idx, row in movies.iterrows():
        movie_genres = row['genres']
        # Check if movie matches ANY target genre
        if any(genre in movie_genres for genre in target_genres):
            # Get average rating from all users
            avg_rating = data[data['title'] == row['title']]['rating'].mean()
            if pd.isna(avg_rating):
                avg_rating = 0
            
            # Get TMDB ID for poster
            movie_id = row['movieId']
            tmdb_match = links[links['movieId'] == movie_id]
            tmdb_id = tmdb_match['tmdbId'].values[0] if not tmdb_match.empty else None
            
            emotion_movies.append({
                'title': row['title'],
                'rating': avg_rating,
                'genres': movie_genres,
                'tmdbId': tmdb_id,
                'year': row['year']
            })
    
    # Sort by rating (highest first)
    emotion_movies.sort(key=lambda x: x['rating'], reverse=True)
    
    # Return top N titles
    return [m['title'] for m in emotion_movies[:top_n]]


# ============================================
# COLLABORATIVE FILTERING (WITH EMOTION BOOST)
# ============================================
user_movie_matrix = data.pivot_table(index="userId", columns="title", values="rating").fillna(0)
movie_similarity_df = pd.DataFrame(
    cosine_similarity(user_movie_matrix.T),
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)

def recommend_collaborative(movie_title, emotion, top_n=10):
    """
    Collaborative Filtering with Emotion Boost
    
    How it works:
    1. Find movies similar to the user's selected movie (what other users liked)
    2. Apply emotion boost: movies matching the mood get higher priority
    3. Sort and return
    """
    if movie_title not in movie_similarity_df.columns:
        return recommend_by_mood_only(emotion, top_n)  # Fallback to mood-only
    
    # Get similarity scores
    similar_scores = movie_similarity_df[movie_title].sort_values(ascending=False)
    similar_scores = similar_scores.drop(movie_title, errors='ignore')
    
    target_genres = emotion_map.get(emotion, [])
    
    scored_movies = []
    consider_count = min(top_n * 3, len(similar_scores))
    
    for movie, similarity in similar_scores.head(consider_count).items():
        # Get movie rating
        movie_rating_data = data[data["title"] == movie]["rating"]
        movie_avg_rating = movie_rating_data.mean() if not movie_rating_data.empty else 2.5
        
        # Calculate base score
        rating_weight = movie_avg_rating / 5.0
        base_score = similarity * rating_weight
        
        # Apply emotion boost
        movie_genres = movies[movies["title"] == movie]["genres"].values
        emotion_match = False
        if len(movie_genres) > 0:
            emotion_match = any(g in movie_genres[0] for g in target_genres)
        
        # Emotion match gives 3x boost
        final_score = base_score * (3.0 if emotion_match else 0.5)
        
        scored_movies.append({
            'title': movie,
            'score': final_score,
            'emotion_match': emotion_match
        })
    
    # Sort by score (emotion-match already factored in)
    scored_movies.sort(key=lambda x: -x['score'])
    
    return [m['title'] for m in scored_movies[:top_n]]


# ============================================
# CONTENT-BASED FILTERING (WITH STRONG EMOTION)
# ============================================
def recommend_content_based(movie_title, emotion, top_n=10):
    """
    Content-Based Filtering with Strong Emotion Influence
    
    How it works:
    1. Find movies with similar genres to the user's selected movie
    2. Apply emotion boost: movies matching mood get 5x boost
    3. Non-matching movies are heavily penalized (0.2x)
    """
    if movie_title not in indices:
        return recommend_by_mood_only(emotion, top_n)  # Fallback to mood-only
    
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim_cb[idx]))
    
    target_genres = [g.lower() for g in emotion_map.get(emotion, [])]
    
    scored_movies = []
    for i, score in sim_scores:
        if i == idx:
            continue
        
        m_genres = movies.iloc[i]["genres"].lower()
        
        # Check emotion match
        if any(g in m_genres for g in target_genres):
            # Emotion match: full score with 5x boost
            final_score = score * 5.0
        else:
            # No emotion match: heavily penalized
            final_score = score * 0.2
        
        scored_movies.append((i, final_score))
    
    scored_movies.sort(key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in scored_movies[:top_n]]
    
    return movies.iloc[recommended_indices]["title"].tolist()


# ============================================
# HYBRID: Mood-First + Reference Movie
# ============================================
def recommend_hybrid(movie_title, emotion, top_n=10, mood_weight=0.6, movie_weight=0.4):
    """
    TRUE HYBRID with MOOD as primary factor (60% weight)
    
    This combines:
    - 60%: Pure mood-based recommendations (movies that match your emotion)
    - 40%: Movie-similarity recommendations (movies similar to your selection)
    
    This ensures recommendations are TRULY emotion-based, not just similarity-based.
    """
    # Get mood-based recommendations (pure emotion, ignores selected movie)
    mood_recs = recommend_by_mood_only(emotion, top_n=top_n * 2)
    
    # Get movie-based recommendations (content-based similarity)
    movie_recs = recommend_content_based(movie_title, emotion, top_n=top_n * 2)
    
    # Combine scores with mood_weight as dominant
    combined_scores = {}
    
    # Score mood-based recs (higher weight = more important)
    for rank, movie in enumerate(mood_recs):
        score = (len(mood_recs) - rank) * mood_weight
        combined_scores[movie] = combined_scores.get(movie, 0) + score
    
    # Score movie-based recs (lower weight)
    for rank, movie in enumerate(movie_recs):
        score = (len(movie_recs) - rank) * movie_weight
        combined_scores[movie] = combined_scores.get(movie, 0) + score
    
    # Sort and return
    sorted_movies = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie, score in sorted_movies[:top_n]]


# ============================================
# LEGACY FUNCTIONS (Keep for compatibility)
# ============================================
def recommend_movies(movie_title, emotion, top_n=10):
    """Legacy wrapper - use recommend_collaborative"""
    return recommend_collaborative(movie_title, emotion, top_n)

def recommend_movies_cb(movie_title, emotion, top_n=10):
    """Legacy wrapper - use recommend_content_based"""
    return recommend_content_based(movie_title, emotion, top_n)

def recommend_movies_hybrid(movie_title, emotion, top_n=10):
    """Legacy wrapper - use recommend_hybrid"""
    return recommend_hybrid(movie_title, emotion, top_n)


# ============================================
# UTILITY FUNCTIONS (Unchanged)
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
    
    movie_id = movie_info["movieId"].values[0]
    tmdb_match = links[links["movieId"] == movie_id]
    tmdb_id = tmdb_match["tmdbId"].values[0] if not tmdb_match.empty else None
    
    avg_rating = data[data["title"] == movie]["rating"].mean()
    
    return {
        "genres": movie_info["genres"].values[0],
        "rating": round(avg_rating, 2) if not pd.isna(avg_rating) else "N/A",
        "tmdbId": tmdb_id
    }

def global_browse_movies(movies_df, ratings_df, selected_genres=None, min_rating=0.0, 
                         sort_by="Rating", sort_order="Descending", year_range=None):
    """Filter and sort movies for the Browse Library tab"""
    df = movies_df.copy()
    
    avg_ratings = ratings_df.groupby("movieId")["rating"].mean().reset_index()
    df = df.merge(avg_ratings, on="movieId", how="left")
    df["rating"] = df["rating"].fillna(0)
    
    global links
    df = df.merge(links[['movieId', 'tmdbId']], on="movieId", how="left")
    
    if year_range:
        df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
    
    if selected_genres:
        genre_regex = "|".join(selected_genres)
        df = df[df["genres"].str.contains(genre_regex, case=False, na=False)]
    
    df = df[df["rating"] >= min_rating]
    
    sort_map = {"Rating": "rating", "A-Z": "title", "Year": "year"}
    target_column = sort_map.get(sort_by, "rating")
    ascending_bool = True if sort_order == "Ascending" else False
    
    df = df.sort_values(by=target_column, ascending=ascending_bool)
    
    return df


# ============================================
# EVALUATION METRICS (Unchanged)
# ============================================
from sklearn.metrics import mean_squared_error
from math import sqrt
import random

def evaluate_recommendations(test_user_id, algorithm_type="collaborative", top_n=10):
    user_actual = data[data["userId"] == test_user_id]
    liked_movies = set(user_actual[user_actual["rating"] >= 4.0]["title"].tolist())
    
    if len(liked_movies) == 0:
        return {"precision": 0, "recall": 0, "f1": 0, "error": "No liked movies found"}
    
    seed_movie = random.choice(list(liked_movies))
    
    if algorithm_type == "collaborative":
        recommendations = recommend_movies(seed_movie, "Happy", top_n=top_n)
    elif algorithm_type == "content":
        recommendations = recommend_movies_cb(seed_movie, "Happy", top_n=top_n)
    else:
        recommendations = recommend_movies_hybrid(seed_movie, "Happy", top_n=top_n)
    
    recommended_set = set(recommendations)
    relevant_recommended = len(liked_movies.intersection(recommended_set))
    
    precision = relevant_recommended / len(recommended_set) if len(recommended_set) > 0 else 0
    recall = relevant_recommended / len(liked_movies) if len(liked_movies) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_liked": len(liked_movies),
        "recommendations_found": len(recommended_set)
    }

def calculate_rmse():
    actual_ratings = data["rating"].values
    movie_avg_ratings = data.groupby("movieId")["rating"].mean().to_dict()
    
    predictions = []
    actuals = []
    
    for _, row in data.iterrows():
        movie_id = row["movieId"]
        actual = row["rating"]
        predicted = movie_avg_ratings.get(movie_id, data["rating"].mean())
        actuals.append(actual)
        predictions.append(predicted)
    
    rmse = sqrt(mean_squared_error(actuals, predictions))
    
    return {
        "rmse": round(rmse, 4),
        "total_predictions": len(actuals),
        "global_avg_rating": round(data["rating"].mean(), 2)
    }
