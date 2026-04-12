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
# PURE EMOTION-BASED RECOMMENDATION
# ============================================
def recommend_by_mood_only(emotion, top_n=10):
    """
    TRUE EMOTION-BASED RECOMMENDATION
    
    This function completely ignores the selected movie.
    It only recommends movies that match the user's emotional state.
    """
    target_genres = emotion_map.get(emotion, [])
    
    if not target_genres:
        popular = data.groupby("title")["rating"].mean().sort_values(ascending=False).head(top_n)
        return popular.index.tolist()
    
    emotion_movies = []
    for idx, row in movies.iterrows():
        movie_genres = row['genres']
        if any(genre in movie_genres for genre in target_genres):
            avg_rating = data[data['title'] == row['title']]['rating'].mean()
            if pd.isna(avg_rating):
                avg_rating = 0
            
            emotion_movies.append({
                'title': row['title'],
                'rating': avg_rating,
                'genres': movie_genres,
                'year': row['year']
            })
    
    emotion_movies.sort(key=lambda x: x['rating'], reverse=True)
    return [m['title'] for m in emotion_movies[:top_n]]


# ============================================
# COLLABORATIVE FILTERING (FIXED - STRONGER BOOST)
# ============================================
user_movie_matrix = data.pivot_table(index="userId", columns="title", values="rating").fillna(0)
movie_similarity_df = pd.DataFrame(
    cosine_similarity(user_movie_matrix.T),
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)

def recommend_collaborative(movie_title, emotion, top_n=10):
    """
    COLLABORATIVE FILTERING - FIXED WITH STRONGER BOOST
    
    HOW IT WORKS:
    1. Find movies similar to the selected movie (what people also liked)
    2. Apply STRONG emotion boost (10x for mood match, 0.1x for no match)
    3. This keeps recommendations similar to Toy Story, but re-ranks them by mood
    """
    
    if movie_title not in movie_similarity_df.columns:
        return recommend_by_mood_only(emotion, top_n)
    
    # Get similar movies
    similar_scores = movie_similarity_df[movie_title].sort_values(ascending=False)
    similar_scores = similar_scores.drop(movie_title, errors='ignore')
    
    target_genres = emotion_map.get(emotion, [])
    
    scored_movies = []
    consider_count = min(top_n * 5, len(similar_scores))
    
    for movie, similarity in similar_scores.head(consider_count).items():
        # Get movie rating
        movie_rating_data = data[data["title"] == movie]["rating"]
        movie_avg_rating = movie_rating_data.mean() if not movie_rating_data.empty else 2.5
        
        # Calculate base score
        rating_weight = movie_avg_rating / 5.0
        base_score = similarity * rating_weight
        
        # Check emotion match
        movie_genres = movies[movies["title"] == movie]["genres"].values
        emotion_match = False
        if len(movie_genres) > 0:
            emotion_match = any(g in movie_genres[0] for g in target_genres)
        
        # STRONGER EMOTION BOOST (10x for match, 0.1x for no match)
        if emotion_match:
            final_score = base_score * 10.0   # Strong boost for mood match
        else:
            final_score = base_score * 0.1    # Heavy penalty for no match
        
        scored_movies.append({
            'title': movie,
            'score': final_score,
            'similarity': similarity,
            'emotion_match': emotion_match,
            'rating': movie_avg_rating
        })
    
    # Sort by score
    scored_movies.sort(key=lambda x: -x['score'])
    
    return [m['title'] for m in scored_movies[:top_n]]


# ============================================
# CONTENT-BASED FILTERING (WITH STRONG EMOTION)
# ============================================
def recommend_content_based(movie_title, emotion, top_n=10):
    """
    Content-Based Filtering with Strong Emotion Influence
    """
    if movie_title not in indices:
        return recommend_by_mood_only(emotion, top_n)
    
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim_cb[idx]))
    
    target_genres = [g.lower() for g in emotion_map.get(emotion, [])]
    
    scored_movies = []
    for i, score in sim_scores:
        if i == idx:
            continue
        
        m_genres = movies.iloc[i]["genres"].lower()
        
        if any(g in m_genres for g in target_genres):
            final_score = score * 5.0
        else:
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
    """
    mood_recs = recommend_by_mood_only(emotion, top_n=top_n * 2)
    movie_recs = recommend_content_based(movie_title, emotion, top_n=top_n * 2)
    
    combined_scores = {}
    
    for rank, movie in enumerate(mood_recs):
        score = (len(mood_recs) - rank) * mood_weight
        combined_scores[movie] = combined_scores.get(movie, 0) + score
    
    for rank, movie in enumerate(movie_recs):
        score = (len(movie_recs) - rank) * movie_weight
        combined_scores[movie] = combined_scores.get(movie, 0) + score
    
    sorted_movies = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie, score in sorted_movies[:top_n]]


# ============================================
# LEGACY FUNCTIONS (Keep for compatibility)
# ============================================
def recommend_movies(movie_title, emotion, top_n=10):
    return recommend_collaborative(movie_title, emotion, top_n)

def recommend_movies_cb(movie_title, emotion, top_n=10):
    return recommend_content_based(movie_title, emotion, top_n)

def recommend_movies_hybrid(movie_title, emotion, top_n=10):
    return recommend_hybrid(movie_title, emotion, top_n)


# ============================================
# UTILITY FUNCTIONS
# ============================================
def get_popular_movies(n=10):
    popular = data.groupby("title")["rating"].mean().sort_values(ascending=False).head(n)
    return popular

def get_movie_details(movie):
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
# EVALUATION METRICS
# ============================================
from sklearn.metrics import mean_squared_error
from math import sqrt
import random

def evaluate_recommendations(test_user_id, algorithm_type="collaborative", top_n=10):
    # Uncomment for consistent results: random.seed(42)
    
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
