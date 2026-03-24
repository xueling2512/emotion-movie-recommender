import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# Merge dataset
data = pd.merge(ratings, movies, on="movieId")

# Emotion → Genre mapping
emotion_map = {
    "Happy": ["Comedy", "Animation"],
    "Sad": ["Drama"],
    "Stressed": ["Comedy"],
    "Excited": ["Action"],
    "Romantic": ["Romance"]
}

# Pre-calculate Collaborative Similarity
user_movie_matrix = data.pivot_table(index="userId", columns="title", values="rating").fillna(0)
movie_similarity_df = pd.DataFrame(cosine_similarity(user_movie_matrix.T), 
                                   index=user_movie_matrix.columns, 
                                   columns=user_movie_matrix.columns)

def recommend_movies_cb(movie_title, emotion):
    if movie_title not in indices: return []
    idx = indices[movie_title]
    
    # ✅ FIX: Increased pool to 150 so the mood filter actually finds matches
    sim_scores = sorted(list(enumerate(cosine_sim_cb[idx])), key=lambda x: x[1], reverse=True)[1:150]
    
    movie_indices = [i[0] for i in sim_scores]
    genres = emotion_map.get(emotion, [])
    
    recommended, fallback = [], []
    for i in movie_indices:
        m_title = movies.iloc[i]["title"]
        m_genres = movies.iloc[i]["genres"]
        
        # Match mood
        if any(g.lower() in m_genres.lower() for g in genres):
            recommended.append(m_title)
        else:
            fallback.append(m_title)
            
    return (recommended + fallback)[:10]

def recommend_movies(movie_title, emotion):

    if movie_title not in movie_similarity_df.columns:
        return []

    similar_scores = movie_similarity_df[movie_title].sort_values(ascending=False)

    similar_scores = similar_scores.drop(movie_title, errors='ignore')

    genres = emotion_map.get(emotion, [])

    recommended = []
    fallback = []   # store similar movies without emotion filter

    for movie in similar_scores.index:

        movie_genres = movies[movies["title"] == movie]["genres"].values

        if len(movie_genres) > 0:

            # Check emotion match
            if any(g in movie_genres[0] for g in genres):
                recommended.append(movie)
            else:
                fallback.append(movie)

        # Stop early if enough
        if len(recommended) >= 10:
            break

    # ⭐ If not enough emotion-matching movies → fill with similar movies
    if len(recommended) < 10:
        for movie in fallback:
            if movie not in recommended:
                recommended.append(movie)
            if len(recommended) >= 10:
                break

    return recommended

# Get top popular movies (based on average rating)
def get_popular_movies():
    popular = data.groupby("title")["rating"].mean().sort_values(ascending=False).head(10)
    return popular

# Get movie details (genre + rating)
def get_movie_details(movie):
    movie_info = movies[movies["title"] == movie]

    avg_rating = data[data["title"] == movie]["rating"].mean()

    return {
        "genres": movie_info["genres"].values[0].replace("|", ", ")
        if len(movie_info) > 0 else "N/A",
        "rating": round(avg_rating, 2) if avg_rating else "N/A"
    }
