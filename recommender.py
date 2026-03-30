import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

links = pd.read_csv("dataset/links.csv")
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")
data = pd.merge(ratings, movies, on="movieId")

movies["genres"] = movies["genres"].fillna("").str.replace("|", " ", regex=False)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies["genres"])
cosine_sim_cb = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

emotion_map = {
    "Happy": ["Comedy", "Animation"],
    "Sad": ["Drama"],
    "Stressed": ["Comedy", "Sci-Fi"], # Added Sci-Fi for variety
    "Excited": ["Action", "Adventure"],
    "Romantic": ["Romance"]
}

# Pre-calculate Collaborative Similarity
user_movie_matrix = data.pivot_table(index="userId", columns="title", values="rating").fillna(0)
movie_similarity_df = pd.DataFrame(cosine_similarity(user_movie_matrix.T), 
                                   index=user_movie_matrix.columns, 
                                   columns=user_movie_matrix.columns)

def recommend_movies_cb(movie_title, emotion):
    if movie_title not in indices: 
        return []
    
    idx = indices[movie_title]
    # Get similarity scores for ALL movies
    sim_scores = list(enumerate(cosine_sim_cb[idx]))
    
    # Get genres for the selected emotion
    target_genres = [g.lower() for g in emotion_map.get(emotion, [])]
    
    scored_movies = []
    for i, score in sim_scores:
        if i == idx: continue  # Skip the movie itself
        
        m_genres = movies.iloc[i]["genres"].lower()
        
        # Calculate Bonus: If the movie matches an emotion genre, boost its score
        # Using a multiplier makes the emotion much more influential
        bonus = 1.0
        if any(g in m_genres for g in target_genres):
            bonus = 5.0 # High multiplier to ensure mood-matching movies rise to the top
            
        final_score = score * bonus
        scored_movies.append((i, final_score))

    # Sort by the new boosted score
    scored_movies = sorted(scored_movies, key=lambda x: x[1], reverse=True)
    
    # Take top 10
    recommended_indices = [i[0] for i in scored_movies[:10]]
    return movies.iloc[recommended_indices]["title"].tolist()

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

def recommend_movies_hybrid(movie_title, emotion, cb_weight=0.5, cf_weight=0.5):
    """
    A Rank-Based Hybrid Recommender.
    Assigns points to movies based on their rank from both algorithms.
    Movies that appear in both lists get a combined higher score.
    """
    # Fetch top 10 from both algorithms
    cb_recs = recommend_movies_cb(movie_title, emotion)
    cf_recs = recommend_movies(movie_title, emotion)
    
    hybrid_scores = {}
    
    # Score Content-Based recommendations (1st place = 10 pts, 10th place = 1 pt)
    for rank, movie in enumerate(cb_recs):
        score = (len(cb_recs) - rank) * cb_weight
        hybrid_scores[movie] = hybrid_scores.get(movie, 0) + score
        
    # Score Collaborative Filtering recommendations
    for rank, movie in enumerate(cf_recs):
        score = (len(cf_recs) - rank) * cf_weight
        hybrid_scores[movie] = hybrid_scores.get(movie, 0) + score
        
    # Sort movies by their new combined hybrid score
    sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Extract just the titles for the top 10
    top_hybrid_movies = [movie for movie, score in sorted_hybrid][:10]
    
    return top_hybrid_movies

# Get top popular movies (based on average rating)
def get_popular_movies():
    popular = data.groupby("title")["rating"].mean().sort_values(ascending=False).head(10)
    return popular

# Get movie details (genre + rating)
# Load links at the top of recommender.py


def get_movie_details(movie):
    movie_info = movies[movies["title"] == movie]
    if movie_info.empty:
        return {"genres": "N/A", "rating": "N/A", "tmdbId": None}

    # Get the movieId to find the tmdbId
    movie_id = movie_info["movieId"].values[0]
    tmdb_id = links[links["movieId"] == movie_id]["tmdbId"].values[0]
    
    avg_rating = data[data["title"] == movie]["rating"].mean()

    return {
        "genres": movie_info["genres"].values[0].replace("|", ", "),
        "rating": round(avg_rating, 2) if not pd.isna(avg_rating) else "N/A",
        "tmdbId": tmdb_id # Add this!
    }
