import streamlit as st
import pandas as pd
from recommender import (
    recommend_movies,
    recommend_movies_cb,
    recommend_movies_hybrid,
    movies,
    get_popular_movies,
    get_movie_details,
    global_browse_movies,
    data
)
from poster import fetch_poster

# 1. Page Config must be FIRST
st.set_page_config(page_title="Emotion Movie Recommender", layout="wide")

# 2. CSS for FIXED SIZE CARDS - Using reliable class names
st.markdown("""
    <style>
    /* Fixed size for each movie card */
    .movie-card {
        height: 450px !important;
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        background-color: #262730;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #444;
        overflow: hidden !important;
    }
    
    /* Fixed size for poster image */
    .movie-poster {
        height: 280px !important;
        width: 100% !important;
        object-fit: cover !important;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    
    /* Fixed height for title (2 lines max) */
    .movie-title {
        font-weight: bold;
        font-size: 0.85rem;
        height: 40px !important;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        text-align: center;
        margin-bottom: 5px;
        line-height: 1.3;
    }
    
    /* Fixed height for rating/genre */
    .movie-info {
        font-size: 0.7rem;
        text-align: center;
        height: 50px !important;
        overflow: hidden;
        color: #aaa;
    }
    
    /* Force all columns to have same height parent */
    .stColumn {
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Make containers inside columns stretch properly */
    .stColumn > div {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Emotion-Based Movie Recommender")

# Helper function to display a movie card (ensures same size every time)
def display_movie_card(poster_url, title, rating, genres):
    """Display a single movie card with fixed dimensions"""
    # Truncate genres to prevent overflow
    if len(genres) > 40:
        genres = genres[:37] + "..."
    
    card_html = f"""
    <div class="movie-card">
        <img class="movie-poster" src="{poster_url}" onerror="this.src='https://via.placeholder.com/500x750?text=No+Poster'">
        <div class="movie-title">{title}</div>
        <div class="movie-info">⭐ {rating} | 🎭 {genres}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["✨ Discover", "📂 Browse Library", "🔥 Trending"])

# ============================================
# TAB 1: DISCOVER
# ============================================
with tab1:
    st.subheader("Personalized for your Mood")
    
    col_mood, col_algo, col_movie = st.columns([2, 2, 4])

    with col_mood:
        emotion = st.selectbox("Select your mood", ["Happy", "Sad", "Stressed", "Excited", "Romantic"], key="tab1_mood")
        
    with col_algo:
        algorithm = st.selectbox("Algorithm", ["Collaborative Filtering", "Content-Based", "Hybrid"], key="tab1_algo")

    with col_movie:
        movie_title = st.selectbox("Choose a movie you like:", movies["title"].values, key="tab1_movie_select")
    
    num_recs = st.slider("Number of Recommendations", 5, 50, 10, key="tab1_res")

    if st.button("Generate Recommendations", type="primary"):
        with st.spinner('AI is thinking...'):
            try:
                if algorithm == "Collaborative Filtering":
                    recs = recommend_movies(movie_title, emotion, top_n=num_recs)
                elif algorithm == "Content-Based":
                    recs = recommend_movies_cb(movie_title, emotion, top_n=num_recs)
                else:
                    recs = recommend_movies_hybrid(movie_title, emotion, top_n=num_recs)

                recs = recs[:num_recs]

                if recs:
                    st.subheader(f"Recommended for You ({len(recs)} movies)")
                    
                    # Display in rows of 5
                    for i in range(0, len(recs), 5):
                        cols = st.columns(5)
                        for j in range(5):
                            idx = i + j
                            if idx < len(recs):
                                movie = recs[idx]
                                details = get_movie_details(movie)
                                poster = fetch_poster(details['tmdbId'])
                                with cols[j]:
                                    display_movie_card(
                                        poster_url=poster,
                                        title=movie,
                                        rating=details['rating'],
                                        genres=details['genres'].replace('|', ', ')
                                    )
                else:
                    st.warning("No recommendations found.")
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================
# TAB 2: BROWSE LIBRARY
# ============================================
with tab2:
    st.header("🔍 Search & Filter Library")
    search = st.text_input("🔍 Search for a specific title in the catalog")
    
    c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 1.5, 1.5, 1.5, 1.5])

    with c1:
        raw_genres = movies["genres"].str.split('|').explode().unique()
        all_genres = sorted([g for g in raw_genres if pd.notna(g) and g != "" and g != "(no genres listed)"])
        selected_genres = st.multiselect("Genre Filter", all_genres, key="br_gen")

    with c2:
        valid_years = movies[movies['year'] > 0]['year']
        min_v, max_v = int(valid_years.min()), int(valid_years.max())
        year_range = st.slider("Year Range", min_v, max_v, (1990, max_v), key="br_yr")

    with c3:
        min_rating = st.slider("Min Rating", 0.0, 5.0, 0.0, 0.1, key="br_rate")

    with c4:
        sort_choice = st.selectbox("Sort By", ["Rating", "A-Z", "Year"], key="br_sort")

    with c5:
        sort_order = st.selectbox("Order", ["Descending", "Ascending"], key="br_order")

    with c6:
        num_results = st.number_input("Limit", min_value=1, max_value=100, value=10, key="br_limit")

    if st.button("Apply Library Filters", type="primary", use_container_width=True):
        with st.spinner('Filtering results...'):
            filtered_df = global_browse_movies(
                movies_df=movies,
                ratings_df=data,
                selected_genres=selected_genres,
                min_rating=min_rating,
                sort_by=sort_choice,
                sort_order=sort_order, 
                year_range=year_range
            )
            
            if search:
                filtered_df = filtered_df[filtered_df["title"].str.lower().str.contains(search.lower(), na=False)]

            display_list = filtered_df.head(num_results)
            
            if not display_list.empty:
                st.markdown(f"### Showing {len(display_list)} results")
                
                # Display in rows of 5
                movies_list = display_list.to_dict('records')
                for i in range(0, len(movies_list), 5):
                    cols = st.columns(5)
                    for j in range(5):
                        idx = i + j
                        if idx < len(movies_list):
                            row = movies_list[idx]
                            poster = fetch_poster(row.get('tmdbId', None))
                            display_movie_card(
                                poster_url=poster,
                                title=row['title'],
                                rating=round(row['rating'], 1),
                                genres=row['genres'].replace('|', ', ')
                            )
            else:
                st.warning("No movies found matching your search and filter criteria.")

# ============================================
# TAB 3: TRENDING
# ============================================
with tab3:
    st.subheader("Top Rated by the Community")
    popular_data = get_popular_movies()
    
    # Convert to list for iteration
    popular_list = list(popular_data.items())
    
    for i in range(0, len(popular_list), 5):
        cols = st.columns(5)
        for j in range(5):
            idx = i + j
            if idx < len(popular_list):
                title, rating = popular_list[idx]
                details = get_movie_details(title)
                poster = fetch_poster(details['tmdbId'])
                display_movie_card(
                    poster_url=poster,
                    title=title,
                    rating=round(rating, 2),
                    genres=details['genres'].replace('|', ', ')
                )
