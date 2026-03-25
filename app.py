import streamlit as st
import re
from recommender import (
    recommend_movies,
    recommend_movies_cb,
    recommend_movies_hybrid,
    movies,
    get_popular_movies,
    get_movie_details
)
from poster import fetch_poster

# 1. Page Config must be FIRST
st.set_page_config(page_title="Emotion Movie Recommender", layout="wide")

# 2. CSS for UI Polish
st.markdown("""
    <style>
    /* Target the container wrapper to ensure uniform card height */
    [data-testid="stVerticalBlockBorderWrapper"] {
        height: 480px !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    /* Standardize image height */
    .stImage > img {
        border-radius: 8px;
        height: 280px !important;
        object-fit: cover;
    }

    /* Force movie titles to stay within 2 lines to prevent layout shifts */
    .movie-title {
        font-weight: bold;
        height: 3em; 
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Emotion-Based Movie Recommender")

# --- SIDEBAR ---
st.sidebar.header("Settings")
emotion = st.sidebar.selectbox("Select your mood", ["Happy", "Sad", "Stressed", "Excited", "Romantic"])
algorithm = st.sidebar.selectbox("Algorithm", ["Collaborative Filtering", "Content-Based", "Hybrid"])

# --- TABS ---
tab1, tab2 = st.tabs(["✨ Discover", "🔥 Trending"])

with tab1:
    search = st.text_input("🔍 Search for a movie you liked")
    
    # Search logic
    if search:
        filtered = movies[movies["title"].str.lower().str.contains(search.lower(), na=False)]
        movie_title = st.selectbox("Pick the movie:", filtered["title"].values) if not filtered.empty else st.selectbox("Choose from list:", movies["title"].values)
    else:
        movie_title = st.selectbox("Choose a movie you like:", movies["title"].values)

    if st.button("Generate Recommendations", type="primary"):
        with st.spinner('Thinking...'):
            try:
                # Get recommendations
                if algorithm == "Collaborative Filtering":
                    recs = recommend_movies(movie_title, emotion)
                elif algorithm == "Content-Based":
                    recs = recommend_movies_cb(movie_title, emotion)
                else:
                    recs = recommend_movies_hybrid(movie_title, emotion)

                if recs:
                    st.subheader("Recommended for You")
                    cols = st.columns(5)
                    for i, movie in enumerate(recs):
                        details = get_movie_details(movie)
                        poster = fetch_poster(details['tmdbId']) 
                        
                        with cols[i % 5]:
                            with st.container(border=True):
                                st.image(poster, use_container_width=True)
                                # Use the CSS class we defined above for uniform title height
                                st.markdown(f'<div class="movie-title">{movie}</div>', unsafe_allow_html=True)
                                st.caption(f"⭐ {details['rating']} | 🎭 {details['genres'].split(',')[0]}")
                else:
                    st.warning("No matches found.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

with tab2:
    st.subheader("Top Rated by the Community")
    popular_data = get_popular_movies()
    p_cols = st.columns(5)
    for i, title in enumerate(popular_data.index):
        details = get_movie_details(title)
        poster = fetch_poster(details['tmdbId'])
        with p_cols[i % 5]:
            # FIX: Removed the manual <div> tags that were creating placeholders
            with st.container(border=True):
                st.image(poster, use_container_width=True)
                st.markdown(f"**{title[:35]}**")
                st.caption(f"⭐ {popular_data[title]:.2f} avg")
