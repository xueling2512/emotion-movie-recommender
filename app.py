import streamlit as st
import pandas as pd
from recommender import (
    recommend_movies,
    recommend_movies_cb,
    recommend_movies_hybrid,
    movies,
    get_popular_movies,
    get_movie_details,
    global_browse_movies, # Use this for independent filtering
    data
)
from poster import fetch_poster

# 1. Page Config must be FIRST
st.set_page_config(page_title="Emotion Movie Recommender", layout="wide")

# 2. CSS for UI Polish
st.markdown("""
    <style>
    [data-testid="stVerticalBlockBorderWrapper"] {
        height: 480px !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .stImage > img {
        border-radius: 8px;
        height: 280px !important;
        object-fit: cover;
    }
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

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["✨ Discover", "📂 Browse Library", "🔥 Trending"])


# --- TAB 1: DISCOVER (AI RECOMMENDATIONS) ---
with tab1:
    st.subheader("Personalized for your Mood")
# Create 3 columns for a single row layout
    # You can adjust the numbers (2, 2, 4) to change the width of each box
    col_mood, col_algo, col_movie = st.columns([2, 2, 4]) 

    with col_mood:
        emotion = st.selectbox("Select your mood", ["Happy", "Sad", "Stressed", "Excited", "Romantic"], key="tab1_mood")
        
    with col_algo:
        algorithm = st.selectbox("Algorithm", ["Collaborative Filtering", "Content-Based", "Hybrid"], key="tab1_algo")

    with col_movie:
        movie_title = st.selectbox("Choose a movie you like:", movies["title"].values, key="tab1_movie_select")
    
    # 🆕 Add a specific slider for recommendations
    num_recs = st.slider("Number of Recommendations", 5, 50, 10, key="tab1_res")

    if st.button("Generate Recommendations", type="primary"):
        with st.spinner('AI is thinking...'):
            try:
                # Get raw AI recommendations
                if algorithm == "Collaborative Filtering":
                    recs = recommend_movies(movie_title, emotion)
                elif algorithm == "Content-Based":
                    recs = recommend_movies_cb(movie_title, emotion)
                else:
                    recs = recommend_movies_hybrid(movie_title, emotion)

                recs = recs[:num_recs]

                if recs:
                    st.subheader("Recommended for You")
                    cols = st.columns(5)
                    for i, movie in enumerate(recs):
                        details = get_movie_details(movie)
                        poster = fetch_poster(details['tmdbId']) 
                        with cols[i % 5]:
                            with st.container(border=True):
                                st.image(poster, use_container_width=True)
                                st.markdown(f'<div class="movie-title">{movie}</div>', unsafe_allow_html=True)
                                st.caption(f"⭐ {details['rating']} | 🎭 {details['genres']}")
                else:
                    st.warning("No recommendations found.")
            except Exception as e:
                st.error(f"Error: {e}")

# --- TAB 2: BROWSE ---
with tab2:
    st.header("🔍 Search & Filter Library")
    search = st.text_input("🔍 Search for a specific title in the catalog")
    
    # Create 6 columns to fit the new sorting options
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
        # Added "Year" to the sorting options
        sort_choice = st.selectbox("Sort By", ["Rating", "A-Z", "Year"], key="br_sort")

    with c5:
        # NEW: Sort Direction
        sort_order = st.selectbox("Order", ["Descending", "Ascending"], key="br_order")

    with c6:
        num_results = st.number_input("Limit", min_value=1, max_value=100, value=10, key="br_limit")

# 3. Apply Button
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
            
            # Apply text search filter if user typed something
            if search:
                filtered_df = filtered_df[filtered_df["title"].str.lower().str.contains(search.lower(), na=False)]

            display_list = filtered_df.head(num_results)
            
            if not display_list.empty:
                st.markdown(f"### Showing {len(display_list)} results")
                cols = st.columns(5)
                for i, row in enumerate(display_list.itertuples()):
                    details = get_movie_details(row.title)
                    poster = fetch_poster(details['tmdbId'])
                    with cols[i % 5]:
                        with st.container(border=True):
                            st.image(poster, use_container_width=True)
                            st.markdown(f'<div class="movie-title">{row.title}</div>', unsafe_allow_html=True)
                            st.caption(f"⭐ {row.rating:.1f} | 🎭 {row.genres.replace('|', ' ')}")
            else:
                st.warning("No movies found matching your search and filter criteria.")

# --- TAB 3: TRENDING ---
with tab3:
    st.subheader("Top Rated by the Community")
    popular_data = get_popular_movies()
    p_cols = st.columns(5)
    for i, title in enumerate(popular_data.index):
        details = get_movie_details(title)
        poster = fetch_poster(details['tmdbId'])
        with p_cols[i % 5]:
            with st.container(border=True):
                st.image(poster, use_container_width=True)
                st.markdown(f"**{title[:35]}**")
                st.caption(f"⭐ {popular_data[title]:.2f} avg")
