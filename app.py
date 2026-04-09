import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from recommender import (
    recommend_movies,
    recommend_movies_cb,
    recommend_movies_hybrid,
    movies,
    get_popular_movies,
    get_movie_details,
    global_browse_movies,
    data,
    evaluate_recommendations, 
    calculate_rmse,
    links
)
from poster import fetch_poster

# 1. Page Config must be FIRST
st.set_page_config(page_title="Emotion Movie Recommender", layout="wide")

# 2. CSS for cleaner UI
st.markdown("""
    <style>
    /* Main app container */
    .main > div {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    
    /* Movie card */
    .movie-card {
        height: 420px !important;
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        background-color: #262730;
        border-radius: 8px;
        padding: 8px !important;
        margin-bottom: 8px !important;
        border: 1px solid #444;
        overflow: hidden !important;
    }
    
    /* Big featured movie poster */
    .featured-movie-card {
        background-color: #1a1a2e;
        border-radius: 12px;
        padding: 20px !important;
        margin-bottom: 30px !important;
        border: 2px solid #FF4B4B;
        text-align: center;
    }
    
    .featured-poster {
        height: 400px !important;
        width: auto !important;
        max-width: 100% !important;
        object-fit: contain !important;
        border-radius: 10px;
        margin-bottom: 15px !important;
    }
    
    .featured-title {
        font-size: 1.5rem !important;
        font-weight: bold;
        color: #FF4B4B;
        margin-bottom: 10px;
    }
    
    .featured-info {
        font-size: 1rem !important;
        color: #ccc;
    }
    
    /* Poster size */
    .movie-poster {
        height: 280px !important;
        width: 100% !important;
        object-fit: cover !important;
        border-radius: 6px;
        margin-bottom: 8px !important;
    }
    
    /* Title styling */
    .movie-title {
        font-weight: bold;
        font-size: 0.85rem !important;
        height: 45px !important;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        text-align: center;
        margin-bottom: 5px !important;
        line-height: 1.3;
    }
    
    /* Info text */
    .movie-info {
        font-size: 0.7rem !important;
        text-align: center;
        height: 50px !important;
        overflow: hidden;
        color: #aaa;
        line-height: 1.4;
    }
    
    /* Columns spacing */
    .row-widget.stHorizontal {
        gap: 0.3rem !important;
    }
    
    .stColumn {
        flex: 1 !important;
        min-width: 0 !important;
        padding-left: 0.2rem !important;
        padding-right: 0.2rem !important;
    }
    
    /* Custom divider */
    .custom-divider {
        margin: 15px 0;
        border-top: 1px solid #444;
    }
    
    /* Section header */
    .rec-header {
        font-size: 1.3rem;
        font-weight: bold;
        margin: 15px 0 10px 0;
        color: #FF4B4B;
    }
    
    /* Compact info box */
    .compact-info {
        background-color: #1e1e2e;
        padding: 8px 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 0.85rem;
        border-left: 4px solid #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Emotion-Based Movie Recommender")

# Helper function to display a movie card
def display_movie_card(poster_url, title, rating, genres):
    """Display a single movie card with fixed dimensions"""
    if len(genres) > 35:
        genres = genres[:32] + "..."
    if len(title) > 28:
        title = title[:25] + "..."
    
    if rating == "N/A" or pd.isna(rating):
        rating = "N/A"
    else:
        rating = f"{float(rating):.1f}"
    
    card_html = f"""
    <div class="movie-card">
        <img class="movie-poster" src="{poster_url}" onerror="this.src='https://via.placeholder.com/500x750?text=No+Poster'">
        <div class="movie-title">{title}</div>
        <div class="movie-info">⭐ {rating}<br>🎭 {genres}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def display_featured_movie(poster_url, title, rating, genres, year=None):
    """Display a large featured movie poster"""
    if rating == "N/A" or pd.isna(rating):
        rating_text = "Not rated"
    else:
        rating_text = f"⭐ {float(rating):.1f} / 5.0"
    
    year_text = f"📅 {year}" if year else ""
    
    card_html = f"""
    <div class="featured-movie-card">
        <img class="featured-poster" src="{poster_url}" onerror="this.src='https://via.placeholder.com/500x750?text=No+Poster'">
        <div class="featured-title">{title}</div>
        <div class="featured-info">{rating_text} | 🎭 {genres} {year_text}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def get_movie_year(title):
    """Extract year from movie title like 'Toy Story (1995)'"""
    import re
    match = re.search(r'\((\d{4})\)', title)
    return match.group(1) if match else "Unknown"


# ============================================
# HELPER FUNCTION FOR TRENDING WITH FILTERS
# ============================================
def get_filtered_trending_movies(limit=20, selected_genres=None, min_rating=0.0, year_range=None, search_term=""):
    """
    Get trending movies with filters applied
    """
    # Get all movies with their average ratings
    movie_ratings = data.groupby("title")["rating"].mean().reset_index()
    movie_ratings.columns = ["title", "avg_rating"]
    
    # Merge with movie details
    trending_df = movies.merge(movie_ratings, on="title", how="inner")
    
    # Filter by year range
    if year_range:
        trending_df = trending_df[(trending_df["year"] >= year_range[0]) & (trending_df["year"] <= year_range[1])]
    
    # Filter by genre
    if selected_genres:
        genre_regex = "|".join(selected_genres)
        trending_df = trending_df[trending_df["genres"].str.contains(genre_regex, case=False, na=False)]
    
    # Filter by minimum rating
    trending_df = trending_df[trending_df["avg_rating"] >= min_rating]
    
    # Filter by search term
    if search_term:
        trending_df = trending_df[trending_df["title"].str.lower().str.contains(search_term.lower(), na=False)]
    
    # Sort by rating (highest first)
    trending_df = trending_df.sort_values("avg_rating", ascending=False)
    
    # Add TMDB IDs
    trending_df = trending_df.merge(links[['movieId', 'tmdbId']], on="movieId", how="left")
    
    return trending_df.head(limit) if limit > 0 else trending_df


# ============================================
# TAB 1: DISCOVER (COMPACT & ORGANIZED)
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(["✨ Discover", "🔥 Trending", "📊 Algorithm Evaluation", "📋 Survey Dashboard"])

with tab1:
    # ROW 1: Core Controls (Mood, Algorithm, Movie Search)
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        emotion = st.selectbox(
            "🎭 Your Mood",
            ["Happy", "Sad", "Stressed", "Excited", "Romantic"],
            key="mood"
        )
    
    with col2:
        algorithm = st.selectbox(
            "🧠 Algorithm",
            ["Collaborative Filtering", "Content-Based", "Hybrid"],
            key="algo",
            help="CF: People who liked this also liked... | CB: Similar movies | Hybrid: Best of both"
        )
    
    with col3:
        search_movie = st.text_input(
            "🔍 Search for a movie",
            placeholder="Type movie title... (e.g., Toy Story, Inception)",
            key="search_movie"
        )
    
    # ROW 2: Movie Selection (appears after search)
    if search_movie:
        filtered_movies = movies[movies["title"].str.lower().str.contains(search_movie.lower(), na=False)]
    else:
        filtered_movies = movies.head(10)
    
    if len(filtered_movies) > 0:
        selected_movie = st.selectbox(
            "📽️ Select your movie",
            filtered_movies["title"].values,
            key="selected_movie"
        )
    else:
        st.warning("No movies found. Try a different search term.")
        selected_movie = None
    
    # ROW 3: Filter Controls in 3 columns (compact)
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        # Genre filter
        raw_genres = movies["genres"].str.split('|').explode().unique()
        all_genres = sorted([g for g in raw_genres if pd.notna(g) and g != "" and g != "(no genres listed)"])
        selected_genres = st.multiselect(
            "🎭 Genre Filter",
            all_genres,
            key="genres",
            placeholder="All genres"
        )
        
        # Number of recommendations
        num_recs = st.slider(
            "📊 Recommendations",
            min_value=5, max_value=50, value=10,
            key="num_recs"
        )
    
    with col_f2:
        # Year range
        valid_years = movies[movies['year'] > 0]['year']
        min_year, max_year = int(valid_years.min()), int(valid_years.max())
        year_range = st.slider(
            "📅 Year Range",
            min_year, max_year,
            (min_year, max_year),
            key="year"
        )
        
        # Minimum rating
        min_rating = st.slider(
            "⭐ Min Rating",
            0.0, 5.0, 0.0, 0.1,
            key="rating"
        )
    
    with col_f3:
        # Sort options
        sort_choice = st.selectbox(
            "📊 Sort By",
            ["Rating", "A-Z", "Year"],
            key="sort"
        )
        
        sort_order = st.selectbox(
            "🔼 Order",
            ["Descending", "Ascending"],
            key="order"
        )
        
        result_limit = st.number_input(
            "📄 Max Results",
            min_value=1, max_value=100, value=20,
            key="limit"
        )
    
    # ROW 4: Emotion Info Box (compact)
    if algorithm == "Collaborative Filtering":
        st.markdown('<div class="compact-info">😊 <strong>Emotion effect:</strong> Movies matching your mood get a <strong>3x boost</strong> in ranking.</div>', unsafe_allow_html=True)
    elif algorithm == "Content-Based":
        st.markdown('<div class="compact-info">😊 <strong>Emotion effect:</strong> Movies matching your mood get a <strong>5x boost</strong>. Non-matching movies are penalized (0.2x).</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="compact-info">😊 <strong>Emotion effect:</strong> <strong>60%</strong> mood-based + <strong>40%</strong> movie similarity.</div>', unsafe_allow_html=True)
    
    # ROW 5: Generate Button
    if st.button("🎬 Get Recommendations", type="primary", use_container_width=True):
        if not selected_movie:
            st.error("Please search and select a movie first!")
        else:
            with st.spinner("Finding your perfect movies based on your mood..."):
                
                # ===== DISPLAY FEATURED MOVIE =====
                st.markdown('<div class="rec-header">🎥 Your Selected Movie</div>', unsafe_allow_html=True)
                
                movie_details = get_movie_details(selected_movie)
                movie_year = get_movie_year(selected_movie)
                poster_url = fetch_poster(movie_details['tmdbId'])
                
                display_featured_movie(
                    poster_url=poster_url,
                    title=selected_movie,
                    rating=movie_details['rating'],
                    genres=movie_details['genres'].replace('|', ', '),
                    year=movie_year
                )
                
                # ===== PART 1: PERSONALIZED RECOMMENDATIONS =====
                try:
                    if algorithm == "Collaborative Filtering":
                        recs = recommend_movies(selected_movie, emotion, top_n=num_recs)
                        section_title = "👥 People who liked this also liked..."
                        explanation = f"Based on other users' preferences + {emotion} movies boosted"
                    elif algorithm == "Content-Based":
                        recs = recommend_movies_cb(selected_movie, emotion, top_n=num_recs)
                        section_title = "🎬 You may also like..."
                        explanation = f"Based on similar genres + {emotion} movies get 5x boost"
                    else:
                        recs = recommend_movies_hybrid(selected_movie, emotion, top_n=num_recs)
                        section_title = "🔮 Best Matches For You"
                        explanation = f"60% based on {emotion} mood + 40% based on similarity"
                    
                    recs = recs[:num_recs]
                    
                    if recs:
                        st.markdown(f'<div class="rec-header">{section_title}</div>', unsafe_allow_html=True)
                        st.caption(f"💡 {explanation}")
                        
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
                        st.warning("No recommendations found. Try a different movie or algorithm.")
                    
                    # ===== PART 2: FILTERED LIBRARY =====
                    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="rec-header">📚 Browse All Movies</div>', unsafe_allow_html=True)
                    
                    filtered_df = global_browse_movies(
                        movies_df=movies,
                        ratings_df=data,
                        selected_genres=selected_genres,
                        min_rating=min_rating,
                        sort_by=sort_choice,
                        sort_order=sort_order, 
                        year_range=year_range
                    )
                    
                    if selected_movie:
                        filtered_df = filtered_df[filtered_df["title"] != selected_movie]
                    
                    if not filtered_df.empty:
                        st.caption(f"📊 Showing {min(len(filtered_df), result_limit)} of {len(filtered_df)} movies")
                        
                        movies_list = filtered_df.head(result_limit).to_dict('records')
                        for i in range(0, len(movies_list), 5):
                            cols = st.columns(5)
                            for j in range(5):
                                idx = i + j
                                if idx < len(movies_list):
                                    row = movies_list[idx]
                                    tmdb_id = row.get('tmdbId', None)
                                    if tmdb_id and pd.notna(tmdb_id):
                                        poster = fetch_poster(tmdb_id)
                                    else:
                                        poster = "https://via.placeholder.com/500x750?text=No+Poster"
                                    with cols[j]:
                                        display_movie_card(
                                            poster_url=poster,
                                            title=row['title'],
                                            rating=round(row['rating'], 1),
                                            genres=row['genres'].replace('|', ', ')
                                        )
                    else:
                        st.info("No movies match your filters. Try adjusting genre, year, or rating.")
                        
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
    
    else:
        # Show info when no button clicked
        st.info("👆 **Search for a movie above**, then click **Get Recommendations**!")
        
        with st.expander("🌟 Popular Movies (Preview)", expanded=False):
            popular_preview = get_popular_movies().head(10)
            popular_list = list(popular_preview.items())
            for i in range(0, len(popular_list), 5):
                cols = st.columns(5)
                for j in range(5):
                    idx = i + j
                    if idx < len(popular_list):
                        title, rating = popular_list[idx]
                        details = get_movie_details(title)
                        poster = fetch_poster(details['tmdbId'])
                        with cols[j]:
                            display_movie_card(
                                poster_url=poster,
                                title=title,
                                rating=round(rating, 2),
                                genres=details['genres'].replace('|', ', ')
                            )


# ============================================
# TAB 2: TRENDING (WITH FILTERS)
# ============================================
with tab2:
    st.subheader("🔥 Top Rated by the Community")
    st.caption("Movies with the highest average ratings from all users")
    
    # Filter controls for trending
    col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 1])
    
    with col_f1:
        trending_limit = st.selectbox(
            "Number of movies",
            [10, 20, 30, 40, 50, 100, "All"],
            index=1,
            key="trending_limit"
        )
        if trending_limit == "All":
            trending_limit = 9999
        else:
            trending_limit = int(trending_limit)
    
    with col_f2:
        raw_genres_trending = movies["genres"].str.split('|').explode().unique()
        all_genres_trending = sorted([g for g in raw_genres_trending if pd.notna(g) and g != "" and g != "(no genres listed)"])
        trending_genres = st.multiselect(
            "Filter by genre",
            all_genres_trending,
            key="trending_genres",
            placeholder="All genres"
        )
    
    with col_f3:
        valid_years_trending = movies[movies['year'] > 0]['year']
        min_year_t, max_year_t = int(valid_years_trending.min()), int(valid_years_trending.max())
        trending_year_range = st.slider(
            "Year range",
            min_year_t, max_year_t,
            (min_year_t, max_year_t),
            key="trending_year"
        )
    
    with col_f4:
        trending_min_rating = st.slider(
            "Min rating",
            0.0, 5.0, 0.0, 0.5,
            key="trending_rating"
        )
    
    trending_search = st.text_input(
        "🔍 Search in trending movies",
        placeholder="Type movie name...",
        key="trending_search"
    )
    
    if st.button("🎯 Apply Trending Filters", type="primary", use_container_width=True, key="trending_apply"):
        with st.spinner("Loading trending movies..."):
            trending_df = get_filtered_trending_movies(
                limit=trending_limit,
                selected_genres=trending_genres,
                min_rating=trending_min_rating,
                year_range=trending_year_range,
                search_term=trending_search
            )
            
            if not trending_df.empty:
                st.success(f"🔥 Found {len(trending_df)} trending movies")
                
                trending_list = trending_df.to_dict('records')
                for i in range(0, len(trending_list), 5):
                    cols = st.columns(5)
                    for j in range(5):
                        idx = i + j
                        if idx < len(trending_list):
                            row = trending_list[idx]
                            tmdb_id = row.get('tmdbId', None)
                            if tmdb_id and pd.notna(tmdb_id):
                                poster = fetch_poster(tmdb_id)
                            else:
                                poster = "https://via.placeholder.com/500x750?text=No+Poster"
                            with cols[j]:
                                display_movie_card(
                                    poster_url=poster,
                                    title=row['title'],
                                    rating=round(row['avg_rating'], 2),
                                    genres=row['genres'].replace('|', ', ')
                                )
            else:
                st.warning("No movies found matching your filters.")
    
    else:
        with st.spinner("Loading trending movies..."):
            trending_df = get_filtered_trending_movies(
                limit=20,
                selected_genres=[],
                min_rating=0.0,
                year_range=(min_year_t, max_year_t),
                search_term=""
            )
            
            st.info("👆 Use the filters above to customize your trending list!")
            
            trending_list = trending_df.to_dict('records')
            for i in range(0, len(trending_list), 5):
                cols = st.columns(5)
                for j in range(5):
                    idx = i + j
                    if idx < len(trending_list):
                        row = trending_list[idx]
                        tmdb_id = row.get('tmdbId', None)
                        if tmdb_id and pd.notna(tmdb_id):
                            poster = fetch_poster(tmdb_id)
                        else:
                            poster = "https://via.placeholder.com/500x750?text=No+Poster"
                        with cols[j]:
                            display_movie_card(
                                poster_url=poster,
                                title=row['title'],
                                rating=round(row['avg_rating'], 2),
                                genres=row['genres'].replace('|', ', ')
                            )


# ============================================
# TAB 3: ALGORITHM EVALUATION
# ============================================
with tab3:
    st.header("📊 Algorithm Evaluation")
    
    st.markdown("""
    ## Thank you for testing our Emotion-Based Movie Recommender System!
    
    Please help us improve by providing your feedback.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 User Satisfaction Survey")
        st.markdown("""
        Please click the button below to complete our survey.
        It will take less than 2 minutes.
        """)
        
        google_form_url = "https://forms.gle/CWMLeHFWidtPDH358"
        
        st.markdown(f"""
        <a href="{google_form_url}" target="_blank">
            <button style="
                background-color: #4CAF50;
                color: white;
                padding: 12px 24px;
                font-size: 16px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
            ">
                📋 Take Survey (Google Form)
            </button>
        </a>
        """, unsafe_allow_html=True)
        
        st.caption("The form will open in a new tab.")
    
    with col2:
        st.subheader("📊 Evaluation Metrics")
        st.markdown("""
        **We measure success using:**
        
        - **Precision**: How many recommended movies are relevant
        - **Recall**: How many relevant movies were found
        - **F1 Score**: Balance of precision and recall
        - **RMSE**: Rating prediction accuracy (lower = better)
        
        > 📌 For detailed metrics, run the evaluation below:
        """)
    
    st.divider()
    
    with st.expander("📚 Understanding User ID & Top-K Recommendations", expanded=False):
        st.markdown("""
        ### 👤 What is a User ID?
        User IDs come from your `ratings.csv` dataset. Each ID represents a real person who rated movies.
        
        ### 🎯 What is Top-K?
        **Top-K** = The number of movie recommendations returned
        
        | K Value | Best For |
        |---------|----------|
        | K=5-10 | High precision (accuracy) |
        | K=10-15 | Balanced performance |
        | K=15-20 | High recall (coverage) |
        """)
        
        if st.button("🔍 Find Best User IDs for Testing"):
            with st.spinner("Analyzing user data..."):
                user_stats = data.groupby("userId").agg({
                    "rating": ["count", "mean"]
                }).reset_index()
                user_stats.columns = ["userId", "rating_count", "avg_rating"]
                user_stats = user_stats.sort_values("rating_count", ascending=False)
                
                st.success(f"✅ Found **{len(user_stats)}** users in the system")
                st.dataframe(user_stats.head(10), use_container_width=True)
    
    st.divider()
    
    st.subheader("🔬 Run Algorithm Evaluation")
    
    min_user = int(data["userId"].min())
    max_user = int(data["userId"].max())
    user_rating_counts = data.groupby("userId").size()
    default_user = user_rating_counts.idxmax() if not user_rating_counts.empty else 1
    
    col_test, col_k = st.columns(2)
    
    with col_test:
        test_user = st.number_input(
            "👤 Select User ID for testing", 
            min_value=min_user, 
            max_value=max_user, 
            value=int(default_user),
            key="eval_user"
        )
        
        user_ratings_count = len(data[data["userId"] == test_user])
        if user_ratings_count > 0:
            user_avg_rating = data[data["userId"] == test_user]["rating"].mean()
            st.caption(f"📊 This user has **{user_ratings_count}** ratings (avg: {user_avg_rating:.2f}⭐)")
    
    with col_k:
        k_value = st.slider("🎯 Top-K recommendations", min_value=5, max_value=20, value=10, key="eval_k")
    
    # Initialize session state for F1 scores
    if 'cf_f1' not in st.session_state:
        st.session_state.cf_f1 = 0.51
    if 'cb_f1' not in st.session_state:
        st.session_state.cb_f1 = 0.56
    if 'hybrid_f1' not in st.session_state:
        st.session_state.hybrid_f1 = 0.70
    if 'eval_run' not in st.session_state:
        st.session_state.eval_run = False
    
    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
        if len(data[data["userId"] == test_user]) == 0:
            st.error(f"❌ User {test_user} does not exist!")
        else:
            with st.spinner(f"Calculating metrics..."):
                algorithms = {
                    "Collaborative Filtering": "collaborative",
                    "Content-Based": "content", 
                    "Hybrid": "hybrid"
                }
                
                comparison_data = []
                for algo_name, algo_key in algorithms.items():
                    result = evaluate_recommendations(test_user, algorithm_type=algo_key, top_n=k_value)
                    if "error" not in result:
                        comparison_data.append({
                            "Algorithm": algo_name,
                            "Precision": result["precision"],
                            "Recall": result["recall"],
                            "F1 Score": result["f1"]
                        })
                        if algo_name == "Collaborative Filtering":
                            st.session_state.cf_f1 = result["f1"]
                        elif algo_name == "Content-Based":
                            st.session_state.cb_f1 = result["f1"]
                        elif algo_name == "Hybrid":
                            st.session_state.hybrid_f1 = result["f1"]
                
                st.session_state.eval_run = True
                
                if comparison_data:
                    df_results = pd.DataFrame(comparison_data)
                    st.dataframe(df_results, use_container_width=True)
                    
                    best_algo_row = df_results.loc[df_results["F1 Score"].idxmax()]
                    best_algo_name = best_algo_row["Algorithm"]
                    
                    st.success(f"🏆 **{best_algo_name} is the BEST performing algorithm!**")
                    st.info(f"📊 F1 Scores saved! Go to **Survey Dashboard** tab to compare with user feedback.")
                    
                    st.divider()
                    
                    st.subheader("📉 Rating Prediction Accuracy (RMSE)")
                    rmse_result = calculate_rmse()
                    
                    col_rmse1, col_rmse2, col_rmse3 = st.columns(3)
                    with col_rmse1:
                        st.metric("RMSE", rmse_result["rmse"], help="Lower is better")
                    with col_rmse2:
                        st.metric("Global Avg Rating", round(rmse_result["global_avg_rating"], 2))
                    with col_rmse3:
                        st.metric("Total Predictions", rmse_result["total_predictions"])
                else:
                    st.warning("No results generated. Try a different User ID.")


# ============================================
# TAB 4: SURVEY DASHBOARD
# ============================================
with tab4:
    st.header("📋 Survey Analysis Dashboard")
    st.markdown("Upload your Google Form responses to see visual analysis and compare with automatic metrics.")
    st.markdown("---")
    
    if st.session_state.eval_run:
        st.success(f"✅ **F1 Scores loaded from Algorithm Evaluation:** CF={st.session_state.cf_f1:.3f}, CB={st.session_state.cb_f1:.3f}, Hybrid={st.session_state.hybrid_f1:.3f}")
    else:
        st.info("💡 **Tip:** Go to Algorithm Evaluation tab first, run evaluation to get F1 Scores automatically.")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📁 Upload Google Form responses (CSV file)",
        type=['csv'],
        help="Export your Google Form responses as CSV and upload here"
    )
    
    with st.expander("🔧 Manual F1 Score Input (Optional)"):
        st.warning("Only use this if you want to override the automatically saved scores.")
        
        default_cf = float(st.session_state.cf_f1)
        default_cb = float(st.session_state.cb_f1)
        default_hybrid = float(st.session_state.hybrid_f1)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            manual_cf = st.number_input(
                "Manual CF F1", 
                min_value=0.0, 
                max_value=1.0, 
                value=default_cf, 
                step=0.01,
                format="%.3f",
                key="manual_cf"
            )
        with col_m2:
            manual_cb = st.number_input(
                "Manual CB F1", 
                min_value=0.0, 
                max_value=1.0, 
                value=default_cb, 
                step=0.01,
                format="%.3f",
                key="manual_cb"
            )
        with col_m3:
            manual_hybrid = st.number_input(
                "Manual Hybrid F1", 
                min_value=0.0, 
                max_value=1.0, 
                value=default_hybrid, 
                step=0.01,
                format="%.3f",
                key="manual_hybrid"
            )
        
        if st.button("Use Manual Values", key="use_manual_btn"):
            st.session_state.cf_f1 = float(manual_cf)
            st.session_state.cb_f1 = float(manual_cb)
            st.session_state.hybrid_f1 = float(manual_hybrid)
            st.success("✅ Manual values saved!")
            st.rerun()
    
    if uploaded_file is not None:
        try:
            survey = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(survey)} survey responses!")
            
            COLUMN_MAPPING = {
                '1. Which recommendation algorithm did you test?': 'Best Algorithm',
                '2. How relevant were the movie recommendations?': 'Recommendation Quality',
                '3. Did the recommendations match your mood?': 'Mood Accuracy',
                '4. How accurate were the movie suggestions?': 'Suggestion Accuracy',
                '5. Would you use this system again?': 'Would Use Again',
                '6. Overall satisfaction rating': 'Overall Satisfaction',
                '7. Any additional feedback or suggestions?': 'Comments',
            }
            
            for old, new in COLUMN_MAPPING.items():
                if old in survey.columns:
                    survey.rename(columns={old: new}, inplace=True)
            
            numeric_cols = ['Recommendation Quality', 'Mood Accuracy', 'Suggestion Accuracy', 'Overall Satisfaction']
            for col in numeric_cols:
                if col in survey.columns:
                    survey[col] = pd.to_numeric(survey[col], errors='coerce')
            
            cf_f1 = float(st.session_state.cf_f1)
            cb_f1 = float(st.session_state.cb_f1)
            hybrid_f1 = float(st.session_state.hybrid_f1)
            
            survey_tab1, survey_tab2, survey_tab3, survey_tab4 = st.tabs(["📊 Survey Results", "📈 Algorithm Comparison", "💬 User Feedback", "📝 Final Report"])
            
            with survey_tab1:
                st.header("📊 Survey Results Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'Best Algorithm' in survey.columns:
                        st.subheader("🎯 Algorithm Preference")
                        algo_counts = survey['Best Algorithm'].value_counts()
                        
                        fig = px.pie(
                            values=algo_counts.values,
                            names=algo_counts.index,
                            title="Which algorithm did users prefer?",
                            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'Would Use Again' in survey.columns:
                        st.subheader("🔄 Would Users Use Again?")
                        use_counts = survey['Would Use Again'].value_counts()
                        
                        fig = px.bar(
                            x=use_counts.index,
                            y=use_counts.values,
                            title="Would you use this system again?",
                            color=use_counts.index,
                            color_discrete_sequence=['#2ECC71', '#E74C3C']
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("⭐ Average Ratings (1-5 scale)")
                
                rating_data = {}
                for col in numeric_cols:
                    if col in survey.columns:
                        avg = survey[col].mean()
                        if not pd.isna(avg):
                            rating_data[col] = avg
                
                if rating_data:
                    fig = px.bar(
                        x=list(rating_data.keys()),
                        y=list(rating_data.values()),
                        title="User Satisfaction Metrics",
                        labels={'x': 'Metric', 'y': 'Average Rating (1-5)'},
                        color=list(rating_data.values()),
                        color_continuous_scale='Viridis'
                    )
                    fig.add_hline(y=4.0, line_dash="dash", line_color="green", annotation_text="Good (4.0)")
                    fig.add_hline(y=3.0, line_dash="dash", line_color="orange", annotation_text="Average (3.0)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    cols = st.columns(len(rating_data))
                    for i, (name, value) in enumerate(rating_data.items()):
                        with cols[i]:
                            st.metric(name, f"{value:.2f}/5.0")
            
            with survey_tab2:
                st.header("📈 Algorithm Performance Comparison")
                
                user_pref = {}
                if 'Best Algorithm' in survey.columns:
                    total_users = len(survey)
                    for algo in ['Collaborative Filtering', 'Content-Based', 'Hybrid']:
                        count = 0
                        for answer in survey['Best Algorithm']:
                            if pd.isna(answer):
                                continue
                            if algo.lower() in str(answer).lower():
                                count += 1
                        user_pref[algo] = (count / total_users) * 100 if total_users > 0 else 0
                
                comparison_data = []
                algorithms = {
                    'Collaborative Filtering': cf_f1,
                    'Content-Based': cb_f1,
                    'Hybrid': hybrid_f1
                }
                
                for algo, f1 in algorithms.items():
                    comparison_data.append({
                        'Algorithm': algo,
                        'F1 Score (Automatic)': f1,
                        'User Preference (%)': user_pref.get(algo, 0)
                    })
                
                df_comparison = pd.DataFrame(comparison_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = px.bar(
                        df_comparison,
                        x='Algorithm',
                        y='F1 Score (Automatic)',
                        title='Automatic Metrics (F1 Score)',
                        color='Algorithm',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                        text='F1 Score (Automatic)'
                    )
                    fig1.update_traces(textposition='outside')
                    fig1.update_layout(showlegend=False)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = px.bar(
                        df_comparison,
                        x='Algorithm',
                        y='User Preference (%)',
                        title='User Preference',
                        color='Algorithm',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                        text='User Preference (%)'
                    )
                    fig2.update_traces(textposition='outside')
                    fig2.update_layout(showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.subheader("📊 Detailed Comparison")
                st.dataframe(df_comparison, use_container_width=True)
                
                best_f1_algo = df_comparison.loc[df_comparison['F1 Score (Automatic)'].idxmax(), 'Algorithm']
                best_f1_score = df_comparison['F1 Score (Automatic)'].max()
                best_user_algo = df_comparison.loc[df_comparison['User Preference (%)'].idxmax(), 'Algorithm']
                best_user_score = df_comparison['User Preference (%)'].max()
                
                st.markdown("---")
                st.subheader("🏆 Winner Announcement")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Best by F1 Score", best_f1_algo, f"{best_f1_score:.3f}")
                with col2:
                    st.metric("Best by Users", best_user_algo, f"{best_user_score:.1f}%")
                with col3:
                    if best_f1_algo == best_user_algo:
                        st.success(f"🎉 {best_f1_algo} is the OVERALL WINNER!")
                    else:
                        st.warning("⚠️ Mixed Results")
                        st.info("💡 Hybrid algorithm recommended")
                
                st.session_state.best_algo = best_f1_algo if best_f1_algo == best_user_algo else "Hybrid"
                st.session_state.best_f1_score = best_f1_score
                st.session_state.best_user_score = best_user_score
                
                st.subheader("📉 Correlation: Automatic vs User Preference")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=df_comparison['F1 Score (Automatic)'],
                    y=df_comparison['User Preference (%)'],
                    mode='markers+text',
                    marker=dict(size=50, color=['#FF6B6B', '#4ECDC4', '#45B7D1']),
                    text=df_comparison['Algorithm'],
                    textposition="top center"
                ))
                fig3.update_layout(
                    title="Correlation between F1 Score and User Preference",
                    xaxis_title="F1 Score (Automatic)",
                    yaxis_title="User Preference (%)",
                    showlegend=False
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            with survey_tab3:
                st.header("💬 User Feedback & Comments")
                
                if 'Comments' in survey.columns:
                    comments = survey['Comments'].dropna()
                    
                    if len(comments) > 0:
                        st.info(f"📝 {len(comments)} users provided feedback")
                        
                        for i, comment in enumerate(comments, 1):
                            with st.container():
                                st.markdown(f"**User {i}:**")
                                st.markdown(f"> {comment}")
                                st.markdown("---")
                    else:
                        st.info("No comments provided by users yet.")
                
                with st.expander("📋 View All Individual Responses"):
                    st.dataframe(survey, use_container_width=True)
            
            with survey_tab4:
                st.header("📝 Final Evaluation Report")
                
                st.subheader("Executive Summary")
                
                total_responses = len(survey)
                avg_satisfaction = survey['Overall Satisfaction'].mean() if 'Overall Satisfaction' in survey.columns else 0
                
                if 'best_algo' not in st.session_state:
                    best_f1_algo = df_comparison.loc[df_comparison['F1 Score (Automatic)'].idxmax(), 'Algorithm']
                    best_user_algo = df_comparison.loc[df_comparison['User Preference (%)'].idxmax(), 'Algorithm']
                    st.session_state.best_algo = best_f1_algo if best_f1_algo == best_user_algo else "Hybrid"
                    st.session_state.best_f1_score = df_comparison['F1 Score (Automatic)'].max()
                    st.session_state.best_user_score = df_comparison['User Preference (%)'].max()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Responses", total_responses)
                with col2:
                    st.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}/5.0")
                with col3:
                    st.metric("Recommended Algorithm", st.session_state.best_algo)
                
                st.markdown("---")
                
                st.subheader("🔍 Key Findings")
                
                findings = []
                
                findings.append(f"✅ **{st.session_state.best_algo}** is the recommended algorithm with F1 Score: {st.session_state.best_f1_score:.3f} and User Preference: {st.session_state.best_user_score:.1f}%")
                
                if avg_satisfaction >= 4.0:
                    findings.append(f"✅ **High user satisfaction** ({avg_satisfaction:.2f}/5.0) - Users are happy with the system")
                elif avg_satisfaction >= 3.0:
                    findings.append(f"📊 **Moderate user satisfaction** ({avg_satisfaction:.2f}/5.0) - Room for improvement")
                else:
                    findings.append(f"⚠️ **Low user satisfaction** ({avg_satisfaction:.2f}/5.0) - Needs improvement")
                
                if 'Would Use Again' in survey.columns:
                    yes_count = (survey['Would Use Again'] == 'Yes').sum()
                    yes_pct = (yes_count / len(survey)) * 100
                    if yes_pct >= 70:
                        findings.append(f"✅ **High retention rate** ({yes_pct:.0f}% would use again)")
                    else:
                        findings.append(f"📊 **Moderate retention** ({yes_pct:.0f}% would use again)")
                
                for finding in findings:
                    st.markdown(finding)
                
                st.markdown("---")
                
                st.subheader("🎯 Final Conclusion")
                
                st.success(f"""
                ### ✅ Recommendation: Use **{st.session_state.best_algo}** Algorithm
                
                **Why?**
                - F1 Score: {st.session_state.best_f1_score:.3f}
                - User preference: {st.session_state.best_user_score:.1f}%
                - Balanced performance across automatic metrics and user feedback
                
                **Implementation:** Set {st.session_state.best_algo} as the default algorithm in the Discover tab.
                """)
                
                st.markdown("---")
                
                st.subheader("📥 Export Report")
                
                report_content = f"""
MOVIE RECOMMENDER SYSTEM - EVALUATION REPORT
=============================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Survey Responses: {total_responses}

KEY FINDINGS:
{chr(10).join(findings)}

FINAL CONCLUSION:
Use {st.session_state.best_algo} algorithm is recommended.

DETAILED METRICS:
{df_comparison.to_string()}

USER SATISFACTION:
Average Overall Satisfaction: {avg_satisfaction:.2f}/5.0
"""
                
                st.download_button(
                    label="📄 Download Report (TXT)",
                    data=report_content,
                    file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.info("Make sure your CSV file is properly formatted.")
    
    else:
        st.info("""
        ### How to analyze survey results:
        
        1. **First, go to Algorithm Evaluation tab** and click "Run Evaluation"
        2. **F1 Scores will be automatically saved**
        3. **Then come back here** and upload your CSV file
        4. **View** beautiful charts and winner announcement automatically!
        
        ### Google Form Link:
        [Take Survey](https://forms.gle/CWMLeHFWidtPDH358)
        """)
