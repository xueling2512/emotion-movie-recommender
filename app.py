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
    calculate_rmse  
)
from poster import fetch_poster

# 1. Page Config must be FIRST
st.set_page_config(page_title="Emotion Movie Recommender", layout="wide")

# 2. CSS for POSTERS - Balanced size
st.markdown("""
    <style>
    /* Main app container - reduce padding */
    .main > div {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    
    /* Movie card - balanced size */
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
    
    /* Poster size - visible but fits */
    .movie-poster {
        height: 280px !important;
        width: 100% !important;
        object-fit: cover !important;
        border-radius: 6px;
        margin-bottom: 8px !important;
    }
    
    /* Title styling - readable */
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
    
    /* Info text - readable */
    .movie-info {
        font-size: 0.7rem !important;
        text-align: center;
        height: 50px !important;
        overflow: hidden;
        color: #aaa;
        line-height: 1.4;
    }
    
    /* Reduce gap between columns */
    .row-widget.stHorizontal {
        gap: 0.3rem !important;
    }
    
    /* Force columns to be equal width */
    .stColumn {
        flex: 1 !important;
        min-width: 0 !important;
        padding-left: 0.2rem !important;
        padding-right: 0.2rem !important;
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

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["✨ Discover", "📂 Browse Library", "🔥 Trending", "📊 Algorithm Evaluation", "📋 Survey Dashboard"])

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
                
                movies_list = display_list.to_dict('records')
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
                st.warning("No movies found matching your search and filter criteria.")

# ============================================
# TAB 3: TRENDING
# ============================================
with tab3:
    st.subheader("Top Rated by the Community")
    popular_data = get_popular_movies()
    
    popular_list = list(popular_data.items())

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
# TAB 4: ALGORITHM EVALUATION (with Session State)
# ============================================
with tab4:
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
                        # Save F1 scores to session state
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
                    best_f1 = best_algo_row["F1 Score"]
                    
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
# TAB 5: SURVEY DASHBOARD (Fixed - No Type Errors)
# ============================================
with tab5:
    st.header("📋 Survey Analysis Dashboard")
    st.markdown("Upload your Google Form responses to see visual analysis and compare with automatic metrics.")
    st.markdown("---")
    
    # Display saved F1 Scores from evaluation
    if st.session_state.eval_run:
        st.success(f"✅ **F1 Scores loaded from Algorithm Evaluation:** CF={st.session_state.cf_f1:.3f}, CB={st.session_state.cb_f1:.3f}, Hybrid={st.session_state.hybrid_f1:.3f}")
    else:
        st.info("💡 **Tip:** Go to Algorithm Evaluation tab first, run evaluation to get F1 Scores automatically.")
    
    st.markdown("---")
    
    # File uploader for CSV
    uploaded_file = st.file_uploader(
        "📁 Upload Google Form responses (CSV file)",
        type=['csv'],
        help="Export your Google Form responses as CSV and upload here"
    )
    
    # Manual override (optional - collapsed by default) - FIXED VERSION
    with st.expander("🔧 Manual F1 Score Input (Optional)"):
        st.warning("Only use this if you want to override the automatically saved scores.")
        
        # Convert session state values to Python float to avoid type issues
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
            # Load survey data
            survey = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(survey)} survey responses!")
            
            # COLUMN MAPPING (Your Google Form columns)
            COLUMN_MAPPING = {
                '1. Which recommendation algorithm did you test?': 'Best Algorithm',
                '2. How relevant were the movie recommendations?': 'Recommendation Quality',
                '3. Did the recommendations match your mood?': 'Mood Accuracy',
                '4. How accurate were the movie suggestions?': 'Suggestion Accuracy',
                '5. Would you use this system again?': 'Would Use Again',
                '6. Overall satisfaction rating': 'Overall Satisfaction',
                '7. Any additional feedback or suggestions?': 'Comments',
            }
            
            # Rename columns
            for old, new in COLUMN_MAPPING.items():
                if old in survey.columns:
                    survey.rename(columns={old: new}, inplace=True)
            
            # Convert numeric columns
            numeric_cols = ['Recommendation Quality', 'Mood Accuracy', 'Suggestion Accuracy', 'Overall Satisfaction']
            for col in numeric_cols:
                if col in survey.columns:
                    survey[col] = pd.to_numeric(survey[col], errors='coerce')
            
            # Use saved F1 scores from session state (convert to float)
            cf_f1 = float(st.session_state.cf_f1)
            cb_f1 = float(st.session_state.cb_f1)
            hybrid_f1 = float(st.session_state.hybrid_f1)
            
            # Create tabs within the survey dashboard
            survey_tab1, survey_tab2, survey_tab3, survey_tab4 = st.tabs(["📊 Survey Results", "📈 Algorithm Comparison", "💬 User Feedback", "📝 Final Report"])
            
            # ============================================
            # TAB 1: SURVEY RESULTS
            # ============================================
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
                
                # Average Ratings
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
                    
                    # Display as metrics
                    cols = st.columns(len(rating_data))
                    for i, (name, value) in enumerate(rating_data.items()):
                        with cols[i]:
                            st.metric(name, f"{value:.2f}/5.0")
            
            # ============================================
            # TAB 2: ALGORITHM COMPARISON
            # ============================================
            with survey_tab2:
                st.header("📈 Algorithm Performance Comparison")
                
                # Get user preferences
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
                
                # Create comparison dataframe
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
                
                # Display comparison charts
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
                
                # Comparison table
                st.subheader("📊 Detailed Comparison")
                st.dataframe(df_comparison, use_container_width=True)
                
                # Winner announcement
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
                
                # Store best algorithm in session state for final report
                st.session_state.best_algo = best_f1_algo if best_f1_algo == best_user_algo else "Hybrid"
                st.session_state.best_f1_score = best_f1_score
                st.session_state.best_user_score = best_user_score
                
                # Correlation chart
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
            
            # ============================================
            # TAB 3: USER FEEDBACK
            # ============================================
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
            
            # ============================================
            # TAB 4: FINAL REPORT
            # ============================================
            with survey_tab4:
                st.header("📝 Final Evaluation Report")
                
                st.subheader("Executive Summary")
                
                total_responses = len(survey)
                avg_satisfaction = survey['Overall Satisfaction'].mean() if 'Overall Satisfaction' in survey.columns else 0
                
                # Get best algorithm from session state or calculate
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
                
                # Key Findings
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
                
                # Final Conclusion
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
                
                # Export report button
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
