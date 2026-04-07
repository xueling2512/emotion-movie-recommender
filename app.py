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
    data,
    evaluate_recommendations, 
    calculate_rmse  
)
from poster import fetch_poster

# 1. Page Config must be FIRST
st.set_page_config(page_title="Emotion Movie Recommender", layout="wide")

# 2. CSS for COMPACT POSTERS - Fits 5 columns at 100% zoom
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
    # Aggressive truncation for small cards
    if len(genres) > 35:
        genres = genres[:32] + "..."
    if len(title) > 28:
        title = title[:25] + "..."
    
    # Handle missing/empty rating
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
tab1, tab2, tab3, tab4 = st.tabs(["✨ Discover", "📂 Browse Library", "🔥 Trending", "📊 Evaluation"])

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
                    
                    # Display in rows of 5 columns
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
                
                movies_list = display_list.to_dict('records')  # ✅ Now inside the if block
                # Display in rows of 5 columns
                for i in range(0, len(movies_list), 5):
                    cols = st.columns(5)
                    for j in range(5):
                        idx = i + j
                        if idx < len(movies_list):
                            row = movies_list[idx]
                            # Make sure tmdbId exists and is valid
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

    # Display in rows of 5 columns
    for i in range(0, len(popular_list), 5):
        cols = st.columns(5)
        for j in range(5):
            idx = i + j
            if idx < len(popular_list):
                title, rating = popular_list[idx]
                details = get_movie_details(title)
                poster = fetch_poster(details['tmdbId'])
                with cols[j]:  # ✅ IMPORTANT: This was missing!
                    display_movie_card(
                        poster_url=poster,
                        title=title,
                        rating=round(rating, 2),
                        genres=details['genres'].replace('|', ', ')
                    )

# ============================================
# TAB 4: EVALUATION
# ============================================
with tab4:
    st.header("📊 System Evaluation")
    
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
        
        # 🔗 REPLACE THIS URL WITH YOUR GOOGLE FORM LINK
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
    
    # Add divider
    st.divider()
    
    # ============================================
    # UNDERSTANDING USER ID & TOP-K SECTION
    # ============================================
    with st.expander("📚 Understanding User ID & Top-K Recommendations", expanded=False):
        st.markdown("""
        ### 👤 What is a User ID?
        
        User IDs come from your `ratings.csv` dataset. Each ID represents a real person who rated movies.
        
        **How to choose a good User ID:**
        - Users with **more ratings** (100+) give better evaluation results
        - Users with **varied ratings** (not all 5 stars) are more realistic
        - The system will automatically suggest the best users below
        
        ### 🎯 What is Top-K?
        
        **Top-K** = The number of movie recommendations returned
        
        | K Value | Best For | Example |
        |---------|----------|---------|
        | K=5-10 | High precision (accuracy) | "Show me my 5 best matches" |
        | K=10-15 | Balanced performance | "Show me 10 good recommendations" |
        | K=15-20 | High recall (coverage) | "Show me many options to choose from" |
        
        **Default K=10** is recommended for the best balance!
        """)
        
        # Auto-detect best users for testing
        if st.button("🔍 Find Best User IDs for Testing"):
            with st.spinner("Analyzing user data..."):
                # Get user statistics
                user_stats = data.groupby("userId").agg({
                    "rating": ["count", "mean"]
                }).reset_index()
                user_stats.columns = ["userId", "rating_count", "avg_rating"]
                user_stats = user_stats.sort_values("rating_count", ascending=False)
                
                st.success(f"✅ Found **{len(user_stats)}** users in the system")
                
                # Show top 10 users with most ratings
                st.write("**🏆 Top 10 Users with Most Ratings (Best for Testing):**")
                
                top_users = user_stats.head(10).copy()
                top_users["recommended"] = "✅ Yes"
                
                # Display as DataFrame
                st.dataframe(
                    top_users[["userId", "rating_count", "avg_rating", "recommended"]],
                    use_container_width=True,
                    column_config={
                        "userId": "User ID",
                        "rating_count": "Number of Ratings",
                        "avg_rating": "Average Rating",
                        "recommended": "Recommended for Testing?"
                    }
                )
                
                st.info("💡 **Tip:** Use User ID with the highest rating count (usually User 1 or 2) for most reliable evaluation results!")
    
    st.divider()
    
    # ============================================
    # EVALUATION CONTROLS
    # ============================================
    st.subheader("🔬 Run Algorithm Evaluation")
    
    # Get user range
    min_user = int(data["userId"].min())
    max_user = int(data["userId"].max())
    
    # Find recommended default user (user with most ratings)
    user_rating_counts = data.groupby("userId").size()
    default_user = user_rating_counts.idxmax() if not user_rating_counts.empty else 1
    
    col_test, col_k, col_info = st.columns([2, 2, 1])
    
    with col_test:
        test_user = st.number_input(
            "👤 Select User ID for testing", 
            min_value=min_user, 
            max_value=max_user, 
            value=int(default_user),
            key="eval_user",
            help="Choose a User ID from your dataset. Users with more ratings give better results!"
        )
        
        # Show selected user's stats
        user_ratings_count = len(data[data["userId"] == test_user])
        if user_ratings_count > 0:
            user_avg_rating = data[data["userId"] == test_user]["rating"].mean()
            st.caption(f"📊 This user has **{user_ratings_count}** ratings (avg: {user_avg_rating:.2f}⭐)")
            
            if user_ratings_count < 50:
                st.warning("⚠️ This user has few ratings. Results may not be reliable. Choose a user with 100+ ratings!")
            elif user_ratings_count > 200:
                st.success("✅ Great choice! This user has many ratings for reliable evaluation.")
        else:
            st.error(f"❌ User {test_user} not found in dataset!")
    
    with col_k:
        k_value = st.slider(
            "🎯 Top-K recommendations", 
            min_value=5, 
            max_value=20, 
            value=10,
            key="eval_k",
            help="Number of recommendations to evaluate. K=10 gives the best balance of precision and recall."
        )
        
        # Explain current K value
        if k_value <= 8:
            st.caption("⚡ **High precision mode** - Fewer but more accurate recommendations")
        elif k_value <= 15:
            st.caption("⚖️ **Balanced mode** - Best trade-off between accuracy and coverage")
        else:
            st.caption("🔍 **High recall mode** - Finds more relevant movies but may include less accurate ones")
    
    with col_info:
        st.metric(
            "📊 Evaluation Scope",
            f"Top-{k_value}",
            help=f"Testing how well algorithms perform when recommending {k_value} movies"
        )
    
    # ============================================
    # RUN EVALUATION BUTTON
    # ============================================
    if st.button("🚀 Run Evaluation", type="primary", key="eval_btn", use_container_width=True):
        
        # Check if user exists
        if len(data[data["userId"] == test_user]) == 0:
            st.error(f"❌ User {test_user} does not exist in the dataset! Please select a valid User ID between {min_user} and {max_user}.")
        else:
            with st.spinner(f"Calculating metrics for User {test_user} with Top-{k_value} recommendations..."):
                
                # Get results for all three algorithms
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
                
                if comparison_data:
                    # Display results table
                    st.subheader(f"📊 Algorithm Performance Comparison for User {test_user} (Top-{k_value})")
                    df_results = pd.DataFrame(comparison_data)
                    st.dataframe(df_results, use_container_width=True)
                    
                    # ============================================
                    # FIND AND HIGHLIGHT THE BEST ALGORITHM
                    # ============================================
                    
                    # Find best algorithm by F1 Score
                    best_algo_row = df_results.loc[df_results["F1 Score"].idxmax()]
                    best_algo_name = best_algo_row["Algorithm"]
                    best_f1 = best_algo_row["F1 Score"]
                    best_precision = best_algo_row["Precision"]
                    best_recall = best_algo_row["Recall"]
                    
                    # Display prominent winner message
                    st.success(f"🏆 **{best_algo_name} is the BEST performing algorithm for User {test_user} with Top-{k_value} recommendations!**")
                    
                    # Create metrics row for the winner
                    col_w1, col_w2, col_w3 = st.columns(3)
                    with col_w1:
                        st.metric("🎯 Precision", f"{best_precision:.4f}", help="Accuracy of recommendations")
                    with col_w2:
                        st.metric("📚 Recall", f"{best_recall:.4f}", help="Coverage of relevant movies")
                    with col_w3:
                        st.metric("⭐ F1 Score", f"{best_f1:.4f}", help="Balance of precision and recall (most important)")
                    
                    # Explanation of why this algorithm is best
                    with st.expander("📖 Why is this algorithm the best?", expanded=True):
                        st.markdown(f"""
                        ### 🎯 **{best_algo_name}** outperforms the other algorithms because:
                        
                        1. **Highest F1 Score ({best_f1:.4f})** - This is the most important metric as it balances:
                           - **Precision** ({best_precision:.4f}): {best_precision*100:.1f}% of recommended movies are relevant
                           - **Recall** ({best_recall:.4f}): Found {best_recall*100:.1f}% of all relevant movies
                        
                        2. **Comparison with other algorithms:**
                        """)
                        
                        # Show comparison table
                        for _, row in df_results.iterrows():
                            if row["Algorithm"] != best_algo_name:
                                diff_f1 = best_f1 - row["F1 Score"]
                                if diff_f1 > 0:
                                    st.markdown(f"- **{row['Algorithm']}** has **{diff_f1:.4f} lower F1 Score** than {best_algo_name}")
                                else:
                                    st.markdown(f"- **{row['Algorithm']}** performs similarly to {best_algo_name}")
                        
                        st.markdown(f"""
                        ### 💡 Recommendation:
                        **Use the *{best_algo_name}* algorithm** in the **Discover tab** for the best movie recommendations tailored to your mood!
                        
                        ---
                        **How Top-{k_value} affects this result:**
                        - With K={k_value}, we're evaluating {k_value} recommendations per algorithm
                        - This K value is {'optimized for balance' if 8 <= k_value <= 12 else 'focused on ' + ('precision' if k_value < 8 else 'recall')}
                        """)
                    
                    # Show runner-up comparison
                    if len(df_results) > 1:
                        with st.expander("🏅 Full Algorithm Ranking"):
                            # Sort by F1 Score descending
                            ranked_df = df_results.sort_values("F1 Score", ascending=False).reset_index(drop=True)
                            ranked_df.index = ranked_df.index + 1
                            ranked_df.columns = ["Algorithm", "Precision", "Recall", "F1 Score"]
                            
                            # Add medal emojis
                            medal_map = {1: "🥇 ", 2: "🥈 ", 3: "🥉 "}
                            ranked_df["Rank"] = [medal_map.get(i, f"{i}. ") for i in ranked_df.index]
                            ranked_df["Algorithm"] = ranked_df["Rank"] + ranked_df["Algorithm"]
                            
                            st.dataframe(ranked_df[["Algorithm", "Precision", "Recall", "F1 Score"]], use_container_width=True)
                            
                            st.markdown("""
                            **Understanding the ranking:**
                            - **🥇 1st Place**: Best overall performance
                            - Higher F1 Score = Better recommendation quality
                            - F1 Score combines both Precision and Recall
                            """)
                    
                    # Divider before RMSE
                    st.divider()
                    
                    # ============================================
                    # RMSE SECTION
                    # ============================================
                    st.subheader("📉 Rating Prediction Accuracy (RMSE)")
                    st.markdown("**What is RMSE?** Root Mean Square Error - measures how accurately the system predicts ratings. **Lower = Better**")
                    
                    rmse_result = calculate_rmse()
                    
                    col_rmse1, col_rmse2, col_rmse3 = st.columns(3)
                    with col_rmse1:
                        st.metric(
                            label="Root Mean Square Error (RMSE)", 
                            value=rmse_result["rmse"],
                            delta=None,
                            help="Lower is better. Measures how accurately the system predicts ratings."
                        )
                    with col_rmse2:
                        st.metric(
                            label="Global Average Rating",
                            value=round(rmse_result["global_avg_rating"], 2),
                            help="Average rating across all movies in the dataset"
                        )
                    with col_rmse3:
                        st.metric(
                            label="Total Predictions",
                            value=rmse_result["total_predictions"],
                            help="Number of ratings used for RMSE calculation"
                        )
                    
                    # RMSE interpretation
                    if rmse_result["rmse"] < 0.8:
                        st.success(f"✅ **Excellent!** RMSE of {rmse_result['rmse']} indicates very accurate rating predictions.")
                    elif rmse_result["rmse"] < 1.0:
                        st.info(f"👍 **Good!** RMSE of {rmse_result['rmse']} indicates reasonably accurate predictions.")
                    else:
                        st.warning(f"⚠️ RMSE of {rmse_result['rmse']} suggests room for improvement in rating predictions.")
                    
                    # ============================================
                    # FINAL CONCLUSION
                    # ============================================
                    st.divider()
                    
                    # Test different K values suggestion
                    with st.expander("🔬 Try Different K Values", expanded=False):
                        st.markdown(f"""
                        ### How K={k_value} compares to other values:
                        
                        You can rerun this evaluation with different K values to see how performance changes:
                        
                        | K Value | What it tests | Best for |
                        |---------|---------------|----------|
                        | K=5 | High precision | When you want only the very best matches |
                        | K=10 | Balanced | General use (recommended) |
                        | K=15 | Balanced recall | When you want more options |
                        | K=20 | High recall | When you want to discover many movies |
                        
                        **Try changing the Top-K slider above and run evaluation again!**
                        """)
                    
                    st.info(f"""
                    **📌 Final Conclusion for User {test_user} with Top-{k_value}:**
                    
                    Based on the evaluation results, the **{best_algo_name}** algorithm achieves the highest 
                    F1 Score (**{best_f1:.4f}**), making it the most effective recommendation strategy for this user.
                    
                    **Recommendation:** Use the **{best_algo_name}** algorithm in the Discover tab for the best 
                    personalized mood-based movie recommendations!
                    """)
                    
                else:
                    st.warning(f"No evaluation results could be generated for User {test_user}. Try a different User ID with more ratings.")
    
    # ============================================
    # ASSIGNMENT INSTRUCTIONS
    # ============================================
    with st.expander("📖 How to Complete the Evaluation for Your Assignment"):
        st.markdown("""
        ### Instructions for collecting user feedback:
        
        1. **Share your app** with 5-10 friends/classmates
        2. **Ask them to**:
           - Test all 3 algorithms (Collaborative, Content-Based, Hybrid)
           - Try different moods (Happy, Sad, Excited, etc.)
           - Try different K values (5, 10, 15, 20)
           - Click the "Take Survey" button above
           - Complete the Google Form honestly
        3. **Collect responses** from Google Forms:
           - Go to your Google Form
           - Click "Responses" tab
           - Click "Link to Sheets" (creates Excel file)
           - Export to CSV/Excel
        4. **Include in your documentation**:
           - Screenshot of the Google Form
           - Summary table of responses
           - Average satisfaction scores
           - Chart/Graph of results
        
        ### For the Algorithm Comparison:
        - The evaluation above automatically compares all three algorithms
        - **F1 Score** is the best metric for overall comparison
        - The system will highlight which algorithm performed best
        - Try different User IDs and K values to see how results change
        - Include these results in your assignment report
        
        ### Understanding Your Results:
        
        **Which algorithm is best?**
        - Look at the **F1 Score** column in the results table
        - Higher F1 Score = Better algorithm
        - The system will automatically highlight the winner
        
        **What if different users prefer different algorithms?**
        - That's normal! Different users have different tastes
        - Report the average across multiple users
        - The Hybrid algorithm often performs best overall
        
        **What K value should I use?**
        - K=10 is standard for most evaluations
        - Test with K=5, 10, 15, 20 to see patterns
        - Report results for K=10 in your main assignment
        """)
