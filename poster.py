import pandas as pd  # <--- THIS WAS MISSING
import requests
import streamlit as st

@st.cache_data
def fetch_poster(tmdb_id):
    # Check if the ID is empty or NaN (Not a Number)
    if not tmdb_id or pd.isna(tmdb_id):
        return "https://via.placeholder.com/500x750?text=No+ID+Found"

    api_key = "9c1c57172e2bf025273e0382a1fc6768"
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
                
    except Exception as e:
        print(f"Error for TMDB ID {tmdb_id}: {e}")
    
    return "https://via.placeholder.com/500x750?text=No+Poster+Found"
