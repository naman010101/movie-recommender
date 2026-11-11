import os

import numpy as np
import pandas as pd
import pickle
import requests
import streamlit as st

PLACEHOLDER_POSTER = "https://placehold.co/500x750?text=No+Poster"
N_RECOMMENDATIONS = 5


def get_api_key():
    """TMDB key from Streamlit secrets or the TMDB_API_KEY environment variable."""
    try:
        if "TMDB_API_KEY" in st.secrets:
            return st.secrets["TMDB_API_KEY"]
    except Exception:
        pass
    return os.getenv("TMDB_API_KEY", "")


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    api_key = get_api_key()
    if not api_key:
        return PLACEHOLDER_POSTER
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": api_key, "language": "en-US"},
            timeout=5,
        )
        poster_path = response.json().get("poster_path")
    except (requests.RequestException, ValueError):
        return PLACEHOLDER_POSTER
    return f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else PLACEHOLDER_POSTER


@st.cache_resource
def load_data():
    movies = pd.DataFrame(pickle.load(open("movie_dict.pkl", "rb")))
    # Top-10 most similar movies per movie (built by build_neighbors.py)
    neighbors = np.load("neighbors.npy")
    return movies, neighbors


def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    names, posters = [], []
    for i in neighbors[movie_index][:N_RECOMMENDATIONS]:
        row = movies.iloc[int(i)]
        names.append(row.title)
        posters.append(fetch_poster(row.movie_id))
    return names, posters


movies, neighbors = load_data()

st.title("Movie Recommender System")

selected_movie_name = st.selectbox(
    "How would you like to get movie recommendations?", movies["title"].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie_name)
    for col, name, poster in zip(st.columns(N_RECOMMENDATIONS), names, posters):
        with col:
            st.text(name)
            st.image(poster)
