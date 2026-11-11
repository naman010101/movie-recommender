# Movie Recommender System

A content-based movie recommender built with Streamlit. Pick a movie and get the 5 most similar titles with posters.

## How it works
- Dataset: [TMDB 5000 Movies & Credits](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
- Each movie's overview, genres, keywords, cast and crew are combined into a stemmed "tags" text.
- Tags are turned into bag-of-words vectors (5000 features) and compared with cosine similarity.
- `neighbors.npy` stores each movie's 10 nearest neighbours, so the app doesn't need the 185 MB similarity matrix.
- Posters are fetched live from the TMDB API.

## Run locally
```bash
pip install -r requirements.txt
export TMDB_API_KEY=your_key_here     # Windows PowerShell: $env:TMDB_API_KEY="your_key_here"
streamlit run app.py
```
Get a free key at https://www.themoviedb.org/settings/api. Without a key the app still works and shows placeholder posters.
On Streamlit Cloud, add `TMDB_API_KEY` under *Secrets*.

## Rebuild the neighbours file
```bash
pip install scikit-learn
python build_neighbors.py
```

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `movie_dict.pkl` / `movies.pkl` | Movie id, title and tags |
| `neighbors.npy` | Precomputed top-10 similar movies |
| `build_neighbors.py` | Regenerates `neighbors.npy` |
| `Procfile`, `setup.sh` | Heroku-style deployment |
