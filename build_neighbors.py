"""Rebuild neighbors.npy from movie_dict.pkl (needs scikit-learn).

Mirrors the notebook: bag-of-words over the pre-stemmed `tags` column (5000 features,
English stop words), cosine similarity, then keep each movie's 10 nearest neighbours.
Run:  pip install scikit-learn && python build_neighbors.py
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame(pickle.load(open("movie_dict.pkl", "rb")))
vectors = CountVectorizer(max_features=5000, stop_words="english").fit_transform(movies["tags"])
similarity = cosine_similarity(vectors)
neighbors = np.argsort(-similarity, axis=1, kind="stable")[:, 1:11].astype(np.int16)
np.save("neighbors.npy", neighbors)
print("saved neighbors.npy", neighbors.shape)
