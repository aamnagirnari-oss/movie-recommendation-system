import streamlit as st
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# ---------------- LOAD DATA (CACHED) ----------------
@st.cache_data
def load_data():
    if os.path.exists("data/ratings.csv"):
        ratings = pd.read_csv("data/ratings.csv")
        movies = pd.read_csv("data/movies.csv")
    else:
        ratings = pd.read_csv("ratings.csv")
        movies = pd.read_csv("movies.csv")

    return pd.merge(ratings, movies, on="movieId")

data = load_data()

# ---------------- CREATE MATRIX ----------------
@st.cache_data
def create_matrix(data):
    return data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

user_movie_matrix = create_matrix(data)

# ---------------- SVD + SIMILARITY ----------------
@st.cache_data
def compute_similarity(matrix):
    svd = TruncatedSVD(n_components=10)  # reduced for cloud
    latent_matrix = svd.fit_transform(matrix)

    similarity = cosine_similarity(latent_matrix)
    return pd.DataFrame(similarity,
                        index=matrix.index,
                        columns=matrix.index)

user_similarity_df = compute_similarity(user_movie_matrix)

# ---------------- RECOMMEND FUNCTION ----------------
def recommend_for_user(user_id, top_n=10, min_ratings=10):
    if user_id not in user_similarity_df.index:
        return pd.DataFrame()

    similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id)
    top_users = similar_users.head(20).index

    rec = data[data['userId'].isin(top_users)]

    watched = data[data['userId'] == user_id]['title']
    rec = rec[~rec['title'].isin(watched)]

    agg = rec.groupby('title').agg(
        mean_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    )

    agg = agg[agg['num_ratings'] >= min_ratings]

    C = data['rating'].mean()
    m = min_ratings

    agg['score'] = (agg['num_ratings']/(agg['num_ratings']+m))*agg['mean_rating'] + \
                   (m/(agg['num_ratings']+m))*C

    return agg.sort_values('score', ascending=False).head(top_n)

# ---------------- UI ----------------
st.title("🎬 Movie Recommendation System")
st.caption("Powered by SVD + Collaborative Filtering")

st.sidebar.title("🎛️ Controls")

# ONLY USER MODE (stable for cloud)
user_id = st.number_input("Enter User ID", min_value=1, step=1)

if st.button("🔥 Recommend"):
    with st.spinner("Analyzing preferences..."):
        time.sleep(1)
        results = recommend_for_user(user_id, min_ratings=5)

    if results.empty:
        st.warning("No recommendations found.")
    else:
        st.success("Top Picks for You 🎯")

        cols = st.columns(2)

        for i, (movie, row) in enumerate(results.iterrows()):
            with cols[i % 2]:
                st.markdown(f"### 🎬 {movie}")
                st.write(f"⭐ Rating: {row['mean_rating']:.2f}")
                st.write(f"📊 Votes: {int(row['num_ratings'])}")
                st.write(f"🔥 Score: {row['score']:.2f}")
                st.markdown("---")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Python & Machine Learning")
