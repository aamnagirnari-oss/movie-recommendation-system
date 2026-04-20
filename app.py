import streamlit as st
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# ---------------- LOAD DATA ----------------
# Try data/ folder first (local), else root (Streamlit)
if os.path.exists("data/ratings.csv"):
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")
else:
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")

data = pd.merge(ratings, movies, on="movieId")

# ---------------- MATRIX ----------------
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# ---------------- SVD ----------------
svd = TruncatedSVD(n_components=20)
latent_matrix = svd.fit_transform(user_movie_matrix)

# ---------------- USER SIMILARITY ----------------
user_similarity = cosine_similarity(latent_matrix)
user_similarity_df = pd.DataFrame(user_similarity,
                                 index=user_movie_matrix.index,
                                 columns=user_movie_matrix.index)

# ---------------- MOVIE SIMILARITY ----------------
movie_matrix = user_movie_matrix.T
movie_similarity = cosine_similarity(movie_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity,
                                  index=movie_matrix.index,
                                  columns=movie_matrix.index)

# ---------------- USER RECOMMEND ----------------
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

# ---------------- MOVIE RECOMMEND ----------------
def recommend_by_movie(movie_name, top_n=10):
    if movie_name not in movie_similarity_df.index:
        return pd.DataFrame()

    similar = movie_similarity_df[movie_name].sort_values(ascending=False).drop(movie_name)

    counts = data.groupby('title')['rating'].count()

    df = pd.DataFrame({'similarity': similar})
    df['num_ratings'] = counts
    df = df[df['num_ratings'] >= 10]

    return df.sort_values('similarity', ascending=False).head(top_n)

# ---------------- HEADER ----------------
st.title("🎬 Movie Recommendation System")
st.caption("Powered by SVD + Collaborative Filtering")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎛️ Controls")
mode = st.sidebar.radio("Choose Mode", ["User-Based", "Movie-Based"])

# ---------------- USER MODE ----------------
if mode == "User-Based":
    st.subheader("👤 Personalized Recommendations")

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

# ---------------- MOVIE MODE ----------------
else:
    st.subheader("🎥 Find Similar Movies")

    movie_list = sorted(data['title'].unique())
    movie_name = st.selectbox("Select a movie", movie_list)

    if st.button("🎯 Find Similar"):
        with st.spinner("Finding similar movies..."):
            time.sleep(1)
            results = recommend_by_movie(movie_name)

        if results.empty:
            st.warning("No similar movies found.")
        else:
            st.success("You may also like 🍿")

            cols = st.columns(2)

            for i, (movie, row) in enumerate(results.iterrows()):
                with cols[i % 2]:
                    st.markdown(f"### 🎬 {movie}")
                    st.write(f"🔗 Similarity: {row['similarity']:.2f}")
                    st.write(f"📊 Ratings: {int(row['num_ratings'])}")
                    st.markdown("---")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Python & Machine Learning")
