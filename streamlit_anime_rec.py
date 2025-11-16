import streamlit as st
import pandas as pd
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_data():
    """Load all necessary data for recommendations"""
    try:
        # Load anime data
        animes = pd.read_csv(r"C:\Users\gouab\Downloads\animes.csv")
        
        # Load KMeans model
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        
        # Load user-item matrix and mappings
        with open('user_item_matrix.pkl', 'rb') as f:
            data = pickle.load(f)
            user_item_matrix_sparse = data['matrix']
            user_to_idx = data['user_to_idx']
            anime_to_idx = data['anime_to_idx']
            unique_anime = data['unique_anime']
            user_cluster_mapping = data['user_cluster_mapping']
        
        # Create name to ID mapping
        name_to_id = pd.Series(animes.animeID.values, index=animes.title).to_dict()
        
        return animes, kmeans, user_item_matrix_sparse, user_to_idx, anime_to_idx, unique_anime, user_cluster_mapping, name_to_id
    
    except FileNotFoundError as e:
        st.error(f"Error loading data files: {e}")
        st.stop()

animes, kmeans, user_item_matrix_sparse, user_to_idx, anime_to_idx, unique_anime, user_cluster_mapping, name_to_id = load_data()

def get_cluster_based_recommendations(watched_anime_ids, user_ratings_dict, n_recommendations=10):
    """Generate collaborative filtering recommendations"""
    row_indices = []
    col_indices = []
    data = []
    
    for anime_id, rating in user_ratings_dict.items():
        if anime_id in anime_to_idx:
            anime_idx = anime_to_idx[anime_id]
            row_indices.append(0)
            col_indices.append(anime_idx)
            data.append(rating)
    
    if not data:
        return pd.DataFrame(), None
    
    new_user_vector = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(1, len(unique_anime))
    )
    
    cluster_id = kmeans.predict(new_user_vector)[0]
    users_in_cluster = [user_id for user_id, cluster in user_cluster_mapping.items() if cluster == cluster_id]
    
    if not users_in_cluster:
        return pd.DataFrame(), cluster_id
    
    cluster_user_indices = [user_to_idx[user_id] for user_id in users_in_cluster]
    cluster_user_matrix = user_item_matrix_sparse[cluster_user_indices]
    
    similarities = cosine_similarity(new_user_vector, cluster_user_matrix).flatten()
    top_n_users = min(50, len(similarities))
    top_similar_indices = similarities.argsort()[::-1][:top_n_users]
    top_similar_scores = similarities[top_similar_indices]
    
    similar_users_ratings = cluster_user_matrix[top_similar_indices]
    weighted_ratings = similar_users_ratings.T.dot(top_similar_scores)
    
    watched_indices = [anime_to_idx[anime_id] for anime_id in watched_anime_ids if anime_id in anime_to_idx]
    weighted_ratings[watched_indices] = -1
    
    top_anime_indices = weighted_ratings.argsort()[::-1][:n_recommendations]
    idx_to_anime = {idx: anime for anime, idx in anime_to_idx.items()}
    recommended_anime_ids = [idx_to_anime[idx] for idx in top_anime_indices]
    
    recommendations = animes[animes['animeID'].isin(recommended_anime_ids)][['animeID', 'title', 'genres', 'score']].copy()
    score_dict = {idx_to_anime[idx]: weighted_ratings[idx] for idx in top_anime_indices}
    recommendations['recommendation_score'] = recommendations['animeID'].map(score_dict)
    recommendations = recommendations.sort_values('recommendation_score', ascending=False)
    
    return recommendations, cluster_id

# App Title
st.title("🎬 Anime Recommendation System")
st.markdown("---")

# Sidebar for user input
st.sidebar.header("Rate Your Anime")
st.sidebar.markdown("Search and rate anime you've watched to get personalized recommendations!")

# Initialize session state for user ratings
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

# Search and add anime
anime_names = sorted(animes['title'].tolist())
selected_anime = st.sidebar.selectbox(
    "Search for an anime:",
    [""] + anime_names,
    index=0
)

if selected_anime:
    rating = st.sidebar.slider(
        f"Rate '{selected_anime}':",
        min_value=1,
        max_value=10,
        value=7,
        step=1
    )
    
    if st.sidebar.button("Add Rating"):
        if selected_anime in name_to_id:
            st.session_state.user_ratings[selected_anime] = rating
            st.sidebar.success(f"Added: {selected_anime} - Rating: {rating}")
        else:
            st.sidebar.error("Anime not found in database!")

# Display current ratings
st.sidebar.markdown("---")
st.sidebar.subheader("Your Ratings")

if st.session_state.user_ratings:
    for anime_name, rating in st.session_state.user_ratings.items():
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(f"{anime_name}: ⭐ {rating}")
        if col2.button("❌", key=f"remove_{anime_name}"):
            del st.session_state.user_ratings[anime_name]
            st.rerun()
else:
    st.sidebar.info("No ratings yet. Add some anime to get started!")

# Clear all ratings button
if st.session_state.user_ratings:
    if st.sidebar.button("Clear All Ratings"):
        st.session_state.user_ratings = {}
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Get Recommendations")
    
    if len(st.session_state.user_ratings) < 3:
        st.warning("⚠️ Please rate at least 3 anime to get accurate recommendations!")
    
    n_recommendations = st.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )

with col2:
    st.header("Stats")
    st.metric("Anime Rated", len(st.session_state.user_ratings))
    if st.session_state.user_ratings:
        avg_rating = np.mean(list(st.session_state.user_ratings.values()))
        st.metric("Average Rating", f"{avg_rating:.1f}")

# Generate recommendations button
if st.button(" Generate Recommendations", type="primary", use_container_width=True):
    if not st.session_state.user_ratings:
        st.error("Please rate at least one anime first!")
    else:
        with st.spinner("Generating personalized recommendations..."):
            # Convert names to IDs
            new_user_watched_by_id = {}
            for name, rating in st.session_state.user_ratings.items():
                if name in name_to_id:
                    new_user_watched_by_id[name_to_id[name]] = rating
            
            watched_ids = list(new_user_watched_by_id.keys())
            
            # Get recommendations
            recommendations, cluster_id = get_cluster_based_recommendations(
                watched_ids,
                new_user_watched_by_id,
                n_recommendations=n_recommendations
            )
            
            if not recommendations.empty:
                st.success(f" Found {len(recommendations)} recommendations based on Cluster #{cluster_id}")
                
                # Display recommendations
                st.markdown("---")
                st.header("Recommended Anime")
                
                for idx, row in recommendations.iterrows():
                    with st.container():
                        col1, col2, col3 = st.columns([3, 2, 1])
                        
                        with col1:
                            st.subheader(row['title'])
                            st.write(f"**Genres:** {row['genres']}")
                        
                        with col2:
                            # Handle non-numeric scores
                            try:
                                score_value = float(row['score'])
                                st.metric("Score", f"{score_value:.2f}")
                            except (ValueError, TypeError):
                                st.metric("Score", str(row['score']))
                        
                        with col3:
                            st.metric("Match", f"{row['recommendation_score']:.1f}")
                        
                        st.markdown("---")
            else:
                st.error("No recommendations found. Try rating more anime!")