# 🎬 Anime Recommendation System

A machine learning-based anime recommendation system that uses collaborative filtering with K-Means clustering to provide personalized anime suggestions. The system includes both a Jupyter notebook for model training and a Streamlit web application for interactive recommendations.

## 📋 Features

- **Collaborative Filtering**: Uses user rating patterns to find similar users and recommend anime
- **K-Means Clustering**: Groups users into clusters for efficient similarity computation (630 clusters optimized via elbow method)
- **Content-Based Filtering**: TF-IDF vectorization on anime genres for content similarity
- **Interactive Web Interface**: Streamlit app for real-time recommendations
- **Sparse Matrix Optimization**: Efficient memory usage for large-scale data processing
- **Batch Processing**: Handles millions of ratings efficiently

## 🚀 Installation

### Prerequisites

- Python 3.13+
- UV package manager

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd PROJET

# Install dependencies
uv sync 
```

### Required Data Files

Place the following CSV files in your downloads folder or update paths in the code:

- `animes.csv` - Contains anime details (animeID, title, genres, score)
- `ratings.csv` - Contains user ratings (userID, animeID, rating)

You can download the datasets and the *.pkl files in the link bellow :

https://drive.google.com/drive/folders/1JHnTDJ6Sxuoc_MJRSn-_BipagIlHSW93?usp=sharing

## 📁 Project Structure

```
PROJET/
├── anime_rec.ipynb              # Main notebook for model training
├── streamlit_anime_rec.py       # Streamlit web application
├── pyproject.toml              # Project dependencies
├── kmeans_model.pkl            # Trained K-Means model (generated)
├── user_item_matrix.pkl        # Sparse user-item matrix (generated)
├── content_similarity.pkl      # Content-based similarity matrix (generated)
└── README.md                   # This file
```

## 🔧 Usage

### 1. Training the Model (Jupyter Notebook)

Open and run `anime_rec.ipynb` to:

1. **Load and Filter Data**

   - Filters animes with at least 50 ratings
   - Filters users with at least 10 ratings
   - Creates sparse user-item matrix
2. **Determine Optimal Clusters**

   - Tests K values from 20 to 1000
   - Uses elbow method to find optimal K (630 clusters)
3. **Train K-Means Model**

   - Uses MiniBatchKMeans for efficiency
   - Batch size: 1000 users
   - Saves model to `kmeans_model.pkl`
4. **Generate Recommendations**

   - Test the recommendation system with sample user profiles
   - Saves processed data for Streamlit app

### 2. Running the Streamlit App

```bash
streamlit run streamlit_anime_rec.py
```

**Using the App:**

1. Search for anime titles in the sidebar
2. Rate anime on a scale of 1-10
3. Add at least 3 ratings for accurate recommendations
4. Click "Generate Recommendations" to get personalized suggestions
5. View recommendations with match scores and genres

## 🧠 Technical Details

### Recommendation Algorithm

**Collaborative Filtering with Clustering:**

1. **User Profiling**: Creates a sparse vector of user's anime ratings
2. **Cluster Assignment**: Assigns user to nearest cluster using trained K-Means model
3. **Similarity Computation**: Calculates cosine similarity with top 50 users in the cluster
4. **Weighted Recommendation**: Aggregates ratings from similar users, weighted by similarity scores
5. **Filtering**: Excludes already-watched anime from recommendations

### Data Processing

- **Sparse Matrix Format**: Uses `scipy.sparse.csr_matrix` for memory efficiency
- **Batch Processing**: Processes users in batches of 1000 to handle large datasets
- **Filtering Thresholds**:
  - Minimum anime ratings: 50
  - Minimum user ratings: 10

### Content-Based Filtering

- Uses TF-IDF vectorization on anime genres
- Computes cosine similarity between anime
- Stored in `content_similarity.pkl`

## 📊 Model Performance

- **Number of Clusters**: 630 (optimized via elbow method)
- **Similarity Metric**: Cosine similarity
- **Top Similar Users**: 50 per recommendation
- **Matrix Shape**: ~1M users × anime count (sparse)

## 🛠️ Dependencies

```
matplotlib >= 3.10.7
numpy >= 2.3.4
pandas >= 2.3.3
pypickle >= 2.0.1
scikit-learn >= 1.7.2
scipy >= 1.16.3
streamlit >= 1.51.0
```

## 📝 Example Usage

```python
# Example: Get recommendations for a new user
new_user_watched = {
    'Cowboy Bebop': 9,
    'Naruto': 8,
    'Fullmetal Alchemist: Brotherhood': 10,
    'Death Note': 7,
}

recommendations, cluster_id = get_cluster_based_recommendations(
    watched_ids,
    new_user_watched,
    n_recommendations=10
)

print(recommendations)
```

## 🎯 Future Improvements

- [ ] Add explanation for why anime was recommended
- [ ] Add anime search with filters (genre, year, score)

## 🐛 Troubleshooting

**Issue: FileNotFoundError for CSV files**

- Update file paths in notebook and Streamlit app to match your data location

**Issue: Out of memory errors**

- Reduce `BATCH_SIZE` in the notebook
- Reduce `MIN_ANIME_RATINGS` or `MIN_USER_RATINGS` thresholds

**Issue: No recommendations generated**

- Ensure at least 3 anime are rated
- Check that rated anime exist in the filtered database

## 👤 Author

GOURICHI Abdellah
