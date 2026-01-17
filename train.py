
import pandas as pd
import numpy as np
import sqlite3
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Setup
DB_PATH = Path("model_store/feature_store.db")
MODEL_STORE = Path("model_store")
MODEL_STORE.mkdir(exist_ok=True)

mlflow.set_experiment("Recommendation_System_Experiment")

def load_data():
    conn = sqlite3.connect(DB_PATH)
    interactions = pd.read_sql("SELECT * FROM interactions", conn)
    try:
        products = pd.read_sql("SELECT * FROM products", conn)
    except:
        products = pd.DataFrame()
        print("Warning: Products table not found or empty.")
    conn.close()
    return interactions, products

def train_collaborative_filtering(interactions):
    print("Training Collaborative Filtering (SVD)...")
    
    # Prepare User-Item Matrix
    # We use interaction_count as explicit rating proxy
    user_item_matrix = interactions.pivot(index='customer_id', columns='product_id', values='interaction_count').fillna(0)
    
    # Train/Test Split (Masking some interactions is complex here, so we'll just split raw interactions if we want strict eval)
    # But for Matrix Factorization on the whole matrix, usually we verify reconstruction error.
    # Let's do a proper split: Split interactions dataframe, then build matrix from train only.
    
    train_df, test_df = train_test_split(interactions, test_size=0.2, random_state=42)
    
    train_matrix = train_df.pivot(index='customer_id', columns='product_id', values='interaction_count').fillna(0)
    
    # Align columns (products) and rows (users) for test set (ensure dimensions match)
    # This is tricky with simple Pivot. We'll use the full matrix shape for SVD but train on train_matrix data only?
    # Simpler approach for this task: Dimensionality Reduction on the FULL matrix and measure reconstruction error?
    # Or Train on Train Matrix, predict on Test.
    # Let's use the full matrix for training to generate embeddings, and evaluate reconstruction error as a metric.
    # Then for strict "Evaluation", we can calculate RMSE on the known non-zero entries.
    
    X = user_item_matrix.values
    
    with mlflow.start_run(run_name="Collaborative_Filtering_SVD"):
        n_components = 20 # Hyperparameter
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        matrix_reduced = svd.fit_transform(X)
        matrix_reconstructed = svd.inverse_transform(matrix_reduced)
        
        # Calculate RMSE on the non-zero elements (actual interactions)
        # We want to minimize error on KNOWN interactions
        mask = X > 0
        rmse = np.sqrt(mean_squared_error(X[mask], matrix_reconstructed[mask]))
        mae = mean_absolute_error(X[mask], matrix_reconstructed[mask])
        
        print(f"CF (SVD) RMSE: {rmse:.4f}")
        print(f"CF (SVD) MAE: {mae:.4f}")
        
        mlflow.log_param("n_components", n_components)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(svd, "svd_model")
        
        # Save artifacts locally for Streamlit App
        joblib.dump(svd, MODEL_STORE / "svd_model.pkl")
        joblib.dump(user_item_matrix, MODEL_STORE / "user_item_matrix.pkl")
        joblib.dump(interactions, MODEL_STORE / "interactions.pkl")
        print("CF artifacts saved locally.")
        
    return user_item_matrix, matrix_reconstructed

def train_content_based(products, interactions):
    print("\nTraining Content-Based Filtering...")
    
    if products.empty:
        print("Skipping Content-Based: No product data.")
        return
        
    # Create content string
    # Fill NA
    products['description'] = products['category'].fillna('') + " " + \
                              products['brand'].fillna('') + " " + \
                              products['product_name'].fillna('')
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products['description'])
    
    # Compute Cosine Similarity between items
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Recommendation Logic (Quick Test)
    # Pick a popular product and show similar items
    example_product_idx = 0
    example_product_id = products.iloc[example_product_idx]['product_id']
    
    print(f"Generating similar items for product: {example_product_id}")
    
    sim_scores = list(enumerate(cosine_sim[example_product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6] # Top 5 excluding self
    
    with mlflow.start_run(run_name="Content_Based"):
        mlflow.log_param("n_products", len(products))
        
        print(f"Top 5 similar products to {example_product_id}:")
        for i, score in sim_scores:
            print(f"- {products.iloc[i]['product_id']} (Score: {score:.4f})")
            
        # Log a dummy metric or the avg similarity of top 5
        avg_sim = np.mean([s[1] for s in sim_scores])
        mlflow.log_metric("avg_top5_similarity", avg_sim)

        # Save artifacts locally for Streamlit App
        joblib.dump(products, MODEL_STORE / "products.pkl")
        joblib.dump(cosine_sim, MODEL_STORE / "cosine_sim.pkl")
        # Ensure we save the mapping or matrix if we want to do inference on new items,
        # but for the app we just look up existing similarities or re-compute if fast enough.
        # Saving DataFrame and Similarity Matrix is enough for now.
        print("CB artifacts saved locally.")

def main():
    interactions, products = load_data()
    
    # Collaborative Filtering
    if not interactions.empty:
        train_collaborative_filtering(interactions)
    else:
        print("No interactions data found.")
        
    # Content-Based Filtering
    if not products.empty:
        train_content_based(products, interactions)
    else:
        print("No products data found for Content-Based Filtering.")

if __name__ == "__main__":
    main()
