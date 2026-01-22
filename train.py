import random
from datetime import datetime
import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
import os
import contextlib
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# --- MLflow Safety Wrapper ---
MLFLOW_AVAILABLE = False
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    print("MLflow imported successfully.")
except ImportError:
    print("WARNING: MLflow not available. Using LocalLogger fallback.")

class LocalLogger:
    def __init__(self, log_path="model_store/training_log.json"):
        self.log_path = log_path
        self.current_run = None
        self.data = {}
        # Load existing if any
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    self.data = json.load(f)
            except:
                self.data = {}

    @contextlib.contextmanager
    def start_run(self, run_name=None):
        run_id = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n[LocalLogger] Starting Run: {run_id}")
        self.current_run = {
            "run_id": run_id,
            "run_name": run_name,
            "timestamp": str(datetime.now()),
            "params": {},
            "metrics": {},
            "artifacts": []
        }
        try:
            yield self
        finally:
            # Save run
            if "runs" not in self.data:
                self.data["runs"] = []
            self.data["runs"].append(self.current_run)
            with open(self.log_path, 'w') as f:
                json.dump(self.data, f, indent=4)
            print(f"[LocalLogger] Run saved to {self.log_path}")
            self.current_run = None

    def log_metric(self, key, value):
        print(f"[LocalLogger] Metric: {key} = {value}")
        if self.current_run:
            self.current_run["metrics"][key] = value

    def log_param(self, key, value):
        print(f"[LocalLogger] Param: {key} = {value}")
        if self.current_run:
            self.current_run["params"][key] = value
            
    def log_model(self, model, artifact_path):
        print(f"[LocalLogger] Model saved: {artifact_path}")
        # Identify model type
        model_name = artifact_path + ".pkl"
        # We rely on manual joblib dumps in code, this just logs the event
        if self.current_run:
            self.current_run["artifacts"].append(artifact_path)
            
    def set_experiment(self, exp_name):
        print(f"[LocalLogger] Experiment: {exp_name}")

# Initialize Global Logger
if MLFLOW_AVAILABLE:
    logger = mlflow
    logger.log_model = mlflow.sklearn.log_model
else:
    logger = LocalLogger()
    # Mock specific sub-functions if needed, or wrap them dynamically
    # For mlflow.sklearn.log_model(model, name), our LocalLogger.log_model takes (model, name)
    # We need to map methods if they differ. 
    # MLflow: mlflow.sklearn.log_model(sk_model, artifact_path, ...)
    # Local: We just use the instance method directly. 
    # To mimic module structure for `mlflow.sklearn.log_model`:
    class MockSklearn:
        def log_model(self, model, artifact_path, **kwargs):
            logger.log_model(model, artifact_path)
    logger.sklearn = MockSklearn()


# Setup
DB_PATH = Path("model_store/feature_store.db")
MODEL_STORE = Path("model_store")
MODEL_STORE.mkdir(exist_ok=True)

try:
    logger.set_experiment("Recommendation_System_Experiment")
except:
    pass # LocalLogger might not strictly fail but just in case

def load_data():
    conn = sqlite3.connect(DB_PATH)
    interactions = pd.read_sql("SELECT * FROM interactions", conn)
    try:
        products = pd.read_sql("SELECT * FROM product_metadata", conn)
    except:
        products = pd.DataFrame()
        print("Warning: product_metadata table not found. Using fallback.")
    conn.close()
    return interactions, products

def train_collaborative_filtering(interactions):
    print("Training Collaborative Filtering (TruncatedSVD)...")
    
    # Pivot for User-Item Matrix
    user_item_matrix = interactions.pivot(index='customer_id', columns='product_id', values='interaction_score').fillna(0)
    
    X = user_item_matrix.values
    
    with logger.start_run(run_name="Collaborative_Filtering"):
        n_components = 20
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors = svd.fit_transform(X)
        item_factors = svd.components_.T # Product embeddings
        
        reconstructed = np.dot(user_factors, item_factors.T)
        
        # Calculate standard regression metrics
        mask = X > 0
        if np.any(mask):
            rmse = np.sqrt(mean_squared_error(X[mask], reconstructed[mask]))
            logger.log_metric("rmse", rmse)
            print(f"SVD RMSE: {rmse:.4f}")
        else:
            print("Warning: Matrix is empty or all zeros.")

        # Calculate Ranking Metrics (Precision@10)
        # We sample 100 users for speed in this demo
        sample_users = random.sample(range(X.shape[0]), min(100, X.shape[0]))
        precisions = []
        
        for u_idx in sample_users:
            actual_items = np.where(X[u_idx, :] > 3)[0] # Threshold for 'actual' interest (e.g., add_to_cart=3)
            if len(actual_items) == 0: continue
            
            # Predict scores for all items
            pred_scores = reconstructed[u_idx, :]
            top_k_indices = np.argsort(pred_scores)[::-1][:10]
            
            p_at_k = len(set(actual_items).intersection(set(top_k_indices))) / 10.0
            precisions.append(p_at_k)
            
        avg_p_at_10 = np.mean(precisions) if precisions else 0
        logger.log_metric("precision_at_10", avg_p_at_10)
        print(f"Avg Precision@10: {avg_p_at_10:.4f}")

        # Save artifacts (ALWAYS save to disk regardless of logger)
        joblib.dump(svd, MODEL_STORE / "svd_model.pkl")
        joblib.dump(user_item_matrix, MODEL_STORE / "user_item_matrix.pkl")
        joblib.dump(interactions, MODEL_STORE / "interactions.pkl")
        
        # Log to "MLflow" (or fallback)
        try:
            logger.sklearn.log_model(svd, "model")
        except:
             logger.log_model(svd, "model")
        
    return user_item_matrix, reconstructed

def train_content_based(products):
    print("\nTraining Content-Based Filtering...")
    
    # Enrichment
    products['soup'] = products['product_name'].fillna('') + " " + \
                       products['category'].fillna('') + " " + \
                       products['brand'].fillna('')
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products['soup'])
    
    from sklearn.metrics.pairwise import linear_kernel
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    with logger.start_run(run_name="Content_Based"):
        logger.log_param("n_products", len(products))
        
        mean_sim = np.mean(cosine_sim)
        logger.log_metric("mean_item_similarity", mean_sim)
        
        joblib.dump(products, MODEL_STORE / "products.pkl")
        joblib.dump(cosine_sim, MODEL_STORE / "cosine_sim.pkl")
        print(f"CB Similarity Matrix calculated ({cosine_sim.shape})")

def main():
    interactions, products = load_data()
    
    if not interactions.empty:
        train_collaborative_filtering(interactions)
    else:
        print("No interactions found. Skipping CF training.")
    
    if not products.empty:
        train_content_based(products)
    else:
        print("No products found. Skipping CB training.")

if __name__ == "__main__":
    main()
