from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import sqlite3
from feature_registry import FeatureRegistry

app = FastAPI(title="Recomart Recommendation API", version="1.0")

# ------------------------------------------------------------------
# Global State (Models & Registry)
# ------------------------------------------------------------------
class AppState:
    svd_model = None
    user_item_matrix = None
    interactions = None
    feature_registry = None

    @classmethod
    def load(cls):
        print("Loading models and registry...")
        try:
            cls.svd_model = joblib.load("model_store/svd_model.pkl")
            # We need the user mapping from the training phase to map ID -> Index
            cls.user_item_matrix = joblib.load("model_store/user_item_matrix.pkl") 
            # Reconstruct mappings
            cls.user_map = {uid: idx for idx, uid in enumerate(cls.user_item_matrix.index)}
            cls.item_map = {idx: iid for idx, iid in enumerate(cls.user_item_matrix.columns)}
            # Inverse map for lookup
            cls.item_inv_map = {iid: idx for idx, iid in enumerate(cls.user_item_matrix.columns)}
        except FileNotFoundError:
            print("Warning: SVD model artifacts not found. /recommend/user will fail.")

        cls.feature_registry = FeatureRegistry()
        print("Startup complete.")

@app.on_event("startup")
async def startup_event():
    AppState.load()

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "online", "model": "SVD + Content-Based"}

@app.get("/features/user/{user_id}")
def get_user_features(user_id: str):
    """Debug endpoint to fetch raw features for a user."""
    result = AppState.feature_registry.get_online_features({'customer_id': [user_id]})
    if result.get('customer_features') is not None and not result['customer_features'].empty:
        return result['customer_features'].to_dict(orient="records")[0]
    raise HTTPException(status_code=404, detail="User not found in feature store")

@app.get("/features/product/{product_id}")
def get_product_features(product_id: str):
    """Debug endpoint to fetch raw features for a product."""
    result = AppState.feature_registry.get_online_features({'product_id': [product_id]})
    if result.get('product_features') is not None and not result['product_features'].empty:
        return result['product_features'].to_dict(orient="records")[0]
    raise HTTPException(status_code=404, detail="Product not found in feature store")

@app.get("/recommend/user/{user_id}")
def recommend_for_user(user_id: str, k: int = 5):
    """
    Returns top-k recommendations for a user using SVD.
    """
    if AppState.svd_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Look up user index
    u_idx = AppState.user_map.get(user_id)
    if u_idx is None:
        return {"user_id": user_id, "recommendations": [], "note": "New user (Cold Start)"}

    # 2. Predict
    # SVD reconstructs the row for this user
    user_vector = AppState.user_item_matrix.iloc[u_idx].values.reshape(1, -1)
    
    # Actually, SVD model transforms X -> factors. To get predictions:
    # dot(user_factors, item_factors.T). 
    # But since we have the Reconstructed Matrix in memory? No, usually we don't keep the full dense matrix in memory for Prod.
    # In 'train.py', we saved 'svd_model'.
    # transform() gives user factors. 
    user_factors = AppState.svd_model.transform(user_vector)
    item_factors = AppState.svd_model.components_.T
    
    pred_scores = np.dot(user_factors, item_factors.T).flatten()
    
    # 3. Rank
    top_indices = np.argsort(pred_scores)[::-1][:k]
    
    recommendations = []
    for idx in top_indices:
        item_id = AppState.item_map.get(idx)
        score = float(pred_scores[idx])
        recommendations.append({"product_id": item_id, "score": score})
        
    return {"user_id": user_id, "recommendations": recommendations}

@app.get("/recommend/item/{product_id}")
def recommend_similar_items(product_id: str, k: int = 5):
    """
    Returns items frequently bought together with the given product.
    Uses the 'co_occurrence' table from Feature Store.
    """
    df = AppState.feature_registry.get_co_occurrence(product_id, limit=k)
    if df.empty:
        return {"product_id": product_id, "similar_items": [], "note": "No associations found"}
    
    return {
        "product_id": product_id,
        "similar_items": df.to_dict(orient="records")
    }
