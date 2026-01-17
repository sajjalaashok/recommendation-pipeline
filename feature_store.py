# • Create features suitable for recommendation algorithms, such as:
#     ◦ User activity frequency
#     ◦ Average rating per user/item
#     ◦ Co-occurrence or similarity-based features
# • Store transformed data in a structured database or warehouse.

import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH = Path("model_store/feature_store.db")
DB_PATH.parent.mkdir(exist_ok=True)

def build_feature_store():
    # Load processed transactions
    df = pd.read_csv("data/processed/transactions_processed.csv")
    
    # Load raw product catalog for content-based filtering (text features)
    # We use the same logic as transform.py to find the file
    try:
        product_path = next(Path("raw_zone").glob("*product_catalog.csv"))
        products_df = pd.read_csv(product_path)
    except StopIteration:
        print("Warning: Product catalog not found in raw_zone. Content-based features might be limited.")
        products_df = pd.DataFrame() # Empty fallback

    base_cols = [
        'gender', 'age', 'super_category', 'category', 'brand',
        'product_price', 'total_price', 'quantity', 'txn_date',
        'txn_id', 'customer_id', 'product_id'
    ]

    # User activity frequency
    df['activity_frequency'] = df.groupby('customer_id')['txn_id'].transform('nunique')

    # Average rating per user/item (using quantity as proxy for implicit rating)
    df['avg_rating'] = df.groupby('customer_id')['quantity'].transform('mean')

    # Co-occurrence or similarity-based features
    df['co_occurrence'] = df.groupby('customer_id')['product_id'].transform('nunique')

    # Content-Based Features: Create a text description for each product
    # We'll use this for TF-IDF later.
    # Ensure no NaN values in text fields
    df['category'] = df['category'].fillna('')
    df['brand'] = df['brand'].fillna('')
    # We might need the original text if we encoded them, but let's check transform.py.
    # transform.py encodes category and brand to codes.
    # If we want content based on text, we need the mapping or use the codes as categorical features.
    # Ideally, we should have kept the text or have a look up.
    # Looking at transform.py, it encodes them:
    # monthly_txns['category'] = monthly_txns['category'].astype('category').cat.codes
    # This loses the semantic meaning for TF-IDF on text unless we have the original strings.
    # However, for this exercise, we might have to work with what we have (codes) or reload raw if needed.
    # Actually, let's just create a "features" table that has the processed data ready for training.
    
    # We will save the processed dataframe as is, but we might want a cleaner interaction matrix query later.
    
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("features", conn, if_exists="replace", index=False)
    
    # Create specific views/tables for models if needed, or just use 'features' table.
    # Let's create a view for interactions (User, Item, Rating/Implicit)
    # Implicit rating = quantity or just presence. 
    # Let's use frequency of purchase as 'rating' for now, or just binary.
    # Or better, let's aggregate to get a 'weight' for user-item pair.
    
    interactions = df.groupby(['customer_id', 'product_id']).size().reset_index(name='interaction_count')
    interactions.to_sql("interactions", conn, if_exists="replace", index=False)

    if not products_df.empty:
        products_df.to_sql("products", conn, if_exists="replace", index=False)

    conn.close()

    print("Feature store created →", DB_PATH)

if __name__ == "__main__":
    build_feature_store()