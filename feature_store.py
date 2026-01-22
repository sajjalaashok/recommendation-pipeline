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
    # Load enriched transactions (produced by transform.py)
    enriched_path = Path("data/processed/transactions_enriched.csv")
    if not enriched_path.exists():
        print(f"Error: {enriched_path} not found. Run transform.py first.")
        return
        
    df = pd.read_csv(enriched_path)
    
    # Feature Engineering
    # 1. User Activity: Frequency of transactions
    df['user_txn_count'] = df.groupby('customer_id')['txn_id'].transform('nunique')
    
    # 2. Product stats: Average volume
    df['prod_avg_quantity'] = df.groupby('product_id')['quantity'].transform('mean')

    # Load unified interactions (Transaction + Clickstream)
    unified_path = Path("data/processed/unified_interactions.csv")
    if unified_path.exists():
        interactions = pd.read_csv(unified_path)
    else:
        # Fallback to transactions only
        interactions = df.groupby(['customer_id', 'product_id'], as_index=False)['quantity'].sum()
        interactions = interactions.rename(columns={'quantity': 'interaction_score'})
        
    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)
    
    # Store Tables
    # 'features' contains denormalized txn-level features
    df.to_sql("features", conn, if_exists="replace", index=False)
    
    # 'interactions' is the core for SVD/Collaborative Filtering
    interactions.to_sql("interactions", conn, if_exists="replace", index=False)
    
    # 'product_metadata' is core for Content-Based Filtering
    # We use the full catalog as the base to ensure metadata is never "missing"
    catalog_path = Path("raw_zone/recomart_product_catalog.csv")
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path)
        # Pull calculated features from df (transactions) if available
        # product_price, sentiment_score, popularity_index are in 'df' (transactions_enriched)
        # We join them into the catalog
        metadata_cols = ['product_id', 'product_price', 'sentiment_score', 'popularity_index']
        # Drop duplicates in df to get per-product values
        df_features = df[metadata_cols].drop_duplicates('product_id')
        
        product_features = pd.merge(catalog, df_features, on='product_id', how='left')
        
        # Fill missing values from transactions (if any)
        # Actually catalog likely already has some of these, but we prioritize the enriched ones
        product_features.to_sql("product_metadata", conn, if_exists="replace", index=False)
    else:
        # Fallback to enriched set only
        product_features = df[['product_id', 'product_name', 'category', 'brand', 'product_price', 
                              'sentiment_score', 'popularity_index']].drop_duplicates()
        product_features.to_sql("product_metadata", conn, if_exists="replace", index=False)

    # Optional: Indexing
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cust ON interactions (customer_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prod ON interactions (product_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_feat_cust ON features (customer_id)")
    
    conn.commit()
    conn.close()

    print("Feature store updated with multi-source data ->", DB_PATH)

if __name__ == "__main__":
    build_feature_store()