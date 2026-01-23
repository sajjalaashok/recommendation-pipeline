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

    # 3. Average Ratings (Scores)
    # We need interactions first to calculate this correctly, but we can do it after loading unified


    # Load unified interactions (Transaction + Clickstream)
    unified_path = Path("data/processed/unified_interactions.csv")
    if unified_path.exists():
        interactions = pd.read_csv(unified_path)
    else:
        # Fallback to transactions only
        interactions = df.groupby(['customer_id', 'product_id'], as_index=False)['quantity'].sum()
        interactions = interactions.rename(columns={'quantity': 'interaction_score'})
        
    # Calculate Average Ratings from Interactions
    user_avg = interactions.groupby('customer_id')['interaction_score'].mean().rename('user_avg_score')
    item_avg = interactions.groupby('product_id')['interaction_score'].mean().rename('item_avg_score')
    
    # Merge back to df (features) - Note: df is txn level, so we map it
    df = df.merge(user_avg, on='customer_id', how='left')
    df = df.merge(item_avg, on='product_id', how='left')
    
    # 4. Co-occurrence (Simplified: Top 10 pairs)
    # Self-join on txn_id to find items bought together
    print("Calculating Co-occurrence...")
    txns_only = df[['txn_id', 'product_id']].drop_duplicates()
    pairs = txns_only.merge(txns_only, on='txn_id')
    # Filter A != B and A < B to avoid double counting (A-B, B-A)
    pairs = pairs[pairs['product_id_x'] < pairs['product_id_y']]
    
    co_counts = pairs.groupby(['product_id_x', 'product_id_y']).size().reset_index(name='frequency')
    co_counts = co_counts.sort_values('frequency', ascending=False)
    # Keep top 1000 global pairs or Top N per product (Global is safer for speed in this demo)
    co_occurrence = co_counts.head(1000)
    co_occurrence = co_occurrence.rename(columns={'product_id_x': 'product_a', 'product_id_y': 'product_b'})

        
    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)
    
    # Store Tables
    # 'features' contains denormalized txn-level features
    df.to_sql("features", conn, if_exists="replace", index=False)
    
    # 'interactions' is the core for SVD/Collaborative Filtering
    interactions.to_sql("interactions", conn, if_exists="replace", index=False)

    # 'co_occurrence' for market basket analysis results
    co_occurrence.to_sql("co_occurrence", conn, if_exists="replace", index=False)

    
    # 'product_metadata' is core for Content-Based Filtering
    # We use the full catalog as the base to ensure metadata is never "missing"
    catalog_path = Path("raw_zone/recomart_product_catalog.csv")
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path)
        # Pull calculated features from df (transactions) if available
        # Pull calculated features from df (transactions) if available
        # product_price, sentiment_score, popularity_index are in 'df' (transactions_enriched)
        # We also added item_avg_score
        metadata_cols = ['product_id', 'product_price', 'sentiment_score', 'popularity_index', 'item_avg_score']

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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prod ON interactions (product_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_feat_cust ON features (customer_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_co_a ON co_occurrence (product_a)")
    
    # Export SQL Schema
    with open("model_store/schema.sql", "w") as f:
        for line in conn.iterdump():
            if line.startswith("CREATE TABLE"):
                f.write(f"{line};\n")
    
    conn.commit()
    conn.close()
    
    print("Schema exported to model_store/schema.sql")


    print("Feature store updated with multi-source data ->", DB_PATH)

if __name__ == "__main__":
    build_feature_store()