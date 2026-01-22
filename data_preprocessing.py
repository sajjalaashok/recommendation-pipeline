import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Config
RAW_PATH = Path("raw_zone")
PROCESSED_PATH = Path("data/processed")
EDA_PATH = PROCESSED_PATH / "eda"

# Ensure output directories exist
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
EDA_PATH.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load datasets from raw_zone."""
    datasets = {}
    
    # Standard files
    for name in ["recomart_product_catalog", "recomart_raw_customers", "recomart_raw_products", "external_metadata"]:
        p = RAW_PATH / f"{name}.csv"
        if p.exists():
            datasets[name] = pd.read_csv(p)
    
    # Transactions (Partitions + Legacy)
    txn_dfs = []
    
    # Partitions
    import glob
    partition_files = glob.glob(str(RAW_PATH / "transactions/**/*.csv"), recursive=True)
    if partition_files:
        txn_dfs.extend([pd.read_csv(f) for f in partition_files])
        
    # Legacy

        
    if txn_dfs:
        datasets["transactions"] = pd.concat(txn_dfs, ignore_index=True)
        
    return datasets

def preprocess_catalog(df):
    """Clean, Encode, and Normalize product catalog."""
    print("Preprocessing Catalog (Encoding & Scaling)...")
    df = df.copy()
    
    # Fill missing
    cols_to_fill_zero = ["discount_percent", "monthly_sales_volume", "return_rate_percent"]
    for c in cols_to_fill_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Label Encoding (Categorical)
    cat_cols = ["category", "brand", "is_perishable", "super_category"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[f"{col}_encoded"] = le.fit_transform(df[col])
            
    # MinMax Scaling (Numerical)
    num_cols = ["base_price", "monthly_sales_volume", "avg_rating", "return_rate_percent"]
    scaler = MinMaxScaler()
    for col in num_cols:
        if col in df.columns:
            df[f"{col}_scaled"] = scaler.fit_transform(df[[col]])
            
    return df

def preprocess_customers(df):
    """Clean and process customer data."""
    print("Preprocessing Customers...")
    df = df.copy()
    
    df = df.drop_duplicates(subset=["customer_id"])
    
    # Encode Gender
    if "gender" in df.columns:
        le = LabelEncoder()
        df["gender_encoded"] = le.fit_transform(df["gender"].astype(str))
        
    # Scale Age
    if "age" in df.columns:
        scaler = MinMaxScaler()
        df["age_scaled"] = scaler.fit_transform(df[["age"]])
        
    return df

def preprocess_transactions(df):
    """Clean transactions."""
    print("Preprocessing Transactions...")
    df = df.copy()
    
    if "txn_date" in df.columns:
        df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")
        
    df = df.dropna(subset=["customer_id", "product_id"])
    return df

def perform_eda(transactions, catalog):
    """Generate EDA plots and stats."""
    print("Performing Exploratory Data Analysis...")
    
    # 1. Sparsity
    n_users = transactions["customer_id"].nunique()
    n_items = transactions["product_id"].nunique()
    n_interactions = len(transactions)
    matrix_size = n_users * n_items
    sparsity = 1 - (n_interactions / matrix_size) if matrix_size > 0 else 0
    
    print(f"  - Users: {n_users}")
    print(f"  - Items: {n_items}")
    print(f"  - Interactions: {n_interactions}")
    print(f"  - Sparsity: {sparsity:.4%}")
    
    # 2. Item Popularity (Top 20)
    plt.figure(figsize=(12, 6))
    top_items = transactions["product_id"].value_counts().head(20)
    # Merge with catalog for names if available
    if not catalog.empty and "product_name" in catalog.columns:
        names = catalog.set_index("product_id")["product_name"]
        labels = [names.get(pid, pid) for pid in top_items.index]
    else:
        labels = top_items.index
        
    sns.barplot(x=top_items.values, y=labels, palette="viridis")
    plt.title("Top 20 Popular Items")
    plt.xlabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(EDA_PATH / "item_popularity.png")
    plt.close()
    
    # 3. User Interaction Dist (Log scale)
    plt.figure(figsize=(10, 5))
    user_counts = transactions["customer_id"].value_counts()
    sns.histplot(user_counts, bins=30, kde=True, log_scale=(True, False))
    plt.title("User Interaction Distribution (Log Scale)")
    plt.xlabel("Interactions per User")
    plt.savefig(EDA_PATH / "user_activity_dist.png")
    plt.close()
    
    print(f"  - EDA plots saved to {EDA_PATH}")

def main():
    dfs = load_data()
    
    catalog = pd.DataFrame()
    if "recomart_product_catalog" in dfs:
        catalog = preprocess_catalog(dfs["recomart_product_catalog"])
        catalog.to_csv(PROCESSED_PATH / "clean_product_catalog.csv", index=False)
        
    if "recomart_raw_customers" in dfs:
        cust = preprocess_customers(dfs["recomart_raw_customers"])
        cust.to_csv(PROCESSED_PATH / "clean_customers.csv", index=False)
        
    if "transactions" in dfs:
        txn = preprocess_transactions(dfs["transactions"])
        txn.to_csv(PROCESSED_PATH / "clean_transactions.csv", index=False)
        
        # Run EDA only if we have transactions
        perform_eda(txn, catalog)
        
    print(f"\nProcessing complete. Outputs in {PROCESSED_PATH}")

if __name__ == "__main__":
    main()
