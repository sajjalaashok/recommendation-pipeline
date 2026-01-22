import pandas as pd
import numpy as np

def corrupt_file(path, key_col, cols_to_null):
    print(f"Corrupting {path}...")
    df = pd.read_csv(path)
    
    # Take last 10 rows to duplicate as base
    bad_rows = df.tail(10).copy()
    
    # Make IDs unique so we don't just get specific "Duplicate ID" errors (unless we want to test that too)
    # But user asked for "null and missing".
    bad_rows[key_col] = bad_rows[key_col].astype(str) + "_TEST_NULL"
    
    # Inject Nulls
    for col in cols_to_null:
        if col in bad_rows.columns:
            bad_rows[col] = np.nan
            
    # Append
    df_new = pd.concat([df, bad_rows], ignore_index=True)
    df_new.to_csv(path, index=False)
    print(f"  -> Added 10 rows. Total: {len(df_new)}")

# 1. Product Catalog
# Null out: product_name (String), base_price (Numeric), category (String)
corrupt_file("raw_zone/recomart_product_catalog.csv", "product_id", ["product_name", "base_price", "category"])

# 2. Customers
# Null out: age (Numeric), gender (Categorical)
corrupt_file("raw_zone/recomart_raw_customers.csv", "customer_id", ["age", "gender"])
