import pandas as pd
import json
import glob
from pathlib import Path

# Config
RAW_PATH = Path("raw_zone")
TXN_PARTITIONS = RAW_PATH / "transactions"
CLICKSTREAM_PATH = RAW_PATH / "clickstream"
EXTERNAL_METADATA = RAW_PATH / "external_metadata.csv"
CATALOG_FILE = RAW_PATH / "recomart_product_catalog.csv"
CUSTOMERS_FILE = RAW_PATH / "recomart_raw_customers.csv"

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def transform():
    print("Loading partitioned transactions...")
    txn_files = list(TXN_PARTITIONS.glob("**/ingested_*.csv"))
    if not txn_files:
        print("Error: No transaction data found.")
        return
    else:
        monthly_txns = pd.concat([pd.read_csv(f) for f in txn_files], ignore_index=True)

    # Static data
    customers = pd.read_csv(CUSTOMERS_FILE)
    products = pd.read_csv(CATALOG_FILE)
    
    # External metadata (Mock API output)
    if EXTERNAL_METADATA.exists():
        metadata = pd.read_csv(EXTERNAL_METADATA)
        print(f"Loaded {len(metadata)} external metadata records.")
    else:
        metadata = pd.DataFrame()

    # Drop records with if record does not contains either customer_id or product_id
    monthly_txns = monthly_txns[monthly_txns['customer_id'].notna() & monthly_txns['product_id'].notna()]
    
    # # Clean numeric values
    monthly_txns['quantity'] = pd.to_numeric(monthly_txns['quantity'], errors='coerce')

    # Merge with static data
    monthly_txns = pd.merge(monthly_txns, customers, on='customer_id', how='left')
    monthly_txns = pd.merge(monthly_txns, products, on='product_id', how='inner') # Inner join to filter ghost products
    
    print(f"Transactions after filtering invalid products: {len(monthly_txns)}")
    
    # Merge with external metadata
    if not metadata.empty:
        monthly_txns = pd.merge(monthly_txns, metadata, on='product_id', how='left')
    
    # Calculate price features
    monthly_txns['product_price'] = monthly_txns['base_price'] - (monthly_txns['base_price'] * (monthly_txns['discount_percent'].fillna(0) / 100))
    monthly_txns['total_price'] = monthly_txns['product_price'] * monthly_txns['quantity']

    # Discretize age column
    monthly_txns['age'] = pd.cut(monthly_txns['age'], bins=[0, 18, 35, 50, 65, 80], labels=['0-18', '19-35', '36-50', '51-65', '66-80'])

    # Preserve labels for feature store (avoid premature encoding)
    monthly_txns.to_csv(OUT_DIR/"transactions_enriched.csv", index=False)
    print("Enriched transaction data saved ->", OUT_DIR/"transactions_enriched.csv")
    
    # Process Clickstream and Create Unified Interactions
    clickstream_df = process_clickstream()
    
    # Create unified interactions (Transaction interactions + Clickstream interactions)
    # Transaction interactions: quantity is proxy for strength? Or just binary?
    # Let's say purchase in transaction = 5 pts per quantity
    txn_interactions = monthly_txns[['customer_id', 'product_id', 'quantity']].copy()
    txn_interactions['interaction_score'] = txn_interactions['quantity'] * 5 
    
    # Merge
    if not clickstream_df.empty:
        # Filter clickstream for valid products only
        valid_product_ids = products['product_id'].unique()
        clickstream_df = clickstream_df[clickstream_df['product_id'].isin(valid_product_ids)]
        
        unified = pd.concat([txn_interactions, clickstream_df], ignore_index=True)
    else:
        unified = txn_interactions
        
    # Aggregate scores (sum) if multiple entries for same user-item
    unified = unified.groupby(['customer_id', 'product_id'], as_index=False)['interaction_score'].sum()
    
    # Save
    unified.to_csv(OUT_DIR/"unified_interactions.csv", index=False)
    print("Unified interactions saved ->", OUT_DIR/"unified_interactions.csv")

def process_clickstream():
    print("Processing clickstream data...")
    json_files = glob.glob(str(CLICKSTREAM_PATH / "**/*.json"), recursive=True)
    
    if not json_files:
        print("No clickstream files found.")
        return pd.DataFrame(columns=['customer_id', 'product_id', 'interaction_score'])
        
    all_events = []
    for f in json_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                all_events.extend(data)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not all_events:
        return pd.DataFrame(columns=['customer_id', 'product_id', 'interaction_score'])
        
    df = pd.DataFrame(all_events)
    
    # Map events to scores
    # event_strength is already in the data from generator, but let's enforce local logic or use it.
    # Generator: view=1, add_to_cart=3, purchase=5
    # The generator provides 'event_strength', we can just use it.
    
    # Aggregate: Sum strength per user-item
    # Rename user_id to customer_id to match transactions
    df = df.rename(columns={'user_id': 'customer_id', 'item_id': 'product_id'})
    
    interactions = df.groupby(['customer_id', 'product_id'], as_index=False)['event_strength'].sum()
    interactions = interactions.rename(columns={'event_strength': 'interaction_score'})
    
    return interactions

if __name__ == "__main__":
    transform()