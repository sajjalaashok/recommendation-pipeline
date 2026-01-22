import pandas as pd
import glob
import os

# Config
PATH = "raw_zone/"
TXN_PATH = PATH + "transactions/"
LEGACY_PATH = PATH + "recomart_raw_transactions_dec_2025.csv"

# Load Legacy
legacy_df = pd.read_csv(LEGACY_PATH)
print(f"Legacy File Rows: {len(legacy_df)}")

# Load Partitions
partition_files = glob.glob(os.path.join(TXN_PATH, "**/*.csv"), recursive=True)
partition_dfs = [pd.read_csv(f) for f in partition_files]
partition_df = pd.concat(partition_dfs, ignore_index=True) if partition_dfs else pd.DataFrame()
print(f"Partition files combined rows: {len(partition_df)}")

# Check Overlap
print("\nChecking for duplicates in Legacy file itself:")
print(legacy_df.duplicated(subset=['txn_id'], keep=False).sum())

print("\nChecking for duplicates in Partitions themselves:")
if not partition_df.empty:
    print(partition_df.duplicated(subset=['txn_id'], keep=False).sum())

# Check Intersection
if not partition_df.empty:
    combined = pd.concat([legacy_df, partition_df], ignore_index=True)
    duplicates = combined.duplicated(subset=['txn_id'], keep=False)
    print(f"\nTotal Combined Rows: {len(combined)}")
    print(f"Total Duplicates detected (txn_id): {duplicates.sum()}")
    
    # Show example
    dup_ids = combined[duplicates]['txn_id'].head(1).values
    if len(dup_ids) > 0:
        print(f"\nExample Duplicate ID: {dup_ids[0]}")
        print(combined[combined['txn_id'] == dup_ids[0]])
