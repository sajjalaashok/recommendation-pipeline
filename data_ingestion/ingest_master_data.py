import shutil
from pathlib import Path
import logging

# Config
SOURCE_DIR = Path("data/raw")
TARGET_DIR = Path("raw_zone")
MASTER_FILES = {
    "recomart_product_catalog.csv": "recomart_product_catalog.csv",
    "recomart_raw_customers.csv": "recomart_raw_customers.csv",
    "recomart_raw_products.csv": "recomart_raw_products.csv",
    "recomart_raw_transactions_dec_2025.csv": "recomart_raw_transactions_dec_2025.csv"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest_master_data():
    """Syncs master data from simulator source to raw zone."""
    # Ensure full directory structure exists
    subdirs = ["landing", "transactions", "clickstream", "landing/archive"]
    for subdir in subdirs:
        (TARGET_DIR / subdir).mkdir(parents=True, exist_ok=True)
    
    for src_name, dest_name in MASTER_FILES.items():
        src_path = SOURCE_DIR / src_name
        dest_path = TARGET_DIR / dest_name
        
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            logging.info(f"Ingested {src_name} → {dest_path}")
        else:
            logging.error(f"Master source {src_path} not found!")

if __name__ == "__main__":
    ingest_master_data()
