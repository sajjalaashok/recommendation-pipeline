import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Config
LANDING_ZONE = Path("raw_zone/landing")
TARGET_DIR = Path("raw_zone/transactions")
ARCHIVE_DIR = LANDING_ZONE / "archive"
CATALOG_FILE = Path("raw_zone/recomart_product_catalog.csv")
CUSTOMERS_FILE = Path("raw_zone/recomart_raw_customers.csv")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest_from_landing():
    """Moves transactions from landing zone into partitioned folders and archives them."""
    # Ensure directories exist
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process all CSVs in landing zone
    txn_files = list(LANDING_ZONE.glob("*.csv"))
    
    if not txn_files:
        logging.info("No new files found in landing zone.")
        return

    for txn_file in txn_files:
        logging.info(f"Processing {txn_file}...")
        try:
            df = pd.read_csv(txn_file)
            if 'txn_date' not in df.columns:
                logging.warning(f"No txn_date in {txn_file}. Using today's date.")
                date_str = datetime.now().strftime("%Y-%m-%d")
            else:
                # Convert to datetime to extract date
                df['date_parsed'] = pd.to_datetime(df['txn_date'])
                # Group by date and save separately
                for date, group in df.groupby(df['date_parsed'].dt.date):
                    date_folder = TARGET_DIR / str(date)
                    date_folder.mkdir(parents=True, exist_ok=True)
                    
                    target_file = date_folder / f"ingested_{txn_file.name}"
                    # Don't overwrite if not necessary, but for ingestion we usually replace or append
                    group.drop(columns=['date_parsed']).to_csv(target_file, index=False)
                    logging.info(f"Saved partition for {date} to {target_file}")
            
            # Move processed file to archive
            shutil.move(str(txn_file), str(ARCHIVE_DIR / txn_file.name))
            logging.info(f"Archived {txn_file} to {ARCHIVE_DIR}")
            
        except Exception as e:
            logging.error(f"Error processing {txn_file}: {e}")

def organize_catalog():
    """Placeholder for catalog organization if needed."""
    pass

if __name__ == "__main__":
    ingest_from_landing()
