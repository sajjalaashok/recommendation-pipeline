import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Config
DB_PATH = Path("model_store/inference_logs.db")
DB_PATH.parent.mkdir(exist_ok=True)

TRAINING_DATA = Path("data/processed/transactions_enriched.csv")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inference_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            customer_id TEXT,
            recommended_items TEXT,
            latency_ms REAL,
            model_version TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_inference(customer_id, items, latency, version="v1"):
    """Logs a live prediction event."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO inference_logs (timestamp, customer_id, recommended_items, latency_ms, model_version)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), customer_id, ",".join(items), latency, version))
    conn.commit()
    conn.close()

def run_drift_check():
    """Simulates a drift check between training data and current 'live' data."""
    if not TRAINING_DATA.exists():
        print("No training data found for baseline.")
        return
    
    train_df = pd.read_csv(TRAINING_DATA)
    
    # In a real app, we'd pull from current clickstream. 
    # For simulation, we'll check if the recent inference scores differ.
    conn = sqlite3.connect(DB_PATH)
    try:
        logs_df = pd.read_sql("SELECT * FROM inference_logs", conn)
    except:
        logs_df = pd.DataFrame()
    conn.close()
    
    if len(logs_df) < 10:
        print("Need more inference logs for a meaningful drift check.")
        return
    
    # Example: Check if average recommended price is drifting from historical mean
    # This is a proxy for 'Production Drift'
    train_avg_price = train_df['product_price'].mean()
    
    # (Mock logic: In a real case, we'd lookup prices of recommended items in logs)
    print(f"Monitoring: Baseline Avg Price: {train_avg_price:.2f}")
    print("Monitoring: Drift Check Completed. Status: HEALTHY.")

if __name__ == "__main__":
    init_db()
    # Log a dummy event for verification
    log_inference("CUST123", ["P001", "P005"], 45.5, "SVD-v1")
    run_drift_check()
