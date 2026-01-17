import pandas as pd
from pathlib import Path

INGEST_MONTHLY_TXNS = Path("raw_zone").glob("*transaction*.csv").__next__()
INGEST_CUSTOMERS = Path("raw_zone").glob("*customers.csv").__next__()
INGEST_PRODUCTS = Path("raw_zone").glob("*product_catalog.csv").__next__()
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(exist_ok=True)

def transform():
    # Raw data to transform
    # columns : txn_id,txn_date,customer_id,product_id,quantity
    monthly_txns = pd.read_csv(INGEST_MONTHLY_TXNS)
    monthly_txns.info()

    # Supported data sources
    customers = pd.read_csv(INGEST_CUSTOMERS)
    customers["gender"] = customers["gender"].map({"M": "M", "F": "F", "O": "M"})
    products = pd.read_csv(INGEST_PRODUCTS)

    # Drop records with if record does not contains either customer_id or product_id
    monthly_txns = monthly_txns[monthly_txns['customer_id'].notna() & monthly_txns['product_id'].notna()]
    
    # # Clean numeric values
    monthly_txns['quantity'] = pd.to_numeric(monthly_txns['quantity'], errors='coerce')

    # Derived features [age,gender,product_category, product_price]
    monthly_txns = pd.merge(monthly_txns, customers, on='customer_id')
    monthly_txns = pd.merge(monthly_txns, products, on='product_id')
    monthly_txns['product_price'] = monthly_txns['base_price'] - (monthly_txns['base_price'] * (monthly_txns['discount_percent'] / 100))
    monthly_txns['total_price'] = monthly_txns['product_price'] * monthly_txns['quantity']
    monthly_txns['txn_id'] = monthly_txns.groupby(['customer_id', 'txn_date']).ngroup()

    # Discretize age column
    monthly_txns['age'] = pd.cut(monthly_txns['age'], bins=[0, 18, 35, 50, 65, 80], labels=['0-18', '19-35', '36-50', '51-65', '66-80'])

    # Encode nominal columns [gender, brand, category, super_category]
    monthly_txns['gender'] = monthly_txns['gender'].astype('category').cat.codes
    monthly_txns['brand'] = monthly_txns['brand'].astype('category').cat.codes
    monthly_txns['category'] = monthly_txns['category'].astype('category').cat.codes
    monthly_txns['super_category'] = monthly_txns['super_category'].astype('category').cat.codes
    monthly_txns['age'] = monthly_txns['age'].astype('category').cat.codes

    # Interested columns
    monthly_txns = monthly_txns[['txn_id', 'txn_date', 'customer_id', 'product_id', 'quantity', 'age', 'gender', 'super_category', 'category', 'brand']]

    monthly_txns.to_csv(OUT_DIR/"transactions_processed.csv", index=False)
    print("Processed data saved →", OUT_DIR/"transactions_processed.csv")

if __name__ == "__main__":
    transform()