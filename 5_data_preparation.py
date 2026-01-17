#1. Imports & Data Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

SRC_ROOT_FOLDER = "data/raw/"
OUTPUT_ROOT_FOLDER = "data/processed/"

# Load datasets
customers = pd.read_csv(SRC_ROOT_FOLDER+"recomart_raw_customers.csv")
products = pd.read_csv(SRC_ROOT_FOLDER+"recomart_raw_products.csv")
transactions = pd.read_csv(SRC_ROOT_FOLDER+"recomart_raw_transactions_dec_2025.csv")
catalog = pd.read_csv(SRC_ROOT_FOLDER+"recomart_product_catalog.csv")

#2. Merge into a unified dataset
# Merge transactions with customers and products
df = transactions.merge(customers, on="customer_id", how="left")
df = df.merge(products, on="product_id", how="left")
df = df.merge(catalog[["product_id","base_price","category","avg_rating"]], on="product_id", how="left")
print("Columns after merge:", df.columns)

#3. Handle Missing Values
# Drop rows missing critical IDs
df = df.dropna(subset=["customer_id","product_id"])

# Fill numeric columns with median
num_cols = ["base_price","avg_rating","quantity","age"]
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

#4. Encode Categorical Attributes
cat_cols = []
if "category" in df.columns: cat_cols.append("category")
if "gender" in df.columns: cat_cols.append("gender")

encoders = {}
for col in cat_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col].astype(str))
    encoders[col] = enc

#5. Normalize Numerical Variables
scaler = MinMaxScaler()
for col in num_cols:
    if col in df.columns:
        df[col] = scaler.fit_transform(df[[col]])

# Normalize transaction timestamp
df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")
df["txn_timestamp"] = df["txn_date"].astype(np.int64) // 10**9
df["txn_timestamp"] = scaler.fit_transform(df[["txn_timestamp"]])

#6. Exploratory Analysis
#Interaction Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["quantity"], bins=30)
plt.title("Distribution of Transaction Quantities")
plt.savefig(OUTPUT_ROOT_FOLDER+"interaction_distribution.png", dpi=300, bbox_inches="tight")
#plt.show()
#Item Popularity
item_counts = df["product_id"].value_counts().head(20)
plt.figure(figsize=(10,5))
sns.barplot(x=item_counts.index, y=item_counts.values)
plt.title("Top 20 Most Popular Products")
plt.xticks(rotation=90)
plt.savefig(OUTPUT_ROOT_FOLDER+"popularity.png", dpi=300, bbox_inches="tight")
#plt.show()
#Sparsity Heatmap
user_item_matrix = df.pivot_table(index="customer_id", columns="product_id", values="quantity", fill_value=0)
plt.figure(figsize=(12,6))
sns.heatmap(user_item_matrix.sample(min(30, len(user_item_matrix)), axis=0).sample(min(30, len(user_item_matrix.columns)), axis=1), cmap="Blues")
plt.title("User-Item Interaction Matrix (Sampled)")
plt.savefig(OUTPUT_ROOT_FOLDER+"Sparsity.png", dpi=300, bbox_inches="tight")
#plt.show()

#7. Save Cleaned Dataset
df.to_csv(OUTPUT_ROOT_FOLDER+"recomart_cleaned_dataset.csv", index=False)


