import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROCESSED = Path("data/processed/transactions_processed.csv")
ANALYTICS = Path("data/analytics")
ANALYTICS.mkdir(exist_ok=True)

def analytics():
    df = pd.read_csv(PROCESSED)

    df.describe(include='all').to_csv(ANALYTICS/"summary.csv")
    
    print("Analytics created →", ANALYTICS)

    # Conduct exploratory analysis showing interaction distributions, item popularity, and sparsity patterns.


if __name__ == "__main__":
    analytics()
