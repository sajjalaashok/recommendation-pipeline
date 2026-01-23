# Feature Store SQL Schema Summary

The Feature Store is a SQLite database (`model_store/feature_store.db`) containing the following tables:

## 1. `features`
Denormalized transaction-level data enriched with user and product statistics.
*   **Key Columns**: `txn_id`, `customer_id`, `product_id`, `txn_date`
*   **Calculated Features**:
    *   `user_txn_count`: Total number of transactions by this user.
    *   `prod_avg_quantity`: Average quantity sold per transaction for this product.
    *   `user_avg_score`: Average interaction score (implicit rating) for this user.
    *   `item_avg_score`: Average interaction score (implicit rating) for this product.

## 2. `interactions`
The clean User-Item Interaction Matrix used for Collaborative Filtering.
*   **Columns**: `customer_id`, `product_id`, `interaction_score`
*   **Description**: Aggregates all interactions (views + add_to_cart + purchase) into a single score.

## 3. `product_metadata`
Enriched product catalog used for Content-Based Filtering.
*   **Columns**: `product_id`, `product_name`, `category`, `brand`, `product_price`, `sentiment_score`, `popularity_index`, `item_avg_score`
*   **Description**: Combines static catalog data with dynamic metadata (sentiment, popularity) and calculated stats.

## 4. `co_occurrence`
Market Basket Analysis results (Top-1000 pairs).
*   **Columns**: `product_a`, `product_b`, `frequency`
*   **Description**: Stores pairs of products frequently purchased in the same transaction. Used for "Frequently Bought Together" recommendations.
