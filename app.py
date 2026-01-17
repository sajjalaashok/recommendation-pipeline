import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Paths
MODEL_STORE = Path("model_store")

# Load Models (Cached)
@st.cache_resource
def load_artifacts():
    try:
        svd = joblib.load(MODEL_STORE / "svd_model.pkl")
        user_item_matrix = joblib.load(MODEL_STORE / "user_item_matrix.pkl")
        products = joblib.load(MODEL_STORE / "products.pkl")
        cosine_sim = joblib.load(MODEL_STORE / "cosine_sim.pkl")
        interactions = joblib.load(MODEL_STORE / "interactions.pkl")
        return svd, user_item_matrix, products, cosine_sim, interactions
    except FileNotFoundError:
        return None, None, None, None, None

svd_model, user_item_matrix, products_df, cosine_sim, interactions_df = load_artifacts()

# App Layout
st.set_page_config(page_title="Recommendation System", layout="wide")
st.title("🛍️ AI Recommendation System")

if svd_model is None:
    st.error("Models not found! Please run the training pipeline first to generate artifacts.")
else:
    # Sidebar
    st.sidebar.header("Configuration")
    rec_type = st.sidebar.radio("Recommendation Type", ["Collaborative Filtering", "Content-Based Filtering"])

    if rec_type == "Collaborative Filtering":
        st.header("👥 User-Based Recommendations (Collaborative)")
        st.write("Using Matrix Factorization (SVD) to recommend items based on user interaction history.")
        
        # User Selection
        users = user_item_matrix.index.tolist()
        selected_user = st.selectbox("Select Customer ID", users)
        
        # Display Purchase History
        st.subheader(f"📜 Purchase History for {selected_user}")
        user_history = interactions_df[interactions_df['customer_id'] == selected_user]
        
        if not user_history.empty:
            # Join with products to get details
            history_enriched = pd.merge(user_history, products_df, on='product_id', how='left')
            # Select relevant columns
            cols_to_show = ['product_id', 'product_name', 'category', 'brand', 'interaction_count']
            # Handle missing columns if any
            existing_cols = [c for c in cols_to_show if c in history_enriched.columns]
            st.dataframe(history_enriched[existing_cols].style.format({"interaction_count": "{:.0f}"}))
        else:
            st.info("No recorded interaction history for this user.")
        
        st.divider()
        
        if st.button("Recommend for User"):
            # Get User Index
            user_idx = users.index(selected_user)
            
            # Predict scores for all items
            # Reconstruct interaction matrix for this user
            # svd.transform expects 2D array. We need the user's vector from the original matrix.
            user_vector = user_item_matrix.iloc[user_idx].values.reshape(1, -1)
            
            # SVD transform gives us the user embedding
            user_embedding = svd_model.transform(user_vector)
            
            # Inverse transform to get predicted scores (reconstructed row)
            predicted_scores = svd_model.inverse_transform(user_embedding)[0]
            
            # Create a DataFrame of products and their predicted scores
            predictions = pd.Series(predicted_scores, index=user_item_matrix.columns, name="score")
            predictions = predictions.sort_values(ascending=False).head(10)
            
            st.subheader(f"Top 10 Picks for {selected_user}")
            
            # Display results
            results = []
            for product_id, score in predictions.items():
                # Get product details if available
                prod_info = products_df[products_df['product_id'] == product_id]
                if not prod_info.empty:
                    name = prod_info.iloc[0].get('product_name', 'Unknown')
                    category = prod_info.iloc[0].get('category', '-')
                    brand = prod_info.iloc[0].get('brand', '-')
                    results.append({"Product ID": product_id, "Score": f"{score:.4f}", "Name": name, "Category": category, "Brand": brand})
                else:
                    results.append({"Product ID": product_id, "Score": f"{score:.4f}", "Name": "Unknown", "Category": "-", "Brand": "-"})
            
            st.table(pd.DataFrame(results))

    elif rec_type == "Content-Based Filtering":
        st.header("📦 Item-Based Recommendations (Content-Based)")
        st.write("Finding similar products based on description (Category + Brand + Name).")
        
        # Product Selection
        product_ids = products_df['product_id'].tolist()
        selected_product = st.selectbox("Select Product ID", product_ids)
        
        # Show selected product details
        sel_prod = products_df[products_df['product_id'] == selected_product].iloc[0]
        st.info(f"**Selected Product**: {sel_prod.get('product_name', 'N/A')} | {sel_prod.get('brand', 'N/A')} | {sel_prod.get('category', 'N/A')}")
        
        if st.button("Find Similar Products"):
            # Find index
            try:
                prod_idx = products_df[products_df['product_id'] == selected_product].index[0]
                
                # Get similarity scores
                sim_scores = list(enumerate(cosine_sim[prod_idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:6] # Top 5
                
                st.subheader("Similar Products")
                
                cols = st.columns(5)
                for i, (idx, score) in enumerate(sim_scores):
                    similar_prod = products_df.iloc[idx]
                    with cols[i]:
                        st.caption(f"Score: {score:.2f}")
                        st.write(f"**{similar_prod.get('product_name', 'N/A')}**")
                        st.write(f"_{similar_prod.get('brand', '-')}_")
                        st.write(f"{similar_prod.get('category', '-')}")
                        st.write(similar_prod.get('product_id'))
            except IndexError:
                st.error("Product index not found in similarity matrix.")
