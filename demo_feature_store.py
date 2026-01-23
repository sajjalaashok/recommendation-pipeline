from feature_registry import FeatureRegistry
import pandas as pd

def demo():
    print("="*50)
    print("   FEATURE STORE DEMO")
    print("="*50)
    
    # 1. Initialize
    fs = FeatureRegistry()
    print("\n[1] Metadata Registry Loaded")
    print(f"Project: {fs.registry['project']} v{fs.registry['version']}")
    
    # 2. Show Documentation
    print("\n[2] Available Features (Documentation)")
    print(fs.list_features().to_string(index=False))
    
    # 3. Retrieval Scenarios
    
    # Scenario A: User Inference (Get features for User CUST001)
    print("\n[3] Retrieve Features for Inference: Customer 'CUST001'")
    user_features = fs.get_online_features({'customer_id': ['CUST001']})
    print(user_features['customer_features'])
    
    # Scenario B: Product Ranking (Get features for Products P00001, P00002)
    print("\n[4] Retrieve Features for Products 'P00001', 'P00002'")
    prod_features = fs.get_online_features({'product_id': ['P00001', 'P00002']})
    print(prod_features['product_features'])
    
    # Scenario C: Recommendations (Get Co-occurrence for P00001)
    print("\n[5] Retrieve Co-occurrence (Recommendations) for 'P00001'")
    recs = fs.get_co_occurrence('P00001')
    if not recs.empty:
        print(recs)
    else:
        print("No co-occurrence data found for P00001 (might be low volume in demo data)")

if __name__ == "__main__":
    demo()
