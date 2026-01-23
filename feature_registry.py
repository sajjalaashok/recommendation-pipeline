import yaml
import sqlite3
import pandas as pd
from pathlib import Path

class FeatureRegistry:
    def __init__(self, metadata_path="feature_definitions.yaml", db_path="model_store/feature_store.db"):
        self.metadata_path = metadata_path
        self.db_path = db_path
        self.registry = self._load_registry()
        
    def _load_registry(self):
        """Loads feature metadata from YAML."""
        if not Path(self.metadata_path).exists():
            raise FileNotFoundError(f"Registry file {self.metadata_path} not found.")
        
        with open(self.metadata_path, 'r') as f:
            return yaml.safe_load(f)

    def list_features(self):
        """Returns a list of all available features and their descriptions."""
        features = []
        for view in self.registry.get('feature_views', []):
            for feat in view.get('features', []):
                features.append({
                    "view": view['name'],
                    "name": feat['name'],
                    "type": feat['type'],
                    "description": feat['description']
                })
        return pd.DataFrame(features)

    def get_online_features(self, entity_rows: dict):
        """
        Retrieves features for specific entities.
        Mimics Feast API: get_online_features(features, entity_rows)
        
        Args:
            entity_rows: Dict with format {'customer_id': ['C001', 'C002'], 'product_id': ['P1']}
                         Currently supports fetching one entity type at a time.
        """
        conn = sqlite3.connect(self.db_path)
        
        results = {}
        
        # 1. Customer Features
        if 'customer_id' in entity_rows:
            ids = tuple(entity_rows['customer_id'])
            if len(ids) == 1:
                ids = f"('{ids[0]}')" 
            
            # Find relevant views
            # Assuming 'user_activity_stats' is the view for customers
            query = f"SELECT customer_id, user_txn_count, user_avg_score FROM features WHERE customer_id IN {ids}"
            df = pd.read_sql(query, conn)
            results['customer_features'] = df
            
        # 2. Product Features
        if 'product_id' in entity_rows:
            ids = tuple(entity_rows['product_id'])
            if len(ids) == 1:
                ids = f"('{ids[0]}')"
            
            # Assuming 'product_metadata' is the view
            query = f"SELECT product_id, item_avg_score, sentiment_score, popularity_index FROM product_metadata WHERE product_id IN {ids}"
            df = pd.read_sql(query, conn)
            results['product_features'] = df
            
        conn.close()
        return results

    def get_co_occurrence(self, product_id, limit=5):
        """Special retrieval for recommendation features."""
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT product_b, frequency 
            FROM co_occurrence 
            WHERE product_a = '{product_id}' 
            ORDER BY frequency DESC 
            LIMIT {limit}
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df

if __name__ == "__main__":
    # Test basic loading
    fs = FeatureRegistry()
    print("Feature Registry Loaded Version:", fs.registry['version'])
    print(fs.list_features().to_markdown())
