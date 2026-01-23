import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    print(f"Testing API at {BASE_URL}...")
    
    # 1. Health Check
    try:
        r = requests.get(f"{BASE_URL}/")
        print(f"[Health] Status: {r.status_code}, Response: {r.json()}")
    except Exception as e:
        print(f"[Health] Failed to connect: {e}")
        return

    # 2. Get Features (Known User/Product)
    # We use CUST001 / P00002 from previous steps
    user_id = "CUST001"
    r = requests.get(f"{BASE_URL}/features/user/{user_id}")
    print(f"\n[Features][User] {user_id}: {r.status_code}")
    if r.status_code == 200:
        print(r.json())
        
    prod_id = "P00002"
    r = requests.get(f"{BASE_URL}/features/product/{prod_id}")
    print(f"\n[Features][Product] {prod_id}: {r.status_code}")
    if r.status_code == 200:
        print(r.json())

    # 3. Recommendations (User)
    print(f"\n[Recommend][User] {user_id}:")
    r = requests.get(f"{BASE_URL}/recommend/user/{user_id}?k=3")
    if r.status_code == 200:
        print(r.json())
    else:
        print(f"Error: {r.text}")

    # 4. Recommendations (Item Co-occurrence)
    # P00002 likely has data? Or maybe not, checking...
    print(f"\n[Recommend][Item] {prod_id}:")
    r = requests.get(f"{BASE_URL}/recommend/item/{prod_id}")
    if r.status_code == 200:
        print(r.json())

if __name__ == "__main__":
    # Wait a bit for server to possibly start if running in CI/pipeline (manual run for us)
    test_api()
