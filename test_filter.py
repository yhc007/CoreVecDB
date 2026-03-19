import requests
import time
import random

BASE_URL = "http://localhost:3000"

def upsert(id, vector):
    payload = {
        "id": id,
        "vector": vector,
        "metadata": {"key": f"val_{id}"}
    }
    r = requests.post(f"{BASE_URL}/upsert", json=payload)
    if r.status_code != 200:
        print(f"Failed to upsert {id}: {r.text}")
    return r.json()

def search(vector, k, filter_ids=None):
    payload = {
        "vector": vector,
        "k": k,
        "filter": {}
    }
    if filter_ids:
        payload["filter_ids"] = filter_ids
    
    r = requests.post(f"{BASE_URL}/search", json=payload)
    if r.status_code != 200:
        print(f"Failed to search: {r.text}")
        return []
    return r.json().get("results", [])

def main():
    print("Wait for server...")
    time.sleep(2)
    
    # 1. Upsert 10 vectors
    print("Upserting 10 vectors...")
    dim = 128
    vectors = {}
    for i in range(1, 11):
        vec = [random.random() for _ in range(dim)]
        upsert(i, vec)
        vectors[i] = vec
        
    print("Upsert complete.")
    
    # 2. Search without filter
    print("Searching without filter...")
    query_vec = vectors[1] # Should match 1 closely
    results = search(query_vec, k=5)
    ids = [r['id'] for r in results]
    print(f"Results (no filter): {ids}")
    
    # 3. Search with filter
    print("Searching with filter_ids=[2, 4, 6]...")
    # Even if 1 is the closest, it should not be returned.
    results = search(query_vec, k=5, filter_ids=[2, 4, 6])
    ids = [r['id'] for r in results]
    print(f"Results (filter=[2, 4, 6]): {ids}")
    
    success = all(id in [2, 4, 6] for id in ids)
    if success and len(ids) > 0:
        print("PASS: Filtered search successful.")
    elif len(ids) == 0:
        print("WARN: No results found (might be distance too far? or just empty intersection)")
    else:
        print("FAIL: Found unexpected IDs.")

if __name__ == "__main__":
    main()
