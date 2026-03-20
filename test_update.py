#!/usr/bin/env python3
"""Test Update Vector functionality."""
import requests
import random

BASE_URL = "http://localhost:3000"

def random_vector(dim=128):
    return [random.random() for _ in range(dim)]

def test_update():
    print("=== Testing Update Vector ===\n")

    # Create collection
    print("1. Creating test collection...")
    resp = requests.post(f"{BASE_URL}/collections", json={
        "name": "update_test",
        "dim": 128,
        "indexed_fields": ["category"],
    })
    print(f"   Response: {resp.status_code}")

    # Insert test vectors
    print("\n2. Inserting test vectors...")
    vectors = [
        {"vector": random_vector(), "metadata": {"category": "A", "name": f"item_{i}"}}
        for i in range(10)
    ]
    resp = requests.post(f"{BASE_URL}/collections/update_test/upsert_batch", json={"vectors": vectors})
    print(f"   Inserted: {resp.json()['count']} vectors")

    # Get original vector
    print("\n3. Getting original vector (ID 5)...")
    resp = requests.get(f"{BASE_URL}/collections/update_test/vectors/5")
    if resp.status_code == 200:
        original = resp.json()
        print(f"   Found: ID={original['id']}")

    # Update vector (PUT - full update with new vector)
    print("\n4. Updating vector (PUT - full update)...")
    new_vector = random_vector()
    resp = requests.put(f"{BASE_URL}/collections/update_test/vectors/5", json={
        "vector": new_vector,
        "metadata": {"category": "B", "name": "updated_item"}
    })
    if resp.status_code == 200:
        result = resp.json()
        print(f"   Result: old_id={result['old_id']}, new_id={result['new_id']}")
    else:
        print(f"   Error: {resp.status_code} - {resp.text}")
        return

    # Try to get old ID (should be 410 Gone)
    print("\n5. Trying to get old ID (should be 410 Gone)...")
    resp = requests.get(f"{BASE_URL}/collections/update_test/vectors/5")
    print(f"   Status: {resp.status_code}")

    # Get new ID
    new_id = result['new_id']
    print(f"\n6. Getting new ID ({new_id})...")
    resp = requests.get(f"{BASE_URL}/collections/update_test/vectors/{new_id}")
    if resp.status_code == 200:
        print(f"   Found: ID={resp.json()['id']}")

    # Update metadata only (PATCH)
    print(f"\n7. Updating metadata only (PATCH ID {new_id})...")
    resp = requests.patch(f"{BASE_URL}/collections/update_test/vectors/{new_id}", json={
        "metadata": {"category": "C", "priority": "high"}
    })
    if resp.status_code == 200:
        print(f"   Result: {resp.json()}")

    # Check collection info
    print("\n8. Checking collection info...")
    resp = requests.get(f"{BASE_URL}/collections/update_test")
    info = resp.json()
    print(f"   Vectors: {info['vector_count']}, Deleted: {info['deleted_count']}")
    print(f"   (Original: 10, After update: 10 active, 1 deleted)")

    # Search to verify new vector is found
    print("\n9. Searching with category=C filter...")
    resp = requests.post(f"{BASE_URL}/collections/update_test/search", json={
        "vector": new_vector,
        "k": 5,
        "filter": {"category": "C"},
    })
    if resp.status_code == 200:
        results = resp.json()["results"]
        print(f"   Found {len(results)} results with category=C")
        if results:
            print(f"   First result: ID={results[0]['id']}, score={results[0]['score']:.4f}")

    # Cleanup
    print("\n10. Cleaning up...")
    requests.delete(f"{BASE_URL}/collections/update_test")
    print("    Done!")

if __name__ == "__main__":
    test_update()
