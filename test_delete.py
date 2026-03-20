#!/usr/bin/env python3
"""Test Delete Vector functionality."""
import requests
import random

BASE_URL = "http://localhost:3000"

def random_vector(dim=128):
    return [random.random() for _ in range(dim)]

def test_delete():
    print("=== Testing Delete Vector ===\n")

    # Create collection
    print("1. Creating test collection...")
    resp = requests.post(f"{BASE_URL}/collections", json={
        "name": "delete_test",
        "dim": 128,
    })
    print(f"   Response: {resp.status_code}")

    # Insert test vectors
    print("\n2. Inserting test vectors...")
    vectors = [{"vector": random_vector(), "metadata": {"idx": str(i)}} for i in range(20)]
    resp = requests.post(f"{BASE_URL}/collections/delete_test/upsert_batch", json={"vectors": vectors})
    print(f"   Inserted: {resp.json()['count']} vectors")

    # Check collection info
    print("\n3. Checking collection info...")
    resp = requests.get(f"{BASE_URL}/collections/delete_test")
    info = resp.json()
    print(f"   Vectors: {info['vector_count']}, Deleted: {info['deleted_count']}")

    # Search before delete
    print("\n4. Searching before delete...")
    query = random_vector()
    resp = requests.post(f"{BASE_URL}/collections/delete_test/search", json={
        "vector": query,
        "k": 5,
        "filter": {},
    })
    results = resp.json()["results"]
    print(f"   Found {len(results)} results: {[r['id'] for r in results]}")

    # Delete single vector
    print("\n5. Deleting vector ID 5...")
    resp = requests.delete(f"{BASE_URL}/collections/delete_test/vectors/5")
    if resp.status_code == 200:
        print(f"   Deleted: {resp.json()}")
    else:
        print(f"   Error: {resp.status_code}")

    # Try to delete again (should return 410 Gone)
    print("\n6. Trying to delete ID 5 again...")
    resp = requests.delete(f"{BASE_URL}/collections/delete_test/vectors/5")
    print(f"   Status: {resp.status_code} (expected 410 Gone)")

    # Try to get deleted vector (should return 410 Gone)
    print("\n7. Trying to get deleted vector...")
    resp = requests.get(f"{BASE_URL}/collections/delete_test/vectors/5")
    print(f"   Status: {resp.status_code} (expected 410 Gone)")

    # Batch delete
    print("\n8. Batch deleting IDs 10, 11, 12...")
    resp = requests.post(f"{BASE_URL}/collections/delete_test/delete_batch", json={
        "ids": [10, 11, 12]
    })
    if resp.status_code == 200:
        print(f"   Result: {resp.json()}")

    # Check collection info after deletes
    print("\n9. Checking collection info after deletes...")
    resp = requests.get(f"{BASE_URL}/collections/delete_test")
    info = resp.json()
    print(f"   Vectors: {info['vector_count']}, Deleted: {info['deleted_count']}")

    # Search after delete (deleted vectors should not appear)
    print("\n10. Searching after delete...")
    resp = requests.post(f"{BASE_URL}/collections/delete_test/search", json={
        "vector": query,
        "k": 10,
        "filter": {},
    })
    results = resp.json()["results"]
    print(f"    Found {len(results)} results: {[r['id'] for r in results]}")

    # Verify deleted IDs are not in results
    deleted_ids = {5, 10, 11, 12}
    found_deleted = [r['id'] for r in results if r['id'] in deleted_ids]
    if found_deleted:
        print(f"    ERROR: Found deleted IDs in results: {found_deleted}")
    else:
        print(f"    OK: No deleted IDs in results")

    # Cleanup
    print("\n11. Cleaning up...")
    requests.delete(f"{BASE_URL}/collections/delete_test")
    print("    Done!")

if __name__ == "__main__":
    test_delete()
