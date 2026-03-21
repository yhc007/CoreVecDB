#!/usr/bin/env python3
"""Test Compaction functionality."""
import requests
import random

BASE_URL = "http://localhost:3000"

def random_vector(dim=128):
    return [random.random() for _ in range(dim)]

def test_compaction():
    print("=== Testing Compaction ===\n")

    # Create collection
    print("1. Creating test collection...")
    resp = requests.post(f"{BASE_URL}/collections", json={
        "name": "compact_test",
        "dim": 128,
        "indexed_fields": ["category"],
    })
    print(f"   Response: {resp.status_code}")

    # Insert vectors
    print("\n2. Inserting 1000 test vectors...")
    vectors = [
        {"vector": random_vector(), "metadata": {"category": f"cat_{i % 10}", "idx": str(i)}}
        for i in range(1000)
    ]
    resp = requests.post(f"{BASE_URL}/collections/compact_test/upsert_batch", json={"vectors": vectors})
    print(f"   Inserted: {resp.json()['count']} vectors")

    # Check initial state
    print("\n3. Initial collection state...")
    resp = requests.get(f"{BASE_URL}/collections/compact_test")
    info = resp.json()
    print(f"   Vectors: {info['vector_count']}, Deleted: {info['deleted_count']}")

    # Delete 300 vectors (30%)
    print("\n4. Deleting 300 vectors (IDs 100-399)...")
    ids_to_delete = list(range(100, 400))
    resp = requests.post(f"{BASE_URL}/collections/compact_test/delete_batch", json={"ids": ids_to_delete})
    print(f"   Deleted: {resp.json()['deleted_count']} vectors")

    # Check state after delete
    print("\n5. Collection state after delete...")
    resp = requests.get(f"{BASE_URL}/collections/compact_test")
    info = resp.json()
    print(f"   Vectors: {info['vector_count']}, Deleted: {info['deleted_count']}")

    # Search before compaction
    print("\n6. Searching before compaction...")
    resp = requests.post(f"{BASE_URL}/collections/compact_test/search", json={
        "vector": random_vector(),
        "k": 5,
    })
    if resp.status_code == 200:
        results = resp.json()["results"]
        print(f"   Found {len(results)} results")
        print(f"   IDs: {[r['id'] for r in results]}")

    # Compact
    print("\n7. Running compaction...")
    resp = requests.post(f"{BASE_URL}/collections/compact_test/compact")
    if resp.status_code == 200:
        result = resp.json()
        print(f"   Vectors before: {result['vectors_before']}")
        print(f"   Vectors after: {result['vectors_after']}")
        print(f"   Vectors removed: {result['vectors_removed']}")
        print(f"   Bytes reclaimed: {result['bytes_reclaimed']}")
    else:
        print(f"   Error: {resp.status_code} - {resp.text}")
        return

    # Check state after compaction
    print("\n8. Collection state after compaction...")
    resp = requests.get(f"{BASE_URL}/collections/compact_test")
    info = resp.json()
    print(f"   Vectors: {info['vector_count']}, Deleted: {info['deleted_count']}")

    # Search after compaction
    print("\n9. Searching after compaction...")
    resp = requests.post(f"{BASE_URL}/collections/compact_test/search", json={
        "vector": random_vector(),
        "k": 5,
    })
    if resp.status_code == 200:
        results = resp.json()["results"]
        print(f"   Found {len(results)} results")
        print(f"   IDs: {[r['id'] for r in results]}")
        print("   (Note: IDs are now 0-699 instead of 0-999)")

    # Verify ID 0 exists (should exist after compaction)
    print("\n10. Verifying vectors exist after compaction...")
    resp = requests.get(f"{BASE_URL}/collections/compact_test/vectors/0")
    if resp.status_code == 200:
        print(f"   Vector 0: Found")
    else:
        print(f"   Vector 0: {resp.status_code}")

    # Cleanup
    print("\n11. Cleaning up...")
    requests.delete(f"{BASE_URL}/collections/compact_test")
    print("    Done!")

def test_compaction_no_deletes():
    """Test that compaction does nothing when there are no deleted vectors."""
    print("\n=== Testing Compaction (No Deletes) ===\n")

    print("1. Creating collection with 100 vectors...")
    requests.post(f"{BASE_URL}/collections", json={
        "name": "compact_no_delete",
        "dim": 128,
    })

    vectors = [{"vector": random_vector(), "metadata": {}} for _ in range(100)]
    requests.post(f"{BASE_URL}/collections/compact_no_delete/upsert_batch", json={"vectors": vectors})

    print("2. Running compaction (no deletes)...")
    resp = requests.post(f"{BASE_URL}/collections/compact_no_delete/compact")
    if resp.status_code == 200:
        result = resp.json()
        print(f"   Vectors before: {result['vectors_before']}")
        print(f"   Vectors after: {result['vectors_after']}")
        print(f"   Vectors removed: {result['vectors_removed']}")
        if result['vectors_removed'] == 0:
            print("   Correctly skipped compaction (no deletes)")

    print("3. Cleaning up...")
    requests.delete(f"{BASE_URL}/collections/compact_no_delete")
    print("   Done!")

if __name__ == "__main__":
    test_compaction()
    test_compaction_no_deletes()
