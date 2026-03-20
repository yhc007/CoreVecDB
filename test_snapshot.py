#!/usr/bin/env python3
"""Test Snapshot/Backup functionality."""
import requests
import random
import time

BASE_URL = "http://localhost:3000"

def random_vector(dim=128):
    return [random.random() for _ in range(dim)]

def test_snapshot():
    print("=== Testing Snapshot/Backup ===\n")

    # Create collection
    print("1. Creating test collection...")
    resp = requests.post(f"{BASE_URL}/collections", json={
        "name": "snapshot_test",
        "dim": 128,
        "indexed_fields": ["category"],
        "numeric_fields": ["price"],
    })
    print(f"   Response: {resp.status_code}")

    # Insert test data
    print("\n2. Inserting test vectors...")
    vectors = [
        {
            "vector": random_vector(),
            "metadata": {"category": ["A", "B", "C"][i % 3], "price": str(i * 10)}
        }
        for i in range(50)
    ]
    resp = requests.post(f"{BASE_URL}/collections/snapshot_test/upsert_batch", json={"vectors": vectors})
    print(f"   Inserted: {resp.json()['count']} vectors")

    # Create snapshot
    print("\n3. Creating snapshot...")
    resp = requests.post(f"{BASE_URL}/collections/snapshot_test/snapshots")
    if resp.status_code == 200:
        snapshot = resp.json()
        print(f"   Snapshot created: {snapshot['name']}")
        print(f"   Size: {snapshot['size_bytes']} bytes")
        print(f"   Vectors: {snapshot['vector_count']}")
        snapshot_name = snapshot['name']
    else:
        print(f"   Error: {resp.status_code} - {resp.text}")
        return

    # List snapshots
    print("\n4. Listing snapshots...")
    resp = requests.get(f"{BASE_URL}/collections/snapshot_test/snapshots")
    if resp.status_code == 200:
        snapshots = resp.json()
        print(f"   Found {len(snapshots)} snapshot(s)")
        for s in snapshots:
            print(f"   - {s['name']} ({s['vector_count']} vectors)")

    # Delete original collection
    print("\n5. Deleting original collection...")
    resp = requests.delete(f"{BASE_URL}/collections/snapshot_test")
    print(f"   Deleted: {resp.status_code == 200}")

    # Verify collection is gone
    resp = requests.get(f"{BASE_URL}/collections/snapshot_test")
    print(f"   Collection exists: {resp.status_code == 200}")

    # Restore from snapshot
    print(f"\n6. Restoring from snapshot: {snapshot_name}")
    resp = requests.post(f"{BASE_URL}/snapshots/{snapshot_name}/restore", json={})
    if resp.status_code == 200:
        restored = resp.json()
        print(f"   Restored: {restored['name']}")
        print(f"   Vectors: {restored['vector_count']}")
    else:
        print(f"   Error: {resp.status_code} - {resp.text}")

    # Verify search works on restored collection
    print("\n7. Verifying search on restored collection...")
    query = random_vector()
    resp = requests.post(f"{BASE_URL}/collections/snapshot_test/search", json={
        "vector": query,
        "k": 5,
        "filter": {},
    })
    if resp.status_code == 200:
        results = resp.json()["results"]
        print(f"   Found {len(results)} results")
    else:
        print(f"   Error: {resp.status_code}")

    # Restore with new name
    print(f"\n8. Restoring with new name...")
    resp = requests.post(f"{BASE_URL}/snapshots/{snapshot_name}/restore", json={
        "new_name": "snapshot_test_copy"
    })
    if resp.status_code == 200:
        restored = resp.json()
        print(f"   Restored as: {restored['name']}")
        print(f"   Vectors: {restored['vector_count']}")

    # List all collections
    print("\n9. Listing all collections...")
    resp = requests.get(f"{BASE_URL}/collections")
    if resp.status_code == 200:
        collections = resp.json()
        for c in collections:
            print(f"   - {c['name']} ({c['vector_count']} vectors)")

    # Delete snapshot
    print(f"\n10. Deleting snapshot: {snapshot_name}")
    resp = requests.delete(f"{BASE_URL}/snapshots/{snapshot_name}")
    print(f"    Deleted: {resp.status_code == 200}")

    # Cleanup
    print("\n11. Cleaning up...")
    requests.delete(f"{BASE_URL}/collections/snapshot_test")
    requests.delete(f"{BASE_URL}/collections/snapshot_test_copy")
    print("    Done!")

if __name__ == "__main__":
    test_snapshot()
