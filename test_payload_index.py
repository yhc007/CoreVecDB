#!/usr/bin/env python3
"""Test Payload Index filtering functionality."""

import requests
import random
import time
import subprocess
import sys
import os

BASE_URL = "http://localhost:3000"
DIM = 128

def wait_for_server(max_wait=10):
    """Wait for server to be ready."""
    for _ in range(max_wait):
        try:
            r = requests.get(f"{BASE_URL}/stats", timeout=1)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def upsert(id, vector, metadata):
    """Upsert a vector with metadata."""
    payload = {
        "id": id,
        "vector": vector,
        "metadata": metadata
    }
    r = requests.post(f"{BASE_URL}/upsert", json=payload)
    return r.status_code == 200

def search(vector, k, filter_dict=None, filter_ids=None):
    """Search with optional filters."""
    payload = {
        "vector": vector,
        "k": k,
        "filter": filter_dict or {}
    }
    if filter_ids:
        payload["filter_ids"] = filter_ids

    r = requests.post(f"{BASE_URL}/search", json=payload)
    if r.status_code == 200:
        return r.json().get("results", [])
    else:
        print(f"Search failed: {r.text}")
        return []

def main():
    print("=" * 60)
    print("Payload Index Test")
    print("=" * 60)

    # Check server
    print("\n1. Checking server...")
    if not wait_for_server():
        print("Server not available. Start with: cargo run --release")
        sys.exit(1)
    print("Server is ready.")

    # Insert test data with indexed fields (category, type, status)
    print("\n2. Inserting test vectors with metadata...")
    categories = ["electronics", "books", "clothing"]
    types = ["new", "used", "refurbished"]
    statuses = ["available", "sold", "pending"]

    vectors = {}
    for i in range(30):
        vec = [random.random() for _ in range(DIM)]
        metadata = {
            "category": categories[i % 3],
            "type": types[i % 3],
            "status": statuses[i % 3],
            "name": f"product_{i}"  # Not indexed
        }
        if upsert(i, vec, metadata):
            vectors[i] = (vec, metadata)
        else:
            print(f"  Failed to insert {i}")

    print(f"  Inserted {len(vectors)} vectors")

    # Wait for indexing
    time.sleep(0.5)

    # Test 1: Search without filter
    print("\n3. Test: Search without filter")
    query = vectors[0][0]
    results = search(query, k=5)
    print(f"  Results: {[r['id'] for r in results]}")

    # Test 2: Filter by indexed field (category)
    print("\n4. Test: Filter by category='electronics' (indexed)")
    results = search(query, k=10, filter_dict={"category": "electronics"})
    ids = [r['id'] for r in results]
    print(f"  Results: {ids}")

    # Verify all results have category=electronics
    expected_ids = [i for i in range(30) if i % 3 == 0]  # 0, 3, 6, 9, ...
    valid = all(id in expected_ids for id in ids)
    print(f"  Valid (all electronics): {valid}")

    # Test 3: Filter by two indexed fields (AND)
    print("\n5. Test: Filter by category='electronics' AND status='available'")
    results = search(query, k=10, filter_dict={
        "category": "electronics",
        "status": "available"
    })
    ids = [r['id'] for r in results]
    print(f"  Results: {ids}")

    # category=electronics -> i%3==0 -> 0,3,6,9,...
    # status=available -> i%3==0 -> 0,3,6,9,...
    # Both: 0,3,6,9,...
    expected = [i for i in range(30) if i % 3 == 0]
    valid = all(id in expected for id in ids)
    print(f"  Valid: {valid}")

    # Test 4: Filter by non-indexed field (should use post-filter)
    print("\n6. Test: Filter by name='product_5' (NOT indexed)")
    results = search(query, k=10, filter_dict={"name": "product_5"})
    ids = [r['id'] for r in results]
    print(f"  Results: {ids}")
    print(f"  Valid (should be [5] or empty): {ids == [5] or ids == []}")

    # Test 5: Combined indexed filter + filter_ids
    print("\n7. Test: Filter by category='books' AND filter_ids=[1,4,7,10]")
    results = search(query, k=10, filter_dict={"category": "books"}, filter_ids=[1,4,7,10])
    ids = [r['id'] for r in results]
    print(f"  Results: {ids}")
    # category=books -> i%3==1 -> 1,4,7,10,13,...
    # filter_ids -> [1,4,7,10]
    # Intersection: [1,4,7,10]
    expected = [1, 4, 7, 10]
    valid = all(id in expected for id in ids)
    print(f"  Valid: {valid}")

    # Summary
    print("\n" + "=" * 60)
    print("Payload Index Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
