#!/usr/bin/env python3
"""Test HTTP Range Filter API."""
import requests
import random

BASE_URL = "http://localhost:3000"

def random_vector(dim=128):
    return [random.random() for _ in range(dim)]

def test_range_filter():
    print("=== Testing HTTP Range Filter API ===\n")

    # Create collection with numeric fields
    print("1. Creating test collection with numeric fields...")
    resp = requests.post(f"{BASE_URL}/collections", json={
        "name": "range_test",
        "dim": 128,
        "indexed_fields": ["category"],
        "numeric_fields": ["price", "rating"],
    })
    print(f"   Response: {resp.status_code}")

    # Insert test vectors with metadata
    print("\n2. Inserting test vectors with price and rating...")
    vectors = []
    for i in range(100):
        vectors.append({
            "vector": random_vector(),
            "metadata": {
                "category": ["electronics", "books", "clothing"][i % 3],
                "price": str(10 + i * 5),  # 10, 15, 20, ..., 505
                "rating": str(1.0 + (i % 50) * 0.1),  # 1.0 ~ 5.9
            }
        })

    resp = requests.post(f"{BASE_URL}/collections/range_test/upsert_batch", json={"vectors": vectors})
    print(f"   Inserted: {resp.json()['count']} vectors")

    # Test range filters
    query = random_vector()

    print("\n3. Testing range filters...")

    # Test: price > 100
    print("\n   a) price > 100 (gt filter)")
    resp = requests.post(f"{BASE_URL}/collections/range_test/search", json={
        "vector": query,
        "k": 10,
        "filter": {},
        "range_filters": [
            {"op": "gt", "field": "price", "value": 100}
        ]
    })
    if resp.status_code == 200:
        results = resp.json()["results"]
        print(f"      Found {len(results)} results")

    # Test: price >= 50 AND price <= 150
    print("\n   b) 50 <= price <= 150 (range filter)")
    resp = requests.post(f"{BASE_URL}/collections/range_test/search", json={
        "vector": query,
        "k": 10,
        "filter": {},
        "range_filters": [
            {"op": "range", "field": "price", "min": 50, "max": 150}
        ]
    })
    if resp.status_code == 200:
        results = resp.json()["results"]
        print(f"      Found {len(results)} results")

    # Test: rating >= 4.0
    print("\n   c) rating >= 4.0 (gte filter)")
    resp = requests.post(f"{BASE_URL}/collections/range_test/search", json={
        "vector": query,
        "k": 10,
        "filter": {},
        "range_filters": [
            {"op": "gte", "field": "rating", "value": 4.0}
        ]
    })
    if resp.status_code == 200:
        results = resp.json()["results"]
        print(f"      Found {len(results)} results")

    # Test: Combined range filters (price > 100 AND rating >= 3.0)
    print("\n   d) price > 100 AND rating >= 3.0 (combined)")
    resp = requests.post(f"{BASE_URL}/collections/range_test/search", json={
        "vector": query,
        "k": 10,
        "filter": {},
        "range_filters": [
            {"op": "gt", "field": "price", "value": 100},
            {"op": "gte", "field": "rating", "value": 3.0}
        ]
    })
    if resp.status_code == 200:
        results = resp.json()["results"]
        print(f"      Found {len(results)} results")

    # Test: Combined with string filter (category=electronics AND price < 200)
    print("\n   e) category=electronics AND price < 200 (string + range)")
    resp = requests.post(f"{BASE_URL}/collections/range_test/search", json={
        "vector": query,
        "k": 10,
        "filter": {"category": "electronics"},
        "range_filters": [
            {"op": "lt", "field": "price", "value": 200}
        ]
    })
    if resp.status_code == 200:
        results = resp.json()["results"]
        print(f"      Found {len(results)} results")

    # Test: Between (exclusive range)
    print("\n   f) 100 < price < 200 (between filter)")
    resp = requests.post(f"{BASE_URL}/collections/range_test/search", json={
        "vector": query,
        "k": 10,
        "filter": {},
        "range_filters": [
            {"op": "between", "field": "price", "min": 100, "max": 200}
        ]
    })
    if resp.status_code == 200:
        results = resp.json()["results"]
        print(f"      Found {len(results)} results")

    # Cleanup
    print("\n4. Cleaning up...")
    requests.delete(f"{BASE_URL}/collections/range_test")
    print("   Done!")

if __name__ == "__main__":
    test_range_filter()
