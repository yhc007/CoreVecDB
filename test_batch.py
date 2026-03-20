#!/usr/bin/env python3
"""Test batch upsert functionality."""
import requests
import random
import time

BASE_URL = "http://localhost:3000"

def random_vector(dim=128):
    return [random.random() for _ in range(dim)]

def test_batch_upsert():
    print("=== Testing Batch Upsert ===\n")

    # Create collection
    print("1. Creating test collection...")
    resp = requests.post(f"{BASE_URL}/collections", json={
        "name": "batch_test",
        "dim": 128,
    })
    print(f"   Response: {resp.status_code}")

    # Test batch sizes
    batch_sizes = [10, 100, 500]

    for batch_size in batch_sizes:
        print(f"\n2. Testing batch upsert with {batch_size} vectors...")

        vectors = [
            {
                "vector": random_vector(),
                "metadata": {"batch": str(batch_size), "idx": str(i)}
            }
            for i in range(batch_size)
        ]

        start = time.time()
        resp = requests.post(
            f"{BASE_URL}/collections/batch_test/upsert_batch",
            json={"vectors": vectors}
        )
        elapsed = time.time() - start

        if resp.status_code == 200:
            data = resp.json()
            print(f"   Success: start_id={data['start_id']}, count={data['count']}")
            print(f"   Time: {elapsed:.3f}s ({batch_size/elapsed:.0f} vectors/sec)")
        else:
            print(f"   Error: {resp.status_code} - {resp.text}")

    # Verify search works
    print("\n3. Verifying search on batch-inserted vectors...")
    query = random_vector()
    resp = requests.post(f"{BASE_URL}/collections/batch_test/search", json={
        "vector": query,
        "k": 5,
        "filter": {},
    })

    if resp.status_code == 200:
        results = resp.json()["results"]
        print(f"   Found {len(results)} results")
        for r in results[:3]:
            print(f"   - id={r['id']}, score={r['score']:.4f}")
    else:
        print(f"   Error: {resp.status_code}")

    # Cleanup
    print("\n4. Cleaning up...")
    requests.delete(f"{BASE_URL}/collections/batch_test")
    print("   Done!")

    # Compare single vs batch performance
    print("\n=== Performance Comparison ===")

    # Create fresh collection
    requests.post(f"{BASE_URL}/collections", json={"name": "perf_test", "dim": 128})

    n = 100

    # Single inserts
    print(f"\nSingle inserts ({n} vectors)...")
    start = time.time()
    for i in range(n):
        requests.post(f"{BASE_URL}/collections/perf_test/upsert", json={
            "id": i,
            "vector": random_vector(),
            "metadata": {"idx": str(i)},
        })
    single_time = time.time() - start
    print(f"   Time: {single_time:.3f}s ({n/single_time:.0f} vectors/sec)")

    # Batch insert
    print(f"\nBatch insert ({n} vectors)...")
    vectors = [{"vector": random_vector(), "metadata": {"idx": str(i)}} for i in range(n)]
    start = time.time()
    requests.post(f"{BASE_URL}/collections/perf_test/upsert_batch", json={"vectors": vectors})
    batch_time = time.time() - start
    print(f"   Time: {batch_time:.3f}s ({n/batch_time:.0f} vectors/sec)")

    speedup = single_time / batch_time if batch_time > 0 else float('inf')
    print(f"\n   Speedup: {speedup:.1f}x faster with batch upsert!")

    requests.delete(f"{BASE_URL}/collections/perf_test")

if __name__ == "__main__":
    test_batch_upsert()
