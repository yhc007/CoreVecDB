#!/usr/bin/env python3
"""Benchmark: filtered vs unfiltered search on CoreVecDB (collection API, port 3100)."""

import urllib.request
import json
import random
import time
import statistics
import concurrent.futures

URL = "http://localhost:3100"
COLLECTION = "bench_filter"
DIM = 128
NUM_VECTORS = 2000
SEARCH_ITERS = 500

def rand_vec():
    return [random.gauss(0, 1) for _ in range(DIM)]

def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{URL}{path}", data=data,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def get(path):
    with urllib.request.urlopen(f"{URL}{path}") as r:
        return json.loads(r.read())

def delete(path):
    req = urllib.request.Request(f"{URL}{path}", method="DELETE")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status
    except:
        return 0

def bench(name, fn, n):
    lats = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        lats.append(time.perf_counter() - t0)
    ops = n / sum(lats)
    avg = statistics.mean(lats) * 1000
    p50 = statistics.median(lats) * 1000
    p99 = sorted(lats)[int(len(lats) * 0.99)] * 1000 if len(lats) >= 100 else sorted(lats)[-1] * 1000
    print(f"  {name:45s}  {ops:8.1f} ops/s  avg={avg:6.2f}ms  p50={p50:6.2f}ms  p99={p99:6.2f}ms")
    return ops

def bench_concurrent(name, fn, n, threads):
    lats = []
    def worker():
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(worker) for _ in range(n)]
        for f in concurrent.futures.as_completed(futs):
            lats.append(f.result())
    total = time.perf_counter() - t_start
    ops = n / total
    avg = statistics.mean(lats) * 1000
    p50 = statistics.median(lats) * 1000
    p99 = sorted(lats)[int(len(lats) * 0.99)] * 1000 if len(lats) >= 100 else sorted(lats)[-1] * 1000
    print(f"  {name:45s}  {ops:8.1f} ops/s  avg={avg:6.2f}ms  p50={p50:6.2f}ms  p99={p99:6.2f}ms")
    return ops

def main():
    random.seed(42)
    print("=" * 90)
    print("CoreVecDB Filter Performance Benchmark")
    print("=" * 90)

    # Cleanup old bench collection
    delete(f"/collections/{COLLECTION}")

    # Create collection with indexed fields
    post("/collections", {
        "name": COLLECTION,
        "dim": DIM,
        "distance": "cosine",
        "indexed_fields": ["category", "status"],
        "numeric_fields": ["price", "rating"],
    })

    categories = ["electronics", "books", "clothing", "food", "sports",
                   "toys", "health", "automotive", "garden", "office"]
    statuses = ["available", "sold", "pending"]

    # Insert vectors
    print(f"\nInserting {NUM_VECTORS} vectors...")
    batch_vecs = []
    for i in range(NUM_VECTORS):
        batch_vecs.append({
            "vector": rand_vec(),
            "metadata": {
                "category": categories[i % len(categories)],
                "status": statuses[i % len(statuses)],
                "name": f"item_{i}",
                "price": str(round(random.uniform(1, 1000), 2)),
                "rating": str(random.randint(1, 5)),
            }
        })

    # Batch upsert in chunks of 100
    t0 = time.perf_counter()
    for start in range(0, NUM_VECTORS, 100):
        chunk = batch_vecs[start:start+100]
        post(f"/collections/{COLLECTION}/upsert_batch", {
            "vectors": [{"vector": v["vector"], "metadata": v["metadata"]} for v in chunk]
        })
    insert_time = time.perf_counter() - t0
    print(f"  Inserted {NUM_VECTORS} vectors in {insert_time:.2f}s ({NUM_VECTORS/insert_time:.0f} vec/s)")

    # Verify
    info = get(f"/collections/{COLLECTION}")
    print(f"  Collection: {info['vector_count']} vectors, {info['dim']}d\n")

    query = rand_vec()

    # --- Sequential Benchmarks ---
    print("-" * 90)
    print("SEQUENTIAL SEARCH (single-thread)")
    print("-" * 90)

    # 1. No filter
    no_filter_ops = bench("No filter", lambda: post(f"/collections/{COLLECTION}/search", {
        "vector": query, "k": 10, "filter": {}, "include_metadata": False
    }), SEARCH_ITERS)

    # 2. Single indexed filter (category=electronics → ~10% selectivity)
    idx1_ops = bench("Indexed filter (category=electronics)", lambda: post(f"/collections/{COLLECTION}/search", {
        "vector": query, "k": 10,
        "filter": {"category": "electronics"}, "include_metadata": False
    }), SEARCH_ITERS)

    # 3. Multi indexed filter (category + status → ~3.3% selectivity)
    idx2_ops = bench("Multi indexed filter (cat+status)", lambda: post(f"/collections/{COLLECTION}/search", {
        "vector": query, "k": 10,
        "filter": {"category": "electronics", "status": "available"}, "include_metadata": False
    }), SEARCH_ITERS)

    # 4. Non-indexed filter (name → post-filter)
    noidx_ops = bench("Non-indexed filter (name=item_5)", lambda: post(f"/collections/{COLLECTION}/search", {
        "vector": query, "k": 10,
        "filter": {"name": "item_5"}, "include_metadata": False
    }), SEARCH_ITERS)

    # 5. filter_ids (500 IDs → 25% selectivity)
    ids_500 = list(range(0, 500))
    fid_ops = bench("filter_ids (500 IDs)", lambda: post(f"/collections/{COLLECTION}/search", {
        "vector": query, "k": 10, "filter": {}, "filter_ids": ids_500, "include_metadata": False
    }), SEARCH_ITERS)

    # 6. filter_id_range
    frange_ops = bench("filter_id_range [0, 499]", lambda: post(f"/collections/{COLLECTION}/search", {
        "vector": query, "k": 10, "filter": {}, "filter_id_range": [0, 499], "include_metadata": False
    }), SEARCH_ITERS)

    # 7. With metadata retrieval
    meta_ops = bench("Indexed filter + include_metadata", lambda: post(f"/collections/{COLLECTION}/search", {
        "vector": query, "k": 10,
        "filter": {"category": "electronics"}, "include_metadata": True
    }), SEARCH_ITERS)

    # --- Concurrent Benchmarks ---
    print(f"\n{'-' * 90}")
    print("CONCURRENT SEARCH (20 threads)")
    print("-" * 90)

    c_no = bench_concurrent("No filter", lambda: post(f"/collections/{COLLECTION}/search", {
        "vector": query, "k": 10, "filter": {}, "include_metadata": False
    }), 1000, 20)

    c_idx = bench_concurrent("Indexed filter (category)", lambda: post(f"/collections/{COLLECTION}/search", {
        "vector": query, "k": 10,
        "filter": {"category": "electronics"}, "include_metadata": False
    }), 1000, 20)

    c_noidx = bench_concurrent("Non-indexed filter (name)", lambda: post(f"/collections/{COLLECTION}/search", {
        "vector": query, "k": 10,
        "filter": {"name": "item_5"}, "include_metadata": False
    }), 1000, 20)

    # --- Mixed workload ---
    print(f"\n{'-' * 90}")
    print("MIXED WORKLOAD (90% filtered search, 10% upsert, 20 threads)")
    print("-" * 90)

    mix_counter = [NUM_VECTORS]
    def mixed():
        if random.random() < 0.1:
            i = mix_counter[0]
            mix_counter[0] += 1
            post(f"/collections/{COLLECTION}/upsert_batch", {
                "vectors": [{"vector": rand_vec(), "metadata": {
                    "category": random.choice(categories),
                    "status": random.choice(statuses),
                }}]
            })
        else:
            post(f"/collections/{COLLECTION}/search", {
                "vector": rand_vec(), "k": 10,
                "filter": {"category": random.choice(categories)},
                "include_metadata": False
            })

    bench_concurrent("Mixed workload", mixed, 1000, 20)

    # --- Summary ---
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    print(f"  No filter baseline:       {no_filter_ops:8.1f} ops/s")
    print(f"  Indexed filter:           {idx1_ops:8.1f} ops/s  ({idx1_ops/no_filter_ops*100:.0f}% of baseline)")
    print(f"  Multi indexed filter:     {idx2_ops:8.1f} ops/s  ({idx2_ops/no_filter_ops*100:.0f}% of baseline)")
    print(f"  Non-indexed filter:       {noidx_ops:8.1f} ops/s  ({noidx_ops/no_filter_ops*100:.0f}% of baseline)")
    print(f"  filter_ids (500):         {fid_ops:8.1f} ops/s  ({fid_ops/no_filter_ops*100:.0f}% of baseline)")
    print(f"  filter_id_range:          {frange_ops:8.1f} ops/s  ({frange_ops/no_filter_ops*100:.0f}% of baseline)")
    print(f"  Concurrent no filter:     {c_no:8.1f} ops/s")
    print(f"  Concurrent indexed:       {c_idx:8.1f} ops/s  ({c_idx/c_no*100:.0f}% of concurrent baseline)")
    print(f"  Concurrent non-indexed:   {c_noidx:8.1f} ops/s  ({c_noidx/c_no*100:.0f}% of concurrent baseline)")
    print(f"{'=' * 90}")

    # Cleanup
    delete(f"/collections/{COLLECTION}")

if __name__ == "__main__":
    main()
