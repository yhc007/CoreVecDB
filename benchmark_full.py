#!/usr/bin/env python3
"""Full benchmark including Payload Index filtering."""

import urllib.request
import json
import random
import time
import statistics
import concurrent.futures

URL = "http://localhost:3000"
DIM = 128

def generate_vector():
    return [random.random() for _ in range(DIM)]

def http_post(endpoint, payload):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"{URL}{endpoint}", data=data,
                                  headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as f:
        return json.loads(f.read())

def http_get(endpoint):
    with urllib.request.urlopen(f"{URL}{endpoint}") as f:
        return json.loads(f.read())

def run_benchmark(name, fn, iterations):
    """Run benchmark and return stats."""
    latencies = []
    start = time.time()

    for _ in range(iterations):
        req_start = time.time()
        fn()
        latencies.append(time.time() - req_start)

    total = time.time() - start
    return {
        "name": name,
        "iterations": iterations,
        "total_time": total,
        "throughput": iterations / total,
        "avg_latency_ms": statistics.mean(latencies) * 1000,
        "p50_ms": statistics.median(latencies) * 1000,
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)] * 1000 if len(latencies) >= 100 else latencies[-1] * 1000
    }

def concurrent_benchmark(name, fn, iterations, threads):
    """Run benchmark with concurrent workers."""
    latencies = []
    start = time.time()

    def worker():
        req_start = time.time()
        fn()
        return time.time() - req_start

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(worker) for _ in range(iterations)]
        for f in concurrent.futures.as_completed(futures):
            latencies.append(f.result())

    total = time.time() - start
    return {
        "name": f"{name} ({threads} threads)",
        "iterations": iterations,
        "total_time": total,
        "throughput": iterations / total,
        "avg_latency_ms": statistics.mean(latencies) * 1000,
        "p50_ms": statistics.median(latencies) * 1000,
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)] * 1000 if len(latencies) >= 100 else latencies[-1] * 1000
    }

def print_result(result):
    print(f"\n  {result['name']}:")
    print(f"    Throughput: {result['throughput']:.2f} ops/sec")
    print(f"    Avg Latency: {result['avg_latency_ms']:.2f} ms")
    print(f"    P50: {result['p50_ms']:.2f} ms | P99: {result['p99_ms']:.2f} ms")

def main():
    print("=" * 60)
    print("VectorDB Full Benchmark")
    print("=" * 60)

    # Check server
    stats = http_get("/stats")
    print(f"\nServer Status: {stats['status']}")
    print(f"Initial Vector Count: {stats['vector_count']}")

    # Categories for payload index testing
    categories = ["electronics", "books", "clothing", "food", "sports"]
    statuses = ["available", "sold", "pending"]

    # 1. Upsert Benchmark
    print("\n" + "-" * 40)
    print("1. UPSERT BENCHMARK")
    print("-" * 40)

    vectors = []
    def upsert_fn():
        i = len(vectors)
        vec = generate_vector()
        vectors.append(vec)
        return http_post("/upsert", {
            "id": i,
            "vector": vec,
            "metadata": {
                "category": categories[i % len(categories)],
                "status": statuses[i % len(statuses)],
                "name": f"item_{i}"
            }
        })

    result = run_benchmark("Sequential Upsert", upsert_fn, 1000)
    print_result(result)

    # 2. Search Benchmarks
    print("\n" + "-" * 40)
    print("2. SEARCH BENCHMARKS")
    print("-" * 40)

    query_vec = generate_vector()

    # Search without filter
    def search_no_filter():
        return http_post("/search", {"vector": query_vec, "k": 10, "filter": {}})

    result = run_benchmark("Search (no filter)", search_no_filter, 200)
    print_result(result)

    # Search with indexed filter
    def search_indexed_filter():
        return http_post("/search", {
            "vector": query_vec,
            "k": 10,
            "filter": {"category": "electronics"}
        })

    result = run_benchmark("Search + Indexed Filter (category)", search_indexed_filter, 200)
    print_result(result)

    # Search with multiple indexed filters
    def search_multi_filter():
        return http_post("/search", {
            "vector": query_vec,
            "k": 10,
            "filter": {"category": "electronics", "status": "available"}
        })

    result = run_benchmark("Search + Multi Indexed Filter", search_multi_filter, 200)
    print_result(result)

    # Search with non-indexed filter (post-filter)
    def search_nonindexed_filter():
        return http_post("/search", {
            "vector": query_vec,
            "k": 10,
            "filter": {"name": "item_5"}
        })

    result = run_benchmark("Search + Non-indexed Filter (name)", search_nonindexed_filter, 200)
    print_result(result)

    # Search with filter_ids
    def search_filter_ids():
        return http_post("/search", {
            "vector": query_vec,
            "k": 10,
            "filter": {},
            "filter_ids": list(range(0, 500))
        })

    result = run_benchmark("Search + filter_ids (500 IDs)", search_filter_ids, 200)
    print_result(result)

    # Search with filter_id_range
    def search_filter_range():
        return http_post("/search", {
            "vector": query_vec,
            "k": 10,
            "filter": {},
            "filter_id_range": [0, 499]
        })

    result = run_benchmark("Search + filter_id_range [0,499]", search_filter_range, 200)
    print_result(result)

    # 3. GET Benchmark
    print("\n" + "-" * 40)
    print("3. GET BENCHMARK")
    print("-" * 40)

    def get_fn():
        return http_get(f"/vectors/{random.randint(0, 999)}")

    result = run_benchmark("GET Random Vector", get_fn, 500)
    print_result(result)

    # 4. Concurrent Benchmarks
    print("\n" + "-" * 40)
    print("4. CONCURRENT BENCHMARKS")
    print("-" * 40)

    for threads in [10, 20]:
        result = concurrent_benchmark("Search (no filter)", search_no_filter, 500, threads)
        print_result(result)

        result = concurrent_benchmark("Search + Indexed Filter", search_indexed_filter, 500, threads)
        print_result(result)

    # 5. Mixed Workload (90% read, 10% write)
    print("\n" + "-" * 40)
    print("5. MIXED WORKLOAD (90% read, 10% write)")
    print("-" * 40)

    def mixed_workload():
        if random.random() < 0.1:
            # Write
            i = len(vectors) + random.randint(10000, 99999)
            return http_post("/upsert", {
                "id": i,
                "vector": generate_vector(),
                "metadata": {"category": random.choice(categories)}
            })
        else:
            # Read
            return http_post("/search", {
                "vector": query_vec,
                "k": 10,
                "filter": {"category": random.choice(categories)}
            })

    result = concurrent_benchmark("Mixed Workload", mixed_workload, 1000, 20)
    print_result(result)

    # Final stats
    print("\n" + "=" * 60)
    stats = http_get("/stats")
    print(f"Final Vector Count: {stats['vector_count']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
