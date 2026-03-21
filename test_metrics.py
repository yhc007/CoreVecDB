#!/usr/bin/env python3
"""Test Prometheus Metrics functionality."""
import requests
import random

BASE_URL = "http://localhost:3000"

def random_vector(dim=128):
    return [random.random() for _ in range(dim)]

def test_metrics():
    print("=== Testing Prometheus Metrics ===\n")

    # Create some test data first
    print("1. Creating test collection with vectors...")
    requests.post(f"{BASE_URL}/collections", json={
        "name": "metrics_test",
        "dim": 128,
    })

    vectors = [{"vector": random_vector(), "metadata": {"idx": str(i)}} for i in range(100)]
    requests.post(f"{BASE_URL}/collections/metrics_test/upsert_batch", json={"vectors": vectors})

    # Delete some vectors
    requests.post(f"{BASE_URL}/collections/metrics_test/delete_batch", json={"ids": [10, 20, 30]})

    # Do a search
    requests.post(f"{BASE_URL}/collections/metrics_test/search", json={
        "vector": random_vector(),
        "k": 10,
    })

    print("   Created collection with 100 vectors, deleted 3, did 1 search")

    # Fetch metrics
    print("\n2. Fetching Prometheus metrics...")
    resp = requests.get(f"{BASE_URL}/metrics")

    if resp.status_code == 200:
        print(f"   Content-Type: {resp.headers.get('Content-Type')}")
        print(f"   Size: {len(resp.text)} bytes")

        # Parse and display key metrics
        print("\n3. Key Metrics:")
        lines = resp.text.split('\n')

        metrics_to_show = [
            'vectordb_collections_total',
            'vectordb_vectors_total',
            'vectordb_deleted_vectors_total',
            'vectordb_active_vectors_total',
            'vectordb_collection_dimension',
            'vectordb_operations_total',
            'vectordb_server_info',
        ]

        for metric_name in metrics_to_show:
            for line in lines:
                if line.startswith(metric_name) and not line.startswith('#'):
                    print(f"   {line}")
                    break

        # Show full output preview
        print("\n4. Full Metrics Output (first 50 lines):")
        for line in lines[:50]:
            if line:
                print(f"   {line}")

    else:
        print(f"   Error: {resp.status_code}")

    # Cleanup
    print("\n5. Cleaning up...")
    requests.delete(f"{BASE_URL}/collections/metrics_test")
    print("   Done!")

def show_metrics_help():
    print("\n=== Prometheus Metrics Reference ===\n")
    print("Available metrics:")
    print()
    print("Collection Metrics:")
    print("  vectordb_collections_total         - Total number of collections")
    print("  vectordb_vectors_total             - Total vectors per collection")
    print("  vectordb_deleted_vectors_total     - Deleted vectors per collection")
    print("  vectordb_active_vectors_total      - Active vectors (total - deleted)")
    print("  vectordb_collection_dimension      - Vector dimension per collection")
    print()
    print("Operation Metrics:")
    print("  vectordb_operations_total          - Operation counts by type/status")
    print("  vectordb_operation_duration_seconds - Operation latency histogram")
    print()
    print("Search Metrics:")
    print("  vectordb_search_results_total      - Results returned per search")
    print("  vectordb_search_k                  - K parameter histogram")
    print()
    print("Batch Metrics:")
    print("  vectordb_batch_size                - Batch operation sizes")
    print()
    print("Storage Metrics:")
    print("  vectordb_storage_bytes             - Storage size per collection")
    print()
    print("WAL/Replication Metrics:")
    print("  vectordb_wal_sequence              - Current WAL sequence number")
    print("  vectordb_wal_size_bytes            - WAL total size")
    print("  vectordb_wal_entries_total         - Total WAL entries")
    print()
    print("Prometheus scrape config example:")
    print("""
  - job_name: 'vectordb'
    static_configs:
      - targets: ['localhost:3000']
    metrics_path: '/metrics'
""")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_metrics_help()
    else:
        test_metrics()
        show_metrics_help()
