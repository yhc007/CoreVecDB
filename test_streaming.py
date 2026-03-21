#!/usr/bin/env python3
"""Test Streaming Insert functionality."""
import requests
import random
import json
import sseclient  # pip install sseclient-py

BASE_URL = "http://localhost:3000"

def random_vector(dim=128):
    return [random.random() for _ in range(dim)]

def test_streaming_upsert():
    print("=== Testing Streaming Insert ===\n")

    # Create collection
    print("1. Creating test collection...")
    resp = requests.post(f"{BASE_URL}/collections", json={
        "name": "stream_test",
        "dim": 128,
        "indexed_fields": ["category"],
    })
    print(f"   Response: {resp.status_code}")

    # Generate large batch of vectors
    print("\n2. Generating 5000 test vectors...")
    vectors = [
        {"vector": random_vector(), "metadata": {"category": f"cat_{i % 10}", "idx": str(i)}}
        for i in range(5000)
    ]
    print(f"   Generated {len(vectors)} vectors")

    # Stream upsert with SSE progress
    print("\n3. Streaming upsert with progress updates...")
    print("   (Using chunk_size=500)")

    # Use requests with stream=True for SSE
    with requests.post(
        f"{BASE_URL}/collections/stream_test/stream_upsert",
        json={"vectors": vectors, "chunk_size": 500},
        stream=True,
        headers={"Accept": "text/event-stream"}
    ) as response:
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.event == "progress":
                data = json.loads(event.data)
                print(f"   Progress: {data['progress_percent']:.1f}% - Inserted {data['inserted_so_far']} vectors")
            elif event.event == "complete":
                data = json.loads(event.data)
                print(f"\n   Complete! Total inserted: {data['total_inserted']}")
                print(f"   Start ID: {data['start_id']}")
            elif event.event == "error":
                print(f"   Error: {event.data}")
                break

    # Verify collection info
    print("\n4. Checking collection info...")
    resp = requests.get(f"{BASE_URL}/collections/stream_test")
    info = resp.json()
    print(f"   Vectors: {info['vector_count']}")

    # Search to verify vectors were indexed
    print("\n5. Searching for vectors...")
    resp = requests.post(f"{BASE_URL}/collections/stream_test/search", json={
        "vector": random_vector(),
        "k": 5,
    })
    if resp.status_code == 200:
        results = resp.json()["results"]
        print(f"   Found {len(results)} results")
        if results:
            print(f"   First result: ID={results[0]['id']}, score={results[0]['score']:.4f}")

    # Search with filter
    print("\n6. Searching with filter (category=cat_5)...")
    resp = requests.post(f"{BASE_URL}/collections/stream_test/search", json={
        "vector": random_vector(),
        "k": 5,
        "filter": {"category": "cat_5"}
    })
    if resp.status_code == 200:
        results = resp.json()["results"]
        print(f"   Found {len(results)} results with category=cat_5")

    # Cleanup
    print("\n7. Cleaning up...")
    requests.delete(f"{BASE_URL}/collections/stream_test")
    print("   Done!")

def test_streaming_simple():
    """Simple test without SSE client dependency."""
    print("\n=== Simple Streaming Test (without SSE client) ===\n")

    print("1. Creating collection...")
    resp = requests.post(f"{BASE_URL}/collections", json={
        "name": "stream_simple",
        "dim": 128,
    })
    print(f"   Status: {resp.status_code}")

    print("\n2. Generating 2000 vectors...")
    vectors = [
        {"vector": random_vector(), "metadata": {"idx": str(i)}}
        for i in range(2000)
    ]

    print("\n3. Streaming upsert...")
    resp = requests.post(
        f"{BASE_URL}/collections/stream_simple/stream_upsert",
        json={"vectors": vectors, "chunk_size": 500},
        stream=True
    )

    # Read raw SSE response
    for line in resp.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data:'):
                data = line_str[5:].strip()
                if data:
                    try:
                        parsed = json.loads(data)
                        if 'progress_percent' in parsed:
                            print(f"   Progress: {parsed['progress_percent']:.1f}%")
                        elif 'total_inserted' in parsed:
                            print(f"   Complete: {parsed['total_inserted']} vectors inserted")
                    except json.JSONDecodeError:
                        print(f"   Raw: {data}")

    # Verify
    print("\n4. Verifying...")
    resp = requests.get(f"{BASE_URL}/collections/stream_simple")
    print(f"   Vectors: {resp.json()['vector_count']}")

    # Cleanup
    requests.delete(f"{BASE_URL}/collections/stream_simple")
    print("   Done!")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        test_streaming_simple()
    else:
        try:
            import sseclient
            test_streaming_upsert()
        except ImportError:
            print("sseclient-py not installed, running simple test...")
            print("Install with: pip install sseclient-py")
            test_streaming_simple()
