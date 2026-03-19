import urllib.request
import json
import random
import time
import statistics

URL = "http://localhost:3000"
DIM = 128
NUM_UPSERTS = 1000
NUM_SEARCHES = 100

def generate_vector():
    return [random.random() for _ in range(DIM)]

def upsert_benchmark():
    print(f"Starting Upsert Benchmark ({NUM_UPSERTS} items)...")
    latencies = []
    start_time = time.time()
    
    for i in range(NUM_UPSERTS):
        payload = {
            "id": i,
            "vector": generate_vector(),
            "metadata": {"bench": "true"}
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(f"{URL}/upsert", data=data, headers={'Content-Type': 'application/json'})
        
        req_start = time.time()
        with urllib.request.urlopen(req) as f:
            f.read()
        latencies.append(time.time() - req_start)
        
        if (i + 1) % 100 == 0:
            print(f"Upserted {i + 1}/{NUM_UPSERTS}")

    total_time = time.time() - start_time
    avg_latency = statistics.mean(latencies) * 1000
    ops = NUM_UPSERTS / total_time
    
    print(f"Upsert Results:")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Throughput: {ops:.2f} upserts/sec")
    print(f"  Avg Latency: {avg_latency:.2f} ms")
    return ops

def search_benchmark():
    print(f"\nStarting Search Benchmark ({NUM_SEARCHES} queries)...")
    latencies = []
    start_time = time.time()
    
    for i in range(NUM_SEARCHES):
        payload = {
            "vector": generate_vector(),
            "k": 10,
            "filter": {}
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(f"{URL}/search", data=data, headers={'Content-Type': 'application/json'})
        
        req_start = time.time()
        with urllib.request.urlopen(req) as f:
            f.read()
        latencies.append(time.time() - req_start)

    total_time = time.time() - start_time
    avg_latency = statistics.mean(latencies) * 1000
    ops = NUM_SEARCHES / total_time
    
    print(f"Search Results:")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Throughput: {ops:.2f} searches/sec")
    print(f"  Avg Latency: {avg_latency:.2f} ms")
    return ops

if __name__ == "__main__":
    # Wait for server to be ready (simple sleep or check)
    print("Waiting for server...")
    time.sleep(2)
    
    try:
        upsert_benchmark()
        search_benchmark()
    except Exception as e:
        print(f"Benchmark failed: {e}")
