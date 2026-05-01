#!/usr/bin/env python3
"""Competitive benchmark: CoreVecDB vs ChromaDB vs Qdrant (local) vs LanceDB vs FAISS.

All tests use the same vectors, same dimensions, same queries.
CoreVecDB runs via HTTP (localhost:3100), others run in-process/embedded.
"""

import json
import random
import time
import statistics
import urllib.request
import tempfile
import shutil
import os
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
DIM = 128
NUM_VECTORS = 10_000
K = 10
SEARCH_ITERS = 500
FILTER_CATEGORIES = ["electronics", "books", "clothing", "food", "sports",
                      "toys", "health", "automotive", "garden", "office"]

random.seed(42)
np.random.seed(42)

# Pre-generate data
VECTORS = np.random.randn(NUM_VECTORS, DIM).astype(np.float32)
CATEGORIES = [FILTER_CATEGORIES[i % len(FILTER_CATEGORIES)] for i in range(NUM_VECTORS)]
QUERY = np.random.randn(DIM).astype(np.float32)

# ── Helpers ─────────────────────────────────────────────────────────────────
def bench(name, fn, n=SEARCH_ITERS):
    """Run fn() n times and report latency stats."""
    # Warmup
    for _ in range(min(20, n)):
        fn()

    lats = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        lats.append(time.perf_counter() - t0)
    ops = n / sum(lats)
    avg = statistics.mean(lats) * 1000
    p50 = statistics.median(lats) * 1000
    p99 = sorted(lats)[int(len(lats) * 0.99)] * 1000 if len(lats) >= 100 else sorted(lats)[-1] * 1000
    return {"name": name, "ops": ops, "avg_ms": avg, "p50_ms": p50, "p99_ms": p99}

def print_result(r):
    print(f"  {r['name']:42s}  {r['ops']:8.1f} ops/s  avg={r['avg_ms']:6.2f}ms  p50={r['p50_ms']:6.2f}ms  p99={r['p99_ms']:6.2f}ms")

# ── CoreVecDB ───────────────────────────────────────────────────────────────
def bench_corevecdb():
    print("\n" + "="*90)
    print("CoreVecDB (HTTP, localhost:3100)")
    print("="*90)

    URL = "http://localhost:3100"
    COLL = "bench_compete"

    def post(path, body):
        data = json.dumps(body).encode()
        req = urllib.request.Request(f"{URL}{path}", data=data,
                                    headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    def delete(path):
        req = urllib.request.Request(f"{URL}{path}", method="DELETE")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status
        except:
            return 0

    # Setup
    delete(f"/collections/{COLL}")
    post("/collections", {
        "name": COLL, "dim": DIM, "distance": "cosine",
        "indexed_fields": ["category"],
    })

    # Insert in batches
    t0 = time.perf_counter()
    for start in range(0, NUM_VECTORS, 200):
        end = min(start + 200, NUM_VECTORS)
        batch = []
        for i in range(start, end):
            batch.append({
                "vector": VECTORS[i].tolist(),
                "metadata": {"category": CATEGORIES[i], "name": f"item_{i}"}
            })
        post(f"/collections/{COLL}/upsert_batch", {"vectors": batch})
    insert_time = time.perf_counter() - t0
    print(f"  Insert {NUM_VECTORS} vectors: {insert_time:.2f}s ({NUM_VECTORS/insert_time:.0f} vec/s)")

    query_list = QUERY.tolist()

    # Search: no filter
    r1 = bench("Search (no filter)", lambda: post(f"/collections/{COLL}/search", {
        "vector": query_list, "k": K, "filter": {}, "include_metadata": False
    }))
    print_result(r1)

    # Search: indexed filter
    r2 = bench("Search (indexed filter: category)", lambda: post(f"/collections/{COLL}/search", {
        "vector": query_list, "k": K,
        "filter": {"category": "electronics"}, "include_metadata": False
    }))
    print_result(r2)

    # Search: with metadata
    r3 = bench("Search (filter + metadata)", lambda: post(f"/collections/{COLL}/search", {
        "vector": query_list, "k": K,
        "filter": {"category": "electronics"}, "include_metadata": True
    }))
    print_result(r3)

    delete(f"/collections/{COLL}")
    return {"insert_time": insert_time, "no_filter": r1, "filtered": r2, "filter_meta": r3}

# ── CoreVecDB Embedded ─────────────────────────────────────────────────────
def bench_corevecdb_embedded():
    print("\n" + "="*90)
    print("CoreVecDB (embedded, in-process, zero HTTP overhead)")
    print("="*90)
    import corevecdb

    tmpdir = tempfile.mkdtemp(prefix="corevecdb_embed_")
    db = corevecdb.CoreVecDB(tmpdir)

    db.create_collection("bench_compete", dim=DIM, distance="cosine",
                         indexed_fields=["category"], max_elements=NUM_VECTORS + 1000)
    col = db.collection("bench_compete")

    # Insert in batches
    t0 = time.perf_counter()
    BATCH = 500
    for start in range(0, NUM_VECTORS, BATCH):
        end = min(start + BATCH, NUM_VECTORS)
        vecs = [VECTORS[i].tolist() for i in range(start, end)]
        metas = [{"category": CATEGORIES[i], "name": f"item_{i}"} for i in range(start, end)]
        col.insert_batch(vecs, metas)
    insert_time = time.perf_counter() - t0
    print(f"  Insert {NUM_VECTORS} vectors: {insert_time:.2f}s ({NUM_VECTORS/insert_time:.0f} vec/s)")

    query_np = QUERY.astype(np.float32)

    # Search: no filter
    r1 = bench("Search (no filter)", lambda: col.search(query_np, k=K))
    print_result(r1)

    # Search: indexed filter
    r2 = bench("Search (indexed filter: category)", lambda: col.search(
        query_np, k=K, filter={"category": "electronics"}))
    print_result(r2)

    # Search: with metadata
    r3 = bench("Search (filter + metadata)", lambda: col.search(
        query_np, k=K, filter={"category": "electronics"}, include_metadata=True))
    print_result(r3)

    db.flush()
    shutil.rmtree(tmpdir, ignore_errors=True)
    return {"insert_time": insert_time, "no_filter": r1, "filtered": r2, "filter_meta": r3}

# ── ChromaDB ────────────────────────────────────────────────────────────────
def bench_chromadb():
    print("\n" + "="*90)
    print("ChromaDB (in-process, ephemeral)")
    print("="*90)
    import chromadb

    client = chromadb.Client()
    try:
        client.delete_collection("bench_compete")
    except:
        pass
    collection = client.create_collection("bench_compete", metadata={"hnsw:space": "cosine"})

    # Insert in batches
    t0 = time.perf_counter()
    for start in range(0, NUM_VECTORS, 500):
        end = min(start + 500, NUM_VECTORS)
        ids = [f"id_{i}" for i in range(start, end)]
        embeddings = VECTORS[start:end].tolist()
        metadatas = [{"category": CATEGORIES[i], "name": f"item_{i}"} for i in range(start, end)]
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    insert_time = time.perf_counter() - t0
    print(f"  Insert {NUM_VECTORS} vectors: {insert_time:.2f}s ({NUM_VECTORS/insert_time:.0f} vec/s)")

    query_list = QUERY.tolist()

    # Search: no filter
    r1 = bench("Search (no filter)", lambda: collection.query(
        query_embeddings=[query_list], n_results=K, include=[]
    ))
    print_result(r1)

    # Search: filtered
    r2 = bench("Search (filtered: category)", lambda: collection.query(
        query_embeddings=[query_list], n_results=K,
        where={"category": "electronics"}, include=[]
    ))
    print_result(r2)

    # Search: with metadata
    r3 = bench("Search (filter + metadata)", lambda: collection.query(
        query_embeddings=[query_list], n_results=K,
        where={"category": "electronics"}, include=["metadatas"]
    ))
    print_result(r3)

    client.delete_collection("bench_compete")
    return {"insert_time": insert_time, "no_filter": r1, "filtered": r2, "filter_meta": r3}

# ── Qdrant (local mode) ────────────────────────────────────────────────────
def bench_qdrant():
    print("\n" + "="*90)
    print("Qdrant (local mode, in-process)")
    print("="*90)
    from qdrant_client import QdrantClient, models

    tmpdir = tempfile.mkdtemp(prefix="qdrant_bench_")
    client = QdrantClient(path=tmpdir)

    client.create_collection(
        collection_name="bench_compete",
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
    )
    # Create payload index
    client.create_payload_index(
        collection_name="bench_compete",
        field_name="category",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    # Insert in batches
    t0 = time.perf_counter()
    for start in range(0, NUM_VECTORS, 500):
        end = min(start + 500, NUM_VECTORS)
        points = []
        for i in range(start, end):
            points.append(models.PointStruct(
                id=i,
                vector=VECTORS[i].tolist(),
                payload={"category": CATEGORIES[i], "name": f"item_{i}"}
            ))
        client.upsert(collection_name="bench_compete", points=points)
    insert_time = time.perf_counter() - t0
    print(f"  Insert {NUM_VECTORS} vectors: {insert_time:.2f}s ({NUM_VECTORS/insert_time:.0f} vec/s)")

    query_list = QUERY.tolist()

    # Search: no filter
    r1 = bench("Search (no filter)", lambda: client.query_points(
        collection_name="bench_compete",
        query=query_list, limit=K, with_payload=False
    ))
    print_result(r1)

    # Search: filtered
    r2 = bench("Search (filtered: category)", lambda: client.query_points(
        collection_name="bench_compete",
        query=query_list, limit=K,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="category", match=models.MatchValue(value="electronics"))]
        ),
        with_payload=False
    ))
    print_result(r2)

    # Search: with metadata
    r3 = bench("Search (filter + metadata)", lambda: client.query_points(
        collection_name="bench_compete",
        query=query_list, limit=K,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="category", match=models.MatchValue(value="electronics"))]
        ),
        with_payload=True
    ))
    print_result(r3)

    client.close()
    shutil.rmtree(tmpdir, ignore_errors=True)
    return {"insert_time": insert_time, "no_filter": r1, "filtered": r2, "filter_meta": r3}

# ── LanceDB ─────────────────────────────────────────────────────────────────
def bench_lancedb():
    print("\n" + "="*90)
    print("LanceDB (embedded, Arrow-native)")
    print("="*90)
    import lancedb
    import pyarrow as pa

    tmpdir = tempfile.mkdtemp(prefix="lance_bench_")
    db = lancedb.connect(tmpdir)

    # Insert
    t0 = time.perf_counter()
    data = pa.table({
        "id": list(range(NUM_VECTORS)),
        "vector": [VECTORS[i].tolist() for i in range(NUM_VECTORS)],
        "category": CATEGORIES,
        "name": [f"item_{i}" for i in range(NUM_VECTORS)],
    })
    tbl = db.create_table("bench_compete", data)
    tbl.create_index("cosine", vector_column_name="vector", num_partitions=4, num_sub_vectors=16,
                     replace=True)
    insert_time = time.perf_counter() - t0
    print(f"  Insert {NUM_VECTORS} vectors + index: {insert_time:.2f}s ({NUM_VECTORS/insert_time:.0f} vec/s)")

    query_list = QUERY.tolist()

    # Search: no filter
    r1 = bench("Search (no filter)", lambda: tbl.search(query_list).select(["id"]).limit(K).to_list())
    print_result(r1)

    # Search: filtered
    r2 = bench("Search (filtered: category)", lambda: tbl.search(query_list)
               .where("category = 'electronics'").select(["id"]).limit(K).to_list())
    print_result(r2)

    # Search: with metadata
    r3 = bench("Search (filter + metadata)", lambda: tbl.search(query_list)
               .where("category = 'electronics'").limit(K).to_list())
    print_result(r3)

    db.drop_table("bench_compete")
    shutil.rmtree(tmpdir, ignore_errors=True)
    return {"insert_time": insert_time, "no_filter": r1, "filtered": r2, "filter_meta": r3}

# ── FAISS ────────────────────────────────────────────────────────────────────
def bench_faiss():
    print("\n" + "="*90)
    print("FAISS (in-memory, brute-force + HNSW)")
    print("="*90)
    import faiss

    # Normalize for cosine sim
    vecs = VECTORS.copy()
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vecs = vecs / norms

    query_norm = QUERY.copy()
    qn = np.linalg.norm(query_norm)
    if qn > 0:
        query_norm = query_norm / qn
    query_2d = query_norm.reshape(1, DIM)

    # --- Flat (brute-force) ---
    t0 = time.perf_counter()
    index_flat = faiss.IndexFlatIP(DIM)
    index_flat.add(vecs)
    insert_flat = time.perf_counter() - t0
    print(f"  Insert {NUM_VECTORS} vectors (Flat): {insert_flat:.4f}s")

    r1_flat = bench("Flat search (no filter)", lambda: index_flat.search(query_2d, K))
    print_result(r1_flat)

    # --- HNSW ---
    t0 = time.perf_counter()
    index_hnsw = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    index_hnsw.hnsw.efSearch = 64
    index_hnsw.add(vecs)
    insert_hnsw = time.perf_counter() - t0
    print(f"  Insert {NUM_VECTORS} vectors (HNSW): {insert_hnsw:.2f}s")

    r1_hnsw = bench("HNSW search (no filter)", lambda: index_hnsw.search(query_2d, K))
    print_result(r1_hnsw)

    # FAISS doesn't natively support filtered search, simulate with IDSelector
    category_ids = np.array([i for i in range(NUM_VECTORS) if CATEGORIES[i] == "electronics"], dtype=np.int64)
    id_selector = faiss.IDSelectorArray(category_ids)
    search_params = faiss.SearchParametersHNSW()
    search_params.sel = id_selector

    r2_hnsw = bench("HNSW search (filtered: IDSelector)", lambda: index_hnsw.search(query_2d, K, params=search_params))
    print_result(r2_hnsw)

    return {
        "insert_time": insert_flat,
        "no_filter": r1_hnsw,
        "filtered": r2_hnsw,
        "flat": r1_flat,
    }

# ── Summary ─────────────────────────────────────────────────────────────────
def print_summary(results):
    print("\n" + "="*90)
    print("COMPETITIVE BENCHMARK SUMMARY")
    print(f"Config: {NUM_VECTORS} vectors, {DIM}-dim, k={K}, {SEARCH_ITERS} iterations")
    print("="*90)

    # Insert speed
    print(f"\n{'─'*90}")
    print(f"  {'INSERT SPEED':42s}  {'Time':>8s}  {'Vec/s':>10s}")
    print(f"{'─'*90}")
    for name, data in results.items():
        t = data["insert_time"]
        print(f"  {name:42s}  {t:7.2f}s  {NUM_VECTORS/t:9.0f}")

    # No-filter search
    print(f"\n{'─'*90}")
    print(f"  {'SEARCH (NO FILTER)':42s}  {'ops/s':>8s}  {'avg':>8s}  {'p50':>8s}  {'p99':>8s}")
    print(f"{'─'*90}")
    for name, data in results.items():
        r = data["no_filter"]
        print(f"  {name:42s}  {r['ops']:8.1f}  {r['avg_ms']:7.2f}ms {r['p50_ms']:7.2f}ms {r['p99_ms']:7.2f}ms")

    # Filtered search
    print(f"\n{'─'*90}")
    print(f"  {'SEARCH (FILTERED: category=electronics)':42s}  {'ops/s':>8s}  {'avg':>8s}  {'p50':>8s}  {'p99':>8s}")
    print(f"{'─'*90}")
    for name, data in results.items():
        r = data.get("filtered")
        if r:
            print(f"  {name:42s}  {r['ops']:8.1f}  {r['avg_ms']:7.2f}ms {r['p50_ms']:7.2f}ms {r['p99_ms']:7.2f}ms")

    # Filter + metadata
    print(f"\n{'─'*90}")
    print(f"  {'SEARCH (FILTER + METADATA)':42s}  {'ops/s':>8s}  {'avg':>8s}  {'p50':>8s}  {'p99':>8s}")
    print(f"{'─'*90}")
    for name, data in results.items():
        r = data.get("filter_meta")
        if r:
            print(f"  {name:42s}  {r['ops']:8.1f}  {r['avg_ms']:7.2f}ms {r['p50_ms']:7.2f}ms {r['p99_ms']:7.2f}ms")

    print(f"\n{'='*90}")

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    results = {}

    print("="*90)
    print("Vector Database Competitive Benchmark")
    print(f"Config: {NUM_VECTORS} vectors, {DIM}-dim, k={K}, {SEARCH_ITERS} search iterations")
    print("="*90)

    # CoreVecDB Embedded (in-process)
    try:
        results["CoreVecDB (embedded)"] = bench_corevecdb_embedded()
    except Exception as e:
        print(f"  ❌ CoreVecDB embedded failed: {e}")

    # CoreVecDB HTTP
    try:
        results["CoreVecDB (HTTP)"] = bench_corevecdb()
    except Exception as e:
        print(f"  ❌ CoreVecDB HTTP failed: {e}")

    # ChromaDB
    try:
        results["ChromaDB"] = bench_chromadb()
    except Exception as e:
        print(f"  ❌ ChromaDB failed: {e}")

    # Qdrant
    try:
        results["Qdrant (local)"] = bench_qdrant()
    except Exception as e:
        print(f"  ❌ Qdrant failed: {e}")

    # LanceDB
    try:
        results["LanceDB"] = bench_lancedb()
    except Exception as e:
        print(f"  ❌ LanceDB failed: {e}")

    # FAISS
    try:
        results["FAISS (HNSW)"] = bench_faiss()
    except Exception as e:
        print(f"  ❌ FAISS failed: {e}")

    print_summary(results)

if __name__ == "__main__":
    main()
