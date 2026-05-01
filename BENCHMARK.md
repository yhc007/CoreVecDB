# CoreVecDB Benchmark Results

Benchmark comparing CoreVecDB embedded (in-process) mode against CoreVecDB HTTP mode and competing vector databases.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Vectors | 10,000 |
| Dimensions | 128 |
| k (top results) | 10 |
| Iterations | 500~1,000 |
| Filter | `category = "electronics"` (10% selectivity, 10 categories) |
| Distance metric | Cosine |
| Hardware | Apple Silicon (M-series), macOS |

All databases use the same pre-generated random vectors (seed=42) and metadata.

## Filtered Search Performance (category = "electronics")

The primary benchmark — this is where CoreVecDB's embedded mode shines.

| Database | Mode | ops/s | avg latency | p50 | p99 |
|----------|------|------:|------------:|----:|----:|
| **CoreVecDB** | **Embedded (in-process)** | **2,340** | **0.43ms** | **0.41ms** | **0.55ms** |
| FAISS | In-memory (IDSelector) | ~1,800 | ~0.56ms | - | - |
| CoreVecDB | HTTP (localhost) | 640 | 1.56ms | 1.50ms | 2.10ms |
| LanceDB | Embedded (Arrow) | ~750 | ~1.33ms | - | - |
| ChromaDB | In-process | ~240 | ~4.17ms | - | - |
| Qdrant | Local mode | ~200 | ~5.00ms | - | - |

### Key Takeaway

CoreVecDB embedded mode achieves **2,340 ops/s** for filtered search — **3.7x faster than HTTP mode** and **3.1x faster than LanceDB**.

## Unfiltered Search Performance

| Database | Mode | ops/s | avg latency | p50 | p99 |
|----------|------|------:|------------:|----:|----:|
| FAISS | In-memory (HNSW) | ~50,000 | ~0.02ms | - | - |
| FAISS | In-memory (Flat/brute) | ~12,000 | ~0.08ms | - | - |
| CoreVecDB | HTTP (cached) | ~1,388 | 0.72ms | 0.70ms | 1.10ms |
| LanceDB | Embedded | ~900 | ~1.11ms | - | - |
| CoreVecDB | Embedded | 382 | 2.62ms | 2.63ms | 3.10ms |
| ChromaDB | In-process | ~350 | ~2.86ms | - | - |
| Qdrant | Local mode | ~300 | ~3.33ms | - | - |

> Note: CoreVecDB HTTP "no filter" benefits from query result caching (same query repeated).
> FAISS operates purely in-memory with no persistence overhead.

## Filter + Metadata Retrieval

| Database | Mode | ops/s | avg latency | p50 | p99 |
|----------|------|------:|------------:|----:|----:|
| **CoreVecDB** | **Embedded** | **2,260** | **0.44ms** | **0.42ms** | **0.58ms** |
| LanceDB | Embedded | ~700 | ~1.43ms | - | - |
| CoreVecDB | HTTP | ~240 | ~4.17ms | - | - |
| ChromaDB | In-process | ~220 | ~4.55ms | - | - |

## Insert Performance

| Database | Mode | vec/s | Time (10K) |
|----------|------|------:|-----------:|
| FAISS | In-memory (Flat) | ~250,000 | 0.04s |
| LanceDB | Embedded | ~5,000 | ~2.0s |
| **CoreVecDB** | **Embedded** | **2,980** | **3.36s** |
| CoreVecDB | HTTP | ~2,500 | ~4.0s |
| ChromaDB | In-process | ~2,000 | ~5.0s |
| Qdrant | Local mode | ~1,500 | ~6.7s |

## Architecture Comparison

| Database | Index Type | Storage | Filter Method | Python API |
|----------|-----------|---------|--------------|------------|
| CoreVecDB | HNSW (adaptive) | mmap + sled | RoaringBitmap pre-filter | PyO3 native |
| LanceDB | IVF-PQ | Arrow/Lance columnar | Arrow SIMD scan | PyArrow native |
| ChromaDB | HNSW | DuckDB + Parquet | Brute-force scan | Native Python |
| Qdrant | HNSW | RocksDB | Payload index | gRPC/REST |
| FAISS | HNSW / Flat | In-memory only | IDSelector bitmap | C++ bindings |

## Why Embedded Mode Is Faster

CoreVecDB HTTP mode incurs ~1ms overhead per request:
- JSON serialization/deserialization (~0.3ms)
- TCP round-trip on localhost (~0.2ms)
- HTTP framework overhead (~0.5ms)

Embedded mode eliminates all of this — function calls go directly to `Collection::search()` with zero-copy filter bitmap construction.

For filtered search at 10K scale, the actual computation is ~0.4ms (bitmap lookup + HNSW traversal). HTTP overhead was **71% of total latency**.

## How to Run

```bash
# Embedded benchmark (recommended)
cd corevecdb-python
python3 -m venv .venv && source .venv/bin/activate
pip install numpy
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
cd .. && python3 bench_competitive.py

# HTTP-only benchmark (requires running server)
cargo run --release  # terminal 1
python3 bench_competitive.py  # terminal 2
```

## Usage

### Python (Embedded)

```python
import corevecdb
import numpy as np

db = corevecdb.CoreVecDB("./my_data")
db.create_collection("products", dim=128,
    indexed_fields=["category"], distance="cosine")

col = db.collection("products")
col.insert(np.array([0.1] * 128, dtype=np.float32),
           metadata={"category": "electronics"})

results = col.search(
    np.array([0.1] * 128, dtype=np.float32),
    k=10,
    filter={"category": "electronics"},
    include_metadata=True
)
```

### Rust (Embedded)

```rust
use vectordb::embedded::CoreVecDB;
use vectordb::collection::{CollectionConfig, SearchParams};

let db = CoreVecDB::open("./my_data")?;
db.create_collection(
    CollectionConfig::new("products", 128)
        .with_distance("cosine")
        .with_indexed_fields(vec!["category"], vec![])
)?;

let col = db.collection("products").unwrap();
let results = col.search(
    SearchParams::new(query_vec, 10)
        .with_filter("category", "electronics")
        .with_metadata()
)?;
```
