# CoreVecDB Benchmark Results

Benchmark comparing CoreVecDB embedded (in-process) mode against CoreVecDB HTTP mode and competing vector databases.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Vectors | 10,000 |
| Dimensions | 128 |
| k (top results) | 10 |
| Iterations | 500 |
| Filter | `category = "electronics"` (10% selectivity, 10 categories) |
| Distance metric | Cosine |
| Hardware | Apple Silicon (M-series), macOS |

All databases use the same pre-generated random vectors (seed=42) and metadata.

## Filtered Search Performance (category = "electronics")

The primary benchmark — this is where CoreVecDB's embedded mode shines.

| Database | Mode | ops/s | avg latency | p50 | p99 |
|----------|------|------:|------------:|----:|----:|
| FAISS | In-memory (IDSelector) | 3,718 | 0.27ms | 0.27ms | 0.34ms |
| **CoreVecDB** | **Embedded (in-process)** | **2,218** | **0.45ms** | **0.45ms** | **0.52ms** |
| LanceDB | Embedded (Arrow) | 767 | 1.30ms | 1.25ms | 1.94ms |
| CoreVecDB | HTTP (localhost) | 263 | 3.80ms | 3.45ms | 8.78ms |
| ChromaDB | In-process | 220 | 4.54ms | 4.47ms | 5.51ms |
| Qdrant | Local mode | 23 | 43.34ms | 44.66ms | 53.74ms |

### Key Takeaway

CoreVecDB embedded mode achieves **2,218 ops/s** for filtered search — **8.4x faster than HTTP mode**, **2.9x faster than LanceDB**, and **10x faster than ChromaDB**. Only FAISS (pure in-memory, no persistence) is faster.

## Unfiltered Search Performance

| Database | Mode | ops/s | avg latency | p50 | p99 |
|----------|------|------:|------------:|----:|----:|
| FAISS | In-memory (HNSW) | 22,005 | 0.05ms | 0.04ms | 0.05ms |
| FAISS | In-memory (Flat/brute) | 9,654 | 0.10ms | 0.10ms | 0.13ms |
| ChromaDB | In-process | 1,575 | 0.63ms | 0.61ms | 0.91ms |
| CoreVecDB | HTTP (cached) | 1,131 | 0.88ms | 0.75ms | 3.99ms |
| LanceDB | Embedded | 1,035 | 0.97ms | 0.93ms | 1.58ms |
| Qdrant | Local mode | 552 | 1.81ms | 1.79ms | 2.08ms |
| CoreVecDB | Embedded | 379 | 2.64ms | 2.64ms | 2.89ms |

> Note: CoreVecDB HTTP "no filter" benefits from query result caching (same query repeated).
> FAISS operates purely in-memory with no persistence overhead.

## Filter + Metadata Retrieval

| Database | Mode | ops/s | avg latency | p50 | p99 |
|----------|------|------:|------------:|----:|----:|
| **CoreVecDB** | **Embedded** | **2,133** | **0.47ms** | **0.47ms** | **0.60ms** |
| LanceDB | Embedded | 535 | 1.87ms | 1.80ms | 2.35ms |
| CoreVecDB | HTTP | 252 | 3.96ms | 3.60ms | 9.66ms |
| ChromaDB | In-process | 208 | 4.80ms | 4.74ms | 5.90ms |
| Qdrant | Local mode | 23 | 43.45ms | 44.61ms | 55.05ms |

## Insert Performance

| Database | Mode | vec/s | Time (10K) |
|----------|------|------:|-----------:|
| FAISS | In-memory (Flat) | 52,140,095 | 0.0002s |
| LanceDB | Embedded + index | 8,400 | 1.19s |
| ChromaDB | In-process | 7,919 | 1.26s |
| **CoreVecDB** | **Embedded** | **3,327** | **3.01s** |
| CoreVecDB | HTTP | 2,538 | 3.94s |
| Qdrant | Local mode | 1,340 | 7.46s |

## Architecture Comparison

| Database | Index Type | Storage | Filter Method | Python API |
|----------|-----------|---------|--------------|------------|
| CoreVecDB | HNSW (adaptive) | mmap + sled | RoaringBitmap pre-filter | PyO3 native |
| LanceDB | IVF-PQ | Arrow/Lance columnar | Arrow SIMD scan | PyArrow native |
| ChromaDB | HNSW | DuckDB + Parquet | Brute-force scan | Native Python |
| Qdrant | HNSW | RocksDB | Payload index | gRPC/REST |
| FAISS | HNSW / Flat | In-memory only | IDSelector bitmap | C++ bindings |

## Why Embedded Mode Is Faster

CoreVecDB HTTP mode incurs ~3.5ms overhead per filtered request:
- JSON serialization/deserialization (~0.3ms)
- TCP round-trip on localhost (~0.2ms)
- HTTP framework overhead (~0.5ms)
- Connection management and buffering (~2.5ms)

Embedded mode eliminates all of this — function calls go directly to `Collection::search()` with zero-copy filter bitmap construction.

For filtered search at 10K scale, the actual computation is ~0.45ms (bitmap lookup + HNSW traversal). HTTP overhead was **88% of total latency**.

## How to Run

```bash
# Full competitive benchmark
cd corevecdb-python
python3 -m venv .venv && source .venv/bin/activate
pip install numpy chromadb qdrant-client lancedb faiss-cpu
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
cd .. && python3 bench_competitive.py

# HTTP benchmark (requires running server)
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

let col = db.collection("products")?;
let results = col.search(
    SearchParams::new(query_vec, 10)
        .with_filter("category", "electronics")
        .with_metadata()
)?;
```
