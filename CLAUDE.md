# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

```bash
# Build (release)
cargo build --release

# Run the server (uses config.toml)
cargo run --release

# Build only (compiles protobuf definitions via build.rs)
cargo build

# Test client (requires server running)
python test_client.py

# Run benchmark
python benchmark.py
```

## Architecture Overview

VectorDB is a Rust-based vector database with HNSW indexing, exposing both gRPC and HTTP APIs.

### Core Layers

1. **Storage** (`src/storage/mod.rs`)
   - `MemmapVectorStore`: Memory-mapped file for vector data (`data/vectors.bin`)
   - `QuantizedMemmapVectorStore`: Scalar quantized storage (75% memory reduction)
   - `SledMetadataStore`: Embedded key-value store for metadata (`data/meta.sled`)
   - `IndexedSledMetadataStore`: SledMetadataStore + PayloadIndex for fast filtering
   - Vectors are append-only; IDs are auto-assigned sequentially starting from 0

2. **Index** (`src/index/mod.rs`)
   - `HnswIndexer`: Wraps `hnsw_rs` crate for approximate nearest neighbor search
   - Uses L2 (Euclidean) distance; for cosine similarity, normalize vectors before insertion
   - Supports filtered search via `RoaringBitmap`
   - Index persists to `data/index.hnsw.*` files on shutdown

3. **Quantization** (`src/quantization/mod.rs`)
   - `ScalarQuantizer`: float32 → uint8 compression (4x memory reduction)
   - Online training: learns min/max per dimension incrementally
   - Asymmetric distance: query in float32, candidates in uint8

4. **Payload Index** (`src/payload/mod.rs`)
   - `PayloadIndex`: Inverted index for metadata filtering
   - `FieldIndex`: Per-field index using `RoaringBitmap` per unique value
   - `FilterQuery` DSL: `Eq`, `In`, `And`, `Or`, `Not` operations
   - Pre-filtering in HNSW search for indexed fields (O(1) bitmap lookup)
   - Automatic fallback to post-filtering for non-indexed fields

5. **API** (`src/api/`)
   - `mod.rs`: gRPC service implementation (`VectorServiceImpl`)
   - `http.rs`: Axum HTTP handlers mirroring gRPC functionality
   - Both share the same underlying service logic

6. **Proto** (`src/proto/vectordb.proto`)
   - Defines `VectorService` with `Upsert`, `Search`, `Get` RPCs
   - Compiled via `tonic-build` in `build.rs`

### Request Flow

```
Client → HTTP (port 3000) or gRPC (port 50051)
      → VectorServiceImpl
      → MemmapVectorStore (vector data)
      → HnswIndexer (similarity search)
      → SledMetadataStore (metadata filtering)
```

### Configuration

`config.toml` controls:
- `server.grpc_port`, `server.http_port`, `server.data_dir`
- `index.dim` (vector dimension), `index.max_elements`, `index.m`, `index.ef_construction`
- `quantization.enabled`, `quantization.keep_originals`, `quantization.rerank_oversample`
- `payload.index_enabled`, `payload.indexed_fields` (list of fields to index)

Environment override: `APP_SERVER__GRPC_PORT=50052` etc.

## HTTP API Endpoints

- `POST /upsert` - Insert vector with metadata
- `POST /search` - k-NN search with optional metadata/ID filtering
- `GET /vectors/:id` - Retrieve vector by ID
- `GET /stats` - Get vector count and status
- Static files served from `ui/` directory at root

## Key Implementation Details

- Vector IDs are internal (0-indexed, auto-incremented); user-provided IDs in upsert are currently ignored
- Search supports two filtering modes: pre-filter via `filter_ids` or `filter_id_range` (RoaringBitmap) and post-filter via metadata `filter`
- Post-filtering uses Rayon for parallel metadata lookups
- Index is saved on graceful shutdown (SIGINT/SIGTERM)

## Performance Optimizations

### Storage Layer
- **Write Buffer**: Batches 256 vectors before flushing to disk (reduces fsync overhead)
- **parking_lot locks**: Uses `parking_lot::Mutex` and `RwLock` instead of std for better contention handling
- **Atomic counters**: Lock-free vector count tracking via `AtomicUsize`
- **Mmap optimization**: Tracks mmap length atomically, auto-refreshes after flush

### Metadata Layer
- **LRU Cache**: 10,000 entry cache for metadata lookups (32x speedup on cache hit)
- **Write-through policy**: Cache updated on insert

### ID Filtering
- **`filter_ids`**: Uses `from_sorted_iter` for O(n) bitmap creation
- **`filter_id_range`**: Uses `insert_range` for O(1) contiguous range filters

## Scalar Quantization

Reduces memory usage by 75% by compressing float32 vectors to uint8.

### Configuration
```toml
[quantization]
enabled = true
keep_originals = false  # Set true for reranking with original vectors
rerank_oversample = 3   # Fetch 3x candidates for reranking
```

### How It Works
1. **Training**: Learns min/max per dimension online
2. **Encoding**: `uint8 = (float32 - min) / (max - min) * 255`
3. **Distance**: Asymmetric computation (query=float32, candidate=uint8)

### Memory Comparison
| Vectors | float32 | uint8 (quantized) | Savings |
|---------|---------|-------------------|---------|
| 10,000 | 5.12 MB | 1.28 MB | 75% |
| 100,000 | 51.2 MB | 12.8 MB | 75% |
| 1,000,000 | 512 MB | 128 MB | 75% |

## Payload Index

Inverted index for fast metadata filtering using Rust functional programming patterns.

### Configuration
```toml
[payload]
index_enabled = true
indexed_fields = ["category", "type", "status"]
```

### FilterQuery DSL
```rust
// Equality
FilterQuery::Eq { field: "category", value: "electronics" }

// In (multiple values)
FilterQuery::In { field: "status", values: vec!["active", "pending"] }

// Logical operators
FilterQuery::And(vec![...])
FilterQuery::Or(vec![...])
FilterQuery::Not(Box::new(...))
```

### Search Filtering Flow
1. **Indexed fields**: Pre-filter with PayloadIndex → RoaringBitmap → HNSW search
2. **Non-indexed fields**: HNSW search → Post-filter with metadata lookup
3. **Combined**: Bitmap intersection for multiple filter types

### Performance
| Filter Type | Complexity | Notes |
|-------------|------------|-------|
| Indexed field (Eq) | O(1) | Bitmap lookup |
| Indexed field (In) | O(k) | k = number of values |
| Non-indexed field | O(N) | Post-filter scan |
| Indexed + filter_ids | O(1) | Bitmap intersection |

### Functional Programming Patterns Used
- **Iterator chains**: `filter_map`, `fold`, `reduce`
- **Pattern matching**: FilterQuery enum dispatch
- **Closures**: Lazy evaluation in bitmap operations
- **Option/Result monads**: Safe error handling

## Performance Benchmarks

Tested with 128-dimensional vectors, ~2000 vectors, Apple Silicon.

### Sequential Operations (Single Thread)
| Operation | Throughput | Avg Latency | P99 |
|-----------|------------|-------------|-----|
| GET | 2,359 ops/s | 0.42ms | 1.22ms |
| Search (no filter) | 1,388 ops/s | 0.72ms | 1.55ms |
| Search + Non-indexed Filter | 1,521 ops/s | 0.66ms | 1.16ms |
| Search + filter_ids (500) | 914 ops/s | 1.09ms | 4.76ms |
| Search + filter_id_range | 804 ops/s | 1.24ms | 5.11ms |
| Search + Indexed Filter | 640 ops/s | 1.56ms | 8.38ms |
| Search + Multi Indexed | 440 ops/s | 2.27ms | 8.63ms |
| Upsert | 712 ops/s | 1.41ms | 2.98ms |

### Concurrent Operations (20 Threads)
| Operation | Throughput | Avg Latency | P99 |
|-----------|------------|-------------|-----|
| Search (no filter) | 3,610 ops/s | 5.30ms | 11.77ms |
| Mixed Workload (90R/10W) | 2,975 ops/s | 6.57ms | 16.89ms |
| Search + Indexed Filter | 2,405 ops/s | 8.06ms | 16.67ms |

### Concurrent Scaling
| Threads | Search Throughput | Search + Filter |
|---------|-------------------|-----------------|
| 10 | 2,946 ops/s | 2,207 ops/s |
| 20 | 3,610 ops/s | 2,405 ops/s |

### Benchmark Commands
```bash
# Basic benchmark
python benchmark.py

# Full benchmark with payload index tests
python benchmark_full.py
```
