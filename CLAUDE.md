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
   - `SledMetadataStore`: Embedded key-value store for metadata (`data/meta.sled`)
   - Vectors are append-only; IDs are auto-assigned sequentially starting from 0

2. **Index** (`src/index/mod.rs`)
   - `HnswIndexer`: Wraps `hnsw_rs` crate for approximate nearest neighbor search
   - Uses L2 (Euclidean) distance; for cosine similarity, normalize vectors before insertion
   - Supports filtered search via `RoaringBitmap`
   - Index persists to `data/index.hnsw.*` files on shutdown

3. **API** (`src/api/`)
   - `mod.rs`: gRPC service implementation (`VectorServiceImpl`)
   - `http.rs`: Axum HTTP handlers mirroring gRPC functionality
   - Both share the same underlying service logic

4. **Proto** (`src/proto/vectordb.proto`)
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

## Performance Benchmarks

Tested with 128-dimensional vectors on Apple Silicon.

### Throughput (20 concurrent threads)
| Operation | ops/sec | Avg Latency |
|-----------|---------|-------------|
| GET | 5,589 | 3.44ms |
| Search+Filter | 3,691 | 4.99ms |
| Search | 3,337 | 5.58ms |
| Upsert | 1,791 | 10.63ms |

### Mixed Workload (90% read, 10% write)
| Threads | Throughput |
|---------|------------|
| 10 | 3,252 ops/s |
| 20 | 3,606 ops/s |
| 50 | 3,287 ops/s |

### Sequential Operations
| Operation | Latency | Throughput |
|-----------|---------|------------|
| GET | 0.36ms | 2,812/s |
| Search | 0.78ms | 1,275/s |
| Upsert | 1.69ms | 593/s |
