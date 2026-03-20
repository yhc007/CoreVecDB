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
   - `FieldIndex`: Per-field string index using `RoaringBitmap` per unique value
   - `NumericFieldIndex`: BTreeMap-based index for range queries (O(log n))
   - `FilterQuery` DSL: `Eq`, `In`, `Gt`, `Gte`, `Lt`, `Lte`, `Range`, `And`, `Or`, `Not`
   - Pre-filtering in HNSW search for indexed fields
   - Automatic fallback to post-filtering for non-indexed fields

5. **Collection** (`src/collection/mod.rs`)
   - `Collection`: Independent vector space with its own storage, index, and metadata
   - `CollectionManager`: Manages multiple collections (create, delete, list)
   - `CollectionConfig`: Per-collection settings (dim, distance, quantization, indexed_fields)
   - Each collection stored in `data/<collection_name>/` directory

6. **API** (`src/api/`)
   - `mod.rs`: gRPC service implementation (`VectorServiceImpl`)
   - `http.rs`: Axum HTTP handlers with multi-collection support
   - Legacy endpoints use "default" collection for backward compatibility

7. **Proto** (`src/proto/vectordb.proto`)
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
- `payload.index_enabled`, `payload.indexed_fields`, `payload.numeric_fields`

Environment override: `APP_SERVER__GRPC_PORT=50052` etc.

## HTTP API Endpoints

### Collection Management
- `GET /collections` - List all collections
- `POST /collections` - Create new collection
- `GET /collections/:name` - Get collection info
- `DELETE /collections/:name` - Delete collection

### Vector Operations (with collection)
- `POST /collections/:name/upsert` - Insert single vector
- `POST /collections/:name/upsert_batch` - Batch insert vectors (~10x faster)
- `POST /collections/:name/search` - Search vectors
- `GET /collections/:name/vectors/:id` - Get vector by ID
- `PUT /collections/:name/vectors/:id` - Update vector (delete + insert, new ID)
- `PATCH /collections/:name/vectors/:id` - Update metadata only (same ID)
- `DELETE /collections/:name/vectors/:id` - Delete vector (soft delete)
- `POST /collections/:name/delete_batch` - Batch delete vectors

### Snapshot Operations
- `GET /collections/:name/snapshots` - List snapshots
- `POST /collections/:name/snapshots` - Create snapshot
- `POST /snapshots/:snapshot_name/restore` - Restore from snapshot
- `DELETE /snapshots/:snapshot_name` - Delete snapshot

### Legacy Endpoints (uses "default" collection)
- `POST /upsert` - Insert vector with metadata
- `POST /upsert_batch` - Batch insert vectors
- `POST /search` - k-NN search with optional metadata/ID filtering
- `GET /vectors/:id` - Retrieve vector by ID
- `GET /stats` - Get vector count and status

Static files served from `ui/` directory at root

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
indexed_fields = ["category", "type", "status"]  # String fields
numeric_fields = ["price", "rating", "timestamp"] # Numeric fields (range queries)
```

### FilterQuery DSL

#### String Filters
```rust
// Equality
FilterQuery::eq("category", "electronics")

// In (multiple values)
FilterQuery::in_set("status", ["active", "pending"])
```

#### Range Filters (Numeric)
```rust
// Greater than: price > 100
FilterQuery::gt("price", 100)

// Greater than or equal: price >= 100
FilterQuery::gte("price", 100)

// Less than: price < 500
FilterQuery::lt("price", 500)

// Less than or equal: price <= 500
FilterQuery::lte("price", 500)

// Range (inclusive): 100 <= price <= 500
FilterQuery::range("price", 100, 500)

// Float variants (auto-converts to i64 with 6 decimal precision)
FilterQuery::gte_f("rating", 4.0)
FilterQuery::range_f("score", 0.5, 0.95)
```

#### Logical Operators
```rust
// AND: category=electronics AND price < 300
FilterQuery::and(vec![
    FilterQuery::eq("category", "electronics"),
    FilterQuery::lt("price", 300),
])

// OR: status=active OR status=pending
FilterQuery::or(vec![
    FilterQuery::eq("status", "active"),
    FilterQuery::eq("status", "pending"),
])

// NOT (requires universe bitmap)
FilterQuery::not(FilterQuery::eq("status", "deleted"))
```

### Search Filtering Flow
1. **Indexed fields**: Pre-filter with PayloadIndex → RoaringBitmap → HNSW search
2. **Non-indexed fields**: HNSW search → Post-filter with metadata lookup
3. **Combined**: Bitmap intersection for multiple filter types

### HTTP Range Filter API

Numeric range queries are available directly through the HTTP API.

#### Request Format
```json
{
  "vector": [...],
  "k": 10,
  "filter": {"category": "electronics"},
  "range_filters": [
    {"op": "gt", "field": "price", "value": 100},
    {"op": "lte", "field": "rating", "value": 5.0}
  ]
}
```

#### Supported Operators
| Operator | JSON Format | Description |
|----------|-------------|-------------|
| `gt` | `{"op": "gt", "field": "price", "value": 100}` | price > 100 |
| `gte` | `{"op": "gte", "field": "price", "value": 100}` | price >= 100 |
| `lt` | `{"op": "lt", "field": "price", "value": 100}` | price < 100 |
| `lte` | `{"op": "lte", "field": "price", "value": 100}` | price <= 100 |
| `range` | `{"op": "range", "field": "price", "min": 50, "max": 200}` | 50 <= price <= 200 |
| `between` | `{"op": "between", "field": "price", "min": 50, "max": 200}` | 50 < price < 200 |

#### Example Usage
```bash
# Find products with price > 100 AND rating >= 4.0
curl -X POST http://localhost:3000/collections/products/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [...],
    "k": 10,
    "filter": {},
    "range_filters": [
      {"op": "gt", "field": "price", "value": 100},
      {"op": "gte", "field": "rating", "value": 4.0}
    ]
  }'

# Combine with string filter
curl -X POST http://localhost:3000/collections/products/search \
  -d '{
    "vector": [...],
    "k": 10,
    "filter": {"category": "electronics"},
    "range_filters": [
      {"op": "range", "field": "price", "min": 50, "max": 500}
    ]
  }'
```

### Performance
| Filter Type | Complexity | Notes |
|-------------|------------|-------|
| String field (Eq) | O(1) | HashMap lookup |
| String field (In) | O(k) | k = number of values |
| Numeric range (Gt/Lt/Range) | O(log n + m) | BTreeMap range scan, m = matches |
| Non-indexed field | O(N) | Post-filter scan |
| Combined filters | O(1) | Bitmap intersection |

### Functional Programming Patterns Used
- **Iterator chains**: `filter_map`, `fold`, `reduce`
- **Pattern matching**: FilterQuery enum dispatch
- **Closures**: Lazy evaluation in bitmap operations
- **Option/Result monads**: Safe error handling

## Multi-Collection

Each collection is an independent vector space with its own storage, index, and configuration.

### Create Collection
```bash
curl -X POST http://localhost:3000/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "products",
    "dim": 128,
    "distance": "euclidean",
    "quantization_enabled": false,
    "indexed_fields": ["category", "brand"],
    "numeric_fields": ["price", "rating"]
  }'
```

### Collection Operations
```bash
# List all collections
curl http://localhost:3000/collections

# Get collection info
curl http://localhost:3000/collections/products

# Delete collection
curl -X DELETE http://localhost:3000/collections/products
```

### Vector Operations with Collection
```bash
# Upsert to specific collection
curl -X POST http://localhost:3000/collections/products/upsert \
  -d '{"vector": [...], "metadata": {"category": "electronics"}}'

# Search in specific collection
curl -X POST http://localhost:3000/collections/products/search \
  -d '{"vector": [...], "k": 10, "filter": {"category": "electronics"}}'

# Get vector from collection
curl http://localhost:3000/collections/products/vectors/0
```

### Directory Structure
```
data/
├── products/           # "products" collection
│   ├── config.json     # Collection configuration
│   ├── vectors.bin     # Vector storage
│   ├── index.hnsw.*    # HNSW index files
│   └── meta.sled/      # Metadata store
├── users/              # "users" collection
│   └── ...
└── default/            # Default collection (legacy endpoints)
    └── ...
```

### Collection Config Options
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Collection name |
| `dim` | int | required | Vector dimension |
| `distance` | string | "euclidean" | Distance metric (euclidean, cosine) |
| `quantization_enabled` | bool | false | Enable scalar quantization |
| `indexed_fields` | string[] | [] | String fields for exact match filtering |
| `numeric_fields` | string[] | [] | Numeric fields for range queries |

## Update Vector

Update vectors or metadata. Two methods available:

### PUT - Full Vector Update
Replaces the vector and metadata. Since storage is append-only, this creates a new ID.

```bash
curl -X PUT http://localhost:3000/collections/products/vectors/123 \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, ...],
    "metadata": {"category": "electronics", "price": "299"}
  }'
```

Response:
```json
{
  "old_id": 123,
  "new_id": 456,
  "collection": "products",
  "updated": true
}
```

**Note**: The old ID is soft-deleted, and a new ID is assigned. Update your references to use the new ID.

### PATCH - Metadata Only Update
Updates metadata without changing the vector. Keeps the same ID.

```bash
curl -X PATCH http://localhost:3000/collections/products/vectors/123 \
  -H "Content-Type: application/json" \
  -d '{
    "metadata": {"price": "249", "on_sale": "true"}
  }'
```

Response:
```json
{
  "id": 123,
  "collection": "products",
  "updated": true
}
```

### When to Use Which
| Method | Use Case | ID Changes |
|--------|----------|------------|
| PUT | Change vector values | Yes (new ID) |
| PATCH | Change metadata only | No (same ID) |

### HTTP Status Codes
| Code | Meaning |
|------|---------|
| 200 | Successfully updated |
| 404 | Vector ID not found |
| 410 | Vector already deleted |

## Delete Vector

Soft delete vectors using tombstones. Deleted vectors are excluded from search results.

### API Endpoints
```bash
# Delete single vector
DELETE /collections/:name/vectors/:id

# Batch delete
POST /collections/:name/delete_batch
```

### Delete Single Vector
```bash
curl -X DELETE http://localhost:3000/collections/products/vectors/123
```

Response:
```json
{"id": 123, "collection": "products", "deleted": true}
```

### Batch Delete
```bash
curl -X POST http://localhost:3000/collections/products/delete_batch \
  -H "Content-Type: application/json" \
  -d '{"ids": [1, 2, 3, 4, 5]}'
```

Response:
```json
{"collection": "products", "deleted_count": 5, "requested_count": 5}
```

### HTTP Status Codes
| Code | Meaning |
|------|---------|
| 200 | Successfully deleted |
| 404 | Vector ID not found |
| 410 | Already deleted (Gone) |

### Collection Info with Delete Count
```json
{
  "name": "products",
  "dim": 128,
  "vector_count": 950,    // Active vectors
  "deleted_count": 50,    // Deleted vectors
  ...
}
```

### Implementation Details
- **Soft delete**: Uses RoaringBitmap to track deleted IDs
- **Persisted**: Deleted IDs saved to `deleted.bin` on flush
- **Search filter**: Deleted IDs automatically excluded from search results
- **Get returns 410**: Attempting to get deleted vector returns HTTP 410 Gone

### Storage
```
data/products/
├── vectors.bin       # Vector data (includes deleted)
├── deleted.bin       # Deleted ID bitmap
└── ...
```

## Snapshot/Backup

Create and restore backups of collections.

### API Endpoints
```bash
# List snapshots for a collection
GET /collections/:name/snapshots

# Create a snapshot
POST /collections/:name/snapshots

# Restore from snapshot
POST /snapshots/:snapshot_name/restore

# Delete a snapshot
DELETE /snapshots/:snapshot_name
```

### Create Snapshot
```bash
curl -X POST http://localhost:3000/collections/products/snapshots
```

Response:
```json
{
  "name": "products_20240320_143052",
  "collection": "products",
  "created_at": "2024-03-20T14:30:52Z",
  "vector_count": 10000,
  "size_bytes": 5242880
}
```

### List Snapshots
```bash
curl http://localhost:3000/collections/products/snapshots
```

### Restore from Snapshot
```bash
# Restore with original name (collection must not exist)
curl -X POST http://localhost:3000/snapshots/products_20240320_143052/restore \
  -H "Content-Type: application/json" \
  -d '{}'

# Restore with new name
curl -X POST http://localhost:3000/snapshots/products_20240320_143052/restore \
  -d '{"new_name": "products_restored"}'
```

### Delete Snapshot
```bash
curl -X DELETE http://localhost:3000/snapshots/products_20240320_143052
```

### Storage Structure
```
data/
├── products/           # Active collection
│   └── ...
└── _snapshots/         # Snapshot storage
    ├── products_20240320_143052/
    │   ├── snapshot.json   # Metadata
    │   ├── config.json
    │   ├── vectors.bin
    │   ├── index.hnsw.*
    │   └── meta.sled/
    └── products_20240319_120000/
        └── ...
```

### Use Cases
- **Backup before risky operations**: Create snapshot before bulk deletes
- **Point-in-time recovery**: Restore to previous state
- **Clone collections**: Restore with new name for testing
- **Migration**: Move collections between environments

## Batch Upsert

Bulk insert API for ~10x performance improvement over individual inserts.

### API Endpoints
```bash
# Batch upsert to specific collection
POST /collections/:name/upsert_batch

# Batch upsert using default collection
POST /upsert_batch
```

### Request Format
```json
{
  "vectors": [
    {
      "vector": [0.1, 0.2, ...],
      "metadata": {"category": "A", "price": "100"}
    },
    {
      "vector": [0.3, 0.4, ...],
      "metadata": {"category": "B", "price": "200"}
    }
  ]
}
```

### Response Format
```json
{
  "start_id": 0,
  "count": 2,
  "success": true
}
```

### Usage Example
```bash
# Insert 100 vectors at once
curl -X POST http://localhost:3000/collections/products/upsert_batch \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      {"vector": [...], "metadata": {"category": "electronics"}},
      {"vector": [...], "metadata": {"category": "books"}},
      ...
    ]
  }'
```

### Python Example
```python
import requests

vectors = [
    {"vector": [random.random() for _ in range(128)], "metadata": {"idx": str(i)}}
    for i in range(1000)
]

resp = requests.post(
    "http://localhost:3000/collections/products/upsert_batch",
    json={"vectors": vectors}
)
print(resp.json())  # {"start_id": 0, "count": 1000, "success": true}
```

### Performance Optimizations
| Layer | Optimization |
|-------|-------------|
| **VectorStore** | Single flush after batch (vs per-vector) |
| **MetadataStore** | Atomic sled::Batch write |
| **HnswIndexer** | Uses `parallel_insert` for multi-threaded indexing |
| **PayloadIndex** | Batch index updates |

### Benchmark (100 vectors, 128-dim)
| Method | Time | Throughput |
|--------|------|------------|
| Individual inserts | ~1.5s | ~65 vectors/sec |
| Batch upsert | ~0.15s | ~650 vectors/sec |
| **Speedup** | **~10x** | |

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
