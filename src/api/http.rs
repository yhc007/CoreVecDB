use axum::{
    routing::{get, post, delete, put, patch},
    Router, Json, extract::{State, Path},
    http::StatusCode,
    response::sse::{Event, Sse},
};
use std::sync::Arc;
use std::convert::Infallible;
use futures::stream::Stream;
use crate::api::{create_filter_bitmap, create_range_bitmap};
use crate::collection::{CollectionManager, CollectionConfig, CollectionInfo, SnapshotInfo, WalConfigWrapper};
use crate::storage::{MetadataEntry, IndexedMetadata};
use crate::payload::FilterQuery;
use crate::replication::{ReplicationManager, ReplicationRole};
use crate::metrics;
use crate::cache::{QueryCacheKey, FilterBitmapCache};
use serde::{Deserialize, Serialize};

// Wrapper structs for JSON compatibility if proto structs don't behave well with Serde automatically
// (Prost types usually need ::serde feature or separate structs if we want clean JSON)
// For MVP we'll try to use the Proto structs if they deserialize well, or define DTOs.
// Tonic/Prost 0.12 might not derive Serialize/Deserialize by default unless configured.
// We accepted `serde` features in Cargo.toml? No, we didn't add `serde` feature to `prost` in `Cargo.toml`.
// Let's rely on mapping.

#[derive(Deserialize)]
pub struct JsonUpsertReq {
    id: u64,
    vector: Vec<f32>,
    metadata: std::collections::HashMap<String, String>,
}

/// Single vector in batch upsert request.
#[derive(Deserialize)]
pub struct BatchVectorReq {
    vector: Vec<f32>,
    #[serde(default)]
    metadata: std::collections::HashMap<String, String>,
}

/// Batch upsert request - insert multiple vectors at once.
#[derive(Deserialize)]
pub struct JsonBatchUpsertReq {
    vectors: Vec<BatchVectorReq>,
}

/// Batch upsert response.
#[derive(Serialize)]
pub struct JsonBatchUpsertResp {
    start_id: u64,
    count: usize,
    success: bool,
}

/// Range filter for numeric fields.
#[derive(Deserialize, Clone)]
#[serde(tag = "op")]
pub enum RangeFilter {
    /// Greater than: field > value
    #[serde(rename = "gt")]
    Gt { field: String, value: f64 },
    /// Greater than or equal: field >= value
    #[serde(rename = "gte")]
    Gte { field: String, value: f64 },
    /// Less than: field < value
    #[serde(rename = "lt")]
    Lt { field: String, value: f64 },
    /// Less than or equal: field <= value
    #[serde(rename = "lte")]
    Lte { field: String, value: f64 },
    /// Range: min <= field <= max (inclusive)
    #[serde(rename = "range")]
    Range { field: String, min: f64, max: f64 },
    /// Between: min < field < max (exclusive)
    #[serde(rename = "between")]
    Between { field: String, min: f64, max: f64 },
}

impl RangeFilter {
    /// Convert to FilterQuery for use with PayloadIndex.
    fn to_filter_query(&self) -> FilterQuery {
        match self {
            RangeFilter::Gt { field, value } => FilterQuery::gt_f(field, *value),
            RangeFilter::Gte { field, value } => FilterQuery::gte_f(field, *value),
            RangeFilter::Lt { field, value } => FilterQuery::lt_f(field, *value),
            RangeFilter::Lte { field, value } => FilterQuery::lte_f(field, *value),
            RangeFilter::Range { field, min, max } => FilterQuery::range_f(field, *min, *max),
            RangeFilter::Between { field, min, max } => {
                // Between is min < field < max, using AND of Gt and Lt
                FilterQuery::and(vec![
                    FilterQuery::gt_f(field, *min),
                    FilterQuery::lt_f(field, *max),
                ])
            }
        }
    }
}

/// Apply range filters using IndexedMetadata.
/// Returns bitmap of matching IDs.
fn apply_range_filters(
    metadata_store: &dyn IndexedMetadata,
    range_filters: &[RangeFilter],
) -> Option<roaring::RoaringBitmap> {
    if range_filters.is_empty() {
        return None;
    }

    // Convert to FilterQuery and combine with AND
    let queries: Vec<FilterQuery> = range_filters
        .iter()
        .map(|rf| rf.to_filter_query())
        .collect();

    let combined = if queries.len() == 1 {
        queries.into_iter().next().unwrap()
    } else {
        FilterQuery::and(queries)
    };

    metadata_store.filter(&combined)
}

#[derive(Deserialize)]
pub struct JsonSearchReq {
    vector: Vec<f32>,
    k: u32,
    /// String field exact match filters (AND).
    #[serde(default)]
    filter: std::collections::HashMap<String, String>,
    /// Numeric range filters (AND with other filters).
    #[serde(default)]
    range_filters: Vec<RangeFilter>,
    filter_ids: Option<Vec<u64>>,
    /// Range filter: [start, end] inclusive. More efficient than listing all IDs.
    filter_id_range: Option<(u64, u64)>,
    /// Include metadata in search results (default: false).
    #[serde(default)]
    include_metadata: bool,
}

#[derive(Serialize)]
pub struct JsonSearchResp {
    results: Vec<JsonSearchResult>,
}

#[derive(Serialize)]
pub struct JsonSearchResult {
    id: u64,
    score: f32,
    /// Metadata (only present when include_metadata=true).
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<std::collections::HashMap<String, String>>,
}

/// Streaming search request for large result sets.
#[derive(Deserialize)]
pub struct StreamSearchReq {
    vector: Vec<f32>,
    k: u32,
    #[serde(default)]
    filter: std::collections::HashMap<String, String>,
    #[serde(default)]
    range_filters: Vec<RangeFilter>,
    #[serde(default)]
    filter_ids: Option<Vec<u64>>,
    /// Results per batch (default: 100)
    #[serde(default)]
    batch_size: Option<u32>,
    /// Include vectors in results
    #[serde(default)]
    include_vectors: bool,
    /// Include metadata in results
    #[serde(default)]
    include_metadata: bool,
}

/// Streaming search result batch.
#[derive(Serialize)]
pub struct StreamSearchResultBatch {
    results: Vec<StreamSearchResultItem>,
    batch_index: u32,
    total_batches: u32,
    results_so_far: u32,
    total_results: u32,
    is_last: bool,
}

/// Single result in streaming search.
#[derive(Serialize)]
pub struct StreamSearchResultItem {
    id: u64,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    vector: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<std::collections::HashMap<String, String>>,
}

// ============================================================================
// Multi-Collection API
// ============================================================================

/// State for collection-aware router.
pub struct CollectionAppState {
    pub manager: Arc<CollectionManager>,
    pub replication: Option<Arc<ReplicationManager>>,
}

/// Request to create a new collection.
#[derive(Deserialize)]
pub struct CreateCollectionReq {
    name: String,
    dim: usize,
    #[serde(default)]
    distance: Option<String>,
    #[serde(default)]
    quantization_enabled: Option<bool>,
    #[serde(default)]
    indexed_fields: Option<Vec<String>>,
    #[serde(default)]
    numeric_fields: Option<Vec<String>>,
    /// Text fields for BM25 full-text search
    #[serde(default)]
    text_fields: Option<Vec<String>>,
    /// Enable WAL for crash recovery (default: true)
    #[serde(default)]
    wal_enabled: Option<bool>,
    /// Sync WAL to disk on each write (safer but slower)
    #[serde(default)]
    wal_sync_on_write: Option<bool>,
}

/// Hybrid search request (vector + text).
#[derive(Deserialize)]
pub struct HybridSearchReq {
    vector: Vec<f32>,
    query: String,
    k: u32,
    /// Weight for vector score (0.0-1.0, default: 0.5)
    #[serde(default)]
    alpha: Option<f32>,
    /// Fusion method: "weighted" or "rrf" (default: "weighted")
    #[serde(default)]
    fusion_method: Option<String>,
    #[serde(default)]
    filter: std::collections::HashMap<String, String>,
    #[serde(default)]
    range_filters: Vec<RangeFilter>,
    #[serde(default)]
    include_scores: bool,
    #[serde(default)]
    include_metadata: bool,
}

/// Hybrid search result.
#[derive(Serialize)]
pub struct HybridSearchResult {
    id: u64,
    combined_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    vector_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<std::collections::HashMap<String, String>>,
}

/// Hybrid search response.
#[derive(Serialize)]
pub struct HybridSearchResp {
    results: Vec<HybridSearchResult>,
    total_candidates: u32,
}

/// Text-only search request using BM25.
#[derive(Deserialize)]
pub struct TextSearchReq {
    query: String,
    k: u32,
    #[serde(default)]
    filter: std::collections::HashMap<String, String>,
    #[serde(default)]
    must_match_all: bool,
    #[serde(default)]
    include_metadata: bool,
}

/// Text search result.
#[derive(Serialize)]
pub struct TextSearchResultItem {
    id: u64,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<std::collections::HashMap<String, String>>,
}

/// Text search response.
#[derive(Serialize)]
pub struct TextSearchResp {
    results: Vec<TextSearchResultItem>,
}

// =============================================================================
// Multi-Vector Search Types
// =============================================================================

/// Multi-vector search request.
#[derive(Deserialize)]
pub struct MultiVectorSearchReq {
    /// Multiple query vectors
    vectors: Vec<Vec<f32>>,
    /// Number of results to return
    k: u32,
    /// Fusion method: "sum", "max", "avg", "rrf", "min"
    #[serde(default = "default_fusion")]
    fusion: String,
    /// Optional weights for each vector (for weighted fusion)
    #[serde(default)]
    weights: Option<Vec<f32>>,
    /// RRF k parameter (default 60)
    #[serde(default = "default_rrf_k")]
    rrf_k: f32,
    /// Oversample factor (fetch more candidates per query)
    #[serde(default = "default_oversample")]
    oversample: u32,
    /// Optional metadata filter
    #[serde(default)]
    filter: std::collections::HashMap<String, String>,
    /// Include per-query scores in response
    #[serde(default)]
    include_per_query_scores: bool,
    /// Include metadata in results
    #[serde(default)]
    include_metadata: bool,
}

fn default_fusion() -> String { "rrf".to_string() }
fn default_rrf_k() -> f32 { 60.0 }
fn default_oversample() -> u32 { 2 }

/// Multi-vector search result.
#[derive(Serialize)]
pub struct MultiVectorSearchResult {
    id: u64,
    score: f32,
    query_matches: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    per_query_scores: Option<Vec<Option<f32>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<std::collections::HashMap<String, String>>,
}

/// Multi-vector search response.
#[derive(Serialize)]
pub struct MultiVectorSearchResp {
    results: Vec<MultiVectorSearchResult>,
    num_queries: usize,
    fusion_method: String,
}

// =============================================================================
// Sparse Vector Types
// =============================================================================

/// Sparse vector upsert request.
#[derive(Deserialize)]
pub struct SparseUpsertReq {
    /// Dimension indices (non-zero positions)
    indices: Vec<u32>,
    /// Values at each index
    values: Vec<f32>,
    /// Optional metadata
    #[serde(default)]
    metadata: std::collections::HashMap<String, String>,
}

/// Batch sparse vector upsert request.
#[derive(Deserialize)]
pub struct SparseBatchUpsertReq {
    vectors: Vec<SparseUpsertReq>,
}

/// Sparse search request.
#[derive(Deserialize)]
pub struct SparseSearchReq {
    /// Query vector indices
    indices: Vec<u32>,
    /// Query vector values
    values: Vec<f32>,
    /// Number of results
    k: u32,
    /// Optional metadata filter
    #[serde(default)]
    filter: std::collections::HashMap<String, String>,
    /// Include metadata in results
    #[serde(default)]
    include_metadata: bool,
}

/// Sparse search result.
#[derive(Serialize)]
pub struct SparseSearchResult {
    id: u64,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<std::collections::HashMap<String, String>>,
}

/// Sparse search response.
#[derive(Serialize)]
pub struct SparseSearchResp {
    results: Vec<SparseSearchResult>,
}

/// Hybrid dense + sparse search request.
#[derive(Deserialize)]
pub struct HybridDenseSparseReq {
    /// Dense query vector
    dense_vector: Vec<f32>,
    /// Sparse query indices
    sparse_indices: Vec<u32>,
    /// Sparse query values
    sparse_values: Vec<f32>,
    /// Number of results
    k: u32,
    /// Weight for dense score (0.0 to 1.0)
    #[serde(default = "default_dense_weight")]
    dense_weight: f32,
    /// Weight for sparse score (0.0 to 1.0)
    #[serde(default = "default_sparse_weight")]
    sparse_weight: f32,
    /// Fusion method: "weighted", "rrf", "max"
    #[serde(default = "default_hybrid_fusion")]
    fusion: String,
    /// Optional metadata filter
    #[serde(default)]
    filter: std::collections::HashMap<String, String>,
    /// Include component scores in results
    #[serde(default)]
    include_scores: bool,
    /// Include metadata in results
    #[serde(default)]
    include_metadata: bool,
}

fn default_dense_weight() -> f32 { 0.5 }
fn default_sparse_weight() -> f32 { 0.5 }
fn default_hybrid_fusion() -> String { "weighted".to_string() }

/// Hybrid search result.
#[derive(Serialize)]
pub struct HybridDenseSparseResult {
    id: u64,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    dense_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sparse_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<std::collections::HashMap<String, String>>,
}

/// Hybrid search response.
#[derive(Serialize)]
pub struct HybridDenseSparseResp {
    results: Vec<HybridDenseSparseResult>,
    fusion_method: String,
}

// =============================================================================
// Versioning Types
// =============================================================================

/// Versioned upsert request.
#[derive(Deserialize)]
pub struct VersionedUpsertReq {
    /// Vector ID to upsert
    vector_id: u64,
    /// Vector data
    vector: Vec<f32>,
    /// Metadata
    #[serde(default)]
    metadata: std::collections::HashMap<String, String>,
    /// Optional change description
    #[serde(default)]
    description: Option<String>,
}

/// Get version at timestamp request.
#[derive(Deserialize)]
pub struct GetAtTimestampReq {
    /// ISO 8601 timestamp
    timestamp: String,
}

/// Rollback request.
#[derive(Deserialize)]
pub struct RollbackReq {
    /// Version ID to rollback to
    to_version_id: u64,
}

/// Snapshot at timestamp request.
#[derive(Deserialize)]
pub struct SnapshotAtReq {
    /// ISO 8601 timestamp
    timestamp: String,
}

/// Version info response.
#[derive(Serialize)]
pub struct VersionInfoResp {
    version_id: u64,
    vector_id: u64,
    timestamp: String,
    is_deleted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    change_description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vector: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<std::collections::HashMap<String, String>>,
}

impl From<crate::versioning::VectorVersion> for VersionInfoResp {
    fn from(v: crate::versioning::VectorVersion) -> Self {
        Self {
            version_id: v.version_id,
            vector_id: v.vector_id,
            timestamp: v.timestamp.to_rfc3339(),
            is_deleted: v.is_deleted,
            change_description: v.change_description,
            vector: if v.is_deleted { None } else { Some(v.vector) },
            metadata: if v.metadata.is_empty() { None } else { Some(v.metadata) },
        }
    }
}

/// Create router with multi-collection support.
pub async fn collection_router(manager: Arc<CollectionManager>) -> Router {
    collection_router_with_replication(manager, None).await
}

/// Create router with multi-collection and replication support.
pub async fn collection_router_with_replication(
    manager: Arc<CollectionManager>,
    replication: Option<Arc<ReplicationManager>>,
) -> Router {
    let state = Arc::new(CollectionAppState { manager, replication });

    Router::new()
        // Collection management
        .route("/collections", get(list_collections))
        .route("/collections", post(create_collection))
        .route("/collections/:name", get(get_collection_info))
        .route("/collections/:name", delete(delete_collection))
        // Vector operations with collection
        .route("/collections/:name/upsert", post(collection_upsert))
        .route("/collections/:name/upsert_batch", post(collection_upsert_batch))
        .route("/collections/:name/search", post(collection_search))
        .route("/collections/:name/vectors/:id", get(collection_get_vector))
        .route("/collections/:name/vectors/:id", put(collection_update_vector))
        .route("/collections/:name/vectors/:id", patch(collection_update_metadata))
        .route("/collections/:name/vectors/:id", delete(collection_delete_vector))
        .route("/collections/:name/delete_batch", post(collection_delete_batch))
        // Streaming upsert with progress (SSE)
        .route("/collections/:name/stream_upsert", post(collection_stream_upsert))
        // Streaming search with SSE (for large k values)
        .route("/collections/:name/stream_search", post(collection_stream_search))
        // Hybrid search (vector + text BM25)
        .route("/collections/:name/hybrid_search", post(collection_hybrid_search))
        // Text-only search (BM25)
        .route("/collections/:name/text_search", post(collection_text_search))
        // Multi-vector search
        .route("/collections/:name/multi_search", post(collection_multi_search))
        // Sparse vector operations
        .route("/collections/:name/sparse_upsert", post(collection_sparse_upsert))
        .route("/collections/:name/sparse_upsert_batch", post(collection_sparse_upsert_batch))
        .route("/collections/:name/sparse_search", post(collection_sparse_search))
        .route("/collections/:name/hybrid_dense_sparse", post(collection_hybrid_dense_sparse))
        // Versioning operations
        .route("/collections/:name/versioned/upsert", post(versioned_upsert))
        .route("/collections/:name/versioned/vectors/:id", get(versioned_get))
        .route("/collections/:name/versioned/vectors/:id/history", get(versioned_history))
        .route("/collections/:name/versioned/vectors/:id/at", post(versioned_at_timestamp))
        .route("/collections/:name/versioned/vectors/:id/rollback", post(versioned_rollback))
        .route("/collections/:name/versioned/snapshot", post(versioned_snapshot))
        .route("/collections/:name/versioned/stats", get(versioned_stats))
        // Snapshot operations
        .route("/collections/:name/snapshots", get(list_snapshots))
        .route("/collections/:name/snapshots", post(create_snapshot))
        .route("/snapshots/:snapshot_name", delete(delete_snapshot))
        .route("/snapshots/:snapshot_name/restore", post(restore_snapshot))
        // Maintenance operations
        .route("/collections/:name/compact", post(collection_compact))
        .route("/collections/:name/checkpoint", post(collection_checkpoint))
        .route("/collections/:name/wal_stats", get(collection_wal_stats))
        .route("/collections/:name/cache_stats", get(collection_cache_stats))
        .route("/checkpoint_all", post(checkpoint_all))
        // Replication status
        .route("/replication/status", get(replication_status))
        .route("/replication/wal", get(get_wal_entries))
        // Metrics
        .route("/metrics", get(prometheus_metrics))
        // SIMD info
        .route("/simd", get(simd_info))
        // Sharding management
        .route("/sharding/config", get(get_sharding_config))
        .route("/sharding/config", post(update_sharding_config))
        .route("/sharding/shards", get(list_shards))
        .route("/sharding/shards/:shard_id", get(get_shard_info))
        .route("/sharding/shards", post(add_shard))
        .route("/sharding/shards/:shard_id", delete(remove_shard))
        .route("/sharding/route/:vector_id", get(get_shard_for_id))
        // Legacy routes (use default collection)
        .route("/upsert", post(legacy_upsert))
        .route("/upsert_batch", post(legacy_upsert_batch))
        .route("/search", post(legacy_search))
        .route("/vectors/:id", get(legacy_get_vector))
        .route("/stats", get(legacy_stats))
        .layer(
            tower_http::cors::CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_methods(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any),
        )
        .nest_service("/", tower_http::services::ServeDir::new("ui"))
        .with_state(state)
}

// ============================================================================
// Collection Management Handlers
// ============================================================================

async fn list_collections(
    State(state): State<Arc<CollectionAppState>>,
) -> Json<Vec<CollectionInfo>> {
    Json(state.manager.list())
}

async fn create_collection(
    State(state): State<Arc<CollectionAppState>>,
    Json(req): Json<CreateCollectionReq>,
) -> Result<Json<CollectionInfo>, StatusCode> {
    let mut config = CollectionConfig::new(&req.name, req.dim);

    if let Some(ref distance) = req.distance {
        config = config.with_distance(distance);
    }

    if let Some(enabled) = req.quantization_enabled {
        config = config.with_quantization(enabled, true);
    }

    if req.indexed_fields.is_some() || req.numeric_fields.is_some() || req.text_fields.is_some() {
        config.payload.indexed_fields = req.indexed_fields.unwrap_or_default();
        config.payload.numeric_fields = req.numeric_fields.unwrap_or_default();
        config.payload.text_fields = req.text_fields.unwrap_or_default();
    }

    // Configure WAL settings
    config.wal = WalConfigWrapper {
        enabled: req.wal_enabled.unwrap_or(true),
        sync_on_write: req.wal_sync_on_write.unwrap_or(false),
        checkpoint_interval: 10000, // default
    };

    state.manager.create(config)
        .map(|c| Json(c.info()))
        .map_err(|_| StatusCode::CONFLICT)
}

async fn get_collection_info(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
) -> Result<Json<CollectionInfo>, StatusCode> {
    state.manager.get(&name)
        .map(|c| Json(c.info()))
        .ok_or(StatusCode::NOT_FOUND)
}

async fn delete_collection(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    state.manager.delete(&name)
        .map(|_| Json(serde_json::json!({ "deleted": true, "name": name })))
        .map_err(|_| StatusCode::NOT_FOUND)
}

// ============================================================================
// Collection Vector Operations
// ============================================================================

async fn collection_upsert(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<JsonUpsertReq>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let internal_id = collection.vector_store.insert(&payload.vector)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    collection.indexer.insert(internal_id, &payload.vector)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    for (k, v) in payload.metadata {
        collection.metadata_store.insert(internal_id, k, v)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }

    Ok(Json(serde_json::json!({
        "id": internal_id,
        "collection": name,
        "success": true
    })))
}

/// Batch upsert - insert multiple vectors at once.
/// ~10x faster than individual upserts for large batches.
async fn collection_upsert_batch(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<JsonBatchUpsertReq>,
) -> Result<Json<JsonBatchUpsertResp>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    if payload.vectors.is_empty() {
        return Ok(Json(JsonBatchUpsertResp {
            start_id: collection.len() as u64,
            count: 0,
            success: true,
        }));
    }

    // Extract vectors for batch insert
    let vectors: Vec<Vec<f32>> = payload.vectors
        .iter()
        .map(|v| v.vector.clone())
        .collect();

    // Batch insert to vector store
    let start_id = collection.vector_store.insert_batch(&vectors)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Prepare index data: (id, vector)
    let index_data: Vec<(u64, Vec<f32>)> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, v)| (start_id + i as u64, v))
        .collect();

    // Batch insert to HNSW index
    collection.indexer.insert_batch(&index_data)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Collect metadata entries
    let metadata_entries: Vec<MetadataEntry> = payload.vectors
        .iter()
        .enumerate()
        .flat_map(|(i, v)| {
            let id = start_id + i as u64;
            v.metadata.iter().map(move |(k, val)| MetadataEntry {
                id,
                key: k.clone(),
                value: val.clone(),
            })
        })
        .collect();

    // Batch insert metadata
    if !metadata_entries.is_empty() {
        collection.metadata_store.insert_batch(&metadata_entries)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }

    Ok(Json(JsonBatchUpsertResp {
        start_id,
        count: payload.vectors.len(),
        success: true,
    }))
}

async fn collection_search(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<JsonSearchReq>,
) -> Result<Json<JsonSearchResp>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    // Check query cache first (if no metadata post-filtering needed)
    let cache_key = QueryCacheKey::new(
        &name,
        &payload.vector,
        payload.k as usize,
        Some(&payload.filter),
    );
    let current_vector_count = collection.len();

    // Only use cache if no range filters, filter_ids, or filter_id_range (complex filters)
    let use_cache = payload.range_filters.is_empty()
        && payload.filter_ids.is_none()
        && payload.filter_id_range.is_none();

    if use_cache {
        if let Some(ref query_cache) = collection.query_cache() {
            if let Some(cached_results) = query_cache.get(&cache_key, current_vector_count) {
                // Cache hit - return cached results
                let include_meta = payload.include_metadata;
                let final_results: Vec<JsonSearchResult> = cached_results
                    .into_iter()
                    .map(|(id, score)| {
                        let metadata = if include_meta {
                            Some(collection.metadata_store.get_all(id))
                        } else {
                            None
                        };
                        JsonSearchResult { id, score, metadata }
                    })
                    .collect();
                return Ok(Json(JsonSearchResp { results: final_results }));
            }
        }
    }

    // Build filter bitmap
    let mut filter_bitmap: Option<roaring::RoaringBitmap> = None;

    // 1. String field filters (exact match) - with filter cache
    let has_metadata_filter = !payload.filter.is_empty();
    let used_index_filter = if has_metadata_filter {
        let conditions: Vec<(&str, &str)> = payload
            .filter
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        // Try filter cache first
        let filter_cache_key = FilterBitmapCache::normalize_key(&conditions);

        if let Some(ref filter_cache) = collection.filter_cache() {
            if let Some(cached_bitmap) = filter_cache.get(&filter_cache_key) {
                filter_bitmap = Some(cached_bitmap);
                true
            } else if let Some(indexed_bitmap) = collection.metadata_store.try_filter_and(&conditions) {
                // Cache the computed bitmap
                filter_cache.put(filter_cache_key, indexed_bitmap.clone());
                filter_bitmap = Some(indexed_bitmap);
                true
            } else {
                false
            }
        } else if let Some(indexed_bitmap) = collection.metadata_store.try_filter_and(&conditions) {
            filter_bitmap = Some(indexed_bitmap);
            true
        } else {
            false
        }
    } else {
        false
    };

    // 2. Range filters (numeric fields)
    let has_range_filter = !payload.range_filters.is_empty();
    if has_range_filter {
        // Try to downcast to IndexedMetadata for range filter support
        // IndexedSledMetadataStore implements IndexedMetadata trait
        if let Some(indexed_store) = collection.metadata_store
            .as_any()
            .and_then(|any| any.downcast_ref::<crate::storage::IndexedSledMetadataStore>())
        {
            if let Some(range_bitmap) = apply_range_filters(indexed_store, &payload.range_filters) {
                filter_bitmap = Some(
                    filter_bitmap
                        .map(|existing| existing & &range_bitmap)
                        .unwrap_or(range_bitmap)
                );
            }
        }
    }

    // 3. ID range filter
    if let Some((start, end)) = payload.filter_id_range {
        let range_bitmap = create_range_bitmap(start, end);
        filter_bitmap = Some(
            filter_bitmap
                .map(|existing| existing & &range_bitmap)
                .unwrap_or(range_bitmap)
        );
    }

    // 4. Explicit ID filter
    if let Some(ref ids) = payload.filter_ids {
        if !ids.is_empty() {
            let id_bitmap = create_filter_bitmap(ids);
            filter_bitmap = Some(
                filter_bitmap
                    .map(|existing| existing & &id_bitmap)
                    .unwrap_or(id_bitmap)
            );
        }
    }

    // 5. Exclude deleted vectors
    let deleted_bitmap = collection.deleted_bitmap();
    if !deleted_bitmap.is_empty() {
        // Create universe bitmap (all valid IDs)
        let total = collection.len() as u64;
        let mut universe = roaring::RoaringBitmap::new();
        universe.insert_range(0..total as u32);

        // Remove deleted IDs from universe
        let active_bitmap = &universe - &deleted_bitmap;

        // Combine with existing filter
        filter_bitmap = Some(
            filter_bitmap
                .map(|existing| existing & &active_bitmap)
                .unwrap_or(active_bitmap)
        );
    }

    let results = collection.indexer.search(&payload.vector, payload.k as usize, filter_bitmap.as_ref())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let include_meta = payload.include_metadata;

    let final_results: Vec<JsonSearchResult> = if has_metadata_filter && !used_index_filter {
        // Post-filter: can't cache since results depend on runtime filtering
        results
            .into_iter()
            .filter(|(id, _)| {
                payload.filter.iter().all(|(fk, fv)| {
                    collection.metadata_store
                        .get(*id, fk)
                        .ok()
                        .flatten()
                        .as_ref()
                        .map(|v| v == fv)
                        .unwrap_or(false)
                })
            })
            .map(|(id, score)| {
                let metadata = if include_meta {
                    Some(collection.metadata_store.get_all(id))
                } else {
                    None
                };
                JsonSearchResult { id, score, metadata }
            })
            .collect()
    } else {
        // Store results in query cache (only for simple queries without post-filtering)
        if use_cache {
            if let Some(ref query_cache) = collection.query_cache() {
                query_cache.put(cache_key, results.clone(), current_vector_count);
            }
        }

        results
            .into_iter()
            .map(|(id, score)| {
                let metadata = if include_meta {
                    Some(collection.metadata_store.get_all(id))
                } else {
                    None
                };
                JsonSearchResult { id, score, metadata }
            })
            .collect()
    };

    Ok(Json(JsonSearchResp { results: final_results }))
}

async fn collection_get_vector(
    State(state): State<Arc<CollectionAppState>>,
    Path((name, id)): Path<(String, u64)>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    // Check if deleted
    if collection.is_deleted(id) {
        return Err(StatusCode::GONE);
    }

    if let Ok(vec) = collection.vector_store.get(id) {
        Ok(Json(serde_json::json!({
            "id": id,
            "collection": name,
            "vector": vec,
            "found": true
        })))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

/// Delete a vector by ID.
async fn collection_delete_vector(
    State(state): State<Arc<CollectionAppState>>,
    Path((name, id)): Path<(String, u64)>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    match collection.delete(id) {
        Ok(true) => Ok(Json(serde_json::json!({
            "id": id,
            "collection": name,
            "deleted": true
        }))),
        Ok(false) => Err(StatusCode::GONE), // Already deleted
        Err(_) => Err(StatusCode::NOT_FOUND),
    }
}

/// Batch delete request.
#[derive(Deserialize)]
pub struct BatchDeleteReq {
    ids: Vec<u64>,
}

/// Update vector request.
#[derive(Deserialize)]
pub struct UpdateVectorReq {
    vector: Vec<f32>,
    #[serde(default)]
    metadata: std::collections::HashMap<String, String>,
}

/// Update metadata only request.
#[derive(Deserialize)]
pub struct UpdateMetadataReq {
    metadata: std::collections::HashMap<String, String>,
}

/// Delete multiple vectors by ID.
async fn collection_delete_batch(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<BatchDeleteReq>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let count = collection.delete_batch(&payload.ids)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(serde_json::json!({
        "collection": name,
        "deleted_count": count,
        "requested_count": payload.ids.len()
    })))
}

/// Streaming progress event.
#[derive(Serialize)]
pub struct StreamProgress {
    pub inserted_so_far: u64,
    pub current_chunk_start_id: u64,
    pub current_chunk_count: u32,
    pub progress_percent: f32,
    pub is_complete: bool,
}

/// Streaming upsert request body.
#[derive(Deserialize)]
pub struct StreamUpsertReq {
    pub vectors: Vec<BatchVectorReq>,
    pub chunk_size: Option<usize>,  // Optional chunk size (default: 1000)
}

/// Streaming upsert with SSE progress updates.
/// Processes large batches in chunks and streams progress back.
async fn collection_stream_upsert(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<StreamUpsertReq>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let chunk_size = payload.chunk_size.unwrap_or(1000);
    let total_vectors = payload.vectors.len();

    // Create async stream for SSE
    let stream = async_stream::stream! {
        let mut total_inserted: u64 = 0;
        let mut first_chunk = true;
        let mut start_id: u64 = 0;

        for chunk in payload.vectors.chunks(chunk_size) {
            // Extract vectors
            let vectors: Vec<Vec<f32>> = chunk
                .iter()
                .map(|v| v.vector.clone())
                .collect();

            if vectors.is_empty() {
                continue;
            }

            // Batch insert to vector store
            let chunk_start_id = match collection.vector_store.insert_batch(&vectors) {
                Ok(id) => id,
                Err(e) => {
                    let err_event = Event::default()
                        .event("error")
                        .data(format!("Storage error: {}", e));
                    yield Ok(err_event);
                    return;
                }
            };

            if first_chunk {
                start_id = chunk_start_id;
                first_chunk = false;
            }

            // Prepare index data
            let index_data: Vec<(u64, Vec<f32>)> = vectors
                .into_iter()
                .enumerate()
                .map(|(i, v)| (chunk_start_id + i as u64, v))
                .collect();

            // Batch insert to HNSW index
            if let Err(e) = collection.indexer.insert_batch(&index_data) {
                let err_event = Event::default()
                    .event("error")
                    .data(format!("Index error: {}", e));
                yield Ok(err_event);
                return;
            }

            // Collect metadata entries
            let metadata_entries: Vec<MetadataEntry> = chunk
                .iter()
                .enumerate()
                .flat_map(|(i, v)| {
                    let id = chunk_start_id + i as u64;
                    v.metadata.iter().map(move |(k, val)| MetadataEntry {
                        id,
                        key: k.clone(),
                        value: val.clone(),
                    })
                })
                .collect();

            if !metadata_entries.is_empty() {
                if let Err(e) = collection.metadata_store.insert_batch(&metadata_entries) {
                    let err_event = Event::default()
                        .event("error")
                        .data(format!("Metadata error: {}", e));
                    yield Ok(err_event);
                    return;
                }
            }

            total_inserted += chunk.len() as u64;
            let progress_percent = (total_inserted as f32 / total_vectors as f32) * 100.0;
            let is_complete = total_inserted >= total_vectors as u64;

            let progress = StreamProgress {
                inserted_so_far: total_inserted,
                current_chunk_start_id: chunk_start_id,
                current_chunk_count: chunk.len() as u32,
                progress_percent,
                is_complete,
            };

            let event = Event::default()
                .event("progress")
                .data(serde_json::to_string(&progress).unwrap_or_default());

            yield Ok(event);
        }

        // Final completion event
        let complete = serde_json::json!({
            "total_inserted": total_inserted,
            "start_id": start_id,
            "collection": name,
            "success": true
        });

        let done_event = Event::default()
            .event("complete")
            .data(serde_json::to_string(&complete).unwrap_or_default());

        yield Ok(done_event);
    };

    Ok(Sse::new(stream))
}

/// Streaming search with SSE for large result sets.
/// Returns results in batches, useful when k is large (e.g., k > 1000).
async fn collection_stream_search(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<StreamSearchReq>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let k = payload.k as usize;
    let batch_size = payload.batch_size.unwrap_or(100) as usize;
    let include_vectors = payload.include_vectors;
    let include_metadata = payload.include_metadata;

    // Build filter bitmap (same as regular search)
    let mut filter_bitmap: Option<roaring::RoaringBitmap> = None;

    // String field filters
    if !payload.filter.is_empty() {
        let conditions: Vec<(&str, &str)> = payload.filter
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        if let Some(indexed_bitmap) = collection.metadata_store.try_filter_and(&conditions) {
            filter_bitmap = Some(indexed_bitmap);
        }
    }

    // Range filters
    if !payload.range_filters.is_empty() {
        if let Some(indexed_store) = collection.metadata_store
            .as_any()
            .and_then(|any| any.downcast_ref::<crate::storage::IndexedSledMetadataStore>())
        {
            if let Some(range_bitmap) = apply_range_filters(indexed_store, &payload.range_filters) {
                filter_bitmap = Some(
                    filter_bitmap
                        .map(|existing| existing & &range_bitmap)
                        .unwrap_or(range_bitmap)
                );
            }
        }
    }

    // Explicit ID filter
    if let Some(ref ids) = payload.filter_ids {
        if !ids.is_empty() {
            let id_bitmap = crate::api::create_filter_bitmap(ids);
            filter_bitmap = Some(
                filter_bitmap
                    .map(|existing| existing & &id_bitmap)
                    .unwrap_or(id_bitmap)
            );
        }
    }

    // Exclude deleted vectors
    let deleted_bitmap = collection.deleted_bitmap();
    if !deleted_bitmap.is_empty() {
        let total = collection.len() as u64;
        let mut universe = roaring::RoaringBitmap::new();
        universe.insert_range(0..total as u32);
        let active_bitmap = &universe - &deleted_bitmap;
        filter_bitmap = Some(
            filter_bitmap
                .map(|existing| existing & &active_bitmap)
                .unwrap_or(active_bitmap)
        );
    }

    // Perform search
    let results = collection.indexer.search(&payload.vector, k, filter_bitmap.as_ref())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let total_results = results.len();
    let total_batches = (total_results + batch_size - 1) / batch_size;

    // Clone for the async stream
    let vector_store = collection.vector_store.clone();
    let metadata_store = collection.metadata_store.clone();

    let stream = async_stream::stream! {
        let mut results_sent: u32 = 0;

        if total_results == 0 {
            let batch = StreamSearchResultBatch {
                results: Vec::new(),
                batch_index: 0,
                total_batches: 0,
                results_so_far: 0,
                total_results: 0,
                is_last: true,
            };

            let event = Event::default()
                .event("batch")
                .data(serde_json::to_string(&batch).unwrap_or_default());

            yield Ok(event);
            return;
        }

        for (batch_idx, chunk) in results.chunks(batch_size).enumerate() {
            let mut batch_results = Vec::with_capacity(chunk.len());

            for &(id, score) in chunk {
                let mut result = StreamSearchResultItem {
                    id,
                    score,
                    vector: None,
                    metadata: None,
                };

                if include_vectors {
                    if let Ok(v) = vector_store.get(id) {
                        result.vector = Some(v);
                    }
                }

                if include_metadata {
                    let meta = metadata_store.get_all(id);
                    if !meta.is_empty() {
                        result.metadata = Some(meta);
                    }
                }

                batch_results.push(result);
            }

            results_sent += batch_results.len() as u32;
            let is_last = batch_idx == total_batches.saturating_sub(1);

            let batch = StreamSearchResultBatch {
                results: batch_results,
                batch_index: batch_idx as u32,
                total_batches: total_batches as u32,
                results_so_far: results_sent,
                total_results: total_results as u32,
                is_last,
            };

            let event = Event::default()
                .event("batch")
                .data(serde_json::to_string(&batch).unwrap_or_default());

            yield Ok(event);
        }

        // Final complete event
        let complete = serde_json::json!({
            "total_results": total_results,
            "batches_sent": total_batches,
            "success": true
        });

        let done_event = Event::default()
            .event("complete")
            .data(serde_json::to_string(&complete).unwrap_or_default());

        yield Ok(done_event);
    };

    Ok(Sse::new(stream))
}

/// Hybrid search combining vector similarity with BM25 text search.
async fn collection_hybrid_search(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<HybridSearchReq>,
) -> Result<Json<HybridSearchResp>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let k = payload.k as usize;
    let alpha = payload.alpha.unwrap_or(0.5).clamp(0.0, 1.0);
    let use_rrf = payload.fusion_method.as_deref() == Some("rrf");
    let include_scores = payload.include_scores;
    let include_metadata = payload.include_metadata;

    // Build filter bitmap for metadata filters
    let mut filter_bitmap: Option<roaring::RoaringBitmap> = None;

    // String field filters
    if !payload.filter.is_empty() {
        let conditions: Vec<(&str, &str)> = payload.filter
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        if let Some(indexed_bitmap) = collection.metadata_store.try_filter_and(&conditions) {
            filter_bitmap = Some(indexed_bitmap);
        }
    }

    // Range filters
    if !payload.range_filters.is_empty() {
        if let Some(indexed_store) = collection.metadata_store
            .as_any()
            .and_then(|any| any.downcast_ref::<crate::storage::IndexedSledMetadataStore>())
        {
            if let Some(range_bitmap) = apply_range_filters(indexed_store, &payload.range_filters) {
                filter_bitmap = Some(
                    filter_bitmap
                        .map(|existing| existing & &range_bitmap)
                        .unwrap_or(range_bitmap)
                );
            }
        }
    }

    // Perform hybrid search
    let results = collection.hybrid_search_filtered(
        &payload.vector,
        &payload.query,
        k,
        alpha,
        use_rrf,
        filter_bitmap.as_ref(),
    ).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Convert to response
    let response_results: Vec<HybridSearchResult> = results
        .into_iter()
        .map(|r| {
            let metadata = if include_metadata {
                let meta = collection.metadata_store.get_all(r.id);
                if meta.is_empty() { None } else { Some(meta) }
            } else {
                None
            };

            HybridSearchResult {
                id: r.id,
                combined_score: r.combined_score,
                vector_score: if include_scores { Some(r.vector_score) } else { None },
                text_score: if include_scores { Some(r.text_score) } else { None },
                metadata,
            }
        })
        .collect();

    Ok(Json(HybridSearchResp {
        results: response_results,
        total_candidates: (k * 2) as u32,
    }))
}

/// Text-only search using BM25 algorithm.
async fn collection_text_search(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<TextSearchReq>,
) -> Result<Json<TextSearchResp>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let k = payload.k as usize;
    let include_metadata = payload.include_metadata;

    // Perform text search
    let results = collection.text_search(&payload.query, k)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Apply metadata filter if present
    let filtered_results: Vec<_> = if !payload.filter.is_empty() {
        results.into_iter()
            .filter(|r| {
                payload.filter.iter().all(|(fk, fv)| {
                    collection.metadata_store
                        .get(r.id, fk)
                        .ok()
                        .flatten()
                        .as_ref()
                        .map(|v| v == fv)
                        .unwrap_or(false)
                })
            })
            .collect()
    } else {
        results
    };

    // Convert to response
    let response_results: Vec<TextSearchResultItem> = filtered_results
        .into_iter()
        .map(|r| {
            let metadata = if include_metadata {
                let meta = collection.metadata_store.get_all(r.id);
                if meta.is_empty() { None } else { Some(meta) }
            } else {
                None
            };

            TextSearchResultItem {
                id: r.id,
                score: r.score,
                metadata,
            }
        })
        .collect();

    Ok(Json(TextSearchResp {
        results: response_results,
    }))
}

/// Multi-vector search - search with multiple query vectors and fuse results.
async fn collection_multi_search(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<MultiVectorSearchReq>,
) -> Result<Json<MultiVectorSearchResp>, StatusCode> {
    use crate::index::multi_vector::{MultiVectorQuery, FusionMethod, parallel_multi_search};

    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let k = payload.k as usize;
    let fusion = FusionMethod::from_str(&payload.fusion);
    let include_per_query = payload.include_per_query_scores;
    let include_metadata = payload.include_metadata;

    // Build query
    let mut query = MultiVectorQuery::new(payload.vectors.clone(), k)
        .with_fusion(fusion)
        .with_rrf_k(payload.rrf_k)
        .with_oversample(payload.oversample as usize);

    if let Some(weights) = payload.weights {
        query = query.with_weights(weights);
    }

    // Build filter bitmap if provided
    let filter_bitmap: Option<roaring::RoaringBitmap> = if !payload.filter.is_empty() {
        // Convert filter to conditions slice format
        let conditions: Vec<(&str, &str)> = payload.filter
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        collection.metadata_store.try_filter_and(&conditions)
    } else {
        None
    };

    // Exclude deleted IDs from filter
    let deleted = collection.deleted_bitmap();
    let effective_filter = if !deleted.is_empty() {
        let total = collection.len() as u64;
        let mut universe = roaring::RoaringBitmap::new();
        universe.insert_range(0..total as u32);
        let allowed = &universe - &deleted;

        match filter_bitmap {
            Some(f) => Some(&f & &allowed),
            None => Some(allowed),
        }
    } else {
        filter_bitmap
    };

    // Perform multi-vector search
    let results = parallel_multi_search(
        &collection.indexer,
        &query,
        effective_filter.as_ref(),
    ).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Convert to response
    let response_results: Vec<MultiVectorSearchResult> = results
        .into_iter()
        .map(|r| {
            let metadata = if include_metadata {
                let meta = collection.metadata_store.get_all(r.id);
                if meta.is_empty() { None } else { Some(meta) }
            } else {
                None
            };

            MultiVectorSearchResult {
                id: r.id,
                score: r.score,
                query_matches: r.query_matches,
                per_query_scores: if include_per_query {
                    Some(r.per_query_scores)
                } else {
                    None
                },
                metadata,
            }
        })
        .collect();

    Ok(Json(MultiVectorSearchResp {
        results: response_results,
        num_queries: payload.vectors.len(),
        fusion_method: payload.fusion,
    }))
}

// =============================================================================
// Sparse Vector Handlers
// =============================================================================

/// Sparse index storage - one per collection.
/// Uses lazy initialization via a global map.
use std::sync::Mutex;
use once_cell::sync::Lazy;
use std::collections::HashMap as StdHashMap;

static SPARSE_INDICES: Lazy<Mutex<StdHashMap<String, crate::sparse::ThreadSafeSparseIndex>>> =
    Lazy::new(|| Mutex::new(StdHashMap::new()));

fn get_or_create_sparse_index(collection_name: &str) -> crate::sparse::ThreadSafeSparseIndex {
    let mut indices = SPARSE_INDICES.lock().unwrap();
    indices
        .entry(collection_name.to_string())
        .or_insert_with(crate::sparse::ThreadSafeSparseIndex::new)
        .clone()
}

/// Insert a sparse vector into a collection.
async fn collection_sparse_upsert(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<SparseUpsertReq>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Verify collection exists
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    // Get or create sparse index for this collection
    let sparse_index = get_or_create_sparse_index(&name);

    // Create sparse vector
    let sparse_vec = crate::sparse::SparseVector::new(payload.indices, payload.values)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    // Generate ID (use next available from collection)
    let id = collection.len() as u64;

    // Insert into sparse index
    sparse_index.insert(id, sparse_vec)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Store metadata
    for (k, v) in payload.metadata {
        collection.metadata_store.insert(id, k, v)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }

    Ok(Json(serde_json::json!({
        "id": id,
        "collection": name,
        "success": true
    })))
}

/// Batch insert sparse vectors.
async fn collection_sparse_upsert_batch(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<SparseBatchUpsertReq>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Verify collection exists
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    // Get or create sparse index
    let sparse_index = get_or_create_sparse_index(&name);

    let start_id = collection.len() as u64;
    let count = payload.vectors.len();

    // Prepare batch
    let mut batch: Vec<(u64, crate::sparse::SparseVector)> = Vec::with_capacity(count);
    let mut metadata_batch: Vec<(u64, std::collections::HashMap<String, String>)> = Vec::with_capacity(count);

    for (i, v) in payload.vectors.into_iter().enumerate() {
        let id = start_id + i as u64;
        let sparse_vec = crate::sparse::SparseVector::new(v.indices, v.values)
            .map_err(|_| StatusCode::BAD_REQUEST)?;
        batch.push((id, sparse_vec));
        metadata_batch.push((id, v.metadata));
    }

    // Batch insert sparse vectors
    sparse_index.insert_batch(&batch)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Insert metadata
    for (id, metadata) in metadata_batch {
        for (k, v) in metadata {
            collection.metadata_store.insert(id, k, v)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        }
    }

    Ok(Json(serde_json::json!({
        "start_id": start_id,
        "count": count,
        "success": true
    })))
}

/// Search sparse vectors.
async fn collection_sparse_search(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<SparseSearchReq>,
) -> Result<Json<SparseSearchResp>, StatusCode> {
    // Verify collection exists
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    // Get sparse index
    let sparse_index = get_or_create_sparse_index(&name);

    if sparse_index.is_empty() {
        return Ok(Json(SparseSearchResp { results: vec![] }));
    }

    // Create query vector
    let query = crate::sparse::SparseVector::new(payload.indices, payload.values)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let k = payload.k as usize;
    let include_metadata = payload.include_metadata;

    // Search
    let results = sparse_index.search(&query, k);

    // Apply filter if present (post-filter for now)
    let filtered_results: Vec<_> = if !payload.filter.is_empty() {
        results.into_iter()
            .filter(|(id, _)| {
                payload.filter.iter().all(|(fk, fv)| {
                    collection.metadata_store
                        .get(*id, fk)
                        .ok()
                        .flatten()
                        .as_ref()
                        .map(|v| v == fv)
                        .unwrap_or(false)
                })
            })
            .collect()
    } else {
        results
    };

    // Convert to response
    let response_results: Vec<SparseSearchResult> = filtered_results
        .into_iter()
        .map(|(id, score)| {
            let metadata = if include_metadata {
                let meta = collection.metadata_store.get_all(id);
                if meta.is_empty() { None } else { Some(meta) }
            } else {
                None
            };

            SparseSearchResult { id, score, metadata }
        })
        .collect();

    Ok(Json(SparseSearchResp { results: response_results }))
}

/// Hybrid dense + sparse search.
async fn collection_hybrid_dense_sparse(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<HybridDenseSparseReq>,
) -> Result<Json<HybridDenseSparseResp>, StatusCode> {
    use crate::sparse::{HybridSearchConfig, HybridFusion, fuse_hybrid_results};

    // Verify collection exists
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let k = payload.k as usize;
    let include_scores = payload.include_scores;
    let include_metadata = payload.include_metadata;

    // Build filter bitmap
    let filter_bitmap: Option<roaring::RoaringBitmap> = if !payload.filter.is_empty() {
        let conditions: Vec<(&str, &str)> = payload.filter
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        collection.metadata_store.try_filter_and(&conditions)
    } else {
        None
    };

    // Exclude deleted IDs
    let deleted = collection.deleted_bitmap();
    let effective_filter = if !deleted.is_empty() {
        let total = collection.len() as u64;
        let mut universe = roaring::RoaringBitmap::new();
        universe.insert_range(0..total as u32);
        let allowed = &universe - &deleted;
        match filter_bitmap {
            Some(f) => Some(&f & &allowed),
            None => Some(allowed),
        }
    } else {
        filter_bitmap
    };

    // Dense search
    let oversample = 3; // Fetch more candidates for fusion
    let dense_results = collection.indexer
        .search(&payload.dense_vector, k * oversample, effective_filter.as_ref())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Sparse search
    let sparse_index = get_or_create_sparse_index(&name);
    let sparse_query = crate::sparse::SparseVector::new(
        payload.sparse_indices.clone(),
        payload.sparse_values.clone(),
    ).map_err(|_| StatusCode::BAD_REQUEST)?;

    let sparse_results = if let Some(ref filter) = effective_filter {
        sparse_index.search_filtered(&sparse_query, k * oversample, filter)
    } else {
        sparse_index.search(&sparse_query, k * oversample)
    };

    // Fuse results
    let config = HybridSearchConfig {
        dense_weight: payload.dense_weight,
        sparse_weight: payload.sparse_weight,
        fusion: HybridFusion::from_str(&payload.fusion),
        oversample,
    };

    let fused = fuse_hybrid_results(&dense_results, &sparse_results, &config, k);

    // Convert to response
    let response_results: Vec<HybridDenseSparseResult> = fused
        .into_iter()
        .map(|r| {
            let metadata = if include_metadata {
                let meta = collection.metadata_store.get_all(r.id);
                if meta.is_empty() { None } else { Some(meta) }
            } else {
                None
            };

            HybridDenseSparseResult {
                id: r.id,
                score: r.score,
                dense_score: if include_scores { r.dense_score } else { None },
                sparse_score: if include_scores { r.sparse_score } else { None },
                metadata,
            }
        })
        .collect();

    Ok(Json(HybridDenseSparseResp {
        results: response_results,
        fusion_method: payload.fusion,
    }))
}

// =============================================================================
// Versioning Handlers
// =============================================================================

/// Versioned store storage - one per collection.
static VERSIONED_STORES: Lazy<Mutex<StdHashMap<String, crate::versioning::ThreadSafeVersionedStore>>> =
    Lazy::new(|| Mutex::new(StdHashMap::new()));

fn get_or_create_versioned_store(collection_name: &str, dim: usize) -> crate::versioning::ThreadSafeVersionedStore {
    let mut stores = VERSIONED_STORES.lock().unwrap();
    stores
        .entry(collection_name.to_string())
        .or_insert_with(|| crate::versioning::ThreadSafeVersionedStore::new(dim))
        .clone()
}

/// Insert or update a versioned vector.
async fn versioned_upsert(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<VersionedUpsertReq>,
) -> Result<Json<VersionInfoResp>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let store = get_or_create_versioned_store(&name, collection.config.dim);

    let version = store.upsert_with_description(
        payload.vector_id,
        payload.vector,
        payload.metadata,
        payload.description.as_deref(),
    ).map_err(|_| StatusCode::BAD_REQUEST)?;

    Ok(Json(version.into()))
}

/// Get the latest version of a vector.
async fn versioned_get(
    State(state): State<Arc<CollectionAppState>>,
    Path((name, vector_id)): Path<(String, u64)>,
) -> Result<Json<VersionInfoResp>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let store = get_or_create_versioned_store(&name, collection.config.dim);

    let version = store.get(vector_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(version.into()))
}

/// Get version history for a vector.
async fn versioned_history(
    State(state): State<Arc<CollectionAppState>>,
    Path((name, vector_id)): Path<(String, u64)>,
) -> Result<Json<Vec<VersionInfoResp>>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let store = get_or_create_versioned_store(&name, collection.config.dim);

    let history = store.get_history(vector_id);
    if history.is_empty() {
        return Err(StatusCode::NOT_FOUND);
    }

    let response: Vec<VersionInfoResp> = history.into_iter().map(|v| v.into()).collect();
    Ok(Json(response))
}

/// Get vector at a specific timestamp.
async fn versioned_at_timestamp(
    State(state): State<Arc<CollectionAppState>>,
    Path((name, vector_id)): Path<(String, u64)>,
    Json(payload): Json<GetAtTimestampReq>,
) -> Result<Json<VersionInfoResp>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let store = get_or_create_versioned_store(&name, collection.config.dim);

    let timestamp = chrono::DateTime::parse_from_rfc3339(&payload.timestamp)
        .map_err(|_| StatusCode::BAD_REQUEST)?
        .with_timezone(&chrono::Utc);

    let version = store.get_at_timestamp(vector_id, timestamp)
        .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(version.into()))
}

/// Rollback a vector to a previous version.
async fn versioned_rollback(
    State(state): State<Arc<CollectionAppState>>,
    Path((name, vector_id)): Path<(String, u64)>,
    Json(payload): Json<RollbackReq>,
) -> Result<Json<VersionInfoResp>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let store = get_or_create_versioned_store(&name, collection.config.dim);

    let version = store.rollback(vector_id, payload.to_version_id)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    Ok(Json(version.into()))
}

/// Get snapshot of all vectors at a timestamp.
async fn versioned_snapshot(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<SnapshotAtReq>,
) -> Result<Json<Vec<VersionInfoResp>>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let store = get_or_create_versioned_store(&name, collection.config.dim);

    let timestamp = chrono::DateTime::parse_from_rfc3339(&payload.timestamp)
        .map_err(|_| StatusCode::BAD_REQUEST)?
        .with_timezone(&chrono::Utc);

    let snapshot = store.snapshot_at(timestamp);
    let response: Vec<VersionInfoResp> = snapshot.into_iter().map(|v| v.into()).collect();

    Ok(Json(response))
}

/// Get versioning statistics.
async fn versioned_stats(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
) -> Result<Json<crate::versioning::VersioningStats>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let store = get_or_create_versioned_store(&name, collection.config.dim);

    Ok(Json(store.stats()))
}

/// Update a vector (delete old + insert new).
/// Returns the new ID since vectors are append-only.
async fn collection_update_vector(
    State(state): State<Arc<CollectionAppState>>,
    Path((name, old_id)): Path<(String, u64)>,
    Json(payload): Json<UpdateVectorReq>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    // Check if vector exists and not already deleted
    if old_id as usize >= collection.len() {
        return Err(StatusCode::NOT_FOUND);
    }
    if collection.is_deleted(old_id) {
        return Err(StatusCode::GONE);
    }

    // Delete old vector
    collection.delete(old_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Insert new vector
    let new_id = collection.vector_store.insert(&payload.vector)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Index new vector
    collection.indexer.insert(new_id, &payload.vector)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Insert metadata
    for (k, v) in payload.metadata {
        collection.metadata_store.insert(new_id, k, v)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }

    Ok(Json(serde_json::json!({
        "old_id": old_id,
        "new_id": new_id,
        "collection": name,
        "updated": true
    })))
}

/// Update metadata only (no vector change, same ID).
async fn collection_update_metadata(
    State(state): State<Arc<CollectionAppState>>,
    Path((name, id)): Path<(String, u64)>,
    Json(payload): Json<UpdateMetadataReq>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    // Check if vector exists and not deleted
    if id as usize >= collection.len() {
        return Err(StatusCode::NOT_FOUND);
    }
    if collection.is_deleted(id) {
        return Err(StatusCode::GONE);
    }

    // Update metadata (overwrites existing keys)
    for (k, v) in payload.metadata {
        collection.metadata_store.insert(id, k, v)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }

    Ok(Json(serde_json::json!({
        "id": id,
        "collection": name,
        "updated": true
    })))
}

// ============================================================================
// Compaction Handler
// ============================================================================

/// Compact response.
#[derive(Serialize)]
pub struct CompactResp {
    pub collection: String,
    pub vectors_before: usize,
    pub vectors_after: usize,
    pub vectors_removed: usize,
    pub bytes_reclaimed: u64,
    pub success: bool,
}

/// Compact a collection by removing deleted vectors.
/// This operation rebuilds the index and may take time for large collections.
///
/// WARNING: All vector IDs will change after compaction.
async fn collection_compact(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
) -> Result<Json<CompactResp>, StatusCode> {
    let result = state.manager.compact(&name)
        .map_err(|e| {
            eprintln!("Compaction error: {:?}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(Json(CompactResp {
        collection: name,
        vectors_before: result.vectors_before,
        vectors_after: result.vectors_after,
        vectors_removed: result.vectors_removed,
        bytes_reclaimed: result.bytes_reclaimed,
        success: true,
    }))
}

// ============================================================================
// Checkpoint Handlers (WAL)
// ============================================================================

/// Response for checkpoint operations.
#[derive(Serialize)]
pub struct CheckpointResp {
    collection: String,
    sequence: u64,
    success: bool,
}

/// Response for WAL stats.
#[derive(Serialize)]
pub struct WalStatsResp {
    collection: String,
    enabled: bool,
    sequence: Option<u64>,
    last_checkpoint: Option<u64>,
    entries_since_checkpoint: Option<u64>,
    file_size: Option<u64>,
}

/// Create a checkpoint for a collection.
/// This saves the index and WAL marker, allowing old WAL entries to be truncated.
async fn collection_checkpoint(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
) -> Result<Json<CheckpointResp>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let sequence = collection.checkpoint()
        .map_err(|e| {
            eprintln!("Checkpoint error: {:?}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(Json(CheckpointResp {
        collection: name,
        sequence,
        success: true,
    }))
}

/// Get WAL statistics for a collection.
async fn collection_wal_stats(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
) -> Result<Json<WalStatsResp>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let wal_stats = collection.wal_stats();
    let info = collection.info();

    Ok(Json(WalStatsResp {
        collection: name,
        enabled: info.wal_enabled,
        sequence: wal_stats.as_ref().map(|s| s.sequence),
        last_checkpoint: wal_stats.as_ref().map(|s| s.last_checkpoint),
        entries_since_checkpoint: wal_stats.as_ref().map(|s| s.entries_since_checkpoint),
        file_size: wal_stats.map(|s| s.file_size),
    }))
}

/// Cache statistics response.
#[derive(Serialize)]
pub struct CacheStatsResp {
    collection: String,
    query_cache: Option<QueryCacheStatsResp>,
    filter_cache: Option<FilterCacheStatsResp>,
}

/// Query cache statistics.
#[derive(Serialize)]
pub struct QueryCacheStatsResp {
    size: usize,
    capacity: usize,
    hits: u64,
    misses: u64,
    invalidations: u64,
    hit_rate: f64,
}

/// Filter cache statistics.
#[derive(Serialize)]
pub struct FilterCacheStatsResp {
    size: usize,
    capacity: usize,
    hits: u64,
    misses: u64,
    invalidations: u64,
    hit_rate: f64,
}

/// Get cache statistics for a collection.
async fn collection_cache_stats(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
) -> Result<Json<CacheStatsResp>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    let query_cache = collection.query_cache_stats().map(|s| QueryCacheStatsResp {
        size: s.size,
        capacity: s.capacity,
        hits: s.hits,
        misses: s.misses,
        invalidations: s.invalidations,
        hit_rate: s.hit_rate(),
    });

    let filter_cache = collection.filter_cache_stats().map(|s| FilterCacheStatsResp {
        size: s.size,
        capacity: s.capacity,
        hits: s.hits,
        misses: s.misses,
        invalidations: s.invalidations,
        hit_rate: s.hit_rate(),
    });

    Ok(Json(CacheStatsResp {
        collection: name,
        query_cache,
        filter_cache,
    }))
}

/// Checkpoint all response.
#[derive(Serialize)]
pub struct CheckpointAllResp {
    checkpointed: Vec<String>,
    success: bool,
}

/// Create checkpoints for all collections.
async fn checkpoint_all(
    State(state): State<Arc<CollectionAppState>>,
) -> Json<CheckpointAllResp> {
    let _ = state.manager.checkpoint_all();
    let names = state.manager.names();

    Json(CheckpointAllResp {
        checkpointed: names,
        success: true,
    })
}

// ============================================================================
// Snapshot Handlers
// ============================================================================

/// List all snapshots for a collection.
async fn list_snapshots(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
) -> Result<Json<Vec<SnapshotInfo>>, StatusCode> {
    // Verify collection exists
    if state.manager.get(&name).is_none() {
        return Err(StatusCode::NOT_FOUND);
    }

    state.manager.list_snapshots(&name)
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// Create a snapshot of a collection.
async fn create_snapshot(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
) -> Result<Json<SnapshotInfo>, StatusCode> {
    state.manager.create_snapshot(&name)
        .map(Json)
        .map_err(|e| {
            eprintln!("Failed to create snapshot: {:?}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })
}

/// Request to restore a snapshot.
#[derive(Deserialize)]
pub struct RestoreSnapshotReq {
    /// Optional new collection name. If not provided, uses original name.
    #[serde(default)]
    new_name: Option<String>,
}

/// Restore a collection from a snapshot.
async fn restore_snapshot(
    State(state): State<Arc<CollectionAppState>>,
    Path(snapshot_name): Path<String>,
    Json(payload): Json<RestoreSnapshotReq>,
) -> Result<Json<CollectionInfo>, StatusCode> {
    state.manager.restore_snapshot(&snapshot_name, payload.new_name.as_deref())
        .map(|c| Json(c.info()))
        .map_err(|e| {
            eprintln!("Failed to restore snapshot: {:?}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })
}

/// Delete a snapshot.
async fn delete_snapshot(
    State(state): State<Arc<CollectionAppState>>,
    Path(snapshot_name): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    state.manager.delete_snapshot(&snapshot_name)
        .map(|_| Json(serde_json::json!({ "deleted": true, "snapshot": snapshot_name })))
        .map_err(|_| StatusCode::NOT_FOUND)
}

// ============================================================================
// Legacy Handlers (Default Collection)
// ============================================================================

const DEFAULT_DIM: usize = 128;

async fn legacy_upsert(
    State(state): State<Arc<CollectionAppState>>,
    Json(payload): Json<JsonUpsertReq>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let collection = state.manager.get_or_create_default(DEFAULT_DIM)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let internal_id = collection.vector_store.insert(&payload.vector)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    collection.indexer.insert(internal_id, &payload.vector)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    for (k, v) in payload.metadata {
        collection.metadata_store.insert(internal_id, k, v)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }

    Ok(Json(serde_json::json!({ "id": internal_id, "success": true })))
}

/// Legacy batch upsert using default collection.
async fn legacy_upsert_batch(
    State(state): State<Arc<CollectionAppState>>,
    Json(payload): Json<JsonBatchUpsertReq>,
) -> Result<Json<JsonBatchUpsertResp>, StatusCode> {
    let collection = state.manager.get_or_create_default(DEFAULT_DIM)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if payload.vectors.is_empty() {
        return Ok(Json(JsonBatchUpsertResp {
            start_id: collection.len() as u64,
            count: 0,
            success: true,
        }));
    }

    // Extract vectors
    let vectors: Vec<Vec<f32>> = payload.vectors
        .iter()
        .map(|v| v.vector.clone())
        .collect();

    // Batch insert to vector store
    let start_id = collection.vector_store.insert_batch(&vectors)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Prepare index data
    let index_data: Vec<(u64, Vec<f32>)> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, v)| (start_id + i as u64, v))
        .collect();

    // Batch insert to HNSW index
    collection.indexer.insert_batch(&index_data)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Collect metadata entries
    let metadata_entries: Vec<MetadataEntry> = payload.vectors
        .iter()
        .enumerate()
        .flat_map(|(i, v)| {
            let id = start_id + i as u64;
            v.metadata.iter().map(move |(k, val)| MetadataEntry {
                id,
                key: k.clone(),
                value: val.clone(),
            })
        })
        .collect();

    // Batch insert metadata
    if !metadata_entries.is_empty() {
        collection.metadata_store.insert_batch(&metadata_entries)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }

    Ok(Json(JsonBatchUpsertResp {
        start_id,
        count: payload.vectors.len(),
        success: true,
    }))
}

async fn legacy_search(
    State(state): State<Arc<CollectionAppState>>,
    Json(payload): Json<JsonSearchReq>,
) -> Result<Json<JsonSearchResp>, StatusCode> {
    let collection = state.manager.get_or_create_default(DEFAULT_DIM)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Build filter bitmap
    let mut filter_bitmap: Option<roaring::RoaringBitmap> = None;

    // 1. String field filters
    let has_metadata_filter = !payload.filter.is_empty();
    let used_index_filter = if has_metadata_filter {
        let conditions: Vec<(&str, &str)> = payload
            .filter
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        if let Some(indexed_bitmap) = collection.metadata_store.try_filter_and(&conditions) {
            filter_bitmap = Some(indexed_bitmap);
            true
        } else {
            false
        }
    } else {
        false
    };

    // 2. Range filters
    let has_range_filter = !payload.range_filters.is_empty();
    if has_range_filter {
        if let Some(indexed_store) = collection.metadata_store
            .as_any()
            .and_then(|any| any.downcast_ref::<crate::storage::IndexedSledMetadataStore>())
        {
            if let Some(range_bitmap) = apply_range_filters(indexed_store, &payload.range_filters) {
                filter_bitmap = Some(
                    filter_bitmap
                        .map(|existing| existing & &range_bitmap)
                        .unwrap_or(range_bitmap)
                );
            }
        }
    }

    // 3. ID range filter
    if let Some((start, end)) = payload.filter_id_range {
        let range_bitmap = create_range_bitmap(start, end);
        filter_bitmap = Some(
            filter_bitmap
                .map(|existing| existing & &range_bitmap)
                .unwrap_or(range_bitmap)
        );
    }

    // 4. Explicit ID filter
    if let Some(ref ids) = payload.filter_ids {
        if !ids.is_empty() {
            let id_bitmap = create_filter_bitmap(ids);
            filter_bitmap = Some(
                filter_bitmap
                    .map(|existing| existing & &id_bitmap)
                    .unwrap_or(id_bitmap)
            );
        }
    }

    // 5. Exclude deleted vectors
    let deleted_bitmap = collection.deleted_bitmap();
    if !deleted_bitmap.is_empty() {
        let total = collection.len() as u64;
        let mut universe = roaring::RoaringBitmap::new();
        universe.insert_range(0..total as u32);
        let active_bitmap = &universe - &deleted_bitmap;
        filter_bitmap = Some(
            filter_bitmap
                .map(|existing| existing & &active_bitmap)
                .unwrap_or(active_bitmap)
        );
    }

    let results = collection.indexer.search(&payload.vector, payload.k as usize, filter_bitmap.as_ref())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let include_meta = payload.include_metadata;

    let final_results: Vec<JsonSearchResult> = if has_metadata_filter && !used_index_filter {
        results
            .into_iter()
            .filter(|(id, _)| {
                payload.filter.iter().all(|(fk, fv)| {
                    collection.metadata_store
                        .get(*id, fk)
                        .ok()
                        .flatten()
                        .as_ref()
                        .map(|v| v == fv)
                        .unwrap_or(false)
                })
            })
            .map(|(id, score)| {
                let metadata = if include_meta {
                    Some(collection.metadata_store.get_all(id))
                } else {
                    None
                };
                JsonSearchResult { id, score, metadata }
            })
            .collect()
    } else {
        results
            .into_iter()
            .map(|(id, score)| {
                let metadata = if include_meta {
                    Some(collection.metadata_store.get_all(id))
                } else {
                    None
                };
                JsonSearchResult { id, score, metadata }
            })
            .collect()
    };

    Ok(Json(JsonSearchResp { results: final_results }))
}

async fn legacy_get_vector(
    State(state): State<Arc<CollectionAppState>>,
    Path(id): Path<u64>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let collection = state.manager.get_or_create_default(DEFAULT_DIM)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if let Ok(vec) = collection.vector_store.get(id) {
        Ok(Json(serde_json::json!({ "id": id, "vector": vec, "found": true })))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn legacy_stats(
    State(state): State<Arc<CollectionAppState>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let collection = state.manager.get_or_create_default(DEFAULT_DIM)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(serde_json::json!({
        "vector_count": collection.len(),
        "dimension": collection.config.dim,
        "collection": collection.name(),
        "status": "Ready"
    })))
}

// ============================================================================
// Replication Handlers
// ============================================================================

/// Replication status response.
#[derive(Serialize)]
pub struct ReplicationStatusResp {
    pub role: String,
    pub current_seq: u64,
    pub wal_entries: usize,
    pub wal_size_bytes: u64,
    pub replica_status: Option<String>,
    pub primary_addr: Option<String>,
    pub last_synced_seq: Option<u64>,
}

/// Get replication status.
async fn replication_status(
    State(state): State<Arc<CollectionAppState>>,
) -> Json<ReplicationStatusResp> {
    if let Some(ref replication) = state.replication {
        let status = replication.status();
        let role_str = match status.role {
            ReplicationRole::Primary => "primary",
            ReplicationRole::Replica => "replica",
            ReplicationRole::Standalone => "standalone",
        };

        let (wal_entries, wal_size) = status.wal_stats
            .map(|s| (s.total_entries, s.total_size_bytes))
            .unwrap_or((0, 0));

        let (replica_status, primary_addr, last_seq) = status.replica_state
            .map(|s| (Some(format!("{:?}", s.status)), Some(s.primary_addr), Some(s.last_seq)))
            .unwrap_or((None, None, None));

        Json(ReplicationStatusResp {
            role: role_str.to_string(),
            current_seq: status.current_seq,
            wal_entries,
            wal_size_bytes: wal_size,
            replica_status,
            primary_addr,
            last_synced_seq: last_seq,
        })
    } else {
        Json(ReplicationStatusResp {
            role: "standalone".to_string(),
            current_seq: 0,
            wal_entries: 0,
            wal_size_bytes: 0,
            replica_status: None,
            primary_addr: None,
            last_synced_seq: None,
        })
    }
}

/// WAL entries query parameters.
#[derive(Deserialize)]
pub struct WalEntriesQuery {
    #[serde(default)]
    pub from_seq: u64,
    #[serde(default = "default_max_entries")]
    pub max_entries: usize,
}

fn default_max_entries() -> usize {
    1000
}

/// WAL entry response.
#[derive(Serialize)]
pub struct WalEntryResp {
    pub seq: u64,
    pub timestamp: String,
    pub operation_type: String,
    pub collection: String,
}

/// Get WAL entries response.
#[derive(Serialize)]
pub struct WalEntriesResp {
    pub entries: Vec<WalEntryResp>,
    pub current_seq: u64,
    pub has_more: bool,
}

/// Get WAL entries for replication.
async fn get_wal_entries(
    State(state): State<Arc<CollectionAppState>>,
    axum::extract::Query(query): axum::extract::Query<WalEntriesQuery>,
) -> Result<Json<WalEntriesResp>, StatusCode> {
    if let Some(ref replication) = state.replication {
        let entries = replication.get_entries_from(query.from_seq)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        let current_seq = replication.current_seq();
        let total_entries = entries.len();
        let has_more = total_entries > query.max_entries;

        let entries: Vec<WalEntryResp> = entries
            .into_iter()
            .take(query.max_entries)
            .map(|e| {
                let (op_type, collection) = match &e.operation {
                    crate::replication::WalOperation::Insert { collection, .. } => ("insert", collection.clone()),
                    crate::replication::WalOperation::InsertBatch { collection, .. } => ("insert_batch", collection.clone()),
                    crate::replication::WalOperation::Delete { collection, .. } => ("delete", collection.clone()),
                    crate::replication::WalOperation::DeleteBatch { collection, .. } => ("delete_batch", collection.clone()),
                    crate::replication::WalOperation::Update { collection, .. } => ("update", collection.clone()),
                    crate::replication::WalOperation::UpdateMetadata { collection, .. } => ("update_metadata", collection.clone()),
                    crate::replication::WalOperation::CreateCollection { name, .. } => ("create_collection", name.clone()),
                    crate::replication::WalOperation::DeleteCollection { name } => ("delete_collection", name.clone()),
                    crate::replication::WalOperation::Compact { collection } => ("compact", collection.clone()),
                };
                WalEntryResp {
                    seq: e.seq,
                    timestamp: e.timestamp.to_rfc3339(),
                    operation_type: op_type.to_string(),
                    collection,
                }
            })
            .collect();

        Ok(Json(WalEntriesResp {
            entries,
            current_seq,
            has_more,
        }))
    } else {
        Ok(Json(WalEntriesResp {
            entries: Vec::new(),
            current_seq: 0,
            has_more: false,
        }))
    }
}

// ============================================================================
// Metrics Handler
// ============================================================================

/// Prometheus metrics endpoint.
/// Returns metrics in Prometheus text exposition format.
async fn prometheus_metrics(
    State(state): State<Arc<CollectionAppState>>,
) -> (StatusCode, [(axum::http::header::HeaderName, &'static str); 1], String) {
    // Update collection metrics before gathering
    for info in state.manager.list() {
        metrics::update_collection_metrics(
            &info.name,
            info.vector_count + info.deleted_count,
            info.deleted_count,
            info.dim,
        );
    }

    // Update total collections count
    metrics::COLLECTIONS_TOTAL.set(state.manager.list().len() as f64);

    // Update WAL metrics if replication is enabled
    if let Some(ref replication) = state.replication {
        if let Some(stats) = replication.wal_stats() {
            metrics::update_wal_metrics(
                stats.current_seq,
                stats.total_entries,
                stats.total_size_bytes,
            );
        }
    }

    // Gather and return metrics
    let body = metrics::gather_metrics();

    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

// ============================================================================
// SIMD Info Handler
// ============================================================================

/// SIMD capability info response.
#[derive(Serialize)]
pub struct SimdInfoResp {
    pub level: String,
    pub l2_implementation: String,
    pub dot_implementation: String,
    pub features: Vec<String>,
}

/// Get SIMD capability information.
/// Returns the detected SIMD level and available implementations.
async fn simd_info() -> Json<SimdInfoResp> {
    let info = crate::simd::simd_info();

    let mut features = Vec::new();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            features.push("avx2".to_string());
        }
        if is_x86_feature_detected!("avx") {
            features.push("avx".to_string());
        }
        if is_x86_feature_detected!("fma") {
            features.push("fma".to_string());
        }
        if is_x86_feature_detected!("sse4.1") {
            features.push("sse4.1".to_string());
        }
        if is_x86_feature_detected!("sse4.2") {
            features.push("sse4.2".to_string());
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        features.push("neon".to_string());
    }

    Json(SimdInfoResp {
        level: info.level.as_str().to_string(),
        l2_implementation: info.l2_impl.to_string(),
        dot_implementation: info.dot_impl.to_string(),
        features,
    })
}

// ============================================================================
// Sharding Management Handlers
// ============================================================================

use crate::sharding::{ShardConfig, ShardingConfig, ShardRouter};

/// Sharding status response (uses ShardingStatus from sharding module).
#[derive(Serialize)]
pub struct ShardingStatusResp {
    pub enabled: bool,
    pub strategy: String,
    pub num_shards: u32,
    pub local_shards: u32,
    pub local_node: String,
    pub shard_details: Vec<ShardDetailResp>,
}

/// Shard detail for API responses.
#[derive(Serialize)]
pub struct ShardDetailResp {
    pub shard_id: u32,
    pub node: String,
    pub collection: String,
    pub is_primary: bool,
    pub is_local: bool,
    pub vector_count: Option<u64>,
}

/// Request to add a new shard.
#[derive(Deserialize)]
pub struct AddShardReq {
    pub shard_id: u32,
    pub collection: String,
    #[serde(default = "default_node")]
    pub node: String,
    #[serde(default = "default_true")]
    pub is_primary: bool,
}

fn default_node() -> String { "local".to_string() }
fn default_true() -> bool { true }

/// Route info response.
#[derive(Serialize)]
pub struct RouteInfoResp {
    pub vector_id: u64,
    pub shard_id: u32,
    pub is_local: bool,
    pub collection_name: Option<String>,
    pub node: Option<String>,
}

/// Global sharding state (using std::sync::LazyLock).
static SHARD_ROUTER: std::sync::LazyLock<parking_lot::RwLock<Option<ShardRouter>>> =
    std::sync::LazyLock::new(|| parking_lot::RwLock::new(None));

/// Initialize sharding with config.
pub fn init_sharding(config: ShardingConfig, local_node: &str) {
    let mut router = SHARD_ROUTER.write();
    *router = Some(ShardRouter::new(config, local_node.to_string()));
}

/// Initialize local-only sharding.
pub fn init_local_sharding(num_shards: u32, base_collection: &str) {
    let mut router = SHARD_ROUTER.write();
    *router = Some(ShardRouter::local(num_shards, base_collection));
}

/// Get sharding configuration/status.
async fn get_sharding_config(
    State(state): State<Arc<CollectionAppState>>,
) -> Result<Json<ShardingStatusResp>, StatusCode> {
    let router = SHARD_ROUTER.read();
    let router = router.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    let status = router.status();

    let shard_details: Vec<ShardDetailResp> = status.shard_details.iter().map(|s| {
        let is_local = router.is_local_shard(s.id);
        let vector_count = if is_local {
            state.manager.get(&s.collection).map(|c| c.len() as u64)
        } else {
            None
        };

        ShardDetailResp {
            shard_id: s.id,
            node: s.node.clone(),
            collection: s.collection.clone(),
            is_primary: s.is_primary,
            is_local,
            vector_count,
        }
    }).collect();

    Ok(Json(ShardingStatusResp {
        enabled: status.enabled,
        strategy: format!("{:?}", status.strategy),
        num_shards: status.num_shards,
        local_shards: status.local_shards,
        local_node: status.local_node,
        shard_details,
    }))
}

/// Update sharding configuration - rebuild with new config.
async fn update_sharding_config(
    State(state): State<Arc<CollectionAppState>>,
    Json(payload): Json<ShardingConfig>,
) -> Result<Json<ShardingStatusResp>, StatusCode> {
    let mut router_guard = SHARD_ROUTER.write();
    let old_router = router_guard.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let local_node = old_router.status().local_node.clone();

    // Rebuild router with new config
    *router_guard = Some(ShardRouter::new(payload, local_node.clone()));
    let router = router_guard.as_ref().unwrap();

    let status = router.status();
    let shard_details: Vec<ShardDetailResp> = status.shard_details.iter().map(|s| {
        let is_local = router.is_local_shard(s.id);
        let vector_count = if is_local {
            state.manager.get(&s.collection).map(|c| c.len() as u64)
        } else {
            None
        };

        ShardDetailResp {
            shard_id: s.id,
            node: s.node.clone(),
            collection: s.collection.clone(),
            is_primary: s.is_primary,
            is_local,
            vector_count,
        }
    }).collect();

    Ok(Json(ShardingStatusResp {
        enabled: status.enabled,
        strategy: format!("{:?}", status.strategy),
        num_shards: status.num_shards,
        local_shards: status.local_shards,
        local_node: status.local_node,
        shard_details,
    }))
}

/// List all shards.
async fn list_shards(
    State(state): State<Arc<CollectionAppState>>,
) -> Result<Json<Vec<ShardDetailResp>>, StatusCode> {
    let router = SHARD_ROUTER.read();
    let router = router.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    let status = router.status();
    let shards: Vec<ShardDetailResp> = status.shard_details.iter().map(|s| {
        let is_local = router.is_local_shard(s.id);
        let vector_count = if is_local {
            state.manager.get(&s.collection).map(|c| c.len() as u64)
        } else {
            None
        };

        ShardDetailResp {
            shard_id: s.id,
            node: s.node.clone(),
            collection: s.collection.clone(),
            is_primary: s.is_primary,
            is_local,
            vector_count,
        }
    }).collect();

    Ok(Json(shards))
}

/// Get info about a specific shard.
async fn get_shard_info(
    State(state): State<Arc<CollectionAppState>>,
    Path(shard_id): Path<u32>,
) -> Result<Json<ShardDetailResp>, StatusCode> {
    let router = SHARD_ROUTER.read();
    let router = router.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    let shard_config = router.get_shard_config(shard_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    let is_local = router.is_local_shard(shard_id);
    let vector_count = if is_local {
        state.manager.get(&shard_config.collection).map(|c| c.len() as u64)
    } else {
        None
    };

    Ok(Json(ShardDetailResp {
        shard_id: shard_config.id,
        node: shard_config.node,
        collection: shard_config.collection,
        is_primary: shard_config.is_primary,
        is_local,
        vector_count,
    }))
}

/// Add a new shard.
async fn add_shard(
    Json(payload): Json<AddShardReq>,
) -> Result<Json<ShardDetailResp>, StatusCode> {
    let router = SHARD_ROUTER.read();
    let router = router.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    let shard_config = ShardConfig {
        id: payload.shard_id,
        node: payload.node.clone(),
        collection: payload.collection.clone(),
        replicas: Vec::new(),
        is_primary: payload.is_primary,
    };

    router.add_shard(shard_config.clone());
    let is_local = router.is_local_shard(payload.shard_id);

    Ok(Json(ShardDetailResp {
        shard_id: payload.shard_id,
        node: payload.node,
        collection: payload.collection,
        is_primary: payload.is_primary,
        is_local,
        vector_count: None,
    }))
}

/// Remove a shard.
async fn remove_shard(
    Path(shard_id): Path<u32>,
) -> Result<StatusCode, StatusCode> {
    let router = SHARD_ROUTER.read();
    let router = router.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    router.remove_shard(shard_id);
    Ok(StatusCode::NO_CONTENT)
}

/// Get shard for a vector ID.
async fn get_shard_for_id(
    Path(vector_id): Path<u64>,
) -> Result<Json<RouteInfoResp>, StatusCode> {
    let router = SHARD_ROUTER.read();
    let router = router.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    let shard_id = router.route_by_id(vector_id)
        .ok_or(StatusCode::NOT_FOUND)?;
    let is_local = router.is_local_shard(shard_id);
    let collection_name = router.get_shard_collection(shard_id);
    let node = router.get_shard_node(shard_id);

    Ok(Json(RouteInfoResp {
        vector_id,
        shard_id,
        is_local,
        collection_name,
        node,
    }))
}
