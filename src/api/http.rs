use axum::{
    routing::{get, post, delete, put, patch},
    Router, Json, extract::{State, Path},
    http::StatusCode,
};
use std::sync::Arc;
use crate::api::{create_filter_bitmap, create_range_bitmap};
use crate::collection::{CollectionManager, CollectionConfig, CollectionInfo, SnapshotInfo};
use crate::storage::{MetadataEntry, IndexedMetadata};
use crate::payload::FilterQuery;
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

// ============================================================================
// Multi-Collection API
// ============================================================================

/// State for collection-aware router.
pub struct CollectionAppState {
    pub manager: Arc<CollectionManager>,
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
}

/// Create router with multi-collection support.
pub async fn collection_router(manager: Arc<CollectionManager>) -> Router {
    let state = Arc::new(CollectionAppState { manager });

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
        // Snapshot operations
        .route("/collections/:name/snapshots", get(list_snapshots))
        .route("/collections/:name/snapshots", post(create_snapshot))
        .route("/snapshots/:snapshot_name", delete(delete_snapshot))
        .route("/snapshots/:snapshot_name/restore", post(restore_snapshot))
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

    if req.indexed_fields.is_some() || req.numeric_fields.is_some() {
        config.payload.indexed_fields = req.indexed_fields.unwrap_or_default();
        config.payload.numeric_fields = req.numeric_fields.unwrap_or_default();
    }

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

    // Build filter bitmap
    let mut filter_bitmap: Option<roaring::RoaringBitmap> = None;

    // 1. String field filters (exact match)
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
