use axum::{
    routing::{get, post, delete},
    Router, Json, extract::{State, Path},
    http::StatusCode,
};
use std::sync::Arc;
use crate::api::{VectorServiceImpl, create_filter_bitmap, create_range_bitmap};
use crate::collection::{CollectionManager, CollectionConfig, CollectionInfo};
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

#[derive(Deserialize)]
pub struct JsonSearchReq {
    vector: Vec<f32>,
    k: u32,
    filter: std::collections::HashMap<String, String>,
    filter_ids: Option<Vec<u64>>,
    /// Range filter: [start, end] inclusive. More efficient than listing all IDs.
    filter_id_range: Option<(u64, u64)>,
}

#[derive(Serialize)]
pub struct JsonSearchResp {
    results: Vec<JsonSearchResult>,
}

#[derive(Serialize)]
pub struct JsonSearchResult {
    id: u64,
    score: f32,
}

pub struct AppState {
    pub service: Arc<VectorServiceImpl>,
}

pub async fn router(service: Arc<VectorServiceImpl>) -> Router {
    let state = Arc::new(AppState { service });

    Router::new()
        .route("/upsert", post(upsert))
        .route("/search", post(search))
        .route("/vectors/:id", get(get_vector))
        .route("/stats", get(get_stats))
        .layer(
            tower_http::cors::CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_methods(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any),
        )
        .nest_service("/", tower_http::services::ServeDir::new("ui"))
        .with_state(state)
}

async fn get_stats(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let count = state.service.vector_store.len();
    // Assuming dim is constant 128 for now as per main.rs
    // Ideally we expose dim from store or config. 
    // MemmapVectorStore has dim field but it is not exposed via trait.
    // We can add dim() to VectorStore trait later if needed.
    Json(serde_json::json!({
        "vector_count": count,
        "dimension": 128, 
        "status": "Ready"
    }))
}

async fn upsert(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<JsonUpsertReq>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let internal_id = state.service.vector_store.insert(&payload.vector)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    state.service.indexer.insert(internal_id, &payload.vector)
         .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    for (k, v) in payload.metadata {
        state.service.metadata_store.insert(internal_id, k, v)
             .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }

    Ok(Json(serde_json::json!({ "id": internal_id, "success": true })))
}

async fn search(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<JsonSearchReq>,
) -> Result<Json<JsonSearchResp>, StatusCode> {

    // Build filter bitmap efficiently - combine all filter sources
    // Priority: payload index (fastest) > id range > explicit ids
    let mut filter_bitmap: Option<roaring::RoaringBitmap> = None;

    // 1. Try payload index pre-filtering (indexed fields)
    //    Functional: collect conditions as Vec of refs, then try index
    let has_metadata_filter = !payload.filter.is_empty();
    let used_index_filter = if has_metadata_filter {
        let conditions: Vec<(&str, &str)> = payload
            .filter
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        if let Some(indexed_bitmap) = state.service.metadata_store.try_filter_and(&conditions) {
            // Index hit - use pre-filtered IDs
            filter_bitmap = Some(indexed_bitmap);
            true
        } else {
            false
        }
    } else {
        false
    };

    // 2. Combine with range filter if present
    if let Some((start, end)) = payload.filter_id_range {
        let range_bitmap = create_range_bitmap(start, end);
        filter_bitmap = Some(
            filter_bitmap
                .map(|existing| existing & &range_bitmap)
                .unwrap_or(range_bitmap)
        );
    }

    // 3. Combine with explicit id filter if present
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

    let results = state.service.indexer.search(&payload.vector, payload.k as usize, filter_bitmap.as_ref())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Post-filter only if metadata filter exists AND index was not used
    let final_results: Vec<JsonSearchResult> = if has_metadata_filter && !used_index_filter {
        // Fallback: post-filter (for non-indexed fields)
        results
            .into_iter()
            .filter(|(id, _)| {
                payload.filter.iter().all(|(fk, fv)| {
                    state.service.metadata_store
                        .get(*id, fk)
                        .ok()
                        .flatten()
                        .as_ref()
                        .map(|v| v == fv)
                        .unwrap_or(false)
                })
            })
            .map(|(id, score)| JsonSearchResult { id, score })
            .collect()
    } else {
        // No post-filter needed (either no metadata filter or index was used)
        results
            .into_iter()
            .map(|(id, score)| JsonSearchResult { id, score })
            .collect()
    };

    Ok(Json(JsonSearchResp { results: final_results }))
}

async fn get_vector(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if let Ok(vec) = state.service.vector_store.get(id) {
        Ok(Json(serde_json::json!({ "id": id, "vector": vec, "found": true })))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
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
        .route("/collections/:name/search", post(collection_search))
        .route("/collections/:name/vectors/:id", get(collection_get_vector))
        // Legacy routes (use default collection)
        .route("/upsert", post(legacy_upsert))
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

async fn collection_search(
    State(state): State<Arc<CollectionAppState>>,
    Path(name): Path<String>,
    Json(payload): Json<JsonSearchReq>,
) -> Result<Json<JsonSearchResp>, StatusCode> {
    let collection = state.manager.get(&name)
        .ok_or(StatusCode::NOT_FOUND)?;

    // Build filter bitmap
    let mut filter_bitmap: Option<roaring::RoaringBitmap> = None;

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

    if let Some((start, end)) = payload.filter_id_range {
        let range_bitmap = create_range_bitmap(start, end);
        filter_bitmap = Some(
            filter_bitmap
                .map(|existing| existing & &range_bitmap)
                .unwrap_or(range_bitmap)
        );
    }

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

    let results = collection.indexer.search(&payload.vector, payload.k as usize, filter_bitmap.as_ref())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

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
            .map(|(id, score)| JsonSearchResult { id, score })
            .collect()
    } else {
        results
            .into_iter()
            .map(|(id, score)| JsonSearchResult { id, score })
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

async fn legacy_search(
    State(state): State<Arc<CollectionAppState>>,
    Json(payload): Json<JsonSearchReq>,
) -> Result<Json<JsonSearchResp>, StatusCode> {
    let collection = state.manager.get_or_create_default(DEFAULT_DIM)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Build filter bitmap
    let mut filter_bitmap: Option<roaring::RoaringBitmap> = None;

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

    if let Some((start, end)) = payload.filter_id_range {
        let range_bitmap = create_range_bitmap(start, end);
        filter_bitmap = Some(
            filter_bitmap
                .map(|existing| existing & &range_bitmap)
                .unwrap_or(range_bitmap)
        );
    }

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

    let results = collection.indexer.search(&payload.vector, payload.k as usize, filter_bitmap.as_ref())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

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
            .map(|(id, score)| JsonSearchResult { id, score })
            .collect()
    } else {
        results
            .into_iter()
            .map(|(id, score)| JsonSearchResult { id, score })
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
