use axum::{
    routing::{get, post},
    Router, Json, extract::{State, Path},
    http::StatusCode,
};
use std::sync::Arc;
use crate::api::{VectorServiceImpl, create_filter_bitmap, create_range_bitmap};
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
