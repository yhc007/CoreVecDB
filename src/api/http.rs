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

    // Build filter bitmap efficiently
    let filter_bitmap = if let Some((start, end)) = payload.filter_id_range {
        // Range filter is most efficient for contiguous IDs
        Some(create_range_bitmap(start, end))
    } else if let Some(ref ids) = payload.filter_ids {
        if !ids.is_empty() {
            Some(create_filter_bitmap(ids))
        } else {
            None
        }
    } else {
        None
    };

    let results = state.service.indexer.search(&payload.vector, payload.k as usize, filter_bitmap.as_ref())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Filter logic same as gRPC
    let mut final_results = Vec::new();
    for (id, score) in results {
        if !payload.filter.is_empty() {
             let mut match_all = true;
             for (fk, fv) in &payload.filter {
                 if let Ok(val) = state.service.metadata_store.get(id, fk) {
                     if val.as_deref() != Some(fv) {
                         match_all = false;
                         break;
                     }
                 } else {
                      match_all = false;
                      break;
                 }
             }
             if !match_all {
                 continue;
             }
        }
        final_results.push(JsonSearchResult { id, score });
    }

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
