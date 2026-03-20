use tonic::{Request, Response, Status};
use crate::proto::vectordb::vector_service_server::VectorService;
use crate::proto::vectordb::{UpsertRequest, UpsertResponse, SearchRequest, SearchResponse, GetRequest, GetResponse, SearchResult, Vector};
use crate::storage::{VectorStore, MetadataStore};
use crate::index::HnswIndexer;
use std::sync::Arc;
use roaring::RoaringBitmap;

pub mod http;

/// Efficiently create a RoaringBitmap from a list of u64 IDs.
/// Uses sorted iterator for optimal performance.
#[inline]
pub fn create_filter_bitmap(ids: &[u64]) -> RoaringBitmap {
    if ids.is_empty() {
        return RoaringBitmap::new();
    }

    // Convert to u32 and sort for optimal bitmap creation
    let mut ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
    ids_u32.sort_unstable();

    // from_sorted_iter is much faster than individual inserts
    RoaringBitmap::from_sorted_iter(ids_u32.into_iter()).unwrap_or_default()
}

/// Create a RoaringBitmap from a range of IDs (inclusive).
/// Very efficient for contiguous ID ranges.
#[inline]
pub fn create_range_bitmap(start: u64, end: u64) -> RoaringBitmap {
    let mut rb = RoaringBitmap::new();
    rb.insert_range(start as u32..=end as u32);
    rb
}

#[derive(Clone)]
pub struct VectorServiceImpl {
    pub vector_store: Arc<dyn VectorStore>,
    pub metadata_store: Arc<dyn MetadataStore>,
    pub indexer: Arc<HnswIndexer>,
}

impl VectorServiceImpl {
    pub fn new(
        vector_store: Arc<dyn VectorStore>,
        metadata_store: Arc<dyn MetadataStore>,
        indexer: Arc<HnswIndexer>,
    ) -> Self {
        Self {
            vector_store,
            metadata_store,
            indexer,
        }
    }
}

#[tonic::async_trait]
impl VectorService for VectorServiceImpl {
    async fn upsert(&self, request: Request<UpsertRequest>) -> Result<Response<UpsertResponse>, Status> {
        let req = request.into_inner();
        let id_val = req.id;
        
        // Basic validation
        if let Some(vec) = &req.vector {
             // 1. Store Vector
             // In a real system, we'd handle ID generation if 0, or check existence.
             // Here we assume Append-Only internally but we need to map external ID to internal ID if they differ.
             // For MVP, let's assume external ID matching if possible or use the returned ID from store.
             // But hnsw-rs and our store usage might differ.
             // Our VectorStore::insert returns a u64 (internal ID).
             // HNSW insert takes usize.
             
             // Issue: If user provides ID 100, we want to store it as 100.
             // My MemmapVectorStore simply appends and returns the next ID. It doesn't support random ID insertion easily.
             // FIX: For MVP, we ignore req.id and return the new internal ID, or we must implement a mapping.
             // Production systems (Milvus) map Primary Key <-> Internal ID.
             // Let's implement mapping in MetadataStore or similar?
             // "SledMetadataStore" can store mappings.
             
             // Simplification for MVP: We use the returned internal ID as the definitive ID.
             let internal_id = self.vector_store.insert(&vec.elements)
                 .map_err(|e| Status::internal(format!("Storage error: {}", e)))?;
                 
             // 2. Index Vector
             self.indexer.insert(internal_id, &vec.elements)
                 .map_err(|e| Status::internal(format!("Index error: {}", e)))?;
                 
             // 3. Store Metadata
             for (k, v) in req.metadata {
                 self.metadata_store.insert(internal_id, k, v)
                    .map_err(|e| Status::internal(format!("Metadata error: {}", e)))?;
             }
             
             Ok(Response::new(UpsertResponse {
                 id: internal_id,
                 success: true,
             }))
        } else {
            Err(Status::invalid_argument("Missing vector"))
        }
    }

    async fn search(&self, request: Request<SearchRequest>) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let k = req.k as usize;

        if let Some(vec) = req.vector {
            // Build filter bitmap - try payload index first, then explicit IDs
            let has_metadata_filter = !req.filter.is_empty();

            // Try payload index pre-filtering (functional style)
            let (filter_bitmap, used_index_filter) = if has_metadata_filter {
                let conditions: Vec<(&str, &str)> = req
                    .filter
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .collect();

                if let Some(indexed_bitmap) = self.metadata_store.try_filter_and(&conditions) {
                    // Combine with explicit filter_ids if present
                    let combined = if !req.filter_ids.is_empty() {
                        let id_bitmap = create_filter_bitmap(&req.filter_ids);
                        indexed_bitmap & id_bitmap
                    } else {
                        indexed_bitmap
                    };
                    (Some(combined), true)
                } else if !req.filter_ids.is_empty() {
                    (Some(create_filter_bitmap(&req.filter_ids)), false)
                } else {
                    (None, false)
                }
            } else if !req.filter_ids.is_empty() {
                (Some(create_filter_bitmap(&req.filter_ids)), false)
            } else {
                (None, false)
            };

            let results = self.indexer.search(&vec.elements, k, filter_bitmap.as_ref())
                .map_err(|e| Status::internal(format!("Search error: {}", e)))?;

            // Post-filtering only if metadata filter exists AND index was not used
            use rayon::prelude::*;

            let filtered_results: Vec<SearchResult> = if has_metadata_filter && !used_index_filter {
                // Fallback: parallel post-filter for non-indexed fields
                results.into_par_iter()
                    .filter_map(|(id, score)| {
                        // Functional: all conditions must match
                        let matches = req.filter.iter().all(|(fk, fv)| {
                            self.metadata_store
                                .get(id, fk)
                                .ok()
                                .flatten()
                                .as_ref()
                                .map(|v| v == fv)
                                .unwrap_or(false)
                        });
                        if matches { Some(SearchResult { id, score }) } else { None }
                    })
                    .collect()
            } else {
                // No post-filter needed
                results.into_iter().map(|(id, score)| SearchResult { id, score }).collect()
            };

            Ok(Response::new(SearchResponse {
                results: filtered_results,
            }))
        } else {
             Err(Status::invalid_argument("Missing query vector"))
        }
    }

    async fn get(&self, request: Request<GetRequest>) -> Result<Response<GetResponse>, Status> {
        let req = request.into_inner();
        let id = req.id;
        
        let vec = self.vector_store.get(id);
        match vec {
            Ok(v) => {
                // Fetch basic metadata?
                // Just return found true, we don't dump all metadata in this MVP RPC unless asked.
                // Or map empty for now.
                Ok(Response::new(GetResponse {
                    vector: Some(Vector { elements: v }),
                    metadata: Default::default(), // TODO: Fetch all metadata
                    found: true,
                }))
            }
            Err(_) => {
                Ok(Response::new(GetResponse {
                    vector: None,
                    metadata: Default::default(),
                    found: false,
                }))
            }
        }
    }
}
