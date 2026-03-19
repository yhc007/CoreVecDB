use tonic::{Request, Response, Status};
use crate::proto::vectordb::vector_service_server::VectorService;
use crate::proto::vectordb::{UpsertRequest, UpsertResponse, SearchRequest, SearchResponse, GetRequest, GetResponse, SearchResult, Vector};
use crate::storage::{VectorStore, MetadataStore};
use crate::index::HnswIndexer;
use std::sync::Arc;
use roaring::RoaringBitmap;

pub mod http;

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
            let filter_bitmap = if !req.filter_ids.is_empty() {
                 let mut rb = RoaringBitmap::new();
                 // Optimize by creating from sorted iter if possible, or just insert
                 for id in req.filter_ids {
                     rb.insert(id as u32);
                 }
                 Some(rb)
            } else {
                 None
            };
            
            let results = self.indexer.search(&vec.elements, k, filter_bitmap.as_ref())
                .map_err(|e| Status::internal(format!("Search error: {}", e)))?;
            
            // Post-filtering with Rayon (Functional & Parallel)
            // We collect results into a Vec first, then par_iter to filter.
            // Since `search` is async and `rayon` is blocking CPU work, 
            // strictly speaking we should use `spawn_blocking` if it takes too long,
            // but for this granular work (metadata lookup) it might be fine or we verify perf.
            
            use rayon::prelude::*;
            
            let filtered_results: Vec<SearchResult> = if !req.filter.is_empty() {
                // Rayon requires the closure to be Send + Sync.
                // Our metadata_store is Arc<dyn MetadataStore> which should be, 
                // but SledMetadataStore needs to handle concurrent reads (which it does).
                results.into_par_iter()
                    .filter_map(|(id, score)| {
                         // Check all filters
                         for (fk, fv) in &req.filter {
                             // Note: get() might block, which is not ideal in async fn if prolonged,
                             // but we are aiming for throughput here.
                             match self.metadata_store.get(id, fk) {
                                 Ok(Some(val)) if val == *fv => {},
                                 _ => return None,
                             }
                         }
                         Some(SearchResult { id, score })
                    })
                    .collect()
            } else {
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
