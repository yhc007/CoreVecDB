use tonic::{Request, Response, Status};
use crate::proto::vectordb::vector_service_server::VectorService;
use crate::proto::vectordb::*;
use crate::collection::{CollectionManager, CollectionConfig};
use crate::storage::{VectorStore, MetadataStore, MetadataEntry, IndexedMetadata};
use crate::payload::FilterQuery;
use std::sync::Arc;
use roaring::RoaringBitmap;

pub mod http;

const DEFAULT_DIM: usize = 128;

/// Efficiently create a RoaringBitmap from a list of u64 IDs.
#[inline]
pub fn create_filter_bitmap(ids: &[u64]) -> RoaringBitmap {
    if ids.is_empty() {
        return RoaringBitmap::new();
    }
    let mut ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
    ids_u32.sort_unstable();
    RoaringBitmap::from_sorted_iter(ids_u32.into_iter()).unwrap_or_default()
}

/// Create a RoaringBitmap from a range of IDs (inclusive).
#[inline]
pub fn create_range_bitmap(start: u64, end: u64) -> RoaringBitmap {
    let mut rb = RoaringBitmap::new();
    rb.insert_range(start as u32..=end as u32);
    rb
}

/// Multi-collection gRPC service implementation.
#[derive(Clone)]
pub struct VectorServiceImpl {
    pub manager: Arc<CollectionManager>,
}

impl VectorServiceImpl {
    pub fn new(manager: Arc<CollectionManager>) -> Self {
        Self { manager }
    }

    fn get_collection(&self, name: &str) -> Result<Arc<crate::collection::Collection>, Status> {
        if name.is_empty() {
            self.manager.get_or_create_default(DEFAULT_DIM)
                .map_err(|e| Status::internal(format!("Failed to get default collection: {}", e)))
        } else {
            self.manager.get(name)
                .ok_or_else(|| Status::not_found(format!("Collection '{}' not found", name)))
        }
    }

    fn to_collection_info(info: &crate::collection::CollectionInfo) -> CollectionInfo {
        CollectionInfo {
            name: info.name.clone(),
            dim: info.dim as u32,
            vector_count: info.vector_count as u64,
            deleted_count: info.deleted_count as u64,
            distance_metric: info.distance_metric.clone(),
            quantization_enabled: info.quantization_enabled,
            indexed_fields: info.indexed_fields.clone(),
            numeric_fields: info.numeric_fields.clone(),
        }
    }

    fn apply_range_filters(
        metadata_store: &dyn IndexedMetadata,
        range_filters: &[RangeFilter],
    ) -> Option<RoaringBitmap> {
        if range_filters.is_empty() {
            return None;
        }

        let queries: Vec<FilterQuery> = range_filters
            .iter()
            .map(|rf| match rf.op.as_str() {
                "gt" => FilterQuery::gt_f(&rf.field, rf.value),
                "gte" => FilterQuery::gte_f(&rf.field, rf.value),
                "lt" => FilterQuery::lt_f(&rf.field, rf.value),
                "lte" => FilterQuery::lte_f(&rf.field, rf.value),
                "range" => FilterQuery::range_f(&rf.field, rf.min, rf.max),
                "between" => FilterQuery::and(vec![
                    FilterQuery::gt_f(&rf.field, rf.min),
                    FilterQuery::lt_f(&rf.field, rf.max),
                ]),
                _ => FilterQuery::gte_f(&rf.field, rf.value),
            })
            .collect();

        let combined = if queries.len() == 1 {
            queries.into_iter().next().unwrap()
        } else {
            FilterQuery::and(queries)
        };

        metadata_store.filter(&combined)
    }
}

#[tonic::async_trait]
impl VectorService for VectorServiceImpl {
    // ========================================================================
    // Collection Management
    // ========================================================================

    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();

        let mut config = CollectionConfig::new(&req.name, req.dim as usize);

        if !req.distance.is_empty() {
            config = config.with_distance(&req.distance);
        }

        if req.quantization_enabled {
            config = config.with_quantization(true, true);
        }

        config.payload.indexed_fields = req.indexed_fields;
        config.payload.numeric_fields = req.numeric_fields;

        match self.manager.create(config) {
            Ok(collection) => Ok(Response::new(CreateCollectionResponse {
                info: Some(Self::to_collection_info(&collection.info())),
                success: true,
            })),
            Err(e) => Err(Status::already_exists(format!("{}", e))),
        }
    }

    async fn list_collections(
        &self,
        _request: Request<ListCollectionsRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        let collections = self.manager.list()
            .into_iter()
            .map(|info| Self::to_collection_info(&info))
            .collect();

        Ok(Response::new(ListCollectionsResponse { collections }))
    }

    async fn get_collection_info(
        &self,
        request: Request<GetCollectionInfoRequest>,
    ) -> Result<Response<CollectionInfo>, Status> {
        let req = request.into_inner();
        let collection = self.get_collection(&req.collection)?;
        Ok(Response::new(Self::to_collection_info(&collection.info())))
    }

    async fn delete_collection(
        &self,
        request: Request<DeleteCollectionRequest>,
    ) -> Result<Response<DeleteCollectionResponse>, Status> {
        let req = request.into_inner();
        match self.manager.delete(&req.collection) {
            Ok(_) => Ok(Response::new(DeleteCollectionResponse { success: true })),
            Err(e) => Err(Status::not_found(format!("{}", e))),
        }
    }

    // ========================================================================
    // Vector Operations
    // ========================================================================

    async fn upsert(
        &self,
        request: Request<UpsertRequest>,
    ) -> Result<Response<UpsertResponse>, Status> {
        let req = request.into_inner();
        let collection = self.get_collection(&req.collection)?;

        let vec = req.vector.ok_or_else(|| Status::invalid_argument("Missing vector"))?;

        let internal_id = collection.vector_store.insert(&vec.elements)
            .map_err(|e| Status::internal(format!("Storage error: {}", e)))?;

        collection.indexer.insert(internal_id, &vec.elements)
            .map_err(|e| Status::internal(format!("Index error: {}", e)))?;

        for (k, v) in req.metadata {
            collection.metadata_store.insert(internal_id, k, v)
                .map_err(|e| Status::internal(format!("Metadata error: {}", e)))?;
        }

        Ok(Response::new(UpsertResponse {
            id: internal_id,
            success: true,
        }))
    }

    async fn upsert_batch(
        &self,
        request: Request<UpsertBatchRequest>,
    ) -> Result<Response<UpsertBatchResponse>, Status> {
        let req = request.into_inner();
        let collection = self.get_collection(&req.collection)?;

        if req.vectors.is_empty() {
            return Ok(Response::new(UpsertBatchResponse {
                start_id: collection.len() as u64,
                count: 0,
                success: true,
            }));
        }

        let vectors: Vec<Vec<f32>> = req.vectors
            .iter()
            .filter_map(|bv| bv.vector.as_ref().map(|v| v.elements.clone()))
            .collect();

        let start_id = collection.vector_store.insert_batch(&vectors)
            .map_err(|e| Status::internal(format!("Storage error: {}", e)))?;

        let index_data: Vec<(u64, Vec<f32>)> = vectors
            .into_iter()
            .enumerate()
            .map(|(i, v)| (start_id + i as u64, v))
            .collect();

        collection.indexer.insert_batch(&index_data)
            .map_err(|e| Status::internal(format!("Index error: {}", e)))?;

        let metadata_entries: Vec<MetadataEntry> = req.vectors
            .iter()
            .enumerate()
            .flat_map(|(i, bv)| {
                let id = start_id + i as u64;
                bv.metadata.iter().map(move |(k, v)| MetadataEntry {
                    id,
                    key: k.clone(),
                    value: v.clone(),
                })
            })
            .collect();

        if !metadata_entries.is_empty() {
            collection.metadata_store.insert_batch(&metadata_entries)
                .map_err(|e| Status::internal(format!("Metadata error: {}", e)))?;
        }

        Ok(Response::new(UpsertBatchResponse {
            start_id,
            count: req.vectors.len() as u32,
            success: true,
        }))
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let collection = self.get_collection(&req.collection)?;
        let k = req.k as usize;

        let vec = req.vector.ok_or_else(|| Status::invalid_argument("Missing query vector"))?;

        let mut filter_bitmap: Option<RoaringBitmap> = None;

        // 1. String field filters
        let has_metadata_filter = !req.filter.is_empty();
        let used_index_filter = if has_metadata_filter {
            let conditions: Vec<(&str, &str)> = req.filter
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
        if !req.range_filters.is_empty() {
            if let Some(indexed_store) = collection.metadata_store
                .as_any()
                .and_then(|any| any.downcast_ref::<crate::storage::IndexedSledMetadataStore>())
            {
                if let Some(range_bitmap) = Self::apply_range_filters(indexed_store, &req.range_filters) {
                    filter_bitmap = Some(
                        filter_bitmap
                            .map(|existing| existing & &range_bitmap)
                            .unwrap_or(range_bitmap)
                    );
                }
            }
        }

        // 3. Explicit ID filter
        if !req.filter_ids.is_empty() {
            let id_bitmap = create_filter_bitmap(&req.filter_ids);
            filter_bitmap = Some(
                filter_bitmap
                    .map(|existing| existing & &id_bitmap)
                    .unwrap_or(id_bitmap)
            );
        }

        // 4. Exclude deleted vectors
        let deleted_bitmap = collection.deleted_bitmap();
        if !deleted_bitmap.is_empty() {
            let total = collection.len() as u64;
            let mut universe = RoaringBitmap::new();
            universe.insert_range(0..total as u32);
            let active_bitmap = &universe - &deleted_bitmap;
            filter_bitmap = Some(
                filter_bitmap
                    .map(|existing| existing & &active_bitmap)
                    .unwrap_or(active_bitmap)
            );
        }

        let results = collection.indexer.search(&vec.elements, k, filter_bitmap.as_ref())
            .map_err(|e| Status::internal(format!("Search error: {}", e)))?;

        use rayon::prelude::*;

        let filtered_results: Vec<SearchResult> = if has_metadata_filter && !used_index_filter {
            results.into_par_iter()
                .filter_map(|(id, score)| {
                    let matches = req.filter.iter().all(|(fk, fv)| {
                        collection.metadata_store
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
            results.into_iter().map(|(id, score)| SearchResult { id, score }).collect()
        };

        Ok(Response::new(SearchResponse { results: filtered_results }))
    }

    async fn get(
        &self,
        request: Request<GetRequest>,
    ) -> Result<Response<GetResponse>, Status> {
        let req = request.into_inner();
        let collection = self.get_collection(&req.collection)?;

        if collection.is_deleted(req.id) {
            return Ok(Response::new(GetResponse {
                id: req.id,
                vector: None,
                metadata: Default::default(),
                found: false,
            }));
        }

        match collection.vector_store.get(req.id) {
            Ok(v) => Ok(Response::new(GetResponse {
                id: req.id,
                vector: Some(Vector { elements: v }),
                metadata: Default::default(),
                found: true,
            })),
            Err(_) => Ok(Response::new(GetResponse {
                id: req.id,
                vector: None,
                metadata: Default::default(),
                found: false,
            })),
        }
    }

    async fn delete(
        &self,
        request: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        let req = request.into_inner();
        let collection = self.get_collection(&req.collection)?;

        match collection.delete(req.id) {
            Ok(deleted) => Ok(Response::new(DeleteResponse {
                id: req.id,
                deleted,
            })),
            Err(e) => Err(Status::not_found(format!("{}", e))),
        }
    }

    async fn delete_batch(
        &self,
        request: Request<DeleteBatchRequest>,
    ) -> Result<Response<DeleteBatchResponse>, Status> {
        let req = request.into_inner();
        let collection = self.get_collection(&req.collection)?;

        let count = collection.delete_batch(&req.ids)
            .map_err(|e| Status::internal(format!("{}", e)))?;

        Ok(Response::new(DeleteBatchResponse {
            deleted_count: count as u32,
            requested_count: req.ids.len() as u32,
        }))
    }

    async fn update(
        &self,
        request: Request<UpdateRequest>,
    ) -> Result<Response<UpdateResponse>, Status> {
        let req = request.into_inner();
        let collection = self.get_collection(&req.collection)?;

        if req.id as usize >= collection.len() {
            return Err(Status::not_found("Vector ID not found"));
        }
        if collection.is_deleted(req.id) {
            return Err(Status::failed_precondition("Vector already deleted"));
        }

        let vec = req.vector.ok_or_else(|| Status::invalid_argument("Missing vector"))?;

        collection.delete(req.id)
            .map_err(|e| Status::internal(format!("{}", e)))?;

        let new_id = collection.vector_store.insert(&vec.elements)
            .map_err(|e| Status::internal(format!("{}", e)))?;

        collection.indexer.insert(new_id, &vec.elements)
            .map_err(|e| Status::internal(format!("{}", e)))?;

        for (k, v) in req.metadata {
            collection.metadata_store.insert(new_id, k, v)
                .map_err(|e| Status::internal(format!("{}", e)))?;
        }

        Ok(Response::new(UpdateResponse {
            old_id: req.id,
            new_id,
            updated: true,
        }))
    }

    async fn update_metadata(
        &self,
        request: Request<UpdateMetadataRequest>,
    ) -> Result<Response<UpdateMetadataResponse>, Status> {
        let req = request.into_inner();
        let collection = self.get_collection(&req.collection)?;

        if req.id as usize >= collection.len() {
            return Err(Status::not_found("Vector ID not found"));
        }
        if collection.is_deleted(req.id) {
            return Err(Status::failed_precondition("Vector already deleted"));
        }

        for (k, v) in req.metadata {
            collection.metadata_store.insert(req.id, k, v)
                .map_err(|e| Status::internal(format!("{}", e)))?;
        }

        Ok(Response::new(UpdateMetadataResponse {
            id: req.id,
            updated: true,
        }))
    }

    // ========================================================================
    // Snapshot Operations
    // ========================================================================

    async fn create_snapshot(
        &self,
        request: Request<CreateSnapshotRequest>,
    ) -> Result<Response<SnapshotInfo>, Status> {
        let req = request.into_inner();

        let info = self.manager.create_snapshot(&req.collection)
            .map_err(|e| Status::internal(format!("{}", e)))?;

        Ok(Response::new(SnapshotInfo {
            name: info.name,
            collection: info.collection,
            created_at: info.created_at.to_rfc3339(),
            vector_count: info.vector_count as u64,
            size_bytes: info.size_bytes,
        }))
    }

    async fn list_snapshots(
        &self,
        request: Request<ListSnapshotsRequest>,
    ) -> Result<Response<ListSnapshotsResponse>, Status> {
        let req = request.into_inner();

        let snapshots = self.manager.list_snapshots(&req.collection)
            .map_err(|e| Status::internal(format!("{}", e)))?
            .into_iter()
            .map(|info| SnapshotInfo {
                name: info.name,
                collection: info.collection,
                created_at: info.created_at.to_rfc3339(),
                vector_count: info.vector_count as u64,
                size_bytes: info.size_bytes,
            })
            .collect();

        Ok(Response::new(ListSnapshotsResponse { snapshots }))
    }

    async fn restore_snapshot(
        &self,
        request: Request<RestoreSnapshotRequest>,
    ) -> Result<Response<CollectionInfo>, Status> {
        let req = request.into_inner();

        let new_name = if req.new_collection_name.is_empty() {
            None
        } else {
            Some(req.new_collection_name.as_str())
        };

        let collection = self.manager.restore_snapshot(&req.snapshot_name, new_name)
            .map_err(|e| Status::internal(format!("{}", e)))?;

        Ok(Response::new(Self::to_collection_info(&collection.info())))
    }

    async fn delete_snapshot(
        &self,
        request: Request<DeleteSnapshotRequest>,
    ) -> Result<Response<DeleteSnapshotResponse>, Status> {
        let req = request.into_inner();

        self.manager.delete_snapshot(&req.snapshot_name)
            .map_err(|e| Status::not_found(format!("{}", e)))?;

        Ok(Response::new(DeleteSnapshotResponse { success: true }))
    }
}
