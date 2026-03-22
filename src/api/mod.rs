use tonic::{Request, Response, Status, Streaming};
use tokio_stream::StreamExt;
use crate::proto::vectordb::vector_service_server::VectorService;
use crate::proto::vectordb::*;
use crate::collection::{CollectionManager, CollectionConfig};
use crate::storage::{MetadataEntry, IndexedMetadata};
use crate::replication::{ReplicationManager, ReplicationRole};
use crate::payload::FilterQuery;
use std::sync::Arc;
use std::pin::Pin;
use roaring::RoaringBitmap;
use tokio::sync::mpsc;

type StreamingResponse<T> = Pin<Box<dyn tokio_stream::Stream<Item = Result<T, Status>> + Send>>;

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
    pub replication: Option<Arc<ReplicationManager>>,
}

impl VectorServiceImpl {
    pub fn new(manager: Arc<CollectionManager>) -> Self {
        Self { manager, replication: None }
    }

    pub fn with_replication(manager: Arc<CollectionManager>, replication: Arc<ReplicationManager>) -> Self {
        Self { manager, replication: Some(replication) }
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
            text_fields: info.text_fields.clone(),
            text_indexed_count: info.text_indexed_count.unwrap_or(0) as u64,
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
        config.payload.text_fields = req.text_fields;

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

        // 4. Exclude deleted vectors (using cached active bitmap)
        if let Some(active_bitmap) = collection.active_bitmap() {
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

    // ========================================================================
    // Maintenance Operations
    // ========================================================================

    async fn compact(
        &self,
        request: Request<CompactRequest>,
    ) -> Result<Response<CompactResponse>, Status> {
        let req = request.into_inner();

        let result = self.manager.compact(&req.collection)
            .map_err(|e| Status::internal(format!("{}", e)))?;

        Ok(Response::new(CompactResponse {
            vectors_before: result.vectors_before as u64,
            vectors_after: result.vectors_after as u64,
            vectors_removed: result.vectors_removed as u64,
            bytes_reclaimed: result.bytes_reclaimed,
            success: true,
        }))
    }

    // ========================================================================
    // Replication Operations
    // ========================================================================

    async fn get_replication_status(
        &self,
        _request: Request<ReplicationStatusRequest>,
    ) -> Result<Response<ReplicationStatusResponse>, Status> {
        if let Some(ref replication) = self.replication {
            let status = replication.status();

            let role_str = match status.role {
                ReplicationRole::Primary => "primary",
                ReplicationRole::Replica => "replica",
                ReplicationRole::Standalone => "standalone",
            };

            let (wal_entries, wal_size) = status.wal_stats
                .map(|s| (s.total_entries as u64, s.total_size_bytes))
                .unwrap_or((0, 0));

            let (replica_status, primary_addr, last_seq) = status.replica_state
                .map(|s| (format!("{:?}", s.status), s.primary_addr, s.last_seq))
                .unwrap_or(("".to_string(), "".to_string(), 0));

            Ok(Response::new(ReplicationStatusResponse {
                role: role_str.to_string(),
                current_seq: status.current_seq,
                wal_entries,
                wal_size_bytes: wal_size,
                replica_status,
                primary_addr,
                last_synced_seq: last_seq,
            }))
        } else {
            Ok(Response::new(ReplicationStatusResponse {
                role: "standalone".to_string(),
                current_seq: 0,
                wal_entries: 0,
                wal_size_bytes: 0,
                replica_status: "".to_string(),
                primary_addr: "".to_string(),
                last_synced_seq: 0,
            }))
        }
    }

    async fn get_wal_entries(
        &self,
        request: Request<GetWalEntriesRequest>,
    ) -> Result<Response<GetWalEntriesResponse>, Status> {
        let req = request.into_inner();

        if let Some(ref replication) = self.replication {
            let entries = replication.get_entries_from(req.from_seq)
                .map_err(|e| Status::internal(format!("{}", e)))?;

            let current_seq = replication.current_seq();
            let max_entries = if req.max_entries == 0 { usize::MAX } else { req.max_entries as usize };
            let has_more = entries.len() > max_entries;

            let entries: Vec<WalEntryProto> = entries
                .into_iter()
                .take(max_entries)
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

                    let operation_data = serde_json::to_vec(&e.operation).unwrap_or_default();

                    WalEntryProto {
                        seq: e.seq,
                        timestamp: e.timestamp.to_rfc3339(),
                        operation_type: op_type.to_string(),
                        collection,
                        operation_data,
                    }
                })
                .collect();

            Ok(Response::new(GetWalEntriesResponse {
                entries,
                current_seq,
                has_more,
            }))
        } else {
            Ok(Response::new(GetWalEntriesResponse {
                entries: Vec::new(),
                current_seq: 0,
                has_more: false,
            }))
        }
    }

    // ========================================================================
    // Streaming Operations
    // ========================================================================

    /// Client streaming: receive chunks of vectors, return final response.
    /// OPTIMIZATION: Uses deferred flush - flushes only at the end of stream
    /// for 20-40% better throughput on large streaming inserts.
    async fn stream_upsert(
        &self,
        request: Request<Streaming<StreamUpsertRequest>>,
    ) -> Result<Response<StreamUpsertResponse>, Status> {
        let mut stream = request.into_inner();
        let mut collection_name = String::new();
        let mut total_inserted: u64 = 0;
        let mut start_id: u64 = 0;
        let mut end_id: u64 = 0;
        let mut first_chunk = true;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;

            // Get collection from first chunk
            if first_chunk {
                collection_name = chunk.collection.clone();
                first_chunk = false;
            }

            if chunk.vectors.is_empty() {
                continue;
            }

            let collection = self.get_collection(&collection_name)?;

            // Extract vectors
            let vectors: Vec<Vec<f32>> = chunk.vectors
                .iter()
                .filter_map(|bv| bv.vector.as_ref().map(|v| v.elements.clone()))
                .collect();

            if vectors.is_empty() {
                continue;
            }

            // OPTIMIZATION: Batch insert without flush - defer to end of stream
            let chunk_start_id = collection.vector_store.insert_batch_no_flush(&vectors)
                .map_err(|e| Status::internal(format!("Storage error: {}", e)))?;

            if total_inserted == 0 {
                start_id = chunk_start_id;
            }
            end_id = chunk_start_id + vectors.len() as u64 - 1;

            // Prepare index data
            let index_data: Vec<(u64, Vec<f32>)> = vectors
                .into_iter()
                .enumerate()
                .map(|(i, v)| (chunk_start_id + i as u64, v))
                .collect();

            // Batch insert to HNSW index
            collection.indexer.insert_batch(&index_data)
                .map_err(|e| Status::internal(format!("Index error: {}", e)))?;

            // Collect metadata entries
            let metadata_entries: Vec<MetadataEntry> = chunk.vectors
                .iter()
                .enumerate()
                .flat_map(|(i, bv)| {
                    let id = chunk_start_id + i as u64;
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

            total_inserted += chunk.vectors.len() as u64;
        }

        // OPTIMIZATION: Single flush at end of stream
        if !collection_name.is_empty() && total_inserted > 0 {
            if let Ok(collection) = self.get_collection(&collection_name) {
                collection.vector_store.flush()
                    .map_err(|e| Status::internal(format!("Flush error: {}", e)))?;
            }
        }

        Ok(Response::new(StreamUpsertResponse {
            total_inserted,
            start_id,
            end_id,
            success: true,
            error: String::new(),
        }))
    }

    /// Bidirectional streaming: receive vectors, send progress updates.
    type StreamUpsertBidiStream = StreamingResponse<StreamUpsertProgress>;

    async fn stream_upsert_bidi(
        &self,
        request: Request<Streaming<StreamUpsertRequest>>,
    ) -> Result<Response<Self::StreamUpsertBidiStream>, Status> {
        let mut stream = request.into_inner();
        let manager = self.manager.clone();

        let (tx, rx) = mpsc::channel(32);

        // Spawn task to process incoming stream
        tokio::spawn(async move {
            let mut collection_name = String::new();
            let mut total_inserted: u64 = 0;
            let mut first_chunk = true;

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                };

                // Get collection from first chunk
                if first_chunk {
                    collection_name = chunk.collection.clone();
                    first_chunk = false;
                }

                if chunk.vectors.is_empty() {
                    continue;
                }

                // Get collection
                let collection = if collection_name.is_empty() {
                    match manager.get_or_create_default(DEFAULT_DIM) {
                        Ok(c) => c,
                        Err(e) => {
                            let _ = tx.send(Err(Status::internal(format!("{}", e)))).await;
                            return;
                        }
                    }
                } else {
                    match manager.get(&collection_name) {
                        Some(c) => c,
                        None => {
                            let _ = tx.send(Err(Status::not_found("Collection not found"))).await;
                            return;
                        }
                    }
                };

                // Extract vectors
                let vectors: Vec<Vec<f32>> = chunk.vectors
                    .iter()
                    .filter_map(|bv| bv.vector.as_ref().map(|v| v.elements.clone()))
                    .collect();

                if vectors.is_empty() {
                    continue;
                }

                let chunk_count = vectors.len() as u32;

                // Batch insert to vector store
                let chunk_start_id = match collection.vector_store.insert_batch(&vectors) {
                    Ok(id) => id,
                    Err(e) => {
                        let _ = tx.send(Err(Status::internal(format!("{}", e)))).await;
                        return;
                    }
                };

                // Prepare index data
                let index_data: Vec<(u64, Vec<f32>)> = vectors
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| (chunk_start_id + i as u64, v))
                    .collect();

                // Batch insert to HNSW index
                if let Err(e) = collection.indexer.insert_batch(&index_data) {
                    let _ = tx.send(Err(Status::internal(format!("{}", e)))).await;
                    return;
                }

                // Collect metadata entries
                let metadata_entries: Vec<MetadataEntry> = chunk.vectors
                    .iter()
                    .enumerate()
                    .flat_map(|(i, bv)| {
                        let id = chunk_start_id + i as u64;
                        bv.metadata.iter().map(move |(k, v)| MetadataEntry {
                            id,
                            key: k.clone(),
                            value: v.clone(),
                        })
                    })
                    .collect();

                if !metadata_entries.is_empty() {
                    if let Err(e) = collection.metadata_store.insert_batch(&metadata_entries) {
                        let _ = tx.send(Err(Status::internal(format!("{}", e)))).await;
                        return;
                    }
                }

                total_inserted += chunk_count as u64;

                // Send progress update
                let progress = StreamUpsertProgress {
                    inserted_so_far: total_inserted,
                    current_chunk_start_id: chunk_start_id,
                    current_chunk_count: chunk_count,
                    progress_percent: 0.0, // Calculated by client if total known
                    is_complete: chunk.is_last,
                };

                if tx.send(Ok(progress)).await.is_err() {
                    return; // Client disconnected
                }

                if chunk.is_last {
                    break;
                }
            }

            // Send final completion message
            let _ = tx.send(Ok(StreamUpsertProgress {
                inserted_so_far: total_inserted,
                current_chunk_start_id: 0,
                current_chunk_count: 0,
                progress_percent: 100.0,
                is_complete: true,
            })).await;
        });

        let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(output_stream) as Self::StreamUpsertBidiStream))
    }

    // ========================================================================
    // Streaming Search
    // ========================================================================

    /// Server-side streaming search for large result sets.
    type StreamSearchStream = StreamingResponse<StreamSearchResponse>;

    async fn stream_search(
        &self,
        request: Request<StreamSearchRequest>,
    ) -> Result<Response<Self::StreamSearchStream>, Status> {
        let req = request.into_inner();
        let collection = self.get_collection(&req.collection)?;
        let k = req.k as usize;
        let batch_size = if req.batch_size == 0 { 100 } else { req.batch_size as usize };
        let include_vectors = req.include_vectors;
        let include_metadata = req.include_metadata;

        let vec = req.vector.ok_or_else(|| Status::invalid_argument("Missing query vector"))?;

        // Build filter bitmap (same logic as regular search)
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

        // 4. Exclude deleted vectors (using cached active bitmap)
        if let Some(active_bitmap) = collection.active_bitmap() {
            filter_bitmap = Some(
                filter_bitmap
                    .map(|existing| existing & &active_bitmap)
                    .unwrap_or(active_bitmap)
            );
        }

        // Perform search
        let results = collection.indexer.search(&vec.elements, k, filter_bitmap.as_ref())
            .map_err(|e| Status::internal(format!("Search error: {}", e)))?;

        // Apply post-filter if needed
        use rayon::prelude::*;
        let filter_clone = req.filter.clone();
        let metadata_store = collection.metadata_store.clone();

        let filtered_results: Vec<(u64, f32)> = if has_metadata_filter && !used_index_filter {
            results.into_par_iter()
                .filter_map(|(id, score)| {
                    let matches = filter_clone.iter().all(|(fk, fv)| {
                        metadata_store
                            .get(id, fk)
                            .ok()
                            .flatten()
                            .as_ref()
                            .map(|v| v == fv)
                            .unwrap_or(false)
                    });
                    if matches { Some((id, score)) } else { None }
                })
                .collect()
        } else {
            results
        };

        let total_results = filtered_results.len();
        let total_batches = (total_results + batch_size - 1) / batch_size;

        // Create channel for streaming
        let (tx, rx) = mpsc::channel(32);

        // Clone what we need for the spawned task
        let vector_store = collection.vector_store.clone();
        let metadata_store = collection.metadata_store.clone();

        // Spawn task to stream results in batches
        tokio::spawn(async move {
            let mut results_sent: u32 = 0;

            for (batch_idx, chunk) in filtered_results.chunks(batch_size).enumerate() {
                let mut batch_results = Vec::with_capacity(chunk.len());

                for &(id, score) in chunk {
                    let mut result = StreamSearchResult {
                        id,
                        score,
                        vector: None,
                        metadata: Default::default(),
                    };

                    // Include vector if requested
                    if include_vectors {
                        if let Ok(v) = vector_store.get(id) {
                            result.vector = Some(Vector { elements: v });
                        }
                    }

                    // Include metadata if requested
                    if include_metadata {
                        result.metadata = metadata_store.get_all(id);
                    }

                    batch_results.push(result);
                }

                results_sent += batch_results.len() as u32;
                let is_last = batch_idx == total_batches.saturating_sub(1);

                let response = StreamSearchResponse {
                    results: batch_results,
                    batch_index: batch_idx as u32,
                    total_batches: total_batches as u32,
                    results_so_far: results_sent,
                    total_results: total_results as u32,
                    is_last,
                };

                if tx.send(Ok(response)).await.is_err() {
                    // Client disconnected
                    return;
                }
            }

            // Handle empty results case
            if total_results == 0 {
                let _ = tx.send(Ok(StreamSearchResponse {
                    results: Vec::new(),
                    batch_index: 0,
                    total_batches: 0,
                    results_so_far: 0,
                    total_results: 0,
                    is_last: true,
                })).await;
            }
        });

        let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(output_stream) as Self::StreamSearchStream))
    }

    // ========================================================================
    // Hybrid Search (Vector + Text)
    // ========================================================================

    async fn hybrid_search(
        &self,
        request: Request<HybridSearchRequest>,
    ) -> Result<Response<HybridSearchResponse>, Status> {
        let req = request.into_inner();
        let collection = self.get_collection(&req.collection)?;

        let vec = req.vector.ok_or_else(|| Status::invalid_argument("Missing query vector"))?;
        let k = req.k as usize;
        let alpha = if req.alpha == 0.0 { 0.5 } else { req.alpha.clamp(0.0, 1.0) };
        let use_rrf = req.fusion_method == "rrf";
        let include_scores = req.include_scores;

        // Build filter bitmap (same logic as regular search)
        let mut filter_bitmap: Option<RoaringBitmap> = None;

        // 1. String field filters
        if !req.filter.is_empty() {
            let conditions: Vec<(&str, &str)> = req.filter
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();

            if let Some(indexed_bitmap) = collection.metadata_store.try_filter_and(&conditions) {
                filter_bitmap = Some(indexed_bitmap);
            }
        }

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

        // Perform hybrid search
        let results = collection.hybrid_search_filtered(
            &vec.elements,
            &req.query,
            k,
            alpha,
            use_rrf,
            filter_bitmap.as_ref(),
        ).map_err(|e| Status::internal(format!("Hybrid search error: {}", e)))?;

        // Convert to proto response
        let proto_results: Vec<HybridSearchResult> = results
            .into_iter()
            .map(|r| {
                let mut metadata = std::collections::HashMap::new();
                if include_scores {
                    // Include metadata for debugging
                    for (k, v) in collection.metadata_store.get_all(r.id) {
                        metadata.insert(k, v);
                    }
                }

                HybridSearchResult {
                    id: r.id,
                    combined_score: r.combined_score,
                    vector_score: if include_scores { r.vector_score } else { 0.0 },
                    text_score: if include_scores { r.text_score } else { 0.0 },
                    metadata,
                }
            })
            .collect();

        Ok(Response::new(HybridSearchResponse {
            results: proto_results,
            total_candidates: k as u32 * 2, // We oversample by 2x
        }))
    }

    async fn text_search(
        &self,
        request: Request<TextSearchRequest>,
    ) -> Result<Response<TextSearchResponse>, Status> {
        let req = request.into_inner();
        let collection = self.get_collection(&req.collection)?;

        let k = req.k as usize;

        // Perform text search
        let results = collection.text_search(&req.query, k)
            .map_err(|e| Status::internal(format!("Text search error: {}", e)))?;

        // Apply metadata filter if present
        let filtered_results: Vec<_> = if !req.filter.is_empty() {
            results.into_iter()
                .filter(|r| {
                    req.filter.iter().all(|(fk, fv)| {
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

        // Convert to proto response
        let proto_results: Vec<TextSearchResult> = filtered_results
            .into_iter()
            .map(|r| {
                let metadata = collection.metadata_store.get_all(r.id);
                TextSearchResult {
                    id: r.id,
                    score: r.score,
                    metadata,
                }
            })
            .collect();

        Ok(Response::new(TextSearchResponse {
            results: proto_results,
        }))
    }
}
