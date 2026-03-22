//! Multi-Collection support for VectorDB.
//!
//! Each collection is an independent vector space with its own:
//! - Vector storage (mmap or quantized)
//! - HNSW index
//! - Metadata store with payload indexing
//! - Configuration (dimension, index params, etc.)

use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::cache::{QueryCache, QueryCacheKey, FilterBitmapCache};
use crate::index::{HnswIndexer, DistanceMetric};
use crate::index::adaptive::{AdaptiveIndexer, AdaptiveIndexConfig, AdaptiveIndexStats, IndexType, DEFAULT_BRUTE_FORCE_THRESHOLD};
use crate::index::prewarm::{prewarm_collection, PrewarmConfig};
use crate::query::{QueryPlanner, FilterOrder, FilterPlan, FilterStrategy};
use crate::storage::{
    MemmapVectorStore, QuantizedMemmapVectorStore, SledMetadataStore,
    IndexedSledMetadataStore, VectorStore, MetadataStore, MetadataEntry,
};
use crate::text::{TextIndex, TextSearchResult, HybridSearchResult, hybrid_combine, rrf_combine};
use crate::wal::{WriteAheadLog, WalConfig, WalOperation, WalStats};
use roaring::RoaringBitmap;
use std::time::Duration;

/// Collection configuration stored on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Collection name
    pub name: String,
    /// Vector dimension
    pub dim: usize,
    /// Distance metric
    pub distance: DistanceMetricConfig,
    /// HNSW parameters
    pub hnsw: HnswConfig,
    /// Quantization settings
    pub quantization: QuantizationConfig,
    /// Payload index settings
    pub payload: PayloadConfig,
    /// WAL settings
    #[serde(default)]
    pub wal: WalConfigWrapper,
    /// Adaptive index settings (brute-force vs HNSW based on size)
    #[serde(default)]
    pub adaptive: AdaptiveConfigWrapper,
}

/// Adaptive index configuration wrapper for Collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfigWrapper {
    /// Enable adaptive index selection (default: true)
    pub enabled: bool,
    /// Threshold for switching from brute-force to HNSW (default: 2000)
    pub brute_force_threshold: usize,
}

impl Default for AdaptiveConfigWrapper {
    fn default() -> Self {
        Self {
            enabled: true,
            brute_force_threshold: DEFAULT_BRUTE_FORCE_THRESHOLD,
        }
    }
}

/// WAL configuration wrapper for Collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalConfigWrapper {
    /// Enable WAL (default: true)
    pub enabled: bool,
    /// Sync to disk after each write (safer but slower)
    pub sync_on_write: bool,
    /// Number of entries before auto-checkpoint (0 = manual only)
    pub checkpoint_interval: usize,
}

impl Default for WalConfigWrapper {
    fn default() -> Self {
        Self {
            enabled: true,
            sync_on_write: false,
            checkpoint_interval: 10000,
        }
    }
}

impl From<WalConfigWrapper> for WalConfig {
    fn from(w: WalConfigWrapper) -> Self {
        WalConfig {
            enabled: w.enabled,
            sync_on_write: w.sync_on_write,
            checkpoint_interval: w.checkpoint_interval,
            max_wal_size: 256 * 1024 * 1024, // 256MB default
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DistanceMetricConfig {
    pub metric: String, // "euclidean", "cosine", "dot"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    pub max_elements: usize,
    pub m: usize,
    pub ef_construction: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub enabled: bool,
    pub keep_originals: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayloadConfig {
    pub index_enabled: bool,
    pub indexed_fields: Vec<String>,
    pub numeric_fields: Vec<String>,
    /// Text fields for BM25 full-text search
    #[serde(default)]
    pub text_fields: Vec<String>,
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            dim: 128,
            distance: DistanceMetricConfig {
                metric: "euclidean".to_string(),
            },
            hnsw: HnswConfig {
                max_elements: 10000,
                m: 24,
                ef_construction: 400,
            },
            quantization: QuantizationConfig {
                enabled: false,
                keep_originals: true,
            },
            payload: PayloadConfig {
                index_enabled: true,
                indexed_fields: vec![],
                numeric_fields: vec![],
                text_fields: vec![],
            },
            wal: WalConfigWrapper::default(),
            adaptive: AdaptiveConfigWrapper::default(),
        }
    }
}

impl CollectionConfig {
    /// Create config with just name and dimension.
    pub fn new(name: impl Into<String>, dim: usize) -> Self {
        Self {
            name: name.into(),
            dim,
            ..Default::default()
        }
    }

    /// Builder: set distance metric.
    pub fn with_distance(mut self, metric: &str) -> Self {
        self.distance.metric = metric.to_string();
        self
    }

    /// Builder: set HNSW parameters.
    pub fn with_hnsw(mut self, max_elements: usize, m: usize, ef_construction: usize) -> Self {
        self.hnsw.max_elements = max_elements;
        self.hnsw.m = m;
        self.hnsw.ef_construction = ef_construction;
        self
    }

    /// Builder: enable quantization.
    pub fn with_quantization(mut self, enabled: bool, keep_originals: bool) -> Self {
        self.quantization.enabled = enabled;
        self.quantization.keep_originals = keep_originals;
        self
    }

    /// Builder: set indexed fields.
    pub fn with_indexed_fields<I, S>(mut self, string_fields: I, numeric_fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.payload.indexed_fields = string_fields.into_iter().map(|s| s.into()).collect();
        self.payload.numeric_fields = numeric_fields.into_iter().map(|s| s.into()).collect();
        self
    }

    /// Builder: set text fields for BM25 full-text search.
    pub fn with_text_fields<I, S>(mut self, text_fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.payload.text_fields = text_fields.into_iter().map(|s| s.into()).collect();
        self
    }

    /// Builder: configure WAL.
    pub fn with_wal(mut self, enabled: bool, sync_on_write: bool, checkpoint_interval: usize) -> Self {
        self.wal = WalConfigWrapper {
            enabled,
            sync_on_write,
            checkpoint_interval,
        };
        self
    }

    /// Get distance metric enum.
    pub fn distance_metric(&self) -> DistanceMetric {
        DistanceMetric::from_str(&self.distance.metric)
    }
}

/// A single collection containing vectors, index, and metadata.
pub struct Collection {
    /// Configuration
    pub config: CollectionConfig,
    /// Vector storage
    pub vector_store: Arc<dyn VectorStore>,
    /// Metadata storage
    pub metadata_store: Arc<dyn MetadataStore>,
    /// HNSW index (used when adaptive is disabled or after threshold)
    pub indexer: Arc<HnswIndexer>,
    /// Adaptive indexer (brute-force for small, HNSW for large)
    adaptive_indexer: Option<AdaptiveIndexer>,
    /// BM25 text index for full-text search
    text_index: Option<TextIndex>,
    /// Base path for this collection
    base_path: PathBuf,
    /// Deleted vector IDs (soft delete tombstones)
    deleted_ids: RwLock<RoaringBitmap>,
    /// Write-Ahead Log for crash recovery
    wal: Option<WriteAheadLog>,
    /// Cached active bitmap (complement of deleted_ids)
    /// Invalidated when vectors are added or deleted
    cached_active_bitmap: RwLock<Option<CachedActiveBitmap>>,
    /// Query result cache for repeated searches
    query_cache: Option<Arc<QueryCache>>,
    /// Filter bitmap cache for repeated filters
    filter_cache: Option<Arc<FilterBitmapCache>>,
    /// Query planner for filter optimization
    query_planner: QueryPlanner,
}

/// Cached active bitmap with version tracking
struct CachedActiveBitmap {
    /// The active (non-deleted) vector IDs
    bitmap: RoaringBitmap,
    /// Total vector count when this cache was created
    vector_count: usize,
    /// Deleted count when this cache was created
    deleted_count: u64,
}

impl Collection {
    /// Create or load a collection at the given path.
    pub fn open(base_path: &Path, config: CollectionConfig) -> Result<Self> {
        fs::create_dir_all(base_path)?;

        let config_path = base_path.join("config.json");
        let vector_path = base_path.join("vectors");
        let index_path = base_path.join("index.hnsw");
        let meta_path = base_path.join("meta.sled");
        let deleted_path = base_path.join("deleted.bin");

        // Save config if new, or load existing
        let config = if config_path.exists() {
            let content = fs::read_to_string(&config_path)?;
            serde_json::from_str(&content)
                .context("Failed to parse collection config")?
        } else {
            let content = serde_json::to_string_pretty(&config)?;
            fs::write(&config_path, content)?;
            config
        };

        // Create vector store
        let dim = config.dim;
        let vector_store: Arc<dyn VectorStore> = if config.quantization.enabled {
            Arc::new(QuantizedMemmapVectorStore::new(
                vector_path.to_str().unwrap(),
                dim,
                config.quantization.keep_originals,
            )?)
        } else {
            let path = format!("{}.bin", vector_path.to_str().unwrap());
            Arc::new(MemmapVectorStore::new(&path, dim)?)
        };

        // Create metadata store
        let metadata_store: Arc<dyn MetadataStore> = if config.payload.index_enabled {
            Arc::new(IndexedSledMetadataStore::with_numeric_fields(
                meta_path.to_str().unwrap(),
                config.payload.indexed_fields.iter().map(|s| s.as_str()),
                config.payload.numeric_fields.iter().map(|s| s.as_str()),
            )?)
        } else {
            Arc::new(SledMetadataStore::new(meta_path.to_str().unwrap())?)
        };

        // Create or load index with the configured distance metric
        let graph_path = format!("{}.hnsw.graph", index_path.to_str().unwrap());
        let distance_metric = config.distance_metric();
        let indexer = if Path::new(&graph_path).exists() {
            Arc::new(HnswIndexer::load(
                &index_path,
                dim,
                distance_metric,
            )?)
        } else {
            Arc::new(HnswIndexer::with_metric(
                dim,
                config.hnsw.max_elements,
                config.hnsw.m,
                config.hnsw.ef_construction,
                distance_metric,
            ))
        };

        // Load deleted IDs if exists
        let deleted_ids = if deleted_path.exists() {
            let bytes = fs::read(&deleted_path)?;
            RoaringBitmap::deserialize_from(&bytes[..])
                .unwrap_or_else(|_| RoaringBitmap::new())
        } else {
            RoaringBitmap::new()
        };

        // Create WAL if enabled
        let wal = if config.wal.enabled {
            let wal_path = base_path.join("wal.log");
            let wal_config: WalConfig = config.wal.clone().into();
            Some(WriteAheadLog::open(&wal_path, wal_config)?)
        } else {
            None
        };

        // Create text index if text fields are configured
        let text_index = if !config.payload.text_fields.is_empty() {
            let text_path = base_path.join("text_index");
            Some(TextIndex::open(text_path, config.payload.text_fields.clone())?)
        } else {
            None
        };

        // Initialize caches with default settings
        // TODO: Make cache settings configurable via CollectionConfig
        let query_cache = Some(Arc::new(QueryCache::new(1000, Duration::from_secs(60))));
        let filter_cache = Some(Arc::new(FilterBitmapCache::new(500)));

        // Create adaptive indexer if enabled
        let adaptive_indexer = if config.adaptive.enabled {
            let adaptive_config = AdaptiveIndexConfig {
                enabled: true,
                brute_force_threshold: config.adaptive.brute_force_threshold,
                distance_metric,
                max_elements: config.hnsw.max_elements,
                m: config.hnsw.m,
                ef_construction: config.hnsw.ef_construction,
            };
            Some(AdaptiveIndexer::with_hnsw(
                adaptive_config,
                vector_store.clone(),
                indexer.clone(),
                dim,
            ))
        } else {
            None
        };

        // Create query planner for filter optimization
        let query_planner = QueryPlanner::new();

        let mut collection = Self {
            config,
            vector_store,
            metadata_store,
            indexer,
            adaptive_indexer,
            text_index,
            base_path: base_path.to_path_buf(),
            deleted_ids: RwLock::new(deleted_ids),
            wal,
            cached_active_bitmap: RwLock::new(None), // Will be computed on first use
            query_cache,
            filter_cache,
            query_planner,
        };

        // Perform WAL recovery if needed
        collection.recover_from_wal()?;

        // Pre-warm index files into OS page cache
        let prewarm_config = PrewarmConfig::default();
        if let Ok(stats) = prewarm_collection(&collection.base_path, &prewarm_config) {
            if stats.pages_loaded > 0 {
                tracing::info!(
                    "Pre-warmed collection '{}': {} pages ({:.1} MB) in {}ms",
                    collection.config.name,
                    stats.pages_loaded,
                    stats.bytes_loaded as f64 / (1024.0 * 1024.0),
                    stats.duration_ms
                );
            }
        }

        Ok(collection)
    }

    /// Recover from WAL entries since last checkpoint
    fn recover_from_wal(&mut self) -> Result<()> {
        let wal = match &self.wal {
            Some(w) => w,
            None => return Ok(()),
        };

        let entries = wal.read_for_recovery()?;
        if entries.is_empty() {
            return Ok(());
        }

        println!(
            "Recovering {} WAL entries for collection '{}'",
            entries.len(),
            self.config.name
        );

        for entry in entries {
            match entry.operation {
                WalOperation::Insert { id, vector, metadata } => {
                    // Re-insert into index (vector store should already have it)
                    if (id as usize) < self.vector_store.len() {
                        let _ = self.indexer.insert(id, &vector);
                    }
                    // Re-insert metadata
                    for (key, value) in metadata {
                        let _ = self.metadata_store.insert(id, key, value);
                    }
                }
                WalOperation::Delete { id } => {
                    let mut deleted = self.deleted_ids.write();
                    deleted.insert(id as u32);
                }
                WalOperation::BatchInsert { start_id, vectors, metadata } => {
                    for (i, vector) in vectors.iter().enumerate() {
                        let id = start_id + i as u64;
                        if (id as usize) < self.vector_store.len() {
                            let _ = self.indexer.insert(id, vector);
                        }
                        if i < metadata.len() {
                            for (key, value) in &metadata[i] {
                                let _ = self.metadata_store.insert(id, key.clone(), value.clone());
                            }
                        }
                    }
                }
                WalOperation::BatchDelete { ids } => {
                    let mut deleted = self.deleted_ids.write();
                    for id in ids {
                        deleted.insert(id as u32);
                    }
                }
                WalOperation::Checkpoint { .. } => {
                    // Checkpoint markers are informational only
                }
            }
        }

        println!(
            "WAL recovery complete for collection '{}'",
            self.config.name
        );

        Ok(())
    }

    /// Get collection name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get total vector count (including deleted).
    pub fn len(&self) -> usize {
        self.vector_store.len()
    }

    /// Get active vector count (excluding deleted).
    pub fn active_count(&self) -> usize {
        let total = self.vector_store.len();
        let deleted = self.deleted_ids.read().len() as usize;
        total.saturating_sub(deleted)
    }

    /// Get deleted vector count.
    pub fn deleted_count(&self) -> usize {
        self.deleted_ids.read().len() as usize
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.active_count() == 0
    }

    /// Check if a vector is deleted.
    pub fn is_deleted(&self, id: u64) -> bool {
        self.deleted_ids.read().contains(id as u32)
    }

    /// Insert a vector with metadata.
    /// Writes to WAL first for durability, then applies to storage and index.
    pub fn insert(&self, vector: &[f32], metadata: &[(String, String)]) -> Result<u64> {
        // Write to WAL first
        if let Some(ref wal) = self.wal {
            wal.append(WalOperation::Insert {
                id: self.vector_store.len() as u64,
                vector: vector.to_vec(),
                metadata: metadata.to_vec(),
            })?;
        }

        // Insert into vector store
        let id = self.vector_store.insert(vector)?;

        // Insert into index
        self.indexer.insert(id, vector)?;

        // Insert metadata
        for (key, value) in metadata {
            self.metadata_store.insert(id, key.clone(), value.clone())?;
        }

        // Update query planner statistics
        let meta_map: std::collections::HashMap<String, String> = metadata.iter().cloned().collect();
        self.query_planner.observe_fields(&meta_map);

        // Index text for BM25 search
        if let Some(ref text_index) = self.text_index {
            text_index.index_document(id, metadata)?;
        }

        // Check if checkpoint is needed
        self.maybe_checkpoint()?;

        // Invalidate query cache (new vector may affect search results)
        self.invalidate_query_cache();

        Ok(id)
    }

    /// Batch insert vectors with metadata.
    /// Writes to WAL first for durability.
    pub fn insert_batch(
        &self,
        vectors: &[Vec<f32>],
        metadata: &[Vec<(String, String)>],
    ) -> Result<u64> {
        if vectors.is_empty() {
            return Ok(self.vector_store.len() as u64);
        }

        // Write to WAL first
        if let Some(ref wal) = self.wal {
            wal.append(WalOperation::BatchInsert {
                start_id: self.vector_store.len() as u64,
                vectors: vectors.to_vec(),
                metadata: metadata.to_vec(),
            })?;
        }

        // Insert into vector store
        let start_id = self.vector_store.insert_batch(vectors)?;

        // Insert into index
        let vectors_with_ids: Vec<(u64, Vec<f32>)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (start_id + i as u64, v.clone()))
            .collect();
        self.indexer.insert_batch(&vectors_with_ids)?;

        // Insert metadata and update query planner statistics
        for (i, meta) in metadata.iter().enumerate() {
            let id = start_id + i as u64;
            for (key, value) in meta {
                self.metadata_store.insert(id, key.clone(), value.clone())?;
            }
            // Update query planner statistics
            let meta_map: std::collections::HashMap<String, String> = meta.iter().cloned().collect();
            self.query_planner.observe_fields(&meta_map);
        }

        // Index text for BM25 search
        if let Some(ref text_index) = self.text_index {
            let docs: Vec<(u64, Vec<(String, String)>)> = metadata
                .iter()
                .enumerate()
                .map(|(i, meta)| (start_id + i as u64, meta.clone()))
                .collect();
            text_index.index_batch(&docs)?;
        }

        // Check if checkpoint is needed
        self.maybe_checkpoint()?;

        // Invalidate query cache (new vectors may affect search results)
        self.invalidate_query_cache();

        Ok(start_id)
    }

    /// Delete a vector by ID (soft delete).
    /// Writes to WAL first for durability.
    /// Returns true if the vector was deleted, false if already deleted or not found.
    pub fn delete(&self, id: u64) -> Result<bool> {
        let total = self.vector_store.len();
        if id as usize >= total {
            return Err(anyhow::anyhow!("Vector ID {} out of bounds", id));
        }

        let mut deleted = self.deleted_ids.write();
        if deleted.contains(id as u32) {
            return Ok(false); // Already deleted
        }

        // Write to WAL first
        if let Some(ref wal) = self.wal {
            wal.append(WalOperation::Delete { id })?;
        }

        deleted.insert(id as u32);
        drop(deleted); // Release lock before invalidating cache

        // Invalidate caches
        self.invalidate_active_cache();
        self.invalidate_query_cache();

        // Remove from text index
        if let Some(ref text_index) = self.text_index {
            let _ = text_index.remove_document(id);
        }

        Ok(true)
    }

    /// Delete multiple vectors by ID.
    /// Writes to WAL first for durability.
    /// Returns the number of vectors actually deleted.
    pub fn delete_batch(&self, ids: &[u64]) -> Result<usize> {
        let total = self.vector_store.len();
        let mut deleted = self.deleted_ids.write();

        // Find IDs that can actually be deleted
        let valid_ids: Vec<u64> = ids
            .iter()
            .copied()
            .filter(|&id| (id as usize) < total && !deleted.contains(id as u32))
            .collect();

        if valid_ids.is_empty() {
            return Ok(0);
        }

        // Write to WAL first
        if let Some(ref wal) = self.wal {
            wal.append(WalOperation::BatchDelete {
                ids: valid_ids.clone(),
            })?;
        }

        // Apply deletes
        for id in &valid_ids {
            deleted.insert(*id as u32);
        }
        drop(deleted); // Release lock before invalidating cache

        // Invalidate caches
        self.invalidate_active_cache();
        self.invalidate_query_cache();

        Ok(valid_ids.len())
    }

    /// Get the deleted IDs bitmap for search filtering.
    pub fn deleted_bitmap(&self) -> RoaringBitmap {
        self.deleted_ids.read().clone()
    }

    /// Get the active (non-deleted) vector IDs bitmap.
    /// Uses caching to avoid O(n) computation on every search.
    ///
    /// Returns None if there are no deleted vectors (no filtering needed).
    pub fn active_bitmap(&self) -> Option<RoaringBitmap> {
        let deleted = self.deleted_ids.read();
        if deleted.is_empty() {
            return None; // No filtering needed
        }

        let total = self.vector_store.len();
        let deleted_count = deleted.len();

        // Check if cache is still valid
        {
            let cache = self.cached_active_bitmap.read();
            if let Some(ref cached) = *cache {
                if cached.vector_count == total && cached.deleted_count == deleted_count {
                    return Some(cached.bitmap.clone());
                }
            }
        }

        // Cache miss or invalid - recompute
        // Drop the deleted read lock before acquiring write lock
        drop(deleted);

        let mut cache = self.cached_active_bitmap.write();

        // Double-check after acquiring write lock
        let deleted = self.deleted_ids.read();
        let total = self.vector_store.len();
        let deleted_count = deleted.len();

        if let Some(ref cached) = *cache {
            if cached.vector_count == total && cached.deleted_count == deleted_count {
                return Some(cached.bitmap.clone());
            }
        }

        // Compute active bitmap
        let mut universe = RoaringBitmap::new();
        universe.insert_range(0..total as u32);
        let active = &universe - &*deleted;

        // Store in cache
        *cache = Some(CachedActiveBitmap {
            bitmap: active.clone(),
            vector_count: total,
            deleted_count,
        });

        Some(active)
    }

    /// Invalidate the active bitmap cache.
    /// Called when vectors are inserted or deleted.
    fn invalidate_active_cache(&self) {
        let mut cache = self.cached_active_bitmap.write();
        *cache = None;
    }

    /// Invalidate query result cache.
    /// Called when vectors are inserted or deleted.
    fn invalidate_query_cache(&self) {
        if let Some(ref cache) = self.query_cache {
            cache.invalidate_collection(&self.config.name);
        }
    }

    /// Invalidate filter bitmap cache for a specific field.
    /// Called when metadata for that field is updated.
    pub fn invalidate_filter_cache_field(&self, field: &str) {
        if let Some(ref cache) = self.filter_cache {
            cache.invalidate_field(field);
        }
    }

    /// Get query cache reference for search operations.
    pub fn query_cache(&self) -> Option<&Arc<QueryCache>> {
        self.query_cache.as_ref()
    }

    /// Get filter bitmap cache reference.
    pub fn filter_cache(&self) -> Option<&Arc<FilterBitmapCache>> {
        self.filter_cache.as_ref()
    }

    /// Get query cache statistics.
    pub fn query_cache_stats(&self) -> Option<crate::cache::QueryCacheStats> {
        self.query_cache.as_ref().map(|c| c.stats())
    }

    /// Get filter cache statistics.
    pub fn filter_cache_stats(&self) -> Option<crate::cache::FilterCacheStats> {
        self.filter_cache.as_ref().map(|c| c.stats())
    }

    /// Check if adaptive indexing is enabled.
    pub fn is_adaptive_enabled(&self) -> bool {
        self.adaptive_indexer.is_some()
    }

    /// Get adaptive index statistics.
    pub fn adaptive_index_stats(&self) -> Option<AdaptiveIndexStats> {
        self.adaptive_indexer.as_ref().map(|ai| ai.stats())
    }

    /// Get current index type (BruteForce or Hnsw).
    pub fn current_index_type(&self) -> Option<IndexType> {
        self.adaptive_indexer.as_ref().map(|ai| ai.current_type())
    }

    /// Search using adaptive indexer if available, otherwise use HNSW directly.
    pub fn adaptive_search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&RoaringBitmap>,
    ) -> Result<Vec<(u64, f32)>> {
        if let Some(ref adaptive) = self.adaptive_indexer {
            adaptive.search(query, k, filter)
        } else {
            self.indexer.search(query, k, filter)
        }
    }

    /// Get query planner reference.
    pub fn query_planner(&self) -> &QueryPlanner {
        &self.query_planner
    }

    /// Plan filters for optimal execution order.
    /// Returns filters sorted by selectivity (most selective first).
    pub fn plan_filters(&self, filters: Vec<(&str, &str)>) -> FilterPlan {
        let filter_orders: Vec<FilterOrder> = filters
            .into_iter()
            .map(|(field, value)| FilterOrder::new(field, value))
            .collect();
        self.query_planner.plan_filters(filter_orders)
    }

    /// Estimate selectivity for a filter condition.
    pub fn estimate_filter_selectivity(&self, field: &str, value: &str) -> f32 {
        self.query_planner.estimate_selectivity(field, value)
    }

    /// Flush all data to disk with atomic writes.
    ///
    /// Uses write-to-temp-then-rename pattern for atomic deleted bitmap updates.
    /// This prevents data corruption if the process crashes mid-write.
    pub fn flush(&self) -> Result<()> {
        self.vector_store.flush()?;

        // Save deleted IDs atomically using temp file + rename
        let deleted_path = self.base_path.join("deleted.bin");
        let deleted_tmp_path = self.base_path.join("deleted.bin.tmp");
        let deleted = self.deleted_ids.read();

        if !deleted.is_empty() {
            let mut bytes = Vec::new();
            deleted.serialize_into(&mut bytes)?;

            // Write to temp file first
            fs::write(&deleted_tmp_path, &bytes)?;

            // Atomic rename (POSIX guarantees atomicity for rename on same filesystem)
            fs::rename(&deleted_tmp_path, &deleted_path)?;
        } else if deleted_path.exists() {
            // Remove file if no deleted IDs
            let _ = fs::remove_file(&deleted_path);
        }

        // Cleanup temp file if it exists (from previous failed attempt)
        if deleted_tmp_path.exists() {
            let _ = fs::remove_file(&deleted_tmp_path);
        }

        // Save text index
        if let Some(ref text_index) = self.text_index {
            text_index.save()?;
        }

        Ok(())
    }

    /// Save index to disk.
    pub fn save_index(&self) -> Result<()> {
        let index_path = self.base_path.join("index.hnsw");
        self.indexer.save(&index_path)?;
        Ok(())
    }

    /// Create a checkpoint (save index + WAL marker).
    /// This allows truncating old WAL entries.
    pub fn checkpoint(&self) -> Result<u64> {
        // Save index first
        self.save_index()?;

        // Flush vector store
        self.flush()?;

        // Write checkpoint to WAL
        if let Some(ref wal) = self.wal {
            let seq = wal.checkpoint(self.vector_store.len())?;
            println!(
                "Checkpoint created for '{}' at sequence {}",
                self.config.name, seq
            );
            return Ok(seq);
        }

        Ok(0)
    }

    /// Check if checkpoint is needed and perform it
    fn maybe_checkpoint(&self) -> Result<()> {
        if let Some(ref wal) = self.wal {
            if wal.needs_checkpoint() {
                self.checkpoint()?;
            }
        }
        Ok(())
    }

    /// Truncate WAL after checkpoint.
    /// Removes WAL entries before the last checkpoint.
    pub fn truncate_wal(&self) -> Result<()> {
        if let Some(ref wal) = self.wal {
            wal.truncate()?;
            println!("WAL truncated for collection '{}'", self.config.name);
        }
        Ok(())
    }

    /// Get WAL statistics.
    pub fn wal_stats(&self) -> Option<WalStats> {
        self.wal.as_ref().map(|w| w.stats())
    }

    /// Sync WAL to disk.
    pub fn sync_wal(&self) -> Result<()> {
        if let Some(ref wal) = self.wal {
            wal.sync()?;
        }
        Ok(())
    }

    /// Check if text search is enabled.
    pub fn has_text_index(&self) -> bool {
        self.text_index.is_some()
    }

    /// Perform BM25 text search.
    /// Returns vector IDs sorted by BM25 relevance score.
    pub fn text_search(&self, query: &str, k: usize) -> Result<Vec<TextSearchResult>> {
        let text_index = self.text_index.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Text search not enabled. Configure text_fields in collection."))?;

        let results = text_index.search(query, k);

        // Filter out deleted vectors
        let deleted = self.deleted_ids.read();
        let filtered: Vec<TextSearchResult> = results
            .into_iter()
            .filter(|r| !deleted.contains(r.id as u32))
            .collect();

        Ok(filtered)
    }

    /// Perform hybrid search combining vector similarity and text relevance.
    ///
    /// # Arguments
    /// * `vector` - Query vector for similarity search
    /// * `query` - Text query for BM25 search
    /// * `k` - Number of results to return
    /// * `alpha` - Weight for vector score (0.0 = text only, 1.0 = vector only, 0.5 = equal)
    /// * `use_rrf` - Use Reciprocal Rank Fusion instead of weighted combination
    pub fn hybrid_search(
        &self,
        vector: &[f32],
        query: &str,
        k: usize,
        alpha: f32,
        use_rrf: bool,
    ) -> Result<Vec<HybridSearchResult>> {
        // Get vector search results (exclude deleted vectors)
        let deleted = self.deleted_bitmap();
        // For search, we need to invert the deleted bitmap to get allowed IDs
        let total = self.vector_store.len() as u64;
        let mut allowed = roaring::RoaringBitmap::new();
        allowed.insert_range(0..total as u32);
        let allowed = &allowed - &deleted;
        let vector_results = self.indexer.search(vector, k * 2, Some(&allowed))?;

        // Get text search results
        let text_results = if let Some(ref text_index) = self.text_index {
            text_index.search(query, k * 2)
        } else {
            Vec::new()
        };

        // Combine results
        let combined = if use_rrf {
            rrf_combine(&vector_results, &text_results, 60.0, k)
        } else {
            hybrid_combine(&vector_results, &text_results, alpha, k)
        };

        Ok(combined)
    }

    /// Perform hybrid search with additional metadata filtering.
    pub fn hybrid_search_filtered(
        &self,
        vector: &[f32],
        query: &str,
        k: usize,
        alpha: f32,
        use_rrf: bool,
        filter_bitmap: Option<&RoaringBitmap>,
    ) -> Result<Vec<HybridSearchResult>> {
        // Combine deleted IDs with filter
        let deleted = self.deleted_bitmap();
        let effective_filter = match filter_bitmap {
            Some(filter) => {
                let mut combined = deleted.clone();
                // Invert filter to get excluded IDs
                for id in 0..self.vector_store.len() as u32 {
                    if !filter.contains(id) {
                        combined.insert(id);
                    }
                }
                combined
            }
            None => deleted,
        };

        // Get vector search results
        // Invert the effective_filter to get allowed IDs (filter contains excluded IDs)
        let total = self.vector_store.len() as u64;
        let mut universe = roaring::RoaringBitmap::new();
        universe.insert_range(0..total as u32);
        let allowed_for_search = &universe - &effective_filter;
        let vector_results = self.indexer.search(vector, k * 2, Some(&allowed_for_search))?;

        // Get text search results and filter
        let text_results = if let Some(ref text_index) = self.text_index {
            let results = text_index.search(query, k * 2);
            results
                .into_iter()
                .filter(|r| !effective_filter.contains(r.id as u32))
                .collect()
        } else {
            Vec::new()
        };

        // Combine results
        let combined = if use_rrf {
            rrf_combine(&vector_results, &text_results, 60.0, k)
        } else {
            hybrid_combine(&vector_results, &text_results, alpha, k)
        };

        Ok(combined)
    }

    /// Get text index statistics.
    pub fn text_index_stats(&self) -> Option<crate::text::TextIndexStats> {
        self.text_index.as_ref().map(|idx| idx.stats())
    }

    /// Get collection info.
    pub fn info(&self) -> CollectionInfo {
        let wal_stats = self.wal_stats();
        let text_stats = self.text_index_stats();
        CollectionInfo {
            name: self.config.name.clone(),
            dim: self.config.dim,
            vector_count: self.active_count(),
            deleted_count: self.deleted_count(),
            distance_metric: self.config.distance.metric.clone(),
            quantization_enabled: self.config.quantization.enabled,
            indexed_fields: self.config.payload.indexed_fields.clone(),
            numeric_fields: self.config.payload.numeric_fields.clone(),
            text_fields: self.config.payload.text_fields.clone(),
            text_indexed_count: text_stats.map(|s| s.document_count),
            wal_enabled: self.config.wal.enabled,
            wal_sequence: wal_stats.as_ref().map(|s| s.sequence),
            wal_entries_since_checkpoint: wal_stats.map(|s| s.entries_since_checkpoint),
        }
    }

    /// Get base path for this collection.
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Compact the collection by removing deleted vectors.
    /// This creates new storage with only active vectors and rebuilds the index.
    /// Returns CompactionResult with statistics.
    ///
    /// WARNING: This operation is destructive and cannot be undone.
    /// It is recommended to create a snapshot before compacting.
    pub fn compact(&self) -> Result<CompactionResult> {
        let deleted_bitmap = self.deleted_ids.read().clone();
        let deleted_count = deleted_bitmap.len() as usize;
        let total_count = self.vector_store.len();
        let active_count = total_count.saturating_sub(deleted_count);

        if deleted_count == 0 {
            return Ok(CompactionResult {
                vectors_before: total_count,
                vectors_after: total_count,
                vectors_removed: 0,
                bytes_reclaimed: 0,
                id_mapping: HashMap::new(),
            });
        }

        // Create temporary directory for new files
        let temp_dir = self.base_path.join(".compact_temp");
        if temp_dir.exists() {
            fs::remove_dir_all(&temp_dir)?;
        }
        fs::create_dir_all(&temp_dir)?;

        // Create new vector store
        let vector_path = temp_dir.join("vectors");
        let dim = self.config.dim;
        let new_vector_store: Box<dyn VectorStore> = if self.config.quantization.enabled {
            Box::new(QuantizedMemmapVectorStore::new(
                vector_path.to_str().unwrap(),
                dim,
                self.config.quantization.keep_originals,
            )?)
        } else {
            let path = format!("{}.bin", vector_path.to_str().unwrap());
            Box::new(MemmapVectorStore::new(&path, dim)?)
        };

        // Create new metadata store
        let meta_path = temp_dir.join("meta.sled");
        let new_metadata_store: Box<dyn MetadataStore> = if self.config.payload.index_enabled {
            Box::new(IndexedSledMetadataStore::with_numeric_fields(
                meta_path.to_str().unwrap(),
                self.config.payload.indexed_fields.iter().map(|s| s.as_str()),
                self.config.payload.numeric_fields.iter().map(|s| s.as_str()),
            )?)
        } else {
            Box::new(SledMetadataStore::new(meta_path.to_str().unwrap())?)
        };

        // OPTIMIZATION: Batch process vectors for compaction (3-5x faster)
        // Step 1: Collect all active vectors first
        let mut active_vectors: Vec<(u64, Vec<f32>)> = Vec::with_capacity(active_count);
        for old_id in 0..total_count as u64 {
            if deleted_bitmap.contains(old_id as u32) {
                continue; // Skip deleted vectors
            }
            let vector = self.vector_store.get(old_id)?;
            active_vectors.push((old_id, vector));
        }

        // Step 2: Batch insert all vectors at once
        let vectors_only: Vec<Vec<f32>> = active_vectors.iter().map(|(_, v)| v.clone()).collect();
        let start_id = new_vector_store.insert_batch(&vectors_only)?;

        // Step 3: Build ID mapping and prepare index data
        let mut id_mapping: HashMap<u64, u64> = HashMap::with_capacity(active_count);
        let mut vectors_for_index: Vec<(u64, Vec<f32>)> = Vec::with_capacity(active_count);

        for (i, (old_id, vector)) in active_vectors.into_iter().enumerate() {
            let new_id = start_id + i as u64;
            id_mapping.insert(old_id, new_id);
            vectors_for_index.push((new_id, vector));
        }

        // Step 4: Batch copy metadata
        let keys_to_copy = ["category", "type", "status", "name", "idx", "tag",
                            "price", "rating", "timestamp", "priority"];

        // Collect all metadata entries for batch insert
        let mut all_metadata_entries: Vec<MetadataEntry> = Vec::new();

        for (&old_id, &new_id) in &id_mapping {
            // Copy common keys
            for key in keys_to_copy.iter() {
                if let Ok(Some(value)) = self.metadata_store.get(old_id, key) {
                    all_metadata_entries.push(MetadataEntry {
                        id: new_id,
                        key: key.to_string(),
                        value,
                    });
                }
            }
            // Copy indexed string fields
            for key in &self.config.payload.indexed_fields {
                if !keys_to_copy.contains(&key.as_str()) {
                    if let Ok(Some(value)) = self.metadata_store.get(old_id, key) {
                        all_metadata_entries.push(MetadataEntry {
                            id: new_id,
                            key: key.clone(),
                            value,
                        });
                    }
                }
            }
            // Copy numeric fields
            for key in &self.config.payload.numeric_fields {
                if !keys_to_copy.contains(&key.as_str()) {
                    if let Ok(Some(value)) = self.metadata_store.get(old_id, key) {
                        all_metadata_entries.push(MetadataEntry {
                            id: new_id,
                            key: key.clone(),
                            value,
                        });
                    }
                }
            }
        }

        // Batch insert all metadata entries
        if !all_metadata_entries.is_empty() {
            new_metadata_store.insert_batch(&all_metadata_entries)?;
        }

        // Flush new stores
        new_vector_store.flush()?;

        // Create new index with compacted vectors (preserve distance metric)
        let new_indexer = HnswIndexer::with_metric(
            dim,
            std::cmp::max(active_count + 10000, self.config.hnsw.max_elements),
            self.config.hnsw.m,
            self.config.hnsw.ef_construction,
            self.config.distance_metric(),
        );

        // Batch insert all vectors into new index
        if !vectors_for_index.is_empty() {
            new_indexer.insert_batch(&vectors_for_index)?;
        }

        // Save new index
        let index_path = temp_dir.join("index.hnsw");
        new_indexer.save(&index_path)?;

        // Calculate bytes reclaimed (approximate)
        let old_size = dir_size(&self.base_path).unwrap_or(0);
        let new_size = dir_size(&temp_dir).unwrap_or(0);
        let bytes_reclaimed = old_size.saturating_sub(new_size);

        // Drop new stores to release file handles
        drop(new_vector_store);
        drop(new_metadata_store);

        // Backup old files
        let backup_dir = self.base_path.join(".compact_backup");
        if backup_dir.exists() {
            fs::remove_dir_all(&backup_dir)?;
        }

        // Move old files to backup (except temp dirs)
        fs::create_dir_all(&backup_dir)?;
        for entry in fs::read_dir(&self.base_path)? {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_str().unwrap_or("");
            if name_str.starts_with('.') {
                continue; // Skip temp/backup dirs
            }
            let src = entry.path();
            let dst = backup_dir.join(&name);
            fs::rename(&src, &dst)?;
        }

        // Move new files to main directory
        for entry in fs::read_dir(&temp_dir)? {
            let entry = entry?;
            let src = entry.path();
            let dst = self.base_path.join(entry.file_name());
            if src.is_dir() {
                copy_dir_recursive(&src, &dst)?;
            } else {
                fs::copy(&src, &dst)?;
            }
        }

        // Copy config back
        let config_src = backup_dir.join("config.json");
        let config_dst = self.base_path.join("config.json");
        if config_src.exists() {
            fs::copy(&config_src, &config_dst)?;
        }

        // Clean up temp and backup directories
        fs::remove_dir_all(&temp_dir)?;
        fs::remove_dir_all(&backup_dir)?;

        // Clear deleted IDs
        self.deleted_ids.write().clear();

        // Remove deleted.bin file
        let deleted_path = self.base_path.join("deleted.bin");
        if deleted_path.exists() {
            fs::remove_file(deleted_path)?;
        }

        // Remove WAL file after compaction (clean slate)
        let wal_path = self.base_path.join("wal.log");
        if wal_path.exists() {
            fs::remove_file(&wal_path)?;
        }

        Ok(CompactionResult {
            vectors_before: total_count,
            vectors_after: active_count,
            vectors_removed: deleted_count,
            bytes_reclaimed,
            id_mapping,
        })
    }

    /// Create a snapshot of this collection.
    /// Copies all files to the snapshot directory.
    pub fn create_snapshot(&self, snapshot_path: &Path) -> Result<SnapshotInfo> {
        // Ensure data is flushed before snapshot
        self.flush()?;
        self.save_index()?;

        // Create snapshot directory
        fs::create_dir_all(snapshot_path)?;

        // Copy all files recursively
        copy_dir_recursive(&self.base_path, snapshot_path)?;

        // Create snapshot metadata
        let info = SnapshotInfo {
            name: snapshot_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            collection: self.config.name.clone(),
            created_at: Utc::now(),
            vector_count: self.len(),
            size_bytes: dir_size(snapshot_path)?,
        };

        // Save snapshot metadata
        let meta_path = snapshot_path.join("snapshot.json");
        let meta_content = serde_json::to_string_pretty(&info)?;
        fs::write(meta_path, meta_content)?;

        Ok(info)
    }
}

/// Recursively copy directory contents.
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    if !dst.exists() {
        fs::create_dir_all(dst)?;
    }

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }

    Ok(())
}

/// Calculate directory size in bytes.
fn dir_size(path: &Path) -> Result<u64> {
    let mut size = 0;
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                size += dir_size(&path)?;
            } else {
                size += entry.metadata()?.len();
            }
        }
    }
    Ok(size)
}

/// Collection information for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub dim: usize,
    pub vector_count: usize,
    pub deleted_count: usize,
    pub distance_metric: String,
    pub quantization_enabled: bool,
    pub indexed_fields: Vec<String>,
    pub numeric_fields: Vec<String>,
    pub text_fields: Vec<String>,
    pub text_indexed_count: Option<usize>,
    pub wal_enabled: bool,
    pub wal_sequence: Option<u64>,
    pub wal_entries_since_checkpoint: Option<u64>,
}

/// Snapshot information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotInfo {
    pub name: String,
    pub collection: String,
    pub created_at: DateTime<Utc>,
    pub vector_count: usize,
    pub size_bytes: u64,
}

/// Compaction result with statistics.
#[derive(Debug, Clone, Serialize)]
pub struct CompactionResult {
    /// Number of vectors before compaction
    pub vectors_before: usize,
    /// Number of vectors after compaction
    pub vectors_after: usize,
    /// Number of deleted vectors removed
    pub vectors_removed: usize,
    /// Approximate bytes reclaimed
    pub bytes_reclaimed: u64,
    /// Mapping from old IDs to new IDs
    #[serde(skip)]  // Don't serialize - can be very large
    pub id_mapping: HashMap<u64, u64>,
}

/// Manager for multiple collections.
pub struct CollectionManager {
    /// Base data directory
    base_dir: PathBuf,
    /// Loaded collections
    collections: RwLock<HashMap<String, Arc<Collection>>>,
    /// Default collection name
    default_collection: String,
}

impl CollectionManager {
    /// Create a new collection manager.
    pub fn new(base_dir: &Path) -> Result<Self> {
        fs::create_dir_all(base_dir)?;

        let manager = Self {
            base_dir: base_dir.to_path_buf(),
            collections: RwLock::new(HashMap::new()),
            default_collection: "default".to_string(),
        };

        // Load existing collections
        manager.load_existing_collections()?;

        Ok(manager)
    }

    /// Load all existing collections from disk.
    fn load_existing_collections(&self) -> Result<()> {
        let entries = fs::read_dir(&self.base_dir)?;

        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                let config_path = path.join("config.json");
                if config_path.exists() {
                    let name = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    match Collection::open(&path, CollectionConfig::new(&name, 128)) {
                        Ok(collection) => {
                            let mut collections = self.collections.write();
                            println!("Loaded collection: {} ({} vectors)", name, collection.len());
                            collections.insert(name, Arc::new(collection));
                        }
                        Err(e) => {
                            eprintln!("Failed to load collection {}: {:?}", name, e);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Create a new collection.
    pub fn create(&self, config: CollectionConfig) -> Result<Arc<Collection>> {
        let name = config.name.clone();

        // Check if already exists
        {
            let collections = self.collections.read();
            if collections.contains_key(&name) {
                return Err(anyhow::anyhow!("Collection '{}' already exists", name));
            }
        }

        // Create collection directory and open
        let path = self.base_dir.join(&name);
        let collection = Arc::new(Collection::open(&path, config)?);

        // Add to manager
        {
            let mut collections = self.collections.write();
            collections.insert(name.clone(), collection.clone());
        }

        println!("Created collection: {}", name);
        Ok(collection)
    }

    /// Get a collection by name.
    pub fn get(&self, name: &str) -> Option<Arc<Collection>> {
        let collections = self.collections.read();
        collections.get(name).cloned()
    }

    /// Get or create default collection.
    pub fn get_or_create_default(&self, dim: usize) -> Result<Arc<Collection>> {
        let name = &self.default_collection;

        // Try to get existing
        if let Some(collection) = self.get(name) {
            return Ok(collection);
        }

        // Create default
        let config = CollectionConfig::new(name, dim);
        self.create(config)
    }

    /// Delete a collection.
    pub fn delete(&self, name: &str) -> Result<()> {
        // Remove from manager
        let collection = {
            let mut collections = self.collections.write();
            collections.remove(name)
        };

        if collection.is_none() {
            return Err(anyhow::anyhow!("Collection '{}' not found", name));
        }

        // Delete from disk
        let path = self.base_dir.join(name);
        if path.exists() {
            fs::remove_dir_all(&path)?;
        }

        println!("Deleted collection: {}", name);
        Ok(())
    }

    /// List all collections.
    pub fn list(&self) -> Vec<CollectionInfo> {
        let collections = self.collections.read();
        collections
            .values()
            .map(|c| c.info())
            .collect()
    }

    /// Get collection names.
    pub fn names(&self) -> Vec<String> {
        let collections = self.collections.read();
        collections.keys().cloned().collect()
    }

    /// Flush all collections.
    pub fn flush_all(&self) -> Result<()> {
        let collections = self.collections.read();
        for (name, collection) in collections.iter() {
            if let Err(e) = collection.flush() {
                eprintln!("Failed to flush collection {}: {:?}", name, e);
            }
        }
        Ok(())
    }

    /// Save all indexes.
    pub fn save_all_indexes(&self) -> Result<()> {
        let collections = self.collections.read();
        for (name, collection) in collections.iter() {
            if let Err(e) = collection.save_index() {
                eprintln!("Failed to save index for {}: {:?}", name, e);
            } else {
                println!("Saved index for collection: {}", name);
            }
        }
        Ok(())
    }

    /// Get snapshots directory.
    pub fn snapshots_dir(&self) -> PathBuf {
        self.base_dir.join("_snapshots")
    }

    /// Create a snapshot of a collection.
    /// Returns snapshot info with timestamp-based name.
    pub fn create_snapshot(&self, collection_name: &str) -> Result<SnapshotInfo> {
        let collection = self.get(collection_name)
            .ok_or_else(|| anyhow::anyhow!("Collection '{}' not found", collection_name))?;

        // Create snapshot directory with timestamp
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let snapshot_name = format!("{}_{}", collection_name, timestamp);
        let snapshot_dir = self.snapshots_dir().join(&snapshot_name);

        let info = collection.create_snapshot(&snapshot_dir)?;
        println!("Created snapshot: {} ({} bytes)", snapshot_name, info.size_bytes);

        Ok(info)
    }

    /// List all snapshots for a collection.
    pub fn list_snapshots(&self, collection_name: &str) -> Result<Vec<SnapshotInfo>> {
        let snapshots_dir = self.snapshots_dir();
        if !snapshots_dir.exists() {
            return Ok(Vec::new());
        }

        let prefix = format!("{}_", collection_name);
        let mut snapshots = Vec::new();

        for entry in fs::read_dir(&snapshots_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");

                if name.starts_with(&prefix) {
                    // Try to load snapshot metadata
                    let meta_path = path.join("snapshot.json");
                    if meta_path.exists() {
                        if let Ok(content) = fs::read_to_string(&meta_path) {
                            if let Ok(info) = serde_json::from_str::<SnapshotInfo>(&content) {
                                snapshots.push(info);
                            }
                        }
                    }
                }
            }
        }

        // Sort by creation time (newest first)
        snapshots.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(snapshots)
    }

    /// Restore a collection from a snapshot.
    /// Creates a new collection with the given name from snapshot data.
    pub fn restore_snapshot(&self, snapshot_name: &str, new_collection_name: Option<&str>) -> Result<Arc<Collection>> {
        let snapshot_path = self.snapshots_dir().join(snapshot_name);
        if !snapshot_path.exists() {
            return Err(anyhow::anyhow!("Snapshot '{}' not found", snapshot_name));
        }

        // Load snapshot metadata
        let meta_path = snapshot_path.join("snapshot.json");
        let meta_content = fs::read_to_string(&meta_path)
            .context("Failed to read snapshot metadata")?;
        let snapshot_info: SnapshotInfo = serde_json::from_str(&meta_content)
            .context("Failed to parse snapshot metadata")?;

        // Determine target collection name
        let target_name = new_collection_name
            .map(|s| s.to_string())
            .unwrap_or_else(|| snapshot_info.collection.clone());

        // Check if collection already exists
        if self.get(&target_name).is_some() {
            return Err(anyhow::anyhow!(
                "Collection '{}' already exists. Delete it first or use a different name.",
                target_name
            ));
        }

        // Copy snapshot to collection directory
        let collection_path = self.base_dir.join(&target_name);
        copy_dir_recursive(&snapshot_path, &collection_path)?;

        // Update config with new name if different
        let config_path = collection_path.join("config.json");
        if let Ok(content) = fs::read_to_string(&config_path) {
            if let Ok(mut config) = serde_json::from_str::<CollectionConfig>(&content) {
                config.name = target_name.clone();
                let new_content = serde_json::to_string_pretty(&config)?;
                fs::write(&config_path, new_content)?;
            }
        }

        // Remove snapshot metadata from restored collection
        let restored_meta_path = collection_path.join("snapshot.json");
        let _ = fs::remove_file(restored_meta_path);

        // Load the restored collection
        let config = CollectionConfig::new(&target_name, 128);
        let collection = Arc::new(Collection::open(&collection_path, config)?);

        // Add to manager
        {
            let mut collections = self.collections.write();
            collections.insert(target_name.clone(), collection.clone());
        }

        println!("Restored collection: {} from snapshot: {}", target_name, snapshot_name);
        Ok(collection)
    }

    /// Delete a snapshot.
    pub fn delete_snapshot(&self, snapshot_name: &str) -> Result<()> {
        let snapshot_path = self.snapshots_dir().join(snapshot_name);
        if !snapshot_path.exists() {
            return Err(anyhow::anyhow!("Snapshot '{}' not found", snapshot_name));
        }

        fs::remove_dir_all(&snapshot_path)?;
        println!("Deleted snapshot: {}", snapshot_name);
        Ok(())
    }

    /// Compact a collection by removing deleted vectors.
    /// This operation is destructive and requires reloading the collection.
    ///
    /// Returns CompactionResult with statistics.
    /// Note: After compaction, all vector IDs will change. The result includes
    /// an id_mapping to translate old IDs to new IDs.
    pub fn compact(&self, collection_name: &str) -> Result<CompactionResult> {
        let collection = self.get(collection_name)
            .ok_or_else(|| anyhow::anyhow!("Collection '{}' not found", collection_name))?;

        // Check if compaction is needed
        if collection.deleted_count() == 0 {
            return Ok(CompactionResult {
                vectors_before: collection.len(),
                vectors_after: collection.len(),
                vectors_removed: 0,
                bytes_reclaimed: 0,
                id_mapping: HashMap::new(),
            });
        }

        // Perform compaction
        let result = collection.compact()?;

        // Reload the collection to pick up new files
        let path = self.base_dir.join(collection_name);
        let config = collection.config.clone();
        drop(collection);

        // Remove old collection from manager
        {
            let mut collections = self.collections.write();
            collections.remove(collection_name);
        }

        // Reload collection
        let new_collection = Arc::new(Collection::open(&path, config)?);
        {
            let mut collections = self.collections.write();
            collections.insert(collection_name.to_string(), new_collection);
        }

        println!(
            "Compacted collection {}: {} -> {} vectors ({} removed, {} bytes reclaimed)",
            collection_name,
            result.vectors_before,
            result.vectors_after,
            result.vectors_removed,
            result.bytes_reclaimed
        );

        Ok(result)
    }

    /// Compact all collections with deleted vectors.
    pub fn compact_all(&self) -> Result<Vec<(String, CompactionResult)>> {
        let names: Vec<String> = self.names();
        let mut results = Vec::new();

        for name in names {
            if let Some(collection) = self.get(&name) {
                if collection.deleted_count() > 0 {
                    match self.compact(&name) {
                        Ok(result) => results.push((name, result)),
                        Err(e) => eprintln!("Failed to compact {}: {:?}", name, e),
                    }
                }
            }
        }

        Ok(results)
    }

    /// Create checkpoints for all collections.
    pub fn checkpoint_all(&self) -> Result<()> {
        let collections = self.collections.read();
        for (name, collection) in collections.iter() {
            if let Err(e) = collection.checkpoint() {
                eprintln!("Failed to checkpoint {}: {:?}", name, e);
            } else {
                println!("Checkpointed collection: {}", name);
            }
        }
        Ok(())
    }

    /// Create checkpoint for a specific collection.
    pub fn checkpoint(&self, collection_name: &str) -> Result<u64> {
        let collection = self.get(collection_name)
            .ok_or_else(|| anyhow::anyhow!("Collection '{}' not found", collection_name))?;
        collection.checkpoint()
    }

    /// Truncate WAL for all collections.
    pub fn truncate_all_wals(&self) -> Result<()> {
        let collections = self.collections.read();
        for (name, collection) in collections.iter() {
            if let Err(e) = collection.truncate_wal() {
                eprintln!("Failed to truncate WAL for {}: {:?}", name, e);
            }
        }
        Ok(())
    }

    /// Get WAL stats for all collections.
    pub fn all_wal_stats(&self) -> Vec<(String, Option<WalStats>)> {
        let collections = self.collections.read();
        collections
            .iter()
            .map(|(name, collection)| (name.clone(), collection.wal_stats()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn test_dir(name: &str) -> std::path::PathBuf {
        let dir = env::temp_dir().join(format!("vectordb_test_{}", name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_collection_create_and_load() {
        let dir = test_dir("collection_create");
        let config = CollectionConfig::new("test", 64);

        // Create
        let collection = Collection::open(&dir, config.clone()).unwrap();
        assert_eq!(collection.name(), "test");
        assert_eq!(collection.config.dim, 64);
        assert_eq!(collection.len(), 0);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_collection_manager() {
        let dir = test_dir("collection_manager");
        let manager = CollectionManager::new(&dir).unwrap();

        // Create collections
        let config1 = CollectionConfig::new("products", 128);
        let config2 = CollectionConfig::new("users", 256);

        manager.create(config1).unwrap();
        manager.create(config2).unwrap();

        // List
        let list = manager.list();
        assert_eq!(list.len(), 2);

        // Get
        let products = manager.get("products").unwrap();
        assert_eq!(products.config.dim, 128);

        // Delete
        manager.delete("users").unwrap();
        assert!(manager.get("users").is_none());

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }
}
