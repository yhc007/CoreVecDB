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

use crate::index::{HnswIndexer, DistanceMetric};
use crate::storage::{
    MemmapVectorStore, QuantizedMemmapVectorStore, SledMetadataStore,
    IndexedSledMetadataStore, VectorStore, MetadataStore,
};
use roaring::RoaringBitmap;

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
            },
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

    /// Get distance metric enum.
    pub fn distance_metric(&self) -> DistanceMetric {
        match self.distance.metric.to_lowercase().as_str() {
            "cosine" => DistanceMetric::Cosine,
            _ => DistanceMetric::Euclidean,
        }
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
    /// HNSW index
    pub indexer: Arc<HnswIndexer>,
    /// Base path for this collection
    base_path: PathBuf,
    /// Deleted vector IDs (soft delete tombstones)
    deleted_ids: RwLock<RoaringBitmap>,
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

        // Create or load index
        let graph_path = format!("{}.hnsw.graph", index_path.to_str().unwrap());
        let indexer = if Path::new(&graph_path).exists() {
            Arc::new(HnswIndexer::load(
                &index_path,
                dim,
                config.distance_metric(),
            )?)
        } else {
            Arc::new(HnswIndexer::new(
                dim,
                config.hnsw.max_elements,
                config.hnsw.m,
                config.hnsw.ef_construction,
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

        Ok(Self {
            config,
            vector_store,
            metadata_store,
            indexer,
            base_path: base_path.to_path_buf(),
            deleted_ids: RwLock::new(deleted_ids),
        })
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

    /// Delete a vector by ID (soft delete).
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

        deleted.insert(id as u32);
        Ok(true)
    }

    /// Delete multiple vectors by ID.
    /// Returns the number of vectors actually deleted.
    pub fn delete_batch(&self, ids: &[u64]) -> Result<usize> {
        let total = self.vector_store.len();
        let mut deleted = self.deleted_ids.write();
        let mut count = 0;

        for &id in ids {
            if (id as usize) < total && !deleted.contains(id as u32) {
                deleted.insert(id as u32);
                count += 1;
            }
        }

        Ok(count)
    }

    /// Get the deleted IDs bitmap for search filtering.
    pub fn deleted_bitmap(&self) -> RoaringBitmap {
        self.deleted_ids.read().clone()
    }

    /// Flush all data to disk.
    pub fn flush(&self) -> Result<()> {
        self.vector_store.flush()?;

        // Save deleted IDs
        let deleted_path = self.base_path.join("deleted.bin");
        let deleted = self.deleted_ids.read();
        if !deleted.is_empty() {
            let mut bytes = Vec::new();
            deleted.serialize_into(&mut bytes)?;
            fs::write(deleted_path, bytes)?;
        } else if deleted_path.exists() {
            // Remove file if no deleted IDs
            let _ = fs::remove_file(deleted_path);
        }

        Ok(())
    }

    /// Save index to disk.
    pub fn save_index(&self) -> Result<()> {
        let index_path = self.base_path.join("index.hnsw");
        self.indexer.save(&index_path)?;
        Ok(())
    }

    /// Get collection info.
    pub fn info(&self) -> CollectionInfo {
        CollectionInfo {
            name: self.config.name.clone(),
            dim: self.config.dim,
            vector_count: self.active_count(),
            deleted_count: self.deleted_count(),
            distance_metric: self.config.distance.metric.clone(),
            quantization_enabled: self.config.quantization.enabled,
            indexed_fields: self.config.payload.indexed_fields.clone(),
            numeric_fields: self.config.payload.numeric_fields.clone(),
        }
    }

    /// Get base path for this collection.
    pub fn base_path(&self) -> &Path {
        &self.base_path
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
