//! Multi-Collection support for VectorDB.
//!
//! Each collection is an independent vector space with its own:
//! - Vector storage (mmap or quantized)
//! - HNSW index
//! - Metadata store with payload indexing
//! - Configuration (dimension, index params, etc.)

use anyhow::{Result, Context};
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
}

impl Collection {
    /// Create or load a collection at the given path.
    pub fn open(base_path: &Path, config: CollectionConfig) -> Result<Self> {
        fs::create_dir_all(base_path)?;

        let config_path = base_path.join("config.json");
        let vector_path = base_path.join("vectors");
        let index_path = base_path.join("index.hnsw");
        let meta_path = base_path.join("meta.sled");

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

        Ok(Self {
            config,
            vector_store,
            metadata_store,
            indexer,
            base_path: base_path.to_path_buf(),
        })
    }

    /// Get collection name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get vector count.
    pub fn len(&self) -> usize {
        self.vector_store.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Flush all data to disk.
    pub fn flush(&self) -> Result<()> {
        self.vector_store.flush()?;
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
            vector_count: self.len(),
            distance_metric: self.config.distance.metric.clone(),
            quantization_enabled: self.config.quantization.enabled,
            indexed_fields: self.config.payload.indexed_fields.clone(),
            numeric_fields: self.config.payload.numeric_fields.clone(),
        }
    }
}

/// Collection information for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub dim: usize,
    pub vector_count: usize,
    pub distance_metric: String,
    pub quantization_enabled: bool,
    pub indexed_fields: Vec<String>,
    pub numeric_fields: Vec<String>,
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
