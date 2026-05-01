//! Embedded (in-process) API for CoreVecDB.
//!
//! Provides `CoreVecDB` as a high-level entry point for using the vector database
//! without HTTP/gRPC overhead. All operations run in-process.
//!
//! # Example
//! ```rust,ignore
//! use vectordb::embedded::CoreVecDB;
//! use vectordb::collection::{CollectionConfig, SearchParams};
//!
//! let db = CoreVecDB::open("./my_data")?;
//!
//! let config = CollectionConfig::new("products", 128)
//!     .with_distance("cosine")
//!     .with_indexed_fields(vec!["category"], vec!["price"]);
//! db.create_collection(config)?;
//!
//! let col = db.collection("products").unwrap();
//! let id = col.insert(&[0.1; 128], &[("category", "electronics")])?;
//!
//! let results = col.search(
//!     SearchParams::new(vec![0.1; 128], 10)
//!         .with_filter("category", "electronics")
//!         .with_metadata()
//! )?;
//! ```

use anyhow::Result;
use std::path::Path;
use std::sync::Arc;

use crate::collection::{
    Collection, CollectionConfig, CollectionInfo, CollectionManager,
    SearchParams, SearchResult,
};
use crate::text::HybridSearchResult;

/// Embedded vector database — zero network overhead.
///
/// Wraps `CollectionManager` with an ergonomic API for in-process usage.
pub struct CoreVecDB {
    manager: Arc<CollectionManager>,
}

impl CoreVecDB {
    /// Open or create a database at the given directory.
    pub fn open(data_dir: impl AsRef<Path>) -> Result<Self> {
        let manager = CollectionManager::new(data_dir.as_ref())?;
        Ok(Self {
            manager: Arc::new(manager),
        })
    }

    /// Create a new collection.
    pub fn create_collection(&self, config: CollectionConfig) -> Result<()> {
        self.manager.create(config)?;
        Ok(())
    }

    /// Get a handle to an existing collection.
    pub fn collection(&self, name: &str) -> Option<CollectionHandle> {
        self.manager.get(name).map(|c| CollectionHandle { inner: c })
    }

    /// Get or create a default collection with the given dimension.
    pub fn default_collection(&self, dim: usize) -> Result<CollectionHandle> {
        let c = self.manager.get_or_create_default(dim)?;
        Ok(CollectionHandle { inner: c })
    }

    /// List all collections.
    pub fn list_collections(&self) -> Vec<CollectionInfo> {
        self.manager.list()
    }

    /// Drop a collection and delete its data.
    pub fn drop_collection(&self, name: &str) -> Result<()> {
        self.manager.delete(name)
    }

    /// Flush all collections to disk.
    pub fn flush(&self) -> Result<()> {
        self.manager.flush_all()
    }

    /// Get the underlying CollectionManager (for advanced usage).
    pub fn manager(&self) -> &Arc<CollectionManager> {
        &self.manager
    }
}

/// Handle to a single collection with ergonomic API.
pub struct CollectionHandle {
    inner: Arc<Collection>,
}

impl CollectionHandle {
    /// Insert a single vector with metadata.
    pub fn insert(&self, vector: &[f32], metadata: &[(&str, &str)]) -> Result<u64> {
        let meta: Vec<(String, String)> = metadata
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        self.inner.insert(vector, &meta)
    }

    /// Insert a batch of vectors with metadata.
    pub fn insert_batch(
        &self,
        vectors: &[Vec<f32>],
        metadata: &[Vec<(String, String)>],
    ) -> Result<u64> {
        self.inner.insert_batch(vectors, metadata)
    }

    /// Search with full filter support.
    pub fn search(&self, params: SearchParams) -> Result<Vec<SearchResult>> {
        self.inner.search(params)
    }

    /// Hybrid vector + text search.
    pub fn hybrid_search(
        &self,
        vector: &[f32],
        query: &str,
        k: usize,
        alpha: f32,
    ) -> Result<Vec<HybridSearchResult>> {
        self.inner.hybrid_search(vector, query, k, alpha, false)
    }

    /// Hybrid search with RRF (Reciprocal Rank Fusion) instead of linear combination.
    pub fn hybrid_search_rrf(
        &self,
        vector: &[f32],
        query: &str,
        k: usize,
    ) -> Result<Vec<HybridSearchResult>> {
        self.inner.hybrid_search(vector, query, k, 0.5, true)
    }

    /// Delete a vector by ID.
    pub fn delete(&self, id: u64) -> Result<bool> {
        self.inner.delete(id)
    }

    /// Delete multiple vectors by ID.
    pub fn delete_batch(&self, ids: &[u64]) -> Result<usize> {
        self.inner.delete_batch(ids)
    }

    /// Get total vector count (including deleted).
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if collection is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Get collection info.
    pub fn info(&self) -> CollectionInfo {
        self.inner.info()
    }

    /// Flush to disk.
    pub fn flush(&self) -> Result<()> {
        self.inner.flush()
    }

    /// Get the underlying Collection (for advanced usage).
    pub fn inner(&self) -> &Arc<Collection> {
        &self.inner
    }
}
