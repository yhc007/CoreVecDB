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
//! let col = db.collection("products")?;
//! let id = col.insert(&[0.1; 128], &[("category", "electronics")])?;
//!
//! let results = col.search(
//!     SearchParams::new(vec![0.1; 128], 10)
//!         .with_filter("category", "electronics")
//!         .with_metadata()
//! )?;
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::collection::{
    Collection, CollectionConfig, CollectionInfo, CollectionManager,
    SearchParams, SearchResult, SnapshotInfo, VectorEntry,
};
use crate::text::HybridSearchResult;
use crate::versioning::{
    ThreadSafeVersionedStore, VectorVersion, VersioningConfig, VersioningStats,
};
use crate::VecDbError;

use chrono::{DateTime, Utc};

/// Result type for embedded API.
pub type Result<T> = std::result::Result<T, VecDbError>;

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
    pub fn collection(&self, name: &str) -> Result<CollectionHandle> {
        self.manager
            .get(name)
            .map(|c| CollectionHandle { inner: c })
            .ok_or_else(|| VecDbError::NotFound(name.to_string()))
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
        self.manager.delete(name)?;
        Ok(())
    }

    /// Flush all collections to disk.
    pub fn flush(&self) -> Result<()> {
        self.manager.flush_all()?;
        Ok(())
    }

    // ─── Snapshots ──────────────────────────────────────────────────

    /// Create a point-in-time snapshot of a collection under `_snapshots/`.
    pub fn create_snapshot(&self, collection: &str) -> Result<SnapshotInfo> {
        self.manager.create_snapshot(collection).map_err(Into::into)
    }

    /// List all snapshots for a collection.
    pub fn list_snapshots(&self, collection: &str) -> Result<Vec<SnapshotInfo>> {
        self.manager.list_snapshots(collection).map_err(Into::into)
    }

    /// Restore a snapshot into a collection.
    ///
    /// With `new_name = None` the snapshot's original collection name is used
    /// (which must not already exist); pass `Some(name)` to restore under a new
    /// name (e.g. to clone). Returns a handle to the restored collection.
    pub fn restore_snapshot(
        &self,
        snapshot_name: &str,
        new_name: Option<&str>,
    ) -> Result<CollectionHandle> {
        let c = self.manager.restore_snapshot(snapshot_name, new_name)?;
        Ok(CollectionHandle { inner: c })
    }

    /// Delete a snapshot by name.
    pub fn delete_snapshot(&self, snapshot_name: &str) -> Result<()> {
        self.manager.delete_snapshot(snapshot_name).map_err(Into::into)
    }

    /// Get the underlying CollectionManager (for advanced usage).
    pub fn manager(&self) -> &Arc<CollectionManager> {
        &self.manager
    }
}

/// In-process vector versioning: full history, time-travel queries, and rollback.
///
/// This is an **independent, in-memory** store — it is not persisted to disk and
/// is decoupled from a [`Collection`]'s vectors (the same design the HTTP
/// `/versioned/*` endpoints use). Construct one per logical vector space and drive
/// it explicitly; nothing in `CoreVecDB` writes to it automatically.
///
/// # Example
/// ```rust,ignore
/// use vectordb::embedded::VersionedStore;
/// let vs = VersionedStore::new(128);
/// let v1 = vs.upsert(0, vec![1.0; 128], &[("note", "first")], None)?;
/// let v2 = vs.upsert(0, vec![2.0; 128], &[], Some("second edit"))?;
/// let history = vs.history(0);              // [v1, v2]
/// let restored = vs.rollback(0, v1.version_id)?;  // new version cloning v1
/// ```
#[derive(Clone)]
pub struct VersionedStore {
    inner: ThreadSafeVersionedStore,
}

impl VersionedStore {
    /// Create a versioned store for `dim`-dimensional vectors (default config).
    pub fn new(dim: usize) -> Self {
        Self { inner: ThreadSafeVersionedStore::new(dim) }
    }

    /// Create a versioned store with a custom retention/version-cap config.
    pub fn with_config(dim: usize, config: VersioningConfig) -> Self {
        Self { inner: ThreadSafeVersionedStore::with_config(dim, config) }
    }

    /// Insert or update a vector, creating a new version. Optional change note.
    pub fn upsert(
        &self,
        vector_id: u64,
        vector: Vec<f32>,
        metadata: &[(&str, &str)],
        description: Option<&str>,
    ) -> Result<VectorVersion> {
        let meta: HashMap<String, String> = metadata
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        self.inner
            .upsert_with_description(vector_id, vector, meta, description)
            .map_err(Into::into)
    }

    /// Record a deletion as a new version.
    pub fn delete(&self, vector_id: u64) -> Result<VectorVersion> {
        self.inner.delete(vector_id).map_err(Into::into)
    }

    /// Get the latest (non-deleted) version of a vector.
    pub fn get(&self, vector_id: u64) -> Option<VectorVersion> {
        self.inner.get(vector_id)
    }

    /// Get a specific version by its version ID.
    pub fn get_version(&self, vector_id: u64, version_id: u64) -> Option<VectorVersion> {
        self.inner.get_version(vector_id, version_id)
    }

    /// Get the version that was active at `timestamp`.
    pub fn get_at_timestamp(
        &self,
        vector_id: u64,
        timestamp: DateTime<Utc>,
    ) -> Option<VectorVersion> {
        self.inner.get_at_timestamp(vector_id, timestamp)
    }

    /// Full version history for a vector, oldest first.
    pub fn history(&self, vector_id: u64) -> Vec<VectorVersion> {
        self.inner.get_history(vector_id)
    }

    /// All active vectors as they existed at `timestamp`.
    pub fn snapshot_at(&self, timestamp: DateTime<Utc>) -> Vec<VectorVersion> {
        self.inner.snapshot_at(timestamp)
    }

    /// Roll a vector back to an earlier version (creates a new version).
    pub fn rollback(&self, vector_id: u64, to_version_id: u64) -> Result<VectorVersion> {
        self.inner.rollback(vector_id, to_version_id).map_err(Into::into)
    }

    /// Statistics across all versioned vectors.
    pub fn stats(&self) -> VersioningStats {
        self.inner.stats()
    }

    /// Compact old versions per the retention config. Returns versions removed.
    pub fn compact(&self) -> usize {
        self.inner.compact().versions_removed
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }
}

// ─── Candle embedding integration ───────────────────────────────────

/// CoreVecDB with an in-process text embedder.
///
/// Wraps `CoreVecDB` + `CandleTextEmbedder` to provide text-to-vector
/// operations directly (insert_text, search_text, etc.).
#[cfg(feature = "candle")]
pub struct EmbeddingDB {
    db: CoreVecDB,
    embedder: Arc<candle_embed::CandleTextEmbedder>,
}

#[cfg(feature = "candle")]
impl EmbeddingDB {
    /// Open a database with an in-process embedder.
    ///
    /// Downloads the model from HuggingFace Hub on first use.
    ///
    /// # Example
    /// ```rust,ignore
    /// let edb = EmbeddingDB::open("./data", "intfloat/e5-small-v2")?;
    /// let col = edb.collection("docs")?;
    /// col.insert_text("hello world", &edb, &[("source", "test")])?;
    /// ```
    pub fn open(
        data_dir: impl AsRef<Path>,
        model_id: &str,
    ) -> Result<Self> {
        let db = CoreVecDB::open(data_dir)?;
        let config = candle_embed::EmbedderConfig::for_model(model_id);
        let embedder = candle_embed::CandleTextEmbedder::new(config)
            .map_err(|e| VecDbError::Internal(anyhow::anyhow!("failed to load model: {}", e)))?;
        Ok(Self {
            db,
            embedder: Arc::new(embedder),
        })
    }

    /// Open with a custom embedder config.
    pub fn open_with_config(
        data_dir: impl AsRef<Path>,
        config: candle_embed::EmbedderConfig,
    ) -> Result<Self> {
        let db = CoreVecDB::open(data_dir)?;
        let embedder = candle_embed::CandleTextEmbedder::new(config)
            .map_err(|e| VecDbError::Internal(anyhow::anyhow!("failed to load model: {}", e)))?;
        Ok(Self {
            db,
            embedder: Arc::new(embedder),
        })
    }

    /// Get the underlying database.
    pub fn db(&self) -> &CoreVecDB {
        &self.db
    }

    /// Get the embedder.
    pub fn embedder(&self) -> &candle_embed::CandleTextEmbedder {
        &self.embedder
    }

    /// Embed a single text.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embedder
            .embed(text)
            .map_err(|e| VecDbError::Internal(anyhow::anyhow!("embedding failed: {}", e)))
    }

    /// Embed a batch of texts.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embedder
            .embed_batch(texts)
            .map_err(|e| VecDbError::Internal(anyhow::anyhow!("batch embedding failed: {}", e)))
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> usize {
        self.embedder.dimension()
    }

    /// Get a collection handle.
    pub fn collection(&self, name: &str) -> Result<CollectionHandle> {
        self.db.collection(name)
    }

    /// Create a collection with the embedder's dimension.
    ///
    /// Defaults to cosine distance: transformer sentence/text embeddings are
    /// direction-based, so cosine is almost always the right metric here (the
    /// generic `CoreVecDB::create_collection` still defaults to euclidean). Use
    /// `create_collection_with_config` to override.
    pub fn create_collection(&self, name: &str) -> Result<()> {
        let config = CollectionConfig::new(name, self.dimension()).with_distance("cosine");
        self.db.create_collection(config)
    }

    /// Create a collection with custom config.
    pub fn create_collection_with_config(&self, config: CollectionConfig) -> Result<()> {
        self.db.create_collection(config)
    }

    /// List all collections.
    pub fn list_collections(&self) -> Vec<CollectionInfo> {
        self.db.list_collections()
    }

    /// Drop a collection.
    pub fn drop_collection(&self, name: &str) -> Result<()> {
        self.db.drop_collection(name)
    }

    /// Flush all collections to disk.
    pub fn flush(&self) -> Result<()> {
        self.db.flush()
    }
}

// Text-based operations on CollectionHandle (requires EmbeddingDB)
#[cfg(feature = "candle")]
impl CollectionHandle {
    /// Insert a text document by embedding it first.
    ///
    /// The text is stored as metadata under the key `_text`.
    pub fn insert_text(
        &self,
        text: &str,
        edb: &EmbeddingDB,
        metadata: &[(&str, &str)],
    ) -> Result<u64> {
        let vector = edb.embed(text)?;
        let mut meta: Vec<(&str, &str)> = metadata.to_vec();
        meta.push(("_text", text));
        self.insert(&vector, &meta)
    }

    /// Search by text query (embeds the query, then vector search).
    pub fn search_text(
        &self,
        query: &str,
        edb: &EmbeddingDB,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let vector = edb.embed(query)?;
        let params = SearchParams::new(vector, k).with_metadata();
        self.search(params)
    }

    /// Hybrid text + vector search.
    pub fn hybrid_search_text(
        &self,
        query: &str,
        edb: &EmbeddingDB,
        k: usize,
        alpha: f32,
    ) -> Result<Vec<HybridSearchResult>> {
        let vector = edb.embed(query)?;
        self.hybrid_search(&vector, query, k, alpha)
    }

    /// Search by text with cross-encoder reranking.
    ///
    /// Two-stage retrieval pipeline:
    /// 1. Retrieve `top_k_initial` candidates via bi-encoder vector search
    /// 2. Rerank candidates with the cross-encoder and return the top `top_k`
    ///
    /// The cross-encoder scores (query, document) pairs directly, producing
    /// more accurate relevance scores than cosine similarity alone.
    pub fn search_and_rerank(
        &self,
        query: &str,
        edb: &EmbeddingDB,
        top_k: usize,
        top_k_initial: usize,
        reranker: &candle_embed::CandleReranker,
    ) -> Result<Vec<SearchResult>> {
        // Stage 1: fast bi-encoder retrieval
        let candidates = self.search_text(query, edb, top_k_initial)?;

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        // Extract text from _text metadata field
        let doc_texts: Vec<String> = candidates
            .iter()
            .map(|r| {
                r.metadata
                    .as_ref()
                    .and_then(|m| m.get("_text").cloned())
                    .unwrap_or_default()
            })
            .collect();

        let doc_refs: Vec<&str> = doc_texts.iter().map(|s| s.as_str()).collect();

        // Stage 2: cross-encoder reranking
        let ranked = reranker
            .rerank(query, &doc_refs, top_k)
            .map_err(|e| VecDbError::Internal(anyhow::anyhow!("reranking failed: {}", e)))?;

        // Build results in reranked order, using cross-encoder score
        let results: Vec<SearchResult> = ranked
            .into_iter()
            .map(|(idx, score)| {
                let orig = &candidates[idx];
                SearchResult {
                    id: orig.id,
                    score,
                    metadata: orig.metadata.clone(),
                }
            })
            .collect();

        Ok(results)
    }
}

/// Handle to a single collection with ergonomic API.
#[derive(Clone)]
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
        self.inner.insert(vector, &meta).map_err(Into::into)
    }

    /// Insert a batch of vectors with metadata.
    pub fn insert_batch(
        &self,
        vectors: &[Vec<f32>],
        metadata: &[Vec<(String, String)>],
    ) -> Result<u64> {
        self.inner.insert_batch(vectors, metadata).map_err(Into::into)
    }

    /// Search with full filter support.
    pub fn search(&self, params: SearchParams) -> Result<Vec<SearchResult>> {
        self.inner.search(params).map_err(Into::into)
    }

    /// Hybrid vector + text search.
    pub fn hybrid_search(
        &self,
        vector: &[f32],
        query: &str,
        k: usize,
        alpha: f32,
    ) -> Result<Vec<HybridSearchResult>> {
        self.inner.hybrid_search(vector, query, k, alpha, false).map_err(Into::into)
    }

    /// Hybrid search with RRF (Reciprocal Rank Fusion) instead of linear combination.
    pub fn hybrid_search_rrf(
        &self,
        vector: &[f32],
        query: &str,
        k: usize,
    ) -> Result<Vec<HybridSearchResult>> {
        self.inner.hybrid_search(vector, query, k, 0.5, true).map_err(Into::into)
    }

    /// Fetch a vector and its metadata by ID.
    ///
    /// Returns `None` if the ID is out of range or has been soft-deleted.
    pub fn get(&self, id: u64) -> Option<VectorEntry> {
        self.inner.get(id)
    }

    /// Replace a vector (and metadata). Storage is append-only, so this returns a
    /// NEW id; the old id is soft-deleted.
    ///
    /// Errors with [`VecDbError::NotFound`] if `old_id` is out of range or deleted.
    pub fn update(
        &self,
        old_id: u64,
        vector: &[f32],
        metadata: &[(&str, &str)],
    ) -> Result<u64> {
        let meta: Vec<(String, String)> = metadata
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        self.inner
            .update(old_id, vector, &meta)?
            .ok_or_else(|| VecDbError::NotFound(format!("vector {old_id}")))
    }

    /// Overwrite metadata keys for an existing vector (vector and ID unchanged).
    ///
    /// Errors with [`VecDbError::NotFound`] if `id` is out of range or deleted.
    pub fn update_metadata(&self, id: u64, metadata: &[(&str, &str)]) -> Result<()> {
        let meta: Vec<(String, String)> = metadata
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        if self.inner.update_metadata(id, &meta)? {
            Ok(())
        } else {
            Err(VecDbError::NotFound(format!("vector {id}")))
        }
    }

    /// Text-only BM25 search (requires the collection to have `text_fields`).
    pub fn text_search(&self, query: &str, k: usize) -> Result<Vec<crate::text::TextSearchResult>> {
        self.inner.text_search(query, k).map_err(Into::into)
    }

    /// Delete a vector by ID.
    pub fn delete(&self, id: u64) -> Result<bool> {
        self.inner.delete(id).map_err(Into::into)
    }

    /// Delete multiple vectors by ID.
    pub fn delete_batch(&self, ids: &[u64]) -> Result<usize> {
        self.inner.delete_batch(ids).map_err(Into::into)
    }

    /// Get the number of active (non-deleted) vectors.
    ///
    /// This is what most callers mean by "how many vectors are in here" — it
    /// excludes soft-deleted tombstones.
    pub fn len(&self) -> usize {
        self.inner.active_count()
    }

    /// Get the total number of vectors ever inserted, including soft-deleted ones.
    pub fn total_len(&self) -> usize {
        self.inner.len()
    }

    /// Get the number of soft-deleted vectors.
    pub fn deleted_len(&self) -> usize {
        self.inner.deleted_count()
    }

    /// Check if the collection has no active vectors.
    pub fn is_empty(&self) -> bool {
        self.inner.active_count() == 0
    }

    /// Get collection info.
    pub fn info(&self) -> CollectionInfo {
        self.inner.info()
    }

    /// Flush to disk.
    pub fn flush(&self) -> Result<()> {
        self.inner.flush().map_err(Into::into)
    }

    /// Get the underlying Collection (for advanced usage).
    pub fn inner(&self) -> &Arc<Collection> {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = env::temp_dir().join(format!("corevecdb_embed_{}", name));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn test_snapshot_create_list_delete() {
        let dir = tmp("snapshot");
        let db = CoreVecDB::open(&dir).unwrap();
        db.create_collection(CollectionConfig::new("docs", 4)).unwrap();
        let col = db.collection("docs").unwrap();
        col.insert(&[1.0; 4], &[("k", "v")]).unwrap();
        col.flush().unwrap();

        let snap = db.create_snapshot("docs").unwrap();
        assert_eq!(snap.collection, "docs");

        let list = db.list_snapshots("docs").unwrap();
        assert!(list.iter().any(|s| s.name == snap.name));

        // Restore under a new name → independent collection with the same data.
        let restored = db.restore_snapshot(&snap.name, Some("docs_copy")).unwrap();
        assert_eq!(restored.len(), 1);

        db.delete_snapshot(&snap.name).unwrap();
        assert!(db.list_snapshots("docs").unwrap().iter().all(|s| s.name != snap.name));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_versioning_history_and_rollback() {
        let vs = VersionedStore::new(4);
        let v1 = vs.upsert(0, vec![1.0; 4], &[("note", "first")], None).unwrap();
        let v2 = vs.upsert(0, vec![2.0; 4], &[], Some("second edit")).unwrap();
        assert_ne!(v1.version_id, v2.version_id);

        // History keeps both; latest is v2.
        assert_eq!(vs.history(0).len(), 2);
        assert_eq!(vs.get(0).unwrap().version_id, v2.version_id);

        // Fetch an old version explicitly.
        assert_eq!(vs.get_version(0, v1.version_id).unwrap().vector, vec![1.0; 4]);

        // Rollback creates a NEW version cloning v1's content.
        vs.rollback(0, v1.version_id).unwrap();
        assert_eq!(vs.get(0).unwrap().vector, vec![1.0; 4]);

        let stats = vs.stats();
        assert_eq!(stats.active_vectors, 1);
        assert!(stats.total_versions >= 3);
    }

    #[test]
    fn test_versioning_delete_marks_inactive() {
        let vs = VersionedStore::new(2);
        vs.upsert(7, vec![0.5, 0.5], &[], None).unwrap();
        vs.delete(7).unwrap();
        assert!(vs.get(7).map_or(true, |v| v.is_deleted));
        assert_eq!(vs.stats().active_vectors, 0);
    }
}
