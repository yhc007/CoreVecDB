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

use std::path::Path;
use std::sync::Arc;

use crate::collection::{
    Collection, CollectionConfig, CollectionInfo, CollectionManager,
    SearchParams, SearchResult,
};
use crate::text::HybridSearchResult;
use crate::VecDbError;

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

    /// Get the underlying CollectionManager (for advanced usage).
    pub fn manager(&self) -> &Arc<CollectionManager> {
        &self.manager
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
    pub fn create_collection(&self, name: &str) -> Result<()> {
        let config = CollectionConfig::new(name, self.dimension());
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

    /// Delete a vector by ID.
    pub fn delete(&self, id: u64) -> Result<bool> {
        self.inner.delete(id).map_err(Into::into)
    }

    /// Delete multiple vectors by ID.
    pub fn delete_batch(&self, ids: &[u64]) -> Result<usize> {
        self.inner.delete_batch(ids).map_err(Into::into)
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
        self.inner.flush().map_err(Into::into)
    }

    /// Get the underlying Collection (for advanced usage).
    pub fn inner(&self) -> &Arc<Collection> {
        &self.inner
    }
}
