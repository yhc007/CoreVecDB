use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyReadonlyArray1;
use std::collections::HashMap;

use vectordb::collection::{
    CollectionConfig, SearchParams,
};
use vectordb::embedded::CoreVecDB as RustCoreVecDB;

/// CoreVecDB — embedded vector database (zero HTTP overhead).
#[pyclass]
struct CoreVecDB {
    inner: RustCoreVecDB,
}

#[pymethods]
impl CoreVecDB {
    /// Open or create a database at the given directory.
    #[new]
    fn new(data_dir: &str) -> PyResult<Self> {
        let inner = RustCoreVecDB::open(data_dir)
            .map_err(|e| PyValueError::new_err(format!("Failed to open database: {}", e)))?;
        Ok(Self { inner })
    }

    /// Create a new collection.
    #[pyo3(signature = (name, dim, distance="cosine", indexed_fields=None, numeric_fields=None, text_fields=None, max_elements=10000))]
    fn create_collection(
        &self,
        name: &str,
        dim: usize,
        distance: &str,
        indexed_fields: Option<Vec<String>>,
        numeric_fields: Option<Vec<String>>,
        text_fields: Option<Vec<String>>,
        max_elements: usize,
    ) -> PyResult<()> {
        let mut config = CollectionConfig::new(name, dim)
            .with_distance(distance)
            .with_hnsw(max_elements, 24, 400);

        let sf = indexed_fields.unwrap_or_default();
        let nf = numeric_fields.unwrap_or_default();
        if !sf.is_empty() || !nf.is_empty() {
            config = config.with_indexed_fields(sf, nf);
        }

        if let Some(tf) = text_fields {
            config = config.with_text_fields(tf);
        }

        self.inner
            .create_collection(config)
            .map_err(|e| PyValueError::new_err(format!("Failed to create collection: {}", e)))
    }

    /// Get a handle to an existing collection.
    fn collection(&self, name: &str) -> PyResult<CollectionHandle> {
        self.inner
            .collection(name)
            .map(|h| CollectionHandle { inner: h })
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    /// List all collections.
    fn list_collections(&self) -> Vec<PyCollectionInfo> {
        self.inner
            .list_collections()
            .into_iter()
            .map(|c| PyCollectionInfo {
                name: c.name,
                dim: c.dim,
                count: c.vector_count,
            })
            .collect()
    }

    /// Drop a collection and delete its data.
    fn drop_collection(&self, name: &str) -> PyResult<()> {
        self.inner
            .drop_collection(name)
            .map_err(|e| PyValueError::new_err(format!("Failed to drop collection: {}", e)))
    }

    /// Flush all collections to disk.
    fn flush(&self) -> PyResult<()> {
        self.inner
            .flush()
            .map_err(|e| PyValueError::new_err(format!("Failed to flush: {}", e)))
    }
}

/// Collection handle for vector operations.
#[pyclass]
#[derive(Clone)]
struct CollectionHandle {
    inner: vectordb::embedded::CollectionHandle,
}

#[pymethods]
impl CollectionHandle {
    /// Insert a single vector with optional metadata.
    #[pyo3(signature = (vector, metadata=None))]
    fn insert(
        &self,
        vector: PyReadonlyArray1<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<u64> {
        let vec = vector.as_slice()
            .map_err(|e| PyValueError::new_err(format!("Invalid vector array: {}", e)))?;
        let meta: Vec<(&str, &str)> = metadata
            .as_ref()
            .map(|m| m.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect())
            .unwrap_or_default();
        self.inner
            .insert(vec, &meta)
            .map_err(|e| PyValueError::new_err(format!("Insert failed: {}", e)))
    }

    /// Insert a batch of vectors with optional metadata.
    /// vectors: list of list of float, or numpy 2D array
    #[pyo3(signature = (vectors, metadata=None))]
    fn insert_batch(
        &self,
        vectors: Vec<Vec<f32>>,
        metadata: Option<Vec<HashMap<String, String>>>,
    ) -> PyResult<u64> {
        let n = vectors.len();
        let meta: Vec<Vec<(String, String)>> = metadata
            .map(|m| {
                m.into_iter()
                    .map(|hm| hm.into_iter().collect())
                    .collect()
            })
            .unwrap_or_else(|| vec![vec![]; n]);

        self.inner
            .insert_batch(&vectors, &meta)
            .map_err(|e| PyValueError::new_err(format!("Batch insert failed: {}", e)))
    }

    /// Search for similar vectors.
    #[pyo3(signature = (vector, k=10, filter=None, filter_ids=None, include_metadata=false))]
    fn search(
        &self,
        vector: PyReadonlyArray1<f32>,
        k: usize,
        filter: Option<HashMap<String, String>>,
        filter_ids: Option<Vec<u64>>,
        include_metadata: bool,
    ) -> PyResult<Vec<PySearchResult>> {
        let vec = vector.as_slice()
            .map_err(|e| PyValueError::new_err(format!("Invalid vector: {}", e)))?;

        let mut params = SearchParams::new(vec.to_vec(), k);

        if let Some(f) = filter {
            for (k, v) in f {
                params = params.with_filter(k, v);
            }
        }

        if let Some(ids) = filter_ids {
            params = params.with_filter_ids(ids);
        }

        if include_metadata {
            params = params.with_metadata();
        }

        let results = self.inner
            .search(params)
            .map_err(|e| PyValueError::new_err(format!("Search failed: {}", e)))?;

        Ok(results
            .into_iter()
            .map(|r| PySearchResult {
                id: r.id,
                score: r.score,
                metadata: r.metadata,
            })
            .collect())
    }

    /// Hybrid vector + text search.
    #[pyo3(signature = (vector, query, k=10, alpha=0.5))]
    fn hybrid_search(
        &self,
        vector: PyReadonlyArray1<f32>,
        query: &str,
        k: usize,
        alpha: f32,
    ) -> PyResult<Vec<PyHybridResult>> {
        let vec = vector.as_slice()
            .map_err(|e| PyValueError::new_err(format!("Invalid vector: {}", e)))?;

        let results = self.inner
            .hybrid_search(vec, query, k, alpha)
            .map_err(|e| PyValueError::new_err(format!("Hybrid search failed: {}", e)))?;

        Ok(results
            .into_iter()
            .map(|r| PyHybridResult {
                id: r.id,
                combined_score: r.combined_score,
                vector_score: r.vector_score,
                text_score: r.text_score,
            })
            .collect())
    }

    /// Delete a vector by ID.
    fn delete(&self, id: u64) -> PyResult<bool> {
        self.inner
            .delete(id)
            .map_err(|e| PyValueError::new_err(format!("Delete failed: {}", e)))
    }

    /// Delete multiple vectors by ID.
    fn delete_batch(&self, ids: Vec<u64>) -> PyResult<usize> {
        self.inner
            .delete_batch(&ids)
            .map_err(|e| PyValueError::new_err(format!("Batch delete failed: {}", e)))
    }

    /// Get total vector count.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Flush to disk.
    fn flush(&self) -> PyResult<()> {
        self.inner
            .flush()
            .map_err(|e| PyValueError::new_err(format!("Flush failed: {}", e)))
    }

    /// Insert a text document (embeds it first, stores text in _text metadata).
    /// Requires candle feature.
    #[cfg(feature = "candle")]
    #[pyo3(signature = (text, edb, metadata=None))]
    fn insert_text(
        &self,
        text: &str,
        edb: &EmbeddingDB,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<u64> {
        let meta: Vec<(&str, &str)> = metadata
            .as_ref()
            .map(|m| m.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect())
            .unwrap_or_default();
        self.inner
            .insert_text(text, &edb.inner, &meta)
            .map_err(|e| PyValueError::new_err(format!("Insert text failed: {}", e)))
    }

    /// Search by text query (embeds query, then vector search).
    /// Requires candle feature.
    #[cfg(feature = "candle")]
    #[pyo3(signature = (query, edb, k=10))]
    fn search_text(
        &self,
        query: &str,
        edb: &EmbeddingDB,
        k: usize,
    ) -> PyResult<Vec<PySearchResult>> {
        let results = self.inner
            .search_text(query, &edb.inner, k)
            .map_err(|e| PyValueError::new_err(format!("Search text failed: {}", e)))?;

        Ok(results
            .into_iter()
            .map(|r| PySearchResult {
                id: r.id,
                score: r.score,
                metadata: r.metadata,
            })
            .collect())
    }

    /// Hybrid text + vector search.
    /// Requires candle feature.
    #[cfg(feature = "candle")]
    #[pyo3(signature = (query, edb, k=10, alpha=0.5))]
    fn hybrid_search_text(
        &self,
        query: &str,
        edb: &EmbeddingDB,
        k: usize,
        alpha: f32,
    ) -> PyResult<Vec<PyHybridResult>> {
        let results = self.inner
            .hybrid_search_text(query, &edb.inner, k, alpha)
            .map_err(|e| PyValueError::new_err(format!("Hybrid search text failed: {}", e)))?;

        Ok(results
            .into_iter()
            .map(|r| PyHybridResult {
                id: r.id,
                combined_score: r.combined_score,
                vector_score: r.vector_score,
                text_score: r.text_score,
            })
            .collect())
    }
}

/// Search result.
#[pyclass]
#[derive(Clone)]
struct PySearchResult {
    #[pyo3(get)]
    id: u64,
    #[pyo3(get)]
    score: f32,
    #[pyo3(get)]
    metadata: Option<HashMap<String, String>>,
}

#[pymethods]
impl PySearchResult {
    fn __repr__(&self) -> String {
        format!("SearchResult(id={}, score={:.4})", self.id, self.score)
    }
}

/// Hybrid search result.
#[pyclass]
#[derive(Clone)]
struct PyHybridResult {
    #[pyo3(get)]
    id: u64,
    #[pyo3(get)]
    combined_score: f32,
    #[pyo3(get)]
    vector_score: f32,
    #[pyo3(get)]
    text_score: f32,
}

/// Collection info.
#[pyclass]
#[derive(Clone)]
struct PyCollectionInfo {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    dim: usize,
    #[pyo3(get)]
    count: usize,
}

// ─── Candle Embedding API ───────────────────────────────────────────

/// EmbeddingDB — CoreVecDB with in-process text embedding.
///
/// Downloads and caches a transformer model from HuggingFace Hub,
/// then provides text-to-vector operations directly.
///
/// Example:
///     edb = corevecdb.EmbeddingDB("./data", model="intfloat/e5-small-v2")
///     edb.create_collection("docs")
///     col = edb.collection("docs")
///     col.insert_text("The quick brown fox", edb, metadata={"source": "test"})
///     results = col.search_text("fast fox", edb, k=5)
#[cfg(feature = "candle")]
#[pyclass]
struct EmbeddingDB {
    inner: vectordb::embedded::EmbeddingDB,
}

#[cfg(feature = "candle")]
#[pymethods]
impl EmbeddingDB {
    /// Open a database with an in-process embedder.
    #[new]
    #[pyo3(signature = (data_dir, model="intfloat/e5-small-v2"))]
    fn new(data_dir: &str, model: &str) -> PyResult<Self> {
        let inner = vectordb::embedded::EmbeddingDB::open(data_dir, model)
            .map_err(|e| PyValueError::new_err(format!("Failed to open EmbeddingDB: {}", e)))?;
        Ok(Self { inner })
    }

    /// Embed a single text into a vector.
    fn embed(&self, text: &str) -> PyResult<Vec<f32>> {
        self.inner
            .embed(text)
            .map_err(|e| PyValueError::new_err(format!("Embedding failed: {}", e)))
    }

    /// Embed a batch of texts.
    fn embed_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner
            .embed_batch(&refs)
            .map_err(|e| PyValueError::new_err(format!("Batch embedding failed: {}", e)))
    }

    /// Get the embedding dimension.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Create a collection with the embedder's dimension.
    fn create_collection(&self, name: &str) -> PyResult<()> {
        self.inner
            .create_collection(name)
            .map_err(|e| PyValueError::new_err(format!("Failed to create collection: {}", e)))
    }

    /// Get a collection handle.
    fn collection(&self, name: &str) -> PyResult<CollectionHandle> {
        self.inner
            .collection(name)
            .map(|h| CollectionHandle { inner: h })
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    /// List all collections.
    fn list_collections(&self) -> Vec<PyCollectionInfo> {
        self.inner
            .list_collections()
            .into_iter()
            .map(|c| PyCollectionInfo {
                name: c.name,
                dim: c.dim,
                count: c.vector_count,
            })
            .collect()
    }

    /// Drop a collection.
    fn drop_collection(&self, name: &str) -> PyResult<()> {
        self.inner
            .drop_collection(name)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    /// Flush all collections.
    fn flush(&self) -> PyResult<()> {
        self.inner
            .flush()
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }
}

#[pymodule]
fn corevecdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CoreVecDB>()?;
    m.add_class::<CollectionHandle>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyHybridResult>()?;
    m.add_class::<PyCollectionInfo>()?;
    #[cfg(feature = "candle")]
    m.add_class::<EmbeddingDB>()?;
    Ok(())
}
