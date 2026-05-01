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

#[pymodule]
fn corevecdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CoreVecDB>()?;
    m.add_class::<CollectionHandle>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyHybridResult>()?;
    m.add_class::<PyCollectionInfo>()?;
    Ok(())
}
