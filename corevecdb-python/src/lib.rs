use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;
use numpy::PyReadonlyArray1;
use std::collections::HashMap;

use vectordb::collection::{
    CollectionConfig, RangeFilter, SearchParams, SnapshotInfo,
};
use vectordb::embedded::{CoreVecDB as RustCoreVecDB, VersionedStore as RustVersionedStore};
use vectordb::versioning::VectorVersion;

/// Parse an RFC 3339 timestamp string into a UTC datetime.
fn parse_rfc3339(ts: &str) -> PyResult<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(ts)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .map_err(|e| PyValueError::new_err(format!("invalid RFC3339 timestamp '{}': {}", ts, e)))
}

/// Parse a Python dict (same shape as the HTTP `range_filters` entries) into a
/// `RangeFilter`. Accepts: {"op": "gt"|"gte"|"lt"|"lte", "field": str, "value": num}
/// or {"op": "range"|"between", "field": str, "min": num, "max": num}.
fn parse_range_filter(d: &Bound<'_, PyDict>) -> PyResult<RangeFilter> {
    let op: String = d
        .get_item("op")?
        .ok_or_else(|| PyValueError::new_err("range filter missing 'op'"))?
        .extract()?;
    let field: String = d
        .get_item("field")?
        .ok_or_else(|| PyValueError::new_err("range filter missing 'field'"))?
        .extract()?;
    let num = |key: &str| -> PyResult<f64> {
        d.get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("range filter '{}' missing '{}'", op, key)))?
            .extract()
    };
    let rf = match op.as_str() {
        "gt" => RangeFilter::Gt { field, value: num("value")? },
        "gte" => RangeFilter::Gte { field, value: num("value")? },
        "lt" => RangeFilter::Lt { field, value: num("value")? },
        "lte" => RangeFilter::Lte { field, value: num("value")? },
        "range" => RangeFilter::Range { field, min: num("min")?, max: num("max")? },
        "between" => RangeFilter::Between { field, min: num("min")?, max: num("max")? },
        other => return Err(PyValueError::new_err(format!("unknown range filter op: {}", other))),
    };
    Ok(rf)
}

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

    /// Flush all collections to disk. Alias for `flush()` for context-manager symmetry.
    ///
    /// The underlying Rust `CollectionManager` also flushes on drop, but Python GC
    /// timing is non-deterministic — prefer `with corevecdb.CoreVecDB(...) as db:`
    /// or an explicit `close()` to guarantee buffered writes hit disk.
    fn close(&self) -> PyResult<()> {
        self.flush()
    }

    /// Create a point-in-time snapshot of a collection.
    fn create_snapshot(&self, collection: &str) -> PyResult<PySnapshotInfo> {
        self.inner
            .create_snapshot(collection)
            .map(PySnapshotInfo::from)
            .map_err(|e| PyValueError::new_err(format!("Failed to create snapshot: {}", e)))
    }

    /// List all snapshots for a collection.
    fn list_snapshots(&self, collection: &str) -> PyResult<Vec<PySnapshotInfo>> {
        self.inner
            .list_snapshots(collection)
            .map(|v| v.into_iter().map(PySnapshotInfo::from).collect())
            .map_err(|e| PyValueError::new_err(format!("Failed to list snapshots: {}", e)))
    }

    /// Restore a snapshot. With `new_name=None` the original name is reused (it must
    /// not already exist); pass a name to clone under a different collection.
    #[pyo3(signature = (snapshot_name, new_name=None))]
    fn restore_snapshot(&self, snapshot_name: &str, new_name: Option<&str>) -> PyResult<CollectionHandle> {
        self.inner
            .restore_snapshot(snapshot_name, new_name)
            .map(|h| CollectionHandle { inner: h })
            .map_err(|e| PyValueError::new_err(format!("Failed to restore snapshot: {}", e)))
    }

    /// Delete a snapshot by name.
    fn delete_snapshot(&self, snapshot_name: &str) -> PyResult<()> {
        self.inner
            .delete_snapshot(snapshot_name)
            .map_err(|e| PyValueError::new_err(format!("Failed to delete snapshot: {}", e)))
    }

    /// Enter a `with` block; returns self.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Exit a `with` block: flush buffered writes to disk.
    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        _exc_type: Option<PyObject>,
        _exc_value: Option<PyObject>,
        _traceback: Option<PyObject>,
    ) -> PyResult<bool> {
        self.flush()?;
        Ok(false)
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
    ///
    /// `range_filters` accepts a list of dicts matching the HTTP API shape, e.g.
    /// `[{"op": "gt", "field": "price", "value": 100},
    ///   {"op": "range", "field": "rating", "min": 4.0, "max": 5.0}]`.
    #[pyo3(signature = (vector, k=10, filter=None, filter_ids=None, range_filters=None, include_metadata=false))]
    fn search(
        &self,
        vector: PyReadonlyArray1<f32>,
        k: usize,
        filter: Option<HashMap<String, String>>,
        filter_ids: Option<Vec<u64>>,
        range_filters: Option<Vec<Bound<'_, PyDict>>>,
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

        if let Some(rfs) = range_filters {
            for d in &rfs {
                params = params.with_range_filter(parse_range_filter(d)?);
            }
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

    /// Fetch a vector and its metadata by ID. Returns None if not found or deleted.
    fn get(&self, id: u64) -> Option<PyVectorEntry> {
        self.inner.get(id).map(|e| PyVectorEntry {
            id: e.id,
            vector: e.vector,
            metadata: e.metadata,
        })
    }

    /// Replace a vector (and metadata). Storage is append-only, so this returns a
    /// NEW id; the old id is soft-deleted. Raises if `old_id` is missing/deleted.
    #[pyo3(signature = (old_id, vector, metadata=None))]
    fn update(
        &self,
        old_id: u64,
        vector: PyReadonlyArray1<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<u64> {
        let vec = vector.as_slice()
            .map_err(|e| PyValueError::new_err(format!("Invalid vector: {}", e)))?;
        let meta: Vec<(&str, &str)> = metadata
            .as_ref()
            .map(|m| m.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect())
            .unwrap_or_default();
        self.inner
            .update(old_id, vec, &meta)
            .map_err(|e| PyValueError::new_err(format!("Update failed: {}", e)))
    }

    /// Overwrite metadata keys for an existing vector (vector and ID unchanged).
    fn update_metadata(&self, id: u64, metadata: HashMap<String, String>) -> PyResult<()> {
        let meta: Vec<(&str, &str)> = metadata.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        self.inner
            .update_metadata(id, &meta)
            .map_err(|e| PyValueError::new_err(format!("Update metadata failed: {}", e)))
    }

    /// Text-only BM25 search (collection must have `text_fields`).
    #[pyo3(signature = (query, k=10, include_metadata=false))]
    fn text_search(
        &self,
        query: &str,
        k: usize,
        include_metadata: bool,
    ) -> PyResult<Vec<PySearchResult>> {
        let results = self.inner
            .text_search(query, k)
            .map_err(|e| PyValueError::new_err(format!("Text search failed: {}", e)))?;

        Ok(results
            .into_iter()
            .map(|r| PySearchResult {
                id: r.id,
                score: r.score,
                metadata: if include_metadata {
                    self.inner.get(r.id).map(|e| e.metadata)
                } else {
                    None
                },
            })
            .collect())
    }

    /// Hybrid vector + text search using Reciprocal Rank Fusion (RRF).
    #[pyo3(signature = (vector, query, k=10))]
    fn hybrid_search_rrf(
        &self,
        vector: PyReadonlyArray1<f32>,
        query: &str,
        k: usize,
    ) -> PyResult<Vec<PyHybridResult>> {
        let vec = vector.as_slice()
            .map_err(|e| PyValueError::new_err(format!("Invalid vector: {}", e)))?;
        let results = self.inner
            .hybrid_search_rrf(vec, query, k)
            .map_err(|e| PyValueError::new_err(format!("Hybrid RRF search failed: {}", e)))?;

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

    /// Number of active (non-deleted) vectors.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of active (non-deleted) vectors.
    fn active_count(&self) -> usize {
        self.inner.len()
    }

    /// Total number of vectors ever inserted, including soft-deleted ones.
    fn total_count(&self) -> usize {
        self.inner.total_len()
    }

    /// Number of soft-deleted vectors.
    fn deleted_count(&self) -> usize {
        self.inner.deleted_len()
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

/// A stored vector with its metadata, returned by `CollectionHandle.get`.
#[pyclass]
#[derive(Clone)]
struct PyVectorEntry {
    #[pyo3(get)]
    id: u64,
    #[pyo3(get)]
    vector: Vec<f32>,
    #[pyo3(get)]
    metadata: HashMap<String, String>,
}

#[pymethods]
impl PyVectorEntry {
    fn __repr__(&self) -> String {
        format!("VectorEntry(id={}, dim={})", self.id, self.vector.len())
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

// ─── Snapshots ──────────────────────────────────────────────────────

/// Snapshot metadata.
#[pyclass]
#[derive(Clone)]
struct PySnapshotInfo {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    collection: String,
    /// Creation time as an RFC 3339 string.
    #[pyo3(get)]
    created_at: String,
    #[pyo3(get)]
    vector_count: usize,
    #[pyo3(get)]
    size_bytes: u64,
}

impl From<SnapshotInfo> for PySnapshotInfo {
    fn from(s: SnapshotInfo) -> Self {
        Self {
            name: s.name,
            collection: s.collection,
            created_at: s.created_at.to_rfc3339(),
            vector_count: s.vector_count,
            size_bytes: s.size_bytes,
        }
    }
}

#[pymethods]
impl PySnapshotInfo {
    fn __repr__(&self) -> String {
        format!("SnapshotInfo(name={:?}, vectors={})", self.name, self.vector_count)
    }
}

// ─── Versioning ─────────────────────────────────────────────────────

/// A single version of a vector.
#[pyclass]
#[derive(Clone)]
struct PyVectorVersion {
    #[pyo3(get)]
    version_id: u64,
    #[pyo3(get)]
    vector_id: u64,
    /// Timestamp as an RFC 3339 string.
    #[pyo3(get)]
    timestamp: String,
    #[pyo3(get)]
    vector: Vec<f32>,
    #[pyo3(get)]
    metadata: HashMap<String, String>,
    #[pyo3(get)]
    is_deleted: bool,
    #[pyo3(get)]
    change_description: Option<String>,
}

impl From<VectorVersion> for PyVectorVersion {
    fn from(v: VectorVersion) -> Self {
        Self {
            version_id: v.version_id,
            vector_id: v.vector_id,
            timestamp: v.timestamp.to_rfc3339(),
            vector: v.vector,
            metadata: v.metadata,
            is_deleted: v.is_deleted,
            change_description: v.change_description,
        }
    }
}

#[pymethods]
impl PyVectorVersion {
    fn __repr__(&self) -> String {
        format!(
            "VectorVersion(version_id={}, vector_id={}, deleted={})",
            self.version_id, self.vector_id, self.is_deleted
        )
    }
}

/// Versioning statistics.
#[pyclass]
#[derive(Clone)]
struct PyVersioningStats {
    #[pyo3(get)]
    total_vectors: usize,
    #[pyo3(get)]
    active_vectors: usize,
    #[pyo3(get)]
    deleted_vectors: usize,
    #[pyo3(get)]
    total_versions: usize,
    #[pyo3(get)]
    avg_versions_per_vector: f64,
    #[pyo3(get)]
    next_version_id: u64,
}

/// In-process vector versioning (history, time-travel, rollback).
///
/// This store is **in-memory and independent** from any collection's vectors —
/// nothing in `CoreVecDB` writes to it automatically. Construct one per logical
/// vector space and drive it explicitly. Timestamps are RFC 3339 strings.
///
/// Example:
///     vs = corevecdb.VersionedStore(128)
///     v1 = vs.upsert(0, [1.0]*128, metadata={"note": "first"})
///     v2 = vs.upsert(0, [2.0]*128, description="second edit")
///     hist = vs.history(0)            # [v1, v2]
///     vs.rollback(0, v1.version_id)   # new version cloning v1
#[pyclass]
struct VersionedStore {
    inner: RustVersionedStore,
}

#[pymethods]
impl VersionedStore {
    /// Create a versioned store for `dim`-dimensional vectors.
    #[new]
    fn new(dim: usize) -> Self {
        Self { inner: RustVersionedStore::new(dim) }
    }

    /// Insert or update a vector, creating a new version.
    #[pyo3(signature = (vector_id, vector, metadata=None, description=None))]
    fn upsert(
        &self,
        vector_id: u64,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        description: Option<String>,
    ) -> PyResult<PyVectorVersion> {
        let meta: Vec<(&str, &str)> = metadata
            .as_ref()
            .map(|m| m.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect())
            .unwrap_or_default();
        self.inner
            .upsert(vector_id, vector, &meta, description.as_deref())
            .map(PyVectorVersion::from)
            .map_err(|e| PyValueError::new_err(format!("Versioned upsert failed: {}", e)))
    }

    /// Record a deletion as a new version.
    fn delete(&self, vector_id: u64) -> PyResult<PyVectorVersion> {
        self.inner
            .delete(vector_id)
            .map(PyVectorVersion::from)
            .map_err(|e| PyValueError::new_err(format!("Versioned delete failed: {}", e)))
    }

    /// Get the latest (non-deleted) version. Returns None if absent/deleted.
    fn get(&self, vector_id: u64) -> Option<PyVectorVersion> {
        self.inner.get(vector_id).map(PyVectorVersion::from)
    }

    /// Get a specific version by its version ID.
    fn get_version(&self, vector_id: u64, version_id: u64) -> Option<PyVectorVersion> {
        self.inner.get_version(vector_id, version_id).map(PyVectorVersion::from)
    }

    /// Get the version active at an RFC 3339 timestamp.
    fn get_at_timestamp(&self, vector_id: u64, timestamp: &str) -> PyResult<Option<PyVectorVersion>> {
        let ts = parse_rfc3339(timestamp)?;
        Ok(self.inner.get_at_timestamp(vector_id, ts).map(PyVectorVersion::from))
    }

    /// Full version history for a vector, oldest first.
    fn history(&self, vector_id: u64) -> Vec<PyVectorVersion> {
        self.inner.history(vector_id).into_iter().map(PyVectorVersion::from).collect()
    }

    /// All active vectors as they existed at an RFC 3339 timestamp.
    fn snapshot_at(&self, timestamp: &str) -> PyResult<Vec<PyVectorVersion>> {
        let ts = parse_rfc3339(timestamp)?;
        Ok(self.inner.snapshot_at(ts).into_iter().map(PyVectorVersion::from).collect())
    }

    /// Roll a vector back to an earlier version (creates a new version).
    fn rollback(&self, vector_id: u64, to_version_id: u64) -> PyResult<PyVectorVersion> {
        self.inner
            .rollback(vector_id, to_version_id)
            .map(PyVectorVersion::from)
            .map_err(|e| PyValueError::new_err(format!("Rollback failed: {}", e)))
    }

    /// Statistics across all versioned vectors.
    fn stats(&self) -> PyVersioningStats {
        let s = self.inner.stats();
        PyVersioningStats {
            total_vectors: s.total_vectors,
            active_vectors: s.active_vectors,
            deleted_vectors: s.deleted_vectors,
            total_versions: s.total_versions,
            avg_versions_per_vector: s.avg_versions_per_vector,
            next_version_id: s.next_version_id,
        }
    }

    /// Compact old versions per config. Returns number of versions removed.
    fn compact(&self) -> usize {
        self.inner.compact()
    }

    /// Vector dimension.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }
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
    m.add_class::<PyVectorEntry>()?;
    m.add_class::<PyHybridResult>()?;
    m.add_class::<PyCollectionInfo>()?;
    m.add_class::<PySnapshotInfo>()?;
    m.add_class::<VersionedStore>()?;
    m.add_class::<PyVectorVersion>()?;
    m.add_class::<PyVersioningStats>()?;
    #[cfg(feature = "candle")]
    m.add_class::<EmbeddingDB>()?;
    Ok(())
}
