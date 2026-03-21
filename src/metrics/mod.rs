//! Prometheus metrics for VectorDB monitoring.
//!
//! Exposes metrics at `/metrics` endpoint in Prometheus text format.
//!
//! Key metrics:
//! - `vectordb_vectors_total` - Total vectors per collection
//! - `vectordb_deleted_vectors_total` - Deleted vectors per collection
//! - `vectordb_operations_total` - Operation counts by type
//! - `vectordb_operation_duration_seconds` - Operation latencies
//! - `vectordb_search_results_total` - Search result counts
//! - `vectordb_storage_bytes` - Storage size per collection

use lazy_static::lazy_static;
use prometheus::{
    self, Encoder, TextEncoder,
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramVec,
    Opts, Registry,
};
use std::sync::Arc;

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();

    // ========================================================================
    // Collection Metrics
    // ========================================================================

    /// Total vectors per collection
    pub static ref VECTORS_TOTAL: GaugeVec = GaugeVec::new(
        Opts::new("vectordb_vectors_total", "Total number of vectors per collection"),
        &["collection"]
    ).expect("metric creation failed");

    /// Deleted vectors per collection
    pub static ref DELETED_VECTORS_TOTAL: GaugeVec = GaugeVec::new(
        Opts::new("vectordb_deleted_vectors_total", "Number of deleted vectors per collection"),
        &["collection"]
    ).expect("metric creation failed");

    /// Active vectors per collection (total - deleted)
    pub static ref ACTIVE_VECTORS_TOTAL: GaugeVec = GaugeVec::new(
        Opts::new("vectordb_active_vectors_total", "Number of active vectors per collection"),
        &["collection"]
    ).expect("metric creation failed");

    /// Collection dimension
    pub static ref COLLECTION_DIMENSION: GaugeVec = GaugeVec::new(
        Opts::new("vectordb_collection_dimension", "Vector dimension per collection"),
        &["collection"]
    ).expect("metric creation failed");

    /// Number of collections
    pub static ref COLLECTIONS_TOTAL: Gauge = Gauge::new(
        "vectordb_collections_total", "Total number of collections"
    ).expect("metric creation failed");

    // ========================================================================
    // Operation Metrics
    // ========================================================================

    /// Operation counter by type
    pub static ref OPERATIONS_TOTAL: CounterVec = CounterVec::new(
        Opts::new("vectordb_operations_total", "Total operations by type"),
        &["operation", "collection", "status"]
    ).expect("metric creation failed");

    /// Operation duration histogram
    pub static ref OPERATION_DURATION: HistogramVec = HistogramVec::new(
        prometheus::HistogramOpts::new(
            "vectordb_operation_duration_seconds",
            "Operation duration in seconds"
        ).buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]),
        &["operation", "collection"]
    ).expect("metric creation failed");

    // ========================================================================
    // Search Metrics
    // ========================================================================

    /// Search results histogram
    pub static ref SEARCH_RESULTS: HistogramVec = HistogramVec::new(
        prometheus::HistogramOpts::new(
            "vectordb_search_results_total",
            "Number of results returned per search"
        ).buckets(vec![0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]),
        &["collection"]
    ).expect("metric creation failed");

    /// Search K parameter histogram
    pub static ref SEARCH_K: HistogramVec = HistogramVec::new(
        prometheus::HistogramOpts::new(
            "vectordb_search_k",
            "K parameter for search requests"
        ).buckets(vec![1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]),
        &["collection"]
    ).expect("metric creation failed");

    // ========================================================================
    // Batch Metrics
    // ========================================================================

    /// Batch size histogram
    pub static ref BATCH_SIZE: HistogramVec = HistogramVec::new(
        prometheus::HistogramOpts::new(
            "vectordb_batch_size",
            "Batch operation sizes"
        ).buckets(vec![1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]),
        &["operation", "collection"]
    ).expect("metric creation failed");

    // ========================================================================
    // Storage Metrics
    // ========================================================================

    /// Storage size in bytes per collection
    pub static ref STORAGE_BYTES: GaugeVec = GaugeVec::new(
        Opts::new("vectordb_storage_bytes", "Storage size in bytes per collection"),
        &["collection", "type"]  // type: vectors, metadata, index
    ).expect("metric creation failed");

    // ========================================================================
    // WAL/Replication Metrics
    // ========================================================================

    /// WAL sequence number
    pub static ref WAL_SEQUENCE: Gauge = Gauge::new(
        "vectordb_wal_sequence", "Current WAL sequence number"
    ).expect("metric creation failed");

    /// WAL size in bytes
    pub static ref WAL_SIZE_BYTES: Gauge = Gauge::new(
        "vectordb_wal_size_bytes", "WAL total size in bytes"
    ).expect("metric creation failed");

    /// WAL entries total
    pub static ref WAL_ENTRIES_TOTAL: Gauge = Gauge::new(
        "vectordb_wal_entries_total", "Total WAL entries"
    ).expect("metric creation failed");

    // ========================================================================
    // Server Metrics
    // ========================================================================

    /// Server info
    pub static ref SERVER_INFO: GaugeVec = GaugeVec::new(
        Opts::new("vectordb_server_info", "Server information"),
        &["version", "role"]
    ).expect("metric creation failed");

    /// Active connections (placeholder)
    pub static ref ACTIVE_CONNECTIONS: Gauge = Gauge::new(
        "vectordb_active_connections", "Number of active connections"
    ).expect("metric creation failed");
}

/// Initialize metrics registry
pub fn init_metrics() {
    // Register all metrics
    REGISTRY.register(Box::new(VECTORS_TOTAL.clone())).ok();
    REGISTRY.register(Box::new(DELETED_VECTORS_TOTAL.clone())).ok();
    REGISTRY.register(Box::new(ACTIVE_VECTORS_TOTAL.clone())).ok();
    REGISTRY.register(Box::new(COLLECTION_DIMENSION.clone())).ok();
    REGISTRY.register(Box::new(COLLECTIONS_TOTAL.clone())).ok();
    REGISTRY.register(Box::new(OPERATIONS_TOTAL.clone())).ok();
    REGISTRY.register(Box::new(OPERATION_DURATION.clone())).ok();
    REGISTRY.register(Box::new(SEARCH_RESULTS.clone())).ok();
    REGISTRY.register(Box::new(SEARCH_K.clone())).ok();
    REGISTRY.register(Box::new(BATCH_SIZE.clone())).ok();
    REGISTRY.register(Box::new(STORAGE_BYTES.clone())).ok();
    REGISTRY.register(Box::new(WAL_SEQUENCE.clone())).ok();
    REGISTRY.register(Box::new(WAL_SIZE_BYTES.clone())).ok();
    REGISTRY.register(Box::new(WAL_ENTRIES_TOTAL.clone())).ok();
    REGISTRY.register(Box::new(SERVER_INFO.clone())).ok();
    REGISTRY.register(Box::new(ACTIVE_CONNECTIONS.clone())).ok();

    // Set server info
    SERVER_INFO.with_label_values(&["0.1.0", "standalone"]).set(1.0);
}

/// Get metrics in Prometheus text format
pub fn gather_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

/// Update collection metrics
pub fn update_collection_metrics(
    collection: &str,
    total: usize,
    deleted: usize,
    dim: usize,
) {
    VECTORS_TOTAL
        .with_label_values(&[collection])
        .set(total as f64);
    DELETED_VECTORS_TOTAL
        .with_label_values(&[collection])
        .set(deleted as f64);
    ACTIVE_VECTORS_TOTAL
        .with_label_values(&[collection])
        .set((total - deleted) as f64);
    COLLECTION_DIMENSION
        .with_label_values(&[collection])
        .set(dim as f64);
}

/// Record operation
pub fn record_operation(
    operation: &str,
    collection: &str,
    status: &str,
    duration_secs: f64,
) {
    OPERATIONS_TOTAL
        .with_label_values(&[operation, collection, status])
        .inc();
    OPERATION_DURATION
        .with_label_values(&[operation, collection])
        .observe(duration_secs);
}

/// Record search results
pub fn record_search(collection: &str, k: usize, results: usize, duration_secs: f64) {
    SEARCH_K
        .with_label_values(&[collection])
        .observe(k as f64);
    SEARCH_RESULTS
        .with_label_values(&[collection])
        .observe(results as f64);
    record_operation("search", collection, "success", duration_secs);
}

/// Record batch operation
pub fn record_batch(operation: &str, collection: &str, size: usize, duration_secs: f64) {
    BATCH_SIZE
        .with_label_values(&[operation, collection])
        .observe(size as f64);
    record_operation(operation, collection, "success", duration_secs);
}

/// Update WAL metrics
pub fn update_wal_metrics(sequence: u64, entries: usize, size_bytes: u64) {
    WAL_SEQUENCE.set(sequence as f64);
    WAL_ENTRIES_TOTAL.set(entries as f64);
    WAL_SIZE_BYTES.set(size_bytes as f64);
}

/// Helper: Timer for measuring operation duration
pub struct Timer {
    start: std::time::Instant,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}
