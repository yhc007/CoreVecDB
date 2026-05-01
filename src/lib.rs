#[cfg(feature = "server")]
pub mod api;
pub mod cache;
pub mod collection;

/// Library-level error type for embedded API consumers.
#[derive(Debug, thiserror::Error)]
pub enum VecDbError {
    /// Collection not found by name.
    #[error("collection not found: {0}")]
    NotFound(String),

    /// Invalid input (dimension mismatch, ID overflow, bad config).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Storage I/O error.
    #[error("storage error: {0}")]
    Storage(#[from] std::io::Error),

    /// Index error (HNSW build/search failure).
    #[error("index error: {0}")]
    Index(String),

    /// Internal error (catch-all for anyhow conversions).
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}
pub mod config;
pub mod diskann;
pub mod embedded;
pub mod gpu;
pub mod hnswpp;
pub mod index;
pub mod ivfpq;
pub mod metrics;
pub mod navix;
pub mod payload;
#[cfg(feature = "server")]
pub mod proto;
pub mod quantization;
pub mod query;
pub mod rabitq;
pub mod replication;
pub mod sharding;
pub mod simd;
pub mod sparse;
pub mod storage;
pub mod text;
pub mod versioning;
pub mod wal;
