//! Index Pre-warming for VectorDB.
//!
//! Loads index data into the OS page cache by touching memory-mapped pages.
//! This reduces latency for the first few queries after server startup.
//!
//! # Usage
//! ```rust,ignore
//! use vectordb::index::prewarm::{prewarm_index, PrewarmConfig};
//!
//! let config = PrewarmConfig::default();
//! prewarm_index(&index_path, &config).await?;
//! ```

use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use memmap2::Mmap;

/// Configuration for index pre-warming.
#[derive(Debug, Clone)]
pub struct PrewarmConfig {
    /// Enable pre-warming.
    pub enabled: bool,
    /// Maximum number of pages to load into cache.
    /// 0 means load all pages.
    pub max_pages: usize,
    /// Page size in bytes (typically 4096 for most systems).
    pub page_size: usize,
    /// Whether to read pages sequentially (better for HDDs) or randomly (doesn't matter for SSDs).
    pub sequential: bool,
}

impl Default for PrewarmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_pages: 0, // Load all
            page_size: 4096,
            sequential: true,
        }
    }
}

/// Statistics from pre-warming operation.
#[derive(Debug, Clone, Default)]
pub struct PrewarmStats {
    /// Number of pages touched.
    pub pages_loaded: usize,
    /// Total bytes touched.
    pub bytes_loaded: usize,
    /// Time taken in milliseconds.
    pub duration_ms: u64,
}

/// Global counter for pre-warm operations (for monitoring).
static PREWARM_OPERATIONS: AtomicUsize = AtomicUsize::new(0);

/// Pre-warm an index file by touching its pages.
///
/// This loads the file's pages into the OS page cache, reducing
/// latency for subsequent reads.
///
/// # Arguments
/// * `path` - Path to the index file (e.g., index.hnsw.graph)
/// * `config` - Pre-warming configuration
///
/// # Returns
/// Statistics about the pre-warming operation.
pub fn prewarm_file(path: &Path, config: &PrewarmConfig) -> io::Result<PrewarmStats> {
    if !config.enabled {
        return Ok(PrewarmStats::default());
    }

    if !path.exists() {
        return Ok(PrewarmStats::default());
    }

    let start = std::time::Instant::now();

    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    let total_size = mmap.len();
    let page_size = config.page_size;
    let total_pages = (total_size + page_size - 1) / page_size;

    let pages_to_load = if config.max_pages > 0 {
        config.max_pages.min(total_pages)
    } else {
        total_pages
    };

    let mut pages_loaded = 0;
    let mut bytes_loaded = 0;

    // Touch pages to load into cache
    // We use volatile read to prevent compiler from optimizing away
    if config.sequential {
        // Sequential access pattern
        for i in 0..pages_to_load {
            let offset = i * page_size;
            if offset < total_size {
                // Volatile read to prevent optimization
                unsafe {
                    std::ptr::read_volatile(&mmap[offset]);
                }
                pages_loaded += 1;
                bytes_loaded += page_size.min(total_size - offset);
            }
        }
    } else {
        // Random access pattern (useful for testing cache behavior)
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for i in 0..pages_to_load {
            // Generate pseudo-random page index
            i.hash(&mut hasher);
            let page_idx = (hasher.finish() as usize) % total_pages;
            let offset = page_idx * page_size;

            if offset < total_size {
                unsafe {
                    std::ptr::read_volatile(&mmap[offset]);
                }
                pages_loaded += 1;
                bytes_loaded += page_size.min(total_size - offset);
            }
        }
    }

    PREWARM_OPERATIONS.fetch_add(1, Ordering::Relaxed);

    let duration = start.elapsed();

    Ok(PrewarmStats {
        pages_loaded,
        bytes_loaded,
        duration_ms: duration.as_millis() as u64,
    })
}

/// Pre-warm all index files for a collection.
///
/// Loads the following files into cache:
/// - index.hnsw.graph - HNSW graph structure
/// - index.hnsw.data - HNSW node data
/// - vectors.bin - Vector storage
///
/// # Arguments
/// * `base_path` - Base path of the collection
/// * `config` - Pre-warming configuration
pub fn prewarm_collection(base_path: &Path, config: &PrewarmConfig) -> io::Result<PrewarmStats> {
    if !config.enabled {
        return Ok(PrewarmStats::default());
    }

    let mut total_stats = PrewarmStats::default();
    let start = std::time::Instant::now();

    // Pre-warm index files
    let index_files = [
        "index.hnsw.graph",
        "index.hnsw.data",
        "vectors.bin",
        "vectors.quant",
    ];

    for file_name in index_files {
        let path = base_path.join(file_name);
        if path.exists() {
            if let Ok(stats) = prewarm_file(&path, config) {
                total_stats.pages_loaded += stats.pages_loaded;
                total_stats.bytes_loaded += stats.bytes_loaded;
            }
        }
    }

    total_stats.duration_ms = start.elapsed().as_millis() as u64;

    Ok(total_stats)
}

/// Get total number of pre-warm operations performed.
pub fn prewarm_operation_count() -> usize {
    PREWARM_OPERATIONS.load(Ordering::Relaxed)
}

/// Async version of pre-warming (runs in a blocking task).
#[cfg(feature = "tokio")]
pub async fn prewarm_file_async(
    path: std::path::PathBuf,
    config: PrewarmConfig,
) -> io::Result<PrewarmStats> {
    tokio::task::spawn_blocking(move || prewarm_file(&path, &config)).await?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_prewarm_file() {
        // Create a test file with some data
        let mut temp_file = NamedTempFile::new().unwrap();
        let data = vec![0u8; 16 * 1024]; // 16KB = 4 pages
        temp_file.write_all(&data).unwrap();

        let config = PrewarmConfig {
            enabled: true,
            max_pages: 0,
            page_size: 4096,
            sequential: true,
        };

        let stats = prewarm_file(temp_file.path(), &config).unwrap();

        assert!(stats.pages_loaded >= 4);
        assert!(stats.bytes_loaded >= 16 * 1024);
        assert!(stats.duration_ms >= 0);
    }

    #[test]
    fn test_prewarm_disabled() {
        let temp_file = NamedTempFile::new().unwrap();

        let config = PrewarmConfig {
            enabled: false,
            ..Default::default()
        };

        let stats = prewarm_file(temp_file.path(), &config).unwrap();

        assert_eq!(stats.pages_loaded, 0);
        assert_eq!(stats.bytes_loaded, 0);
    }

    #[test]
    fn test_prewarm_max_pages() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let data = vec![0u8; 16 * 1024]; // 16KB = 4 pages
        temp_file.write_all(&data).unwrap();

        let config = PrewarmConfig {
            enabled: true,
            max_pages: 2, // Only load 2 pages
            page_size: 4096,
            sequential: true,
        };

        let stats = prewarm_file(temp_file.path(), &config).unwrap();

        assert_eq!(stats.pages_loaded, 2);
    }

    #[test]
    fn test_prewarm_nonexistent_file() {
        let config = PrewarmConfig::default();
        let stats = prewarm_file(Path::new("/nonexistent/file"), &config).unwrap();

        assert_eq!(stats.pages_loaded, 0);
    }
}
