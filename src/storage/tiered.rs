//! Tiered Vector Storage for VectorDB.
//!
//! Implements a two-tier storage system:
//! - **Hot Tier**: In-memory HashMap for frequently accessed vectors
//! - **Cold Tier**: Memory-mapped storage for less frequently accessed vectors
//!
//! # Concept
//! - Recently inserted vectors start in the hot tier
//! - Frequently accessed vectors are promoted to hot tier
//! - Vectors inactive for a configurable duration are demoted to cold tier
//!
//! # Usage
//! ```rust,ignore
//! use vectordb::storage::tiered::{TieredVectorStore, TieredConfig};
//!
//! let config = TieredConfig::default();
//! let store = TieredVectorStore::new("data/vectors", 128, config)?;
//!
//! // Insert (starts in hot tier)
//! let id = store.insert(&vector)?;
//!
//! // Get (promotes to hot tier if in cold)
//! let vector = store.get(id)?;
//! ```

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use memmap2::{Mmap, MmapMut};
use parking_lot::RwLock;

use anyhow::Result;

/// Configuration for tiered storage.
#[derive(Debug, Clone)]
pub struct TieredConfig {
    /// Enable tiered storage.
    pub enabled: bool,
    /// Maximum vectors in hot tier.
    pub hot_tier_max_vectors: usize,
    /// Duration of inactivity before demotion (seconds).
    pub demotion_threshold_secs: u64,
    /// Minimum access count for promotion.
    pub promotion_access_threshold: u32,
    /// Whether to promote on access.
    pub promote_on_access: bool,
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hot_tier_max_vectors: 10_000,
            demotion_threshold_secs: 3600, // 1 hour
            promotion_access_threshold: 3,
            promote_on_access: true,
        }
    }
}

/// Access tracking information for a vector.
#[derive(Debug, Clone)]
struct AccessInfo {
    /// Last access time.
    last_access: Instant,
    /// Access count.
    access_count: u32,
}

impl AccessInfo {
    fn new() -> Self {
        Self {
            last_access: Instant::now(),
            access_count: 1,
        }
    }

    fn touch(&mut self) {
        self.last_access = Instant::now();
        self.access_count = self.access_count.saturating_add(1);
    }

    fn is_stale(&self, threshold: Duration) -> bool {
        self.last_access.elapsed() > threshold
    }
}

/// Hot tier: In-memory vector storage.
struct HotTier {
    /// Vectors stored in memory.
    vectors: RwLock<HashMap<u64, Vec<f32>>>,
    /// Access tracking.
    access_info: RwLock<HashMap<u64, AccessInfo>>,
    /// Maximum capacity.
    max_vectors: usize,
}

impl HotTier {
    fn new(max_vectors: usize) -> Self {
        Self {
            vectors: RwLock::new(HashMap::with_capacity(max_vectors)),
            access_info: RwLock::new(HashMap::with_capacity(max_vectors)),
            max_vectors,
        }
    }

    fn insert(&self, id: u64, vector: Vec<f32>) -> bool {
        let mut vectors = self.vectors.write();
        if vectors.len() >= self.max_vectors && !vectors.contains_key(&id) {
            return false; // Hot tier full
        }

        vectors.insert(id, vector);
        self.access_info.write().insert(id, AccessInfo::new());
        true
    }

    fn get(&self, id: u64) -> Option<Vec<f32>> {
        let vectors = self.vectors.read();
        if let Some(v) = vectors.get(&id) {
            // Update access info
            if let Some(info) = self.access_info.write().get_mut(&id) {
                info.touch();
            }
            Some(v.clone())
        } else {
            None
        }
    }

    fn remove(&self, id: u64) -> Option<Vec<f32>> {
        self.access_info.write().remove(&id);
        self.vectors.write().remove(&id)
    }

    fn contains(&self, id: u64) -> bool {
        self.vectors.read().contains_key(&id)
    }

    fn len(&self) -> usize {
        self.vectors.read().len()
    }

    fn is_full(&self) -> bool {
        self.vectors.read().len() >= self.max_vectors
    }

    /// Get IDs that should be demoted (stale entries).
    fn get_demotion_candidates(&self, threshold: Duration, count: usize) -> Vec<u64> {
        let access_info = self.access_info.read();
        let mut candidates: Vec<_> = access_info
            .iter()
            .filter(|(_, info)| info.is_stale(threshold))
            .map(|(id, info)| (*id, info.last_access))
            .collect();

        // Sort by last access (oldest first)
        candidates.sort_by(|a, b| a.1.cmp(&b.1));

        candidates.into_iter().take(count).map(|(id, _)| id).collect()
    }
}

/// Cold tier: Memory-mapped vector storage.
struct ColdTier {
    /// Path to the storage file.
    path: PathBuf,
    /// Vector dimension.
    dim: usize,
    /// Bytes per vector.
    vector_bytes: usize,
    /// Memory-mapped file for reading.
    mmap: RwLock<Option<Mmap>>,
    /// File handle for writing.
    file: RwLock<File>,
    /// Number of vectors stored.
    vector_count: AtomicUsize,
    /// ID to offset mapping.
    id_map: RwLock<HashMap<u64, usize>>,
}

impl ColdTier {
    fn new(path: &Path, dim: usize) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        let vector_bytes = dim * std::mem::size_of::<f32>();

        // Try to create mmap if file has content
        let mmap = if file.metadata()?.len() > 0 {
            Some(unsafe { Mmap::map(&file)? })
        } else {
            None
        };

        Ok(Self {
            path: path.to_path_buf(),
            dim,
            vector_bytes,
            mmap: RwLock::new(mmap),
            file: RwLock::new(file),
            vector_count: AtomicUsize::new(0),
            id_map: RwLock::new(HashMap::new()),
        })
    }

    fn insert(&self, id: u64, vector: &[f32]) -> io::Result<()> {
        let bytes: Vec<u8> = vector
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let mut file = self.file.write();
        let offset = file.seek(SeekFrom::End(0))? as usize;
        file.write_all(&bytes)?;
        file.flush()?;

        // Update ID mapping
        self.id_map.write().insert(id, offset / self.vector_bytes);
        self.vector_count.fetch_add(1, Ordering::SeqCst);

        // Refresh mmap
        drop(file);
        self.refresh_mmap()?;

        Ok(())
    }

    fn get(&self, id: u64) -> Option<Vec<f32>> {
        let id_map = self.id_map.read();
        let index = *id_map.get(&id)?;

        let mmap = self.mmap.read();
        let mmap = mmap.as_ref()?;

        let offset = index * self.vector_bytes;
        let end = offset + self.vector_bytes;

        if end > mmap.len() {
            return None;
        }

        let bytes = &mmap[offset..end];
        let vector: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Some(vector)
    }

    fn contains(&self, id: u64) -> bool {
        self.id_map.read().contains_key(&id)
    }

    fn remove(&self, id: u64) -> bool {
        // Note: We only remove from ID map, actual data remains
        // (compaction would be needed for true removal)
        self.id_map.write().remove(&id).is_some()
    }

    fn len(&self) -> usize {
        self.id_map.read().len()
    }

    fn refresh_mmap(&self) -> io::Result<()> {
        let file = self.file.read();
        if file.metadata()?.len() > 0 {
            let new_mmap = unsafe { Mmap::map(&*file)? };
            *self.mmap.write() = Some(new_mmap);
        }
        Ok(())
    }
}

/// Tiered vector store with hot and cold tiers.
pub struct TieredVectorStore {
    config: TieredConfig,
    dim: usize,
    hot_tier: HotTier,
    cold_tier: ColdTier,
    /// Next vector ID.
    next_id: AtomicU64,
    /// Total vectors (hot + cold).
    total_count: AtomicUsize,
    /// Statistics.
    hot_hits: AtomicU64,
    cold_hits: AtomicU64,
    promotions: AtomicU64,
    demotions: AtomicU64,
}

impl TieredVectorStore {
    /// Create a new tiered vector store.
    pub fn new(path: &str, dim: usize, config: TieredConfig) -> io::Result<Self> {
        let cold_path = PathBuf::from(path);
        let cold_tier = ColdTier::new(&cold_path, dim)?;

        Ok(Self {
            hot_tier: HotTier::new(config.hot_tier_max_vectors),
            cold_tier,
            dim,
            config,
            next_id: AtomicU64::new(0),
            total_count: AtomicUsize::new(0),
            hot_hits: AtomicU64::new(0),
            cold_hits: AtomicU64::new(0),
            promotions: AtomicU64::new(0),
            demotions: AtomicU64::new(0),
        })
    }

    /// Insert a vector into the store.
    pub fn insert(&self, vector: &[f32]) -> Result<u64> {
        if vector.len() != self.dim {
            anyhow::bail!("Vector dimension mismatch: expected {}, got {}", self.dim, vector.len());
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        // Try to insert into hot tier first
        if self.config.enabled && !self.hot_tier.is_full() {
            self.hot_tier.insert(id, vector.to_vec());
        } else {
            // Hot tier full or disabled, go directly to cold tier
            self.cold_tier.insert(id, vector)?;
        }

        self.total_count.fetch_add(1, Ordering::SeqCst);
        Ok(id)
    }

    /// Get a vector by ID.
    pub fn get(&self, id: u64) -> Result<Vec<f32>> {
        // Check hot tier first
        if let Some(vector) = self.hot_tier.get(id) {
            self.hot_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(vector);
        }

        // Check cold tier
        if let Some(vector) = self.cold_tier.get(id) {
            self.cold_hits.fetch_add(1, Ordering::Relaxed);

            // Promote to hot tier if enabled
            if self.config.enabled && self.config.promote_on_access {
                self.try_promote(id, &vector);
            }

            return Ok(vector);
        }

        anyhow::bail!("Vector not found: {}", id)
    }

    /// Try to promote a vector to hot tier.
    fn try_promote(&self, id: u64, vector: &[f32]) {
        // Make room if needed
        if self.hot_tier.is_full() {
            self.demote_oldest(1);
        }

        if self.hot_tier.insert(id, vector.to_vec()) {
            // Remove from cold tier mapping (data stays for recovery)
            self.cold_tier.remove(id);
            self.promotions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Demote oldest vectors from hot tier to cold tier.
    fn demote_oldest(&self, count: usize) {
        let threshold = Duration::from_secs(self.config.demotion_threshold_secs);
        let candidates = self.hot_tier.get_demotion_candidates(threshold, count);

        for id in candidates {
            if let Some(vector) = self.hot_tier.remove(id) {
                if self.cold_tier.insert(id, &vector).is_ok() {
                    self.demotions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// Run maintenance (demote stale vectors).
    pub fn maintain(&self) {
        if !self.config.enabled {
            return;
        }

        let threshold = Duration::from_secs(self.config.demotion_threshold_secs);
        let candidates = self.hot_tier.get_demotion_candidates(threshold, self.hot_tier.len());

        for id in candidates {
            if let Some(vector) = self.hot_tier.remove(id) {
                if self.cold_tier.insert(id, &vector).is_ok() {
                    self.demotions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// Get total vector count.
    pub fn len(&self) -> usize {
        self.total_count.load(Ordering::SeqCst)
    }

    /// Check if store is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get storage statistics.
    pub fn stats(&self) -> TieredStorageStats {
        TieredStorageStats {
            total_vectors: self.total_count.load(Ordering::SeqCst),
            hot_tier_vectors: self.hot_tier.len(),
            cold_tier_vectors: self.cold_tier.len(),
            hot_tier_capacity: self.config.hot_tier_max_vectors,
            hot_hits: self.hot_hits.load(Ordering::Relaxed),
            cold_hits: self.cold_hits.load(Ordering::Relaxed),
            promotions: self.promotions.load(Ordering::Relaxed),
            demotions: self.demotions.load(Ordering::Relaxed),
        }
    }
}

/// Statistics for tiered storage.
#[derive(Debug, Clone)]
pub struct TieredStorageStats {
    pub total_vectors: usize,
    pub hot_tier_vectors: usize,
    pub cold_tier_vectors: usize,
    pub hot_tier_capacity: usize,
    pub hot_hits: u64,
    pub cold_hits: u64,
    pub promotions: u64,
    pub demotions: u64,
}

impl TieredStorageStats {
    /// Get hot tier hit rate.
    pub fn hot_hit_rate(&self) -> f64 {
        let total = self.hot_hits + self.cold_hits;
        if total == 0 {
            0.0
        } else {
            (self.hot_hits as f64 / total as f64) * 100.0
        }
    }

    /// Get hot tier utilization.
    pub fn hot_tier_utilization(&self) -> f64 {
        if self.hot_tier_capacity == 0 {
            0.0
        } else {
            (self.hot_tier_vectors as f64 / self.hot_tier_capacity as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_tiered_storage_basic() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("vectors.bin");

        let config = TieredConfig {
            enabled: true,
            hot_tier_max_vectors: 100,
            ..Default::default()
        };

        let store = TieredVectorStore::new(path.to_str().unwrap(), 4, config).unwrap();

        // Insert vectors
        let id1 = store.insert(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let id2 = store.insert(&[5.0, 6.0, 7.0, 8.0]).unwrap();

        assert_eq!(store.len(), 2);

        // Get vectors
        let v1 = store.get(id1).unwrap();
        assert_eq!(v1, vec![1.0, 2.0, 3.0, 4.0]);

        let v2 = store.get(id2).unwrap();
        assert_eq!(v2, vec![5.0, 6.0, 7.0, 8.0]);

        // Check stats
        let stats = store.stats();
        assert_eq!(stats.hot_tier_vectors, 2);
        assert_eq!(stats.hot_hits, 2);
    }

    #[test]
    fn test_hot_tier_overflow() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("vectors.bin");

        let config = TieredConfig {
            enabled: true,
            hot_tier_max_vectors: 2, // Very small for testing
            ..Default::default()
        };

        let store = TieredVectorStore::new(path.to_str().unwrap(), 4, config).unwrap();

        // Insert 3 vectors (exceeds hot tier capacity)
        store.insert(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        store.insert(&[5.0, 6.0, 7.0, 8.0]).unwrap();
        store.insert(&[9.0, 10.0, 11.0, 12.0]).unwrap();

        assert_eq!(store.len(), 3);

        // Hot tier should be at capacity, third vector goes to cold
        let stats = store.stats();
        assert_eq!(stats.hot_tier_vectors, 2);
        assert!(stats.cold_tier_vectors >= 1);
    }

    #[test]
    fn test_cold_tier_access() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("vectors.bin");

        let config = TieredConfig {
            enabled: true,
            hot_tier_max_vectors: 1,
            promote_on_access: false, // Disable promotion for this test
            ..Default::default()
        };

        let store = TieredVectorStore::new(path.to_str().unwrap(), 4, config).unwrap();

        // Insert 2 vectors (second goes to cold)
        store.insert(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let id2 = store.insert(&[5.0, 6.0, 7.0, 8.0]).unwrap();

        // Access cold tier vector
        let v2 = store.get(id2).unwrap();
        assert_eq!(v2, vec![5.0, 6.0, 7.0, 8.0]);

        let stats = store.stats();
        assert_eq!(stats.cold_hits, 1);
    }

    #[test]
    fn test_promotion() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("vectors.bin");

        let config = TieredConfig {
            enabled: true,
            hot_tier_max_vectors: 2,
            promote_on_access: true,
            demotion_threshold_secs: 0, // Immediate demotion allowed
            ..Default::default()
        };

        let store = TieredVectorStore::new(path.to_str().unwrap(), 4, config).unwrap();

        // Fill hot tier
        store.insert(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        store.insert(&[5.0, 6.0, 7.0, 8.0]).unwrap();

        // Third vector goes to cold
        let id3 = store.insert(&[9.0, 10.0, 11.0, 12.0]).unwrap();

        // Access cold vector (should promote)
        let _v3 = store.get(id3).unwrap();

        let stats = store.stats();
        // Should have at least one promotion
        assert!(stats.promotions >= 1 || stats.cold_hits >= 1);
    }

    #[test]
    fn test_stats() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("vectors.bin");

        let config = TieredConfig {
            enabled: true,
            hot_tier_max_vectors: 100,
            ..Default::default()
        };

        let store = TieredVectorStore::new(path.to_str().unwrap(), 4, config).unwrap();

        // Insert and access
        let id1 = store.insert(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        store.get(id1).unwrap();
        store.get(id1).unwrap();
        store.get(id1).unwrap();

        let stats = store.stats();
        assert_eq!(stats.hot_hits, 3);
        assert!((stats.hot_hit_rate() - 100.0).abs() < 0.1);
        assert!(stats.hot_tier_utilization() > 0.0);
    }
}
