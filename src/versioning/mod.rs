//! Vector Versioning for version history and time-travel queries.
//!
//! Features:
//! - Version history for each vector ID
//! - Time-travel queries (query at specific timestamp)
//! - Version metadata (timestamp, version number)
//! - Efficient storage using delta encoding (optional)
//!
//! ## Storage Model
//! Each vector can have multiple versions. Versions are identified by:
//! - `version_id`: Auto-incrementing version number
//! - `timestamp`: When the version was created
//! - `is_deleted`: Soft delete marker for this version

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

// =============================================================================
// Version Types
// =============================================================================

/// Version information for a vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorVersion {
    /// Unique version ID (auto-incrementing)
    pub version_id: u64,
    /// Vector ID this version belongs to
    pub vector_id: u64,
    /// Timestamp when this version was created
    pub timestamp: DateTime<Utc>,
    /// The vector data
    pub vector: Vec<f32>,
    /// Metadata at this version
    pub metadata: HashMap<String, String>,
    /// Whether this version represents a deletion
    pub is_deleted: bool,
    /// Optional description of the change
    pub change_description: Option<String>,
}

impl VectorVersion {
    /// Create a new version.
    pub fn new(
        version_id: u64,
        vector_id: u64,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            version_id,
            vector_id,
            timestamp: Utc::now(),
            vector,
            metadata,
            is_deleted: false,
            change_description: None,
        }
    }

    /// Create a deletion marker version.
    pub fn deletion(version_id: u64, vector_id: u64) -> Self {
        Self {
            version_id,
            vector_id,
            timestamp: Utc::now(),
            vector: vec![],
            metadata: HashMap::new(),
            is_deleted: true,
            change_description: Some("Deleted".to_string()),
        }
    }

    /// Create version with description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.change_description = Some(desc.to_string());
        self
    }
}

/// Version history for a single vector.
#[derive(Debug, Clone, Default)]
pub struct VersionHistory {
    /// All versions, ordered by version_id (oldest first)
    versions: Vec<VectorVersion>,
}

impl VersionHistory {
    pub fn new() -> Self {
        Self { versions: vec![] }
    }

    /// Add a new version.
    pub fn add(&mut self, version: VectorVersion) {
        self.versions.push(version);
    }

    /// Get the latest version.
    pub fn latest(&self) -> Option<&VectorVersion> {
        self.versions.last()
    }

    /// Get version by version_id.
    pub fn get_version(&self, version_id: u64) -> Option<&VectorVersion> {
        self.versions.iter().find(|v| v.version_id == version_id)
    }

    /// Get version at a specific timestamp (latest version before or at timestamp).
    pub fn at_timestamp(&self, timestamp: DateTime<Utc>) -> Option<&VectorVersion> {
        self.versions
            .iter()
            .rev()
            .find(|v| v.timestamp <= timestamp)
    }

    /// Get all versions.
    pub fn all(&self) -> &[VectorVersion] {
        &self.versions
    }

    /// Get version count.
    pub fn count(&self) -> usize {
        self.versions.len()
    }

    /// Check if the vector is currently deleted.
    pub fn is_deleted(&self) -> bool {
        self.latest().map(|v| v.is_deleted).unwrap_or(false)
    }

    /// Get versions in a time range.
    pub fn in_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<&VectorVersion> {
        self.versions
            .iter()
            .filter(|v| v.timestamp >= start && v.timestamp <= end)
            .collect()
    }
}

// =============================================================================
// Versioned Vector Store
// =============================================================================

/// Configuration for versioned store.
#[derive(Debug, Clone)]
pub struct VersioningConfig {
    /// Maximum versions to keep per vector (0 = unlimited)
    pub max_versions: usize,
    /// Whether to keep deleted versions
    pub keep_deleted: bool,
    /// Retention period for old versions (None = keep forever)
    pub retention_days: Option<u32>,
}

impl Default for VersioningConfig {
    fn default() -> Self {
        Self {
            max_versions: 100,
            keep_deleted: true,
            retention_days: None,
        }
    }
}

/// Versioned vector store.
pub struct VersionedVectorStore {
    /// Version history per vector ID
    histories: RwLock<HashMap<u64, VersionHistory>>,
    /// Global version counter
    next_version_id: AtomicU64,
    /// Configuration
    config: VersioningConfig,
    /// Vector dimension
    dim: usize,
}

impl VersionedVectorStore {
    /// Create a new versioned store.
    pub fn new(dim: usize) -> Self {
        Self::with_config(dim, VersioningConfig::default())
    }

    /// Create with custom config.
    pub fn with_config(dim: usize, config: VersioningConfig) -> Self {
        Self {
            histories: RwLock::new(HashMap::new()),
            next_version_id: AtomicU64::new(1),
            config,
            dim,
        }
    }

    /// Insert or update a vector, creating a new version.
    pub fn upsert(
        &self,
        vector_id: u64,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Result<VectorVersion> {
        self.upsert_with_description(vector_id, vector, metadata, None)
    }

    /// Insert or update with change description.
    pub fn upsert_with_description(
        &self,
        vector_id: u64,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
        description: Option<&str>,
    ) -> Result<VectorVersion> {
        if vector.len() != self.dim {
            return Err(anyhow::anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            ));
        }

        let version_id = self.next_version_id.fetch_add(1, Ordering::SeqCst);
        let mut version = VectorVersion::new(version_id, vector_id, vector, metadata);

        if let Some(desc) = description {
            version = version.with_description(desc);
        }

        let mut histories = self.histories.write().unwrap();
        let history = histories.entry(vector_id).or_insert_with(VersionHistory::new);

        // Apply max versions limit
        if self.config.max_versions > 0 && history.count() >= self.config.max_versions {
            // Remove oldest version
            if !history.versions.is_empty() {
                history.versions.remove(0);
            }
        }

        history.add(version.clone());
        Ok(version)
    }

    /// Delete a vector (creates a deletion marker version).
    pub fn delete(&self, vector_id: u64) -> Result<VectorVersion> {
        let version_id = self.next_version_id.fetch_add(1, Ordering::SeqCst);
        let version = VectorVersion::deletion(version_id, vector_id);

        let mut histories = self.histories.write().unwrap();
        let history = histories.entry(vector_id).or_insert_with(VersionHistory::new);
        history.add(version.clone());

        Ok(version)
    }

    /// Get the latest version of a vector.
    pub fn get(&self, vector_id: u64) -> Option<VectorVersion> {
        let histories = self.histories.read().unwrap();
        histories.get(&vector_id)?.latest().cloned()
    }

    /// Get a specific version.
    pub fn get_version(&self, vector_id: u64, version_id: u64) -> Option<VectorVersion> {
        let histories = self.histories.read().unwrap();
        histories.get(&vector_id)?.get_version(version_id).cloned()
    }

    /// Get vector at a specific timestamp.
    pub fn get_at_timestamp(
        &self,
        vector_id: u64,
        timestamp: DateTime<Utc>,
    ) -> Option<VectorVersion> {
        let histories = self.histories.read().unwrap();
        histories.get(&vector_id)?.at_timestamp(timestamp).cloned()
    }

    /// Get all versions for a vector.
    pub fn get_history(&self, vector_id: u64) -> Vec<VectorVersion> {
        let histories = self.histories.read().unwrap();
        histories
            .get(&vector_id)
            .map(|h| h.all().to_vec())
            .unwrap_or_default()
    }

    /// Get all active (non-deleted) vector IDs.
    pub fn active_ids(&self) -> Vec<u64> {
        let histories = self.histories.read().unwrap();
        histories
            .iter()
            .filter(|(_, h)| !h.is_deleted())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all vector IDs (including deleted).
    pub fn all_ids(&self) -> Vec<u64> {
        let histories = self.histories.read().unwrap();
        histories.keys().cloned().collect()
    }

    /// Time-travel query: get all active vectors at a specific timestamp.
    pub fn snapshot_at(&self, timestamp: DateTime<Utc>) -> Vec<VectorVersion> {
        let histories = self.histories.read().unwrap();
        histories
            .values()
            .filter_map(|h| {
                let version = h.at_timestamp(timestamp)?;
                if version.is_deleted {
                    None
                } else {
                    Some(version.clone())
                }
            })
            .collect()
    }

    /// Get vectors modified since a timestamp.
    pub fn modified_since(&self, timestamp: DateTime<Utc>) -> Vec<VectorVersion> {
        let histories = self.histories.read().unwrap();
        histories
            .values()
            .filter_map(|h| {
                let latest = h.latest()?;
                if latest.timestamp > timestamp {
                    Some(latest.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Rollback a vector to a specific version.
    pub fn rollback(&self, vector_id: u64, to_version_id: u64) -> Result<VectorVersion> {
        let old_version = self.get_version(vector_id, to_version_id)
            .ok_or_else(|| anyhow::anyhow!("Version not found"))?;

        if old_version.is_deleted {
            return Err(anyhow::anyhow!("Cannot rollback to a deleted version"));
        }

        self.upsert_with_description(
            vector_id,
            old_version.vector.clone(),
            old_version.metadata.clone(),
            Some(&format!("Rollback to version {}", to_version_id)),
        )
    }

    /// Get statistics.
    pub fn stats(&self) -> VersioningStats {
        let histories = self.histories.read().unwrap();
        let total_vectors = histories.len();
        let active_vectors = histories.values().filter(|h| !h.is_deleted()).count();
        let total_versions: usize = histories.values().map(|h| h.count()).sum();
        let avg_versions = if total_vectors > 0 {
            total_versions as f64 / total_vectors as f64
        } else {
            0.0
        };

        VersioningStats {
            total_vectors,
            active_vectors,
            deleted_vectors: total_vectors - active_vectors,
            total_versions,
            avg_versions_per_vector: avg_versions,
            next_version_id: self.next_version_id.load(Ordering::SeqCst),
        }
    }

    /// Compact: remove old versions based on retention policy.
    pub fn compact(&self) -> CompactionResult {
        let mut versions_removed = 0;
        let mut histories = self.histories.write().unwrap();

        let retention_cutoff = self.config.retention_days.map(|days| {
            Utc::now() - chrono::Duration::days(days as i64)
        });

        for history in histories.values_mut() {
            let original_count = history.versions.len();

            // Keep at least the latest version
            if history.versions.len() <= 1 {
                continue;
            }

            // Remove old versions beyond retention
            if let Some(cutoff) = retention_cutoff {
                let latest = history.versions.last().cloned();
                history.versions.retain(|v| {
                    v.timestamp >= cutoff || Some(v.version_id) == latest.as_ref().map(|l| l.version_id)
                });
            }

            // Remove deleted versions if not keeping them
            if !self.config.keep_deleted {
                let latest_id = history.versions.last().map(|l| l.version_id).unwrap_or(0);
                history.versions.retain(|v| {
                    !v.is_deleted || v.version_id == latest_id
                });
            }

            versions_removed += original_count - history.versions.len();
        }

        CompactionResult { versions_removed }
    }

    /// Get dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Versioning statistics.
#[derive(Debug, Clone, Serialize)]
pub struct VersioningStats {
    pub total_vectors: usize,
    pub active_vectors: usize,
    pub deleted_vectors: usize,
    pub total_versions: usize,
    pub avg_versions_per_vector: f64,
    pub next_version_id: u64,
}

/// Compaction result.
#[derive(Debug, Clone, Serialize)]
pub struct CompactionResult {
    pub versions_removed: usize,
}

// =============================================================================
// Thread-Safe Wrapper
// =============================================================================

/// Thread-safe versioned store wrapper.
#[derive(Clone)]
pub struct ThreadSafeVersionedStore {
    inner: Arc<VersionedVectorStore>,
}

impl ThreadSafeVersionedStore {
    pub fn new(dim: usize) -> Self {
        Self {
            inner: Arc::new(VersionedVectorStore::new(dim)),
        }
    }

    pub fn with_config(dim: usize, config: VersioningConfig) -> Self {
        Self {
            inner: Arc::new(VersionedVectorStore::with_config(dim, config)),
        }
    }

    pub fn upsert(
        &self,
        vector_id: u64,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Result<VectorVersion> {
        self.inner.upsert(vector_id, vector, metadata)
    }

    pub fn upsert_with_description(
        &self,
        vector_id: u64,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
        description: Option<&str>,
    ) -> Result<VectorVersion> {
        self.inner.upsert_with_description(vector_id, vector, metadata, description)
    }

    pub fn delete(&self, vector_id: u64) -> Result<VectorVersion> {
        self.inner.delete(vector_id)
    }

    pub fn get(&self, vector_id: u64) -> Option<VectorVersion> {
        self.inner.get(vector_id)
    }

    pub fn get_version(&self, vector_id: u64, version_id: u64) -> Option<VectorVersion> {
        self.inner.get_version(vector_id, version_id)
    }

    pub fn get_at_timestamp(
        &self,
        vector_id: u64,
        timestamp: DateTime<Utc>,
    ) -> Option<VectorVersion> {
        self.inner.get_at_timestamp(vector_id, timestamp)
    }

    pub fn get_history(&self, vector_id: u64) -> Vec<VectorVersion> {
        self.inner.get_history(vector_id)
    }

    pub fn active_ids(&self) -> Vec<u64> {
        self.inner.active_ids()
    }

    pub fn snapshot_at(&self, timestamp: DateTime<Utc>) -> Vec<VectorVersion> {
        self.inner.snapshot_at(timestamp)
    }

    pub fn modified_since(&self, timestamp: DateTime<Utc>) -> Vec<VectorVersion> {
        self.inner.modified_since(timestamp)
    }

    pub fn rollback(&self, vector_id: u64, to_version_id: u64) -> Result<VectorVersion> {
        self.inner.rollback(vector_id, to_version_id)
    }

    pub fn stats(&self) -> VersioningStats {
        self.inner.stats()
    }

    pub fn compact(&self) -> CompactionResult {
        self.inner.compact()
    }

    pub fn dim(&self) -> usize {
        self.inner.dim()
    }
}

// =============================================================================
// Version Diff
// =============================================================================

/// Diff between two versions.
#[derive(Debug, Clone, Serialize)]
pub struct VersionDiff {
    pub vector_id: u64,
    pub from_version: u64,
    pub to_version: u64,
    /// Vector changed
    pub vector_changed: bool,
    /// L2 distance between vectors (if both exist)
    pub vector_distance: Option<f32>,
    /// Metadata keys added
    pub metadata_added: Vec<String>,
    /// Metadata keys removed
    pub metadata_removed: Vec<String>,
    /// Metadata keys changed
    pub metadata_changed: Vec<String>,
}

impl VersionDiff {
    /// Compute diff between two versions.
    pub fn compute(from: &VectorVersion, to: &VectorVersion) -> Self {
        let vector_changed = from.vector != to.vector;
        let vector_distance = if !from.is_deleted && !to.is_deleted && vector_changed {
            Some(l2_distance(&from.vector, &to.vector))
        } else {
            None
        };

        let from_keys: std::collections::HashSet<_> = from.metadata.keys().collect();
        let to_keys: std::collections::HashSet<_> = to.metadata.keys().collect();

        let metadata_added: Vec<String> = to_keys
            .difference(&from_keys)
            .map(|k| (*k).clone())
            .collect();

        let metadata_removed: Vec<String> = from_keys
            .difference(&to_keys)
            .map(|k| (*k).clone())
            .collect();

        let metadata_changed: Vec<String> = from_keys
            .intersection(&to_keys)
            .filter(|k| from.metadata.get(**k) != to.metadata.get(**k))
            .map(|k| (*k).clone())
            .collect();

        Self {
            vector_id: from.vector_id,
            from_version: from.version_id,
            to_version: to.version_id,
            vector_changed,
            vector_distance,
            metadata_added,
            metadata_removed,
            metadata_changed,
        }
    }
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_version_creation() {
        let store = VersionedVectorStore::new(3);

        let v1 = store.upsert(0, vec![1.0, 2.0, 3.0], HashMap::new()).unwrap();
        assert_eq!(v1.version_id, 1);
        assert_eq!(v1.vector_id, 0);
        assert!(!v1.is_deleted);
    }

    #[test]
    fn test_multiple_versions() {
        let store = VersionedVectorStore::new(3);

        // Insert first version
        store.upsert(0, vec![1.0, 2.0, 3.0], HashMap::new()).unwrap();

        // Update creates new version
        let v2 = store.upsert(0, vec![4.0, 5.0, 6.0], HashMap::new()).unwrap();
        assert_eq!(v2.version_id, 2);

        // History should have 2 versions
        let history = store.get_history(0);
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_delete_creates_version() {
        let store = VersionedVectorStore::new(3);

        store.upsert(0, vec![1.0, 2.0, 3.0], HashMap::new()).unwrap();
        let del = store.delete(0).unwrap();

        assert!(del.is_deleted);
        assert_eq!(del.version_id, 2);

        // Latest should be deleted
        let latest = store.get(0).unwrap();
        assert!(latest.is_deleted);
    }

    #[test]
    fn test_time_travel() {
        let store = VersionedVectorStore::new(3);

        // Insert at time T1
        store.upsert(0, vec![1.0, 2.0, 3.0], HashMap::new()).unwrap();
        let t1 = Utc::now();

        sleep(Duration::from_millis(10));

        // Update at time T2
        store.upsert(0, vec![4.0, 5.0, 6.0], HashMap::new()).unwrap();

        // Query at T1 should return first version
        let at_t1 = store.get_at_timestamp(0, t1).unwrap();
        assert_eq!(at_t1.vector, vec![1.0, 2.0, 3.0]);

        // Query at now should return latest
        let at_now = store.get_at_timestamp(0, Utc::now()).unwrap();
        assert_eq!(at_now.vector, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_snapshot_at() {
        let store = VersionedVectorStore::new(3);

        store.upsert(0, vec![1.0, 2.0, 3.0], HashMap::new()).unwrap();
        store.upsert(1, vec![4.0, 5.0, 6.0], HashMap::new()).unwrap();
        let t1 = Utc::now();

        sleep(Duration::from_millis(10));

        // Delete one vector
        store.delete(0).unwrap();

        // Snapshot at T1 should have both vectors
        let snapshot = store.snapshot_at(t1);
        assert_eq!(snapshot.len(), 2);

        // Current snapshot should only have vector 1
        let active = store.active_ids();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0], 1);
    }

    #[test]
    fn test_rollback() {
        let store = VersionedVectorStore::new(3);

        let v1 = store.upsert(0, vec![1.0, 2.0, 3.0], HashMap::new()).unwrap();
        store.upsert(0, vec![4.0, 5.0, 6.0], HashMap::new()).unwrap();

        // Rollback to v1
        let rollback = store.rollback(0, v1.version_id).unwrap();
        assert_eq!(rollback.vector, vec![1.0, 2.0, 3.0]);
        assert!(rollback.change_description.unwrap().contains("Rollback"));
    }

    #[test]
    fn test_version_diff() {
        let mut meta1 = HashMap::new();
        meta1.insert("key1".to_string(), "val1".to_string());
        meta1.insert("key2".to_string(), "val2".to_string());

        let mut meta2 = HashMap::new();
        meta2.insert("key2".to_string(), "val2_changed".to_string());
        meta2.insert("key3".to_string(), "val3".to_string());

        let v1 = VectorVersion::new(1, 0, vec![1.0, 2.0, 3.0], meta1);
        let v2 = VectorVersion::new(2, 0, vec![4.0, 5.0, 6.0], meta2);

        let diff = VersionDiff::compute(&v1, &v2);

        assert!(diff.vector_changed);
        assert!(diff.vector_distance.is_some());
        assert!(diff.metadata_added.contains(&"key3".to_string()));
        assert!(diff.metadata_removed.contains(&"key1".to_string()));
        assert!(diff.metadata_changed.contains(&"key2".to_string()));
    }

    #[test]
    fn test_max_versions() {
        let config = VersioningConfig {
            max_versions: 3,
            keep_deleted: true,
            retention_days: None,
        };
        let store = VersionedVectorStore::with_config(3, config);

        // Insert 5 versions
        for i in 0..5 {
            store.upsert(0, vec![i as f32, 0.0, 0.0], HashMap::new()).unwrap();
        }

        // Should only keep 3 versions
        let history = store.get_history(0);
        assert_eq!(history.len(), 3);

        // First version should be version 3 (versions 1,2 pruned)
        assert_eq!(history[0].version_id, 3);
    }

    #[test]
    fn test_stats() {
        let store = VersionedVectorStore::new(3);

        store.upsert(0, vec![1.0, 2.0, 3.0], HashMap::new()).unwrap();
        store.upsert(0, vec![4.0, 5.0, 6.0], HashMap::new()).unwrap();
        store.upsert(1, vec![7.0, 8.0, 9.0], HashMap::new()).unwrap();
        store.delete(1).unwrap();

        let stats = store.stats();
        assert_eq!(stats.total_vectors, 2);
        assert_eq!(stats.active_vectors, 1);
        assert_eq!(stats.deleted_vectors, 1);
        assert_eq!(stats.total_versions, 4);
    }

    #[test]
    fn test_modified_since() {
        let store = VersionedVectorStore::new(3);

        store.upsert(0, vec![1.0, 2.0, 3.0], HashMap::new()).unwrap();
        let t1 = Utc::now();

        sleep(Duration::from_millis(10));

        store.upsert(1, vec![4.0, 5.0, 6.0], HashMap::new()).unwrap();

        let modified = store.modified_since(t1);
        assert_eq!(modified.len(), 1);
        assert_eq!(modified[0].vector_id, 1);
    }
}
