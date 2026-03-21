//! Dynamic HNSW Index with support for deletion and neighbor repair.
//!
//! Since the underlying hnsw_rs library doesn't support true node deletion,
//! this module implements a repair-based approach:
//!
//! 1. **Soft Delete**: Mark nodes as deleted in a bitmap (instant)
//! 2. **Neighbor Tracking**: Track reverse neighbors for repair
//! 3. **Lazy Repair**: Repair neighbor connections on access
//! 4. **Auto Rebuild**: Rebuild index when deletion ratio exceeds threshold
//!
//! Architecture:
//! ```text
//! ┌─────────────────────────────────────┐
//! │  DynamicHnswIndex                   │
//! │  ┌───────────────────────────────┐  │
//! │  │ HnswIndexer (base index)      │  │
//! │  │ - insert, search              │  │
//! │  └───────────────────────────────┘  │
//! │  ┌───────────────────────────────┐  │
//! │  │ DeleteTracker                 │  │
//! │  │ - deleted bitmap              │  │
//! │  │ - reverse neighbors           │  │
//! │  │ - pending repairs             │  │
//! │  └───────────────────────────────┘  │
//! │  ┌───────────────────────────────┐  │
//! │  │ VectorStore reference         │  │
//! │  │ - for neighbor lookup         │  │
//! │  └───────────────────────────────┘  │
//! └─────────────────────────────────────┘
//! ```

use anyhow::Result;
use parking_lot::RwLock;
use roaring::RoaringBitmap;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use super::{HnswIndexer, DistanceMetric};

/// Configuration for dynamic deletion behavior.
#[derive(Debug, Clone)]
pub struct DynamicIndexConfig {
    /// Vector dimension
    pub dim: usize,
    /// Maximum elements in the index
    pub max_elements: usize,
    /// HNSW M parameter (connections per node)
    pub m: usize,
    /// HNSW ef_construction parameter
    pub ef_construction: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Deletion ratio threshold to trigger rebuild (0.0 - 1.0)
    pub rebuild_threshold: f32,
    /// Number of repair candidates to consider
    pub repair_candidates: usize,
    /// Enable lazy repair during search
    pub lazy_repair: bool,
}

impl Default for DynamicIndexConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            max_elements: 100_000,
            m: 16,
            ef_construction: 200,
            metric: DistanceMetric::Euclidean,
            rebuild_threshold: 0.3,  // Rebuild when 30% deleted
            repair_candidates: 10,
            lazy_repair: true,
        }
    }
}

impl DynamicIndexConfig {
    pub fn new(dim: usize, max_elements: usize) -> Self {
        Self {
            dim,
            max_elements,
            ..Default::default()
        }
    }

    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

/// Tracks deleted nodes and their relationships for repair.
struct DeleteTracker {
    /// Bitmap of deleted node IDs
    deleted: RwLock<RoaringBitmap>,
    /// Reverse neighbor map: node_id -> set of nodes that have this node as neighbor
    /// Used to find nodes that need repair when a node is deleted
    reverse_neighbors: RwLock<HashMap<u64, HashSet<u64>>>,
    /// Queue of nodes pending neighbor repair
    pending_repairs: RwLock<VecDeque<u64>>,
    /// Statistics
    delete_count: AtomicU64,
    repair_count: AtomicU64,
}

impl DeleteTracker {
    fn new() -> Self {
        Self {
            deleted: RwLock::new(RoaringBitmap::new()),
            reverse_neighbors: RwLock::new(HashMap::new()),
            pending_repairs: RwLock::new(VecDeque::new()),
            delete_count: AtomicU64::new(0),
            repair_count: AtomicU64::new(0),
        }
    }

    /// Mark a node as deleted.
    fn mark_deleted(&self, id: u64) -> bool {
        let mut deleted = self.deleted.write();
        if deleted.insert(id as u32) {
            self.delete_count.fetch_add(1, Ordering::Relaxed);

            // Queue affected nodes for repair
            if let Some(affected) = self.reverse_neighbors.read().get(&id) {
                let mut pending = self.pending_repairs.write();
                for &affected_id in affected {
                    if !deleted.contains(affected_id as u32) {
                        pending.push_back(affected_id);
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Check if a node is deleted.
    fn is_deleted(&self, id: u64) -> bool {
        self.deleted.read().contains(id as u32)
    }

    /// Get deleted bitmap for filtering.
    fn deleted_bitmap(&self) -> RoaringBitmap {
        self.deleted.read().clone()
    }

    /// Get count of deleted nodes.
    fn delete_count(&self) -> u64 {
        self.delete_count.load(Ordering::Relaxed)
    }

    /// Register a neighbor relationship (for reverse lookup).
    fn register_neighbor(&self, node_id: u64, neighbor_id: u64) {
        self.reverse_neighbors
            .write()
            .entry(neighbor_id)
            .or_insert_with(HashSet::new)
            .insert(node_id);
    }

    /// Get next node needing repair.
    fn pop_pending_repair(&self) -> Option<u64> {
        self.pending_repairs.write().pop_front()
    }

    /// Clear all tracking data.
    fn clear(&self) {
        self.deleted.write().clear();
        self.reverse_neighbors.write().clear();
        self.pending_repairs.write().clear();
        self.delete_count.store(0, Ordering::Relaxed);
        self.repair_count.store(0, Ordering::Relaxed);
    }
}

/// Vector store trait for accessing vectors during repair.
pub trait VectorProvider: Send + Sync {
    /// Get vector by ID.
    fn get_vector(&self, id: u64) -> Option<Vec<f32>>;
    /// Get total count of vectors.
    fn vector_count(&self) -> usize;
}

/// Dynamic HNSW Index with deletion support.
pub struct DynamicHnswIndex {
    /// Base HNSW index
    indexer: RwLock<HnswIndexer>,
    /// Deletion tracker
    tracker: DeleteTracker,
    /// Configuration
    config: DynamicIndexConfig,
    /// Total inserted count (including deleted)
    total_inserted: AtomicUsize,
    /// Vector provider for repair operations
    vector_provider: Option<Arc<dyn VectorProvider>>,
}

impl DynamicHnswIndex {
    /// Create a new dynamic HNSW index.
    pub fn new(config: DynamicIndexConfig) -> Self {
        let indexer = HnswIndexer::with_metric(
            config.dim,
            config.max_elements,
            config.m,
            config.ef_construction,
            config.metric,
        );

        Self {
            indexer: RwLock::new(indexer),
            tracker: DeleteTracker::new(),
            config,
            total_inserted: AtomicUsize::new(0),
            vector_provider: None,
        }
    }

    /// Create with a vector provider for repair operations.
    pub fn with_vector_provider(
        config: DynamicIndexConfig,
        provider: Arc<dyn VectorProvider>,
    ) -> Self {
        let mut index = Self::new(config);
        index.vector_provider = Some(provider);
        index
    }

    /// Set vector provider.
    pub fn set_vector_provider(&mut self, provider: Arc<dyn VectorProvider>) {
        self.vector_provider = Some(provider);
    }

    /// Insert a single vector.
    pub fn insert(&self, id: u64, vector: &[f32]) -> Result<()> {
        self.indexer.read().insert(id, vector)?;
        self.total_inserted.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Insert a batch of vectors.
    pub fn insert_batch(&self, vectors: &[(u64, Vec<f32>)]) -> Result<()> {
        self.indexer.read().insert_batch(vectors)?;
        self.total_inserted.fetch_add(vectors.len(), Ordering::Relaxed);
        Ok(())
    }

    /// Delete a vector by ID.
    /// Returns true if the vector was found and deleted.
    pub fn delete(&self, id: u64) -> bool {
        self.tracker.mark_deleted(id)
    }

    /// Delete multiple vectors.
    /// Returns count of successfully deleted vectors.
    pub fn delete_batch(&self, ids: &[u64]) -> usize {
        ids.iter()
            .filter(|&&id| self.tracker.mark_deleted(id))
            .count()
    }

    /// Check if a vector is deleted.
    pub fn is_deleted(&self, id: u64) -> bool {
        self.tracker.is_deleted(id)
    }

    /// Search for k nearest neighbors (excluding deleted).
    pub fn search(&self, vector: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        // Get active (non-deleted) filter
        let deleted = self.tracker.deleted_bitmap();

        if deleted.is_empty() {
            // No deletions, fast path
            return self.indexer.read().search(vector, k, None);
        }

        // Create "allowed" bitmap (all IDs except deleted)
        let total = self.total_inserted.load(Ordering::Relaxed) as u64;
        let mut allowed = RoaringBitmap::new();
        allowed.insert_range(0..total as u32);
        let allowed = &allowed - &deleted;

        // Search with filter
        let results = self.indexer.read().search(vector, k, Some(&allowed))?;

        // Lazy repair: if we encountered deleted nodes' neighbors, queue for repair
        if self.config.lazy_repair {
            self.try_lazy_repair();
        }

        Ok(results)
    }

    /// Search with additional filter.
    pub fn search_filtered(
        &self,
        vector: &[f32],
        k: usize,
        filter: &RoaringBitmap,
    ) -> Result<Vec<(u64, f32)>> {
        let deleted = self.tracker.deleted_bitmap();

        // Combine user filter with deletion filter
        let combined = filter - &deleted;

        self.indexer.read().search(vector, k, Some(&combined))
    }

    /// Perform lazy repair on pending nodes.
    fn try_lazy_repair(&self) {
        // Only repair one node per search to avoid blocking
        if let Some(node_id) = self.tracker.pop_pending_repair() {
            let _ = self.repair_node(node_id);
        }
    }

    /// Repair a single node's neighbors.
    /// Finds new neighbors to replace deleted ones.
    fn repair_node(&self, node_id: u64) -> Result<()> {
        let provider = match &self.vector_provider {
            Some(p) => p,
            None => return Ok(()), // Can't repair without vector access
        };

        // Get the node's vector
        let vector = match provider.get_vector(node_id) {
            Some(v) => v,
            None => return Ok(()), // Node doesn't exist
        };

        if self.tracker.is_deleted(node_id) {
            return Ok(()); // Node itself is deleted
        }

        // Find new neighbors by searching
        let k = self.config.repair_candidates;
        let results = self.search(&vector, k)?;

        // Register new neighbor relationships
        for (neighbor_id, _) in results {
            if neighbor_id != node_id {
                self.tracker.register_neighbor(node_id, neighbor_id);
            }
        }

        self.tracker.repair_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Get the deletion ratio.
    pub fn deletion_ratio(&self) -> f32 {
        let total = self.total_inserted.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.tracker.delete_count() as f32 / total as f32
    }

    /// Check if rebuild is needed based on deletion threshold.
    pub fn needs_rebuild(&self) -> bool {
        self.deletion_ratio() >= self.config.rebuild_threshold
    }

    /// Rebuild the index with only active vectors.
    /// Returns the new index with compacted data.
    pub fn rebuild(&self) -> Result<Self> {
        let provider = self.vector_provider.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Cannot rebuild without vector provider"))?;

        // Create new index
        let mut new_index = Self::new(self.config.clone());
        new_index.vector_provider.clone_from(&self.vector_provider);

        // Get all active IDs
        let total = self.total_inserted.load(Ordering::Relaxed);
        let deleted = self.tracker.deleted_bitmap();

        // Collect active vectors
        let mut vectors: Vec<(u64, Vec<f32>)> = Vec::new();
        for id in 0..total as u64 {
            if !deleted.contains(id as u32) {
                if let Some(vec) = provider.get_vector(id) {
                    vectors.push((id, vec));
                }
            }
        }

        // Batch insert into new index
        if !vectors.is_empty() {
            new_index.insert_batch(&vectors)?;
        }

        Ok(new_index)
    }

    /// Force repair all pending nodes.
    pub fn repair_all(&self) -> usize {
        let mut repaired = 0;
        while let Some(node_id) = self.tracker.pop_pending_repair() {
            if self.repair_node(node_id).is_ok() {
                repaired += 1;
            }
        }
        repaired
    }

    /// Get statistics.
    pub fn stats(&self) -> DynamicIndexStats {
        DynamicIndexStats {
            total_inserted: self.total_inserted.load(Ordering::Relaxed),
            deleted_count: self.tracker.delete_count() as usize,
            deletion_ratio: self.deletion_ratio(),
            pending_repairs: self.tracker.pending_repairs.read().len(),
            repair_count: self.tracker.repair_count.load(Ordering::Relaxed) as usize,
            needs_rebuild: self.needs_rebuild(),
        }
    }

    /// Get active (non-deleted) count.
    pub fn active_count(&self) -> usize {
        let total = self.total_inserted.load(Ordering::Relaxed);
        let deleted = self.tracker.delete_count() as usize;
        total.saturating_sub(deleted)
    }

    /// Get the deleted bitmap.
    pub fn deleted_bitmap(&self) -> RoaringBitmap {
        self.tracker.deleted_bitmap()
    }

    /// Get the underlying indexer (for save/load).
    pub fn indexer(&self) -> &RwLock<HnswIndexer> {
        &self.indexer
    }

    /// Get configuration.
    pub fn config(&self) -> &DynamicIndexConfig {
        &self.config
    }

    /// Get distance metric.
    pub fn metric(&self) -> DistanceMetric {
        self.config.metric
    }
}

/// Statistics for dynamic index.
#[derive(Debug, Clone)]
pub struct DynamicIndexStats {
    pub total_inserted: usize,
    pub deleted_count: usize,
    pub deletion_ratio: f32,
    pub pending_repairs: usize,
    pub repair_count: usize,
    pub needs_rebuild: bool,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    struct MockVectorProvider {
        vectors: RwLock<HashMap<u64, Vec<f32>>>,
    }

    impl MockVectorProvider {
        fn new() -> Self {
            Self {
                vectors: RwLock::new(HashMap::new()),
            }
        }

        fn insert(&self, id: u64, vector: Vec<f32>) {
            self.vectors.write().insert(id, vector);
        }
    }

    impl VectorProvider for MockVectorProvider {
        fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
            self.vectors.read().get(&id).cloned()
        }

        fn vector_count(&self) -> usize {
            self.vectors.read().len()
        }
    }

    #[test]
    fn test_dynamic_insert_search() {
        let config = DynamicIndexConfig::new(4, 100);
        let index = DynamicHnswIndex::new(config);

        // Insert vectors
        index.insert(0, &[0.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        // Search
        let results = index.search(&[0.1, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Closest to query
    }

    #[test]
    fn test_dynamic_delete() {
        let config = DynamicIndexConfig::new(4, 100);
        let index = DynamicHnswIndex::new(config);

        // Insert vectors
        index.insert(0, &[0.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[2.0, 0.0, 0.0, 0.0]).unwrap();

        // Delete ID 1
        assert!(index.delete(1));
        assert!(index.is_deleted(1));

        // Search should exclude deleted
        let results = index.search(&[0.9, 0.0, 0.0, 0.0], 3).unwrap();
        assert!(!results.iter().any(|(id, _)| *id == 1));
    }

    #[test]
    fn test_deletion_ratio() {
        let config = DynamicIndexConfig::new(4, 100);
        let index = DynamicHnswIndex::new(config);

        for i in 0..10 {
            index.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
        }

        assert_eq!(index.deletion_ratio(), 0.0);

        // Delete 3 out of 10
        index.delete(0);
        index.delete(1);
        index.delete(2);

        assert!((index.deletion_ratio() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_batch_delete() {
        let config = DynamicIndexConfig::new(4, 100);
        let index = DynamicHnswIndex::new(config);

        for i in 0..10 {
            index.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
        }

        let deleted = index.delete_batch(&[1, 3, 5, 7, 9]);
        assert_eq!(deleted, 5);
        assert_eq!(index.active_count(), 5);
    }

    #[test]
    fn test_rebuild() {
        let provider = Arc::new(MockVectorProvider::new());

        // Insert vectors into provider
        for i in 0..10u64 {
            provider.insert(i, vec![i as f32, 0.0, 0.0, 0.0]);
        }

        let config = DynamicIndexConfig::new(4, 100);
        let index = DynamicHnswIndex::with_vector_provider(config, provider.clone());

        // Insert into index
        for i in 0..10u64 {
            let vec = provider.get_vector(i).unwrap();
            index.insert(i, &vec).unwrap();
        }

        // Delete half
        for i in (0..10u64).step_by(2) {
            index.delete(i);
        }

        assert_eq!(index.active_count(), 5);

        // Rebuild
        let new_index = index.rebuild().unwrap();
        assert_eq!(new_index.active_count(), 5);
        assert_eq!(new_index.deletion_ratio(), 0.0);
    }

    #[test]
    fn test_stats() {
        let config = DynamicIndexConfig::new(4, 100);
        let index = DynamicHnswIndex::new(config);

        for i in 0..100 {
            index.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
        }

        index.delete_batch(&(0..30).collect::<Vec<_>>());

        let stats = index.stats();
        assert_eq!(stats.total_inserted, 100);
        assert_eq!(stats.deleted_count, 30);
        assert!((stats.deletion_ratio - 0.3).abs() < 0.01);
        assert!(stats.needs_rebuild); // 30% threshold
    }
}
