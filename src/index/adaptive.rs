//! Adaptive Index Selection for VectorDB.
//!
//! Automatically switches between brute-force and HNSW based on collection size.
//! - < 2000 vectors: Brute-force (SIMD-accelerated linear scan)
//! - >= 2000 vectors: HNSW (approximate nearest neighbor)
//!
//! The brute-force search is faster for small datasets due to lower overhead,
//! while HNSW scales better for large datasets.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use parking_lot::RwLock;
use roaring::RoaringBitmap;

use crate::index::{HnswIndexer, DistanceMetric};
use crate::simd;
use crate::storage::VectorStore;

/// Default threshold for switching from brute-force to HNSW.
pub const DEFAULT_BRUTE_FORCE_THRESHOLD: usize = 2000;

/// Index type used for search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    /// Linear scan with SIMD acceleration.
    BruteForce,
    /// Hierarchical Navigable Small World graph.
    Hnsw,
}

/// Configuration for adaptive index.
#[derive(Debug, Clone)]
pub struct AdaptiveIndexConfig {
    /// Enable adaptive index selection.
    pub enabled: bool,
    /// Threshold for switching to HNSW (number of vectors).
    pub brute_force_threshold: usize,
    /// Distance metric.
    pub distance_metric: DistanceMetric,
    /// HNSW max elements.
    pub max_elements: usize,
    /// HNSW m parameter.
    pub m: usize,
    /// HNSW ef_construction parameter.
    pub ef_construction: usize,
}

impl Default for AdaptiveIndexConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            brute_force_threshold: DEFAULT_BRUTE_FORCE_THRESHOLD,
            distance_metric: DistanceMetric::Euclidean,
            max_elements: 100_000,
            m: 24,
            ef_construction: 400,
        }
    }
}

/// Adaptive indexer that switches between brute-force and HNSW.
pub struct AdaptiveIndexer {
    config: AdaptiveIndexConfig,
    /// Current index type in use.
    current_type: RwLock<IndexType>,
    /// HNSW index (lazily initialized when threshold is crossed).
    hnsw: RwLock<Option<Arc<HnswIndexer>>>,
    /// Vector store reference for brute-force search.
    vector_store: Arc<dyn VectorStore>,
    /// Current vector count (tracked for threshold detection).
    vector_count: AtomicUsize,
    /// Dimension of vectors.
    dim: usize,
}

impl AdaptiveIndexer {
    /// Create a new adaptive indexer.
    pub fn new(
        config: AdaptiveIndexConfig,
        vector_store: Arc<dyn VectorStore>,
        dim: usize,
    ) -> Self {
        let current_count = vector_store.len();
        let index_type = if config.enabled && current_count < config.brute_force_threshold {
            IndexType::BruteForce
        } else {
            IndexType::Hnsw
        };

        Self {
            config,
            current_type: RwLock::new(index_type),
            hnsw: RwLock::new(None),
            vector_store,
            vector_count: AtomicUsize::new(current_count),
            dim,
        }
    }

    /// Create with an existing HNSW index.
    pub fn with_hnsw(
        config: AdaptiveIndexConfig,
        vector_store: Arc<dyn VectorStore>,
        hnsw: Arc<HnswIndexer>,
        dim: usize,
    ) -> Self {
        let current_count = vector_store.len();
        let index_type = if config.enabled && current_count < config.brute_force_threshold {
            IndexType::BruteForce
        } else {
            IndexType::Hnsw
        };

        Self {
            config,
            current_type: RwLock::new(index_type),
            hnsw: RwLock::new(Some(hnsw)),
            vector_store,
            vector_count: AtomicUsize::new(current_count),
            dim,
        }
    }

    /// Get current index type.
    pub fn current_type(&self) -> IndexType {
        *self.current_type.read()
    }

    /// Check if HNSW is initialized.
    pub fn has_hnsw(&self) -> bool {
        self.hnsw.read().is_some()
    }

    /// Get HNSW index reference.
    pub fn hnsw(&self) -> Option<Arc<HnswIndexer>> {
        self.hnsw.read().clone()
    }

    /// Insert a vector. Returns true if index type switched.
    pub fn insert(&self, id: u64, vector: &[f32]) -> Result<bool> {
        let new_count = self.vector_count.fetch_add(1, Ordering::SeqCst) + 1;
        let mut switched = false;

        // Check if we need to switch to HNSW
        if self.config.enabled && new_count >= self.config.brute_force_threshold {
            let current = *self.current_type.read();
            if current == IndexType::BruteForce {
                // Need to build HNSW index
                switched = self.build_hnsw_index()?;
            }
        }

        // If HNSW exists, insert into it
        if let Some(ref hnsw) = *self.hnsw.read() {
            hnsw.insert(id, vector)?;
        }

        Ok(switched)
    }

    /// Batch insert vectors.
    pub fn insert_batch(&self, vectors: &[(u64, Vec<f32>)]) -> Result<bool> {
        if vectors.is_empty() {
            return Ok(false);
        }

        let new_count = self.vector_count.fetch_add(vectors.len(), Ordering::SeqCst) + vectors.len();
        let mut switched = false;

        // Check if we need to switch to HNSW
        if self.config.enabled && new_count >= self.config.brute_force_threshold {
            let current = *self.current_type.read();
            if current == IndexType::BruteForce {
                switched = self.build_hnsw_index()?;
            }
        }

        // If HNSW exists, insert into it
        if let Some(ref hnsw) = *self.hnsw.read() {
            hnsw.insert_batch(vectors)?;
        }

        Ok(switched)
    }

    /// Build HNSW index from current vectors.
    fn build_hnsw_index(&self) -> Result<bool> {
        let mut current_type = self.current_type.write();

        // Double-check after acquiring write lock
        if *current_type == IndexType::Hnsw {
            return Ok(false);
        }

        // Check if HNSW already exists
        if self.hnsw.read().is_some() {
            *current_type = IndexType::Hnsw;
            return Ok(true);
        }

        // Build new HNSW index
        let hnsw = HnswIndexer::with_metric(
            self.dim,
            self.config.max_elements,
            self.config.m,
            self.config.ef_construction,
            self.config.distance_metric,
        );

        // Insert all existing vectors
        let count = self.vector_store.len();
        let mut vectors_with_ids = Vec::with_capacity(count);

        for id in 0..count as u64 {
            if let Ok(vector) = self.vector_store.get(id) {
                vectors_with_ids.push((id, vector));
            }
        }

        if !vectors_with_ids.is_empty() {
            hnsw.insert_batch(&vectors_with_ids)?;
        }

        *self.hnsw.write() = Some(Arc::new(hnsw));
        *current_type = IndexType::Hnsw;

        Ok(true)
    }

    /// Search for k nearest neighbors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&RoaringBitmap>,
    ) -> Result<Vec<(u64, f32)>> {
        let current_type = *self.current_type.read();

        match current_type {
            IndexType::BruteForce => self.brute_force_search(query, k, filter),
            IndexType::Hnsw => {
                if let Some(ref hnsw) = *self.hnsw.read() {
                    hnsw.search(query, k, filter)
                } else {
                    // Fallback to brute-force if HNSW not ready
                    self.brute_force_search(query, k, filter)
                }
            }
        }
    }

    /// SIMD-accelerated brute-force search.
    fn brute_force_search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&RoaringBitmap>,
    ) -> Result<Vec<(u64, f32)>> {
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        // Min-heap to keep track of top-k
        #[derive(PartialEq)]
        struct Candidate {
            id: u64,
            distance: f32,
        }

        impl Eq for Candidate {}

        impl PartialOrd for Candidate {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                // Max-heap by distance - larger distances at top to be popped
                // This keeps the k smallest distances in the heap
                self.distance.partial_cmp(&other.distance)
            }
        }

        impl Ord for Candidate {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        let count = self.vector_store.len();
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);

        for id in 0..count as u64 {
            // Apply filter if provided
            if let Some(filter_bitmap) = filter {
                if !filter_bitmap.contains(id as u32) {
                    continue;
                }
            }

            // Get vector from store
            let vector = match self.vector_store.get(id) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Compute distance using SIMD
            let distance = match self.config.distance_metric {
                DistanceMetric::Euclidean => simd::l2_squared(query, &vector),
                DistanceMetric::Cosine => simd::cosine_distance(query, &vector),
                DistanceMetric::DotProduct => -simd::dot_product(query, &vector), // Negative for max similarity
            };

            // Add to heap
            heap.push(Candidate { id, distance });

            // Keep only top-k
            if heap.len() > k {
                heap.pop();
            }
        }

        // Convert to result vector (sorted by distance)
        let mut results: Vec<(u64, f32)> = heap
            .into_iter()
            .map(|c| (c.id, c.distance))
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        Ok(results)
    }

    /// Get index statistics.
    pub fn stats(&self) -> AdaptiveIndexStats {
        AdaptiveIndexStats {
            current_type: *self.current_type.read(),
            vector_count: self.vector_count.load(Ordering::SeqCst),
            brute_force_threshold: self.config.brute_force_threshold,
            hnsw_initialized: self.hnsw.read().is_some(),
            adaptive_enabled: self.config.enabled,
        }
    }
}

/// Statistics for adaptive indexer.
#[derive(Debug, Clone)]
pub struct AdaptiveIndexStats {
    pub current_type: IndexType,
    pub vector_count: usize,
    pub brute_force_threshold: usize,
    pub hnsw_initialized: bool,
    pub adaptive_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::MemmapVectorStore;
    use tempfile::TempDir;

    fn create_test_store(dim: usize) -> (Arc<dyn VectorStore>, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("vectors.bin");
        let store = MemmapVectorStore::new(path.to_str().unwrap(), dim).unwrap();
        (Arc::new(store), temp_dir)
    }

    #[test]
    fn test_adaptive_starts_with_brute_force() {
        let (store, _tmp) = create_test_store(128);
        let config = AdaptiveIndexConfig {
            enabled: true,
            brute_force_threshold: 100,
            ..Default::default()
        };

        let indexer = AdaptiveIndexer::new(config, store, 128);
        assert_eq!(indexer.current_type(), IndexType::BruteForce);
        assert!(!indexer.has_hnsw());
    }

    #[test]
    fn test_brute_force_search() {
        let (store, _tmp) = create_test_store(4);
        let config = AdaptiveIndexConfig {
            enabled: true,
            brute_force_threshold: 1000, // Won't switch
            ..Default::default()
        };

        // Insert some vectors
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.5, 0.5, 0.0, 0.0],
        ];

        for v in &vectors {
            store.insert(v).unwrap();
        }

        let indexer = AdaptiveIndexer::new(config, store, 4);

        // Search for nearest to [1, 0, 0, 0]
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = indexer.search(&query, 2, None).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Should be exact match
        assert!(results[0].1 < 0.001); // Distance should be ~0
    }

    #[test]
    fn test_switch_to_hnsw_on_threshold() {
        let (store, _tmp) = create_test_store(4);
        let config = AdaptiveIndexConfig {
            enabled: true,
            brute_force_threshold: 5,
            ..Default::default()
        };

        let indexer = AdaptiveIndexer::new(config, store.clone(), 4);
        assert_eq!(indexer.current_type(), IndexType::BruteForce);

        // Insert vectors below threshold
        for i in 0..4 {
            let v = vec![i as f32; 4];
            store.insert(&v).unwrap();
            indexer.insert(i as u64, &v).unwrap();
        }
        assert_eq!(indexer.current_type(), IndexType::BruteForce);

        // Insert one more to cross threshold
        let v = vec![4.0; 4];
        store.insert(&v).unwrap();
        let switched = indexer.insert(4, &v).unwrap();

        assert!(switched);
        assert_eq!(indexer.current_type(), IndexType::Hnsw);
        assert!(indexer.has_hnsw());
    }

    #[test]
    fn test_search_with_filter() {
        let (store, _tmp) = create_test_store(4);
        let config = AdaptiveIndexConfig {
            enabled: true,
            brute_force_threshold: 1000,
            ..Default::default()
        };

        // Insert vectors
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        for v in &vectors {
            store.insert(v).unwrap();
        }

        let indexer = AdaptiveIndexer::new(config, store, 4);

        // Create filter that only includes IDs 0 and 2
        let mut filter = RoaringBitmap::new();
        filter.insert(0);
        filter.insert(2);

        let query = vec![0.5, 0.5, 0.0, 0.0];
        let results = indexer.search(&query, 4, Some(&filter)).unwrap();

        // Should only return 2 results (filtered)
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(id, _)| *id == 0 || *id == 2));
    }
}
