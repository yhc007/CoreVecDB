//! Upper Layer Caching for HNSW Index.
//!
//! Caches entry points and region seeds for faster search initialization.
//! Since the hnsw_rs crate manages its own graph structure, this cache
//! provides hints for better starting points based on query history.
//!
//! # Concept
//! - **Entry Points**: Top-layer nodes that serve as starting points for search
//! - **Region Seeds**: Clusters of similar vectors that can accelerate search
//!   for queries in that region

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use lru::LruCache;
use parking_lot::RwLock;
use std::num::NonZeroUsize;

/// Hash of a vector region (based on first few dimensions).
type RegionHash = u64;

/// Configuration for layer caching.
#[derive(Debug, Clone)]
pub struct LayerCacheConfig {
    /// Maximum number of entry points to cache.
    pub max_entry_points: usize,
    /// Maximum number of region seeds to cache.
    pub max_regions: usize,
    /// Number of dimensions to use for region hashing.
    pub region_hash_dims: usize,
    /// Number of buckets per dimension for region hashing.
    pub region_buckets: usize,
}

impl Default for LayerCacheConfig {
    fn default() -> Self {
        Self {
            max_entry_points: 100,
            max_regions: 1000,
            region_hash_dims: 8, // Use first 8 dimensions
            region_buckets: 10,  // 10 buckets per dimension
        }
    }
}

/// Entry point information.
#[derive(Debug, Clone)]
pub struct EntryPoint {
    /// Vector ID.
    pub id: u64,
    /// Number of times this entry point led to good results.
    pub success_count: u64,
    /// Last access timestamp.
    pub last_access: u64,
}

/// Region seed information.
#[derive(Debug, Clone)]
pub struct RegionSeed {
    /// Best starting vector IDs for this region.
    pub seed_ids: Vec<u64>,
    /// Number of successful searches from this region.
    pub hit_count: u64,
}

/// Upper layer cache for HNSW.
pub struct HnswLayerCache {
    config: LayerCacheConfig,
    /// Global entry points (good starting points regardless of query).
    entry_points: RwLock<Vec<EntryPoint>>,
    /// Region-specific seeds (good starting points for queries in that region).
    region_seeds: RwLock<LruCache<RegionHash, RegionSeed>>,
    /// Access counter for timestamps.
    access_counter: AtomicU64,
    /// Statistics.
    hits: AtomicU64,
    misses: AtomicU64,
}

impl HnswLayerCache {
    /// Create a new layer cache.
    pub fn new(config: LayerCacheConfig) -> Self {
        let region_size = NonZeroUsize::new(config.max_regions)
            .unwrap_or(NonZeroUsize::new(1000).unwrap());

        Self {
            config,
            entry_points: RwLock::new(Vec::new()),
            region_seeds: RwLock::new(LruCache::new(region_size)),
            access_counter: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Hash a vector to a region.
    fn hash_region(&self, vector: &[f32]) -> RegionHash {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Use first N dimensions for region hashing
        let dims = self.config.region_hash_dims.min(vector.len());
        for i in 0..dims {
            // Quantize to bucket
            let bucket = ((vector[i] + 1.0) * self.config.region_buckets as f32 / 2.0) as i32;
            bucket.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Get suggested starting points for a query.
    pub fn get_starting_points(&self, query: &[f32], max_points: usize) -> Vec<u64> {
        let mut points = Vec::with_capacity(max_points);

        // Check region-specific seeds first
        let region_hash = self.hash_region(query);
        {
            let mut cache = self.region_seeds.write();
            if let Some(seed) = cache.get(&region_hash) {
                points.extend(seed.seed_ids.iter().take(max_points / 2));
                self.hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.misses.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Add global entry points
        {
            let entry_points = self.entry_points.read();
            let remaining = max_points.saturating_sub(points.len());
            for ep in entry_points.iter().take(remaining) {
                if !points.contains(&ep.id) {
                    points.push(ep.id);
                }
            }
        }

        points
    }

    /// Record a successful search result to update cache.
    pub fn record_success(
        &self,
        query: &[f32],
        results: &[(u64, f32)],
        entry_point_used: Option<u64>,
    ) {
        if results.is_empty() {
            return;
        }

        let timestamp = self.access_counter.fetch_add(1, Ordering::Relaxed);

        // Update entry point success if one was used
        if let Some(ep_id) = entry_point_used {
            let mut entry_points = self.entry_points.write();
            if let Some(ep) = entry_points.iter_mut().find(|e| e.id == ep_id) {
                ep.success_count += 1;
                ep.last_access = timestamp;
            }
        }

        // Update region seeds with top results
        let region_hash = self.hash_region(query);
        let seed_ids: Vec<u64> = results.iter().take(5).map(|(id, _)| *id).collect();

        let mut cache = self.region_seeds.write();
        if let Some(seed) = cache.get_mut(&region_hash) {
            seed.hit_count += 1;
            // Merge with existing seeds, keeping most common
            for id in &seed_ids {
                if !seed.seed_ids.contains(id) {
                    if seed.seed_ids.len() < 10 {
                        seed.seed_ids.push(*id);
                    }
                }
            }
        } else {
            cache.put(
                region_hash,
                RegionSeed {
                    seed_ids,
                    hit_count: 1,
                },
            );
        }
    }

    /// Add a global entry point.
    pub fn add_entry_point(&self, id: u64) {
        let timestamp = self.access_counter.fetch_add(1, Ordering::Relaxed);

        let mut entry_points = self.entry_points.write();

        // Check if already exists
        if entry_points.iter().any(|e| e.id == id) {
            return;
        }

        // Add new entry point
        entry_points.push(EntryPoint {
            id,
            success_count: 0,
            last_access: timestamp,
        });

        // Trim if too many
        if entry_points.len() > self.config.max_entry_points {
            // Sort by success count and keep top N
            entry_points.sort_by(|a, b| b.success_count.cmp(&a.success_count));
            entry_points.truncate(self.config.max_entry_points);
        }
    }

    /// Clear all cached data.
    pub fn clear(&self) {
        self.entry_points.write().clear();
        self.region_seeds.write().clear();
    }

    /// Get cache statistics.
    pub fn stats(&self) -> LayerCacheStats {
        let entry_points = self.entry_points.read();
        let region_seeds = self.region_seeds.read();

        LayerCacheStats {
            entry_point_count: entry_points.len(),
            region_count: region_seeds.len(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }
}

/// Statistics for layer cache.
#[derive(Debug, Clone)]
pub struct LayerCacheStats {
    pub entry_point_count: usize,
    pub region_count: usize,
    pub hits: u64,
    pub misses: u64,
}

impl LayerCacheStats {
    /// Get hit rate as percentage.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_cache_basic() {
        let cache = HnswLayerCache::new(LayerCacheConfig::default());

        // Add some entry points
        cache.add_entry_point(0);
        cache.add_entry_point(1);
        cache.add_entry_point(2);

        let stats = cache.stats();
        assert_eq!(stats.entry_point_count, 3);
    }

    #[test]
    fn test_region_hashing() {
        let config = LayerCacheConfig {
            region_hash_dims: 4,
            region_buckets: 10,
            ..Default::default()
        };
        let cache = HnswLayerCache::new(config);

        // Similar vectors should hash to same region
        let v1 = vec![0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.51, 0.49, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]; // Similar first 4 dims

        let h1 = cache.hash_region(&v1);
        let h2 = cache.hash_region(&v2);

        // Should be same or very similar
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_get_starting_points_miss() {
        let cache = HnswLayerCache::new(LayerCacheConfig::default());

        cache.add_entry_point(0);
        cache.add_entry_point(1);

        let query = vec![0.5; 128];
        let points = cache.get_starting_points(&query, 5);

        // Should return entry points on miss
        assert!(!points.is_empty());
        assert!(points.contains(&0) || points.contains(&1));

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_record_success() {
        let cache = HnswLayerCache::new(LayerCacheConfig::default());

        let query = vec![0.5; 128];
        let results = vec![(10, 0.1), (20, 0.2), (30, 0.3)];

        // Record a successful search
        cache.record_success(&query, &results, None);

        // Should now have cached region
        let points = cache.get_starting_points(&query, 5);
        assert!(points.contains(&10));

        let stats = cache.stats();
        assert_eq!(stats.region_count, 1);
        assert_eq!(stats.hits, 1);
    }
}
