//! Query result cache for VectorDB.
//!
//! Caches search results based on query vector hash, k, and filter conditions.
//! Uses LRU eviction with optional TTL-based expiration.
//!
//! # Example
//! ```rust,ignore
//! let cache = QueryCache::new(1000, Duration::from_secs(60));
//!
//! // Check cache before search
//! if let Some(results) = cache.get(&key) {
//!     return results;
//! }
//!
//! // Perform search and cache result
//! let results = indexer.search(&query, k, filter)?;
//! cache.put(key, results.clone(), vector_count);
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use lru::LruCache;
use parking_lot::RwLock;
use std::num::NonZeroUsize;

/// Key for query cache lookup.
/// Combines collection name, query vector hash, k, and filter hash.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct QueryCacheKey {
    pub collection: String,
    pub vector_hash: u64,
    pub k: usize,
    pub filter_hash: u64,
}

impl QueryCacheKey {
    /// Create a new cache key from query parameters.
    pub fn new(
        collection: &str,
        query_vector: &[f32],
        k: usize,
        filter: Option<&HashMap<String, String>>,
    ) -> Self {
        Self {
            collection: collection.to_string(),
            vector_hash: Self::hash_vector(query_vector),
            k,
            filter_hash: Self::hash_filter(filter),
        }
    }

    /// Hash a vector using xxHash-style fast hashing.
    /// Uses first, middle, and last elements + length for quick hash.
    fn hash_vector(vector: &[f32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();

        // Hash length
        vector.len().hash(&mut hasher);

        if !vector.is_empty() {
            // Hash first element
            vector[0].to_bits().hash(&mut hasher);

            // Hash middle element
            let mid = vector.len() / 2;
            vector[mid].to_bits().hash(&mut hasher);

            // Hash last element
            vector[vector.len() - 1].to_bits().hash(&mut hasher);

            // Hash a few more samples for better distribution
            if vector.len() > 10 {
                vector[vector.len() / 4].to_bits().hash(&mut hasher);
                vector[3 * vector.len() / 4].to_bits().hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Hash filter conditions for cache key.
    fn hash_filter(filter: Option<&HashMap<String, String>>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();

        match filter {
            None => 0u64.hash(&mut hasher),
            Some(f) if f.is_empty() => 0u64.hash(&mut hasher),
            Some(f) => {
                // Sort keys for consistent hashing
                let mut pairs: Vec<_> = f.iter().collect();
                pairs.sort_by_key(|(k, _)| *k);
                for (k, v) in pairs {
                    k.hash(&mut hasher);
                    v.hash(&mut hasher);
                }
            }
        }

        hasher.finish()
    }
}

/// Cached search result entry.
struct CacheEntry {
    results: Vec<(u64, f32)>,
    created_at: Instant,
    vector_count_at_cache: usize,
}

/// LRU cache for query results with TTL support.
pub struct QueryCache {
    cache: RwLock<LruCache<QueryCacheKey, CacheEntry>>,
    ttl: Duration,
    hits: AtomicU64,
    misses: AtomicU64,
    invalidations: AtomicU64,
}

impl QueryCache {
    /// Create a new query cache with specified size and TTL.
    pub fn new(max_size: usize, ttl: Duration) -> Self {
        let size = NonZeroUsize::new(max_size).unwrap_or(NonZeroUsize::new(1000).unwrap());
        Self {
            cache: RwLock::new(LruCache::new(size)),
            ttl,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
        }
    }

    /// Get cached results if valid.
    /// Returns None if not found, expired, or collection state changed.
    pub fn get(&self, key: &QueryCacheKey, current_vector_count: usize) -> Option<Vec<(u64, f32)>> {
        let mut cache = self.cache.write();

        if let Some(entry) = cache.get(key) {
            // Check TTL
            if entry.created_at.elapsed() > self.ttl {
                cache.pop(key);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            // Check if collection state changed (new vectors inserted)
            if entry.vector_count_at_cache != current_vector_count {
                cache.pop(key);
                self.invalidations.fetch_add(1, Ordering::Relaxed);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            self.hits.fetch_add(1, Ordering::Relaxed);
            return Some(entry.results.clone());
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Put search results into cache.
    pub fn put(&self, key: QueryCacheKey, results: Vec<(u64, f32)>, vector_count: usize) {
        let entry = CacheEntry {
            results,
            created_at: Instant::now(),
            vector_count_at_cache: vector_count,
        };

        self.cache.write().put(key, entry);
    }

    /// Invalidate all entries for a collection.
    /// Called when vectors are inserted or deleted.
    pub fn invalidate_collection(&self, collection: &str) {
        let mut cache = self.cache.write();
        let keys_to_remove: Vec<_> = cache
            .iter()
            .filter(|(k, _)| k.collection == collection)
            .map(|(k, _)| k.clone())
            .collect();

        for key in keys_to_remove {
            cache.pop(&key);
            self.invalidations.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Clear all cache entries.
    pub fn clear(&self) {
        self.cache.write().clear();
    }

    /// Get cache statistics.
    pub fn stats(&self) -> QueryCacheStats {
        let cache = self.cache.read();
        QueryCacheStats {
            size: cache.len(),
            capacity: cache.cap().get(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            invalidations: self.invalidations.load(Ordering::Relaxed),
        }
    }

    /// Get hit rate as percentage.
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            (hits as f64 / total as f64) * 100.0
        }
    }
}

/// Statistics for query cache.
#[derive(Debug, Clone)]
pub struct QueryCacheStats {
    pub size: usize,
    pub capacity: usize,
    pub hits: u64,
    pub misses: u64,
    pub invalidations: u64,
}

impl QueryCacheStats {
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
    fn test_query_cache_basic() {
        let cache = QueryCache::new(100, Duration::from_secs(60));

        let key = QueryCacheKey::new("test", &[1.0, 2.0, 3.0], 10, None);
        let results = vec![(0, 0.5), (1, 0.7), (2, 0.9)];

        // Miss on first access
        assert!(cache.get(&key, 100).is_none());
        assert_eq!(cache.stats().misses, 1);

        // Put and get
        cache.put(key.clone(), results.clone(), 100);
        let cached = cache.get(&key, 100);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), results);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_query_cache_invalidation_on_insert() {
        let cache = QueryCache::new(100, Duration::from_secs(60));

        let key = QueryCacheKey::new("test", &[1.0, 2.0, 3.0], 10, None);
        let results = vec![(0, 0.5)];

        // Cache with vector_count = 100
        cache.put(key.clone(), results, 100);

        // Should miss when vector_count changes
        assert!(cache.get(&key, 101).is_none());
        assert_eq!(cache.stats().invalidations, 1);
    }

    #[test]
    fn test_query_cache_collection_invalidation() {
        let cache = QueryCache::new(100, Duration::from_secs(60));

        let key1 = QueryCacheKey::new("coll1", &[1.0], 10, None);
        let key2 = QueryCacheKey::new("coll2", &[1.0], 10, None);

        cache.put(key1.clone(), vec![(0, 0.5)], 100);
        cache.put(key2.clone(), vec![(0, 0.5)], 100);

        // Invalidate coll1
        cache.invalidate_collection("coll1");

        // coll1 should be gone, coll2 should remain
        assert!(cache.get(&key1, 100).is_none());
        assert!(cache.get(&key2, 100).is_some());
    }

    #[test]
    fn test_filter_hashing() {
        let mut filter1 = HashMap::new();
        filter1.insert("a".to_string(), "1".to_string());
        filter1.insert("b".to_string(), "2".to_string());

        let mut filter2 = HashMap::new();
        filter2.insert("b".to_string(), "2".to_string());
        filter2.insert("a".to_string(), "1".to_string());

        // Same filters in different order should produce same hash
        let key1 = QueryCacheKey::new("test", &[1.0], 10, Some(&filter1));
        let key2 = QueryCacheKey::new("test", &[1.0], 10, Some(&filter2));

        assert_eq!(key1.filter_hash, key2.filter_hash);
    }
}
