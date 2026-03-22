//! Filter bitmap cache for VectorDB.
//!
//! Caches computed RoaringBitmaps for frequently used filter conditions.
//! Reduces overhead of repeated filter lookups on PayloadIndex.
//!
//! # Example
//! ```rust,ignore
//! let cache = FilterBitmapCache::new(500);
//!
//! let conditions = vec![("category".to_string(), "electronics".to_string())];
//!
//! // Check cache before computing
//! if let Some(bitmap) = cache.get(&conditions) {
//!     return bitmap;
//! }
//!
//! // Compute and cache
//! let bitmap = payload_index.filter(&conditions);
//! cache.put(conditions, bitmap.clone());
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::num::NonZeroUsize;

use lru::LruCache;
use parking_lot::RwLock;
use roaring::RoaringBitmap;

/// Key for filter bitmap cache.
/// Vector of sorted (field, value) pairs for consistent hashing.
pub type FilterCacheKey = Vec<(String, String)>;

/// LRU cache for filter bitmaps.
pub struct FilterBitmapCache {
    cache: RwLock<LruCache<FilterCacheKey, RoaringBitmap>>,
    hits: AtomicU64,
    misses: AtomicU64,
    invalidations: AtomicU64,
}

impl FilterBitmapCache {
    /// Create a new filter bitmap cache with specified max size.
    pub fn new(max_size: usize) -> Self {
        let size = NonZeroUsize::new(max_size).unwrap_or(NonZeroUsize::new(500).unwrap());
        Self {
            cache: RwLock::new(LruCache::new(size)),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
        }
    }

    /// Normalize filter conditions for consistent cache key.
    /// Sorts by field name for order-independent matching.
    pub fn normalize_key(conditions: &[(&str, &str)]) -> FilterCacheKey {
        let mut key: Vec<_> = conditions
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        key.sort_by(|a, b| a.0.cmp(&b.0));
        key
    }

    /// Get cached bitmap for filter conditions.
    pub fn get(&self, key: &FilterCacheKey) -> Option<RoaringBitmap> {
        let mut cache = self.cache.write();
        if let Some(bitmap) = cache.get(key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(bitmap.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Put computed bitmap into cache.
    pub fn put(&self, key: FilterCacheKey, bitmap: RoaringBitmap) {
        self.cache.write().put(key, bitmap);
    }

    /// Invalidate entries containing a specific field.
    /// Called when metadata for that field is modified.
    pub fn invalidate_field(&self, field: &str) {
        let mut cache = self.cache.write();
        let keys_to_remove: Vec<_> = cache
            .iter()
            .filter(|(key, _)| key.iter().any(|(k, _)| k == field))
            .map(|(key, _)| key.clone())
            .collect();

        for key in keys_to_remove {
            cache.pop(&key);
            self.invalidations.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Invalidate all entries.
    pub fn invalidate_all(&self) {
        let mut cache = self.cache.write();
        let count = cache.len();
        cache.clear();
        self.invalidations.fetch_add(count as u64, Ordering::Relaxed);
    }

    /// Clear all cache entries.
    pub fn clear(&self) {
        self.cache.write().clear();
    }

    /// Get cache statistics.
    pub fn stats(&self) -> FilterCacheStats {
        let cache = self.cache.read();
        FilterCacheStats {
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

/// Statistics for filter bitmap cache.
#[derive(Debug, Clone)]
pub struct FilterCacheStats {
    pub size: usize,
    pub capacity: usize,
    pub hits: u64,
    pub misses: u64,
    pub invalidations: u64,
}

impl FilterCacheStats {
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
    fn test_filter_cache_basic() {
        let cache = FilterBitmapCache::new(100);

        let key = FilterBitmapCache::normalize_key(&[("category", "electronics")]);
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(1);
        bitmap.insert(5);
        bitmap.insert(10);

        // Miss on first access
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats().misses, 1);

        // Put and get
        cache.put(key.clone(), bitmap.clone());
        let cached = cache.get(&key);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 3);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_filter_cache_field_invalidation() {
        let cache = FilterBitmapCache::new(100);

        let key1 = FilterBitmapCache::normalize_key(&[("category", "electronics")]);
        let key2 = FilterBitmapCache::normalize_key(&[("status", "active")]);
        let key3 = FilterBitmapCache::normalize_key(&[("category", "books"), ("status", "active")]);

        let bitmap = RoaringBitmap::new();
        cache.put(key1.clone(), bitmap.clone());
        cache.put(key2.clone(), bitmap.clone());
        cache.put(key3.clone(), bitmap.clone());

        // Invalidate category field
        cache.invalidate_field("category");

        // category keys should be gone
        assert!(cache.get(&key1).is_none());
        assert!(cache.get(&key3).is_none());
        // status-only key should remain
        assert!(cache.get(&key2).is_some());
    }

    #[test]
    fn test_key_normalization() {
        // Different order should produce same key
        let key1 = FilterBitmapCache::normalize_key(&[("a", "1"), ("b", "2")]);
        let key2 = FilterBitmapCache::normalize_key(&[("b", "2"), ("a", "1")]);

        assert_eq!(key1, key2);
    }
}
