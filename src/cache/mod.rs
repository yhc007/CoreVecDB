//! Cache module for VectorDB performance optimization.
//!
//! Provides:
//! - Query result caching for repeated searches
//! - Filter bitmap caching for frequent filter conditions
//!
//! All caches are LRU-based with configurable size and TTL.

pub mod query_cache;
pub mod filter_cache;

pub use query_cache::{QueryCache, QueryCacheKey, QueryCacheStats};
pub use filter_cache::{FilterBitmapCache, FilterCacheStats};
