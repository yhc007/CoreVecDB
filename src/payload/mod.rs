//! Payload Index for efficient metadata filtering.
//!
//! Implements inverted indexes on metadata fields for O(1) lookups
//! instead of O(n) sequential scans.
//!
//! Uses Rust functional programming patterns throughout:
//! - Iterator combinators (map, filter, flat_map, fold)
//! - Option/Result monads
//! - Closures and higher-order functions
//! - Pattern matching

use parking_lot::RwLock;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Inverted index for a single field.
/// Maps field values to sets of vector IDs containing that value.
#[derive(Default)]
pub struct FieldIndex {
    /// value -> bitmap of IDs
    index: RwLock<HashMap<String, RoaringBitmap>>,
    /// Statistics
    cardinality: AtomicU64,
}

impl FieldIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a value for an ID.
    /// Functional: uses entry API with closures.
    pub fn insert(&self, id: u64, value: &str) {
        let mut index = self.index.write();
        index
            .entry(value.to_string())
            .or_insert_with(RoaringBitmap::new)
            .insert(id as u32);
    }

    /// Remove an ID from a value's bitmap.
    /// Functional: uses if-let chain with mutable reference.
    pub fn remove(&self, id: u64, value: &str) {
        let mut index = self.index.write();
        if let Some(bitmap) = index.get_mut(value) {
            bitmap.remove(id as u32);
            // Clean up empty bitmaps
            if bitmap.is_empty() {
                index.remove(value);
            }
        }
    }

    /// Get all IDs matching a value.
    /// Functional: returns cloned bitmap wrapped in Option.
    pub fn get(&self, value: &str) -> Option<RoaringBitmap> {
        self.index
            .read()
            .get(value)
            .cloned()
    }

    /// Get all IDs matching any of the values (OR).
    /// Functional: fold over iterator to union bitmaps.
    pub fn get_any<'a, I>(&self, values: I) -> RoaringBitmap
    where
        I: IntoIterator<Item = &'a str>,
    {
        let index = self.index.read();
        values
            .into_iter()
            .filter_map(|v| index.get(v))
            .fold(RoaringBitmap::new(), |acc, bitmap| &acc | bitmap)
    }

    /// Get all IDs matching all of the values (AND).
    /// Functional: reduce with intersection, short-circuit on empty.
    pub fn get_all<'a, I>(&self, values: I) -> Option<RoaringBitmap>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let index = self.index.read();
        values
            .into_iter()
            .filter_map(|v| index.get(v).cloned())
            .reduce(|acc, bitmap| &acc & &bitmap)
    }

    /// Get cardinality (number of unique values).
    pub fn cardinality(&self) -> usize {
        self.index.read().len()
    }

    /// Get all unique values.
    /// Functional: collects keys via iterator.
    pub fn values(&self) -> Vec<String> {
        self.index
            .read()
            .keys()
            .cloned()
            .collect()
    }

    /// Get value with highest cardinality (most IDs).
    /// Functional: max_by_key with closure.
    pub fn most_common(&self) -> Option<(String, u64)> {
        self.index
            .read()
            .iter()
            .max_by_key(|(_, bitmap)| bitmap.len())
            .map(|(k, v)| (k.clone(), v.len()))
    }

    /// Get statistics.
    /// Functional: fold to compute aggregates.
    pub fn stats(&self) -> FieldIndexStats {
        let index = self.index.read();
        let (total_ids, min_ids, max_ids) = index
            .values()
            .map(|b| b.len())
            .fold((0u64, u64::MAX, 0u64), |(sum, min, max), len| {
                (sum + len, min.min(len), max.max(len))
            });

        let unique_values = index.len();
        FieldIndexStats {
            unique_values,
            total_ids,
            avg_ids_per_value: if unique_values > 0 {
                total_ids as f64 / unique_values as f64
            } else {
                0.0
            },
            min_ids_per_value: if unique_values > 0 { min_ids } else { 0 },
            max_ids_per_value: max_ids,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FieldIndexStats {
    pub unique_values: usize,
    pub total_ids: u64,
    pub avg_ids_per_value: f64,
    pub min_ids_per_value: u64,
    pub max_ids_per_value: u64,
}

/// Payload index manager for multiple fields.
/// Provides efficient filtering across all indexed fields.
pub struct PayloadIndex {
    /// field_name -> FieldIndex
    fields: RwLock<HashMap<String, FieldIndex>>,
    /// Fields that are indexed
    indexed_fields: RwLock<Vec<String>>,
    /// Statistics
    stats: PayloadIndexStats,
}

#[derive(Default)]
struct PayloadIndexStats {
    inserts: AtomicU64,
    lookups: AtomicU64,
    cache_hits: AtomicU64,
}

impl Default for PayloadIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl PayloadIndex {
    pub fn new() -> Self {
        Self {
            fields: RwLock::new(HashMap::new()),
            indexed_fields: RwLock::new(Vec::new()),
            stats: PayloadIndexStats::default(),
        }
    }

    /// Create index with pre-defined indexed fields.
    /// Functional: uses iterator to initialize fields.
    pub fn with_fields<I, S>(fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let field_names: Vec<String> = fields
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();

        let field_map: HashMap<String, FieldIndex> = field_names
            .iter()
            .map(|name| (name.clone(), FieldIndex::new()))
            .collect();

        Self {
            fields: RwLock::new(field_map),
            indexed_fields: RwLock::new(field_names),
            stats: PayloadIndexStats::default(),
        }
    }

    /// Add a field to be indexed.
    pub fn add_field(&self, field_name: &str) {
        let mut fields = self.fields.write();
        let mut indexed = self.indexed_fields.write();

        if !fields.contains_key(field_name) {
            fields.insert(field_name.to_string(), FieldIndex::new());
            indexed.push(field_name.to_string());
        }
    }

    /// Check if a field is indexed.
    pub fn is_indexed(&self, field_name: &str) -> bool {
        self.fields.read().contains_key(field_name)
    }

    /// Insert a payload entry.
    /// Functional: uses if-let with entry pattern.
    pub fn insert(&self, id: u64, field: &str, value: &str) {
        let fields = self.fields.read();
        if let Some(index) = fields.get(field) {
            index.insert(id, value);
            self.stats.inserts.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Insert multiple fields at once.
    /// Functional: iterates over key-value pairs.
    pub fn insert_many<'a, I>(&self, id: u64, entries: I)
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let fields = self.fields.read();
        entries
            .into_iter()
            .filter_map(|(field, value)| {
                fields.get(field).map(|index| (index, value))
            })
            .for_each(|(index, value)| {
                index.insert(id, value);
                self.stats.inserts.fetch_add(1, Ordering::Relaxed);
            });
    }

    /// Remove an entry.
    pub fn remove(&self, id: u64, field: &str, value: &str) {
        let fields = self.fields.read();
        if let Some(index) = fields.get(field) {
            index.remove(id, value);
        }
    }

    /// Filter by a single field-value pair.
    /// Functional: Option monad for optional result.
    pub fn filter_eq(&self, field: &str, value: &str) -> Option<RoaringBitmap> {
        self.stats.lookups.fetch_add(1, Ordering::Relaxed);
        self.fields
            .read()
            .get(field)
            .and_then(|index| index.get(value))
    }

    /// Filter by multiple conditions (AND).
    /// Functional: fold with intersection, using Option monad.
    pub fn filter_and<'a, I>(&self, conditions: I) -> Option<RoaringBitmap>
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        self.stats.lookups.fetch_add(1, Ordering::Relaxed);
        let fields = self.fields.read();

        conditions
            .into_iter()
            .filter_map(|(field, value)| {
                fields.get(field).and_then(|index| index.get(value))
            })
            .reduce(|acc, bitmap| &acc & &bitmap)
    }

    /// Filter by multiple conditions (OR on same field).
    /// Functional: union via fold.
    pub fn filter_or<'a, I>(&self, field: &str, values: I) -> RoaringBitmap
    where
        I: IntoIterator<Item = &'a str>,
    {
        self.stats.lookups.fetch_add(1, Ordering::Relaxed);
        self.fields
            .read()
            .get(field)
            .map(|index| index.get_any(values))
            .unwrap_or_default()
    }

    /// Complex filter with AND/OR combinations.
    /// Functional: recursive evaluation with pattern matching.
    pub fn filter(&self, query: &FilterQuery) -> Option<RoaringBitmap> {
        self.stats.lookups.fetch_add(1, Ordering::Relaxed);
        self.evaluate_query(query)
    }

    /// Evaluate a filter query recursively.
    /// Functional: pattern matching with recursive calls.
    fn evaluate_query(&self, query: &FilterQuery) -> Option<RoaringBitmap> {
        match query {
            FilterQuery::Eq { field, value } => self.filter_eq(field, value),

            FilterQuery::In { field, values } => {
                let result = self.filter_or(field, values.iter().map(|s| s.as_str()));
                if result.is_empty() {
                    None
                } else {
                    Some(result)
                }
            }

            FilterQuery::And(queries) => {
                queries
                    .iter()
                    .filter_map(|q| self.evaluate_query(q))
                    .reduce(|acc, bitmap| &acc & &bitmap)
            }

            FilterQuery::Or(queries) => {
                let result = queries
                    .iter()
                    .filter_map(|q| self.evaluate_query(q))
                    .fold(RoaringBitmap::new(), |acc, bitmap| &acc | &bitmap);

                if result.is_empty() {
                    None
                } else {
                    Some(result)
                }
            }

            FilterQuery::Not(inner) => {
                // NOT requires knowing the universe of IDs
                // For now, return None (caller handles as "no filter")
                self.evaluate_query(inner).map(|_| {
                    // Would need universe bitmap to compute NOT
                    RoaringBitmap::new()
                })
            }
        }
    }

    /// Get statistics for all indexed fields.
    /// Functional: map over fields to collect stats.
    pub fn stats(&self) -> PayloadIndexFullStats {
        let fields = self.fields.read();
        let field_stats: HashMap<String, FieldIndexStats> = fields
            .iter()
            .map(|(name, index)| (name.clone(), index.stats()))
            .collect();

        PayloadIndexFullStats {
            indexed_fields: self.indexed_fields.read().clone(),
            field_stats,
            total_inserts: self.stats.inserts.load(Ordering::Relaxed),
            total_lookups: self.stats.lookups.load(Ordering::Relaxed),
        }
    }

    /// Build index from existing data.
    /// Functional: processes iterator of (id, field, value) tuples.
    pub fn build_from<'a, I>(&self, data: I)
    where
        I: IntoIterator<Item = (u64, &'a str, &'a str)>,
    {
        let fields = self.fields.read();
        data.into_iter()
            .filter_map(|(id, field, value)| {
                fields.get(field).map(|index| (id, index, value))
            })
            .for_each(|(id, index, value)| {
                index.insert(id, value);
            });
    }
}

/// Filter query DSL using algebraic data types.
/// Enables complex filter expressions with AND/OR/NOT.
#[derive(Debug, Clone)]
pub enum FilterQuery {
    /// Exact match: field == value
    Eq { field: String, value: String },
    /// In set: field IN (values...)
    In { field: String, values: Vec<String> },
    /// All conditions must match
    And(Vec<FilterQuery>),
    /// Any condition must match
    Or(Vec<FilterQuery>),
    /// Negation
    Not(Box<FilterQuery>),
}

impl FilterQuery {
    /// Create equality filter.
    /// Functional: builder pattern.
    pub fn eq(field: impl Into<String>, value: impl Into<String>) -> Self {
        FilterQuery::Eq {
            field: field.into(),
            value: value.into(),
        }
    }

    /// Create IN filter.
    pub fn in_set<I, S>(field: impl Into<String>, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        FilterQuery::In {
            field: field.into(),
            values: values.into_iter().map(|s| s.into()).collect(),
        }
    }

    /// Create AND filter.
    pub fn and(queries: Vec<FilterQuery>) -> Self {
        FilterQuery::And(queries)
    }

    /// Create OR filter.
    pub fn or(queries: Vec<FilterQuery>) -> Self {
        FilterQuery::Or(queries)
    }

    /// Create NOT filter.
    pub fn not(query: FilterQuery) -> Self {
        FilterQuery::Not(Box::new(query))
    }

    /// Parse from simple HashMap (legacy compatibility).
    /// Functional: transforms map entries to AND of Eq filters.
    pub fn from_map(filters: &HashMap<String, String>) -> Option<Self> {
        let conditions: Vec<FilterQuery> = filters
            .iter()
            .map(|(k, v)| FilterQuery::eq(k.clone(), v.clone()))
            .collect();

        match conditions.len() {
            0 => None,
            1 => Some(conditions.into_iter().next().unwrap()),
            _ => Some(FilterQuery::And(conditions)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PayloadIndexFullStats {
    pub indexed_fields: Vec<String>,
    pub field_stats: HashMap<String, FieldIndexStats>,
    pub total_inserts: u64,
    pub total_lookups: u64,
}

/// Helper trait for functional bitmap operations.
pub trait BitmapExt {
    /// Apply filter if Some, otherwise return original.
    fn apply_filter(self, filter: Option<&RoaringBitmap>) -> Self;

    /// Intersect with another bitmap if not empty.
    fn intersect_non_empty(self, other: &RoaringBitmap) -> Self;
}

impl BitmapExt for RoaringBitmap {
    fn apply_filter(self, filter: Option<&RoaringBitmap>) -> Self {
        filter
            .map(|f| &self & f)
            .unwrap_or(self)
    }

    fn intersect_non_empty(self, other: &RoaringBitmap) -> Self {
        if other.is_empty() {
            self
        } else {
            &self & other
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_index_basic() {
        let index = FieldIndex::new();

        // Insert
        index.insert(1, "red");
        index.insert(2, "red");
        index.insert(3, "blue");

        // Query
        let red = index.get("red").unwrap();
        assert!(red.contains(1));
        assert!(red.contains(2));
        assert!(!red.contains(3));

        let blue = index.get("blue").unwrap();
        assert!(blue.contains(3));
    }

    #[test]
    fn test_payload_index_and() {
        let index = PayloadIndex::with_fields(["color", "size"]);

        index.insert(1, "color", "red");
        index.insert(1, "size", "large");
        index.insert(2, "color", "red");
        index.insert(2, "size", "small");
        index.insert(3, "color", "blue");
        index.insert(3, "size", "large");

        // AND: color=red AND size=large -> should be ID 1
        let result = index.filter_and([("color", "red"), ("size", "large")]).unwrap();
        assert!(result.contains(1));
        assert!(!result.contains(2));
        assert!(!result.contains(3));
    }

    #[test]
    fn test_filter_query_dsl() {
        let index = PayloadIndex::with_fields(["category", "status"]);

        index.insert(1, "category", "electronics");
        index.insert(1, "status", "active");
        index.insert(2, "category", "clothing");
        index.insert(2, "status", "active");
        index.insert(3, "category", "electronics");
        index.insert(3, "status", "inactive");

        // Query: category=electronics AND status=active
        let query = FilterQuery::and(vec![
            FilterQuery::eq("category", "electronics"),
            FilterQuery::eq("status", "active"),
        ]);

        let result = index.filter(&query).unwrap();
        assert!(result.contains(1));
        assert!(!result.contains(2));
        assert!(!result.contains(3));
    }

    #[test]
    fn test_filter_or() {
        let index = PayloadIndex::with_fields(["color"]);

        index.insert(1, "color", "red");
        index.insert(2, "color", "green");
        index.insert(3, "color", "blue");

        // OR: color IN (red, blue)
        let result = index.filter_or("color", ["red", "blue"].iter().copied());
        assert!(result.contains(1));
        assert!(!result.contains(2));
        assert!(result.contains(3));
    }
}
