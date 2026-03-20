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
use std::collections::{BTreeMap, HashMap};
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

// ============================================================================
// Numeric Field Index (for Range Queries)
// ============================================================================

/// Ordered index for numeric fields supporting range queries.
/// Uses BTreeMap for O(log n) range lookups.
///
/// Functional patterns:
/// - BTreeMap's range iterator for efficient range scans
/// - fold/reduce for bitmap aggregation
#[derive(Default)]
pub struct NumericFieldIndex {
    /// value -> bitmap of IDs (sorted by value)
    index: RwLock<BTreeMap<i64, RoaringBitmap>>,
    /// ID -> value (for updates/deletes)
    reverse: RwLock<HashMap<u32, i64>>,
    /// Statistics
    min_value: RwLock<Option<i64>>,
    max_value: RwLock<Option<i64>>,
}

impl NumericFieldIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a numeric value for an ID.
    pub fn insert(&self, id: u64, value: i64) {
        let id_u32 = id as u32;

        // Remove old value if exists
        {
            let reverse = self.reverse.read();
            if let Some(&old_value) = reverse.get(&id_u32) {
                drop(reverse);
                let mut index = self.index.write();
                if let Some(bitmap) = index.get_mut(&old_value) {
                    bitmap.remove(id_u32);
                    if bitmap.is_empty() {
                        index.remove(&old_value);
                    }
                }
            }
        }

        // Insert new value
        {
            let mut index = self.index.write();
            index
                .entry(value)
                .or_insert_with(RoaringBitmap::new)
                .insert(id_u32);
        }

        // Update reverse index
        {
            let mut reverse = self.reverse.write();
            reverse.insert(id_u32, value);
        }

        // Update min/max
        {
            let mut min = self.min_value.write();
            let mut max = self.max_value.write();
            *min = Some(min.map(|m| m.min(value)).unwrap_or(value));
            *max = Some(max.map(|m| m.max(value)).unwrap_or(value));
        }
    }

    /// Insert from float (converts to i64 with 6 decimal precision).
    pub fn insert_f64(&self, id: u64, value: f64) {
        self.insert(id, float_to_int(value));
    }

    /// Remove an ID from the index.
    pub fn remove(&self, id: u64) {
        let id_u32 = id as u32;

        let value = {
            let mut reverse = self.reverse.write();
            reverse.remove(&id_u32)
        };

        if let Some(v) = value {
            let mut index = self.index.write();
            if let Some(bitmap) = index.get_mut(&v) {
                bitmap.remove(id_u32);
                if bitmap.is_empty() {
                    index.remove(&v);
                }
            }
        }
    }

    /// Get IDs with exact value.
    pub fn get_eq(&self, value: i64) -> Option<RoaringBitmap> {
        self.index.read().get(&value).cloned()
    }

    /// Get IDs > value.
    /// Functional: range iterator with fold.
    pub fn get_gt(&self, value: i64) -> RoaringBitmap {
        let index = self.index.read();
        index
            .range((std::ops::Bound::Excluded(value), std::ops::Bound::Unbounded))
            .map(|(_, bitmap)| bitmap)
            .fold(RoaringBitmap::new(), |acc, b| &acc | b)
    }

    /// Get IDs >= value.
    pub fn get_gte(&self, value: i64) -> RoaringBitmap {
        let index = self.index.read();
        index
            .range(value..)
            .map(|(_, bitmap)| bitmap)
            .fold(RoaringBitmap::new(), |acc, b| &acc | b)
    }

    /// Get IDs < value.
    pub fn get_lt(&self, value: i64) -> RoaringBitmap {
        let index = self.index.read();
        index
            .range(..value)
            .map(|(_, bitmap)| bitmap)
            .fold(RoaringBitmap::new(), |acc, b| &acc | b)
    }

    /// Get IDs <= value.
    pub fn get_lte(&self, value: i64) -> RoaringBitmap {
        let index = self.index.read();
        index
            .range(..=value)
            .map(|(_, bitmap)| bitmap)
            .fold(RoaringBitmap::new(), |acc, b| &acc | b)
    }

    /// Get IDs in range [min, max] (inclusive).
    /// Functional: uses BTreeMap range with fold.
    pub fn get_range(&self, min: i64, max: i64) -> RoaringBitmap {
        let index = self.index.read();
        index
            .range(min..=max)
            .map(|(_, bitmap)| bitmap)
            .fold(RoaringBitmap::new(), |acc, b| &acc | b)
    }

    /// Get IDs in range (min, max) (exclusive).
    pub fn get_range_exclusive(&self, min: i64, max: i64) -> RoaringBitmap {
        use std::ops::Bound;
        let index = self.index.read();
        index
            .range((Bound::Excluded(min), Bound::Excluded(max)))
            .map(|(_, bitmap)| bitmap)
            .fold(RoaringBitmap::new(), |acc, b| &acc | b)
    }

    /// Get statistics.
    pub fn stats(&self) -> NumericFieldIndexStats {
        let index = self.index.read();
        let count = self.reverse.read().len();

        NumericFieldIndexStats {
            unique_values: index.len(),
            total_ids: count as u64,
            min_value: *self.min_value.read(),
            max_value: *self.max_value.read(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NumericFieldIndexStats {
    pub unique_values: usize,
    pub total_ids: u64,
    pub min_value: Option<i64>,
    pub max_value: Option<i64>,
}

/// Convert float to int with 6 decimal precision.
/// Allows storing floats in BTreeMap while maintaining order.
#[inline]
pub fn float_to_int(f: f64) -> i64 {
    (f * 1_000_000.0) as i64
}

/// Convert int back to float.
#[inline]
pub fn int_to_float(i: i64) -> f64 {
    i as f64 / 1_000_000.0
}

// ============================================================================
// Payload Index Manager
// ============================================================================

/// Payload index manager for multiple fields.
/// Provides efficient filtering across all indexed fields.
/// Supports both string fields (exact/in) and numeric fields (range queries).
pub struct PayloadIndex {
    /// field_name -> FieldIndex (for string values)
    fields: RwLock<HashMap<String, FieldIndex>>,
    /// field_name -> NumericFieldIndex (for numeric values)
    numeric_fields: RwLock<HashMap<String, NumericFieldIndex>>,
    /// Fields that are indexed (string)
    indexed_fields: RwLock<Vec<String>>,
    /// Fields that are indexed (numeric)
    numeric_indexed_fields: RwLock<Vec<String>>,
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
            numeric_fields: RwLock::new(HashMap::new()),
            indexed_fields: RwLock::new(Vec::new()),
            numeric_indexed_fields: RwLock::new(Vec::new()),
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
            numeric_fields: RwLock::new(HashMap::new()),
            indexed_fields: RwLock::new(field_names),
            numeric_indexed_fields: RwLock::new(Vec::new()),
            stats: PayloadIndexStats::default(),
        }
    }

    /// Create index with both string and numeric fields.
    pub fn with_fields_and_numeric<I1, I2, S>(string_fields: I1, numeric_fields: I2) -> Self
    where
        I1: IntoIterator<Item = S>,
        I2: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let string_names: Vec<String> = string_fields
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();

        let numeric_names: Vec<String> = numeric_fields
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();

        let string_map: HashMap<String, FieldIndex> = string_names
            .iter()
            .map(|name| (name.clone(), FieldIndex::new()))
            .collect();

        let numeric_map: HashMap<String, NumericFieldIndex> = numeric_names
            .iter()
            .map(|name| (name.clone(), NumericFieldIndex::new()))
            .collect();

        Self {
            fields: RwLock::new(string_map),
            numeric_fields: RwLock::new(numeric_map),
            indexed_fields: RwLock::new(string_names),
            numeric_indexed_fields: RwLock::new(numeric_names),
            stats: PayloadIndexStats::default(),
        }
    }

    /// Add a string field to be indexed.
    pub fn add_field(&self, field_name: &str) {
        let mut fields = self.fields.write();
        let mut indexed = self.indexed_fields.write();

        if !fields.contains_key(field_name) {
            fields.insert(field_name.to_string(), FieldIndex::new());
            indexed.push(field_name.to_string());
        }
    }

    /// Add a numeric field to be indexed.
    pub fn add_numeric_field(&self, field_name: &str) {
        let mut fields = self.numeric_fields.write();
        let mut indexed = self.numeric_indexed_fields.write();

        if !fields.contains_key(field_name) {
            fields.insert(field_name.to_string(), NumericFieldIndex::new());
            indexed.push(field_name.to_string());
        }
    }

    /// Check if a string field is indexed.
    pub fn is_indexed(&self, field_name: &str) -> bool {
        self.fields.read().contains_key(field_name)
    }

    /// Check if a numeric field is indexed.
    pub fn is_numeric_indexed(&self, field_name: &str) -> bool {
        self.numeric_fields.read().contains_key(field_name)
    }

    /// Insert a string payload entry.
    /// Functional: uses if-let with entry pattern.
    pub fn insert(&self, id: u64, field: &str, value: &str) {
        let fields = self.fields.read();
        if let Some(index) = fields.get(field) {
            index.insert(id, value);
            self.stats.inserts.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Insert a numeric payload entry (i64).
    pub fn insert_numeric(&self, id: u64, field: &str, value: i64) {
        let fields = self.numeric_fields.read();
        if let Some(index) = fields.get(field) {
            index.insert(id, value);
            self.stats.inserts.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Insert a numeric payload entry (f64).
    pub fn insert_numeric_f64(&self, id: u64, field: &str, value: f64) {
        let fields = self.numeric_fields.read();
        if let Some(index) = fields.get(field) {
            index.insert_f64(id, value);
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

    /// Remove a string entry.
    pub fn remove(&self, id: u64, field: &str, value: &str) {
        let fields = self.fields.read();
        if let Some(index) = fields.get(field) {
            index.remove(id, value);
        }
    }

    /// Remove a numeric entry.
    pub fn remove_numeric(&self, id: u64, field: &str) {
        let fields = self.numeric_fields.read();
        if let Some(index) = fields.get(field) {
            index.remove(id);
        }
    }

    // ========================================================================
    // Numeric Range Query Methods
    // ========================================================================

    /// Filter numeric field: value > threshold.
    pub fn filter_gt(&self, field: &str, value: i64) -> Option<RoaringBitmap> {
        self.stats.lookups.fetch_add(1, Ordering::Relaxed);
        self.numeric_fields
            .read()
            .get(field)
            .map(|idx| idx.get_gt(value))
            .filter(|b| !b.is_empty())
    }

    /// Filter numeric field: value >= threshold.
    pub fn filter_gte(&self, field: &str, value: i64) -> Option<RoaringBitmap> {
        self.stats.lookups.fetch_add(1, Ordering::Relaxed);
        self.numeric_fields
            .read()
            .get(field)
            .map(|idx| idx.get_gte(value))
            .filter(|b| !b.is_empty())
    }

    /// Filter numeric field: value < threshold.
    pub fn filter_lt(&self, field: &str, value: i64) -> Option<RoaringBitmap> {
        self.stats.lookups.fetch_add(1, Ordering::Relaxed);
        self.numeric_fields
            .read()
            .get(field)
            .map(|idx| idx.get_lt(value))
            .filter(|b| !b.is_empty())
    }

    /// Filter numeric field: value <= threshold.
    pub fn filter_lte(&self, field: &str, value: i64) -> Option<RoaringBitmap> {
        self.stats.lookups.fetch_add(1, Ordering::Relaxed);
        self.numeric_fields
            .read()
            .get(field)
            .map(|idx| idx.get_lte(value))
            .filter(|b| !b.is_empty())
    }

    /// Filter numeric field: min <= value <= max (inclusive).
    pub fn filter_range(&self, field: &str, min: i64, max: i64) -> Option<RoaringBitmap> {
        self.stats.lookups.fetch_add(1, Ordering::Relaxed);
        self.numeric_fields
            .read()
            .get(field)
            .map(|idx| idx.get_range(min, max))
            .filter(|b| !b.is_empty())
    }

    /// Filter numeric field with float values.
    pub fn filter_range_f64(&self, field: &str, min: f64, max: f64) -> Option<RoaringBitmap> {
        self.filter_range(field, float_to_int(min), float_to_int(max))
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
            // String filters
            FilterQuery::Eq { field, value } => self.filter_eq(field, value),

            FilterQuery::In { field, values } => {
                let result = self.filter_or(field, values.iter().map(|s| s.as_str()));
                if result.is_empty() {
                    None
                } else {
                    Some(result)
                }
            }

            // Numeric range filters (i64)
            FilterQuery::Gt { field, value } => self.filter_gt(field, *value),
            FilterQuery::Gte { field, value } => self.filter_gte(field, *value),
            FilterQuery::Lt { field, value } => self.filter_lt(field, *value),
            FilterQuery::Lte { field, value } => self.filter_lte(field, *value),
            FilterQuery::Range { field, min, max } => self.filter_range(field, *min, *max),

            // Numeric range filters (f64 -> i64)
            FilterQuery::GtF { field, value } => self.filter_gt(field, float_to_int(*value)),
            FilterQuery::GteF { field, value } => self.filter_gte(field, float_to_int(*value)),
            FilterQuery::LtF { field, value } => self.filter_lt(field, float_to_int(*value)),
            FilterQuery::LteF { field, value } => self.filter_lte(field, float_to_int(*value)),
            FilterQuery::RangeF { field, min, max } => {
                self.filter_range(field, float_to_int(*min), float_to_int(*max))
            }

            // Logical operators
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
        let numeric = self.numeric_fields.read();

        let field_stats: HashMap<String, FieldIndexStats> = fields
            .iter()
            .map(|(name, index)| (name.clone(), index.stats()))
            .collect();

        let numeric_field_stats: HashMap<String, NumericFieldIndexStats> = numeric
            .iter()
            .map(|(name, index)| (name.clone(), index.stats()))
            .collect();

        PayloadIndexFullStats {
            indexed_fields: self.indexed_fields.read().clone(),
            numeric_indexed_fields: self.numeric_indexed_fields.read().clone(),
            field_stats,
            numeric_field_stats,
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
/// Enables complex filter expressions with AND/OR/NOT and range queries.
#[derive(Debug, Clone)]
pub enum FilterQuery {
    // ========== String Filters ==========
    /// Exact match: field == value
    Eq { field: String, value: String },
    /// In set: field IN (values...)
    In { field: String, values: Vec<String> },

    // ========== Numeric Range Filters ==========
    /// Greater than: field > value
    Gt { field: String, value: i64 },
    /// Greater than or equal: field >= value
    Gte { field: String, value: i64 },
    /// Less than: field < value
    Lt { field: String, value: i64 },
    /// Less than or equal: field <= value
    Lte { field: String, value: i64 },
    /// Range: min <= field <= max (inclusive)
    Range { field: String, min: i64, max: i64 },

    // ========== Float Range Filters (auto-converted to i64) ==========
    /// Greater than (float): field > value
    GtF { field: String, value: f64 },
    /// Greater than or equal (float): field >= value
    GteF { field: String, value: f64 },
    /// Less than (float): field < value
    LtF { field: String, value: f64 },
    /// Less than or equal (float): field <= value
    LteF { field: String, value: f64 },
    /// Range (float): min <= field <= max
    RangeF { field: String, min: f64, max: f64 },

    // ========== Logical Operators ==========
    /// All conditions must match
    And(Vec<FilterQuery>),
    /// Any condition must match
    Or(Vec<FilterQuery>),
    /// Negation
    Not(Box<FilterQuery>),
}

impl FilterQuery {
    // ========== String Filter Builders ==========

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

    // ========== Numeric Range Filter Builders (i64) ==========

    /// Create greater than filter: field > value
    pub fn gt(field: impl Into<String>, value: i64) -> Self {
        FilterQuery::Gt { field: field.into(), value }
    }

    /// Create greater than or equal filter: field >= value
    pub fn gte(field: impl Into<String>, value: i64) -> Self {
        FilterQuery::Gte { field: field.into(), value }
    }

    /// Create less than filter: field < value
    pub fn lt(field: impl Into<String>, value: i64) -> Self {
        FilterQuery::Lt { field: field.into(), value }
    }

    /// Create less than or equal filter: field <= value
    pub fn lte(field: impl Into<String>, value: i64) -> Self {
        FilterQuery::Lte { field: field.into(), value }
    }

    /// Create range filter: min <= field <= max
    pub fn range(field: impl Into<String>, min: i64, max: i64) -> Self {
        FilterQuery::Range { field: field.into(), min, max }
    }

    // ========== Numeric Range Filter Builders (f64) ==========

    /// Create greater than filter (float): field > value
    pub fn gt_f(field: impl Into<String>, value: f64) -> Self {
        FilterQuery::GtF { field: field.into(), value }
    }

    /// Create greater than or equal filter (float): field >= value
    pub fn gte_f(field: impl Into<String>, value: f64) -> Self {
        FilterQuery::GteF { field: field.into(), value }
    }

    /// Create less than filter (float): field < value
    pub fn lt_f(field: impl Into<String>, value: f64) -> Self {
        FilterQuery::LtF { field: field.into(), value }
    }

    /// Create less than or equal filter (float): field <= value
    pub fn lte_f(field: impl Into<String>, value: f64) -> Self {
        FilterQuery::LteF { field: field.into(), value }
    }

    /// Create range filter (float): min <= field <= max
    pub fn range_f(field: impl Into<String>, min: f64, max: f64) -> Self {
        FilterQuery::RangeF { field: field.into(), min, max }
    }

    // ========== Logical Operator Builders ==========

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
    pub numeric_indexed_fields: Vec<String>,
    pub field_stats: HashMap<String, FieldIndexStats>,
    pub numeric_field_stats: HashMap<String, NumericFieldIndexStats>,
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

    #[test]
    fn test_numeric_range_basic() {
        let index = PayloadIndex::with_fields_and_numeric(
            Vec::<&str>::new(),  // no string fields
            ["price"],           // numeric field
        );

        // Insert prices: 10, 20, 30, 40, 50
        index.insert_numeric(1, "price", 10);
        index.insert_numeric(2, "price", 20);
        index.insert_numeric(3, "price", 30);
        index.insert_numeric(4, "price", 40);
        index.insert_numeric(5, "price", 50);

        // Test: price > 25 -> should match 3, 4, 5
        let result = index.filter_gt("price", 25).unwrap();
        assert!(!result.contains(1));
        assert!(!result.contains(2));
        assert!(result.contains(3));
        assert!(result.contains(4));
        assert!(result.contains(5));

        // Test: price >= 30 -> should match 3, 4, 5
        let result = index.filter_gte("price", 30).unwrap();
        assert!(!result.contains(1));
        assert!(!result.contains(2));
        assert!(result.contains(3));
        assert!(result.contains(4));
        assert!(result.contains(5));

        // Test: price < 30 -> should match 1, 2
        let result = index.filter_lt("price", 30).unwrap();
        assert!(result.contains(1));
        assert!(result.contains(2));
        assert!(!result.contains(3));

        // Test: price <= 30 -> should match 1, 2, 3
        let result = index.filter_lte("price", 30).unwrap();
        assert!(result.contains(1));
        assert!(result.contains(2));
        assert!(result.contains(3));
        assert!(!result.contains(4));
    }

    #[test]
    fn test_numeric_range_query() {
        let index = PayloadIndex::with_fields_and_numeric(
            Vec::<&str>::new(),
            ["score"],
        );

        // Insert scores: 100, 200, 300, 400, 500
        for i in 1..=5 {
            index.insert_numeric(i, "score", (i * 100) as i64);
        }

        // Test: 200 <= score <= 400 -> should match 2, 3, 4
        let result = index.filter_range("score", 200, 400).unwrap();
        assert!(!result.contains(1));
        assert!(result.contains(2));
        assert!(result.contains(3));
        assert!(result.contains(4));
        assert!(!result.contains(5));
    }

    #[test]
    fn test_filter_query_with_range() {
        let index = PayloadIndex::with_fields_and_numeric(
            ["category"],
            ["price"],
        );

        // Insert products
        index.insert(1, "category", "electronics");
        index.insert_numeric(1, "price", 100);

        index.insert(2, "category", "electronics");
        index.insert_numeric(2, "price", 500);

        index.insert(3, "category", "clothing");
        index.insert_numeric(3, "price", 50);

        index.insert(4, "category", "electronics");
        index.insert_numeric(4, "price", 200);

        // Query: category=electronics AND price < 300
        let query = FilterQuery::and(vec![
            FilterQuery::eq("category", "electronics"),
            FilterQuery::lt("price", 300),
        ]);

        let result = index.filter(&query).unwrap();
        assert!(result.contains(1));  // electronics, 100
        assert!(!result.contains(2)); // electronics, 500 (too expensive)
        assert!(!result.contains(3)); // clothing
        assert!(result.contains(4));  // electronics, 200
    }

    #[test]
    fn test_float_range_query() {
        let index = PayloadIndex::with_fields_and_numeric(
            Vec::<&str>::new(),
            ["rating"],
        );

        // Insert ratings as floats
        index.insert_numeric_f64(1, "rating", 4.5);
        index.insert_numeric_f64(2, "rating", 3.2);
        index.insert_numeric_f64(3, "rating", 4.8);
        index.insert_numeric_f64(4, "rating", 2.1);
        index.insert_numeric_f64(5, "rating", 4.0);

        // Query: rating >= 4.0 using FilterQuery DSL
        let query = FilterQuery::gte_f("rating", 4.0);
        let result = index.filter(&query).unwrap();

        assert!(result.contains(1));  // 4.5
        assert!(!result.contains(2)); // 3.2
        assert!(result.contains(3));  // 4.8
        assert!(!result.contains(4)); // 2.1
        assert!(result.contains(5));  // 4.0
    }
}
