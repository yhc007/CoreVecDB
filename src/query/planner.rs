//! Query Planner for VectorDB.
//!
//! Estimates filter selectivity and optimizes filter ordering for better search performance.
//!
//! # Concept
//! - **Selectivity**: Fraction of vectors that pass a filter (0.0 = very selective, 1.0 = no filtering)
//! - **Cardinality**: Number of unique values for a field
//! - **Filter Order**: Most selective filters applied first for early termination
//!
//! # Usage
//! ```rust,ignore
//! use vectordb::query::{QueryPlanner, FilterOrder};
//!
//! let planner = QueryPlanner::new();
//!
//! // Update statistics when inserting
//! planner.observe_field("category", "electronics");
//!
//! // Get optimized filter order
//! let filters = vec![
//!     FilterOrder::new("status", "active"),
//!     FilterOrder::new("category", "electronics"),
//! ];
//! let optimized = planner.optimize_filter_order(filters);
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

/// Statistics for a single field.
#[derive(Debug, Clone)]
pub struct FieldStatistics {
    /// Number of unique values observed for this field.
    pub cardinality: usize,
    /// Total number of vectors with this field.
    pub total_vectors: usize,
    /// Estimated selectivity (cardinality / total_vectors).
    /// Lower values mean more selective (fewer results).
    pub selectivity: f32,
    /// Value frequency map for common values.
    pub value_frequencies: HashMap<String, usize>,
}

impl FieldStatistics {
    /// Create new field statistics.
    pub fn new() -> Self {
        Self {
            cardinality: 0,
            total_vectors: 0,
            selectivity: 1.0,
            value_frequencies: HashMap::new(),
        }
    }

    /// Update statistics with a new observed value.
    pub fn observe(&mut self, value: &str) {
        self.total_vectors += 1;

        let count = self.value_frequencies.entry(value.to_string()).or_insert(0);
        if *count == 0 {
            self.cardinality += 1;
        }
        *count += 1;

        self.update_selectivity();
    }

    /// Remove a value observation.
    pub fn remove(&mut self, value: &str) {
        if let Some(count) = self.value_frequencies.get_mut(value) {
            if *count > 0 {
                *count -= 1;
                self.total_vectors = self.total_vectors.saturating_sub(1);

                if *count == 0 {
                    self.value_frequencies.remove(value);
                    self.cardinality = self.cardinality.saturating_sub(1);
                }

                self.update_selectivity();
            }
        }
    }

    /// Update selectivity based on current statistics.
    fn update_selectivity(&mut self) {
        if self.total_vectors == 0 || self.cardinality == 0 {
            self.selectivity = 1.0;
        } else {
            // Selectivity = 1 / cardinality (how many vectors match one value on average)
            self.selectivity = 1.0 / self.cardinality as f32;
        }
    }

    /// Estimate selectivity for a specific value.
    pub fn estimate_value_selectivity(&self, value: &str) -> f32 {
        if self.total_vectors == 0 {
            return 1.0;
        }

        if let Some(&count) = self.value_frequencies.get(value) {
            count as f32 / self.total_vectors as f32
        } else {
            // Unknown value - assume worst case (entire dataset)
            self.selectivity
        }
    }
}

impl Default for FieldStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// A filter condition for planning.
#[derive(Debug, Clone)]
pub struct FilterOrder {
    /// Field name.
    pub field: String,
    /// Filter value.
    pub value: String,
    /// Estimated selectivity (set by planner).
    pub estimated_selectivity: f32,
}

impl FilterOrder {
    /// Create a new filter order entry.
    pub fn new(field: &str, value: &str) -> Self {
        Self {
            field: field.to_string(),
            value: value.to_string(),
            estimated_selectivity: 1.0,
        }
    }

    /// Create with a known selectivity.
    pub fn with_selectivity(field: &str, value: &str, selectivity: f32) -> Self {
        Self {
            field: field.to_string(),
            value: value.to_string(),
            estimated_selectivity: selectivity,
        }
    }
}

/// Filter execution plan.
#[derive(Debug, Clone)]
pub struct FilterPlan {
    /// Ordered list of filters (most selective first).
    pub filters: Vec<FilterOrder>,
    /// Estimated total selectivity.
    pub total_selectivity: f32,
    /// Recommended strategy based on selectivity.
    pub strategy: FilterStrategy,
}

/// Recommended filtering strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterStrategy {
    /// Very selective - pre-filter before search.
    PreFilter,
    /// Moderately selective - use adaptive/hybrid approach.
    Hybrid,
    /// Low selectivity - post-filter after search.
    PostFilter,
    /// No filtering needed.
    NoFilter,
}

impl FilterStrategy {
    /// Determine strategy based on selectivity.
    pub fn from_selectivity(selectivity: f32) -> Self {
        if selectivity <= 0.0 {
            FilterStrategy::NoFilter
        } else if selectivity < 0.1 {
            FilterStrategy::PreFilter
        } else if selectivity < 0.5 {
            FilterStrategy::Hybrid
        } else {
            FilterStrategy::PostFilter
        }
    }
}

/// Query planner that tracks field statistics and optimizes filter execution.
pub struct QueryPlanner {
    /// Statistics per field.
    field_stats: RwLock<HashMap<String, FieldStatistics>>,
    /// Total observations.
    total_observations: AtomicU64,
}

impl QueryPlanner {
    /// Create a new query planner.
    pub fn new() -> Self {
        Self {
            field_stats: RwLock::new(HashMap::new()),
            total_observations: AtomicU64::new(0),
        }
    }

    /// Observe a field value (called during insert).
    pub fn observe_field(&self, field: &str, value: &str) {
        let mut stats = self.field_stats.write();
        let field_stats = stats.entry(field.to_string()).or_insert_with(FieldStatistics::new);
        field_stats.observe(value);
        self.total_observations.fetch_add(1, Ordering::Relaxed);
    }

    /// Observe multiple field values (called during batch insert).
    pub fn observe_fields(&self, metadata: &HashMap<String, String>) {
        let mut stats = self.field_stats.write();
        for (field, value) in metadata {
            let field_stats = stats.entry(field.to_string()).or_insert_with(FieldStatistics::new);
            field_stats.observe(value);
        }
        self.total_observations.fetch_add(metadata.len() as u64, Ordering::Relaxed);
    }

    /// Remove field value observation (called during delete).
    pub fn remove_field(&self, field: &str, value: &str) {
        let mut stats = self.field_stats.write();
        if let Some(field_stats) = stats.get_mut(field) {
            field_stats.remove(value);
        }
    }

    /// Remove multiple field value observations.
    pub fn remove_fields(&self, metadata: &HashMap<String, String>) {
        let mut stats = self.field_stats.write();
        for (field, value) in metadata {
            if let Some(field_stats) = stats.get_mut(field) {
                field_stats.remove(value);
            }
        }
    }

    /// Get statistics for a field.
    pub fn get_field_stats(&self, field: &str) -> Option<FieldStatistics> {
        self.field_stats.read().get(field).cloned()
    }

    /// Estimate selectivity for a single filter.
    pub fn estimate_selectivity(&self, field: &str, value: &str) -> f32 {
        let stats = self.field_stats.read();
        if let Some(field_stats) = stats.get(field) {
            field_stats.estimate_value_selectivity(value)
        } else {
            // Unknown field - assume no selectivity
            1.0
        }
    }

    /// Optimize filter order by selectivity (most selective first).
    pub fn optimize_filter_order(&self, mut filters: Vec<FilterOrder>) -> Vec<FilterOrder> {
        let stats = self.field_stats.read();

        // Set estimated selectivity for each filter
        for filter in &mut filters {
            if let Some(field_stats) = stats.get(&filter.field) {
                filter.estimated_selectivity = field_stats.estimate_value_selectivity(&filter.value);
            }
        }

        // Sort by selectivity (most selective = lowest value first)
        filters.sort_by(|a, b| {
            a.estimated_selectivity
                .partial_cmp(&b.estimated_selectivity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        filters
    }

    /// Create a filter execution plan.
    pub fn plan_filters(&self, filters: Vec<FilterOrder>) -> FilterPlan {
        if filters.is_empty() {
            return FilterPlan {
                filters: vec![],
                total_selectivity: 1.0,
                strategy: FilterStrategy::NoFilter,
            };
        }

        let optimized = self.optimize_filter_order(filters);

        // Estimate combined selectivity (multiply individual selectivities)
        let total_selectivity = optimized
            .iter()
            .map(|f| f.estimated_selectivity)
            .fold(1.0, |acc, s| acc * s);

        let strategy = FilterStrategy::from_selectivity(total_selectivity);

        FilterPlan {
            filters: optimized,
            total_selectivity,
            strategy,
        }
    }

    /// Get all field statistics.
    pub fn all_stats(&self) -> HashMap<String, FieldStatistics> {
        self.field_stats.read().clone()
    }

    /// Clear all statistics.
    pub fn clear(&self) {
        self.field_stats.write().clear();
        self.total_observations.store(0, Ordering::Relaxed);
    }

    /// Get planner statistics.
    pub fn stats(&self) -> QueryPlannerStats {
        let field_stats = self.field_stats.read();
        QueryPlannerStats {
            total_fields: field_stats.len(),
            total_observations: self.total_observations.load(Ordering::Relaxed),
            field_cardinalities: field_stats
                .iter()
                .map(|(k, v)| (k.clone(), v.cardinality))
                .collect(),
        }
    }
}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the query planner.
#[derive(Debug, Clone)]
pub struct QueryPlannerStats {
    /// Number of tracked fields.
    pub total_fields: usize,
    /// Total observations recorded.
    pub total_observations: u64,
    /// Cardinality per field.
    pub field_cardinalities: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_statistics_basic() {
        let mut stats = FieldStatistics::new();

        stats.observe("electronics");
        stats.observe("electronics");
        stats.observe("books");
        stats.observe("clothing");

        assert_eq!(stats.cardinality, 3);
        assert_eq!(stats.total_vectors, 4);

        // electronics appears 2/4 = 0.5
        assert!((stats.estimate_value_selectivity("electronics") - 0.5).abs() < 0.01);
        // books appears 1/4 = 0.25
        assert!((stats.estimate_value_selectivity("books") - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_field_statistics_remove() {
        let mut stats = FieldStatistics::new();

        stats.observe("a");
        stats.observe("a");
        stats.observe("b");

        assert_eq!(stats.cardinality, 2);
        assert_eq!(stats.total_vectors, 3);

        stats.remove("a");
        assert_eq!(stats.total_vectors, 2);
        assert_eq!(stats.cardinality, 2); // Still 2 because "a" still has count 1

        stats.remove("a");
        assert_eq!(stats.total_vectors, 1);
        assert_eq!(stats.cardinality, 1); // Now 1 because "a" is gone
    }

    #[test]
    fn test_query_planner_observe() {
        let planner = QueryPlanner::new();

        planner.observe_field("category", "electronics");
        planner.observe_field("category", "electronics");
        planner.observe_field("category", "books");
        planner.observe_field("status", "active");
        planner.observe_field("status", "active");
        planner.observe_field("status", "active");

        let cat_stats = planner.get_field_stats("category").unwrap();
        assert_eq!(cat_stats.cardinality, 2);

        let status_stats = planner.get_field_stats("status").unwrap();
        assert_eq!(status_stats.cardinality, 1); // Only "active"
    }

    #[test]
    fn test_filter_order_optimization() {
        let planner = QueryPlanner::new();

        // Create skewed distribution
        // "status" = "active" appears 90 times (90% of vectors)
        // "category" = "electronics" appears 10 times (10% of vectors)
        for _ in 0..90 {
            planner.observe_field("status", "active");
            planner.observe_field("category", "other");
        }
        for _ in 0..10 {
            planner.observe_field("status", "inactive");
            planner.observe_field("category", "electronics");
        }

        let filters = vec![
            FilterOrder::new("status", "active"),      // High selectivity (90%)
            FilterOrder::new("category", "electronics"), // Low selectivity (10%)
        ];

        let optimized = planner.optimize_filter_order(filters);

        // "electronics" should come first (more selective)
        assert_eq!(optimized[0].field, "category");
        assert_eq!(optimized[1].field, "status");
    }

    #[test]
    fn test_filter_plan_strategy() {
        let planner = QueryPlanner::new();

        // Create a field with high cardinality (very selective)
        for i in 0..1000 {
            planner.observe_field("user_id", &format!("user_{}", i));
        }

        // Create a field with low cardinality (not selective)
        for _ in 0..500 {
            planner.observe_field("active", "true");
        }
        for _ in 0..500 {
            planner.observe_field("active", "false");
        }

        // Plan for a specific user (highly selective)
        let plan = planner.plan_filters(vec![
            FilterOrder::new("user_id", "user_42"),
        ]);
        assert_eq!(plan.strategy, FilterStrategy::PreFilter);

        // Plan for active=true (not selective)
        let plan = planner.plan_filters(vec![
            FilterOrder::new("active", "true"),
        ]);
        assert_eq!(plan.strategy, FilterStrategy::PostFilter);
    }

    #[test]
    fn test_combined_selectivity() {
        let planner = QueryPlanner::new();

        // Field A: 10 unique values out of 100 (selectivity = 0.1)
        for i in 0..100 {
            planner.observe_field("field_a", &format!("val_{}", i % 10));
        }

        // Field B: 5 unique values out of 100 (selectivity = 0.2)
        for i in 0..100 {
            planner.observe_field("field_b", &format!("val_{}", i % 5));
        }

        let plan = planner.plan_filters(vec![
            FilterOrder::new("field_a", "val_0"),
            FilterOrder::new("field_b", "val_0"),
        ]);

        // Combined selectivity should be ~0.1 * 0.2 = 0.02
        assert!(plan.total_selectivity < 0.05);
        assert_eq!(plan.strategy, FilterStrategy::PreFilter);
    }
}
