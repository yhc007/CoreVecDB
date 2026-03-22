//! Query Planning and Optimization for VectorDB.
//!
//! Provides intelligent query planning based on filter selectivity estimation.

pub mod planner;

pub use planner::{QueryPlanner, FieldStatistics, FilterPlan, FilterOrder};
