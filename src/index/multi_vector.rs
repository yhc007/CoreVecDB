//! Multi-Vector Search for querying with multiple vectors simultaneously.
//!
//! Use cases:
//! - **Query Expansion**: Search with multiple representations of a concept
//! - **Multi-modal Search**: Combine text and image embeddings
//! - **Ensemble Search**: Use multiple embedding models
//! - **ColBERT-style**: Late interaction with token-level vectors
//!
//! Fusion methods:
//! - **Sum**: Add scores from all queries
//! - **Max**: Take maximum score across queries
//! - **Average**: Average scores across queries
//! - **RRF**: Reciprocal Rank Fusion (robust to score scale differences)
//! - **Weighted**: Weighted combination with per-query weights

use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Fusion method for combining multi-vector search results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionMethod {
    /// Sum scores from all queries (good when scores are comparable)
    Sum,
    /// Take maximum score across queries (good for OR-style semantics)
    Max,
    /// Average scores across queries
    Average,
    /// Reciprocal Rank Fusion - robust to different score scales
    /// score = sum(1 / (k + rank_i)) where k is typically 60
    RRF,
    /// Minimum score (AND-style semantics - must match all queries well)
    Min,
}

impl Default for FusionMethod {
    fn default() -> Self {
        FusionMethod::RRF
    }
}

impl FusionMethod {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "sum" | "add" => FusionMethod::Sum,
            "max" | "maximum" => FusionMethod::Max,
            "avg" | "average" | "mean" => FusionMethod::Average,
            "rrf" | "reciprocal_rank_fusion" => FusionMethod::RRF,
            "min" | "minimum" => FusionMethod::Min,
            _ => FusionMethod::RRF,
        }
    }
}

/// Multi-vector query configuration.
#[derive(Debug, Clone)]
pub struct MultiVectorQuery {
    /// Query vectors
    pub vectors: Vec<Vec<f32>>,
    /// Optional weights for each vector (used with weighted fusion)
    pub weights: Option<Vec<f32>>,
    /// Number of results to return
    pub k: usize,
    /// Fusion method
    pub fusion: FusionMethod,
    /// RRF k parameter (default 60)
    pub rrf_k: f32,
    /// Oversample factor - fetch more candidates per query for better fusion
    pub oversample: usize,
}

impl MultiVectorQuery {
    /// Create a new multi-vector query.
    pub fn new(vectors: Vec<Vec<f32>>, k: usize) -> Self {
        Self {
            vectors,
            weights: None,
            k,
            fusion: FusionMethod::RRF,
            rrf_k: 60.0,
            oversample: 2,
        }
    }

    /// Set fusion method.
    pub fn with_fusion(mut self, fusion: FusionMethod) -> Self {
        self.fusion = fusion;
        self
    }

    /// Set weights for weighted fusion.
    pub fn with_weights(mut self, weights: Vec<f32>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set RRF k parameter.
    pub fn with_rrf_k(mut self, k: f32) -> Self {
        self.rrf_k = k;
        self
    }

    /// Set oversample factor.
    pub fn with_oversample(mut self, factor: usize) -> Self {
        self.oversample = factor;
        self
    }

    /// Get the number of candidates to fetch per query.
    pub fn candidates_per_query(&self) -> usize {
        self.k * self.oversample
    }
}

/// Result from multi-vector search.
#[derive(Debug, Clone)]
pub struct MultiVectorResult {
    /// Vector ID
    pub id: u64,
    /// Fused score
    pub score: f32,
    /// Individual scores from each query vector (for debugging)
    pub per_query_scores: Vec<Option<f32>>,
    /// Number of queries that matched this result
    pub query_matches: usize,
}

/// Fuse results from multiple single-vector searches.
pub fn fuse_results(
    results_per_query: Vec<Vec<(u64, f32)>>,
    query: &MultiVectorQuery,
) -> Vec<MultiVectorResult> {
    if results_per_query.is_empty() {
        return Vec::new();
    }

    let num_queries = results_per_query.len();

    // Collect all unique IDs and their scores per query
    let mut id_scores: HashMap<u64, Vec<Option<f32>>> = HashMap::new();
    let mut id_ranks: HashMap<u64, Vec<Option<usize>>> = HashMap::new();

    for (query_idx, results) in results_per_query.iter().enumerate() {
        for (rank, &(id, score)) in results.iter().enumerate() {
            // Initialize vectors if first time seeing this ID
            id_scores.entry(id).or_insert_with(|| vec![None; num_queries]);
            id_ranks.entry(id).or_insert_with(|| vec![None; num_queries]);

            // Record score and rank for this query
            id_scores.get_mut(&id).unwrap()[query_idx] = Some(score);
            id_ranks.get_mut(&id).unwrap()[query_idx] = Some(rank);
        }
    }

    // Compute fused scores
    let weights = query.weights.as_ref();
    let mut fused_results: Vec<MultiVectorResult> = id_scores
        .iter()
        .map(|(&id, scores)| {
            let (fused_score, query_matches) = compute_fused_score(
                scores,
                id_ranks.get(&id),
                query.fusion,
                query.rrf_k,
                weights,
            );

            MultiVectorResult {
                id,
                score: fused_score,
                per_query_scores: scores.clone(),
                query_matches,
            }
        })
        .collect();

    // Sort by fused score
    // For distance-based metrics (lower is better), sort ascending
    // For RRF/Sum (higher is better), sort descending
    match query.fusion {
        FusionMethod::RRF | FusionMethod::Sum | FusionMethod::Max => {
            // Higher is better
            fused_results.sort_by(|a, b| {
                b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        FusionMethod::Average | FusionMethod::Min => {
            // Lower is better (distance-like)
            fused_results.sort_by(|a, b| {
                a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    // Return top k
    fused_results.truncate(query.k);
    fused_results
}

/// Compute fused score from per-query scores.
fn compute_fused_score(
    scores: &[Option<f32>],
    ranks: Option<&Vec<Option<usize>>>,
    method: FusionMethod,
    rrf_k: f32,
    weights: Option<&Vec<f32>>,
) -> (f32, usize) {
    let valid_scores: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.map(|score| (i, score)))
        .collect();

    let query_matches = valid_scores.len();

    if valid_scores.is_empty() {
        return (f32::MAX, 0);
    }

    let fused = match method {
        FusionMethod::Sum => {
            if let Some(w) = weights {
                valid_scores.iter().map(|&(i, s)| s * w.get(i).unwrap_or(&1.0)).sum()
            } else {
                valid_scores.iter().map(|(_, s)| s).sum()
            }
        }
        FusionMethod::Max => {
            valid_scores.iter().map(|(_, s)| *s).fold(f32::MIN, f32::max)
        }
        FusionMethod::Min => {
            valid_scores.iter().map(|(_, s)| *s).fold(f32::MAX, f32::min)
        }
        FusionMethod::Average => {
            let sum: f32 = valid_scores.iter().map(|(_, s)| s).sum();
            sum / valid_scores.len() as f32
        }
        FusionMethod::RRF => {
            // RRF uses ranks, not scores
            if let Some(ranks) = ranks {
                let mut rrf_score = 0.0f32;
                for (i, rank_opt) in ranks.iter().enumerate() {
                    if let Some(rank) = rank_opt {
                        let weight = weights.map(|w| w.get(i).unwrap_or(&1.0)).unwrap_or(&1.0);
                        rrf_score += weight / (rrf_k + *rank as f32 + 1.0);
                    }
                }
                rrf_score
            } else {
                // Fallback to sum if ranks not available
                valid_scores.iter().map(|(_, s)| s).sum()
            }
        }
    };

    (fused, query_matches)
}

/// Convert multi-vector results to simple (id, score) pairs.
pub fn to_simple_results(results: &[MultiVectorResult]) -> Vec<(u64, f32)> {
    results.iter().map(|r| (r.id, r.score)).collect()
}

// =============================================================================
// Parallel Multi-Vector Search
// =============================================================================

use rayon::prelude::*;
use super::HnswIndexer;

/// Perform multi-vector search in parallel.
pub fn parallel_multi_search(
    indexer: &HnswIndexer,
    query: &MultiVectorQuery,
    filter: Option<&roaring::RoaringBitmap>,
) -> Result<Vec<MultiVectorResult>> {
    let candidates_per_query = query.candidates_per_query();

    // Search in parallel for each query vector
    let results_per_query: Vec<Vec<(u64, f32)>> = query
        .vectors
        .par_iter()
        .map(|vector| {
            indexer.search(vector, candidates_per_query, filter)
                .unwrap_or_default()
        })
        .collect();

    // Fuse results
    Ok(fuse_results(results_per_query, query))
}

/// Multi-vector search with different weights per query.
pub fn weighted_multi_search(
    indexer: &HnswIndexer,
    vectors: &[Vec<f32>],
    weights: &[f32],
    k: usize,
    filter: Option<&roaring::RoaringBitmap>,
) -> Result<Vec<(u64, f32)>> {
    let query = MultiVectorQuery::new(vectors.to_vec(), k)
        .with_weights(weights.to_vec())
        .with_fusion(FusionMethod::Sum);

    let results = parallel_multi_search(indexer, &query, filter)?;
    Ok(to_simple_results(&results))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_sum() {
        let results = vec![
            vec![(1, 0.1), (2, 0.2), (3, 0.3)],
            vec![(1, 0.15), (3, 0.25), (4, 0.35)],
        ];

        let query = MultiVectorQuery::new(vec![vec![], vec![]], 3)
            .with_fusion(FusionMethod::Sum);

        let fused = fuse_results(results, &query);

        // ID 1 appears in both: 0.1 + 0.15 = 0.25
        // ID 3 appears in both: 0.3 + 0.25 = 0.55
        assert!(fused.iter().any(|r| r.id == 1));
        assert!(fused.iter().any(|r| r.id == 3));
    }

    #[test]
    fn test_fusion_max() {
        let results = vec![
            vec![(1, 0.5), (2, 0.3)],
            vec![(1, 0.8), (3, 0.6)],
        ];

        let query = MultiVectorQuery::new(vec![vec![], vec![]], 3)
            .with_fusion(FusionMethod::Max);

        let fused = fuse_results(results, &query);

        // ID 1's max score should be 0.8
        let id1_result = fused.iter().find(|r| r.id == 1).unwrap();
        assert!((id1_result.score - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_fusion_rrf() {
        let results = vec![
            vec![(1, 0.1), (2, 0.2), (3, 0.3)],  // Ranks: 1->0, 2->1, 3->2
            vec![(3, 0.05), (1, 0.1), (4, 0.15)], // Ranks: 3->0, 1->1, 4->2
        ];

        let query = MultiVectorQuery::new(vec![vec![], vec![]], 3)
            .with_fusion(FusionMethod::RRF)
            .with_rrf_k(60.0);

        let fused = fuse_results(results, &query);

        // Both ID 1 and 3 appear in both result sets
        let id1 = fused.iter().find(|r| r.id == 1).unwrap();
        let id3 = fused.iter().find(|r| r.id == 3).unwrap();

        // ID 3: rank 2 in query 0, rank 0 in query 1 -> 1/63 + 1/61
        // ID 1: rank 0 in query 0, rank 1 in query 1 -> 1/61 + 1/62
        assert!(id1.score > 0.0);
        assert!(id3.score > 0.0);
    }

    #[test]
    fn test_fusion_weighted() {
        let results = vec![
            vec![(1, 1.0)],
            vec![(1, 1.0)],
        ];

        let query = MultiVectorQuery::new(vec![vec![], vec![]], 1)
            .with_fusion(FusionMethod::Sum)
            .with_weights(vec![0.7, 0.3]);

        let fused = fuse_results(results, &query);

        // 1.0 * 0.7 + 1.0 * 0.3 = 1.0
        assert!((fused[0].score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_query_matches() {
        let results = vec![
            vec![(1, 0.1), (2, 0.2)],
            vec![(1, 0.15), (3, 0.25)],
            vec![(1, 0.12), (4, 0.22)],
        ];

        let query = MultiVectorQuery::new(vec![vec![], vec![], vec![]], 4)
            .with_fusion(FusionMethod::Sum);

        let fused = fuse_results(results, &query);

        // ID 1 appears in all 3 queries
        let id1 = fused.iter().find(|r| r.id == 1).unwrap();
        assert_eq!(id1.query_matches, 3);

        // ID 2 appears in only 1 query
        let id2 = fused.iter().find(|r| r.id == 2).unwrap();
        assert_eq!(id2.query_matches, 1);
    }

    #[test]
    fn test_to_simple_results() {
        let results = vec![
            MultiVectorResult {
                id: 1,
                score: 0.5,
                per_query_scores: vec![Some(0.5)],
                query_matches: 1,
            },
            MultiVectorResult {
                id: 2,
                score: 0.3,
                per_query_scores: vec![Some(0.3)],
                query_matches: 1,
            },
        ];

        let simple = to_simple_results(&results);
        assert_eq!(simple.len(), 2);
        assert_eq!(simple[0], (1, 0.5));
        assert_eq!(simple[1], (2, 0.3));
    }
}
