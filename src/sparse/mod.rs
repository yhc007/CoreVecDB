//! Sparse Vector support for hybrid dense+sparse retrieval.
//!
//! Sparse vectors store only non-zero elements, making them efficient for:
//! - Lexical features (BM25, TF-IDF)
//! - Learned sparse representations (SPLADE, DeepImpact)
//! - Keyword matching combined with semantic search
//!
//! ## Storage Format
//! Sparse vectors are stored as sorted (index, value) pairs.
//! This enables efficient dot product computation and compression.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use anyhow::Result;
use serde::{Deserialize, Serialize};

// =============================================================================
// Sparse Vector Types
// =============================================================================

/// A sparse vector represented as sorted (index, value) pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    /// Sorted dimension indices (ascending order)
    pub indices: Vec<u32>,
    /// Values corresponding to each index
    pub values: Vec<f32>,
}

impl SparseVector {
    /// Create a new sparse vector from indices and values.
    /// Indices must be sorted in ascending order.
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Result<Self> {
        if indices.len() != values.len() {
            return Err(anyhow::anyhow!(
                "Indices and values must have same length"
            ));
        }

        // Verify sorted order
        for i in 1..indices.len() {
            if indices[i] <= indices[i - 1] {
                return Err(anyhow::anyhow!(
                    "Indices must be sorted in ascending order with no duplicates"
                ));
            }
        }

        Ok(Self { indices, values })
    }

    /// Create from unsorted (index, value) pairs.
    pub fn from_pairs(mut pairs: Vec<(u32, f32)>) -> Self {
        // Sort by index
        pairs.sort_by_key(|(idx, _)| *idx);

        // Remove zeros and duplicates (keep last value for duplicates)
        let mut indices = Vec::with_capacity(pairs.len());
        let mut values = Vec::with_capacity(pairs.len());

        for (idx, val) in pairs {
            if val.abs() < 1e-10 {
                continue; // Skip zeros
            }
            if let Some(last_idx) = indices.last() {
                if *last_idx == idx {
                    // Duplicate - replace value
                    *values.last_mut().unwrap() = val;
                    continue;
                }
            }
            indices.push(idx);
            values.push(val);
        }

        Self { indices, values }
    }

    /// Create from a HashMap.
    pub fn from_map(map: HashMap<u32, f32>) -> Self {
        Self::from_pairs(map.into_iter().collect())
    }

    /// Create from a dense vector (non-zero elements only).
    pub fn from_dense(dense: &[f32]) -> Self {
        let pairs: Vec<(u32, f32)> = dense
            .iter()
            .enumerate()
            .filter(|(_, &v)| v.abs() > 1e-10)
            .map(|(i, &v)| (i as u32, v))
            .collect();
        Self::from_pairs(pairs)
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Get maximum dimension index.
    pub fn max_dim(&self) -> Option<u32> {
        self.indices.last().copied()
    }

    /// Compute dot product with another sparse vector.
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let mut result = 0.0f32;
        let mut i = 0;
        let mut j = 0;

        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
            }
        }

        result
    }

    /// Compute L2 norm.
    pub fn l2_norm(&self) -> f32 {
        self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Normalize to unit length.
    pub fn normalize(&mut self) {
        let norm = self.l2_norm();
        if norm > 1e-10 {
            for v in &mut self.values {
                *v /= norm;
            }
        }
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4 + self.nnz() * 8);

        // Number of elements (4 bytes)
        bytes.extend_from_slice(&(self.nnz() as u32).to_le_bytes());

        // Indices and values interleaved
        for (&idx, &val) in self.indices.iter().zip(&self.values) {
            bytes.extend_from_slice(&idx.to_le_bytes());
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Err(anyhow::anyhow!("Invalid sparse vector bytes"));
        }

        let nnz = u32::from_le_bytes(bytes[0..4].try_into()?) as usize;

        if bytes.len() != 4 + nnz * 8 {
            return Err(anyhow::anyhow!("Invalid sparse vector byte length"));
        }

        let mut indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for i in 0..nnz {
            let offset = 4 + i * 8;
            let idx = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
            let val = f32::from_le_bytes(bytes[offset + 4..offset + 8].try_into()?);
            indices.push(idx);
            values.push(val);
        }

        Ok(Self { indices, values })
    }
}

// =============================================================================
// Sparse Vector Index (Inverted Index)
// =============================================================================

/// Entry in the inverted index posting list.
#[derive(Debug, Clone)]
struct PostingEntry {
    doc_id: u64,
    weight: f32,
}

/// Inverted index for sparse vector search.
/// Maps dimension -> list of (doc_id, weight).
pub struct SparseIndex {
    /// Inverted index: dimension -> posting list
    postings: RwLock<HashMap<u32, Vec<PostingEntry>>>,
    /// Document vectors for exact scoring
    vectors: RwLock<HashMap<u64, SparseVector>>,
    /// Total document count
    doc_count: std::sync::atomic::AtomicUsize,
}

impl SparseIndex {
    /// Create a new sparse index.
    pub fn new() -> Self {
        Self {
            postings: RwLock::new(HashMap::new()),
            vectors: RwLock::new(HashMap::new()),
            doc_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Insert a sparse vector.
    pub fn insert(&self, id: u64, vector: SparseVector) -> Result<()> {
        // Add to inverted index
        {
            let mut postings = self.postings.write().unwrap();
            for (&idx, &val) in vector.indices.iter().zip(&vector.values) {
                postings
                    .entry(idx)
                    .or_insert_with(Vec::new)
                    .push(PostingEntry {
                        doc_id: id,
                        weight: val,
                    });
            }
        }

        // Store vector for exact scoring
        {
            let mut vectors = self.vectors.write().unwrap();
            if vectors.insert(id, vector).is_none() {
                self.doc_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        Ok(())
    }

    /// Batch insert sparse vectors.
    pub fn insert_batch(&self, vectors: &[(u64, SparseVector)]) -> Result<()> {
        let mut postings = self.postings.write().unwrap();
        let mut stored = self.vectors.write().unwrap();
        let mut new_count = 0;

        for (id, vector) in vectors {
            // Add to inverted index
            for (&idx, &val) in vector.indices.iter().zip(&vector.values) {
                postings
                    .entry(idx)
                    .or_insert_with(Vec::new)
                    .push(PostingEntry {
                        doc_id: *id,
                        weight: val,
                    });
            }

            // Store vector
            if stored.insert(*id, vector.clone()).is_none() {
                new_count += 1;
            }
        }

        self.doc_count.fetch_add(new_count, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Search for top-k similar vectors using dot product.
    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<(u64, f32)> {
        let postings = self.postings.read().unwrap();

        // Accumulate scores for each candidate
        let mut scores: HashMap<u64, f32> = HashMap::new();

        for (&query_idx, &query_val) in query.indices.iter().zip(&query.values) {
            if let Some(posting_list) = postings.get(&query_idx) {
                for entry in posting_list {
                    *scores.entry(entry.doc_id).or_insert(0.0) +=
                        query_val * entry.weight;
                }
            }
        }

        // Get top-k
        let mut results: Vec<(u64, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        results
    }

    /// Search with filter.
    pub fn search_filtered(
        &self,
        query: &SparseVector,
        k: usize,
        filter: &roaring::RoaringBitmap,
    ) -> Vec<(u64, f32)> {
        let postings = self.postings.read().unwrap();

        let mut scores: HashMap<u64, f32> = HashMap::new();

        for (&query_idx, &query_val) in query.indices.iter().zip(&query.values) {
            if let Some(posting_list) = postings.get(&query_idx) {
                for entry in posting_list {
                    // Apply filter
                    if filter.contains(entry.doc_id as u32) {
                        *scores.entry(entry.doc_id).or_insert(0.0) +=
                            query_val * entry.weight;
                    }
                }
            }
        }

        let mut results: Vec<(u64, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        results
    }

    /// Get a sparse vector by ID.
    pub fn get(&self, id: u64) -> Option<SparseVector> {
        self.vectors.read().unwrap().get(&id).cloned()
    }

    /// Delete a vector from the index.
    pub fn delete(&self, id: u64) -> bool {
        let removed = {
            let mut vectors = self.vectors.write().unwrap();
            vectors.remove(&id)
        };

        if let Some(vector) = removed {
            // Remove from postings
            let mut postings = self.postings.write().unwrap();
            for &idx in &vector.indices {
                if let Some(list) = postings.get_mut(&idx) {
                    list.retain(|e| e.doc_id != id);
                }
            }
            self.doc_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Get document count.
    pub fn len(&self) -> usize {
        self.doc_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get statistics.
    pub fn stats(&self) -> SparseIndexStats {
        let postings = self.postings.read().unwrap();
        let vectors = self.vectors.read().unwrap();

        let total_postings: usize = postings.values().map(|v| v.len()).sum();
        let unique_dims = postings.len();
        let total_nnz: usize = vectors.values().map(|v| v.nnz()).sum();
        let avg_nnz = if vectors.is_empty() {
            0.0
        } else {
            total_nnz as f64 / vectors.len() as f64
        };

        SparseIndexStats {
            doc_count: vectors.len(),
            unique_dimensions: unique_dims,
            total_postings,
            avg_nnz_per_vector: avg_nnz,
        }
    }
}

impl Default for SparseIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for sparse index.
#[derive(Debug, Clone, Serialize)]
pub struct SparseIndexStats {
    pub doc_count: usize,
    pub unique_dimensions: usize,
    pub total_postings: usize,
    pub avg_nnz_per_vector: f64,
}

// =============================================================================
// Hybrid Dense + Sparse Search
// =============================================================================

/// Configuration for hybrid search.
#[derive(Debug, Clone)]
pub struct HybridSearchConfig {
    /// Weight for dense vector score (0.0 to 1.0)
    pub dense_weight: f32,
    /// Weight for sparse vector score (0.0 to 1.0)
    pub sparse_weight: f32,
    /// Fusion method
    pub fusion: HybridFusion,
    /// Oversample factor for candidate retrieval
    pub oversample: usize,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            dense_weight: 0.5,
            sparse_weight: 0.5,
            fusion: HybridFusion::Weighted,
            oversample: 3,
        }
    }
}

/// Fusion method for hybrid search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridFusion {
    /// Weighted linear combination
    Weighted,
    /// Reciprocal Rank Fusion
    RRF,
    /// Max score
    Max,
}

impl HybridFusion {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "weighted" | "linear" => HybridFusion::Weighted,
            "rrf" | "reciprocal_rank_fusion" => HybridFusion::RRF,
            "max" | "maximum" => HybridFusion::Max,
            _ => HybridFusion::Weighted,
        }
    }
}

/// Result from hybrid search.
#[derive(Debug, Clone, Serialize)]
pub struct HybridResult {
    pub id: u64,
    pub score: f32,
    pub dense_score: Option<f32>,
    pub sparse_score: Option<f32>,
}

/// Fuse dense and sparse search results.
pub fn fuse_hybrid_results(
    dense_results: &[(u64, f32)],
    sparse_results: &[(u64, f32)],
    config: &HybridSearchConfig,
    k: usize,
) -> Vec<HybridResult> {
    match config.fusion {
        HybridFusion::Weighted => {
            fuse_weighted(dense_results, sparse_results, config, k)
        }
        HybridFusion::RRF => {
            fuse_rrf(dense_results, sparse_results, config, k)
        }
        HybridFusion::Max => {
            fuse_max(dense_results, sparse_results, k)
        }
    }
}

fn fuse_weighted(
    dense_results: &[(u64, f32)],
    sparse_results: &[(u64, f32)],
    config: &HybridSearchConfig,
    k: usize,
) -> Vec<HybridResult> {
    // Normalize scores to [0, 1] range
    let dense_max = dense_results.iter().map(|(_, s)| *s).fold(f32::MIN, f32::max);
    let dense_min = dense_results.iter().map(|(_, s)| *s).fold(f32::MAX, f32::min);
    let sparse_max = sparse_results.iter().map(|(_, s)| *s).fold(f32::MIN, f32::max);
    let sparse_min = sparse_results.iter().map(|(_, s)| *s).fold(f32::MAX, f32::min);

    let normalize_dense = |s: f32| -> f32 {
        if dense_max > dense_min {
            // For distance metrics (lower is better), invert
            1.0 - (s - dense_min) / (dense_max - dense_min)
        } else {
            1.0
        }
    };

    let normalize_sparse = |s: f32| -> f32 {
        if sparse_max > sparse_min {
            // For similarity metrics (higher is better)
            (s - sparse_min) / (sparse_max - sparse_min)
        } else {
            1.0
        }
    };

    // Collect all candidates
    let mut scores: HashMap<u64, (Option<f32>, Option<f32>)> = HashMap::new();

    for &(id, score) in dense_results {
        scores.entry(id).or_insert((None, None)).0 = Some(score);
    }
    for &(id, score) in sparse_results {
        scores.entry(id).or_insert((None, None)).1 = Some(score);
    }

    // Compute weighted scores
    let mut results: Vec<HybridResult> = scores
        .into_iter()
        .map(|(id, (dense, sparse))| {
            let dense_norm = dense.map(normalize_dense).unwrap_or(0.0);
            let sparse_norm = sparse.map(normalize_sparse).unwrap_or(0.0);

            let score = config.dense_weight * dense_norm
                      + config.sparse_weight * sparse_norm;

            HybridResult {
                id,
                score,
                dense_score: dense,
                sparse_score: sparse,
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

fn fuse_rrf(
    dense_results: &[(u64, f32)],
    sparse_results: &[(u64, f32)],
    config: &HybridSearchConfig,
    k: usize,
) -> Vec<HybridResult> {
    const RRF_K: f32 = 60.0;

    // Build rank maps
    let dense_ranks: HashMap<u64, usize> = dense_results
        .iter()
        .enumerate()
        .map(|(rank, (id, _))| (*id, rank))
        .collect();

    let sparse_ranks: HashMap<u64, usize> = sparse_results
        .iter()
        .enumerate()
        .map(|(rank, (id, _))| (*id, rank))
        .collect();

    // Build score maps for reporting
    let dense_scores: HashMap<u64, f32> = dense_results.iter().cloned().collect();
    let sparse_scores: HashMap<u64, f32> = sparse_results.iter().cloned().collect();

    // Collect all candidates
    let mut all_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
    all_ids.extend(dense_ranks.keys());
    all_ids.extend(sparse_ranks.keys());

    // Compute RRF scores
    let mut results: Vec<HybridResult> = all_ids
        .into_iter()
        .map(|id| {
            let dense_rrf = dense_ranks
                .get(&id)
                .map(|&rank| config.dense_weight / (RRF_K + rank as f32 + 1.0))
                .unwrap_or(0.0);

            let sparse_rrf = sparse_ranks
                .get(&id)
                .map(|&rank| config.sparse_weight / (RRF_K + rank as f32 + 1.0))
                .unwrap_or(0.0);

            HybridResult {
                id,
                score: dense_rrf + sparse_rrf,
                dense_score: dense_scores.get(&id).copied(),
                sparse_score: sparse_scores.get(&id).copied(),
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

fn fuse_max(
    dense_results: &[(u64, f32)],
    sparse_results: &[(u64, f32)],
    k: usize,
) -> Vec<HybridResult> {
    // Normalize to comparable scale
    let dense_max = dense_results.iter().map(|(_, s)| *s).fold(f32::MIN, f32::max);
    let sparse_max = sparse_results.iter().map(|(_, s)| *s).fold(f32::MIN, f32::max);

    let normalize_dense = |s: f32| -> f32 {
        if dense_max > 0.0 { 1.0 - s / dense_max } else { 0.0 }
    };

    let normalize_sparse = |s: f32| -> f32 {
        if sparse_max > 0.0 { s / sparse_max } else { 0.0 }
    };

    let mut scores: HashMap<u64, (Option<f32>, Option<f32>)> = HashMap::new();

    for &(id, score) in dense_results {
        scores.entry(id).or_insert((None, None)).0 = Some(score);
    }
    for &(id, score) in sparse_results {
        scores.entry(id).or_insert((None, None)).1 = Some(score);
    }

    let mut results: Vec<HybridResult> = scores
        .into_iter()
        .map(|(id, (dense, sparse))| {
            let dense_norm = dense.map(normalize_dense).unwrap_or(0.0);
            let sparse_norm = sparse.map(normalize_sparse).unwrap_or(0.0);

            HybridResult {
                id,
                score: dense_norm.max(sparse_norm),
                dense_score: dense,
                sparse_score: sparse,
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

// =============================================================================
// Thread-Safe Wrapper
// =============================================================================

/// Thread-safe sparse index wrapper.
pub struct ThreadSafeSparseIndex {
    inner: Arc<SparseIndex>,
}

impl ThreadSafeSparseIndex {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(SparseIndex::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: SparseVector) -> Result<()> {
        self.inner.insert(id, vector)
    }

    pub fn insert_batch(&self, vectors: &[(u64, SparseVector)]) -> Result<()> {
        self.inner.insert_batch(vectors)
    }

    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<(u64, f32)> {
        self.inner.search(query, k)
    }

    pub fn search_filtered(
        &self,
        query: &SparseVector,
        k: usize,
        filter: &roaring::RoaringBitmap,
    ) -> Vec<(u64, f32)> {
        self.inner.search_filtered(query, k, filter)
    }

    pub fn get(&self, id: u64) -> Option<SparseVector> {
        self.inner.get(id)
    }

    pub fn delete(&self, id: u64) -> bool {
        self.inner.delete(id)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn stats(&self) -> SparseIndexStats {
        self.inner.stats()
    }
}

impl Default for ThreadSafeSparseIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ThreadSafeSparseIndex {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vector_creation() {
        let sv = SparseVector::new(vec![0, 5, 10], vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(sv.nnz(), 3);
        assert_eq!(sv.max_dim(), Some(10));
    }

    #[test]
    fn test_sparse_vector_from_pairs() {
        let pairs = vec![(10, 1.0), (5, 2.0), (0, 3.0)];
        let sv = SparseVector::from_pairs(pairs);

        // Should be sorted
        assert_eq!(sv.indices, vec![0, 5, 10]);
        assert_eq!(sv.values, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_sparse_vector_from_dense() {
        let dense = vec![0.0, 1.0, 0.0, 2.0, 0.0];
        let sv = SparseVector::from_dense(&dense);

        assert_eq!(sv.indices, vec![1, 3]);
        assert_eq!(sv.values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_sparse_dot_product() {
        let a = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]).unwrap();
        let b = SparseVector::new(vec![1, 2, 4], vec![1.0, 3.0, 2.0]).unwrap();

        // Dot = 0 + 2*3 + 3*2 = 12
        let dot = a.dot(&b);
        assert!((dot - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_serialization() {
        let sv = SparseVector::new(vec![0, 5, 10], vec![1.0, 2.0, 3.0]).unwrap();
        let bytes = sv.to_bytes();
        let restored = SparseVector::from_bytes(&bytes).unwrap();

        assert_eq!(sv.indices, restored.indices);
        assert_eq!(sv.values, restored.values);
    }

    #[test]
    fn test_sparse_index_search() {
        let index = SparseIndex::new();

        // Insert documents
        index.insert(0, SparseVector::from_pairs(vec![
            (0, 1.0), (1, 2.0), (2, 1.0)
        ])).unwrap();
        index.insert(1, SparseVector::from_pairs(vec![
            (1, 1.0), (2, 3.0), (3, 1.0)
        ])).unwrap();
        index.insert(2, SparseVector::from_pairs(vec![
            (0, 2.0), (3, 2.0)
        ])).unwrap();

        // Query: emphasize dimensions 0 and 1
        let query = SparseVector::from_pairs(vec![(0, 1.0), (1, 1.0)]);
        let results = index.search(&query, 3);

        // Doc 0 should rank highest (matches both dimensions)
        assert_eq!(results[0].0, 0);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_hybrid_fusion_weighted() {
        let dense = vec![(0, 0.1), (1, 0.5), (2, 0.3)];  // Lower = better (distance)
        let sparse = vec![(0, 0.8), (2, 0.6), (3, 0.4)];  // Higher = better (similarity)

        let config = HybridSearchConfig {
            dense_weight: 0.5,
            sparse_weight: 0.5,
            fusion: HybridFusion::Weighted,
            oversample: 2,
        };

        let results = fuse_hybrid_results(&dense, &sparse, &config, 4);

        // Should have 4 unique IDs
        assert_eq!(results.len(), 4);
        // All should have scores
        assert!(results.iter().all(|r| r.score >= 0.0));
    }

    #[test]
    fn test_hybrid_fusion_rrf() {
        let dense = vec![(0, 0.1), (1, 0.5), (2, 0.3)];
        let sparse = vec![(0, 0.8), (2, 0.6), (3, 0.4)];

        let config = HybridSearchConfig {
            dense_weight: 1.0,
            sparse_weight: 1.0,
            fusion: HybridFusion::RRF,
            oversample: 2,
        };

        let results = fuse_hybrid_results(&dense, &sparse, &config, 4);

        // ID 0 should rank high (appears in both lists at top positions)
        let id0_result = results.iter().find(|r| r.id == 0).unwrap();
        assert!(id0_result.score > 0.0);
    }

    #[test]
    fn test_sparse_delete() {
        let index = SparseIndex::new();

        index.insert(0, SparseVector::from_pairs(vec![(0, 1.0)])).unwrap();
        index.insert(1, SparseVector::from_pairs(vec![(0, 2.0)])).unwrap();

        assert_eq!(index.len(), 2);

        index.delete(0);
        assert_eq!(index.len(), 1);
        assert!(index.get(0).is_none());
        assert!(index.get(1).is_some());
    }
}
