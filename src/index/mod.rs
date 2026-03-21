//! HNSW Index implementation with multiple distance metric support.
//!
//! Supports:
//! - Euclidean (L2) distance
//! - Cosine similarity
//! - Dot product (inner product)
//!
//! Uses enum dispatch pattern to handle different hnsw_rs generic types.
//!
//! ## Dynamic Deletion
//! The `dynamic` submodule provides deletion support with automatic neighbor repair.
//!
//! ## Multi-Vector Search
//! The `multi_vector` submodule enables searching with multiple query vectors.

pub mod dynamic;
pub mod multi_vector;

use std::path::Path;
use anyhow::Result;
use hnsw_rs::prelude::*;
use hnsw_rs::hnswio::HnswIo;
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// Distance metric options for vector similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance - sqrt(sum((a-b)^2))
    Euclidean,
    /// Cosine similarity - 1 - (a·b)/(|a||b|)
    /// Note: For normalized vectors, Cosine ranking == L2 ranking
    Cosine,
    /// Dot product (inner product) - a·b
    /// Note: Maximizes similarity (negative distance)
    DotProduct,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        DistanceMetric::Euclidean
    }
}

impl DistanceMetric {
    /// Parse from string (case-insensitive).
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cosine" | "cos" => DistanceMetric::Cosine,
            "dot" | "dotproduct" | "dot_product" | "inner" | "ip" => DistanceMetric::DotProduct,
            _ => DistanceMetric::Euclidean, // default
        }
    }

    /// Convert to string for serialization.
    pub fn as_str(&self) -> &'static str {
        match self {
            DistanceMetric::Euclidean => "euclidean",
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::DotProduct => "dot",
        }
    }
}

/// Roaring bitmap filter for HNSW search.
pub struct RoaringFilter<'a>(pub &'a roaring::RoaringBitmap);

impl<'a> hnsw_rs::prelude::FilterT for RoaringFilter<'a> {
    fn hnsw_filter(&self, id: &usize) -> bool {
        self.0.contains(*id as u32)
    }
}

/// Enum wrapper for different HNSW index types.
/// This allows runtime selection of distance metric while maintaining type safety.
enum HnswInner {
    L2(Arc<Hnsw<'static, f32, DistL2>>),
    Cosine(Arc<Hnsw<'static, f32, DistCosine>>),
    Dot(Arc<Hnsw<'static, f32, DistDot>>),
}

impl Clone for HnswInner {
    fn clone(&self) -> Self {
        match self {
            HnswInner::L2(h) => HnswInner::L2(Arc::clone(h)),
            HnswInner::Cosine(h) => HnswInner::Cosine(Arc::clone(h)),
            HnswInner::Dot(h) => HnswInner::Dot(Arc::clone(h)),
        }
    }
}

/// HNSW Indexer with multi-distance metric support.
#[derive(Clone)]
pub struct HnswIndexer {
    inner: HnswInner,
    dim: usize,
    metric: DistanceMetric,
}

impl HnswIndexer {
    /// Create a new HNSW index with the specified distance metric.
    pub fn new(dim: usize, max_elements: usize, m: usize, ef_construction: usize) -> Self {
        Self::with_metric(dim, max_elements, m, ef_construction, DistanceMetric::Euclidean)
    }

    /// Create a new HNSW index with the specified distance metric.
    pub fn with_metric(
        dim: usize,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
        metric: DistanceMetric,
    ) -> Self {
        let inner = match metric {
            DistanceMetric::Euclidean => {
                let h: Hnsw<'static, f32, DistL2> = Hnsw::new(
                    m,
                    max_elements,
                    16, // max_layer
                    ef_construction,
                    DistL2 {},
                );
                HnswInner::L2(Arc::new(h))
            }
            DistanceMetric::Cosine => {
                let h: Hnsw<'static, f32, DistCosine> = Hnsw::new(
                    m,
                    max_elements,
                    16,
                    ef_construction,
                    DistCosine {},
                );
                HnswInner::Cosine(Arc::new(h))
            }
            DistanceMetric::DotProduct => {
                let h: Hnsw<'static, f32, DistDot> = Hnsw::new(
                    m,
                    max_elements,
                    16,
                    ef_construction,
                    DistDot {},
                );
                HnswInner::Dot(Arc::new(h))
            }
        };

        Self { inner, dim, metric }
    }

    /// Get the distance metric used by this index.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Get the vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Insert a single vector with its ID.
    pub fn insert(&self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(anyhow::anyhow!("Vector dimension mismatch"));
        }

        match &self.inner {
            HnswInner::L2(h) => h.insert((vector, id as usize)),
            HnswInner::Cosine(h) => h.insert((vector, id as usize)),
            HnswInner::Dot(h) => h.insert((vector, id as usize)),
        }
        Ok(())
    }

    /// Batch insert using parallel_insert for ~10x speedup.
    /// Takes vectors with their pre-assigned IDs.
    pub fn insert_batch(&self, vectors: &[(u64, Vec<f32>)]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        // Validate dimensions
        for (id, v) in vectors {
            if v.len() != self.dim {
                return Err(anyhow::anyhow!(
                    "Vector {} dimension mismatch: expected {}, got {}",
                    id, self.dim, v.len()
                ));
            }
        }

        // Prepare data for parallel_insert
        let data: Vec<(&Vec<f32>, usize)> = vectors
            .iter()
            .map(|(id, v)| (v, *id as usize))
            .collect();

        match &self.inner {
            HnswInner::L2(h) => h.parallel_insert(&data),
            HnswInner::Cosine(h) => h.parallel_insert(&data),
            HnswInner::Dot(h) => h.parallel_insert(&data),
        }
        Ok(())
    }

    /// Search for k nearest neighbors.
    pub fn search(
        &self,
        vector: &[f32],
        k: usize,
        filter: Option<&roaring::RoaringBitmap>,
    ) -> Result<Vec<(u64, f32)>> {
        if vector.len() != self.dim {
            return Err(anyhow::anyhow!("Vector dimension mismatch"));
        }

        let ef_search = 32.max(k);

        let results = match &self.inner {
            HnswInner::L2(h) => {
                if let Some(bitmap) = filter {
                    let f = RoaringFilter(bitmap);
                    h.search_possible_filter(vector, k, ef_search, Some(&f))
                } else {
                    h.search(vector, k, ef_search)
                }
            }
            HnswInner::Cosine(h) => {
                if let Some(bitmap) = filter {
                    let f = RoaringFilter(bitmap);
                    h.search_possible_filter(vector, k, ef_search, Some(&f))
                } else {
                    h.search(vector, k, ef_search)
                }
            }
            HnswInner::Dot(h) => {
                if let Some(bitmap) = filter {
                    let f = RoaringFilter(bitmap);
                    h.search_possible_filter(vector, k, ef_search, Some(&f))
                } else {
                    h.search(vector, k, ef_search)
                }
            }
        };

        let converted = results
            .iter()
            .map(|n| (n.d_id as u64, n.distance))
            .collect();

        Ok(converted)
    }

    /// Save the index to disk.
    pub fn save(&self, path: &Path) -> Result<()> {
        let parent = path.parent().unwrap_or(Path::new("."));
        let filename = path
            .file_name()
            .ok_or(anyhow::anyhow!("Invalid path"))?
            .to_str()
            .ok_or(anyhow::anyhow!("Invalid filename"))?;

        match &self.inner {
            HnswInner::L2(h) => { h.file_dump(parent, filename)?; },
            HnswInner::Cosine(h) => { h.file_dump(parent, filename)?; },
            HnswInner::Dot(h) => { h.file_dump(parent, filename)?; },
        }
        Ok(())
    }

    /// Load an index from disk.
    /// The distance metric must match the saved index.
    pub fn load(path: &Path, dim: usize, metric: DistanceMetric) -> Result<Self> {
        let parent = path.parent().unwrap_or(Path::new("."));
        let filename = path
            .file_name()
            .ok_or(anyhow::anyhow!("Invalid path"))?
            .to_str()
            .ok_or(anyhow::anyhow!("Invalid filename"))?;

        let inner = match metric {
            DistanceMetric::Euclidean => {
                let reloader = Box::new(HnswIo::new(parent, filename));
                let reloader: &'static mut HnswIo = Box::leak(reloader);
                let h: Hnsw<'static, f32, DistL2> = reloader
                    .load_hnsw::<f32, DistL2>()
                    .map_err(|e| anyhow::anyhow!("Failed to load L2 index: {:?}", e))?;
                HnswInner::L2(Arc::new(h))
            }
            DistanceMetric::Cosine => {
                let reloader = Box::new(HnswIo::new(parent, filename));
                let reloader: &'static mut HnswIo = Box::leak(reloader);
                let h: Hnsw<'static, f32, DistCosine> = reloader
                    .load_hnsw::<f32, DistCosine>()
                    .map_err(|e| anyhow::anyhow!("Failed to load Cosine index: {:?}", e))?;
                HnswInner::Cosine(Arc::new(h))
            }
            DistanceMetric::DotProduct => {
                let reloader = Box::new(HnswIo::new(parent, filename));
                let reloader: &'static mut HnswIo = Box::leak(reloader);
                let h: Hnsw<'static, f32, DistDot> = reloader
                    .load_hnsw::<f32, DistDot>()
                    .map_err(|e| anyhow::anyhow!("Failed to load Dot index: {:?}", e))?;
                HnswInner::Dot(Arc::new(h))
            }
        };

        Ok(Self { inner, dim, metric })
    }
}

/// Normalize a vector to unit length (for cosine similarity via L2).
/// Returns the original norm.
#[inline]
pub fn normalize_vector(vector: &mut [f32]) -> f32 {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in vector.iter_mut() {
            *x /= norm;
        }
    }
    norm
}

/// Compute cosine similarity between two vectors.
/// Returns value in range [-1, 1] where 1 means identical direction.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Compute dot product between two vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute L2 (Euclidean) distance between two vectors.
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_metric_parse() {
        assert_eq!(DistanceMetric::from_str("euclidean"), DistanceMetric::Euclidean);
        assert_eq!(DistanceMetric::from_str("cosine"), DistanceMetric::Cosine);
        assert_eq!(DistanceMetric::from_str("dot"), DistanceMetric::DotProduct);
        assert_eq!(DistanceMetric::from_str("unknown"), DistanceMetric::Euclidean);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hnsw_l2() {
        let indexer = HnswIndexer::with_metric(4, 100, 8, 100, DistanceMetric::Euclidean);

        indexer.insert(0, &[0.0, 0.0, 0.0, 0.0]).unwrap();
        indexer.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        indexer.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let results = indexer.search(&[0.1, 0.0, 0.0, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Closest to origin
    }

    #[test]
    fn test_hnsw_cosine() {
        let indexer = HnswIndexer::with_metric(3, 100, 16, 200, DistanceMetric::Cosine);

        // Insert more vectors for better HNSW graph connectivity
        indexer.insert(0, &[1.0, 0.0, 0.0]).unwrap();
        indexer.insert(1, &[0.0, 1.0, 0.0]).unwrap();
        indexer.insert(2, &[0.707, 0.707, 0.0]).unwrap();
        indexer.insert(3, &[0.5, 0.5, 0.707]).unwrap();
        indexer.insert(4, &[-1.0, 0.0, 0.0]).unwrap();

        let results = indexer.search(&[1.0, 0.0, 0.0], 3, None).unwrap();
        // Should find at least some results
        assert!(!results.is_empty());
        // ID 0 should be closest (same direction, cosine distance = 0)
        assert_eq!(results[0].0, 0);
        // Verify the cosine distance is near 0 for identical direction
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_normalize_vector() {
        let mut v = vec![3.0, 4.0];
        let norm = normalize_vector(&mut v);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }
}
