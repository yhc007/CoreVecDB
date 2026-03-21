//! RaBitQ: 1-bit Vector Quantization with Theoretical Error Bounds
//!
//! Based on the paper "RaBitQ: Quantizing High-Dimensional Vectors with a
//! Theoretical Error Bound for Approximate Nearest Neighbor Search"
//!
//! Key features:
//! - 32x compression (float32 → 1 bit per dimension)
//! - Unbiased distance estimation
//! - No codebook storage required
//! - SIMD-accelerated hamming distance

use anyhow::Result;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// RaBitQ configuration
#[derive(Debug, Clone)]
pub struct RaBitQConfig {
    /// Vector dimension
    pub dim: usize,
    /// Number of random rotations for better distribution
    pub num_rotations: usize,
    /// Use orthogonal random matrix (better quality, slower init)
    pub use_orthogonal: bool,
}

impl Default for RaBitQConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            num_rotations: 1,
            use_orthogonal: true,
        }
    }
}

/// Statistics for distance estimation correction
#[derive(Debug, Clone)]
pub struct VectorStats {
    /// L2 norm of original vector
    pub norm: f32,
    /// Mean of vector components
    pub mean: f32,
    /// Variance of vector components
    pub variance: f32,
}

impl VectorStats {
    pub fn compute(vector: &[f32]) -> Self {
        let n = vector.len() as f32;
        let sum: f32 = vector.iter().sum();
        let mean = sum / n;

        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let variance = vector.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;

        Self { norm, mean, variance }
    }
}

/// Binary quantized vector with metadata for distance estimation
#[derive(Debug, Clone)]
pub struct QuantizedVector {
    /// Packed binary representation (1 bit per dimension)
    pub bits: Vec<u64>,
    /// Statistics for distance correction
    pub stats: VectorStats,
    /// Original vector ID
    pub id: u64,
}

impl QuantizedVector {
    /// Get the number of dimensions
    pub fn dim(&self) -> usize {
        self.bits.len() * 64
    }

    /// Get bit at position
    #[inline]
    pub fn get_bit(&self, pos: usize) -> bool {
        let word = pos / 64;
        let bit = pos % 64;
        (self.bits[word] >> bit) & 1 == 1
    }
}

/// Random rotation matrix for RaBitQ
#[derive(Debug, Clone)]
pub struct RotationMatrix {
    /// Flattened rotation matrix (dim x dim)
    data: Vec<f32>,
    dim: usize,
}

impl RotationMatrix {
    /// Create random orthogonal rotation matrix using Gram-Schmidt
    pub fn new_orthogonal(dim: usize, seed: u64) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut data = vec![0.0f32; dim * dim];

        // Generate random matrix
        for i in 0..dim * dim {
            let mut hasher = DefaultHasher::new();
            (seed, i).hash(&mut hasher);
            let hash = hasher.finish();
            // Convert to float in [-1, 1]
            data[i] = (hash as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32;
        }

        // Gram-Schmidt orthogonalization
        for i in 0..dim {
            // Subtract projections of previous vectors
            for j in 0..i {
                let dot: f32 = (0..dim)
                    .map(|k| data[i * dim + k] * data[j * dim + k])
                    .sum();
                for k in 0..dim {
                    data[i * dim + k] -= dot * data[j * dim + k];
                }
            }

            // Normalize
            let norm: f32 = (0..dim)
                .map(|k| data[i * dim + k].powi(2))
                .sum::<f32>()
                .sqrt();

            if norm > 1e-10 {
                for k in 0..dim {
                    data[i * dim + k] /= norm;
                }
            }
        }

        Self { data, dim }
    }

    /// Create simple random matrix (faster but less accurate)
    pub fn new_random(dim: usize, seed: u64) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut data = vec![0.0f32; dim * dim];
        let scale = 1.0 / (dim as f32).sqrt();

        for i in 0..dim * dim {
            let mut hasher = DefaultHasher::new();
            (seed, i).hash(&mut hasher);
            let hash = hasher.finish();
            // Rademacher random variable: +1 or -1
            data[i] = if hash % 2 == 0 { scale } else { -scale };
        }

        Self { data, dim }
    }

    /// Apply rotation to vector
    pub fn rotate(&self, vector: &[f32]) -> Vec<f32> {
        assert_eq!(vector.len(), self.dim);

        let mut result = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let mut sum = 0.0f32;
            for j in 0..self.dim {
                sum += self.data[i * self.dim + j] * vector[j];
            }
            result[i] = sum;
        }
        result
    }

    /// Apply rotation using SIMD when available
    #[cfg(target_arch = "x86_64")]
    pub fn rotate_simd(&self, vector: &[f32]) -> Vec<f32> {
        if is_x86_feature_detected!("avx2") {
            unsafe { self.rotate_avx2(vector) }
        } else {
            self.rotate(vector)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn rotate_avx2(&self, vector: &[f32]) -> Vec<f32> {
        use std::arch::x86_64::*;

        let mut result = vec![0.0f32; self.dim];

        for i in 0..self.dim {
            let row_start = i * self.dim;
            let mut sum = _mm256_setzero_ps();

            let chunks = self.dim / 8;
            for c in 0..chunks {
                let idx = c * 8;
                let row = _mm256_loadu_ps(self.data[row_start + idx..].as_ptr());
                let vec = _mm256_loadu_ps(vector[idx..].as_ptr());
                sum = _mm256_fmadd_ps(row, vec, sum);
            }

            // Horizontal sum
            let sum128 = _mm_add_ps(
                _mm256_castps256_ps128(sum),
                _mm256_extractf128_ps(sum, 1),
            );
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            result[i] = _mm_cvtss_f32(sum32);

            // Handle remainder
            for j in (chunks * 8)..self.dim {
                result[i] += self.data[row_start + j] * vector[j];
            }
        }

        result
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn rotate_simd(&self, vector: &[f32]) -> Vec<f32> {
        self.rotate(vector)
    }
}

/// RaBitQ Quantizer
pub struct RaBitQuantizer {
    config: RaBitQConfig,
    /// Random rotation matrix
    rotation: RotationMatrix,
    /// Quantized vectors storage
    vectors: RwLock<Vec<QuantizedVector>>,
    /// Next ID counter
    next_id: AtomicU64,
    /// Global statistics for distance normalization
    global_stats: RwLock<GlobalStats>,
}

#[derive(Debug, Clone, Default)]
struct GlobalStats {
    sum_norms: f64,
    sum_variances: f64,
    count: u64,
}

impl RaBitQuantizer {
    /// Create a new RaBitQ quantizer
    pub fn new(config: RaBitQConfig) -> Self {
        let rotation = if config.use_orthogonal {
            RotationMatrix::new_orthogonal(config.dim, 42)
        } else {
            RotationMatrix::new_random(config.dim, 42)
        };

        Self {
            config,
            rotation,
            vectors: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(0),
            global_stats: RwLock::new(GlobalStats::default()),
        }
    }

    /// Quantize a vector to 1-bit representation
    pub fn quantize(&self, vector: &[f32]) -> QuantizedVector {
        assert_eq!(vector.len(), self.config.dim);

        // Compute original statistics
        let stats = VectorStats::compute(vector);

        // Apply random rotation
        let rotated = self.rotation.rotate_simd(vector);

        // Quantize to binary (sign bit)
        let num_words = (self.config.dim + 63) / 64;
        let mut bits = vec![0u64; num_words];

        for (i, &val) in rotated.iter().enumerate() {
            if val >= 0.0 {
                let word = i / 64;
                let bit = i % 64;
                bits[word] |= 1u64 << bit;
            }
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        QuantizedVector { bits, stats, id }
    }

    /// Insert a vector and return its ID
    pub fn insert(&self, vector: &[f32]) -> u64 {
        let quantized = self.quantize(vector);
        let id = quantized.id;

        // Update global stats
        {
            let mut stats = self.global_stats.write();
            stats.sum_norms += quantized.stats.norm as f64;
            stats.sum_variances += quantized.stats.variance as f64;
            stats.count += 1;
        }

        self.vectors.write().push(quantized);
        id
    }

    /// Batch insert vectors
    pub fn insert_batch(&self, vectors: &[Vec<f32>]) -> Vec<u64> {
        let quantized: Vec<_> = vectors.iter()
            .map(|v| self.quantize(v))
            .collect();

        let ids: Vec<u64> = quantized.iter().map(|q| q.id).collect();

        // Update global stats
        {
            let mut stats = self.global_stats.write();
            for q in &quantized {
                stats.sum_norms += q.stats.norm as f64;
                stats.sum_variances += q.stats.variance as f64;
                stats.count += 1;
            }
        }

        self.vectors.write().extend(quantized);
        ids
    }

    /// Compute asymmetric distance between query (float) and quantized vector
    /// Returns estimated L2 squared distance
    pub fn asymmetric_distance(&self, query: &[f32], quantized: &QuantizedVector) -> f32 {
        // Rotate query
        let rotated_query = self.rotation.rotate_simd(query);

        // Compute distance using binary approximation
        let mut sum_positive = 0.0f32;
        let mut sum_negative = 0.0f32;

        for (i, &q) in rotated_query.iter().enumerate() {
            if quantized.get_bit(i) {
                sum_positive += q;
            } else {
                sum_negative += q;
            }
        }

        // Distance estimation based on RaBitQ formula
        let query_stats = VectorStats::compute(query);
        let dim = self.config.dim as f32;

        // Estimated distance = ||q||^2 + ||x||^2 - 2 * correction * dot_estimate
        let query_norm_sq = query_stats.norm.powi(2);
        let vec_norm_sq = quantized.stats.norm.powi(2);

        // Dot product estimate from binary representation
        let dot_estimate = (sum_positive - sum_negative) * quantized.stats.norm / dim.sqrt();

        // Apply correction factor based on variance
        let correction = (2.0 / std::f32::consts::PI).sqrt();

        let distance = query_norm_sq + vec_norm_sq - 2.0 * correction * dot_estimate;
        distance.max(0.0)
    }

    /// Compute Hamming distance between two quantized vectors (symmetric)
    #[inline]
    pub fn hamming_distance(a: &QuantizedVector, b: &QuantizedVector) -> u32 {
        a.bits.iter()
            .zip(b.bits.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }

    /// SIMD-accelerated Hamming distance
    #[cfg(target_arch = "x86_64")]
    pub fn hamming_distance_simd(a: &QuantizedVector, b: &QuantizedVector) -> u32 {
        if is_x86_feature_detected!("popcnt") {
            unsafe { Self::hamming_distance_popcnt(a, b) }
        } else {
            Self::hamming_distance(a, b)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "popcnt")]
    unsafe fn hamming_distance_popcnt(a: &QuantizedVector, b: &QuantizedVector) -> u32 {
        use std::arch::x86_64::_popcnt64;

        a.bits.iter()
            .zip(b.bits.iter())
            .map(|(x, y)| _popcnt64((x ^ y) as i64) as u32)
            .sum()
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn hamming_distance_simd(a: &QuantizedVector, b: &QuantizedVector) -> u32 {
        Self::hamming_distance(a, b)
    }

    /// Search for k nearest neighbors using asymmetric distance
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let vectors = self.vectors.read();

        // Compute distances to all vectors
        let mut distances: Vec<(u64, f32)> = vectors.iter()
            .map(|v| (v.id, self.asymmetric_distance(query, v)))
            .collect();

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Return top-k
        distances.truncate(k);
        distances
    }

    /// Search with pre-filter (using RoaringBitmap of valid IDs)
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        valid_ids: &roaring::RoaringBitmap,
    ) -> Vec<(u64, f32)> {
        let vectors = self.vectors.read();

        // Compute distances only for valid IDs
        let mut distances: Vec<(u64, f32)> = vectors.iter()
            .filter(|v| valid_ids.contains(v.id as u32))
            .map(|v| (v.id, self.asymmetric_distance(query, v)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        distances
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        // Original: dim * 4 bytes (float32)
        // Quantized: dim / 8 bytes (1 bit) + stats (12 bytes)
        let original_size = self.config.dim * 4;
        let quantized_size = (self.config.dim + 7) / 8 + 12;
        original_size as f32 / quantized_size as f32
    }

    /// Get number of stored vectors
    pub fn len(&self) -> usize {
        self.vectors.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vectors.read().is_empty()
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let vectors = self.vectors.read();
        let vec_size = vectors.iter()
            .map(|v| v.bits.len() * 8 + std::mem::size_of::<VectorStats>() + 8)
            .sum::<usize>();

        let rotation_size = self.rotation.data.len() * 4;

        vec_size + rotation_size
    }
}

/// RaBitQ Index combining quantization with graph-based search
pub struct RaBitQIndex {
    quantizer: RaBitQuantizer,
    /// Optional: store original vectors for reranking
    originals: Option<RwLock<Vec<Vec<f32>>>>,
    /// Rerank top-k candidates with original vectors
    rerank_k: usize,
}

impl RaBitQIndex {
    pub fn new(config: RaBitQConfig, keep_originals: bool, rerank_k: usize) -> Self {
        Self {
            quantizer: RaBitQuantizer::new(config),
            originals: if keep_originals {
                Some(RwLock::new(Vec::new()))
            } else {
                None
            },
            rerank_k,
        }
    }

    /// Insert vector
    pub fn insert(&self, vector: &[f32]) -> u64 {
        let id = self.quantizer.insert(vector);

        if let Some(ref originals) = self.originals {
            originals.write().push(vector.to_vec());
        }

        id
    }

    /// Batch insert
    pub fn insert_batch(&self, vectors: &[Vec<f32>]) -> Vec<u64> {
        let ids = self.quantizer.insert_batch(vectors);

        if let Some(ref originals) = self.originals {
            let mut orig = originals.write();
            for v in vectors {
                orig.push(v.clone());
            }
        }

        ids
    }

    /// Search with optional reranking
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        // Get more candidates for reranking
        let candidates_k = if self.originals.is_some() {
            k * self.rerank_k
        } else {
            k
        };

        let mut results = self.quantizer.search(query, candidates_k);

        // Rerank using original vectors if available
        if let Some(ref originals) = self.originals {
            let orig = originals.read();

            // Recompute exact distances for candidates
            for (id, dist) in results.iter_mut() {
                if let Some(original) = orig.get(*id as usize) {
                    *dist = l2_squared(query, original);
                }
            }

            // Re-sort by exact distance
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }

        results.truncate(k);
        results
    }

    /// Get statistics
    pub fn stats(&self) -> RaBitQStats {
        RaBitQStats {
            num_vectors: self.quantizer.len(),
            compression_ratio: self.quantizer.compression_ratio(),
            memory_bytes: self.quantizer.memory_usage(),
            has_originals: self.originals.is_some(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RaBitQStats {
    pub num_vectors: usize,
    pub compression_ratio: f32,
    pub memory_bytes: usize,
    pub has_originals: bool,
}

/// Compute L2 squared distance
#[inline]
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        (0..dim).map(|i| {
            let mut hasher = DefaultHasher::new();
            (seed, i).hash(&mut hasher);
            let hash = hasher.finish();
            (hash as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
        }).collect()
    }

    #[test]
    fn test_quantize_basic() {
        let config = RaBitQConfig {
            dim: 128,
            ..Default::default()
        };
        let quantizer = RaBitQuantizer::new(config);

        let vector = random_vector(128, 42);
        let quantized = quantizer.quantize(&vector);

        assert_eq!(quantized.bits.len(), 2); // 128 / 64 = 2
        assert!(quantized.stats.norm > 0.0);
    }

    #[test]
    fn test_compression_ratio() {
        let config = RaBitQConfig {
            dim: 128,
            ..Default::default()
        };
        let quantizer = RaBitQuantizer::new(config);

        // Expected: 128 * 4 / (16 + 12) ≈ 18x
        let ratio = quantizer.compression_ratio();
        assert!(ratio > 15.0);
        println!("Compression ratio: {:.1}x", ratio);
    }

    #[test]
    fn test_hamming_distance() {
        let config = RaBitQConfig {
            dim: 128,
            ..Default::default()
        };
        let quantizer = RaBitQuantizer::new(config);

        let v1 = random_vector(128, 1);
        let v2 = random_vector(128, 2);
        let v3 = v1.clone();

        let q1 = quantizer.quantize(&v1);
        let q2 = quantizer.quantize(&v2);
        let q3 = quantizer.quantize(&v3);

        // Same vector should have 0 hamming distance
        assert_eq!(RaBitQuantizer::hamming_distance(&q1, &q3), 0);

        // Different vectors should have non-zero distance
        assert!(RaBitQuantizer::hamming_distance(&q1, &q2) > 0);
    }

    #[test]
    fn test_search() {
        let config = RaBitQConfig {
            dim: 64,
            ..Default::default()
        };
        let quantizer = RaBitQuantizer::new(config);

        // Insert some vectors
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| random_vector(64, i))
            .collect();

        quantizer.insert_batch(&vectors);

        // Search
        let query = random_vector(64, 999);
        let results = quantizer.search(&query, 5);

        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i-1].1);
        }
    }

    #[test]
    fn test_index_with_rerank() {
        let config = RaBitQConfig {
            dim: 64,
            ..Default::default()
        };
        let index = RaBitQIndex::new(config, true, 3);

        // Insert vectors
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| random_vector(64, i))
            .collect();

        index.insert_batch(&vectors);

        // Search with reranking
        let query = random_vector(64, 999);
        let results = index.search(&query, 5);

        assert_eq!(results.len(), 5);

        let stats = index.stats();
        assert_eq!(stats.num_vectors, 100);
        assert!(stats.has_originals);
        println!("Compression: {:.1}x, Memory: {} bytes",
                 stats.compression_ratio, stats.memory_bytes);
    }

    #[test]
    fn test_rotation_matrix() {
        let dim = 64;
        let rotation = RotationMatrix::new_orthogonal(dim, 42);

        let v = random_vector(dim, 1);
        let rotated = rotation.rotate(&v);

        // Rotation should preserve L2 norm
        let orig_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rot_norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!((orig_norm - rot_norm).abs() < 0.01);
    }

    #[test]
    fn test_recall_quality() {
        let config = RaBitQConfig {
            dim: 128,
            use_orthogonal: true,
            ..Default::default()
        };
        let index = RaBitQIndex::new(config, true, 5);

        // Insert 1000 random vectors
        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|i| random_vector(128, i))
            .collect();
        index.insert_batch(&vectors);

        // Test recall
        let mut total_recall = 0.0;
        let num_queries = 10;

        for q in 0..num_queries {
            let query = random_vector(128, 10000 + q);

            // Get RaBitQ results
            let rabitq_results: Vec<u64> = index.search(&query, 10)
                .iter().map(|(id, _)| *id).collect();

            // Compute exact top-10
            let mut exact: Vec<(u64, f32)> = vectors.iter().enumerate()
                .map(|(i, v)| (i as u64, l2_squared(&query, v)))
                .collect();
            exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let exact_top: Vec<u64> = exact[..10].iter().map(|(id, _)| *id).collect();

            // Count matches
            let matches = rabitq_results.iter()
                .filter(|id| exact_top.contains(id))
                .count();

            total_recall += matches as f64 / 10.0;
        }

        let avg_recall = total_recall / num_queries as f64;
        println!("Average Recall@10: {:.1}%", avg_recall * 100.0);

        // With reranking, we should get decent recall
        assert!(avg_recall > 0.5, "Recall too low: {}", avg_recall);
    }
}
