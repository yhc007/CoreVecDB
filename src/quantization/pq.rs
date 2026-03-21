//! Product Quantization (PQ) for high compression vector storage.
//!
//! Achieves 90%+ memory reduction by dividing vectors into subvectors
//! and quantizing each independently using k-means clustering.
//!
//! Architecture:
//! ```text
//! Vector (128-dim, 512 bytes)
//!    │
//!    ▼ Split into M subvectors
//! ┌────┬────┬────┬────┬────┬────┬────┬────┐
//! │ 16 │ 16 │ 16 │ 16 │ 16 │ 16 │ 16 │ 16 │  (M=8 subvectors)
//! └────┴────┴────┴────┴────┴────┴────┴────┘
//!    │
//!    ▼ K-means (K=256 centroids per subvector)
//! ┌────┬────┬────┬────┬────┬────┬────┬────┐
//! │ u8 │ u8 │ u8 │ u8 │ u8 │ u8 │ u8 │ u8 │  PQ code (8 bytes)
//! └────┴────┴────┴────┴────┴────┴────┴────┘
//!
//! Compression: 512 bytes → 8 bytes = 98.4% reduction
//! ```
//!
//! Key concepts:
//! - **Codebook**: K centroids per subvector learned via k-means
//! - **PQ Code**: Vector of centroid indices (one per subvector)
//! - **ADC (Asymmetric Distance Computation)**: Query in float32, candidates in PQ codes

use anyhow::{Result, Context};
use parking_lot::RwLock;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Product Quantizer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// Vector dimension
    pub dim: usize,
    /// Number of subvectors (M). dim must be divisible by M.
    pub num_subvectors: usize,
    /// Number of centroids per subvector (K). Typically 256 for u8 codes.
    pub num_centroids: usize,
    /// Number of k-means iterations for training
    pub kmeans_iterations: usize,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            num_subvectors: 8,      // 8 subvectors of 16 dims each
            num_centroids: 256,     // u8 codes (0-255)
            kmeans_iterations: 25,
        }
    }
}

impl PQConfig {
    /// Create PQ config for given dimension.
    pub fn for_dim(dim: usize) -> Self {
        // Choose M based on dimension for good balance
        let num_subvectors = if dim <= 32 {
            4
        } else if dim <= 128 {
            8
        } else if dim <= 512 {
            16
        } else {
            32
        };

        assert!(dim % num_subvectors == 0, "dim must be divisible by num_subvectors");

        Self {
            dim,
            num_subvectors,
            num_centroids: 256,
            kmeans_iterations: 25,
        }
    }

    /// Subvector dimension.
    pub fn subvector_dim(&self) -> usize {
        self.dim / self.num_subvectors
    }

    /// PQ code size in bytes.
    pub fn code_size(&self) -> usize {
        if self.num_centroids <= 256 {
            self.num_subvectors  // u8 per subvector
        } else {
            self.num_subvectors * 2  // u16 per subvector
        }
    }

    /// Compression ratio (original_size / compressed_size).
    pub fn compression_ratio(&self) -> f32 {
        let original = self.dim * 4;  // f32 = 4 bytes
        let compressed = self.code_size();
        original as f32 / compressed as f32
    }
}

/// Codebook for a single subvector space.
/// Contains K centroids, each of dimension D/M.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codebook {
    /// Centroids: [num_centroids][subvector_dim]
    centroids: Vec<Vec<f32>>,
    /// Subvector dimension
    subvector_dim: usize,
}

impl Codebook {
    /// Create empty codebook.
    pub fn new(num_centroids: usize, subvector_dim: usize) -> Self {
        Self {
            centroids: vec![vec![0.0; subvector_dim]; num_centroids],
            subvector_dim,
        }
    }

    /// Initialize centroids randomly from training data.
    pub fn init_random(&mut self, subvectors: &[Vec<f32>], rng: &mut impl Rng) {
        let n = subvectors.len();
        if n == 0 {
            return;
        }

        // Random sampling without replacement
        let indices: Vec<usize> = (0..n).collect();
        let selected: Vec<usize> = indices
            .choose_multiple(rng, self.centroids.len().min(n))
            .cloned()
            .collect();

        for (i, &idx) in selected.iter().enumerate() {
            self.centroids[i] = subvectors[idx].clone();
        }

        // If not enough samples, duplicate with noise
        for i in selected.len()..self.centroids.len() {
            let src = &subvectors[rng.gen_range(0..n)];
            self.centroids[i] = src.iter()
                .map(|&v| v + rng.gen_range(-0.01..0.01))
                .collect();
        }
    }

    /// Find nearest centroid for a subvector.
    #[inline]
    pub fn find_nearest(&self, subvector: &[f32]) -> u8 {
        let mut best_idx = 0u8;
        let mut best_dist = f32::MAX;

        for (idx, centroid) in self.centroids.iter().enumerate() {
            let dist = l2_squared(subvector, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx as u8;
            }
        }

        best_idx
    }

    /// Get centroid by index.
    #[inline]
    pub fn get_centroid(&self, idx: u8) -> &[f32] {
        &self.centroids[idx as usize]
    }

    /// Compute distance table for ADC.
    /// Returns distances from query subvector to all centroids.
    pub fn compute_distance_table(&self, query_subvector: &[f32]) -> Vec<f32> {
        self.centroids
            .iter()
            .map(|c| l2_squared(query_subvector, c))
            .collect()
    }

    /// Run k-means iteration, returns whether converged.
    pub fn kmeans_step(&mut self, subvectors: &[Vec<f32>]) -> bool {
        if subvectors.is_empty() {
            return true;
        }

        let k = self.centroids.len();
        let dim = self.subvector_dim;

        // Assign each subvector to nearest centroid
        let mut assignments: Vec<usize> = subvectors
            .iter()
            .map(|sv| self.find_nearest(sv) as usize)
            .collect();

        // Compute new centroids
        let mut new_centroids = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for (sv, &cluster) in subvectors.iter().zip(assignments.iter()) {
            counts[cluster] += 1;
            for (j, &val) in sv.iter().enumerate() {
                new_centroids[cluster][j] += val;
            }
        }

        // Average and check convergence
        let mut converged = true;
        let threshold = 1e-6;

        for i in 0..k {
            if counts[i] > 0 {
                for j in 0..dim {
                    new_centroids[i][j] /= counts[i] as f32;
                }
            } else {
                // Empty cluster - reinitialize from random point
                let random_idx = rand::thread_rng().gen_range(0..subvectors.len());
                new_centroids[i] = subvectors[random_idx].clone();
            }

            // Check if centroid moved significantly
            let diff: f32 = self.centroids[i]
                .iter()
                .zip(new_centroids[i].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if diff > threshold {
                converged = false;
            }
        }

        self.centroids = new_centroids;
        converged
    }
}

/// L2 squared distance (no sqrt for efficiency).
#[inline]
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

/// Product Quantizer - encodes vectors to compact PQ codes.
#[derive(Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    /// Configuration
    config: PQConfig,
    /// Codebooks for each subvector space
    codebooks: Vec<Codebook>,
    /// Whether the quantizer is trained
    trained: bool,
    /// Number of vectors used for training
    training_count: usize,
}

impl ProductQuantizer {
    /// Create a new Product Quantizer.
    pub fn new(config: PQConfig) -> Self {
        let subvector_dim = config.subvector_dim();
        let codebooks = (0..config.num_subvectors)
            .map(|_| Codebook::new(config.num_centroids, subvector_dim))
            .collect();

        Self {
            config,
            codebooks,
            trained: false,
            training_count: 0,
        }
    }

    /// Create with default config for dimension.
    pub fn for_dim(dim: usize) -> Self {
        Self::new(PQConfig::for_dim(dim))
    }

    /// Train the quantizer on a set of vectors.
    pub fn train(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Err(anyhow::anyhow!("Cannot train on empty dataset"));
        }

        let first_dim = vectors[0].len();
        if first_dim != self.config.dim {
            return Err(anyhow::anyhow!(
                "Dimension mismatch: expected {}, got {}",
                self.config.dim,
                first_dim
            ));
        }

        let m = self.config.num_subvectors;
        let subvector_dim = self.config.subvector_dim();
        let iterations = self.config.kmeans_iterations;

        // Train each codebook independently
        for i in 0..m {
            // Extract subvectors for this codebook
            let subvectors: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| v[i * subvector_dim..(i + 1) * subvector_dim].to_vec())
                .collect();

            // Initialize centroids randomly
            let mut rng = rand::thread_rng();
            self.codebooks[i].init_random(&subvectors, &mut rng);

            // Run k-means
            for _ in 0..iterations {
                if self.codebooks[i].kmeans_step(&subvectors) {
                    break;  // Converged
                }
            }
        }

        self.trained = true;
        self.training_count = vectors.len();
        Ok(())
    }

    /// Encode a vector to PQ code.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        if !self.trained {
            return Err(anyhow::anyhow!("Quantizer not trained"));
        }

        if vector.len() != self.config.dim {
            return Err(anyhow::anyhow!(
                "Dimension mismatch: expected {}, got {}",
                self.config.dim,
                vector.len()
            ));
        }

        let m = self.config.num_subvectors;
        let subvector_dim = self.config.subvector_dim();

        let code: Vec<u8> = (0..m)
            .map(|i| {
                let subvector = &vector[i * subvector_dim..(i + 1) * subvector_dim];
                self.codebooks[i].find_nearest(subvector)
            })
            .collect();

        Ok(code)
    }

    /// Encode multiple vectors in batch.
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<u8>>> {
        vectors.iter().map(|v| self.encode(v)).collect()
    }

    /// Decode PQ code to approximate vector.
    pub fn decode(&self, code: &[u8]) -> Result<Vec<f32>> {
        if !self.trained {
            return Err(anyhow::anyhow!("Quantizer not trained"));
        }

        if code.len() != self.config.num_subvectors {
            return Err(anyhow::anyhow!(
                "Code length mismatch: expected {}, got {}",
                self.config.num_subvectors,
                code.len()
            ));
        }

        let subvector_dim = self.config.subvector_dim();
        let mut vector = Vec::with_capacity(self.config.dim);

        for (i, &idx) in code.iter().enumerate() {
            let centroid = self.codebooks[i].get_centroid(idx);
            vector.extend_from_slice(centroid);
        }

        Ok(vector)
    }

    /// Compute distance tables for ADC (Asymmetric Distance Computation).
    /// Returns M tables, each with K distances.
    pub fn compute_distance_tables(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let m = self.config.num_subvectors;
        let subvector_dim = self.config.subvector_dim();

        (0..m)
            .map(|i| {
                let query_sub = &query[i * subvector_dim..(i + 1) * subvector_dim];
                self.codebooks[i].compute_distance_table(query_sub)
            })
            .collect()
    }

    /// Compute L2 squared distance using ADC (query in float32, code in PQ).
    /// This is the key optimization - O(M) lookups instead of O(D) operations.
    #[inline]
    pub fn distance_adc(&self, distance_tables: &[Vec<f32>], code: &[u8]) -> f32 {
        code.iter()
            .enumerate()
            .map(|(i, &idx)| distance_tables[i][idx as usize])
            .sum()
    }

    /// Search for k nearest neighbors using ADC.
    pub fn search_adc(
        &self,
        query: &[f32],
        codes: &[Vec<u8>],
        k: usize,
    ) -> Vec<(usize, f32)> {
        let distance_tables = self.compute_distance_tables(query);

        let mut distances: Vec<(usize, f32)> = codes
            .iter()
            .enumerate()
            .map(|(idx, code)| (idx, self.distance_adc(&distance_tables, code)))
            .collect();

        // Partial sort to get top k
        if k < distances.len() {
            distances.select_nth_unstable_by(k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            distances.truncate(k);
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances
    }

    /// Get config.
    pub fn config(&self) -> &PQConfig {
        &self.config
    }

    /// Check if trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get training count.
    pub fn training_count(&self) -> usize {
        self.training_count
    }

    /// Get code size in bytes.
    pub fn code_size(&self) -> usize {
        self.config.code_size()
    }

    /// Get compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        self.config.compression_ratio()
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).context("Failed to serialize ProductQuantizer")
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).context("Failed to deserialize ProductQuantizer")
    }
}

/// Thread-safe Product Quantizer with incremental training support.
pub struct ThreadSafePQ {
    /// The quantizer
    quantizer: RwLock<ProductQuantizer>,
    /// Pending vectors for incremental training
    pending_vectors: RwLock<Vec<Vec<f32>>>,
    /// Threshold to trigger retraining
    retrain_threshold: usize,
}

impl ThreadSafePQ {
    /// Create new thread-safe PQ.
    pub fn new(config: PQConfig) -> Self {
        Self {
            quantizer: RwLock::new(ProductQuantizer::new(config)),
            pending_vectors: RwLock::new(Vec::new()),
            retrain_threshold: 10000,
        }
    }

    /// Create with dimension.
    pub fn for_dim(dim: usize) -> Self {
        Self::new(PQConfig::for_dim(dim))
    }

    /// Add vectors for training (batched incremental training).
    pub fn add_training_vectors(&self, vectors: Vec<Vec<f32>>) {
        let mut pending = self.pending_vectors.write();
        pending.extend(vectors);

        // Check if we should retrain
        if pending.len() >= self.retrain_threshold {
            drop(pending);
            self.retrain();
        }
    }

    /// Force retraining on pending vectors.
    pub fn retrain(&self) {
        let vectors: Vec<Vec<f32>> = {
            let mut pending = self.pending_vectors.write();
            std::mem::take(&mut *pending)
        };

        if vectors.is_empty() {
            return;
        }

        let mut quantizer = self.quantizer.write();
        let _ = quantizer.train(&vectors);
    }

    /// Encode vector.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        self.quantizer.read().encode(vector)
    }

    /// Encode batch.
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<u8>>> {
        self.quantizer.read().encode_batch(vectors)
    }

    /// Decode code.
    pub fn decode(&self, code: &[u8]) -> Result<Vec<f32>> {
        self.quantizer.read().decode(code)
    }

    /// Search using ADC.
    pub fn search_adc(
        &self,
        query: &[f32],
        codes: &[Vec<u8>],
        k: usize,
    ) -> Vec<(usize, f32)> {
        self.quantizer.read().search_adc(query, codes, k)
    }

    /// Check if trained.
    pub fn is_trained(&self) -> bool {
        self.quantizer.read().is_trained()
    }

    /// Get compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        self.quantizer.read().compression_ratio()
    }

    /// Serialize.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.quantizer.read().to_bytes()
    }

    /// Load from bytes.
    pub fn load(&self, bytes: &[u8]) -> Result<()> {
        let pq = ProductQuantizer::from_bytes(bytes)?;
        *self.quantizer.write() = pq;
        Ok(())
    }
}

// =============================================================================
// PQ-based Vector Store
// =============================================================================

/// Memory-efficient vector store using Product Quantization.
pub struct PQVectorStore {
    /// Product Quantizer
    pq: Arc<ThreadSafePQ>,
    /// PQ codes: [vector_id] -> code
    codes: RwLock<Vec<Vec<u8>>>,
    /// Original vectors kept for reranking (optional)
    originals: RwLock<Option<Vec<Vec<f32>>>>,
    /// Keep originals for reranking
    keep_originals: bool,
    /// Rerank oversample factor
    rerank_oversample: usize,
}

impl PQVectorStore {
    /// Create new PQ vector store.
    pub fn new(dim: usize, keep_originals: bool, rerank_oversample: usize) -> Self {
        Self {
            pq: Arc::new(ThreadSafePQ::for_dim(dim)),
            codes: RwLock::new(Vec::new()),
            originals: RwLock::new(if keep_originals { Some(Vec::new()) } else { None }),
            keep_originals,
            rerank_oversample,
        }
    }

    /// Insert a vector.
    pub fn insert(&self, vector: Vec<f32>) -> Result<u64> {
        let id = {
            let codes = self.codes.read();
            codes.len() as u64
        };

        // Train if needed (first batch)
        if !self.pq.is_trained() {
            self.pq.add_training_vectors(vec![vector.clone()]);
            if self.codes.read().len() >= 1000 {
                self.pq.retrain();
            }
        }

        // Encode and store
        if self.pq.is_trained() {
            let code = self.pq.encode(&vector)?;
            self.codes.write().push(code);
        } else {
            // Before training, store dummy code
            let code_size = self.pq.quantizer.read().code_size();
            self.codes.write().push(vec![0u8; code_size]);
        }

        // Keep original if needed
        if self.keep_originals {
            if let Some(ref mut originals) = *self.originals.write() {
                originals.push(vector);
            }
        }

        Ok(id)
    }

    /// Insert batch of vectors.
    pub fn insert_batch(&self, vectors: Vec<Vec<f32>>) -> Result<u64> {
        let start_id = self.codes.read().len() as u64;

        // Add to training pool
        if !self.pq.is_trained() {
            self.pq.add_training_vectors(vectors.clone());
            if self.codes.read().len() + vectors.len() >= 1000 {
                self.pq.retrain();
            }
        }

        // Encode and store
        if self.pq.is_trained() {
            let codes = self.pq.encode_batch(&vectors)?;
            self.codes.write().extend(codes);
        } else {
            let code_size = self.pq.quantizer.read().code_size();
            let dummy_codes: Vec<Vec<u8>> = (0..vectors.len())
                .map(|_| vec![0u8; code_size])
                .collect();
            self.codes.write().extend(dummy_codes);
        }

        // Keep originals if needed
        if self.keep_originals {
            if let Some(ref mut originals) = *self.originals.write() {
                originals.extend(vectors);
            }
        }

        Ok(start_id)
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        if !self.pq.is_trained() {
            return Ok(Vec::new());
        }

        let codes = self.codes.read();

        // Get candidates using ADC
        let oversample_k = if self.keep_originals {
            k * self.rerank_oversample
        } else {
            k
        };

        let candidates = self.pq.search_adc(query, &codes, oversample_k);

        // Rerank with original vectors if available
        if self.keep_originals {
            if let Some(ref originals) = *self.originals.read() {
                let mut reranked: Vec<(u64, f32)> = candidates
                    .iter()
                    .map(|&(idx, _)| {
                        let dist = l2_squared(query, &originals[idx]);
                        (idx as u64, dist)
                    })
                    .collect();

                reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                reranked.truncate(k);
                return Ok(reranked);
            }
        }

        Ok(candidates
            .into_iter()
            .take(k)
            .map(|(idx, dist)| (idx as u64, dist))
            .collect())
    }

    /// Get vector by ID (decoded).
    pub fn get(&self, id: u64) -> Result<Vec<f32>> {
        // Try original first
        if self.keep_originals {
            if let Some(ref originals) = *self.originals.read() {
                if (id as usize) < originals.len() {
                    return Ok(originals[id as usize].clone());
                }
            }
        }

        // Decode from PQ code
        let codes = self.codes.read();
        if (id as usize) >= codes.len() {
            return Err(anyhow::anyhow!("Vector ID {} not found", id));
        }

        self.pq.decode(&codes[id as usize])
    }

    /// Get number of vectors.
    pub fn len(&self) -> usize {
        self.codes.read().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.codes.read().is_empty()
    }

    /// Get compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        self.pq.compression_ratio()
    }

    /// Get memory usage estimate.
    pub fn memory_usage(&self) -> usize {
        let codes = self.codes.read();
        let code_memory: usize = codes.iter().map(|c| c.len()).sum();

        let original_memory: usize = if self.keep_originals {
            self.originals.read()
                .as_ref()
                .map(|o| o.iter().map(|v| v.len() * 4).sum())
                .unwrap_or(0)
        } else {
            0
        };

        code_memory + original_memory
    }

    /// Force training on current data.
    pub fn force_train(&self) {
        self.pq.retrain();

        // Re-encode all vectors if we have originals
        if self.pq.is_trained() && self.keep_originals {
            if let Some(ref originals) = *self.originals.read() {
                if let Ok(new_codes) = self.pq.encode_batch(originals) {
                    *self.codes.write() = new_codes;
                }
            }
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
    fn test_pq_config() {
        let config = PQConfig::for_dim(128);
        assert_eq!(config.dim, 128);
        assert_eq!(config.num_subvectors, 8);
        assert_eq!(config.subvector_dim(), 16);
        assert_eq!(config.code_size(), 8);
        assert!(config.compression_ratio() > 60.0);  // 512/8 = 64x
    }

    #[test]
    fn test_codebook_find_nearest() {
        let mut codebook = Codebook::new(4, 2);
        codebook.centroids = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        assert_eq!(codebook.find_nearest(&[0.1, 0.1]), 0);
        assert_eq!(codebook.find_nearest(&[0.9, 0.1]), 1);
        assert_eq!(codebook.find_nearest(&[0.1, 0.9]), 2);
        assert_eq!(codebook.find_nearest(&[0.9, 0.9]), 3);
    }

    #[test]
    fn test_pq_train_encode_decode() {
        let mut pq = ProductQuantizer::new(PQConfig {
            dim: 8,
            num_subvectors: 2,
            num_centroids: 4,
            kmeans_iterations: 10,
        });

        // Generate training data
        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..8).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        pq.train(&vectors).unwrap();
        assert!(pq.is_trained());

        // Test encode/decode
        let test_vector: Vec<f32> = (0..8).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let code = pq.encode(&test_vector).unwrap();
        assert_eq!(code.len(), 2);

        let decoded = pq.decode(&code).unwrap();
        assert_eq!(decoded.len(), 8);
    }

    #[test]
    fn test_pq_adc_search() {
        let mut pq = ProductQuantizer::new(PQConfig {
            dim: 8,
            num_subvectors: 2,
            num_centroids: 16,
            kmeans_iterations: 10,
        });

        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..8).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        pq.train(&vectors).unwrap();

        let codes: Vec<Vec<u8>> = vectors.iter()
            .map(|v| pq.encode(v).unwrap())
            .collect();

        // Search
        let query = &vectors[0];
        let results = pq.search_adc(query, &codes, 5);

        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1, "Results not sorted");
        }
        // First result's distance should be small (PQ is lossy, so not exactly 0)
        assert!(results[0].1 < 1.0, "First result distance too large: {}", results[0].1);
    }

    #[test]
    fn test_pq_vector_store() {
        let store = PQVectorStore::new(16, true, 3);

        // Insert vectors
        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..1500)
            .map(|_| (0..16).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        store.insert_batch(vectors.clone()).unwrap();

        assert_eq!(store.len(), 1500);
        assert!(store.pq.is_trained());

        // Search
        let query = &vectors[0];
        let results = store.search(query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // With reranking, first result should be exact match
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_pq_serialization() {
        let mut pq = ProductQuantizer::new(PQConfig {
            dim: 8,
            num_subvectors: 2,
            num_centroids: 4,
            kmeans_iterations: 5,
        });

        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..8).map(|j| (i * 8 + j) as f32 / 400.0).collect())
            .collect();

        pq.train(&vectors).unwrap();

        // Serialize and deserialize
        let bytes = pq.to_bytes().unwrap();
        let pq2 = ProductQuantizer::from_bytes(&bytes).unwrap();

        assert!(pq2.is_trained());
        assert_eq!(pq2.config().dim, 8);

        // Check encoding consistency
        let test = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let code1 = pq.encode(&test).unwrap();
        let code2 = pq2.encode(&test).unwrap();
        assert_eq!(code1, code2);
    }
}
