//! IVF-PQ: Inverted File with Product Quantization for billion-scale search.
//!
//! Combines coarse quantization (IVF) with fine quantization (PQ) to enable
//! efficient approximate nearest neighbor search on massive datasets.
//!
//! ## Architecture
//! ```text
//! Query Vector
//!      │
//!      ▼
//! ┌─────────────────┐
//! │ Coarse Quantizer│  k-means centroids (nlist clusters)
//! │   (IVF layer)   │
//! └────────┬────────┘
//!          │ Find nprobe closest clusters
//!          ▼
//! ┌─────────────────┐
//! │ Inverted Lists  │  Vectors grouped by cluster
//! │  [c0] [c1]...[cn]│
//! └────────┬────────┘
//!          │ Search within clusters using PQ
//!          ▼
//! ┌─────────────────┐
//! │  PQ Quantizer   │  Compressed vectors (M bytes each)
//! │   (ADC search)  │
//! └─────────────────┘
//! ```
//!
//! ## Memory Usage
//! For 1 billion 128-dim vectors:
//! - Raw: 512 GB
//! - IVF-PQ (M=8): ~8 GB + centroids

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// =============================================================================
// IVF-PQ Configuration
// =============================================================================

/// IVF-PQ configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfPqConfig {
    /// Vector dimension
    pub dim: usize,
    /// Number of coarse centroids (clusters)
    pub nlist: usize,
    /// Number of clusters to search (1 <= nprobe <= nlist)
    pub nprobe: usize,
    /// Number of PQ subvectors
    pub m: usize,
    /// Number of PQ centroids per subvector (typically 256)
    pub ksub: usize,
    /// K-means iterations for training
    pub kmeans_iters: usize,
}

impl IvfPqConfig {
    /// Create config for a given dimension.
    pub fn for_dim(dim: usize) -> Self {
        Self {
            dim,
            nlist: 256,      // 256 coarse clusters
            nprobe: 8,       // Search 8 clusters
            m: 8,            // 8 subvectors
            ksub: 256,       // 256 centroids per subvector
            kmeans_iters: 20,
        }
    }

    /// Set number of clusters.
    pub fn with_nlist(mut self, nlist: usize) -> Self {
        self.nlist = nlist;
        self
    }

    /// Set number of clusters to probe during search.
    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe.min(self.nlist);
        self
    }

    /// Set PQ subvectors.
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Get subvector dimension.
    pub fn subvector_dim(&self) -> usize {
        self.dim / self.m
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.dim % self.m != 0 {
            return Err(anyhow::anyhow!(
                "Dimension {} must be divisible by M {}",
                self.dim, self.m
            ));
        }
        if self.nprobe > self.nlist {
            return Err(anyhow::anyhow!(
                "nprobe {} cannot exceed nlist {}",
                self.nprobe, self.nlist
            ));
        }
        if self.ksub > 256 {
            return Err(anyhow::anyhow!(
                "ksub {} must be <= 256 for u8 codes",
                self.ksub
            ));
        }
        Ok(())
    }
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self::for_dim(128)
    }
}

// =============================================================================
// Coarse Quantizer (IVF Layer)
// =============================================================================

/// Coarse quantizer using k-means centroids.
#[derive(Debug, Clone)]
pub struct CoarseQuantizer {
    /// Cluster centroids [nlist x dim]
    centroids: Vec<Vec<f32>>,
    /// Number of clusters
    nlist: usize,
    /// Vector dimension
    dim: usize,
    /// Whether trained
    trained: bool,
}

impl CoarseQuantizer {
    /// Create a new coarse quantizer.
    pub fn new(nlist: usize, dim: usize) -> Self {
        Self {
            centroids: vec![],
            nlist,
            dim,
            trained: false,
        }
    }

    /// Train the coarse quantizer using k-means.
    pub fn train(&mut self, vectors: &[Vec<f32>], iterations: usize) -> Result<()> {
        if vectors.is_empty() {
            return Err(anyhow::anyhow!("Cannot train with empty dataset"));
        }

        let n = vectors.len();
        let actual_nlist = self.nlist.min(n);

        // Initialize centroids with random selection
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(actual_nlist);
        let step = n / actual_nlist;
        for i in 0..actual_nlist {
            centroids.push(vectors[i * step].clone());
        }

        // K-means iterations
        for _ in 0..iterations {
            // Assign vectors to nearest centroid
            let assignments: Vec<usize> = vectors
                .par_iter()
                .map(|v| self.find_nearest_centroid(v, &centroids))
                .collect();

            // Update centroids
            let mut new_centroids: Vec<Vec<f32>> = vec![vec![0.0; self.dim]; actual_nlist];
            let mut counts: Vec<usize> = vec![0; actual_nlist];

            for (vec_idx, &cluster) in assignments.iter().enumerate() {
                for (d, &val) in vectors[vec_idx].iter().enumerate() {
                    new_centroids[cluster][d] += val;
                }
                counts[cluster] += 1;
            }

            // Normalize
            for (cluster, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[cluster] > 0 {
                    for val in centroid.iter_mut() {
                        *val /= counts[cluster] as f32;
                    }
                } else {
                    // Empty cluster: reinitialize with random vector
                    *centroid = vectors[cluster % n].clone();
                }
            }

            centroids = new_centroids;
        }

        self.centroids = centroids;
        self.nlist = actual_nlist;
        self.trained = true;
        Ok(())
    }

    /// Find nearest centroid index.
    fn find_nearest_centroid(&self, vector: &[f32], centroids: &[Vec<f32>]) -> usize {
        centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_squared(vector, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Assign a vector to its nearest cluster.
    pub fn assign(&self, vector: &[f32]) -> usize {
        self.find_nearest_centroid(vector, &self.centroids)
    }

    /// Find the nprobe closest clusters.
    pub fn search(&self, vector: &[f32], nprobe: usize) -> Vec<(usize, f32)> {
        let mut distances: Vec<(usize, f32)> = self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_squared(vector, c)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(nprobe);
        distances
    }

    /// Get number of clusters.
    pub fn nlist(&self) -> usize {
        self.nlist
    }

    /// Check if trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get centroid.
    pub fn centroid(&self, cluster: usize) -> Option<&Vec<f32>> {
        self.centroids.get(cluster)
    }
}

// =============================================================================
// Product Quantizer (PQ Layer)
// =============================================================================

/// PQ codebook for a single subvector.
#[derive(Debug, Clone)]
pub struct PqCodebook {
    /// Centroids [ksub x subvector_dim]
    centroids: Vec<Vec<f32>>,
    /// Number of centroids
    ksub: usize,
    /// Subvector dimension
    subvector_dim: usize,
}

impl PqCodebook {
    pub fn new(ksub: usize, subvector_dim: usize) -> Self {
        Self {
            centroids: vec![],
            ksub,
            subvector_dim,
        }
    }

    /// Train the codebook using k-means.
    pub fn train(&mut self, subvectors: &[Vec<f32>], iterations: usize) {
        if subvectors.is_empty() {
            return;
        }

        let n = subvectors.len();
        let actual_ksub = self.ksub.min(n);

        // Initialize centroids
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(actual_ksub);
        let step = n.max(1) / actual_ksub.max(1);
        for i in 0..actual_ksub {
            centroids.push(subvectors[(i * step) % n].clone());
        }

        // K-means
        for _ in 0..iterations {
            let assignments: Vec<usize> = subvectors
                .iter()
                .map(|sv| self.find_nearest(&centroids, sv))
                .collect();

            let mut new_centroids: Vec<Vec<f32>> = vec![vec![0.0; self.subvector_dim]; actual_ksub];
            let mut counts: Vec<usize> = vec![0; actual_ksub];

            for (idx, &cluster) in assignments.iter().enumerate() {
                for (d, &val) in subvectors[idx].iter().enumerate() {
                    new_centroids[cluster][d] += val;
                }
                counts[cluster] += 1;
            }

            for (cluster, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[cluster] > 0 {
                    for val in centroid.iter_mut() {
                        *val /= counts[cluster] as f32;
                    }
                } else {
                    *centroid = subvectors[cluster % n].clone();
                }
            }

            centroids = new_centroids;
        }

        self.centroids = centroids;
        self.ksub = actual_ksub;
    }

    fn find_nearest(&self, centroids: &[Vec<f32>], subvector: &[f32]) -> usize {
        centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_squared(subvector, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Encode a subvector to its nearest centroid index.
    pub fn encode(&self, subvector: &[f32]) -> u8 {
        self.find_nearest(&self.centroids, subvector) as u8
    }

    /// Get centroid for a code.
    pub fn decode(&self, code: u8) -> &[f32] {
        &self.centroids[code as usize]
    }

    /// Precompute distances from query subvector to all centroids.
    /// Returns a lookup table for ADC.
    pub fn compute_distance_table(&self, query_subvector: &[f32]) -> Vec<f32> {
        self.centroids
            .iter()
            .map(|c| l2_squared(query_subvector, c))
            .collect()
    }
}

/// Product Quantizer with M codebooks.
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    /// Codebooks (one per subvector)
    codebooks: Vec<PqCodebook>,
    /// Configuration
    config: IvfPqConfig,
    /// Whether trained
    trained: bool,
}

impl ProductQuantizer {
    pub fn new(config: IvfPqConfig) -> Self {
        let subvector_dim = config.subvector_dim();
        let codebooks: Vec<PqCodebook> = (0..config.m)
            .map(|_| PqCodebook::new(config.ksub, subvector_dim))
            .collect();

        Self {
            codebooks,
            config,
            trained: false,
        }
    }

    /// Train all codebooks.
    pub fn train(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        let subvector_dim = self.config.subvector_dim();

        // Train each codebook on its subvector slice
        for (m, codebook) in self.codebooks.iter_mut().enumerate() {
            let start = m * subvector_dim;
            let end = start + subvector_dim;

            let subvectors: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| v[start..end].to_vec())
                .collect();

            codebook.train(&subvectors, self.config.kmeans_iters);
        }

        self.trained = true;
        Ok(())
    }

    /// Encode a vector to PQ codes.
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let subvector_dim = self.config.subvector_dim();
        self.codebooks
            .iter()
            .enumerate()
            .map(|(m, codebook)| {
                let start = m * subvector_dim;
                let end = start + subvector_dim;
                codebook.encode(&vector[start..end])
            })
            .collect()
    }

    /// Decode PQ codes to approximate vector.
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.config.dim);
        for (m, &code) in codes.iter().enumerate() {
            result.extend_from_slice(self.codebooks[m].decode(code));
        }
        result
    }

    /// Compute distance tables for ADC.
    pub fn compute_distance_tables(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let subvector_dim = self.config.subvector_dim();
        self.codebooks
            .iter()
            .enumerate()
            .map(|(m, codebook)| {
                let start = m * subvector_dim;
                let end = start + subvector_dim;
                codebook.compute_distance_table(&query[start..end])
            })
            .collect()
    }

    /// Compute distance using ADC (lookup tables).
    pub fn adc_distance(&self, tables: &[Vec<f32>], codes: &[u8]) -> f32 {
        codes
            .iter()
            .enumerate()
            .map(|(m, &code)| tables[m][code as usize])
            .sum()
    }

    /// Check if trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }
}

// =============================================================================
// Inverted List
// =============================================================================

/// Entry in an inverted list.
#[derive(Debug, Clone)]
struct IvfEntry {
    /// Original vector ID
    id: u64,
    /// PQ codes
    codes: Vec<u8>,
}

/// Inverted list for a cluster.
#[derive(Debug, Default)]
struct InvertedList {
    entries: Vec<IvfEntry>,
}

impl InvertedList {
    fn new() -> Self {
        Self { entries: vec![] }
    }

    fn add(&mut self, id: u64, codes: Vec<u8>) {
        self.entries.push(IvfEntry { id, codes });
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// =============================================================================
// IVF-PQ Index
// =============================================================================

/// IVF-PQ Index for billion-scale search.
pub struct IvfPqIndex {
    /// Configuration
    config: IvfPqConfig,
    /// Coarse quantizer (IVF layer)
    coarse: RwLock<CoarseQuantizer>,
    /// Product quantizer (PQ layer)
    pq: RwLock<ProductQuantizer>,
    /// Inverted lists (one per cluster)
    lists: RwLock<Vec<InvertedList>>,
    /// Total vector count
    count: AtomicUsize,
    /// Next ID
    next_id: AtomicU64,
    /// Training status
    trained: RwLock<bool>,
}

impl IvfPqIndex {
    /// Create a new IVF-PQ index.
    pub fn new(config: IvfPqConfig) -> Result<Self> {
        config.validate()?;

        let coarse = CoarseQuantizer::new(config.nlist, config.dim);
        let pq = ProductQuantizer::new(config.clone());
        let lists: Vec<InvertedList> = (0..config.nlist)
            .map(|_| InvertedList::new())
            .collect();

        Ok(Self {
            config,
            coarse: RwLock::new(coarse),
            pq: RwLock::new(pq),
            lists: RwLock::new(lists),
            count: AtomicUsize::new(0),
            next_id: AtomicU64::new(0),
            trained: RwLock::new(false),
        })
    }

    /// Train the index on a sample dataset.
    pub fn train(&self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Err(anyhow::anyhow!("Cannot train with empty dataset"));
        }

        // Validate dimensions
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.config.dim {
                return Err(anyhow::anyhow!(
                    "Vector {} has dimension {}, expected {}",
                    i, v.len(), self.config.dim
                ));
            }
        }

        // Train coarse quantizer
        {
            let mut coarse = self.coarse.write().unwrap();
            coarse.train(vectors, self.config.kmeans_iters)?;
        }

        // Compute residuals and train PQ
        let residuals: Vec<Vec<f32>> = {
            let coarse = self.coarse.read().unwrap();
            vectors
                .iter()
                .map(|v| {
                    let cluster = coarse.assign(v);
                    let centroid = coarse.centroid(cluster).unwrap();
                    v.iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| a - b)
                        .collect()
                })
                .collect()
        };

        {
            let mut pq = self.pq.write().unwrap();
            pq.train(&residuals)?;
        }

        // Reinitialize inverted lists
        {
            let mut lists = self.lists.write().unwrap();
            let coarse = self.coarse.read().unwrap();
            *lists = (0..coarse.nlist())
                .map(|_| InvertedList::new())
                .collect();
        }

        *self.trained.write().unwrap() = true;
        Ok(())
    }

    /// Add a vector to the index.
    pub fn add(&self, vector: &[f32]) -> Result<u64> {
        if !*self.trained.read().unwrap() {
            return Err(anyhow::anyhow!("Index must be trained before adding vectors"));
        }

        if vector.len() != self.config.dim {
            return Err(anyhow::anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.dim, vector.len()
            ));
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        // Assign to cluster and compute residual
        let (cluster, residual) = {
            let coarse = self.coarse.read().unwrap();
            let cluster = coarse.assign(vector);
            let centroid = coarse.centroid(cluster).unwrap();
            let residual: Vec<f32> = vector
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| a - b)
                .collect();
            (cluster, residual)
        };

        // Encode residual with PQ
        let codes = {
            let pq = self.pq.read().unwrap();
            pq.encode(&residual)
        };

        // Add to inverted list
        {
            let mut lists = self.lists.write().unwrap();
            lists[cluster].add(id, codes);
        }

        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(id)
    }

    /// Batch add vectors.
    pub fn add_batch(&self, vectors: &[Vec<f32>]) -> Result<Vec<u64>> {
        vectors.iter().map(|v| self.add(v)).collect()
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        self.search_with_nprobe(query, k, self.config.nprobe)
    }

    /// Search with custom nprobe.
    pub fn search_with_nprobe(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<(u64, f32)>> {
        if !*self.trained.read().unwrap() {
            return Err(anyhow::anyhow!("Index must be trained before searching"));
        }

        if query.len() != self.config.dim {
            return Err(anyhow::anyhow!(
                "Query dimension mismatch: expected {}, got {}",
                self.config.dim, query.len()
            ));
        }

        // Find nprobe closest clusters
        let clusters = {
            let coarse = self.coarse.read().unwrap();
            coarse.search(query, nprobe)
        };

        // Precompute residual query for each cluster and search
        let mut all_results: Vec<(u64, f32)> = Vec::new();

        let lists = self.lists.read().unwrap();
        let pq = self.pq.read().unwrap();
        let coarse = self.coarse.read().unwrap();

        for (cluster_id, _cluster_dist) in clusters {
            let list = &lists[cluster_id];
            if list.is_empty() {
                continue;
            }

            // Compute residual query (query - centroid)
            let centroid = coarse.centroid(cluster_id).unwrap();
            let residual_query: Vec<f32> = query
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| a - b)
                .collect();

            // Compute distance tables for ADC
            let tables = pq.compute_distance_tables(&residual_query);

            // Search within cluster using ADC
            for entry in &list.entries {
                let dist = pq.adc_distance(&tables, &entry.codes);
                all_results.push((entry.id, dist));
            }
        }

        // Sort by distance and return top-k
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_results.truncate(k);

        Ok(all_results)
    }

    /// Get total vector count.
    pub fn len(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if trained.
    pub fn is_trained(&self) -> bool {
        *self.trained.read().unwrap()
    }

    /// Get statistics.
    pub fn stats(&self) -> IvfPqStats {
        let lists = self.lists.read().unwrap();
        let list_sizes: Vec<usize> = lists.iter().map(|l| l.len()).collect();
        let non_empty = list_sizes.iter().filter(|&&s| s > 0).count();
        let total: usize = list_sizes.iter().sum();
        let avg = if non_empty > 0 {
            total as f64 / non_empty as f64
        } else {
            0.0
        };
        let max = *list_sizes.iter().max().unwrap_or(&0);
        let min = *list_sizes.iter().filter(|&&s| s > 0).min().unwrap_or(&0);

        IvfPqStats {
            total_vectors: self.len(),
            nlist: self.config.nlist,
            nprobe: self.config.nprobe,
            pq_m: self.config.m,
            non_empty_lists: non_empty,
            avg_list_size: avg,
            max_list_size: max,
            min_list_size: min,
            bytes_per_vector: self.config.m, // PQ codes only
            is_trained: self.is_trained(),
        }
    }

    /// Get config.
    pub fn config(&self) -> &IvfPqConfig {
        &self.config
    }
}

/// IVF-PQ statistics.
#[derive(Debug, Clone, Serialize)]
pub struct IvfPqStats {
    pub total_vectors: usize,
    pub nlist: usize,
    pub nprobe: usize,
    pub pq_m: usize,
    pub non_empty_lists: usize,
    pub avg_list_size: f64,
    pub max_list_size: usize,
    pub min_list_size: usize,
    pub bytes_per_vector: usize,
    pub is_trained: bool,
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute L2 squared distance.
#[inline]
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

// =============================================================================
// Thread-Safe Wrapper
// =============================================================================

/// Thread-safe IVF-PQ index wrapper.
#[derive(Clone)]
pub struct ThreadSafeIvfPq {
    inner: Arc<IvfPqIndex>,
}

impl ThreadSafeIvfPq {
    pub fn new(config: IvfPqConfig) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(IvfPqIndex::new(config)?),
        })
    }

    pub fn train(&self, vectors: &[Vec<f32>]) -> Result<()> {
        self.inner.train(vectors)
    }

    pub fn add(&self, vector: &[f32]) -> Result<u64> {
        self.inner.add(vector)
    }

    pub fn add_batch(&self, vectors: &[Vec<f32>]) -> Result<Vec<u64>> {
        self.inner.add_batch(vectors)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        self.inner.search(query, k)
    }

    pub fn search_with_nprobe(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<(u64, f32)>> {
        self.inner.search_with_nprobe(query, k, nprobe)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn is_trained(&self) -> bool {
        self.inner.is_trained()
    }

    pub fn stats(&self) -> IvfPqStats {
        self.inner.stats()
    }

    pub fn config(&self) -> &IvfPqConfig {
        self.inner.config()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn test_config_validation() {
        let config = IvfPqConfig::for_dim(128);
        assert!(config.validate().is_ok());

        let bad_config = IvfPqConfig {
            dim: 100,
            m: 8, // 100 not divisible by 8
            ..Default::default()
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_coarse_quantizer() {
        let vectors = random_vectors(100, 32);
        let mut cq = CoarseQuantizer::new(10, 32);

        cq.train(&vectors, 5).unwrap();
        assert!(cq.is_trained());
        assert_eq!(cq.nlist(), 10);

        // Should assign to some cluster
        let cluster = cq.assign(&vectors[0]);
        assert!(cluster < 10);

        // Search should return sorted results
        let results = cq.search(&vectors[0], 3);
        assert_eq!(results.len(), 3);
        assert!(results[0].1 <= results[1].1);
    }

    #[test]
    fn test_pq_encode_decode() {
        let config = IvfPqConfig {
            dim: 32,
            nlist: 10,
            nprobe: 2,
            m: 4,
            ksub: 16,
            kmeans_iters: 5,
        };

        let vectors = random_vectors(100, 32);
        let mut pq = ProductQuantizer::new(config);
        pq.train(&vectors).unwrap();

        // Encode and decode
        let original = &vectors[0];
        let codes = pq.encode(original);
        assert_eq!(codes.len(), 4); // M subvectors

        let decoded = pq.decode(&codes);
        assert_eq!(decoded.len(), 32);

        // Decoded should be somewhat close to original
        let dist = l2_squared(original, &decoded);
        assert!(dist < 10.0); // Not exact but reasonable
    }

    #[test]
    fn test_ivfpq_train_and_add() {
        let config = IvfPqConfig {
            dim: 32,
            nlist: 10,
            nprobe: 3,
            m: 4,
            ksub: 16,
            kmeans_iters: 5,
        };

        let index = IvfPqIndex::new(config).unwrap();
        let vectors = random_vectors(100, 32);

        // Train
        index.train(&vectors).unwrap();
        assert!(index.is_trained());

        // Add vectors
        for v in &vectors {
            index.add(v).unwrap();
        }
        assert_eq!(index.len(), 100);
    }

    #[test]
    fn test_ivfpq_search() {
        let config = IvfPqConfig {
            dim: 32,
            nlist: 10,
            nprobe: 5,
            m: 4,
            ksub: 16,
            kmeans_iters: 10,
        };

        let index = IvfPqIndex::new(config).unwrap();
        let vectors = random_vectors(200, 32);

        index.train(&vectors).unwrap();
        for v in &vectors {
            index.add(v).unwrap();
        }

        // Search for nearest neighbors
        let query = &vectors[0];
        let results = index.search(query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_ivfpq_stats() {
        let config = IvfPqConfig {
            dim: 32,
            nlist: 10,
            nprobe: 3,
            m: 4,
            ksub: 16,
            kmeans_iters: 5,
        };

        let index = IvfPqIndex::new(config).unwrap();
        let vectors = random_vectors(100, 32);

        index.train(&vectors).unwrap();
        for v in &vectors {
            index.add(v).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.total_vectors, 100);
        assert_eq!(stats.nlist, 10);
        assert_eq!(stats.pq_m, 4);
        assert!(stats.is_trained);
        assert!(stats.non_empty_lists > 0);
    }

    #[test]
    fn test_thread_safe_wrapper() {
        let config = IvfPqConfig {
            dim: 32,
            nlist: 10,
            nprobe: 3,
            m: 4,
            ksub: 16,
            kmeans_iters: 5,
        };

        let index = ThreadSafeIvfPq::new(config).unwrap();
        let vectors = random_vectors(50, 32);

        index.train(&vectors).unwrap();
        let ids = index.add_batch(&vectors).unwrap();
        assert_eq!(ids.len(), 50);

        let results = index.search(&vectors[0], 3).unwrap();
        assert_eq!(results.len(), 3);
    }
}
