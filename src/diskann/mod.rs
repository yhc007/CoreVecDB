//! DiskANN: Graph-Based Billion-Scale Vector Search on SSD
//!
//! Based on Microsoft's DiskANN paper: "DiskANN: Fast Accurate Billion-point
//! Nearest Neighbor Search on a Single Node"
//!
//! Key features:
//! - Vamana graph construction for SSD-friendly access patterns
//! - Compressed in-memory navigation with full vectors on disk
//! - Beam search with SSD-optimized I/O
//! - Integration with RaBitQ for in-memory compression

use anyhow::Result;
use memmap2::{Mmap, MmapMut};
use parking_lot::RwLock;
use std::collections::{BinaryHeap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::cmp::Reverse;

use crate::rabitq::{RaBitQConfig, RaBitQuantizer, QuantizedVector};

/// DiskANN configuration
#[derive(Debug, Clone)]
pub struct DiskANNConfig {
    /// Vector dimension
    pub dim: usize,
    /// Maximum out-degree of graph nodes (R in paper)
    pub max_degree: usize,
    /// Search list size during construction (L in paper)
    pub build_list_size: usize,
    /// Beam width for search
    pub beam_width: usize,
    /// Alpha parameter for pruning (typically 1.2)
    pub alpha: f32,
    /// Use RaBitQ for in-memory compression
    pub use_rabitq: bool,
    /// Number of PQ bytes for compressed representation (if not using RaBitQ)
    pub pq_bytes: usize,
}

impl Default for DiskANNConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            max_degree: 64,
            build_list_size: 100,
            beam_width: 4,
            alpha: 1.2,
            use_rabitq: true,
            pq_bytes: 32,
        }
    }
}

/// Compressed vector for in-memory navigation
#[derive(Debug, Clone)]
pub struct CompressedNode {
    /// Node ID
    pub id: u64,
    /// Neighbors (graph edges)
    pub neighbors: Vec<u64>,
    /// Compressed vector (RaBitQ or PQ)
    pub compressed: CompressedVector,
}

/// Compressed vector representation
#[derive(Debug, Clone)]
pub enum CompressedVector {
    /// RaBitQ 1-bit quantization
    RaBitQ(QuantizedVector),
    /// Product Quantization codes
    PQ(Vec<u8>),
}

/// Disk-stored full precision vector
#[derive(Debug, Clone)]
pub struct DiskVector {
    pub id: u64,
    pub vector: Vec<f32>,
}

/// Graph edge for Vamana construction
#[derive(Debug, Clone, Copy)]
struct Edge {
    target: u64,
    distance: f32,
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target
    }
}

impl Eq for Edge {}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// DiskANN Index
pub struct DiskANNIndex {
    config: DiskANNConfig,
    /// Base path for storage
    base_path: PathBuf,
    /// In-memory compressed graph
    graph: RwLock<Vec<CompressedNode>>,
    /// RaBitQ quantizer (if enabled)
    rabitq: Option<RaBitQuantizer>,
    /// Medoid (entry point) for search
    medoid: AtomicU64,
    /// Memory-mapped vector file for disk access
    vector_mmap: RwLock<Option<Mmap>>,
    /// Vector count
    count: AtomicU64,
}

impl DiskANNIndex {
    /// Create a new DiskANN index
    pub fn new(config: DiskANNConfig, base_path: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&base_path)?;

        let rabitq = if config.use_rabitq {
            Some(RaBitQuantizer::new(RaBitQConfig {
                dim: config.dim,
                use_orthogonal: true,
                num_rotations: 1,
            }))
        } else {
            None
        };

        Ok(Self {
            config,
            base_path,
            graph: RwLock::new(Vec::new()),
            rabitq,
            medoid: AtomicU64::new(0),
            vector_mmap: RwLock::new(None),
            count: AtomicU64::new(0),
        })
    }

    /// Load existing index from disk
    pub fn load(config: DiskANNConfig, base_path: PathBuf) -> Result<Self> {
        let index = Self::new(config, base_path.clone())?;

        // Load graph
        let graph_path = base_path.join("graph.bin");
        if graph_path.exists() {
            let mut file = File::open(&graph_path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;

            // Deserialize graph (simplified - in production use bincode/serde)
            let graph = Self::deserialize_graph(&buffer, &index)?;
            *index.graph.write() = graph;
        }

        // Memory-map vector file
        let vector_path = base_path.join("vectors.bin");
        if vector_path.exists() {
            let file = File::open(&vector_path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            let count = mmap.len() / (index.config.dim * 4);
            index.count.store(count as u64, Ordering::SeqCst);
            *index.vector_mmap.write() = Some(mmap);
        }

        // Load medoid
        let medoid_path = base_path.join("medoid.bin");
        if medoid_path.exists() {
            let mut file = File::open(&medoid_path)?;
            let mut buffer = [0u8; 8];
            file.read_exact(&mut buffer)?;
            index.medoid.store(u64::from_le_bytes(buffer), Ordering::SeqCst);
        }

        Ok(index)
    }

    /// Deserialize graph from bytes
    fn deserialize_graph(buffer: &[u8], index: &DiskANNIndex) -> Result<Vec<CompressedNode>> {
        let mut graph = Vec::new();
        let mut offset = 0;

        while offset < buffer.len() {
            // Read ID
            if offset + 8 > buffer.len() { break; }
            let id = u64::from_le_bytes(buffer[offset..offset+8].try_into()?);
            offset += 8;

            // Read neighbor count
            if offset + 4 > buffer.len() { break; }
            let num_neighbors = u32::from_le_bytes(buffer[offset..offset+4].try_into()?) as usize;
            offset += 4;

            // Read neighbors
            let mut neighbors = Vec::with_capacity(num_neighbors);
            for _ in 0..num_neighbors {
                if offset + 8 > buffer.len() { break; }
                let neighbor = u64::from_le_bytes(buffer[offset..offset+8].try_into()?);
                neighbors.push(neighbor);
                offset += 8;
            }

            // Read compressed vector (RaBitQ bits)
            let bits_len = (index.config.dim + 63) / 64;
            let mut bits = Vec::with_capacity(bits_len);
            for _ in 0..bits_len {
                if offset + 8 > buffer.len() { break; }
                let word = u64::from_le_bytes(buffer[offset..offset+8].try_into()?);
                bits.push(word);
                offset += 8;
            }

            // Read stats (norm, mean, variance)
            if offset + 12 > buffer.len() { break; }
            let norm = f32::from_le_bytes(buffer[offset..offset+4].try_into()?);
            let mean = f32::from_le_bytes(buffer[offset+4..offset+8].try_into()?);
            let variance = f32::from_le_bytes(buffer[offset+8..offset+12].try_into()?);
            offset += 12;

            let compressed = CompressedVector::RaBitQ(QuantizedVector {
                bits,
                stats: crate::rabitq::VectorStats { norm, mean, variance },
                id,
            });

            graph.push(CompressedNode { id, neighbors, compressed });
        }

        Ok(graph)
    }

    /// Get vector from disk
    fn get_disk_vector(&self, id: u64) -> Result<Vec<f32>> {
        let mmap = self.vector_mmap.read();
        if let Some(ref mmap) = *mmap {
            let vec_size = self.config.dim * 4;
            let offset = id as usize * vec_size;

            if offset + vec_size <= mmap.len() {
                let bytes = &mmap[offset..offset + vec_size];
                let vector: Vec<f32> = bytes.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                return Ok(vector);
            }
        }

        Err(anyhow::anyhow!("Vector {} not found", id))
    }

    /// Append vector to disk file
    fn append_vector(&self, vector: &[f32]) -> Result<u64> {
        let vector_path = self.base_path.join("vectors.bin");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&vector_path)?;

        let bytes: Vec<u8> = vector.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        file.write_all(&bytes)?;

        let id = self.count.fetch_add(1, Ordering::SeqCst);
        Ok(id)
    }

    /// Compute distance between query and compressed node
    fn compressed_distance(&self, query: &[f32], node: &CompressedNode) -> f32 {
        match &node.compressed {
            CompressedVector::RaBitQ(quantized) => {
                if let Some(ref rabitq) = self.rabitq {
                    rabitq.asymmetric_distance(query, quantized)
                } else {
                    f32::MAX
                }
            }
            CompressedVector::PQ(_codes) => {
                // PQ distance computation would go here
                f32::MAX
            }
        }
    }

    /// L2 squared distance
    #[inline]
    fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum()
    }

    /// Greedy search on graph (used during construction)
    fn greedy_search(
        &self,
        query: &[f32],
        start: u64,
        list_size: usize,
    ) -> Vec<(u64, f32)> {
        let graph = self.graph.read();

        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat, u64)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedFloat, u64)> = BinaryHeap::new();

        // Start from entry point
        if let Some(node) = graph.get(start as usize) {
            let dist = self.compressed_distance(query, node);
            candidates.push(Reverse((OrderedFloat(dist), start)));
            results.push((OrderedFloat(dist), start));
            visited.insert(start);
        }

        while let Some(Reverse((OrderedFloat(current_dist), current_id))) = candidates.pop() {
            // Early termination if we have enough good results
            if results.len() >= list_size {
                if let Some(&(OrderedFloat(worst_dist), _)) = results.peek() {
                    if current_dist > worst_dist {
                        break;
                    }
                }
            }

            // Explore neighbors
            if let Some(node) = graph.get(current_id as usize) {
                for &neighbor_id in &node.neighbors {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    visited.insert(neighbor_id);

                    if let Some(neighbor_node) = graph.get(neighbor_id as usize) {
                        let dist = self.compressed_distance(query, neighbor_node);

                        candidates.push(Reverse((OrderedFloat(dist), neighbor_id)));
                        results.push((OrderedFloat(dist), neighbor_id));

                        // Keep only top list_size results
                        while results.len() > list_size {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Convert to sorted vec
        let mut result_vec: Vec<_> = results.into_iter()
            .map(|(OrderedFloat(d), id)| (id, d))
            .collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        result_vec
    }

    /// Robust prune (Vamana algorithm)
    fn robust_prune(&self, node_id: u64, candidates: &[(u64, f32)]) -> Vec<u64> {
        let alpha = self.config.alpha;
        let max_degree = self.config.max_degree;

        let mut neighbors = Vec::new();
        let mut remaining: Vec<_> = candidates.to_vec();

        while !remaining.is_empty() && neighbors.len() < max_degree {
            // Find closest
            let (idx, _) = remaining.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            let (best_id, best_dist) = remaining.remove(idx);
            neighbors.push(best_id);

            // Prune dominated candidates
            remaining.retain(|(id, dist)| {
                // Get distance from best to this candidate
                // In a full implementation, we'd compute this exactly
                // Here we use a heuristic based on triangle inequality
                *dist < alpha * best_dist || *id == best_id
            });
        }

        neighbors
    }

    /// Insert a vector into the index
    pub fn insert(&self, vector: &[f32]) -> Result<u64> {
        // Append to disk
        let id = self.append_vector(vector)?;

        // Compress
        let compressed = if let Some(ref rabitq) = self.rabitq {
            CompressedVector::RaBitQ(rabitq.quantize(vector))
        } else {
            CompressedVector::PQ(vec![0; self.config.pq_bytes])
        };

        // If this is the first vector, it becomes the medoid
        if id == 0 {
            let node = CompressedNode {
                id,
                neighbors: Vec::new(),
                compressed,
            };
            self.graph.write().push(node);
            self.medoid.store(0, Ordering::SeqCst);

            // Refresh mmap
            self.refresh_mmap()?;
            return Ok(id);
        }

        // Find neighbors using greedy search
        let medoid = self.medoid.load(Ordering::SeqCst);
        let candidates = self.greedy_search(vector, medoid, self.config.build_list_size);

        // Prune to get neighbors
        let neighbors = self.robust_prune(id, &candidates);

        // Create node
        let node = CompressedNode {
            id,
            neighbors: neighbors.clone(),
            compressed,
        };

        // Add node to graph
        {
            let mut graph = self.graph.write();
            graph.push(node);

            // Add reverse edges
            for &neighbor_id in &neighbors {
                if let Some(neighbor) = graph.get_mut(neighbor_id as usize) {
                    if !neighbor.neighbors.contains(&id) {
                        neighbor.neighbors.push(id);

                        // Prune if over degree
                        if neighbor.neighbors.len() > self.config.max_degree {
                            // Simple truncation - in production, use robust_prune
                            neighbor.neighbors.truncate(self.config.max_degree);
                        }
                    }
                }
            }
        }

        // Refresh mmap periodically
        if id % 1000 == 0 {
            self.refresh_mmap()?;
        }

        Ok(id)
    }

    /// Batch insert vectors
    pub fn insert_batch(&self, vectors: &[Vec<f32>]) -> Result<Vec<u64>> {
        let mut ids = Vec::with_capacity(vectors.len());

        for vector in vectors {
            let id = self.insert(vector)?;
            ids.push(id);
        }

        // Final mmap refresh
        self.refresh_mmap()?;

        Ok(ids)
    }

    /// Refresh memory-mapped file
    fn refresh_mmap(&self) -> Result<()> {
        let vector_path = self.base_path.join("vectors.bin");
        if vector_path.exists() {
            let file = File::open(&vector_path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            *self.vector_mmap.write() = Some(mmap);
        }
        Ok(())
    }

    /// Beam search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>> {
        // Ensure mmap is fresh
        self.refresh_mmap()?;

        let beam_width = self.config.beam_width;
        let medoid = self.medoid.load(Ordering::SeqCst);
        let search_k = (k * beam_width).max(k * 4);

        // Phase 1: Graph navigation with compressed vectors
        let candidates = self.greedy_search(query, medoid, search_k);

        // If greedy search returned nothing, do brute force on all nodes
        let candidates = if candidates.is_empty() {
            let graph = self.graph.read();
            let mut all: Vec<(u64, f32)> = graph.iter()
                .map(|node| (node.id, self.compressed_distance(query, node)))
                .collect();
            all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            all
        } else {
            candidates
        };

        // Phase 2: Rerank with full vectors from disk
        let mut reranked: Vec<(u64, f32)> = Vec::with_capacity(candidates.len().min(k * 2));

        for (id, _approx_dist) in candidates.iter().take(k * 2) {
            if let Ok(full_vector) = self.get_disk_vector(*id) {
                let exact_dist = Self::l2_squared(query, &full_vector);
                reranked.push((*id, exact_dist));
            }
        }

        // Sort by exact distance
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        reranked.truncate(k);

        Ok(reranked)
    }

    /// Save index to disk
    pub fn save(&self) -> Result<()> {
        // Save graph
        let graph = self.graph.read();
        let mut buffer = Vec::new();

        for node in graph.iter() {
            // ID
            buffer.extend_from_slice(&node.id.to_le_bytes());

            // Neighbor count
            buffer.extend_from_slice(&(node.neighbors.len() as u32).to_le_bytes());

            // Neighbors
            for &neighbor in &node.neighbors {
                buffer.extend_from_slice(&neighbor.to_le_bytes());
            }

            // Compressed vector
            if let CompressedVector::RaBitQ(ref quantized) = node.compressed {
                for &word in &quantized.bits {
                    buffer.extend_from_slice(&word.to_le_bytes());
                }
                buffer.extend_from_slice(&quantized.stats.norm.to_le_bytes());
                buffer.extend_from_slice(&quantized.stats.mean.to_le_bytes());
                buffer.extend_from_slice(&quantized.stats.variance.to_le_bytes());
            }
        }

        let graph_path = self.base_path.join("graph.bin");
        let mut file = File::create(&graph_path)?;
        file.write_all(&buffer)?;

        // Save medoid
        let medoid_path = self.base_path.join("medoid.bin");
        let mut file = File::create(&medoid_path)?;
        file.write_all(&self.medoid.load(Ordering::SeqCst).to_le_bytes())?;

        Ok(())
    }

    /// Get index statistics
    pub fn stats(&self) -> DiskANNStats {
        let graph = self.graph.read();
        let total_edges: usize = graph.iter().map(|n| n.neighbors.len()).sum();
        let avg_degree = if graph.is_empty() { 0.0 } else {
            total_edges as f64 / graph.len() as f64
        };

        DiskANNStats {
            num_vectors: self.count.load(Ordering::SeqCst) as usize,
            num_nodes: graph.len(),
            avg_degree,
            max_degree: self.config.max_degree,
            beam_width: self.config.beam_width,
            uses_rabitq: self.config.use_rabitq,
        }
    }

    /// Get number of vectors
    pub fn len(&self) -> usize {
        self.count.load(Ordering::SeqCst) as usize
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Wrapper for f32 ordering
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// DiskANN statistics
#[derive(Debug, Clone)]
pub struct DiskANNStats {
    pub num_vectors: usize,
    pub num_nodes: usize,
    pub avg_degree: f64,
    pub max_degree: usize,
    pub beam_width: usize,
    pub uses_rabitq: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

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
    fn test_diskann_create() {
        let tmp = TempDir::new().unwrap();
        let config = DiskANNConfig {
            dim: 64,
            max_degree: 32,
            build_list_size: 50,
            beam_width: 4,
            ..Default::default()
        };

        let index = DiskANNIndex::new(config, tmp.path().to_path_buf()).unwrap();
        assert!(index.is_empty());
    }

    #[test]
    fn test_diskann_insert() {
        let tmp = TempDir::new().unwrap();
        let config = DiskANNConfig {
            dim: 64,
            max_degree: 32,
            build_list_size: 50,
            beam_width: 4,
            ..Default::default()
        };

        let index = DiskANNIndex::new(config, tmp.path().to_path_buf()).unwrap();

        // Insert some vectors
        for i in 0..100 {
            let v = random_vector(64, i);
            let id = index.insert(&v).unwrap();
            assert_eq!(id, i);
        }

        assert_eq!(index.len(), 100);

        let stats = index.stats();
        println!("Stats: {:?}", stats);
        assert!(stats.avg_degree > 0.0);
    }

    #[test]
    fn test_diskann_search() {
        let tmp = TempDir::new().unwrap();
        let config = DiskANNConfig {
            dim: 64,
            max_degree: 32,
            build_list_size: 50,
            beam_width: 4,
            ..Default::default()
        };

        let index = DiskANNIndex::new(config, tmp.path().to_path_buf()).unwrap();

        // Insert vectors
        let vectors: Vec<Vec<f32>> = (0..200)
            .map(|i| random_vector(64, i))
            .collect();

        for v in &vectors {
            index.insert(v).unwrap();
        }

        // Search
        let query = random_vector(64, 9999);
        let results = index.search(&query, 10).unwrap();

        assert!(!results.is_empty(), "Search should return results");
        assert!(results.len() <= 10, "Should return at most k results");

        // Verify results are sorted
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i-1].1, "Results should be sorted");
        }

        println!("Search returned {} results", results.len());
    }

    #[test]
    fn test_diskann_save_load() {
        let tmp = TempDir::new().unwrap();
        let config = DiskANNConfig {
            dim: 32,
            max_degree: 16,
            build_list_size: 30,
            beam_width: 2,
            ..Default::default()
        };

        // Create and populate index
        {
            let index = DiskANNIndex::new(config.clone(), tmp.path().to_path_buf()).unwrap();

            for i in 0..50 {
                let v = random_vector(32, i);
                index.insert(&v).unwrap();
            }

            index.save().unwrap();
        }

        // Load and verify
        {
            let index = DiskANNIndex::load(config, tmp.path().to_path_buf()).unwrap();

            assert_eq!(index.len(), 50);

            // Search should work
            let query = random_vector(32, 9999);
            let results = index.search(&query, 5).unwrap();
            assert!(!results.is_empty());
        }
    }

    #[test]
    fn test_diskann_recall() {
        let tmp = TempDir::new().unwrap();
        let config = DiskANNConfig {
            dim: 64,
            max_degree: 32,
            build_list_size: 75,
            beam_width: 8,
            ..Default::default()
        };

        let index = DiskANNIndex::new(config, tmp.path().to_path_buf()).unwrap();

        // Insert vectors
        let vectors: Vec<Vec<f32>> = (0..500)
            .map(|i| random_vector(64, i))
            .collect();

        for v in &vectors {
            index.insert(v).unwrap();
        }

        // Test recall
        let mut total_recall = 0.0;
        let num_queries = 10;

        for q in 0..num_queries {
            let query = random_vector(64, 10000 + q);

            // Get DiskANN results
            let diskann_results: Vec<u64> = index.search(&query, 10).unwrap()
                .iter().map(|(id, _)| *id).collect();

            // Compute exact top-10
            let mut exact: Vec<(u64, f32)> = vectors.iter().enumerate()
                .map(|(i, v)| (i as u64, DiskANNIndex::l2_squared(&query, v)))
                .collect();
            exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let exact_top: Vec<u64> = exact[..10].iter().map(|(id, _)| *id).collect();

            // Count matches
            let matches = diskann_results.iter()
                .filter(|id| exact_top.contains(id))
                .count();

            total_recall += matches as f64 / 10.0;
        }

        let avg_recall = total_recall / num_queries as f64;
        println!("DiskANN Recall@10: {:.1}%", avg_recall * 100.0);

        // Should have some recall (graph-based search is heuristic)
        // With fallback to brute force on small graphs, recall should be decent
        assert!(avg_recall > 0.2, "Recall too low: {}", avg_recall);
    }
}
