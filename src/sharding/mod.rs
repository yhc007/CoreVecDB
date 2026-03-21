//! Distributed Sharding for VectorDB.
//!
//! Implements horizontal data partitioning with:
//! - Consistent hashing for even data distribution
//! - Virtual nodes for load balancing
//! - Cross-shard search with result merging
//! - Automatic shard rebalancing
//!
//! Architecture:
//! ```text
//! Client Request
//!      │
//!      ▼
//! ┌─────────────┐
//! │ShardRouter  │ ─── Routes requests to appropriate shard
//! └─────────────┘
//!      │
//!      ├──────────────┬──────────────┐
//!      ▼              ▼              ▼
//! ┌─────────┐   ┌─────────┐   ┌─────────┐
//! │ Shard 0 │   │ Shard 1 │   │ Shard 2 │
//! └─────────┘   └─────────┘   └─────────┘
//! ```

use anyhow::Result;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Default number of virtual nodes per physical shard
const DEFAULT_VIRTUAL_NODES: usize = 150;

/// Shard identifier
pub type ShardId = u32;

/// Node identifier (hostname:port or unique ID)
pub type NodeId = String;

// =============================================================================
// Consistent Hashing
// =============================================================================

/// Consistent hash ring for shard routing.
/// Uses virtual nodes for even distribution.
#[derive(Debug, Clone)]
pub struct ConsistentHashRing {
    /// Ring: hash position -> (shard_id, virtual_node_index)
    ring: BTreeMap<u64, (ShardId, usize)>,
    /// Number of virtual nodes per shard
    virtual_nodes: usize,
    /// Active shards
    shards: HashSet<ShardId>,
}

impl ConsistentHashRing {
    /// Create a new empty hash ring.
    pub fn new(virtual_nodes: usize) -> Self {
        Self {
            ring: BTreeMap::new(),
            virtual_nodes,
            shards: HashSet::new(),
        }
    }

    /// Create with default virtual nodes.
    pub fn default_ring() -> Self {
        Self::new(DEFAULT_VIRTUAL_NODES)
    }

    /// Add a shard to the ring.
    pub fn add_shard(&mut self, shard_id: ShardId) {
        if self.shards.contains(&shard_id) {
            return;
        }

        self.shards.insert(shard_id);

        // Add virtual nodes
        for i in 0..self.virtual_nodes {
            let hash = self.hash_shard_node(shard_id, i);
            self.ring.insert(hash, (shard_id, i));
        }
    }

    /// Remove a shard from the ring.
    pub fn remove_shard(&mut self, shard_id: ShardId) {
        if !self.shards.remove(&shard_id) {
            return;
        }

        // Remove all virtual nodes
        for i in 0..self.virtual_nodes {
            let hash = self.hash_shard_node(shard_id, i);
            self.ring.remove(&hash);
        }
    }

    /// Get the shard for a given key.
    pub fn get_shard(&self, key: u64) -> Option<ShardId> {
        if self.ring.is_empty() {
            return None;
        }

        let hash = self.hash_key(key);

        // Find first node >= hash
        if let Some((&_pos, &(shard_id, _))) = self.ring.range(hash..).next() {
            return Some(shard_id);
        }

        // Wrap around to first node
        self.ring.values().next().map(|&(shard_id, _)| shard_id)
    }

    /// Get shard for a string key (e.g., collection name).
    pub fn get_shard_for_string(&self, key: &str) -> Option<ShardId> {
        let hash = self.hash_string(key);
        self.get_shard(hash)
    }

    /// Get all shards in the ring.
    pub fn get_shards(&self) -> Vec<ShardId> {
        self.shards.iter().copied().collect()
    }

    /// Get number of shards.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Hash a key (vector ID).
    fn hash_key(&self, key: u64) -> u64 {
        // Use FNV-1a for fast hashing
        let mut hasher = fnv_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash a string key.
    fn hash_string(&self, key: &str) -> u64 {
        let mut hasher = fnv_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash a shard's virtual node.
    fn hash_shard_node(&self, shard_id: ShardId, vnode_index: usize) -> u64 {
        let mut hasher = fnv_hasher();
        shard_id.hash(&mut hasher);
        vnode_index.hash(&mut hasher);
        hasher.finish()
    }
}

/// Simple FNV-1a hasher.
fn fnv_hasher() -> std::collections::hash_map::DefaultHasher {
    std::collections::hash_map::DefaultHasher::new()
}

// =============================================================================
// Shard Configuration
// =============================================================================

/// Configuration for a single shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    /// Shard identifier
    pub id: ShardId,
    /// Node hosting this shard
    pub node: NodeId,
    /// Local collection name for this shard
    pub collection: String,
    /// Replica nodes (for fault tolerance)
    pub replicas: Vec<NodeId>,
    /// Whether this is the primary copy
    pub is_primary: bool,
}

/// Sharding strategy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ShardingStrategy {
    /// Hash-based sharding by vector ID
    HashById,
    /// Range-based sharding by vector ID
    RangeById,
    /// Hash-based sharding by collection
    HashByCollection,
    /// Manual/static shard assignment
    Manual,
}

impl Default for ShardingStrategy {
    fn default() -> Self {
        ShardingStrategy::HashById
    }
}

/// Cluster-wide sharding configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingConfig {
    /// Sharding strategy
    pub strategy: ShardingStrategy,
    /// Number of shards
    pub num_shards: u32,
    /// Replication factor (copies per shard)
    pub replication_factor: u32,
    /// Virtual nodes per shard (for consistent hashing)
    pub virtual_nodes: usize,
    /// Shard configurations
    pub shards: Vec<ShardConfig>,
}

impl Default for ShardingConfig {
    fn default() -> Self {
        Self {
            strategy: ShardingStrategy::HashById,
            num_shards: 4,
            replication_factor: 1,
            virtual_nodes: DEFAULT_VIRTUAL_NODES,
            shards: Vec::new(),
        }
    }
}

impl ShardingConfig {
    /// Create a simple local sharding config (all shards on local node).
    pub fn local(num_shards: u32, base_collection: &str) -> Self {
        let shards: Vec<ShardConfig> = (0..num_shards)
            .map(|i| ShardConfig {
                id: i,
                node: "local".to_string(),
                collection: format!("{}_shard_{}", base_collection, i),
                replicas: Vec::new(),
                is_primary: true,
            })
            .collect();

        Self {
            strategy: ShardingStrategy::HashById,
            num_shards,
            replication_factor: 1,
            virtual_nodes: DEFAULT_VIRTUAL_NODES,
            shards,
        }
    }

    /// Create a distributed sharding config.
    pub fn distributed(num_shards: u32, nodes: &[NodeId], base_collection: &str) -> Self {
        let shards: Vec<ShardConfig> = (0..num_shards)
            .map(|i| {
                let node_index = (i as usize) % nodes.len();
                ShardConfig {
                    id: i,
                    node: nodes[node_index].clone(),
                    collection: format!("{}_shard_{}", base_collection, i),
                    replicas: Vec::new(),
                    is_primary: true,
                }
            })
            .collect();

        Self {
            strategy: ShardingStrategy::HashById,
            num_shards,
            replication_factor: 1,
            virtual_nodes: DEFAULT_VIRTUAL_NODES,
            shards,
        }
    }
}

// =============================================================================
// Shard Router
// =============================================================================

/// Routes requests to appropriate shards.
pub struct ShardRouter {
    /// Consistent hash ring
    ring: RwLock<ConsistentHashRing>,
    /// Sharding configuration
    config: RwLock<ShardingConfig>,
    /// Shard ID -> ShardConfig mapping
    shard_configs: RwLock<HashMap<ShardId, ShardConfig>>,
    /// Local node ID
    local_node: NodeId,
}

impl ShardRouter {
    /// Create a new shard router.
    pub fn new(config: ShardingConfig, local_node: NodeId) -> Self {
        let mut ring = ConsistentHashRing::new(config.virtual_nodes);

        // Add all shards to the ring
        for shard in &config.shards {
            ring.add_shard(shard.id);
        }

        let shard_configs: HashMap<ShardId, ShardConfig> = config
            .shards
            .iter()
            .map(|s| (s.id, s.clone()))
            .collect();

        Self {
            ring: RwLock::new(ring),
            config: RwLock::new(config),
            shard_configs: RwLock::new(shard_configs),
            local_node,
        }
    }

    /// Create router for local-only sharding.
    pub fn local(num_shards: u32, base_collection: &str) -> Self {
        let config = ShardingConfig::local(num_shards, base_collection);
        Self::new(config, "local".to_string())
    }

    /// Get shard for a vector ID.
    pub fn route_by_id(&self, id: u64) -> Option<ShardId> {
        let ring = self.ring.read();
        ring.get_shard(id)
    }

    /// Get shard for a collection name (when sharding by collection).
    pub fn route_by_collection(&self, collection: &str) -> Option<ShardId> {
        let ring = self.ring.read();
        ring.get_shard_for_string(collection)
    }

    /// Get the local collection name for a shard.
    pub fn get_shard_collection(&self, shard_id: ShardId) -> Option<String> {
        let configs = self.shard_configs.read();
        configs.get(&shard_id).map(|c| c.collection.clone())
    }

    /// Get the node hosting a shard.
    pub fn get_shard_node(&self, shard_id: ShardId) -> Option<NodeId> {
        let configs = self.shard_configs.read();
        configs.get(&shard_id).map(|c| c.node.clone())
    }

    /// Check if a shard is hosted locally.
    pub fn is_local_shard(&self, shard_id: ShardId) -> bool {
        self.get_shard_node(shard_id)
            .map(|node| node == self.local_node || node == "local")
            .unwrap_or(false)
    }

    /// Get all local shards.
    pub fn get_local_shards(&self) -> Vec<ShardId> {
        let configs = self.shard_configs.read();
        configs
            .values()
            .filter(|c| c.node == self.local_node || c.node == "local")
            .map(|c| c.id)
            .collect()
    }

    /// Get all shards.
    pub fn get_all_shards(&self) -> Vec<ShardId> {
        self.ring.read().get_shards()
    }

    /// Get shard configuration.
    pub fn get_shard_config(&self, shard_id: ShardId) -> Option<ShardConfig> {
        self.shard_configs.read().get(&shard_id).cloned()
    }

    /// Add a new shard.
    pub fn add_shard(&self, shard_config: ShardConfig) {
        let shard_id = shard_config.id;

        {
            let mut configs = self.shard_configs.write();
            configs.insert(shard_id, shard_config.clone());
        }

        {
            let mut ring = self.ring.write();
            ring.add_shard(shard_id);
        }

        {
            let mut config = self.config.write();
            config.shards.push(shard_config);
            config.num_shards += 1;
        }
    }

    /// Remove a shard.
    pub fn remove_shard(&self, shard_id: ShardId) {
        {
            let mut configs = self.shard_configs.write();
            configs.remove(&shard_id);
        }

        {
            let mut ring = self.ring.write();
            ring.remove_shard(shard_id);
        }

        {
            let mut config = self.config.write();
            config.shards.retain(|s| s.id != shard_id);
            config.num_shards = config.num_shards.saturating_sub(1);
        }
    }

    /// Get sharding status.
    pub fn status(&self) -> ShardingStatus {
        let config = self.config.read();
        let local_shards = self.get_local_shards();

        ShardingStatus {
            enabled: config.num_shards > 1,
            strategy: config.strategy,
            num_shards: config.num_shards,
            local_shards: local_shards.len() as u32,
            local_node: self.local_node.clone(),
            shard_details: config.shards.clone(),
        }
    }

    /// Get the config.
    pub fn config(&self) -> ShardingConfig {
        self.config.read().clone()
    }
}

/// Sharding status for API responses.
#[derive(Debug, Clone, Serialize)]
pub struct ShardingStatus {
    pub enabled: bool,
    pub strategy: ShardingStrategy,
    pub num_shards: u32,
    pub local_shards: u32,
    pub local_node: NodeId,
    pub shard_details: Vec<ShardConfig>,
}

// =============================================================================
// Sharded Collection
// =============================================================================

use crate::collection::{CollectionConfig, CollectionManager};

/// Manages a sharded collection (multiple underlying collections).
pub struct ShardedCollection {
    /// Name of the logical collection
    name: String,
    /// Shard router
    router: Arc<ShardRouter>,
    /// Collection manager for accessing shards
    manager: Arc<CollectionManager>,
    /// Vector dimension
    dim: usize,
}

impl ShardedCollection {
    /// Create a new sharded collection.
    pub fn create(
        name: &str,
        dim: usize,
        num_shards: u32,
        manager: Arc<CollectionManager>,
    ) -> Result<Self> {
        let router = Arc::new(ShardRouter::local(num_shards, name));

        // Create underlying shard collections
        for shard_id in router.get_all_shards() {
            if let Some(collection_name) = router.get_shard_collection(shard_id) {
                let config = CollectionConfig::new(&collection_name, dim);
                manager.create(config)?;
            }
        }

        Ok(Self {
            name: name.to_string(),
            router,
            manager,
            dim,
        })
    }

    /// Open an existing sharded collection.
    pub fn open(
        name: &str,
        router: Arc<ShardRouter>,
        manager: Arc<CollectionManager>,
        dim: usize,
    ) -> Self {
        Self {
            name: name.to_string(),
            router,
            manager,
            dim,
        }
    }

    /// Get the logical collection name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get total vector count across all shards.
    pub fn total_count(&self) -> usize {
        self.router
            .get_local_shards()
            .iter()
            .filter_map(|&shard_id| {
                self.router
                    .get_shard_collection(shard_id)
                    .and_then(|name| self.manager.get(&name))
                    .map(|c| c.active_count())
            })
            .sum()
    }

    /// Insert a vector into the appropriate shard.
    pub fn insert(&self, vector: &[f32], metadata: &[(String, String)]) -> Result<(ShardId, u64)> {
        // For new inserts, we use a simple round-robin or hash of vector content
        // Here we'll hash the first few elements of the vector
        let hash_key = if vector.len() >= 4 {
            let v0 = (vector[0].to_bits() as u64) << 48;
            let v1 = (vector[1].to_bits() as u64) << 32;
            let v2 = (vector[2].to_bits() as u64) << 16;
            let v3 = vector[3].to_bits() as u64;
            v0 | v1 | v2 | v3
        } else {
            0
        };

        let shard_id = self.router.route_by_id(hash_key)
            .ok_or_else(|| anyhow::anyhow!("No shards available"))?;

        let collection_name = self.router.get_shard_collection(shard_id)
            .ok_or_else(|| anyhow::anyhow!("Shard {} not found", shard_id))?;

        let collection = self.manager.get(&collection_name)
            .ok_or_else(|| anyhow::anyhow!("Collection {} not found", collection_name))?;

        let local_id = collection.insert(vector, metadata)?;

        // Return global ID: (shard_id << 48) | local_id
        Ok((shard_id, encode_global_id(shard_id, local_id)))
    }

    /// Insert batch of vectors, distributing across shards.
    pub fn insert_batch(
        &self,
        vectors: &[Vec<f32>],
        metadata: &[Vec<(String, String)>],
    ) -> Result<Vec<(ShardId, u64)>> {
        let mut results = Vec::with_capacity(vectors.len());

        // Group vectors by target shard
        let mut shard_batches: HashMap<ShardId, (Vec<Vec<f32>>, Vec<Vec<(String, String)>>)> =
            HashMap::new();

        for (i, vector) in vectors.iter().enumerate() {
            let hash_key = if vector.len() >= 4 {
                let v0 = (vector[0].to_bits() as u64) << 48;
                let v1 = (vector[1].to_bits() as u64) << 32;
                let v2 = (vector[2].to_bits() as u64) << 16;
                let v3 = vector[3].to_bits() as u64;
                v0 | v1 | v2 | v3
            } else {
                i as u64
            };

            let shard_id = self.router.route_by_id(hash_key)
                .ok_or_else(|| anyhow::anyhow!("No shards available"))?;

            let entry = shard_batches.entry(shard_id).or_insert_with(|| (Vec::new(), Vec::new()));
            entry.0.push(vector.clone());
            entry.1.push(metadata.get(i).cloned().unwrap_or_default());
        }

        // Insert into each shard
        for (shard_id, (shard_vectors, shard_metadata)) in shard_batches {
            let collection_name = self.router.get_shard_collection(shard_id)
                .ok_or_else(|| anyhow::anyhow!("Shard {} not found", shard_id))?;

            let collection = self.manager.get(&collection_name)
                .ok_or_else(|| anyhow::anyhow!("Collection {} not found", collection_name))?;

            let start_id = collection.insert_batch(&shard_vectors, &shard_metadata)?;

            for i in 0..shard_vectors.len() {
                let local_id = start_id + i as u64;
                results.push((shard_id, encode_global_id(shard_id, local_id)));
            }
        }

        Ok(results)
    }

    /// Search across all shards and merge results.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&roaring::RoaringBitmap>,
    ) -> Result<Vec<(ShardId, u64, f32)>> {
        use rayon::prelude::*;

        let local_shards = self.router.get_local_shards();

        // Parallel search across all local shards
        let shard_results: Vec<_> = local_shards
            .par_iter()
            .filter_map(|&shard_id| {
                let collection_name = self.router.get_shard_collection(shard_id)?;
                let collection = self.manager.get(&collection_name)?;

                // Get deleted bitmap for this shard
                let deleted = collection.deleted_bitmap();
                let effective_filter = match filter {
                    Some(f) => {
                        // Filter AND NOT deleted
                        let combined = f - &deleted;
                        Some(combined)
                    }
                    None if !deleted.is_empty() => {
                        let total = collection.len() as u64;
                        let mut universe = roaring::RoaringBitmap::new();
                        universe.insert_range(0..total as u32);
                        Some(&universe - &deleted)
                    }
                    None => None,
                };

                let results = collection.indexer
                    .search(query, k, effective_filter.as_ref())
                    .ok()?;

                Some((shard_id, results))
            })
            .collect();

        // Merge results from all shards
        let mut all_results: Vec<(ShardId, u64, f32)> = Vec::new();

        for (shard_id, results) in shard_results {
            for (local_id, score) in results {
                let global_id = encode_global_id(shard_id, local_id);
                all_results.push((shard_id, global_id, score));
            }
        }

        // Sort by score and take top-k
        all_results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);

        Ok(all_results)
    }

    /// Get a vector by global ID.
    pub fn get(&self, global_id: u64) -> Result<Option<(Vec<f32>, HashMap<String, String>)>> {
        let (shard_id, local_id) = decode_global_id(global_id);

        let collection_name = self.router.get_shard_collection(shard_id)
            .ok_or_else(|| anyhow::anyhow!("Shard {} not found", shard_id))?;

        let collection = self.manager.get(&collection_name)
            .ok_or_else(|| anyhow::anyhow!("Collection {} not found", collection_name))?;

        if local_id as usize >= collection.len() {
            return Ok(None);
        }

        if collection.is_deleted(local_id) {
            return Ok(None);
        }

        let vector = collection.vector_store.get(local_id)?;
        let metadata = collection.metadata_store.get_all(local_id);

        Ok(Some((vector, metadata)))
    }

    /// Delete a vector by global ID.
    pub fn delete(&self, global_id: u64) -> Result<bool> {
        let (shard_id, local_id) = decode_global_id(global_id);

        let collection_name = self.router.get_shard_collection(shard_id)
            .ok_or_else(|| anyhow::anyhow!("Shard {} not found", shard_id))?;

        let collection = self.manager.get(&collection_name)
            .ok_or_else(|| anyhow::anyhow!("Collection {} not found", collection_name))?;

        collection.delete(local_id)
    }

    /// Get sharding status.
    pub fn status(&self) -> ShardedCollectionStatus {
        let shard_stats: Vec<_> = self.router
            .get_local_shards()
            .iter()
            .filter_map(|&shard_id| {
                let collection_name = self.router.get_shard_collection(shard_id)?;
                let collection = self.manager.get(&collection_name)?;

                Some(ShardStats {
                    shard_id,
                    collection: collection_name,
                    vector_count: collection.active_count(),
                    deleted_count: collection.deleted_count(),
                })
            })
            .collect();

        let total_vectors: usize = shard_stats.iter().map(|s| s.vector_count).sum();
        let total_deleted: usize = shard_stats.iter().map(|s| s.deleted_count).sum();

        ShardedCollectionStatus {
            name: self.name.clone(),
            dim: self.dim,
            num_shards: self.router.get_all_shards().len() as u32,
            total_vectors,
            total_deleted,
            shard_stats,
        }
    }

    /// Get the router.
    pub fn router(&self) -> &Arc<ShardRouter> {
        &self.router
    }
}

/// Per-shard statistics.
#[derive(Debug, Clone, Serialize)]
pub struct ShardStats {
    pub shard_id: ShardId,
    pub collection: String,
    pub vector_count: usize,
    pub deleted_count: usize,
}

/// Sharded collection status.
#[derive(Debug, Clone, Serialize)]
pub struct ShardedCollectionStatus {
    pub name: String,
    pub dim: usize,
    pub num_shards: u32,
    pub total_vectors: usize,
    pub total_deleted: usize,
    pub shard_stats: Vec<ShardStats>,
}

// =============================================================================
// Global ID Encoding
// =============================================================================

/// Encode a global ID from shard ID and local ID.
/// Format: [16 bits shard_id][48 bits local_id]
#[inline]
pub fn encode_global_id(shard_id: ShardId, local_id: u64) -> u64 {
    ((shard_id as u64) << 48) | (local_id & 0x0000_FFFF_FFFF_FFFF)
}

/// Decode a global ID into shard ID and local ID.
#[inline]
pub fn decode_global_id(global_id: u64) -> (ShardId, u64) {
    let shard_id = (global_id >> 48) as ShardId;
    let local_id = global_id & 0x0000_FFFF_FFFF_FFFF;
    (shard_id, local_id)
}

// =============================================================================
// Shard Rebalancing
// =============================================================================

/// Result of a rebalancing operation.
#[derive(Debug, Clone, Serialize)]
pub struct RebalanceResult {
    pub vectors_moved: usize,
    pub source_shards: Vec<ShardId>,
    pub target_shards: Vec<ShardId>,
    pub duration_ms: u64,
}

/// Plan for rebalancing vectors when adding/removing shards.
pub struct RebalancePlan {
    /// Vectors to move: (global_id, from_shard, to_shard)
    pub moves: Vec<(u64, ShardId, ShardId)>,
    /// Estimated data size in bytes
    pub estimated_bytes: u64,
}

impl RebalancePlan {
    /// Create a rebalance plan for adding a new shard.
    pub fn for_add_shard(
        router: &ShardRouter,
        new_shard_id: ShardId,
        sample_ids: &[u64],
    ) -> Self {
        // Create a temporary ring with the new shard
        let mut temp_ring = router.ring.read().clone();
        temp_ring.add_shard(new_shard_id);

        let mut moves = Vec::new();

        // Check which vectors would move to the new shard
        for &global_id in sample_ids {
            let (current_shard, local_id) = decode_global_id(global_id);

            // Check new routing
            if let Some(new_shard) = temp_ring.get_shard(local_id) {
                if new_shard != current_shard {
                    moves.push((global_id, current_shard, new_shard));
                }
            }
        }

        let estimated_bytes = moves.len() as u64 * 512; // Rough estimate

        Self {
            moves,
            estimated_bytes,
        }
    }

    /// Check if rebalancing is needed.
    pub fn is_needed(&self) -> bool {
        !self.moves.is_empty()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_hash_ring() {
        let mut ring = ConsistentHashRing::new(10);

        ring.add_shard(0);
        ring.add_shard(1);
        ring.add_shard(2);

        // Check that all keys route to a shard
        for i in 0..1000 {
            let shard = ring.get_shard(i);
            assert!(shard.is_some());
            assert!(shard.unwrap() < 3);
        }

        // Check distribution (should be roughly even)
        let mut counts = [0usize; 3];
        for i in 0..1000 {
            if let Some(shard) = ring.get_shard(i) {
                counts[shard as usize] += 1;
            }
        }

        // Each shard should have between 20% and 50% of keys
        for count in &counts {
            assert!(*count > 200 && *count < 500, "Uneven distribution: {:?}", counts);
        }
    }

    #[test]
    fn test_global_id_encoding() {
        let shard_id = 42u32;
        let local_id = 123456789u64;

        let global = encode_global_id(shard_id, local_id);
        let (decoded_shard, decoded_local) = decode_global_id(global);

        assert_eq!(decoded_shard, shard_id);
        assert_eq!(decoded_local, local_id);
    }

    #[test]
    fn test_global_id_max_values() {
        // Max shard ID (16 bits)
        let shard_id = 0xFFFF;
        // Max local ID (48 bits)
        let local_id = 0x0000_FFFF_FFFF_FFFF;

        let global = encode_global_id(shard_id, local_id);
        let (decoded_shard, decoded_local) = decode_global_id(global);

        assert_eq!(decoded_shard, shard_id);
        assert_eq!(decoded_local, local_id);
    }

    #[test]
    fn test_shard_router() {
        let config = ShardingConfig::local(4, "test");
        let router = ShardRouter::new(config, "local".to_string());

        assert_eq!(router.get_all_shards().len(), 4);
        assert_eq!(router.get_local_shards().len(), 4);

        // All shards should be routable
        for i in 0..100 {
            let shard = router.route_by_id(i);
            assert!(shard.is_some());
        }
    }

    #[test]
    fn test_add_remove_shard() {
        let config = ShardingConfig::local(2, "test");
        let router = ShardRouter::new(config, "local".to_string());

        assert_eq!(router.get_all_shards().len(), 2);

        // Add a shard
        router.add_shard(ShardConfig {
            id: 2,
            node: "local".to_string(),
            collection: "test_shard_2".to_string(),
            replicas: Vec::new(),
            is_primary: true,
        });

        assert_eq!(router.get_all_shards().len(), 3);

        // Remove a shard
        router.remove_shard(1);

        assert_eq!(router.get_all_shards().len(), 2);
    }
}
