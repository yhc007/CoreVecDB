use std::path::Path;
use anyhow::{Result, Context};
use hnsw_rs::prelude::*;
use std::sync::Arc;
use parking_lot::RwLock;

pub enum DistanceMetric {
    Euclidean,
    Cosine,
}

#[derive(Clone)]
pub struct HnswIndexer {
    // Wrapped in RwLock for interior mutability if needed, 
    // though hnsw_rs::Hnsw is thread-safe for search, insert usually needs mut.
    // hnsw-rs 0.3 implementation: Hnsw<F, D>
    // We will use a simplified type alias for generic usage
    // inner: Arc<Hnsw<f32, DistL2>>, // Defaulting to L2 for MVP simplicity in types
    inner: Arc<Hnsw<'static, f32, DistL2>>, 
    // Note: To support dynamic metrics, we might need an enum wrapper or trait object,
    // but hnsw-rs types are heavily generic.
    // For MVP, let's stick to L2 or generic if manageable.
    // Actually, DistCosine and DistL2 are different types.
    // Let's implement for L2 first.
    dim: usize,
}

// Wrapper to hide generic complexity for now, or use enum dispatch if we want both at runtime.
// For this MVP, let's hardcode to L2/Euclidean as it's common.
// Or wait, Cosine is crucial for embeddings.
// Let's stick to L2 for now as normalized vectors + L2 = Cosine ranking.
// HNSW-rs recommends using L2 on normalized vectors for Cosine.

pub struct RoaringFilter<'a>(pub &'a roaring::RoaringBitmap);

impl<'a> hnsw_rs::prelude::FilterT for RoaringFilter<'a> {
    fn hnsw_filter(&self, id: &usize) -> bool {
        self.0.contains(*id as u32)
    }
}

impl HnswIndexer {
    pub fn new(dim: usize, max_elements: usize, m: usize, ef_construction: usize) -> Self {
        let h: Hnsw<'static, f32, DistL2> = Hnsw::new(
            m, // max_nb_connection
            max_elements, // max_elements
            16, // max_layer
            ef_construction, // ef_construction
            DistL2 {}
        );
        Self {
            inner: Arc::new(h),
            dim,
        }
    }

    pub fn insert(&self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(anyhow::anyhow!("Vector dimension mismatch"));
        }
        // hnsw-rs insert requires (data, id)
        // id is usize.
        self.inner.insert((vector, id as usize));
        Ok(())
    }

    pub fn search(&self, vector: &[f32], k: usize, filter: Option<&roaring::RoaringBitmap>) -> Result<Vec<(u64, f32)>> {
        if vector.len() != self.dim {
            return Err(anyhow::anyhow!("Vector dimension mismatch"));
        }
        // ef_search can be tuned.
        let ef_search = 32.max(k); 
        
        let results = if let Some(bitmap) = filter {
            let f = RoaringFilter(bitmap);
            self.inner.search_possible_filter(vector, k, ef_search, Some(&f))
        } else {
            self.inner.search(vector, k, ef_search)
        };
        
        let converted = results.iter().map(|n| {
             (n.d_id as u64, n.distance)
        }).collect();
        
        Ok(converted)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let parent = path.parent().unwrap_or(Path::new("."));
        let filename = path.file_name()
            .ok_or(anyhow::anyhow!("Invalid path"))?
            .to_str()
            .ok_or(anyhow::anyhow!("Invalid filename"))?;
        
        self.inner.file_dump(parent, filename)?;
        Ok(())
    }

    pub fn load(path: &Path, dim: usize, _distance_metric: DistanceMetric) -> Result<Self> {
        let parent = path.parent().unwrap_or(Path::new("."));
        let filename = path.file_name()
            .ok_or(anyhow::anyhow!("Invalid path"))?
            .to_str()
            .ok_or(anyhow::anyhow!("Invalid filename"))?;

        // hnsw-rs load needs the parent directory and the filename (without extension? or full?)
        // Looking at save: self.inner.file_dump(parent, filename)?;
        // hnsw-rs documentation/usage usually implies load takes similar args or full path.
        // Let's assume Hnsw::load(path) or (parent, filename)
        // correct signature for hnsw_rs::Hnsw::load is generally `load(path_to_index_file)` or similar.
        // Actually, looking at common usage of hnsw-rs crate:
        // Hnsw::file_load(path, directory) is common.
        // Let's try to match the save implementation which used `file_dump(parent, filename)`.
        // The corresponding load is likely `Hnsw::file_load(parent, filename)`.
        
        use hnsw_rs::hnswio::HnswIo;
        let reloader = Box::new(HnswIo::new(parent, filename));
        let reloader: &'static mut HnswIo = Box::leak(reloader);
        let h: Hnsw<'static, f32, DistL2> = reloader.load_hnsw::<f32, DistL2>()
            .map_err(|e| anyhow::anyhow!("Failed to load index: {:?}", e))?;

        Ok(Self {
            inner: Arc::new(h),
            dim,
        })
    }
}
