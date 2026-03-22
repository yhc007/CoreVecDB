//! Segment-based Vector Storage for VectorDB.
//!
//! Organizes vectors into fixed-size segments for better scalability:
//! - **Segments**: Fixed-size chunks of vectors
//! - **Active Segment**: Currently accepting new vectors
//! - **Sealed Segments**: Read-only, can be compressed
//!
//! # Concept
//! - Each segment contains up to `segment_size` vectors
//! - Global ID = segment_id * segment_size + local_id
//! - Sealed segments can be compressed or archived
//!
//! # Usage
//! ```rust,ignore
//! use vectordb::storage::segmented::{SegmentedVectorStore, SegmentConfig};
//!
//! let config = SegmentConfig::default();
//! let store = SegmentedVectorStore::new("data/vectors", 128, config)?;
//!
//! // Insert vector
//! let id = store.insert(&vector)?;
//!
//! // Get vector (routes to correct segment)
//! let vector = store.get(id)?;
//! ```

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use memmap2::{Mmap, MmapMut};
use parking_lot::RwLock;

use anyhow::Result;

/// Segment state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentState {
    /// Actively accepting new vectors.
    Active,
    /// Read-only, no new vectors accepted.
    Sealed,
    /// Compressed for storage efficiency.
    Compressed,
}

/// Configuration for segmented storage.
#[derive(Debug, Clone)]
pub struct SegmentConfig {
    /// Maximum vectors per segment.
    pub segment_size: usize,
    /// Auto-seal when segment reaches capacity.
    pub auto_seal: bool,
    /// Enable compression for sealed segments.
    pub compress_sealed: bool,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            segment_size: 100_000,
            auto_seal: true,
            compress_sealed: false,
        }
    }
}

/// A single segment of vector storage.
pub struct Segment {
    /// Segment ID.
    id: u64,
    /// Base path for this segment's files.
    path: PathBuf,
    /// Vector dimension.
    dim: usize,
    /// Bytes per vector.
    vector_bytes: usize,
    /// Maximum capacity.
    capacity: usize,
    /// Current state.
    state: RwLock<SegmentState>,
    /// Memory-mapped data for reading.
    mmap: RwLock<Option<Mmap>>,
    /// File handle for writing.
    file: RwLock<File>,
    /// Current vector count.
    vector_count: AtomicUsize,
}

impl Segment {
    /// Create a new segment.
    pub fn new(id: u64, base_path: &Path, dim: usize, capacity: usize) -> io::Result<Self> {
        let path = base_path.join(format!("segment_{}.bin", id));

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;

        let vector_bytes = dim * std::mem::size_of::<f32>();

        // Count existing vectors
        let file_len = file.metadata()?.len() as usize;
        let vector_count = file_len / vector_bytes;

        // Create mmap if file has content
        let mmap = if file_len > 0 {
            Some(unsafe { Mmap::map(&file)? })
        } else {
            None
        };

        // Determine initial state
        let state = if vector_count >= capacity {
            SegmentState::Sealed
        } else {
            SegmentState::Active
        };

        Ok(Self {
            id,
            path,
            dim,
            vector_bytes,
            capacity,
            state: RwLock::new(state),
            mmap: RwLock::new(mmap),
            file: RwLock::new(file),
            vector_count: AtomicUsize::new(vector_count),
        })
    }

    /// Open an existing segment.
    pub fn open(id: u64, base_path: &Path, dim: usize, capacity: usize) -> io::Result<Self> {
        Self::new(id, base_path, dim, capacity)
    }

    /// Get segment ID.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get segment state.
    pub fn state(&self) -> SegmentState {
        *self.state.read()
    }

    /// Get vector count.
    pub fn len(&self) -> usize {
        self.vector_count.load(Ordering::SeqCst)
    }

    /// Check if segment is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if segment is full.
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Check if segment can accept new vectors.
    pub fn can_insert(&self) -> bool {
        *self.state.read() == SegmentState::Active && !self.is_full()
    }

    /// Insert a vector into this segment.
    pub fn insert(&self, vector: &[f32]) -> io::Result<usize> {
        if *self.state.read() != SegmentState::Active {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot insert into non-active segment",
            ));
        }

        if vector.len() != self.dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Vector dimension mismatch: expected {}, got {}", self.dim, vector.len()),
            ));
        }

        let local_id = self.vector_count.fetch_add(1, Ordering::SeqCst);

        if local_id >= self.capacity {
            self.vector_count.fetch_sub(1, Ordering::SeqCst);
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "Segment is full",
            ));
        }

        // Write vector
        let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut file = self.file.write();
        file.seek(SeekFrom::End(0))?;
        file.write_all(&bytes)?;

        Ok(local_id)
    }

    /// Get a vector by local ID.
    pub fn get(&self, local_id: usize) -> io::Result<Vec<f32>> {
        if local_id >= self.len() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Local ID {} not found in segment {}", local_id, self.id),
            ));
        }

        let mmap = self.mmap.read();

        if let Some(ref mmap) = *mmap {
            let offset = local_id * self.vector_bytes;
            let end = offset + self.vector_bytes;

            if end > mmap.len() {
                // Data not yet in mmap, refresh needed
                drop(mmap);
                self.refresh_mmap()?;
                return self.get(local_id);
            }

            let bytes = &mmap[offset..end];
            let vector: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            return Ok(vector);
        }

        // No mmap, try refresh
        drop(mmap);
        self.refresh_mmap()?;
        self.get(local_id)
    }

    /// Refresh the memory map.
    fn refresh_mmap(&self) -> io::Result<()> {
        let file = self.file.read();
        let len = file.metadata()?.len();

        if len > 0 {
            let new_mmap = unsafe { Mmap::map(&*file)? };
            *self.mmap.write() = Some(new_mmap);
        }

        Ok(())
    }

    /// Seal the segment (make read-only).
    pub fn seal(&self) {
        *self.state.write() = SegmentState::Sealed;
    }

    /// Flush writes to disk.
    pub fn flush(&self) -> io::Result<()> {
        self.file.write().flush()?;
        self.refresh_mmap()
    }

    /// Get segment info.
    pub fn info(&self) -> SegmentInfo {
        SegmentInfo {
            id: self.id,
            state: *self.state.read(),
            vector_count: self.len(),
            capacity: self.capacity,
            bytes_used: self.len() * self.vector_bytes,
        }
    }
}

/// Information about a segment.
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    pub id: u64,
    pub state: SegmentState,
    pub vector_count: usize,
    pub capacity: usize,
    pub bytes_used: usize,
}

/// Segmented vector store.
pub struct SegmentedVectorStore {
    /// Base directory for segments.
    base_path: PathBuf,
    /// Vector dimension.
    dim: usize,
    /// Configuration.
    config: SegmentConfig,
    /// All segments.
    segments: RwLock<Vec<Segment>>,
    /// Index of active segment.
    active_segment_idx: RwLock<usize>,
    /// Next segment ID.
    next_segment_id: AtomicU64,
    /// Total vector count.
    total_count: AtomicUsize,
}

impl SegmentedVectorStore {
    /// Create a new segmented vector store.
    pub fn new(base_path: &str, dim: usize, config: SegmentConfig) -> io::Result<Self> {
        let path = PathBuf::from(base_path);
        fs::create_dir_all(&path)?;

        // Create initial segment
        let segment = Segment::new(0, &path, dim, config.segment_size)?;
        let total_count = segment.len();

        Ok(Self {
            base_path: path,
            dim,
            config,
            segments: RwLock::new(vec![segment]),
            active_segment_idx: RwLock::new(0),
            next_segment_id: AtomicU64::new(1),
            total_count: AtomicUsize::new(total_count),
        })
    }

    /// Open an existing segmented store.
    pub fn open(base_path: &str, dim: usize, config: SegmentConfig) -> io::Result<Self> {
        let path = PathBuf::from(base_path);

        if !path.exists() {
            return Self::new(base_path, dim, config);
        }

        // Discover existing segments
        let mut segments = Vec::new();
        let mut max_id = 0u64;
        let mut active_idx = 0;

        for entry in fs::read_dir(&path)? {
            let entry = entry?;
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();

            if name.starts_with("segment_") && name.ends_with(".bin") {
                let id_str = name
                    .strip_prefix("segment_")
                    .and_then(|s| s.strip_suffix(".bin"))
                    .unwrap_or("0");

                if let Ok(id) = id_str.parse::<u64>() {
                    let segment = Segment::open(id, &path, dim, config.segment_size)?;

                    if segment.state() == SegmentState::Active {
                        active_idx = segments.len();
                    }

                    max_id = max_id.max(id);
                    segments.push(segment);
                }
            }
        }

        // Sort by ID
        segments.sort_by_key(|s| s.id());

        // Ensure at least one segment
        if segments.is_empty() {
            let segment = Segment::new(0, &path, dim, config.segment_size)?;
            segments.push(segment);
        }

        // Find active segment
        active_idx = segments
            .iter()
            .position(|s| s.state() == SegmentState::Active)
            .unwrap_or(segments.len() - 1);

        let total_count: usize = segments.iter().map(|s| s.len()).sum();

        Ok(Self {
            base_path: path,
            dim,
            config,
            segments: RwLock::new(segments),
            active_segment_idx: RwLock::new(active_idx),
            next_segment_id: AtomicU64::new(max_id + 1),
            total_count: AtomicUsize::new(total_count),
        })
    }

    /// Convert global ID to (segment_index, local_id).
    fn global_to_local(&self, global_id: u64) -> (usize, usize) {
        let segment_idx = (global_id as usize) / self.config.segment_size;
        let local_id = (global_id as usize) % self.config.segment_size;
        (segment_idx, local_id)
    }

    /// Convert (segment_index, local_id) to global ID.
    fn local_to_global(&self, segment_idx: usize, local_id: usize) -> u64 {
        (segment_idx * self.config.segment_size + local_id) as u64
    }

    /// Insert a vector.
    pub fn insert(&self, vector: &[f32]) -> Result<u64> {
        if vector.len() != self.dim {
            anyhow::bail!("Vector dimension mismatch: expected {}, got {}", self.dim, vector.len());
        }

        let mut active_idx = self.active_segment_idx.write();
        let segments = self.segments.read();

        // Get active segment
        let active = &segments[*active_idx];

        // Try to insert
        match active.insert(vector) {
            Ok(local_id) => {
                self.total_count.fetch_add(1, Ordering::SeqCst);
                return Ok(self.local_to_global(*active_idx, local_id));
            }
            Err(e) if e.kind() == io::ErrorKind::Other => {
                // Segment full, need new segment
            }
            Err(e) => return Err(e.into()),
        }

        // Need new segment
        drop(segments);

        // Seal current and create new
        self.seal_and_create_new(&mut active_idx)?;

        // Retry insert
        let segments = self.segments.read();
        let active = &segments[*active_idx];
        let local_id = active.insert(vector)?;
        self.total_count.fetch_add(1, Ordering::SeqCst);

        Ok(self.local_to_global(*active_idx, local_id))
    }

    /// Seal current segment and create a new one.
    fn seal_and_create_new(&self, active_idx: &mut usize) -> io::Result<()> {
        let mut segments = self.segments.write();

        // Seal current
        if *active_idx < segments.len() {
            segments[*active_idx].seal();
        }

        // Create new segment
        let new_id = self.next_segment_id.fetch_add(1, Ordering::SeqCst);
        let new_segment = Segment::new(new_id, &self.base_path, self.dim, self.config.segment_size)?;

        segments.push(new_segment);
        *active_idx = segments.len() - 1;

        Ok(())
    }

    /// Get a vector by global ID.
    pub fn get(&self, global_id: u64) -> Result<Vec<f32>> {
        let (segment_idx, local_id) = self.global_to_local(global_id);

        let segments = self.segments.read();

        if segment_idx >= segments.len() {
            anyhow::bail!("Segment {} not found for ID {}", segment_idx, global_id);
        }

        let segment = &segments[segment_idx];
        let vector = segment.get(local_id)?;

        Ok(vector)
    }

    /// Get total vector count.
    pub fn len(&self) -> usize {
        self.total_count.load(Ordering::SeqCst)
    }

    /// Check if store is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Flush all segments.
    pub fn flush(&self) -> Result<()> {
        let segments = self.segments.read();
        for segment in segments.iter() {
            segment.flush()?;
        }
        Ok(())
    }

    /// Get store statistics.
    pub fn stats(&self) -> SegmentedStoreStats {
        let segments = self.segments.read();

        let segment_infos: Vec<SegmentInfo> = segments.iter().map(|s| s.info()).collect();

        let total_capacity: usize = segments.len() * self.config.segment_size;
        let total_vectors = self.total_count.load(Ordering::SeqCst);

        SegmentedStoreStats {
            total_segments: segments.len(),
            active_segments: segment_infos.iter().filter(|s| s.state == SegmentState::Active).count(),
            sealed_segments: segment_infos.iter().filter(|s| s.state == SegmentState::Sealed).count(),
            total_vectors,
            total_capacity,
            utilization: if total_capacity > 0 {
                total_vectors as f64 / total_capacity as f64
            } else {
                0.0
            },
            segment_infos,
        }
    }
}

/// Statistics for segmented store.
#[derive(Debug, Clone)]
pub struct SegmentedStoreStats {
    pub total_segments: usize,
    pub active_segments: usize,
    pub sealed_segments: usize,
    pub total_vectors: usize,
    pub total_capacity: usize,
    pub utilization: f64,
    pub segment_infos: Vec<SegmentInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_segment_basic() {
        let temp_dir = TempDir::new().unwrap();

        let segment = Segment::new(0, temp_dir.path(), 4, 100).unwrap();

        // Insert vectors
        let id1 = segment.insert(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let id2 = segment.insert(&[5.0, 6.0, 7.0, 8.0]).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(segment.len(), 2);

        // Flush and read
        segment.flush().unwrap();

        let v1 = segment.get(0).unwrap();
        assert_eq!(v1, vec![1.0, 2.0, 3.0, 4.0]);

        let v2 = segment.get(1).unwrap();
        assert_eq!(v2, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_segment_seal() {
        let temp_dir = TempDir::new().unwrap();

        let segment = Segment::new(0, temp_dir.path(), 4, 100).unwrap();
        segment.insert(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        assert_eq!(segment.state(), SegmentState::Active);

        segment.seal();

        assert_eq!(segment.state(), SegmentState::Sealed);

        // Cannot insert into sealed segment
        let result = segment.insert(&[5.0, 6.0, 7.0, 8.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_segmented_store_basic() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("segments");

        let config = SegmentConfig {
            segment_size: 100,
            ..Default::default()
        };

        let store = SegmentedVectorStore::new(path.to_str().unwrap(), 4, config).unwrap();

        // Insert vectors
        let id1 = store.insert(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let id2 = store.insert(&[5.0, 6.0, 7.0, 8.0]).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(store.len(), 2);

        // Get vectors
        let v1 = store.get(id1).unwrap();
        assert_eq!(v1, vec![1.0, 2.0, 3.0, 4.0]);

        let v2 = store.get(id2).unwrap();
        assert_eq!(v2, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_segment_overflow() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("segments");

        let config = SegmentConfig {
            segment_size: 3, // Very small for testing
            auto_seal: true,
            compress_sealed: false,
        };

        let store = SegmentedVectorStore::new(path.to_str().unwrap(), 4, config).unwrap();

        // Insert 5 vectors (should create 2 segments)
        for i in 0..5 {
            store.insert(&[i as f32; 4]).unwrap();
        }

        assert_eq!(store.len(), 5);

        let stats = store.stats();
        assert_eq!(stats.total_segments, 2);
        assert_eq!(stats.sealed_segments, 1);
        assert_eq!(stats.active_segments, 1);

        // Verify all vectors accessible
        for i in 0..5 {
            let v = store.get(i).unwrap();
            assert_eq!(v, vec![i as f32; 4]);
        }
    }

    #[test]
    fn test_global_local_id_conversion() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("segments");

        let config = SegmentConfig {
            segment_size: 100,
            ..Default::default()
        };

        let store = SegmentedVectorStore::new(path.to_str().unwrap(), 4, config).unwrap();

        // Test conversion
        let (seg_idx, local_id) = store.global_to_local(150);
        assert_eq!(seg_idx, 1);
        assert_eq!(local_id, 50);

        let global = store.local_to_global(2, 75);
        assert_eq!(global, 275);
    }

    #[test]
    fn test_store_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("segments");

        let config = SegmentConfig {
            segment_size: 100,
            ..Default::default()
        };

        // Create and populate store
        {
            let store = SegmentedVectorStore::new(path.to_str().unwrap(), 4, config.clone()).unwrap();

            for i in 0..10 {
                store.insert(&[i as f32; 4]).unwrap();
            }

            store.flush().unwrap();
        }

        // Reopen and verify
        {
            let store = SegmentedVectorStore::open(path.to_str().unwrap(), 4, config).unwrap();

            assert_eq!(store.len(), 10);

            for i in 0..10 {
                let v = store.get(i).unwrap();
                assert_eq!(v, vec![i as f32; 4]);
            }
        }
    }
}
