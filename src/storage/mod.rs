use std::fs::{File, OpenOptions};
use std::io::{Write, Seek, SeekFrom, Read as IoRead};
use std::num::NonZeroUsize;
use std::path::Path;
use memmap2::Mmap;
use parking_lot::{RwLock, Mutex};
use sled::Db;
use anyhow::Result;
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use lru::LruCache;

use crate::quantization::{ScalarQuantizer, QuantizerStats};
use crate::payload::{PayloadIndex, FilterQuery, PayloadIndexFullStats};
use roaring::RoaringBitmap;

/// Trait for storing vectors
pub trait VectorStore: Send + Sync {
    fn insert(&self, vector: &[f32]) -> Result<u64>;
    fn get(&self, id: u64) -> Result<Vec<f32>>;
    fn len(&self) -> usize;
    fn flush(&self) -> Result<()>;

    /// Batch insert for bulk operations.
    /// Returns starting ID; vectors get sequential IDs from start_id.
    /// Default implementation calls insert() in a loop.
    fn insert_batch(&self, vectors: &[Vec<f32>]) -> Result<u64> {
        if vectors.is_empty() {
            return Ok(self.len() as u64);
        }
        let start_id = self.insert(&vectors[0])?;
        for v in vectors.iter().skip(1) {
            self.insert(v)?;
        }
        self.flush()?;
        Ok(start_id)
    }
}

/// Metadata entry for batch operations.
#[derive(Debug, Clone)]
pub struct MetadataEntry {
    pub id: u64,
    pub key: String,
    pub value: String,
}

/// Trait for storing metadata
pub trait MetadataStore: Send + Sync {
    fn insert(&self, id: u64, key: String, value: String) -> Result<()>;
    fn get(&self, id: u64, key: &str) -> Result<Option<String>>;

    /// Get all metadata for a given ID.
    fn get_all(&self, id: u64) -> std::collections::HashMap<String, String>;

    /// Try to filter by key-value conditions using payload index.
    /// Returns None if indexing is not available (will fallback to post-filter).
    /// Returns Some(bitmap) with matching IDs if index is available.
    fn try_filter_and(&self, _conditions: &[(&str, &str)]) -> Option<RoaringBitmap> {
        None // Default: no indexing available
    }

    /// Batch insert metadata entries.
    /// Default implementation calls insert() in a loop.
    fn insert_batch(&self, entries: &[MetadataEntry]) -> Result<()> {
        for entry in entries {
            self.insert(entry.id, entry.key.clone(), entry.value.clone())?;
        }
        Ok(())
    }

    /// Downcast to concrete type for advanced features (e.g., range queries).
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}

/// Extended trait for indexed metadata store
pub trait IndexedMetadata: MetadataStore {
    /// Filter by metadata conditions, returns matching IDs
    fn filter(&self, query: &FilterQuery) -> Option<RoaringBitmap>;

    /// Filter by simple key-value conditions (AND)
    fn filter_and(&self, conditions: &[(&str, &str)]) -> Option<RoaringBitmap>;

    /// Get index statistics
    fn index_stats(&self) -> PayloadIndexFullStats;
}

const WRITE_BUFFER_SIZE: usize = 256; // Flush after 256 vectors
const FLUSH_THRESHOLD_BYTES: usize = 256 * 1024; // Or 256KB

pub struct MemmapVectorStore {
    file: Mutex<File>,              // Mutex for write-heavy workload
    mmap: RwLock<Option<Mmap>>,     // parking_lot RwLock
    mmap_len: AtomicUsize,          // Track current mmap length
    dim: usize,
    count: AtomicUsize,             // Atomic for lock-free reads
    write_buffer: Mutex<WriteBuffer>,
    flushing: AtomicUsize,          // Flag to track if flush is in progress
}

struct WriteBuffer {
    data: Vec<u8>,
    pending_count: usize,
}

impl MemmapVectorStore {
    pub fn new(path: &str, dim: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        let meta = file.metadata()?;
        let len = meta.len() as usize;
        let vec_size = dim * size_of::<f32>();

        let count = if len > 0 { len / vec_size } else { 0 };

        let mmap = if len > 0 {
            unsafe { Some(Mmap::map(&file)?) }
        } else {
            None
        };

        Ok(Self {
            file: Mutex::new(file),
            mmap: RwLock::new(mmap),
            mmap_len: AtomicUsize::new(len),
            dim,
            count: AtomicUsize::new(count),
            write_buffer: Mutex::new(WriteBuffer {
                data: Vec::with_capacity(FLUSH_THRESHOLD_BYTES),
                pending_count: 0,
            }),
            flushing: AtomicUsize::new(0),
        })
    }

    fn refresh_mmap_if_needed(&self, needed_len: usize) -> Result<bool> {
        let current_len = self.mmap_len.load(Ordering::Acquire);
        if current_len >= needed_len {
            return Ok(false); // No refresh needed
        }

        let file = self.file.lock();
        let file_len = file.metadata()?.len() as usize;
        if file_len == 0 {
            return Ok(false);
        }

        let mmap = unsafe { Mmap::map(&*file)? };
        let new_len = mmap.len();
        *self.mmap.write() = Some(mmap);
        self.mmap_len.store(new_len, Ordering::Release);
        Ok(true)
    }

    fn flush_buffer_internal(&self, buffer: &mut WriteBuffer) -> Result<()> {
        if buffer.data.is_empty() {
            return Ok(());
        }

        let new_data_len = buffer.data.len();

        {
            let mut file = self.file.lock();
            file.seek(SeekFrom::End(0))?;
            file.write_all(&buffer.data)?;
            file.flush()?;
        }

        buffer.data.clear();
        buffer.pending_count = 0;

        // Auto-refresh mmap after flush for better read performance
        let current_len = self.mmap_len.load(Ordering::Acquire);
        let _ = self.refresh_mmap_if_needed(current_len + new_data_len);

        Ok(())
    }
}

impl VectorStore for MemmapVectorStore {
    fn insert(&self, vector: &[f32]) -> Result<u64> {
        if vector.len() != self.dim {
            return Err(anyhow::anyhow!("Vector dimension mismatch"));
        }

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                vector.as_ptr() as *const u8,
                vector.len() * size_of::<f32>(),
            )
        };

        // Atomic increment first - ensures ID is reserved
        let id = self.count.fetch_add(1, Ordering::SeqCst) as u64;

        // Buffered write - check if flush is needed
        let should_flush = {
            let mut buffer = self.write_buffer.lock();
            buffer.data.extend_from_slice(bytes);
            buffer.pending_count += 1;
            buffer.pending_count >= WRITE_BUFFER_SIZE || buffer.data.len() >= FLUSH_THRESHOLD_BYTES
        };

        // Only one thread flushes at a time; others skip
        if should_flush {
            // Try to acquire flush lock (non-blocking)
            if self.flushing.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                let result = {
                    let mut buffer = self.write_buffer.lock();
                    self.flush_buffer_internal(&mut buffer)
                };
                self.flushing.store(0, Ordering::SeqCst);
                result?;
            }
            // If another thread is flushing, we just continue - data is in buffer
        }

        Ok(id)
    }

    fn get(&self, id: u64) -> Result<Vec<f32>> {
        let id_usize = id as usize;
        let count = self.count.load(Ordering::SeqCst);
        if id_usize >= count {
            return Err(anyhow::anyhow!("Vector ID out of bounds"));
        }

        let vec_size = self.dim * size_of::<f32>();
        let needed_len = (id_usize + 1) * vec_size;

        // First check if data is in the write buffer (not yet flushed)
        {
            let buffer = self.write_buffer.lock();
            let flushed_count = count - buffer.pending_count;
            if id_usize >= flushed_count {
                // Data is in the buffer
                let buffer_idx = id_usize - flushed_count;
                let start = buffer_idx * vec_size;
                let end = start + vec_size;
                if end <= buffer.data.len() {
                    let bytes = &buffer.data[start..end];
                    let mut vector = vec![0.0f32; self.dim];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            vector.as_mut_ptr() as *mut u8,
                            bytes.len(),
                        );
                    }
                    return Ok(vector);
                }
            }
        }

        // Fast path: check if mmap is large enough using atomic
        let mmap_len = self.mmap_len.load(Ordering::Acquire);
        if mmap_len >= needed_len {
            let mmap_guard = self.mmap.read();
            if let Some(ref mmap) = *mmap_guard {
                let start = id_usize * vec_size;
                let end = start + vec_size;
                let bytes = &mmap[start..end];

                let mut vector = vec![0.0f32; self.dim];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        vector.as_mut_ptr() as *mut u8,
                        bytes.len(),
                    );
                }
                return Ok(vector);
            }
        }

        // Mmap is stale, refresh it if needed
        self.refresh_mmap_if_needed(needed_len)?;

        let mmap_guard = self.mmap.read();
        if let Some(ref mmap) = *mmap_guard {
            let start = id_usize * vec_size;
            let end = start + vec_size;
            if end <= mmap.len() {
                let bytes = &mmap[start..end];
                let mut vector = vec![0.0f32; self.dim];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        vector.as_mut_ptr() as *mut u8,
                        bytes.len(),
                    );
                }
                return Ok(vector);
            }
        }

        Err(anyhow::anyhow!("Storage is empty or not mapped"))
    }

    fn len(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }

    fn flush(&self) -> Result<()> {
        let mut buffer = self.write_buffer.lock();
        self.flush_buffer_internal(&mut buffer)
    }

    /// Optimized batch insert - writes all vectors to buffer, single flush.
    /// ~10x faster than individual inserts for large batches.
    fn insert_batch(&self, vectors: &[Vec<f32>]) -> Result<u64> {
        if vectors.is_empty() {
            return Ok(self.len() as u64);
        }

        // Validate all dimensions first
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.dim {
                return Err(anyhow::anyhow!(
                    "Vector {} dimension mismatch: expected {}, got {}",
                    i, self.dim, v.len()
                ));
            }
        }

        let vec_size = self.dim * size_of::<f32>();
        let total_bytes = vectors.len() * vec_size;

        // Reserve starting ID atomically
        let start_id = self.count.fetch_add(vectors.len(), Ordering::SeqCst) as u64;

        // Collect all bytes into buffer
        let mut all_bytes = Vec::with_capacity(total_bytes);
        for v in vectors {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * size_of::<f32>(),
                )
            };
            all_bytes.extend_from_slice(bytes);
        }

        // Add to write buffer
        {
            let mut buffer = self.write_buffer.lock();
            buffer.data.extend_from_slice(&all_bytes);
            buffer.pending_count += vectors.len();
        }

        // Force flush after batch
        if self.flushing.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            let result = {
                let mut buffer = self.write_buffer.lock();
                self.flush_buffer_internal(&mut buffer)
            };
            self.flushing.store(0, Ordering::SeqCst);
            result?;
        }

        Ok(start_id)
    }
}

const METADATA_CACHE_SIZE: usize = 10_000;

pub struct SledMetadataStore {
    db: Db,
    cache: Mutex<LruCache<String, Option<String>>>,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
}

impl SledMetadataStore {
    pub fn new(path: &str) -> Result<Self> {
        let db = sled::open(path)?;
        let cache = LruCache::new(NonZeroUsize::new(METADATA_CACHE_SIZE).unwrap());
        Ok(Self {
            db,
            cache: Mutex::new(cache),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        })
    }

    pub fn cache_stats(&self) -> (u64, u64) {
        (
            self.cache_hits.load(Ordering::Relaxed),
            self.cache_misses.load(Ordering::Relaxed),
        )
    }
}

impl MetadataStore for SledMetadataStore {
    fn insert(&self, id: u64, key: String, value: String) -> Result<()> {
        let k = format!("{}:{}", id, key);

        // Update cache
        {
            let mut cache = self.cache.lock();
            cache.put(k.clone(), Some(value.clone()));
        }

        // Write to DB
        self.db.insert(k.as_bytes(), value.as_bytes())?;
        Ok(())
    }

    fn get(&self, id: u64, key: &str) -> Result<Option<String>> {
        let k = format!("{}:{}", id, key);

        // Check cache first
        {
            let mut cache = self.cache.lock();
            if let Some(cached) = cache.get(&k) {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(cached.clone());
            }
        }

        // Cache miss - read from DB
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        let result = if let Some(v) = self.db.get(k.as_bytes())? {
            Some(String::from_utf8(v.to_vec())?)
        } else {
            None
        };

        // Update cache
        {
            let mut cache = self.cache.lock();
            cache.put(k, result.clone());
        }

        Ok(result)
    }

    /// Optimized batch insert using sled Batch for atomic writes.
    fn insert_batch(&self, entries: &[MetadataEntry]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        // Use sled batch for atomic writes
        let mut batch = sled::Batch::default();

        // Update cache and prepare batch
        {
            let mut cache = self.cache.lock();
            for entry in entries {
                let k = format!("{}:{}", entry.id, entry.key);
                cache.put(k.clone(), Some(entry.value.clone()));
                batch.insert(k.as_bytes(), entry.value.as_bytes());
            }
        }

        // Atomic batch write
        self.db.apply_batch(batch)?;
        Ok(())
    }

    fn get_all(&self, id: u64) -> std::collections::HashMap<String, String> {
        let prefix = format!("{}:", id);
        let mut result = std::collections::HashMap::new();

        // Scan all keys with the given ID prefix
        for item in self.db.scan_prefix(prefix.as_bytes()) {
            if let Ok((key, value)) = item {
                if let (Ok(k), Ok(v)) = (
                    String::from_utf8(key.to_vec()),
                    String::from_utf8(value.to_vec()),
                ) {
                    // Extract key name (remove "id:" prefix)
                    if let Some(key_name) = k.strip_prefix(&prefix) {
                        result.insert(key_name.to_string(), v);
                    }
                }
            }
        }

        result
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

// ============================================================================
// Indexed Metadata Store (with PayloadIndex)
// ============================================================================

/// Metadata store with inverted index for efficient filtering.
/// Combines SledMetadataStore with PayloadIndex.
/// Supports both string fields (exact match) and numeric fields (range queries).
pub struct IndexedSledMetadataStore {
    /// Underlying storage
    inner: SledMetadataStore,
    /// Inverted index for fast filtering
    index: PayloadIndex,
    /// String fields to index
    indexed_fields: Vec<String>,
    /// Numeric fields to index
    numeric_fields: Vec<String>,
}

impl IndexedSledMetadataStore {
    /// Create new indexed store with specified fields to index.
    /// Functional: uses iterator to configure fields.
    pub fn new<I, S>(path: &str, indexed_fields: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let inner = SledMetadataStore::new(path)?;
        let field_names: Vec<String> = indexed_fields
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();

        let index = PayloadIndex::with_fields(field_names.iter().map(|s| s.as_str()));

        // Rebuild index from existing data
        let rebuild_data: Vec<(u64, String, String)> = inner
            .db
            .iter()
            .filter_map(|res| res.ok())
            .filter_map(|(key, value)| {
                let key_str = String::from_utf8(key.to_vec()).ok()?;
                let value_str = String::from_utf8(value.to_vec()).ok()?;
                let mut parts = key_str.splitn(2, ':');
                let id: u64 = parts.next()?.parse().ok()?;
                let field = parts.next()?.to_string();
                Some((id, field, value_str))
            })
            .collect();

        rebuild_data
            .iter()
            .filter(|(_, field, _)| field_names.contains(field))
            .for_each(|(id, field, value)| {
                index.insert(*id, field, value);
            });

        Ok(Self {
            inner,
            index,
            indexed_fields: field_names,
            numeric_fields: Vec::new(),
        })
    }

    /// Create new indexed store with both string and numeric fields.
    pub fn with_numeric_fields<I1, I2, S>(
        path: &str,
        indexed_fields: I1,
        numeric_fields: I2,
    ) -> Result<Self>
    where
        I1: IntoIterator<Item = S>,
        I2: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let inner = SledMetadataStore::new(path)?;
        let field_names: Vec<String> = indexed_fields
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();
        let numeric_names: Vec<String> = numeric_fields
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();

        let index = PayloadIndex::with_fields_and_numeric(
            field_names.iter().map(|s| s.as_str()),
            numeric_names.iter().map(|s| s.as_str()),
        );

        // Rebuild index from existing data
        // Functional: scan DB and extract entries via iterator chain
        let rebuild_data: Vec<(u64, String, String)> = inner
            .db
            .iter()
            .filter_map(|res| res.ok())
            .filter_map(|(key, value)| {
                let key_str = String::from_utf8(key.to_vec()).ok()?;
                let value_str = String::from_utf8(value.to_vec()).ok()?;

                // Parse "id:field" format
                let mut parts = key_str.splitn(2, ':');
                let id: u64 = parts.next()?.parse().ok()?;
                let field = parts.next()?.to_string();

                Some((id, field, value_str))
            })
            .collect();

        // Build string index from collected data
        rebuild_data
            .iter()
            .filter(|(_, field, _)| field_names.contains(field))
            .for_each(|(id, field, value)| {
                index.insert(*id, field, value);
            });

        // Build numeric index from collected data
        // Attempts to parse value as f64 for numeric fields
        rebuild_data
            .iter()
            .filter(|(_, field, _)| numeric_names.contains(field))
            .for_each(|(id, field, value)| {
                if let Ok(num) = value.parse::<f64>() {
                    index.insert_numeric_f64(*id, field, num);
                }
            });

        Ok(Self {
            inner,
            index,
            indexed_fields: field_names,
            numeric_fields: numeric_names,
        })
    }

    /// Add a new string field to index (for future inserts).
    pub fn add_indexed_field(&mut self, field: &str) {
        if !self.indexed_fields.contains(&field.to_string()) {
            self.indexed_fields.push(field.to_string());
            self.index.add_field(field);
        }
    }

    /// Add a new numeric field to index (for future inserts).
    pub fn add_numeric_field(&mut self, field: &str) {
        if !self.numeric_fields.contains(&field.to_string()) {
            self.numeric_fields.push(field.to_string());
            self.index.add_numeric_field(field);
        }
    }

    /// Get cache statistics from inner store.
    pub fn cache_stats(&self) -> (u64, u64) {
        self.inner.cache_stats()
    }

    /// Get index statistics.
    pub fn index_stats(&self) -> PayloadIndexFullStats {
        self.index.stats()
    }

    /// Check if a string field is indexed.
    pub fn is_indexed(&self, field: &str) -> bool {
        self.index.is_indexed(field)
    }

    /// Check if a numeric field is indexed.
    pub fn is_numeric_indexed(&self, field: &str) -> bool {
        self.index.is_numeric_indexed(field)
    }
}

impl MetadataStore for IndexedSledMetadataStore {
    /// Insert with automatic index update.
    /// Functional: conditionally updates index based on field membership.
    /// Handles both string and numeric fields.
    fn insert(&self, id: u64, key: String, value: String) -> Result<()> {
        // Update string index if field is indexed
        if self.indexed_fields.contains(&key) {
            self.index.insert(id, &key, &value);
        }

        // Update numeric index if field is indexed
        // Attempts to parse value as f64
        if self.numeric_fields.contains(&key) {
            if let Ok(num) = value.parse::<f64>() {
                self.index.insert_numeric_f64(id, &key, num);
            }
        }

        // Delegate to inner store
        self.inner.insert(id, key, value)
    }

    fn get(&self, id: u64, key: &str) -> Result<Option<String>> {
        self.inner.get(id, key)
    }

    /// Pre-filter using payload index.
    /// Returns matching IDs as RoaringBitmap for efficient HNSW filtering.
    fn try_filter_and(&self, conditions: &[(&str, &str)]) -> Option<RoaringBitmap> {
        if conditions.is_empty() {
            return None;
        }
        self.index.filter_and(conditions.iter().copied())
    }

    /// Optimized batch insert with index updates.
    /// Updates payload index in batch, then delegates to inner store.
    fn insert_batch(&self, entries: &[MetadataEntry]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        // Update indexes in batch
        // Functional: partition entries by field type
        for entry in entries {
            // String index
            if self.indexed_fields.contains(&entry.key) {
                self.index.insert(entry.id, &entry.key, &entry.value);
            }
            // Numeric index
            if self.numeric_fields.contains(&entry.key) {
                if let Ok(num) = entry.value.parse::<f64>() {
                    self.index.insert_numeric_f64(entry.id, &entry.key, num);
                }
            }
        }

        // Delegate to inner store for atomic batch write
        self.inner.insert_batch(entries)
    }

    fn get_all(&self, id: u64) -> std::collections::HashMap<String, String> {
        self.inner.get_all(id)
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

impl IndexedMetadata for IndexedSledMetadataStore {
    /// Filter using PayloadIndex.
    /// Functional: delegates to index's filter.
    fn filter(&self, query: &FilterQuery) -> Option<RoaringBitmap> {
        self.index.filter(query)
    }

    /// Filter by simple AND conditions.
    /// Functional: transforms slice to iterator of tuples.
    fn filter_and(&self, conditions: &[(&str, &str)]) -> Option<RoaringBitmap> {
        self.index.filter_and(conditions.iter().copied())
    }

    fn index_stats(&self) -> PayloadIndexFullStats {
        self.index.stats()
    }
}

// ============================================================================
// Quantized Vector Store
// ============================================================================

const QUANTIZED_WRITE_BUFFER_SIZE: usize = 512;

/// Memory-efficient vector store using scalar quantization.
/// Stores uint8 quantized vectors with optional original vectors for reranking.
pub struct QuantizedMemmapVectorStore {
    /// Quantized vectors file
    quant_file: Mutex<File>,
    quant_mmap: RwLock<Option<Mmap>>,
    quant_mmap_len: AtomicUsize,

    /// Original vectors file (optional, for reranking)
    orig_file: Option<Mutex<File>>,
    orig_mmap: RwLock<Option<Mmap>>,
    orig_mmap_len: AtomicUsize,

    /// Quantizer
    quantizer: ScalarQuantizer,
    quantizer_path: String,

    dim: usize,
    count: AtomicUsize,
    keep_originals: bool,

    /// Write buffers
    quant_buffer: Mutex<Vec<u8>>,
    orig_buffer: Mutex<Vec<u8>>,
    pending_count: AtomicUsize,
    flushing: AtomicUsize,
}

impl QuantizedMemmapVectorStore {
    pub fn new(base_path: &str, dim: usize, keep_originals: bool) -> Result<Self> {
        let quant_path = format!("{}.quant", base_path);
        let orig_path = format!("{}.orig", base_path);
        let quantizer_path = format!("{}.quantizer", base_path);

        // Open quantized vectors file
        let quant_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&quant_path)?;

        let quant_meta = quant_file.metadata()?;
        let quant_len = quant_meta.len() as usize;
        let count = if quant_len > 0 { quant_len / dim } else { 0 };

        let quant_mmap = if quant_len > 0 {
            unsafe { Some(Mmap::map(&quant_file)?) }
        } else {
            None
        };

        // Open original vectors file if keeping originals
        let (orig_file, orig_mmap, orig_mmap_len) = if keep_originals {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&orig_path)?;

            let meta = file.metadata()?;
            let len = meta.len() as usize;
            let mmap = if len > 0 {
                unsafe { Some(Mmap::map(&file)?) }
            } else {
                None
            };

            (Some(Mutex::new(file)), RwLock::new(mmap), AtomicUsize::new(len))
        } else {
            (None, RwLock::new(None), AtomicUsize::new(0))
        };

        // Load or create quantizer
        let quantizer = if Path::new(&quantizer_path).exists() {
            let mut file = std::fs::File::open(&quantizer_path)?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;
            ScalarQuantizer::from_bytes(&bytes)?
        } else {
            ScalarQuantizer::new(dim)
        };

        Ok(Self {
            quant_file: Mutex::new(quant_file),
            quant_mmap: RwLock::new(quant_mmap),
            quant_mmap_len: AtomicUsize::new(quant_len),
            orig_file,
            orig_mmap,
            orig_mmap_len,
            quantizer,
            quantizer_path,
            dim,
            count: AtomicUsize::new(count),
            keep_originals,
            quant_buffer: Mutex::new(Vec::with_capacity(QUANTIZED_WRITE_BUFFER_SIZE * dim)),
            orig_buffer: Mutex::new(Vec::with_capacity(QUANTIZED_WRITE_BUFFER_SIZE * dim * 4)),
            pending_count: AtomicUsize::new(0),
            flushing: AtomicUsize::new(0),
        })
    }

    /// Get the quantizer for external use (e.g., distance computation).
    pub fn quantizer(&self) -> &ScalarQuantizer {
        &self.quantizer
    }

    /// Get quantizer statistics.
    pub fn quantizer_stats(&self) -> QuantizerStats {
        self.quantizer.stats()
    }

    /// Get quantized vector by ID.
    pub fn get_quantized(&self, id: u64) -> Result<Vec<u8>> {
        let id_usize = id as usize;
        let count = self.count.load(Ordering::SeqCst);
        if id_usize >= count {
            return Err(anyhow::anyhow!("Vector ID out of bounds"));
        }

        let needed_len = (id_usize + 1) * self.dim;

        // Check write buffer first
        {
            let buffer = self.quant_buffer.lock();
            let pending = self.pending_count.load(Ordering::SeqCst);
            let flushed_count = count - pending;
            if id_usize >= flushed_count {
                let buffer_idx = id_usize - flushed_count;
                let start = buffer_idx * self.dim;
                let end = start + self.dim;
                if end <= buffer.len() {
                    return Ok(buffer[start..end].to_vec());
                }
            }
        }

        // Read from mmap
        let mmap_len = self.quant_mmap_len.load(Ordering::Acquire);
        if mmap_len >= needed_len {
            let mmap_guard = self.quant_mmap.read();
            if let Some(ref mmap) = *mmap_guard {
                let start = id_usize * self.dim;
                let end = start + self.dim;
                return Ok(mmap[start..end].to_vec());
            }
        }

        // Refresh mmap
        self.refresh_quant_mmap(needed_len)?;

        let mmap_guard = self.quant_mmap.read();
        if let Some(ref mmap) = *mmap_guard {
            let start = id_usize * self.dim;
            let end = start + self.dim;
            if end <= mmap.len() {
                return Ok(mmap[start..end].to_vec());
            }
        }

        Err(anyhow::anyhow!("Storage is empty or not mapped"))
    }

    /// Get original (unquantized) vector by ID, if available.
    pub fn get_original(&self, id: u64) -> Result<Vec<f32>> {
        if !self.keep_originals {
            // Decode from quantized
            let quantized = self.get_quantized(id)?;
            return self.quantizer.decode(&quantized);
        }

        let id_usize = id as usize;
        let count = self.count.load(Ordering::SeqCst);
        if id_usize >= count {
            return Err(anyhow::anyhow!("Vector ID out of bounds"));
        }

        let vec_size = self.dim * size_of::<f32>();
        let needed_len = (id_usize + 1) * vec_size;

        // Check write buffer first
        {
            let buffer = self.orig_buffer.lock();
            let pending = self.pending_count.load(Ordering::SeqCst);
            let flushed_count = count - pending;
            if id_usize >= flushed_count {
                let buffer_idx = id_usize - flushed_count;
                let start = buffer_idx * vec_size;
                let end = start + vec_size;
                if end <= buffer.len() {
                    let bytes = &buffer[start..end];
                    let mut vector = vec![0.0f32; self.dim];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            vector.as_mut_ptr() as *mut u8,
                            bytes.len(),
                        );
                    }
                    return Ok(vector);
                }
            }
        }

        // Read from mmap
        let mmap_len = self.orig_mmap_len.load(Ordering::Acquire);
        if mmap_len >= needed_len {
            let mmap_guard = self.orig_mmap.read();
            if let Some(ref mmap) = *mmap_guard {
                let start = id_usize * vec_size;
                let end = start + vec_size;
                let bytes = &mmap[start..end];
                let mut vector = vec![0.0f32; self.dim];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        vector.as_mut_ptr() as *mut u8,
                        bytes.len(),
                    );
                }
                return Ok(vector);
            }
        }

        // Fallback to decoded quantized
        let quantized = self.get_quantized(id)?;
        self.quantizer.decode(&quantized)
    }

    fn refresh_quant_mmap(&self, needed_len: usize) -> Result<()> {
        let current_len = self.quant_mmap_len.load(Ordering::Acquire);
        if current_len >= needed_len {
            return Ok(());
        }

        let file = self.quant_file.lock();
        let file_len = file.metadata()?.len() as usize;
        if file_len == 0 {
            return Ok(());
        }

        let mmap = unsafe { Mmap::map(&*file)? };
        let new_len = mmap.len();
        *self.quant_mmap.write() = Some(mmap);
        self.quant_mmap_len.store(new_len, Ordering::Release);
        Ok(())
    }

    fn flush_buffers(&self) -> Result<()> {
        // Flush quantized buffer
        {
            let mut buffer = self.quant_buffer.lock();
            if !buffer.is_empty() {
                let mut file = self.quant_file.lock();
                file.seek(SeekFrom::End(0))?;
                file.write_all(&buffer)?;
                file.flush()?;
                buffer.clear();
            }
        }

        // Flush original buffer if keeping originals
        if self.keep_originals {
            if let Some(ref orig_file) = self.orig_file {
                let mut buffer = self.orig_buffer.lock();
                if !buffer.is_empty() {
                    let mut file = orig_file.lock();
                    file.seek(SeekFrom::End(0))?;
                    file.write_all(&buffer)?;
                    file.flush()?;
                    buffer.clear();
                }
            }
        }

        self.pending_count.store(0, Ordering::SeqCst);

        // Refresh mmaps
        let quant_len = self.quant_mmap_len.load(Ordering::Acquire);
        let _ = self.refresh_quant_mmap(quant_len + 1);

        Ok(())
    }

    /// Save quantizer parameters to disk.
    pub fn save_quantizer(&self) -> Result<()> {
        let bytes = self.quantizer.to_bytes();
        std::fs::write(&self.quantizer_path, bytes)?;
        Ok(())
    }

    /// Get memory usage statistics.
    pub fn memory_stats(&self) -> QuantizedMemoryStats {
        let count = self.count.load(Ordering::SeqCst);
        let quant_size = count * self.dim;
        let orig_size = if self.keep_originals {
            count * self.dim * 4
        } else {
            0
        };

        QuantizedMemoryStats {
            vector_count: count,
            quantized_bytes: quant_size,
            original_bytes: orig_size,
            total_bytes: quant_size + orig_size,
            unquantized_would_be: count * self.dim * 4,
            savings_bytes: (count * self.dim * 4).saturating_sub(quant_size + orig_size),
            compression_ratio: if quant_size + orig_size > 0 {
                (count * self.dim * 4) as f32 / (quant_size + orig_size) as f32
            } else {
                0.0
            },
        }
    }
}

impl VectorStore for QuantizedMemmapVectorStore {
    fn insert(&self, vector: &[f32]) -> Result<u64> {
        if vector.len() != self.dim {
            return Err(anyhow::anyhow!("Vector dimension mismatch"));
        }

        // Train quantizer with this vector
        self.quantizer.train_single(vector);

        // Encode to quantized
        let quantized = self.quantizer.encode(vector)?;

        // Reserve ID
        let id = self.count.fetch_add(1, Ordering::SeqCst) as u64;

        // Add to buffers
        {
            let mut quant_buffer = self.quant_buffer.lock();
            quant_buffer.extend_from_slice(&quantized);
        }

        if self.keep_originals {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    vector.as_ptr() as *const u8,
                    vector.len() * size_of::<f32>(),
                )
            };
            let mut orig_buffer = self.orig_buffer.lock();
            orig_buffer.extend_from_slice(bytes);
        }

        let pending = self.pending_count.fetch_add(1, Ordering::SeqCst) + 1;

        // Check if flush needed
        if pending >= QUANTIZED_WRITE_BUFFER_SIZE {
            if self.flushing.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                let result = self.flush_buffers();
                self.flushing.store(0, Ordering::SeqCst);
                result?;
            }
        }

        Ok(id)
    }

    fn get(&self, id: u64) -> Result<Vec<f32>> {
        self.get_original(id)
    }

    fn len(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }

    fn flush(&self) -> Result<()> {
        self.flush_buffers()?;
        self.save_quantizer()?;
        Ok(())
    }

    /// Optimized batch insert for quantized vectors.
    fn insert_batch(&self, vectors: &[Vec<f32>]) -> Result<u64> {
        if vectors.is_empty() {
            return Ok(self.len() as u64);
        }

        // Validate dimensions
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.dim {
                return Err(anyhow::anyhow!(
                    "Vector {} dimension mismatch: expected {}, got {}",
                    i, self.dim, v.len()
                ));
            }
        }

        // Train quantizer with all vectors first
        for v in vectors {
            self.quantizer.train_single(v);
        }

        // Reserve IDs atomically
        let start_id = self.count.fetch_add(vectors.len(), Ordering::SeqCst) as u64;

        // Encode all vectors and collect bytes
        let vec_size = self.dim * size_of::<f32>();
        let mut all_quant = Vec::with_capacity(vectors.len() * self.dim);
        let mut all_orig = if self.keep_originals {
            Vec::with_capacity(vectors.len() * vec_size)
        } else {
            Vec::new()
        };

        for v in vectors {
            let quantized = self.quantizer.encode(v)?;
            all_quant.extend_from_slice(&quantized);

            if self.keep_originals {
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        v.as_ptr() as *const u8,
                        v.len() * size_of::<f32>(),
                    )
                };
                all_orig.extend_from_slice(bytes);
            }
        }

        // Add to buffers
        {
            let mut quant_buffer = self.quant_buffer.lock();
            quant_buffer.extend_from_slice(&all_quant);
        }

        if self.keep_originals {
            let mut orig_buffer = self.orig_buffer.lock();
            orig_buffer.extend_from_slice(&all_orig);
        }

        self.pending_count.fetch_add(vectors.len(), Ordering::SeqCst);

        // Force flush after batch
        if self.flushing.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            let result = self.flush_buffers();
            self.flushing.store(0, Ordering::SeqCst);
            result?;
        }

        Ok(start_id)
    }
}

/// Memory statistics for quantized storage.
#[derive(Debug, Clone)]
pub struct QuantizedMemoryStats {
    pub vector_count: usize,
    pub quantized_bytes: usize,
    pub original_bytes: usize,
    pub total_bytes: usize,
    pub unquantized_would_be: usize,
    pub savings_bytes: usize,
    pub compression_ratio: f32,
}
