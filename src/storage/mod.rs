use std::fs::{File, OpenOptions};
use std::io::{Write, Seek, SeekFrom};
use memmap2::Mmap;
use parking_lot::{RwLock, Mutex};
use sled::Db;
use anyhow::Result;
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Trait for storing vectors
pub trait VectorStore: Send + Sync {
    fn insert(&self, vector: &[f32]) -> Result<u64>;
    fn get(&self, id: u64) -> Result<Vec<f32>>;
    fn len(&self) -> usize;
    fn flush(&self) -> Result<()>;
}

/// Trait for storing metadata
pub trait MetadataStore: Send + Sync {
    fn insert(&self, id: u64, key: String, value: String) -> Result<()>;
    fn get(&self, id: u64, key: &str) -> Result<Option<String>>;
}

const WRITE_BUFFER_SIZE: usize = 256; // Flush after 256 vectors
const FLUSH_THRESHOLD_BYTES: usize = 256 * 1024; // Or 256KB

pub struct MemmapVectorStore {
    file: Mutex<File>,              // Mutex for write-heavy workload
    mmap: RwLock<Option<Mmap>>,     // parking_lot RwLock
    dim: usize,
    count: AtomicUsize,             // Atomic for lock-free reads
    write_buffer: Mutex<WriteBuffer>,
    flushing: AtomicUsize,          // Flag to track if flush is in progress
    path: String,
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
        let len = meta.len();
        let vec_size = dim * size_of::<f32>();

        let count = if len > 0 {
            (len as usize) / vec_size
        } else {
            0
        };

        let mmap = if len > 0 {
            unsafe { Some(Mmap::map(&file)?) }
        } else {
            None
        };

        Ok(Self {
            file: Mutex::new(file),
            mmap: RwLock::new(mmap),
            dim,
            count: AtomicUsize::new(count),
            write_buffer: Mutex::new(WriteBuffer {
                data: Vec::with_capacity(FLUSH_THRESHOLD_BYTES),
                pending_count: 0,
            }),
            flushing: AtomicUsize::new(0),
            path: path.to_string(),
        })
    }

    fn refresh_mmap(&self) -> Result<()> {
        let file = self.file.lock();
        if file.metadata()?.len() == 0 {
            return Ok(());
        }
        let mmap = unsafe { Mmap::map(&*file)? };
        *self.mmap.write() = Some(mmap);
        Ok(())
    }

    fn flush_buffer_internal(&self, buffer: &mut WriteBuffer) -> Result<()> {
        if buffer.data.is_empty() {
            return Ok(());
        }

        let mut file = self.file.lock();
        file.seek(SeekFrom::End(0))?;
        file.write_all(&buffer.data)?;
        file.flush()?;

        buffer.data.clear();
        buffer.pending_count = 0;
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

        // Try optimistic read from mmap
        {
            let mmap_guard = self.mmap.read();
            if let Some(ref mmap) = *mmap_guard {
                if mmap.len() >= needed_len {
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
        }

        // Mmap is stale, refresh it
        self.refresh_mmap()?;

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
}

pub struct SledMetadataStore {
    db: Db,
}

impl SledMetadataStore {
    pub fn new(path: &str) -> Result<Self> {
        let db = sled::open(path)?;
        Ok(Self { db })
    }
}

impl MetadataStore for SledMetadataStore {
    fn insert(&self, id: u64, key: String, value: String) -> Result<()> {
        let k = format!("{}:{}", id, key);
        self.db.insert(k.as_bytes(), value.as_bytes())?;
        Ok(())
    }

    fn get(&self, id: u64, key: &str) -> Result<Option<String>> {
        let k = format!("{}:{}", id, key);
        if let Some(v) = self.db.get(k.as_bytes())? {
            Ok(Some(String::from_utf8(v.to_vec())?))
        } else {
            Ok(None)
        }
    }
}
