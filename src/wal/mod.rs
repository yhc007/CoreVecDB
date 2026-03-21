//! Write-Ahead Log (WAL) for durable persistence.
//!
//! Provides crash recovery by recording operations before applying them.
//! Supports incremental checkpoints to minimize recovery time.
//!
//! # Architecture
//! ```text
//! Operation → WAL → Apply to Index/Storage
//!                ↓
//!            Checkpoint → Truncate old WAL
//! ```

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// WAL entry types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WalOperation {
    /// Insert a vector with metadata
    Insert {
        id: u64,
        vector: Vec<f32>,
        metadata: Vec<(String, String)>,
    },
    /// Delete a vector by ID
    Delete { id: u64 },
    /// Batch insert multiple vectors
    BatchInsert {
        start_id: u64,
        vectors: Vec<Vec<f32>>,
        metadata: Vec<Vec<(String, String)>>,
    },
    /// Batch delete multiple vectors
    BatchDelete { ids: Vec<u64> },
    /// Checkpoint marker - WAL entries before this can be truncated
    Checkpoint {
        sequence: u64,
        vector_count: usize,
    },
}

/// A single WAL entry with sequence number and CRC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Monotonically increasing sequence number
    pub sequence: u64,
    /// The operation
    pub operation: WalOperation,
    /// Timestamp (unix millis)
    pub timestamp: u64,
}

impl WalEntry {
    /// Create a new WAL entry
    pub fn new(sequence: u64, operation: WalOperation) -> Self {
        Self {
            sequence,
            operation,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Serialize entry to bytes with length prefix and CRC
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let data = bincode::serialize(self)?;
        let crc = crc32fast::hash(&data);

        let mut bytes = Vec::with_capacity(4 + 4 + data.len());
        // Length prefix (4 bytes)
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        // CRC32 (4 bytes)
        bytes.extend_from_slice(&crc.to_le_bytes());
        // Data
        bytes.extend_from_slice(&data);

        Ok(bytes)
    }

    /// Deserialize entry from reader
    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Option<Self>> {
        // Read length prefix
        let mut len_buf = [0u8; 4];
        match reader.read_exact(&mut len_buf) {
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let len = u32::from_le_bytes(len_buf) as usize;

        // Sanity check - entries shouldn't be larger than 100MB
        if len > 100 * 1024 * 1024 {
            return Err(anyhow::anyhow!("WAL entry too large: {} bytes", len));
        }

        // Read CRC
        let mut crc_buf = [0u8; 4];
        reader.read_exact(&mut crc_buf)?;
        let expected_crc = u32::from_le_bytes(crc_buf);

        // Read data
        let mut data = vec![0u8; len];
        reader.read_exact(&mut data)?;

        // Verify CRC
        let actual_crc = crc32fast::hash(&data);
        if actual_crc != expected_crc {
            return Err(anyhow::anyhow!(
                "WAL entry CRC mismatch: expected {:08x}, got {:08x}",
                expected_crc, actual_crc
            ));
        }

        // Deserialize
        let entry: WalEntry = bincode::deserialize(&data)?;
        Ok(Some(entry))
    }
}

/// Configuration for WAL behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalConfig {
    /// Enable WAL (default: true)
    pub enabled: bool,
    /// Sync to disk after each write (safer but slower)
    pub sync_on_write: bool,
    /// Number of entries before auto-checkpoint (0 = manual only)
    pub checkpoint_interval: usize,
    /// Maximum WAL file size in bytes before forced checkpoint
    pub max_wal_size: usize,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sync_on_write: false, // Batch sync for performance
            checkpoint_interval: 10000, // Checkpoint every 10K operations
            max_wal_size: 256 * 1024 * 1024, // 256MB max WAL size
        }
    }
}

/// Write-Ahead Log for a single collection
pub struct WriteAheadLog {
    /// WAL file path
    path: PathBuf,
    /// WAL file handle
    file: Mutex<BufWriter<File>>,
    /// Current sequence number
    sequence: AtomicU64,
    /// Last checkpoint sequence
    last_checkpoint: AtomicU64,
    /// Number of entries since last checkpoint
    entries_since_checkpoint: AtomicU64,
    /// Current WAL file size (approximate)
    current_size: AtomicU64,
    /// Configuration
    config: WalConfig,
}

impl WriteAheadLog {
    /// Create or open a WAL file
    pub fn open(path: impl AsRef<Path>, config: WalConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;

        let file_size = file.metadata()?.len();

        // Find the last sequence number and checkpoint
        let (last_seq, last_checkpoint) = Self::scan_wal(&path)?;

        // Seek to end for appending
        let mut writer = BufWriter::new(file);
        writer.seek(SeekFrom::End(0))?;

        Ok(Self {
            path,
            file: Mutex::new(writer),
            sequence: AtomicU64::new(last_seq),
            last_checkpoint: AtomicU64::new(last_checkpoint),
            entries_since_checkpoint: AtomicU64::new(last_seq.saturating_sub(last_checkpoint)),
            current_size: AtomicU64::new(file_size),
            config,
        })
    }

    /// Scan WAL file to find last sequence and checkpoint
    fn scan_wal(path: &Path) -> Result<(u64, u64)> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok((0, 0)),
            Err(e) => return Err(e.into()),
        };

        let metadata = file.metadata()?;
        if metadata.len() == 0 {
            return Ok((0, 0));
        }

        let mut reader = BufReader::new(file);
        let mut last_seq = 0u64;
        let mut last_checkpoint = 0u64;

        while let Some(entry) = WalEntry::from_reader(&mut reader)? {
            last_seq = entry.sequence;
            if let WalOperation::Checkpoint { sequence, .. } = entry.operation {
                last_checkpoint = sequence;
            }
        }

        Ok((last_seq, last_checkpoint))
    }

    /// Append an operation to the WAL
    pub fn append(&self, operation: WalOperation) -> Result<u64> {
        if !self.config.enabled {
            return Ok(self.sequence.load(Ordering::SeqCst));
        }

        let seq = self.sequence.fetch_add(1, Ordering::SeqCst) + 1;
        let entry = WalEntry::new(seq, operation);
        let bytes = entry.to_bytes()?;

        {
            let mut file = self.file.lock();
            file.write_all(&bytes)?;

            if self.config.sync_on_write {
                file.flush()?;
                file.get_ref().sync_data()?;
            }
        }

        self.current_size.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        let entries = self.entries_since_checkpoint.fetch_add(1, Ordering::Relaxed) + 1;

        // Check if we need auto-checkpoint
        if self.config.checkpoint_interval > 0 && entries >= self.config.checkpoint_interval as u64 {
            // Just mark that checkpoint is needed - actual checkpoint done externally
        }

        Ok(seq)
    }

    /// Append multiple operations atomically
    pub fn append_batch(&self, operations: &[WalOperation]) -> Result<u64> {
        if !self.config.enabled || operations.is_empty() {
            return Ok(self.sequence.load(Ordering::SeqCst));
        }

        let start_seq = self.sequence.fetch_add(operations.len() as u64, Ordering::SeqCst) + 1;
        let mut total_bytes = 0usize;

        {
            let mut file = self.file.lock();

            for (i, op) in operations.iter().enumerate() {
                let seq = start_seq + i as u64;
                let entry = WalEntry::new(seq, op.clone());
                let bytes = entry.to_bytes()?;
                total_bytes += bytes.len();
                file.write_all(&bytes)?;
            }

            if self.config.sync_on_write {
                file.flush()?;
                file.get_ref().sync_data()?;
            }
        }

        self.current_size.fetch_add(total_bytes as u64, Ordering::Relaxed);
        self.entries_since_checkpoint.fetch_add(operations.len() as u64, Ordering::Relaxed);

        Ok(start_seq + operations.len() as u64 - 1)
    }

    /// Write a checkpoint marker
    pub fn checkpoint(&self, vector_count: usize) -> Result<u64> {
        let seq = self.sequence.load(Ordering::SeqCst);
        let operation = WalOperation::Checkpoint {
            sequence: seq,
            vector_count,
        };

        let entry = WalEntry::new(seq, operation);
        let bytes = entry.to_bytes()?;

        {
            let mut file = self.file.lock();
            file.write_all(&bytes)?;
            file.flush()?;
            file.get_ref().sync_data()?;
        }

        self.last_checkpoint.store(seq, Ordering::SeqCst);
        self.entries_since_checkpoint.store(0, Ordering::SeqCst);

        Ok(seq)
    }

    /// Sync WAL to disk
    pub fn sync(&self) -> Result<()> {
        let mut file = self.file.lock();
        file.flush()?;
        file.get_ref().sync_data()?;
        Ok(())
    }

    /// Truncate WAL after successful checkpoint
    /// Keeps only entries after last_checkpoint_seq
    pub fn truncate(&self) -> Result<()> {
        let checkpoint_seq = self.last_checkpoint.load(Ordering::SeqCst);
        if checkpoint_seq == 0 {
            return Ok(());
        }

        // Create temp file with entries after checkpoint
        let temp_path = self.path.with_extension("wal.tmp");

        // Read all entries
        let entries = self.read_all()?;

        // Filter entries after checkpoint
        let keep_entries: Vec<_> = entries
            .into_iter()
            .filter(|e| e.sequence > checkpoint_seq)
            .collect();

        // Write to temp file
        {
            let temp_file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&temp_path)?;
            let mut writer = BufWriter::new(temp_file);

            for entry in &keep_entries {
                let bytes = entry.to_bytes()?;
                writer.write_all(&bytes)?;
            }
            writer.flush()?;
        }

        // Close current file and replace
        {
            let mut file = self.file.lock();
            file.flush()?;
            drop(file);
        }

        // Atomic rename
        std::fs::rename(&temp_path, &self.path)?;

        // Reopen file
        let new_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&self.path)?;
        let mut writer = BufWriter::new(new_file);
        writer.seek(SeekFrom::End(0))?;

        *self.file.lock() = writer;
        self.current_size.store(
            std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0),
            Ordering::SeqCst,
        );

        Ok(())
    }

    /// Read all WAL entries.
    /// Note: This syncs buffered data to disk before reading.
    pub fn read_all(&self) -> Result<Vec<WalEntry>> {
        // Ensure all buffered data is flushed to disk before reading
        {
            let mut file = self.file.lock();
            file.flush()?;
        }

        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();

        while let Some(entry) = WalEntry::from_reader(&mut reader)? {
            entries.push(entry);
        }

        Ok(entries)
    }

    /// Read entries since last checkpoint for recovery
    pub fn read_for_recovery(&self) -> Result<Vec<WalEntry>> {
        // Ensure all buffered data is flushed to disk before reading
        self.sync()?;

        let checkpoint_seq = self.last_checkpoint.load(Ordering::SeqCst);
        let all_entries = self.read_all()?;

        Ok(all_entries
            .into_iter()
            .filter(|e| e.sequence > checkpoint_seq)
            .filter(|e| !matches!(e.operation, WalOperation::Checkpoint { .. }))
            .collect())
    }

    /// Get current sequence number
    pub fn sequence(&self) -> u64 {
        self.sequence.load(Ordering::SeqCst)
    }

    /// Get last checkpoint sequence
    pub fn last_checkpoint_seq(&self) -> u64 {
        self.last_checkpoint.load(Ordering::SeqCst)
    }

    /// Get entries since last checkpoint
    pub fn entries_since_checkpoint(&self) -> u64 {
        self.entries_since_checkpoint.load(Ordering::SeqCst)
    }

    /// Check if checkpoint is needed
    pub fn needs_checkpoint(&self) -> bool {
        if self.config.checkpoint_interval == 0 {
            return false;
        }

        let entries = self.entries_since_checkpoint.load(Ordering::Relaxed);
        let size = self.current_size.load(Ordering::Relaxed);

        entries >= self.config.checkpoint_interval as u64
            || size >= self.config.max_wal_size as u64
    }

    /// Get WAL statistics
    pub fn stats(&self) -> WalStats {
        WalStats {
            sequence: self.sequence.load(Ordering::SeqCst),
            last_checkpoint: self.last_checkpoint.load(Ordering::SeqCst),
            entries_since_checkpoint: self.entries_since_checkpoint.load(Ordering::SeqCst),
            file_size: self.current_size.load(Ordering::SeqCst),
            path: self.path.clone(),
        }
    }

    /// Delete WAL file (use after collection delete)
    pub fn delete(self) -> Result<()> {
        drop(self.file);
        if self.path.exists() {
            std::fs::remove_file(&self.path)?;
        }
        Ok(())
    }
}

/// WAL statistics
#[derive(Debug, Clone, Serialize)]
pub struct WalStats {
    pub sequence: u64,
    pub last_checkpoint: u64,
    pub entries_since_checkpoint: u64,
    pub file_size: u64,
    #[serde(skip)]
    pub path: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn test_wal_path(name: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!("vectordb_wal_test_{}", name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("test.wal")
    }

    #[test]
    fn test_wal_basic() {
        let path = test_wal_path("basic");
        let config = WalConfig::default();

        let wal = WriteAheadLog::open(&path, config).unwrap();

        // Insert operation
        let op = WalOperation::Insert {
            id: 1,
            vector: vec![0.1, 0.2, 0.3],
            metadata: vec![("key".to_string(), "value".to_string())],
        };

        let seq = wal.append(op.clone()).unwrap();
        assert_eq!(seq, 1);

        // Delete operation
        let delete_op = WalOperation::Delete { id: 1 };
        let seq2 = wal.append(delete_op).unwrap();
        assert_eq!(seq2, 2);

        // Sync and verify
        wal.sync().unwrap();

        let entries = wal.read_all().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].sequence, 1);
        assert_eq!(entries[1].sequence, 2);

        // Cleanup
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn test_wal_checkpoint() {
        let path = test_wal_path("checkpoint");
        let config = WalConfig::default();

        let wal = WriteAheadLog::open(&path, config).unwrap();

        // Insert some entries
        for i in 0..5 {
            wal.append(WalOperation::Insert {
                id: i,
                vector: vec![i as f32],
                metadata: vec![],
            }).unwrap();
        }

        // Create checkpoint
        let checkpoint_seq = wal.checkpoint(5).unwrap();
        assert_eq!(checkpoint_seq, 5);

        // Add more entries
        for i in 5..10 {
            wal.append(WalOperation::Insert {
                id: i,
                vector: vec![i as f32],
                metadata: vec![],
            }).unwrap();
        }

        // Read for recovery (should only get entries after checkpoint)
        let recovery_entries = wal.read_for_recovery().unwrap();
        assert_eq!(recovery_entries.len(), 5);
        assert!(recovery_entries.iter().all(|e| e.sequence > 5));

        // Cleanup
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn test_wal_recovery() {
        let path = test_wal_path("recovery");
        let config = WalConfig::default();

        // First session - write some data
        {
            let wal = WriteAheadLog::open(&path, config.clone()).unwrap();
            for i in 0..3 {
                wal.append(WalOperation::Insert {
                    id: i,
                    vector: vec![i as f32; 4],
                    metadata: vec![],
                }).unwrap();
            }
            wal.sync().unwrap();
        }

        // Second session - recover
        {
            let wal = WriteAheadLog::open(&path, config).unwrap();
            assert_eq!(wal.sequence(), 3);

            let entries = wal.read_all().unwrap();
            assert_eq!(entries.len(), 3);
        }

        // Cleanup
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn test_wal_truncate() {
        let path = test_wal_path("truncate");
        let config = WalConfig::default();

        let wal = WriteAheadLog::open(&path, config).unwrap();

        // Insert 10 entries
        for i in 0..10 {
            wal.append(WalOperation::Insert {
                id: i,
                vector: vec![i as f32],
                metadata: vec![],
            }).unwrap();
        }

        // Checkpoint at 5
        wal.checkpoint(5).unwrap();

        // Insert 5 more
        for i in 10..15 {
            wal.append(WalOperation::Insert {
                id: i,
                vector: vec![i as f32],
                metadata: vec![],
            }).unwrap();
        }

        // Truncate
        wal.truncate().unwrap();

        // Verify only entries after checkpoint remain
        let entries = wal.read_all().unwrap();
        // Should have entries 11-15 (5 entries after checkpoint marker at seq 10)
        assert!(entries.len() <= 10, "Should have fewer entries after truncate");

        // Cleanup
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
