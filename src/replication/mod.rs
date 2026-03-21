//! Replication support for VectorDB.
//!
//! Implements Primary-Replica architecture with:
//! - Write-Ahead Log (WAL) for durability and replication
//! - Follower mode for syncing from primary
//! - Snapshot-based initial sync
//!
//! Architecture:
//! - Primary: Accepts writes, maintains WAL, serves reads
//! - Replica: Syncs from primary, serves reads only

use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Replication role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationRole {
    /// Primary node - accepts writes
    Primary,
    /// Replica node - read-only, syncs from primary
    Replica,
    /// Standalone - no replication
    Standalone,
}

impl Default for ReplicationRole {
    fn default() -> Self {
        ReplicationRole::Standalone
    }
}

/// Operation type in WAL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOperation {
    /// Insert a vector
    Insert {
        collection: String,
        id: u64,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    },
    /// Insert batch of vectors
    InsertBatch {
        collection: String,
        start_id: u64,
        vectors: Vec<Vec<f32>>,
        metadata: Vec<HashMap<String, String>>,
    },
    /// Delete a vector
    Delete {
        collection: String,
        id: u64,
    },
    /// Delete batch of vectors
    DeleteBatch {
        collection: String,
        ids: Vec<u64>,
    },
    /// Update vector
    Update {
        collection: String,
        old_id: u64,
        new_id: u64,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    },
    /// Update metadata only
    UpdateMetadata {
        collection: String,
        id: u64,
        metadata: HashMap<String, String>,
    },
    /// Create collection
    CreateCollection {
        name: String,
        dim: usize,
        config_json: String,
    },
    /// Delete collection
    DeleteCollection {
        name: String,
    },
    /// Compact collection
    Compact {
        collection: String,
    },
}

/// WAL entry with sequence number and timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Monotonically increasing sequence number
    pub seq: u64,
    /// Timestamp of the operation
    pub timestamp: DateTime<Utc>,
    /// The operation
    pub operation: WalOperation,
}

/// Write-Ahead Log for replication
pub struct WriteAheadLog {
    /// Base directory for WAL files
    base_dir: PathBuf,
    /// Current WAL file writer
    writer: RwLock<Option<BufWriter<File>>>,
    /// Current sequence number
    current_seq: AtomicU64,
    /// Maximum entries per WAL segment
    max_entries_per_segment: usize,
    /// Current segment number
    current_segment: AtomicU64,
    /// Entries in current segment
    entries_in_segment: AtomicU64,
}

impl WriteAheadLog {
    /// Create or open WAL at the given directory
    pub fn new(base_dir: &Path) -> Result<Self> {
        fs::create_dir_all(base_dir)?;

        let wal = Self {
            base_dir: base_dir.to_path_buf(),
            writer: RwLock::new(None),
            current_seq: AtomicU64::new(0),
            max_entries_per_segment: 10000,
            current_segment: AtomicU64::new(0),
            entries_in_segment: AtomicU64::new(0),
        };

        // Load state from existing WAL files
        wal.recover_state()?;

        Ok(wal)
    }

    /// Recover state from existing WAL files
    fn recover_state(&self) -> Result<()> {
        let mut max_seq: u64 = 0;
        let mut max_segment: u64 = 0;

        // Find all WAL segment files
        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("wal_") && name.ends_with(".log") {
                    // Parse segment number
                    if let Some(num_str) = name.strip_prefix("wal_").and_then(|s| s.strip_suffix(".log")) {
                        if let Ok(segment_num) = num_str.parse::<u64>() {
                            max_segment = max_segment.max(segment_num);

                            // Read last sequence number from segment
                            if let Ok(file) = File::open(&path) {
                                let reader = BufReader::new(file);
                                for line in reader.lines().filter_map(|l| l.ok()) {
                                    if let Ok(entry) = serde_json::from_str::<WalEntry>(&line) {
                                        max_seq = max_seq.max(entry.seq);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        self.current_seq.store(max_seq, Ordering::SeqCst);
        self.current_segment.store(max_segment, Ordering::SeqCst);

        // Count entries in current segment
        let current_segment_path = self.segment_path(max_segment);
        if current_segment_path.exists() {
            let file = File::open(&current_segment_path)?;
            let reader = BufReader::new(file);
            let count = reader.lines().count() as u64;
            self.entries_in_segment.store(count, Ordering::SeqCst);
        }

        Ok(())
    }

    /// Get path for a segment file
    fn segment_path(&self, segment: u64) -> PathBuf {
        self.base_dir.join(format!("wal_{:08}.log", segment))
    }

    /// Ensure writer is open for current segment
    fn ensure_writer(&self) -> Result<()> {
        let mut writer_guard = self.writer.write();

        // Check if we need to rotate
        let entries = self.entries_in_segment.load(Ordering::SeqCst);
        if entries >= self.max_entries_per_segment as u64 {
            // Close current segment
            if let Some(ref mut w) = *writer_guard {
                w.flush()?;
            }
            *writer_guard = None;

            // Start new segment
            self.current_segment.fetch_add(1, Ordering::SeqCst);
            self.entries_in_segment.store(0, Ordering::SeqCst);
        }

        if writer_guard.is_none() {
            let segment = self.current_segment.load(Ordering::SeqCst);
            let path = self.segment_path(segment);
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)?;
            *writer_guard = Some(BufWriter::new(file));
        }

        Ok(())
    }

    /// Append an operation to the WAL
    pub fn append(&self, operation: WalOperation) -> Result<u64> {
        self.ensure_writer()?;

        let seq = self.current_seq.fetch_add(1, Ordering::SeqCst) + 1;
        let entry = WalEntry {
            seq,
            timestamp: Utc::now(),
            operation,
        };

        let mut writer_guard = self.writer.write();
        if let Some(ref mut w) = *writer_guard {
            let json = serde_json::to_string(&entry)?;
            writeln!(w, "{}", json)?;
            w.flush()?;
        }

        self.entries_in_segment.fetch_add(1, Ordering::SeqCst);
        Ok(seq)
    }

    /// Get current sequence number
    pub fn current_seq(&self) -> u64 {
        self.current_seq.load(Ordering::SeqCst)
    }

    /// Read entries from a given sequence number
    pub fn read_from(&self, from_seq: u64) -> Result<Vec<WalEntry>> {
        let mut entries = Vec::new();

        // Read from all segments
        let mut segments: Vec<u64> = Vec::new();
        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("wal_") && name.ends_with(".log") {
                    if let Some(num_str) = name.strip_prefix("wal_").and_then(|s| s.strip_suffix(".log")) {
                        if let Ok(segment_num) = num_str.parse::<u64>() {
                            segments.push(segment_num);
                        }
                    }
                }
            }
        }

        segments.sort();

        for segment in segments {
            let path = self.segment_path(segment);
            if path.exists() {
                let file = File::open(&path)?;
                let reader = BufReader::new(file);

                for line in reader.lines() {
                    let line = line?;
                    if let Ok(entry) = serde_json::from_str::<WalEntry>(&line) {
                        if entry.seq > from_seq {
                            entries.push(entry);
                        }
                    }
                }
            }
        }

        // Sort by sequence number
        entries.sort_by_key(|e| e.seq);
        Ok(entries)
    }

    /// Truncate WAL up to (and including) a sequence number
    /// Used after successful snapshot
    pub fn truncate_before(&self, before_seq: u64) -> Result<()> {
        let mut segments_to_delete: Vec<PathBuf> = Vec::new();

        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("wal_") && name.ends_with(".log") {
                    // Check if all entries in this segment are before the truncation point
                    let file = File::open(&path)?;
                    let reader = BufReader::new(file);

                    let mut max_seq_in_segment: u64 = 0;
                    for line in reader.lines().filter_map(|l| l.ok()) {
                        if let Ok(entry) = serde_json::from_str::<WalEntry>(&line) {
                            max_seq_in_segment = max_seq_in_segment.max(entry.seq);
                        }
                    }

                    if max_seq_in_segment < before_seq {
                        segments_to_delete.push(path);
                    }
                }
            }
        }

        for path in segments_to_delete {
            fs::remove_file(path)?;
        }

        Ok(())
    }

    /// Get WAL statistics
    pub fn stats(&self) -> WalStats {
        let mut total_entries = 0;
        let mut total_segments = 0;
        let mut total_size = 0;

        if let Ok(entries) = fs::read_dir(&self.base_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("wal_") && name.ends_with(".log") {
                        total_segments += 1;
                        if let Ok(meta) = fs::metadata(&path) {
                            total_size += meta.len();
                        }
                        if let Ok(file) = File::open(&path) {
                            let reader = BufReader::new(file);
                            total_entries += reader.lines().count();
                        }
                    }
                }
            }
        }

        WalStats {
            current_seq: self.current_seq.load(Ordering::SeqCst),
            total_entries,
            total_segments,
            total_size_bytes: total_size,
        }
    }
}

/// WAL statistics
#[derive(Debug, Clone, Serialize)]
pub struct WalStats {
    pub current_seq: u64,
    pub total_entries: usize,
    pub total_segments: usize,
    pub total_size_bytes: u64,
}

/// Replication state for a replica
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaState {
    /// Primary node address
    pub primary_addr: String,
    /// Last synced sequence number
    pub last_seq: u64,
    /// Last sync timestamp
    pub last_sync: DateTime<Utc>,
    /// Sync status
    pub status: SyncStatus,
}

/// Sync status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncStatus {
    /// In sync with primary
    Synced,
    /// Syncing from primary
    Syncing,
    /// Needs full resync (snapshot)
    NeedsSnapshot,
    /// Disconnected from primary
    Disconnected,
    /// Error state
    Error,
}

/// Replication manager
pub struct ReplicationManager {
    /// Role of this node
    role: RwLock<ReplicationRole>,
    /// WAL (only active when primary or standalone)
    wal: Option<Arc<WriteAheadLog>>,
    /// Replica state (only active when replica)
    replica_state: RwLock<Option<ReplicaState>>,
    /// Data directory
    data_dir: PathBuf,
}

impl ReplicationManager {
    /// Create a new replication manager
    pub fn new(data_dir: &Path, role: ReplicationRole) -> Result<Self> {
        let wal = if role == ReplicationRole::Primary || role == ReplicationRole::Standalone {
            let wal_dir = data_dir.join("wal");
            Some(Arc::new(WriteAheadLog::new(&wal_dir)?))
        } else {
            None
        };

        Ok(Self {
            role: RwLock::new(role),
            wal,
            replica_state: RwLock::new(None),
            data_dir: data_dir.to_path_buf(),
        })
    }

    /// Get current role
    pub fn role(&self) -> ReplicationRole {
        *self.role.read()
    }

    /// Check if writes are allowed
    pub fn is_writable(&self) -> bool {
        let role = self.role();
        role == ReplicationRole::Primary || role == ReplicationRole::Standalone
    }

    /// Log an operation to WAL (primary only)
    pub fn log_operation(&self, operation: WalOperation) -> Result<Option<u64>> {
        if let Some(ref wal) = self.wal {
            Ok(Some(wal.append(operation)?))
        } else {
            Ok(None)
        }
    }

    /// Get WAL entries from a sequence number
    pub fn get_entries_from(&self, from_seq: u64) -> Result<Vec<WalEntry>> {
        if let Some(ref wal) = self.wal {
            wal.read_from(from_seq)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get current WAL sequence
    pub fn current_seq(&self) -> u64 {
        self.wal.as_ref().map(|w| w.current_seq()).unwrap_or(0)
    }

    /// Get WAL statistics
    pub fn wal_stats(&self) -> Option<WalStats> {
        self.wal.as_ref().map(|w| w.stats())
    }

    /// Set replica state
    pub fn set_replica_state(&self, state: ReplicaState) {
        *self.replica_state.write() = Some(state);
    }

    /// Get replica state
    pub fn replica_state(&self) -> Option<ReplicaState> {
        self.replica_state.read().clone()
    }

    /// Get replication status
    pub fn status(&self) -> ReplicationStatus {
        let role = self.role();
        let wal_stats = self.wal_stats();
        let replica_state = self.replica_state();

        ReplicationStatus {
            role,
            current_seq: self.current_seq(),
            wal_stats,
            replica_state,
        }
    }
}

/// Replication status
#[derive(Debug, Clone, Serialize)]
pub struct ReplicationStatus {
    pub role: ReplicationRole,
    pub current_seq: u64,
    pub wal_stats: Option<WalStats>,
    pub replica_state: Option<ReplicaState>,
}
