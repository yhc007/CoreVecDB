//! LZ4 Compression for VectorDB Storage.
//!
//! Provides fast compression for vector data segments, reducing storage size
//! while maintaining reasonable decompression speed for random access.
//!
//! # Concept
//! - **Block Compression**: Vectors are grouped into blocks and compressed
//! - **Offset Table**: Maintains byte offsets for random access
//! - **Lazy Decompression**: Only decompress accessed blocks
//!
//! # Usage
//! ```rust,ignore
//! use vectordb::storage::compression::{CompressedSegment, CompressionConfig};
//!
//! let config = CompressionConfig::default();
//! let mut segment = CompressedSegment::new(128, config);
//!
//! // Add vectors
//! segment.add_vector(&vector)?;
//!
//! // Finalize compression
//! segment.finalize()?;
//!
//! // Read vector (decompresses block on demand)
//! let vector = segment.get_vector(0)?;
//! ```

use std::io::{self, Read, Write};

use lz4_flex::{compress_prepend_size, decompress_size_prepended};

/// Configuration for compression.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Number of vectors per compressed block.
    pub block_size: usize,
    /// Enable compression.
    pub enabled: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            block_size: 256, // 256 vectors per block
            enabled: true,
        }
    }
}

/// A compressed block of vectors.
#[derive(Debug, Clone)]
struct CompressedBlock {
    /// Compressed data.
    compressed_data: Vec<u8>,
    /// Original (uncompressed) size in bytes.
    original_size: usize,
    /// Number of vectors in this block.
    vector_count: usize,
}

impl CompressedBlock {
    /// Create a new compressed block from raw vector data.
    fn compress(data: &[u8], vector_count: usize) -> Self {
        let compressed = compress_prepend_size(data);
        Self {
            original_size: data.len(),
            compressed_data: compressed,
            vector_count,
        }
    }

    /// Decompress the block.
    fn decompress(&self) -> io::Result<Vec<u8>> {
        decompress_size_prepended(&self.compressed_data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Get compression ratio.
    fn compression_ratio(&self) -> f64 {
        if self.original_size == 0 {
            1.0
        } else {
            self.compressed_data.len() as f64 / self.original_size as f64
        }
    }
}

/// Compressed segment for vector storage.
pub struct CompressedSegment {
    /// Vector dimension.
    dim: usize,
    /// Bytes per vector.
    vector_bytes: usize,
    /// Configuration.
    config: CompressionConfig,
    /// Compressed blocks.
    blocks: Vec<CompressedBlock>,
    /// Current block being built (uncompressed).
    current_block: Vec<u8>,
    /// Number of vectors in current block.
    current_block_count: usize,
    /// Total vector count.
    total_vectors: usize,
    /// Block offset table (vector index to block index).
    block_offsets: Vec<usize>,
}

impl CompressedSegment {
    /// Create a new compressed segment.
    pub fn new(dim: usize, config: CompressionConfig) -> Self {
        let vector_bytes = dim * std::mem::size_of::<f32>();
        let block_capacity = config.block_size * vector_bytes;

        Self {
            dim,
            vector_bytes,
            config,
            blocks: Vec::new(),
            current_block: Vec::with_capacity(block_capacity),
            current_block_count: 0,
            total_vectors: 0,
            block_offsets: Vec::new(),
        }
    }

    /// Add a vector to the segment.
    pub fn add_vector(&mut self, vector: &[f32]) -> io::Result<usize> {
        if vector.len() != self.dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Vector dimension mismatch: expected {}, got {}", self.dim, vector.len()),
            ));
        }

        // Record block offset for this vector
        self.block_offsets.push(self.blocks.len());

        // Add vector bytes to current block
        for &v in vector {
            self.current_block.extend_from_slice(&v.to_le_bytes());
        }
        self.current_block_count += 1;

        let vector_id = self.total_vectors;
        self.total_vectors += 1;

        // Compress block if full
        if self.current_block_count >= self.config.block_size {
            self.flush_current_block();
        }

        Ok(vector_id)
    }

    /// Flush the current block (compress and store).
    fn flush_current_block(&mut self) {
        if self.current_block_count == 0 {
            return;
        }

        if self.config.enabled {
            let block = CompressedBlock::compress(&self.current_block, self.current_block_count);
            self.blocks.push(block);
        } else {
            // Store uncompressed
            let block = CompressedBlock {
                compressed_data: self.current_block.clone(),
                original_size: self.current_block.len(),
                vector_count: self.current_block_count,
            };
            self.blocks.push(block);
        }

        self.current_block.clear();
        self.current_block_count = 0;
    }

    /// Finalize the segment (flush remaining data).
    pub fn finalize(&mut self) {
        self.flush_current_block();
    }

    /// Get a vector by index.
    pub fn get_vector(&self, index: usize) -> io::Result<Vec<f32>> {
        if index >= self.total_vectors {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Vector index out of bounds: {}", index),
            ));
        }

        // Find which block contains this vector
        let block_idx = self.block_offsets[index];

        // Check if vector is in unflushed current block
        if block_idx >= self.blocks.len() {
            // Vector is in current (unflushed) block
            let offset_in_block = (index - self.first_vector_in_current_block()) * self.vector_bytes;
            return self.read_vector_from_bytes(&self.current_block, offset_in_block);
        }

        // Decompress the block
        let block = &self.blocks[block_idx];
        let decompressed = if self.config.enabled {
            block.decompress()?
        } else {
            block.compressed_data.clone()
        };

        // Find offset within block
        let first_in_block = self.first_vector_in_block(block_idx);
        let offset_in_block = (index - first_in_block) * self.vector_bytes;

        self.read_vector_from_bytes(&decompressed, offset_in_block)
    }

    /// Get first vector index in a block.
    fn first_vector_in_block(&self, block_idx: usize) -> usize {
        self.block_offsets
            .iter()
            .position(|&b| b == block_idx)
            .unwrap_or(0)
    }

    /// Get first vector index in current (unflushed) block.
    fn first_vector_in_current_block(&self) -> usize {
        self.total_vectors - self.current_block_count
    }

    /// Read a vector from a byte slice.
    fn read_vector_from_bytes(&self, bytes: &[u8], offset: usize) -> io::Result<Vec<f32>> {
        let end = offset + self.vector_bytes;
        if end > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Not enough data in block",
            ));
        }

        let vector: Vec<f32> = bytes[offset..end]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(vector)
    }

    /// Get total number of vectors.
    pub fn len(&self) -> usize {
        self.total_vectors
    }

    /// Check if segment is empty.
    pub fn is_empty(&self) -> bool {
        self.total_vectors == 0
    }

    /// Get compression statistics.
    pub fn stats(&self) -> CompressionStats {
        let total_compressed: usize = self.blocks.iter().map(|b| b.compressed_data.len()).sum();
        let total_original: usize = self.blocks.iter().map(|b| b.original_size).sum();

        // Include current block in original size
        let total_original = total_original + self.current_block.len();

        let compression_ratio = if total_original == 0 {
            1.0
        } else {
            (total_compressed + self.current_block.len()) as f64 / total_original as f64
        };

        CompressionStats {
            total_vectors: self.total_vectors,
            total_blocks: self.blocks.len() + if self.current_block_count > 0 { 1 } else { 0 },
            compressed_bytes: total_compressed + self.current_block.len(),
            original_bytes: total_original,
            compression_ratio,
            space_saved_percent: (1.0 - compression_ratio) * 100.0,
        }
    }

    /// Serialize the segment to bytes.
    pub fn to_bytes(&self) -> io::Result<Vec<u8>> {
        let mut buffer = Vec::new();

        // Header
        buffer.extend_from_slice(&(self.dim as u32).to_le_bytes());
        buffer.extend_from_slice(&(self.total_vectors as u64).to_le_bytes());
        buffer.extend_from_slice(&(self.blocks.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&[if self.config.enabled { 1u8 } else { 0u8 }]);

        // Block offsets
        for offset in &self.block_offsets {
            buffer.extend_from_slice(&(*offset as u32).to_le_bytes());
        }

        // Blocks
        for block in &self.blocks {
            buffer.extend_from_slice(&(block.compressed_data.len() as u32).to_le_bytes());
            buffer.extend_from_slice(&(block.original_size as u32).to_le_bytes());
            buffer.extend_from_slice(&(block.vector_count as u32).to_le_bytes());
            buffer.extend_from_slice(&block.compressed_data);
        }

        // Current block (if any)
        buffer.extend_from_slice(&(self.current_block.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&(self.current_block_count as u32).to_le_bytes());
        buffer.extend_from_slice(&self.current_block);

        Ok(buffer)
    }

    /// Deserialize a segment from bytes.
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let mut cursor = 0;

        // Helper to read bytes
        let read_u32 = |cursor: &mut usize| -> io::Result<u32> {
            if *cursor + 4 > data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated data"));
            }
            let bytes = [data[*cursor], data[*cursor + 1], data[*cursor + 2], data[*cursor + 3]];
            *cursor += 4;
            Ok(u32::from_le_bytes(bytes))
        };

        let read_u64 = |cursor: &mut usize| -> io::Result<u64> {
            if *cursor + 8 > data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated data"));
            }
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&data[*cursor..*cursor + 8]);
            *cursor += 8;
            Ok(u64::from_le_bytes(bytes))
        };

        // Header
        let dim = read_u32(&mut cursor)? as usize;
        let total_vectors = read_u64(&mut cursor)? as usize;
        let block_count = read_u32(&mut cursor)? as usize;
        let enabled = data[cursor] != 0;
        cursor += 1;

        // Block offsets
        let mut block_offsets = Vec::with_capacity(total_vectors);
        for _ in 0..total_vectors {
            block_offsets.push(read_u32(&mut cursor)? as usize);
        }

        // Blocks
        let mut blocks = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            let compressed_len = read_u32(&mut cursor)? as usize;
            let original_size = read_u32(&mut cursor)? as usize;
            let vector_count = read_u32(&mut cursor)? as usize;

            if cursor + compressed_len > data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Truncated block data"));
            }
            let compressed_data = data[cursor..cursor + compressed_len].to_vec();
            cursor += compressed_len;

            blocks.push(CompressedBlock {
                compressed_data,
                original_size,
                vector_count,
            });
        }

        // Current block
        let current_block_len = read_u32(&mut cursor)? as usize;
        let current_block_count = read_u32(&mut cursor)? as usize;

        let current_block = if cursor + current_block_len <= data.len() {
            data[cursor..cursor + current_block_len].to_vec()
        } else {
            Vec::new()
        };

        let vector_bytes = dim * std::mem::size_of::<f32>();
        Ok(Self {
            dim,
            vector_bytes,
            config: CompressionConfig {
                block_size: 256,
                enabled,
            },
            blocks,
            current_block,
            current_block_count,
            total_vectors,
            block_offsets,
        })
    }
}

/// Compression statistics.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub total_vectors: usize,
    pub total_blocks: usize,
    pub compressed_bytes: usize,
    pub original_bytes: usize,
    pub compression_ratio: f64,
    pub space_saved_percent: f64,
}

/// Utility functions for compressing/decompressing raw data.
pub mod utils {
    use super::*;

    /// Compress a byte slice using LZ4.
    pub fn compress(data: &[u8]) -> Vec<u8> {
        compress_prepend_size(data)
    }

    /// Decompress LZ4-compressed data.
    pub fn decompress(data: &[u8]) -> io::Result<Vec<u8>> {
        decompress_size_prepended(data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Estimate compression ratio for typical vector data.
    pub fn estimate_ratio(dim: usize, sample_vectors: &[Vec<f32>]) -> f64 {
        if sample_vectors.is_empty() {
            return 0.7; // Default estimate
        }

        let mut original_bytes = 0;
        let mut compressed_bytes = 0;

        // Compress sample vectors
        for vector in sample_vectors {
            let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
            original_bytes += bytes.len();

            let compressed = compress(&bytes);
            compressed_bytes += compressed.len();
        }

        if original_bytes == 0 {
            0.7
        } else {
            compressed_bytes as f64 / original_bytes as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressed_segment_basic() {
        let config = CompressionConfig {
            block_size: 10,
            enabled: true,
        };
        let mut segment = CompressedSegment::new(4, config);

        // Add vectors
        for i in 0..25 {
            let vector = vec![i as f32; 4];
            segment.add_vector(&vector).unwrap();
        }
        segment.finalize();

        assert_eq!(segment.len(), 25);

        // Retrieve and verify
        for i in 0..25 {
            let vector = segment.get_vector(i).unwrap();
            assert_eq!(vector, vec![i as f32; 4]);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let config = CompressionConfig {
            block_size: 100,
            enabled: true,
        };
        let mut segment = CompressedSegment::new(128, config);

        // Add many similar vectors (should compress well)
        for i in 0..500 {
            let vector = vec![1.0; 128]; // All same values
            segment.add_vector(&vector).unwrap();
        }
        segment.finalize();

        let stats = segment.stats();
        // LZ4 should achieve some compression on repetitive data
        assert!(stats.compression_ratio < 1.0, "Expected compression, got ratio: {}", stats.compression_ratio);
    }

    #[test]
    fn test_uncompressed_segment() {
        let config = CompressionConfig {
            block_size: 10,
            enabled: false,
        };
        let mut segment = CompressedSegment::new(4, config);

        for i in 0..15 {
            let vector = vec![i as f32; 4];
            segment.add_vector(&vector).unwrap();
        }
        segment.finalize();

        // Verify retrieval
        let vector = segment.get_vector(7).unwrap();
        assert_eq!(vector, vec![7.0; 4]);
    }

    #[test]
    fn test_serialization() {
        let config = CompressionConfig {
            block_size: 5,
            enabled: true,
        };
        let mut segment = CompressedSegment::new(4, config);

        for i in 0..12 {
            segment.add_vector(&vec![i as f32; 4]).unwrap();
        }
        segment.finalize();

        // Serialize
        let bytes = segment.to_bytes().unwrap();

        // Deserialize
        let restored = CompressedSegment::from_bytes(&bytes).unwrap();

        assert_eq!(restored.len(), 12);
        for i in 0..12 {
            let vector = restored.get_vector(i).unwrap();
            assert_eq!(vector, vec![i as f32; 4]);
        }
    }

    #[test]
    fn test_utils_compress_decompress() {
        let original = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8];

        let compressed = utils::compress(&original);
        let decompressed = utils::decompress(&compressed).unwrap();

        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_partial_block() {
        let config = CompressionConfig {
            block_size: 10,
            enabled: true,
        };
        let mut segment = CompressedSegment::new(4, config);

        // Add only 3 vectors (less than block size)
        for i in 0..3 {
            segment.add_vector(&vec![i as f32; 4]).unwrap();
        }

        // Access before finalize
        let vector = segment.get_vector(1).unwrap();
        assert_eq!(vector, vec![1.0; 4]);

        // Finalize and access again
        segment.finalize();
        let vector = segment.get_vector(2).unwrap();
        assert_eq!(vector, vec![2.0; 4]);
    }
}
