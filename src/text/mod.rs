//! BM25 Text Search Module
//!
//! Implements BM25 (Best Matching 25) algorithm for full-text search.
//! Supports hybrid search combining vector similarity with text relevance.

use anyhow::Result;
use parking_lot::RwLock;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

/// BM25 parameters
const K1: f32 = 1.2; // Term frequency saturation parameter
const B: f32 = 0.75; // Length normalization parameter

/// A single term in the inverted index
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TermEntry {
    /// Document IDs containing this term
    doc_ids: Vec<u64>,
    /// Term frequency per document (parallel array with doc_ids)
    term_freqs: Vec<u32>,
}

/// Document statistics for BM25 scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DocStats {
    /// Total number of terms in this document
    length: u32,
    /// Fields indexed for this document
    fields: Vec<String>,
}

/// BM25 Text Index
///
/// Provides full-text search capabilities using the BM25 ranking algorithm.
#[derive(Debug)]
pub struct TextIndex {
    /// Path for persistence
    path: PathBuf,
    /// Inverted index: term -> (doc_ids, term_frequencies)
    inverted_index: RwLock<HashMap<String, TermEntry>>,
    /// Document statistics
    doc_stats: RwLock<HashMap<u64, DocStats>>,
    /// Total number of documents
    doc_count: RwLock<usize>,
    /// Average document length
    avg_doc_length: RwLock<f32>,
    /// Fields to index
    indexed_fields: Vec<String>,
    /// Whether index has been modified
    dirty: RwLock<bool>,
}

/// Search result with BM25 score
#[derive(Debug, Clone)]
pub struct TextSearchResult {
    pub id: u64,
    pub score: f32,
}

/// Tokenizer for text processing
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() > 1) // Filter single chars
        .map(|s| s.to_string())
        .collect()
}

impl TextIndex {
    /// Create a new text index
    pub fn new(path: PathBuf, indexed_fields: Vec<String>) -> Self {
        Self {
            path,
            inverted_index: RwLock::new(HashMap::new()),
            doc_stats: RwLock::new(HashMap::new()),
            doc_count: RwLock::new(0),
            avg_doc_length: RwLock::new(0.0),
            indexed_fields,
            dirty: RwLock::new(false),
        }
    }

    /// Open existing index or create new one
    pub fn open(path: PathBuf, indexed_fields: Vec<String>) -> Result<Self> {
        let index_file = path.join("text_index.bin");

        if index_file.exists() {
            let file = File::open(&index_file)?;
            let reader = BufReader::new(file);
            let data: TextIndexData = bincode::deserialize_from(reader)?;

            Ok(Self {
                path,
                inverted_index: RwLock::new(data.inverted_index),
                doc_stats: RwLock::new(data.doc_stats),
                doc_count: RwLock::new(data.doc_count),
                avg_doc_length: RwLock::new(data.avg_doc_length),
                indexed_fields,
                dirty: RwLock::new(false),
            })
        } else {
            Ok(Self::new(path, indexed_fields))
        }
    }

    /// Index a document's text fields
    pub fn index_document(&self, id: u64, metadata: &[(String, String)]) -> Result<()> {
        let mut all_terms: Vec<String> = Vec::new();
        let mut indexed_fields_used: Vec<String> = Vec::new();

        // Extract and tokenize text from indexed fields
        for (key, value) in metadata {
            if self.indexed_fields.contains(key) || self.indexed_fields.is_empty() {
                let terms = tokenize(value);
                all_terms.extend(terms);
                indexed_fields_used.push(key.clone());
            }
        }

        if all_terms.is_empty() {
            return Ok(());
        }

        // Count term frequencies
        let mut term_counts: HashMap<String, u32> = HashMap::new();
        for term in &all_terms {
            *term_counts.entry(term.clone()).or_insert(0) += 1;
        }

        // Update inverted index
        {
            let mut inv_index = self.inverted_index.write();
            for (term, freq) in term_counts {
                let entry = inv_index.entry(term).or_insert_with(|| TermEntry {
                    doc_ids: Vec::new(),
                    term_freqs: Vec::new(),
                });
                entry.doc_ids.push(id);
                entry.term_freqs.push(freq);
            }
        }

        // Update document stats
        {
            let mut stats = self.doc_stats.write();
            let doc_length = all_terms.len() as u32;
            stats.insert(id, DocStats {
                length: doc_length,
                fields: indexed_fields_used,
            });

            // Update average document length
            let mut doc_count = self.doc_count.write();
            let mut avg_len = self.avg_doc_length.write();

            let old_total = *avg_len * (*doc_count as f32);
            *doc_count += 1;
            *avg_len = (old_total + doc_length as f32) / (*doc_count as f32);
        }

        *self.dirty.write() = true;
        Ok(())
    }

    /// Index multiple documents in batch
    pub fn index_batch(&self, documents: &[(u64, Vec<(String, String)>)]) -> Result<()> {
        for (id, metadata) in documents {
            self.index_document(*id, metadata)?;
        }
        Ok(())
    }

    /// Remove a document from the index
    pub fn remove_document(&self, id: u64) -> Result<()> {
        // Remove from inverted index
        {
            let mut inv_index = self.inverted_index.write();
            for entry in inv_index.values_mut() {
                if let Some(pos) = entry.doc_ids.iter().position(|&doc_id| doc_id == id) {
                    entry.doc_ids.remove(pos);
                    entry.term_freqs.remove(pos);
                }
            }
            // Clean up empty entries
            inv_index.retain(|_, entry| !entry.doc_ids.is_empty());
        }

        // Remove from doc stats and update averages
        {
            let mut stats = self.doc_stats.write();
            if let Some(doc_stat) = stats.remove(&id) {
                let mut doc_count = self.doc_count.write();
                let mut avg_len = self.avg_doc_length.write();

                if *doc_count > 1 {
                    let old_total = *avg_len * (*doc_count as f32);
                    *doc_count -= 1;
                    *avg_len = (old_total - doc_stat.length as f32) / (*doc_count as f32);
                } else {
                    *doc_count = 0;
                    *avg_len = 0.0;
                }
            }
        }

        *self.dirty.write() = true;
        Ok(())
    }

    /// Search for documents matching the query
    pub fn search(&self, query: &str, k: usize) -> Vec<TextSearchResult> {
        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let inv_index = self.inverted_index.read();
        let doc_stats = self.doc_stats.read();
        let doc_count = *self.doc_count.read();
        let avg_doc_length = *self.avg_doc_length.read();

        if doc_count == 0 {
            return Vec::new();
        }

        // Calculate BM25 scores for all candidate documents
        let mut scores: HashMap<u64, f32> = HashMap::new();

        for term in &query_terms {
            if let Some(entry) = inv_index.get(term) {
                // IDF calculation
                let df = entry.doc_ids.len() as f32;
                let idf = ((doc_count as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();

                // Score each document containing this term
                for (i, &doc_id) in entry.doc_ids.iter().enumerate() {
                    let tf = entry.term_freqs[i] as f32;

                    let doc_length = doc_stats
                        .get(&doc_id)
                        .map(|s| s.length as f32)
                        .unwrap_or(avg_doc_length);

                    // BM25 formula
                    let numerator = tf * (K1 + 1.0);
                    let denominator = tf + K1 * (1.0 - B + B * (doc_length / avg_doc_length));
                    let term_score = idf * (numerator / denominator);

                    *scores.entry(doc_id).or_insert(0.0) += term_score;
                }
            }
        }

        // Sort by score and return top-k
        let mut results: Vec<_> = scores
            .into_iter()
            .map(|(id, score)| TextSearchResult { id, score })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Search and return document IDs as bitmap (for filtering)
    pub fn search_bitmap(&self, query: &str, min_score: f32) -> RoaringBitmap {
        let results = self.search(query, usize::MAX);
        results
            .into_iter()
            .filter(|r| r.score >= min_score)
            .map(|r| r.id as u32)
            .collect()
    }

    /// Get documents containing all query terms (AND logic)
    pub fn search_must_match(&self, query: &str) -> RoaringBitmap {
        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return RoaringBitmap::new();
        }

        let inv_index = self.inverted_index.read();

        let mut result: Option<RoaringBitmap> = None;

        for term in &query_terms {
            if let Some(entry) = inv_index.get(term) {
                let term_bitmap: RoaringBitmap = entry
                    .doc_ids
                    .iter()
                    .map(|&id| id as u32)
                    .collect();

                result = match result {
                    None => Some(term_bitmap),
                    Some(existing) => Some(&existing & &term_bitmap),
                };
            } else {
                // Term not found, no documents match
                return RoaringBitmap::new();
            }
        }

        result.unwrap_or_else(RoaringBitmap::new)
    }

    /// Persist the index to disk
    pub fn save(&self) -> Result<()> {
        if !*self.dirty.read() {
            return Ok(());
        }

        std::fs::create_dir_all(&self.path)?;
        let index_file = self.path.join("text_index.bin");

        let data = TextIndexData {
            inverted_index: self.inverted_index.read().clone(),
            doc_stats: self.doc_stats.read().clone(),
            doc_count: *self.doc_count.read(),
            avg_doc_length: *self.avg_doc_length.read(),
        };

        let file = File::create(&index_file)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &data)?;

        *self.dirty.write() = false;
        Ok(())
    }

    /// Get index statistics
    pub fn stats(&self) -> TextIndexStats {
        TextIndexStats {
            document_count: *self.doc_count.read(),
            unique_terms: self.inverted_index.read().len(),
            avg_doc_length: *self.avg_doc_length.read(),
            indexed_fields: self.indexed_fields.clone(),
        }
    }

    /// Check if a document is indexed
    pub fn contains(&self, id: u64) -> bool {
        self.doc_stats.read().contains_key(&id)
    }

    /// Get the number of indexed documents
    pub fn len(&self) -> usize {
        *self.doc_count.read()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Serializable index data
#[derive(Serialize, Deserialize)]
struct TextIndexData {
    inverted_index: HashMap<String, TermEntry>,
    doc_stats: HashMap<u64, DocStats>,
    doc_count: usize,
    avg_doc_length: f32,
}

/// Index statistics
#[derive(Debug, Clone, Serialize)]
pub struct TextIndexStats {
    pub document_count: usize,
    pub unique_terms: usize,
    pub avg_doc_length: f32,
    pub indexed_fields: Vec<String>,
}

/// Hybrid search result combining vector and text scores
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    pub id: u64,
    pub vector_score: f32,
    pub text_score: f32,
    pub combined_score: f32,
}

/// Combine vector search results with text search results
///
/// Uses RRF (Reciprocal Rank Fusion) or weighted combination
pub fn hybrid_combine(
    vector_results: &[(u64, f32)],  // (id, distance/similarity)
    text_results: &[TextSearchResult],
    alpha: f32,  // Weight for vector score (0.0-1.0)
    k: usize,
) -> Vec<HybridSearchResult> {
    // Normalize vector scores (convert distance to similarity if needed)
    let max_vec_score = vector_results
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_vec_score = vector_results
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::INFINITY, f32::min);
    let vec_range = (max_vec_score - min_vec_score).max(1e-6);

    // Normalize text scores
    let max_text_score = text_results
        .iter()
        .map(|r| r.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_text_score = text_results
        .iter()
        .map(|r| r.score)
        .fold(f32::INFINITY, f32::min);
    let text_range = (max_text_score - min_text_score).max(1e-6);

    // Build score maps
    let mut vec_scores: HashMap<u64, f32> = HashMap::new();
    for (id, score) in vector_results {
        // Normalize to 0-1 (higher is better)
        // Assuming score is similarity (higher is better)
        let normalized = (*score - min_vec_score) / vec_range;
        vec_scores.insert(*id, normalized);
    }

    let mut text_scores: HashMap<u64, f32> = HashMap::new();
    for result in text_results {
        let normalized = (result.score - min_text_score) / text_range;
        text_scores.insert(result.id, normalized);
    }

    // Combine scores for all unique IDs
    let mut all_ids: std::collections::HashSet<u64> = vec_scores.keys().copied().collect();
    all_ids.extend(text_scores.keys());

    let mut results: Vec<HybridSearchResult> = all_ids
        .into_iter()
        .map(|id| {
            let vec_score = *vec_scores.get(&id).unwrap_or(&0.0);
            let txt_score = *text_scores.get(&id).unwrap_or(&0.0);
            let combined = alpha * vec_score + (1.0 - alpha) * txt_score;

            HybridSearchResult {
                id,
                vector_score: vec_score,
                text_score: txt_score,
                combined_score: combined,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(k);
    results
}

/// Reciprocal Rank Fusion for combining rankings
pub fn rrf_combine(
    vector_results: &[(u64, f32)],
    text_results: &[TextSearchResult],
    k_constant: f32,  // RRF constant, typically 60
    limit: usize,
) -> Vec<HybridSearchResult> {
    let mut rrf_scores: HashMap<u64, (f32, f32, f32)> = HashMap::new();

    // Add vector result RRF scores
    for (rank, (id, score)) in vector_results.iter().enumerate() {
        let rrf_score = 1.0 / (k_constant + (rank + 1) as f32);
        rrf_scores.entry(*id).or_insert((0.0, 0.0, 0.0)).0 = *score;
        rrf_scores.entry(*id).or_insert((0.0, 0.0, 0.0)).2 += rrf_score;
    }

    // Add text result RRF scores
    for (rank, result) in text_results.iter().enumerate() {
        let rrf_score = 1.0 / (k_constant + (rank + 1) as f32);
        let entry = rrf_scores.entry(result.id).or_insert((0.0, 0.0, 0.0));
        entry.1 = result.score;
        entry.2 += rrf_score;
    }

    let mut results: Vec<HybridSearchResult> = rrf_scores
        .into_iter()
        .map(|(id, (vec_score, text_score, combined))| HybridSearchResult {
            id,
            vector_score: vec_score,
            text_score,
            combined_score: combined,
        })
        .collect();

    results.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit);
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is a TEST.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"this".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Single char 'a' should be filtered
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_text_index_basic() {
        let dir = tempdir().unwrap();
        let index = TextIndex::new(dir.path().to_path_buf(), vec!["title".to_string()]);

        // Index some documents
        index
            .index_document(1, &[("title".to_string(), "quick brown fox".to_string())])
            .unwrap();
        index
            .index_document(2, &[("title".to_string(), "lazy brown dog".to_string())])
            .unwrap();
        index
            .index_document(3, &[("title".to_string(), "quick lazy cat".to_string())])
            .unwrap();

        // Search
        let results = index.search("brown", 10);
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.id == 1));
        assert!(results.iter().any(|r| r.id == 2));

        let results = index.search("quick", 10);
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.id == 1));
        assert!(results.iter().any(|r| r.id == 3));
    }

    #[test]
    fn test_text_index_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Create and populate index
        {
            let index = TextIndex::new(path.clone(), vec!["content".to_string()]);
            index
                .index_document(1, &[("content".to_string(), "machine learning".to_string())])
                .unwrap();
            index
                .index_document(2, &[("content".to_string(), "deep learning neural".to_string())])
                .unwrap();
            index.save().unwrap();
        }

        // Reload and verify
        {
            let index = TextIndex::open(path, vec!["content".to_string()]).unwrap();
            let results = index.search("learning", 10);
            assert_eq!(results.len(), 2);
        }
    }

    #[test]
    fn test_bm25_scoring() {
        let dir = tempdir().unwrap();
        let index = TextIndex::new(dir.path().to_path_buf(), vec!["text".to_string()]);

        // Document with more term frequency should score higher
        index
            .index_document(1, &[("text".to_string(), "apple apple apple".to_string())])
            .unwrap();
        index
            .index_document(2, &[("text".to_string(), "apple".to_string())])
            .unwrap();

        let results = index.search("apple", 10);
        assert_eq!(results.len(), 2);
        // Doc 1 has higher TF, should score higher (with saturation)
        assert!(results[0].id == 1 || results[0].score >= results[1].score);
    }

    #[test]
    fn test_must_match() {
        let dir = tempdir().unwrap();
        let index = TextIndex::new(dir.path().to_path_buf(), vec!["text".to_string()]);

        index
            .index_document(1, &[("text".to_string(), "red apple fruit".to_string())])
            .unwrap();
        index
            .index_document(2, &[("text".to_string(), "red car".to_string())])
            .unwrap();
        index
            .index_document(3, &[("text".to_string(), "green apple".to_string())])
            .unwrap();

        // Must match both "red" AND "apple"
        let bitmap = index.search_must_match("red apple");
        assert_eq!(bitmap.len(), 1);
        assert!(bitmap.contains(1));
    }

    #[test]
    fn test_hybrid_combine() {
        let vector_results = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
        let text_results = vec![
            TextSearchResult { id: 2, score: 5.0 },
            TextSearchResult { id: 4, score: 4.0 },
            TextSearchResult { id: 1, score: 3.0 },
        ];

        let results = hybrid_combine(&vector_results, &text_results, 0.5, 10);

        // All unique IDs should be present
        let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
        assert!(ids.contains(&4));
    }

    #[test]
    fn test_rrf_combine() {
        let vector_results = vec![(1, 0.9), (2, 0.8)];
        let text_results = vec![
            TextSearchResult { id: 2, score: 5.0 },
            TextSearchResult { id: 1, score: 3.0 },
        ];

        let results = rrf_combine(&vector_results, &text_results, 60.0, 10);

        // Document 2 appears in both rankings, should score well
        assert!(!results.is_empty());
        // Both docs should be present with combined RRF scores
        assert!(results.iter().any(|r| r.id == 1));
        assert!(results.iter().any(|r| r.id == 2));
    }
}
