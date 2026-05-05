//! GPU Acceleration for vector distance computation.
//!
//! Provides hardware-accelerated distance functions using available GPU backends.
//!
//! ## Supported Backends
//! - **Metal**: Apple Silicon GPUs (macOS/iOS)
//! - **CUDA**: NVIDIA GPUs
//! - **CPU**: SIMD fallback for all platforms
//!
//! ## Architecture
//! ```text
//! ┌─────────────────────────────────────┐
//! │         GpuAccelerator              │
//! │  (Runtime backend selection)        │
//! └──────────────┬──────────────────────┘
//!                │
//!       ┌────────┴────────┐
//!       ▼                 ▼
//! ┌───────────┐    ┌───────────┐
//! │   Metal   │    │   CUDA    │
//! │  Backend  │    │  Backend  │
//! └───────────┘    └───────────┘
//!       │                 │
//!       └────────┬────────┘
//!                ▼
//!       ┌───────────────┐
//!       │ CPU Fallback  │
//!       │   (SIMD)      │
//!       └───────────────┘
//! ```
//!
//! ## Usage
//! ```rust,ignore
//! use vectordb::gpu::{GpuAccelerator, GpuConfig};
//!
//! let gpu = GpuAccelerator::new(GpuConfig::default())?;
//!
//! // Batch distance computation
//! let distances = gpu.batch_l2_distance(&query, &vectors)?;
//!
//! // Matrix-vector multiplication (for projections)
//! let result = gpu.matrix_vector_mul(&matrix, &vector)?;
//! ```

use std::sync::Arc;
use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "candle")]
use candle_core::{Device, Tensor};

// =============================================================================
// GPU Configuration
// =============================================================================

/// GPU backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBackend {
    /// Apple Metal (macOS/iOS)
    Metal,
    /// NVIDIA CUDA
    Cuda,
    /// CPU fallback with SIMD
    Cpu,
    /// Automatic selection
    Auto,
}

impl Default for GpuBackend {
    fn default() -> Self {
        GpuBackend::Auto
    }
}

/// GPU acceleration configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Preferred backend
    pub backend: GpuBackend,
    /// Minimum batch size to use GPU (smaller batches use CPU)
    pub min_batch_size: usize,
    /// Maximum vectors to process in single GPU call
    pub max_batch_size: usize,
    /// Number of CPU threads for parallel processing
    pub cpu_threads: usize,
    /// Enable async GPU operations
    pub async_enabled: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::Auto,
            // GPU has CPU→GPU transfer overhead, so only use for larger batches.
            // With candle, 1000+ vectors is the break-even point.
            #[cfg(feature = "candle")]
            min_batch_size: 1000,
            #[cfg(not(feature = "candle"))]
            min_batch_size: 100,
            max_batch_size: 100_000,
            cpu_threads: num_cpus::get(),
            async_enabled: true,
        }
    }
}

impl GpuConfig {
    /// Create config for Metal backend.
    pub fn metal() -> Self {
        Self {
            backend: GpuBackend::Metal,
            ..Default::default()
        }
    }

    /// Create config for CUDA backend.
    pub fn cuda() -> Self {
        Self {
            backend: GpuBackend::Cuda,
            ..Default::default()
        }
    }

    /// Create config for CPU only.
    pub fn cpu_only() -> Self {
        Self {
            backend: GpuBackend::Cpu,
            ..Default::default()
        }
    }
}

// =============================================================================
// GPU Capabilities
// =============================================================================

/// Detected GPU capabilities.
#[derive(Debug, Clone, Serialize)]
pub struct GpuCapabilities {
    /// Available backends
    pub available_backends: Vec<GpuBackend>,
    /// Selected backend
    pub selected_backend: GpuBackend,
    /// GPU device name (if available)
    pub device_name: Option<String>,
    /// GPU memory in bytes (if available)
    pub memory_bytes: Option<u64>,
    /// Compute units/cores (if available)
    pub compute_units: Option<u32>,
    /// Maximum work group size
    pub max_workgroup_size: Option<u32>,
    /// SIMD width for CPU backend
    pub cpu_simd_width: u32,
}

impl GpuCapabilities {
    /// Detect available GPU capabilities.
    pub fn detect() -> Self {
        let mut available = vec![GpuBackend::Cpu];

        // Check for Metal (macOS only)
        #[cfg(target_os = "macos")]
        {
            available.push(GpuBackend::Metal);
        }

        // CUDA detection would require cuda-sys or similar
        // For now, we check environment variable
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
            available.push(GpuBackend::Cuda);
        }

        let selected = if available.contains(&GpuBackend::Metal) {
            GpuBackend::Metal
        } else if available.contains(&GpuBackend::Cuda) {
            GpuBackend::Cuda
        } else {
            GpuBackend::Cpu
        };

        // Detect SIMD width
        let simd_width = detect_simd_width();

        Self {
            available_backends: available,
            selected_backend: selected,
            device_name: detect_device_name(),
            memory_bytes: None,
            compute_units: None,
            max_workgroup_size: None,
            cpu_simd_width: simd_width,
        }
    }

    /// Check if GPU acceleration is available.
    pub fn has_gpu(&self) -> bool {
        self.available_backends.iter().any(|b| {
            matches!(b, GpuBackend::Metal | GpuBackend::Cuda)
        })
    }
}

fn detect_simd_width() -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return 16; // 512 bits / 32 bits
        }
        if is_x86_feature_detected!("avx2") {
            return 8; // 256 bits / 32 bits
        }
        if is_x86_feature_detected!("sse4.1") {
            return 4; // 128 bits / 32 bits
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return 4; // NEON: 128 bits / 32 bits
    }

    1 // Scalar
}

fn detect_device_name() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        // Would use Metal API to get device name
        return Some("Apple Silicon GPU".to_string());
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

// =============================================================================
// Distance Functions
// =============================================================================

/// Distance metric type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// L2 (Euclidean) squared distance
    L2Squared,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
    /// Dot product (inner product)
    DotProduct,
}

// =============================================================================
// GPU Accelerator
// =============================================================================

/// GPU-accelerated computation engine.
pub struct GpuAccelerator {
    config: GpuConfig,
    capabilities: GpuCapabilities,
}

impl GpuAccelerator {
    /// Create a new GPU accelerator.
    pub fn new(config: GpuConfig) -> Result<Self> {
        let capabilities = GpuCapabilities::detect();

        // Validate backend selection
        let selected = match config.backend {
            GpuBackend::Auto => capabilities.selected_backend,
            GpuBackend::Metal => {
                if capabilities.available_backends.contains(&GpuBackend::Metal) {
                    GpuBackend::Metal
                } else {
                    GpuBackend::Cpu
                }
            }
            GpuBackend::Cuda => {
                if capabilities.available_backends.contains(&GpuBackend::Cuda) {
                    GpuBackend::Cuda
                } else {
                    GpuBackend::Cpu
                }
            }
            GpuBackend::Cpu => GpuBackend::Cpu,
        };

        let mut caps = capabilities;
        caps.selected_backend = selected;

        Ok(Self {
            config,
            capabilities: caps,
        })
    }

    /// Get capabilities.
    pub fn capabilities(&self) -> &GpuCapabilities {
        &self.capabilities
    }

    /// Get selected backend.
    pub fn backend(&self) -> GpuBackend {
        self.capabilities.selected_backend
    }

    /// Compute L2 squared distance between query and all vectors.
    pub fn batch_l2_distance(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }

        // Use GPU for large batches, CPU for small
        if vectors.len() < self.config.min_batch_size {
            return Ok(self.cpu_batch_l2(query, vectors));
        }

        match self.capabilities.selected_backend {
            GpuBackend::Metal => self.metal_batch_l2(query, vectors),
            GpuBackend::Cuda => self.cuda_batch_l2(query, vectors),
            GpuBackend::Cpu | GpuBackend::Auto => Ok(self.cpu_batch_l2(query, vectors)),
        }
    }

    /// Compute dot product between query and all vectors.
    pub fn batch_dot_product(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }

        if vectors.len() < self.config.min_batch_size {
            return Ok(self.cpu_batch_dot(query, vectors));
        }

        match self.capabilities.selected_backend {
            GpuBackend::Metal => self.metal_batch_dot(query, vectors),
            GpuBackend::Cuda => self.cuda_batch_dot(query, vectors),
            GpuBackend::Cpu | GpuBackend::Auto => Ok(self.cpu_batch_dot(query, vectors)),
        }
    }

    /// Compute cosine distance between query and all vectors.
    pub fn batch_cosine_distance(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }

        if vectors.len() < self.config.min_batch_size {
            return Ok(self.cpu_batch_cosine(query, vectors));
        }

        match self.capabilities.selected_backend {
            GpuBackend::Metal => self.metal_batch_cosine(query, vectors),
            GpuBackend::Cuda => self.cuda_batch_cosine(query, vectors),
            GpuBackend::Cpu | GpuBackend::Auto => Ok(self.cpu_batch_cosine(query, vectors)),
        }
    }

    /// Generic batch distance computation.
    pub fn batch_distance(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        metric: DistanceMetric,
    ) -> Result<Vec<f32>> {
        match metric {
            DistanceMetric::L2Squared => self.batch_l2_distance(query, vectors),
            DistanceMetric::DotProduct => self.batch_dot_product(query, vectors),
            DistanceMetric::Cosine => self.batch_cosine_distance(query, vectors),
        }
    }

    /// Find k-nearest neighbors using GPU acceleration.
    pub fn knn_search(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<(usize, f32)>> {
        let distances = self.batch_distance(query, vectors, metric)?;

        // Find top-k
        let mut indexed: Vec<(usize, f32)> = distances
            .into_iter()
            .enumerate()
            .collect();

        // For dot product, higher is better (sort descending)
        // For L2 and cosine, lower is better (sort ascending)
        match metric {
            DistanceMetric::DotProduct => {
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }
            _ => {
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            }
        }

        indexed.truncate(k);
        Ok(indexed)
    }

    /// Matrix-vector multiplication (for projections).
    pub fn matrix_vector_mul(&self, matrix: &[Vec<f32>], vector: &[f32]) -> Result<Vec<f32>> {
        // matrix is [rows x cols], vector is [cols]
        // result is [rows]
        if matrix.is_empty() {
            return Ok(vec![]);
        }

        let cols = vector.len();
        if matrix[0].len() != cols {
            return Err(anyhow::anyhow!(
                "Dimension mismatch: matrix cols {} != vector len {}",
                matrix[0].len(),
                cols
            ));
        }

        // Use parallel CPU for now (GPU implementation would go here)
        let result: Vec<f32> = matrix
            .par_iter()
            .map(|row| dot_product(row, vector))
            .collect();

        Ok(result)
    }

    // =========================================================================
    // CPU Implementations (SIMD-optimized)
    // =========================================================================

    fn cpu_batch_l2(&self, query: &[f32], vectors: &[Vec<f32>]) -> Vec<f32> {
        vectors
            .par_iter()
            .map(|v| l2_squared(query, v))
            .collect()
    }

    fn cpu_batch_dot(&self, query: &[f32], vectors: &[Vec<f32>]) -> Vec<f32> {
        vectors
            .par_iter()
            .map(|v| dot_product(query, v))
            .collect()
    }

    fn cpu_batch_cosine(&self, query: &[f32], vectors: &[Vec<f32>]) -> Vec<f32> {
        let query_norm = l2_norm(query);
        vectors
            .par_iter()
            .map(|v| {
                let dot = dot_product(query, v);
                let v_norm = l2_norm(v);
                if query_norm > 1e-10 && v_norm > 1e-10 {
                    1.0 - (dot / (query_norm * v_norm))
                } else {
                    1.0
                }
            })
            .collect()
    }

    // =========================================================================
    // Candle GPU Implementations (Metal / CUDA via candle tensors)
    // =========================================================================

    #[cfg(feature = "candle")]
    fn candle_device(&self) -> Result<Device> {
        match self.capabilities.selected_backend {
            GpuBackend::Metal => {
                #[cfg(feature = "candle-metal")]
                {
                    Device::new_metal(0).map_err(|e| anyhow::anyhow!("Metal device: {}", e))
                }
                #[cfg(not(feature = "candle-metal"))]
                {
                    Ok(Device::Cpu)
                }
            }
            GpuBackend::Cuda => {
                #[cfg(feature = "candle-cuda")]
                {
                    Device::new_cuda(0).map_err(|e| anyhow::anyhow!("CUDA device: {}", e))
                }
                #[cfg(not(feature = "candle-cuda"))]
                {
                    Ok(Device::Cpu)
                }
            }
            _ => Ok(Device::Cpu),
        }
    }

    /// Flatten Vec<Vec<f32>> into a contiguous tensor [n, dim] on the target device.
    #[cfg(feature = "candle")]
    fn vectors_to_tensor(&self, vectors: &[Vec<f32>], device: &Device) -> Result<Tensor> {
        let n = vectors.len();
        let dim = vectors[0].len();
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        Tensor::from_vec(flat, (n, dim), device)
            .map_err(|e| anyhow::anyhow!("tensor from vectors: {}", e))
    }

    /// Candle-accelerated batch L2 squared distance.
    /// distance_i = ||q - v_i||² = ||q||² + ||v_i||² - 2·q·v_iᵀ
    #[cfg(feature = "candle")]
    fn candle_batch_l2(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        let device = self.candle_device()?;
        let dim = query.len();
        let n = vectors.len();

        let q = Tensor::from_vec(query.to_vec(), (1, dim), &device)?;
        let vecs = self.vectors_to_tensor(vectors, &device)?;

        // ||q||²  (scalar, broadcast)
        let q_sq = q.sqr()?.sum_keepdim(1)?; // [1, 1]

        // ||v_i||²  per row
        let v_sq = vecs.sqr()?.sum_keepdim(1)?; // [n, 1]

        // q · Vᵀ  →  [1, n]
        let dot = q.matmul(&vecs.t()?)?; // [1, n]

        // distances = ||q||² + ||v||² - 2·dot  →  broadcast to [1, n]
        let two = Tensor::from_vec(vec![2.0f32], 1, &device)?;
        let distances = v_sq
            .t()? // [1, n]
            .broadcast_add(&q_sq)? // [1, n]
            .broadcast_sub(&dot.broadcast_mul(&two)?)?; // [1, n]

        let result = distances.squeeze(0)?.to_vec1::<f32>()?;
        // Clamp negatives from floating point errors
        Ok(result.into_iter().map(|d| d.max(0.0)).collect())
    }

    /// Candle-accelerated batch dot product.
    #[cfg(feature = "candle")]
    fn candle_batch_dot(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        let device = self.candle_device()?;
        let dim = query.len();

        let q = Tensor::from_vec(query.to_vec(), (1, dim), &device)?;
        let vecs = self.vectors_to_tensor(vectors, &device)?;

        // q · Vᵀ  →  [1, n]
        let dots = q.matmul(&vecs.t()?)?;
        Ok(dots.squeeze(0)?.to_vec1::<f32>()?)
    }

    /// Candle-accelerated batch cosine distance.
    /// cosine_distance = 1 - (q·v) / (||q|| · ||v||)
    #[cfg(feature = "candle")]
    fn candle_batch_cosine(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        let device = self.candle_device()?;
        let dim = query.len();

        let q = Tensor::from_vec(query.to_vec(), (1, dim), &device)?;
        let vecs = self.vectors_to_tensor(vectors, &device)?;

        // Norms
        let q_norm = q.sqr()?.sum_keepdim(1)?.sqrt()?; // [1, 1]
        let v_norms = vecs.sqr()?.sum_keepdim(1)?.sqrt()?; // [n, 1]

        // Dot products
        let dots = q.matmul(&vecs.t()?)?; // [1, n]

        // Normalize: dots / (q_norm * v_norms^T)
        let norms_product = q_norm.broadcast_mul(&v_norms.t()?)?; // [1, n]
        let epsilon = Tensor::from_vec(vec![1e-10f32], 1, &device)?;
        let norms_clamped = norms_product.clamp(1e-10f32, f32::MAX)?;
        let similarities = dots.broadcast_div(&norms_clamped)?;

        // cosine_distance = 1 - similarity
        let ones = Tensor::ones_like(&similarities)?;
        let distances = ones.sub(&similarities)?;

        Ok(distances.squeeze(0)?.to_vec1::<f32>()?)
    }

    // =========================================================================
    // Metal Implementations
    // =========================================================================

    fn metal_batch_l2(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        #[cfg(feature = "candle")]
        {
            return self.candle_batch_l2(query, vectors);
        }
        #[cfg(not(feature = "candle"))]
        {
            Ok(self.cpu_batch_l2(query, vectors))
        }
    }

    fn metal_batch_dot(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        #[cfg(feature = "candle")]
        {
            return self.candle_batch_dot(query, vectors);
        }
        #[cfg(not(feature = "candle"))]
        {
            Ok(self.cpu_batch_dot(query, vectors))
        }
    }

    fn metal_batch_cosine(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        #[cfg(feature = "candle")]
        {
            return self.candle_batch_cosine(query, vectors);
        }
        #[cfg(not(feature = "candle"))]
        {
            Ok(self.cpu_batch_cosine(query, vectors))
        }
    }

    // =========================================================================
    // CUDA Implementations
    // =========================================================================

    fn cuda_batch_l2(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        #[cfg(feature = "candle")]
        {
            return self.candle_batch_l2(query, vectors);
        }
        #[cfg(not(feature = "candle"))]
        {
            Ok(self.cpu_batch_l2(query, vectors))
        }
    }

    fn cuda_batch_dot(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        #[cfg(feature = "candle")]
        {
            return self.candle_batch_dot(query, vectors);
        }
        #[cfg(not(feature = "candle"))]
        {
            Ok(self.cpu_batch_dot(query, vectors))
        }
    }

    fn cuda_batch_cosine(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        #[cfg(feature = "candle")]
        {
            return self.candle_batch_cosine(query, vectors);
        }
        #[cfg(not(feature = "candle"))]
        {
            Ok(self.cpu_batch_cosine(query, vectors))
        }
    }
}

// =============================================================================
// SIMD-Optimized Distance Functions
// =============================================================================

/// L2 squared distance.
#[inline]
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    // Use SIMD module if available
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { l2_squared_avx2(a, b) };
        }
    }

    // Scalar fallback
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

/// Dot product.
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_product_avx2(a, b) };
        }
    }

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm.
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// AVX2 implementations
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 floats at a time
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
        i += 8;
    }

    // Horizontal sum
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut total: f32 = result.iter().sum();

    // Handle remaining elements
    while i < n {
        let diff = a[i] - b[i];
        total += diff * diff;
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_fmadd_ps(va, vb, sum);
        i += 8;
    }

    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut total: f32 = result.iter().sum();

    while i < n {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

// =============================================================================
// Cached GPU Accelerator
// =============================================================================

/// GPU accelerator with pre-cached vectors tensor for fast repeated searches.
///
/// Unlike `GpuAccelerator` which transfers all vectors to GPU on every call,
/// `CachedGpuAccelerator` keeps vectors on GPU memory permanently. Only the
/// query vector (1 x dim) is transferred per search, eliminating the dominant
/// bottleneck of flatten + copy for the stored vectors.
#[cfg(feature = "candle")]
pub struct CachedGpuAccelerator {
    /// Vectors tensor [n, dim] on GPU device
    vectors_tensor: Tensor,
    /// Pre-computed ||v_i||² [n, 1]
    vectors_sq_norms: Tensor,
    /// Vectors transposed [dim, n] - cached to avoid repeated transpose
    vectors_t: Tensor,
    /// Device (Metal/CUDA/CPU)
    device: Device,
    /// Vector count
    n: usize,
    /// Vector dimension
    dim: usize,
}

// Tensor is Send+Sync in candle, so CachedGpuAccelerator is too.
#[cfg(feature = "candle")]
unsafe impl Send for CachedGpuAccelerator {}
#[cfg(feature = "candle")]
unsafe impl Sync for CachedGpuAccelerator {}

#[cfg(feature = "candle")]
impl CachedGpuAccelerator {
    /// Resolve the best available device (Metal > CUDA > CPU).
    fn resolve_device() -> Device {
        #[cfg(feature = "candle-metal")]
        {
            if let Ok(d) = Device::new_metal(0) {
                return d;
            }
        }
        #[cfg(feature = "candle-cuda")]
        {
            if let Ok(d) = Device::new_cuda(0) {
                return d;
            }
        }
        Device::Cpu
    }

    /// Create from vectors using the best available device.
    /// Transfers all data to GPU once.
    pub fn new(vectors: &[Vec<f32>]) -> Result<Self> {
        let device = Self::resolve_device();
        Self::with_device(vectors, device)
    }

    /// Create with explicit device selection.
    pub fn with_device(vectors: &[Vec<f32>], device: Device) -> Result<Self> {
        if vectors.is_empty() {
            return Err(anyhow::anyhow!("Cannot create CachedGpuAccelerator with empty vectors"));
        }

        let n = vectors.len();
        let dim = vectors[0].len();

        // Flatten Vec<Vec<f32>> into contiguous buffer and transfer to device once
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let vectors_tensor = Tensor::from_vec(flat, (n, dim), &device)
            .map_err(|e| anyhow::anyhow!("Failed to create vectors tensor: {}", e))?;

        // Pre-compute ||v_i||² [n, 1]
        let vectors_sq_norms = vectors_tensor
            .sqr()
            .and_then(|t| t.sum_keepdim(1))
            .map_err(|e| anyhow::anyhow!("Failed to compute squared norms: {}", e))?;

        // Pre-compute transpose [dim, n]
        let vectors_t = vectors_tensor
            .t()
            .map_err(|e| anyhow::anyhow!("Failed to transpose vectors: {}", e))?;

        Ok(Self {
            vectors_tensor,
            vectors_sq_norms,
            vectors_t,
            device,
            n,
            dim,
        })
    }

    /// Batch L2 squared distance. Only query is transferred per call.
    ///
    /// dist_i = ||q||² + ||v_i||² - 2·q·v_iᵀ
    pub fn batch_l2_distance(&self, query: &[f32]) -> Result<Vec<f32>> {
        if query.len() != self.dim {
            return Err(anyhow::anyhow!(
                "Query dimension {} != cached dimension {}",
                query.len(),
                self.dim
            ));
        }

        let q = Tensor::from_vec(query.to_vec(), (1, self.dim), &self.device)
            .map_err(|e| anyhow::anyhow!("query tensor: {}", e))?;

        // ||q||² [1, 1]
        let q_sq = q.sqr()?.sum_keepdim(1)?;

        // q · Vᵀ → [1, n]  (uses cached transpose)
        let dot = q.matmul(&self.vectors_t)?;

        // distances = ||v||²ᵀ + ||q||² - 2·dot → [1, n]
        let two = Tensor::from_vec(vec![2.0f32], 1, &self.device)?;
        let distances = self
            .vectors_sq_norms
            .t()?                           // [1, n]
            .broadcast_add(&q_sq)?          // [1, n]
            .broadcast_sub(&dot.broadcast_mul(&two)?)?; // [1, n]

        let result = distances.squeeze(0)?.to_vec1::<f32>()?;
        Ok(result.into_iter().map(|d| d.max(0.0)).collect())
    }

    /// Batch dot product. Only query is transferred per call.
    pub fn batch_dot_product(&self, query: &[f32]) -> Result<Vec<f32>> {
        if query.len() != self.dim {
            return Err(anyhow::anyhow!(
                "Query dimension {} != cached dimension {}",
                query.len(),
                self.dim
            ));
        }

        let q = Tensor::from_vec(query.to_vec(), (1, self.dim), &self.device)
            .map_err(|e| anyhow::anyhow!("query tensor: {}", e))?;

        // q · Vᵀ → [1, n]
        let dots = q.matmul(&self.vectors_t)?;
        Ok(dots.squeeze(0)?.to_vec1::<f32>()?)
    }

    /// Batch cosine distance. Only query is transferred per call.
    ///
    /// cosine_distance = 1 - (q·v) / (||q|| · ||v||)
    pub fn batch_cosine_distance(&self, query: &[f32]) -> Result<Vec<f32>> {
        if query.len() != self.dim {
            return Err(anyhow::anyhow!(
                "Query dimension {} != cached dimension {}",
                query.len(),
                self.dim
            ));
        }

        let q = Tensor::from_vec(query.to_vec(), (1, self.dim), &self.device)
            .map_err(|e| anyhow::anyhow!("query tensor: {}", e))?;

        // ||q|| [1, 1]
        let q_norm = q.sqr()?.sum_keepdim(1)?.sqrt()?;

        // ||v_i|| [n, 1] — derived from cached squared norms
        let v_norms = self.vectors_sq_norms.sqrt()?;

        // q · Vᵀ → [1, n]
        let dots = q.matmul(&self.vectors_t)?;

        // norm products [1, n]
        let norms_product = q_norm.broadcast_mul(&v_norms.t()?)?;
        let norms_clamped = norms_product.clamp(1e-10f32, f32::MAX)?;
        let similarities = dots.broadcast_div(&norms_clamped)?;

        // cosine_distance = 1 - similarity
        let ones = Tensor::ones_like(&similarities)?;
        let distances = ones.sub(&similarities)?;

        Ok(distances.squeeze(0)?.to_vec1::<f32>()?)
    }

    /// K-NN search using cached vectors.
    pub fn knn_search(
        &self,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<(usize, f32)>> {
        let distances = match metric {
            DistanceMetric::L2Squared => self.batch_l2_distance(query)?,
            DistanceMetric::DotProduct => self.batch_dot_product(query)?,
            DistanceMetric::Cosine => self.batch_cosine_distance(query)?,
        };

        let mut indexed: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();

        match metric {
            DistanceMetric::DotProduct => {
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }
            _ => {
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            }
        }

        indexed.truncate(k);
        Ok(indexed)
    }

    /// Number of cached vectors.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

// =============================================================================
// Batch Processor
// =============================================================================

/// Batch processor for streaming GPU operations.
pub struct BatchProcessor {
    accelerator: Arc<GpuAccelerator>,
    batch_size: usize,
}

impl BatchProcessor {
    pub fn new(accelerator: Arc<GpuAccelerator>) -> Self {
        Self {
            batch_size: accelerator.config.max_batch_size,
            accelerator,
        }
    }

    /// Process vectors in batches.
    pub fn process_batches<F, T>(&self, vectors: &[Vec<f32>], mut f: F) -> Vec<T>
    where
        F: FnMut(&[Vec<f32>]) -> Vec<T>,
        T: Send,
    {
        let mut results = Vec::with_capacity(vectors.len());

        for chunk in vectors.chunks(self.batch_size) {
            results.extend(f(chunk));
        }

        results
    }

    /// Batch k-NN search.
    pub fn batch_knn(
        &self,
        queries: &[Vec<f32>],
        vectors: &[Vec<f32>],
        k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<Vec<(usize, f32)>>> {
        queries
            .par_iter()
            .map(|q| self.accelerator.knn_search(q, vectors, k, metric))
            .collect()
    }
}

// =============================================================================
// Thread-Safe Wrapper
// =============================================================================

/// Thread-safe GPU accelerator wrapper.
#[derive(Clone)]
pub struct ThreadSafeGpu {
    inner: Arc<GpuAccelerator>,
}

impl ThreadSafeGpu {
    pub fn new(config: GpuConfig) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(GpuAccelerator::new(config)?),
        })
    }

    pub fn capabilities(&self) -> &GpuCapabilities {
        self.inner.capabilities()
    }

    pub fn backend(&self) -> GpuBackend {
        self.inner.backend()
    }

    pub fn batch_l2_distance(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        self.inner.batch_l2_distance(query, vectors)
    }

    pub fn batch_dot_product(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        self.inner.batch_dot_product(query, vectors)
    }

    pub fn batch_cosine_distance(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        self.inner.batch_cosine_distance(query, vectors)
    }

    pub fn knn_search(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<(usize, f32)>> {
        self.inner.knn_search(query, vectors, k, metric)
    }

    pub fn batch_processor(&self) -> BatchProcessor {
        BatchProcessor::new(Arc::clone(&self.inner))
    }
}

// =============================================================================
// num_cpus helper
// =============================================================================

mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn test_capabilities_detection() {
        let caps = GpuCapabilities::detect();
        assert!(!caps.available_backends.is_empty());
        assert!(caps.available_backends.contains(&GpuBackend::Cpu));
        assert!(caps.cpu_simd_width >= 1);
    }

    #[test]
    fn test_accelerator_creation() {
        let config = GpuConfig::cpu_only();
        let gpu = GpuAccelerator::new(config).unwrap();
        assert_eq!(gpu.backend(), GpuBackend::Cpu);
    }

    #[test]
    fn test_batch_l2_distance() {
        let config = GpuConfig::cpu_only();
        let gpu = GpuAccelerator::new(config).unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],  // distance = 0
            vec![0.0, 1.0, 0.0, 0.0],  // distance = 2
            vec![2.0, 0.0, 0.0, 0.0],  // distance = 1
        ];

        let distances = gpu.batch_l2_distance(&query, &vectors).unwrap();
        assert_eq!(distances.len(), 3);
        assert!((distances[0] - 0.0).abs() < 1e-6);
        assert!((distances[1] - 2.0).abs() < 1e-6);
        assert!((distances[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_dot_product() {
        let config = GpuConfig::cpu_only();
        let gpu = GpuAccelerator::new(config).unwrap();

        let query = vec![1.0, 2.0, 3.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0],  // dot = 1
            vec![0.0, 1.0, 0.0],  // dot = 2
            vec![0.0, 0.0, 1.0],  // dot = 3
        ];

        let dots = gpu.batch_dot_product(&query, &vectors).unwrap();
        assert_eq!(dots.len(), 3);
        assert!((dots[0] - 1.0).abs() < 1e-6);
        assert!((dots[1] - 2.0).abs() < 1e-6);
        assert!((dots[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_knn_search() {
        let config = GpuConfig::cpu_only();
        let gpu = GpuAccelerator::new(config).unwrap();

        let query = vec![0.0; 8];
        let vectors = random_vectors(100, 8);

        let results = gpu.knn_search(&query, &vectors, 5, DistanceMetric::L2Squared).unwrap();
        assert_eq!(results.len(), 5);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_batch_cosine() {
        let config = GpuConfig::cpu_only();
        let gpu = GpuAccelerator::new(config).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0],   // cosine distance = 0
            vec![-1.0, 0.0, 0.0],  // cosine distance = 2
            vec![0.0, 1.0, 0.0],   // cosine distance = 1
        ];

        let distances = gpu.batch_cosine_distance(&query, &vectors).unwrap();
        assert_eq!(distances.len(), 3);
        assert!(distances[0] < 0.01); // Nearly 0
        assert!((distances[1] - 2.0).abs() < 0.01); // Nearly 2
        assert!((distances[2] - 1.0).abs() < 0.01); // Nearly 1
    }

    #[test]
    fn test_thread_safe_wrapper() {
        let gpu = ThreadSafeGpu::new(GpuConfig::cpu_only()).unwrap();

        let query = vec![1.0, 2.0, 3.0, 4.0];
        let vectors = random_vectors(50, 4);

        let results = gpu.knn_search(&query, &vectors, 3, DistanceMetric::L2Squared).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_large_batch() {
        let config = GpuConfig {
            min_batch_size: 10,
            ..GpuConfig::cpu_only()
        };
        let gpu = GpuAccelerator::new(config).unwrap();

        let query = vec![0.0; 128];
        let vectors = random_vectors(1000, 128);

        let distances = gpu.batch_l2_distance(&query, &vectors).unwrap();
        assert_eq!(distances.len(), 1000);
    }

    // =========================================================================
    // CachedGpuAccelerator tests
    // =========================================================================

    #[cfg(feature = "candle")]
    mod cached_tests {
        use super::*;

        #[test]
        fn test_cached_creation() {
            let vectors = vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ];
            let cached = CachedGpuAccelerator::new(&vectors).unwrap();
            assert_eq!(cached.len(), 3);
            assert_eq!(cached.dim(), 3);
            assert!(!cached.is_empty());
        }

        #[test]
        fn test_cached_empty_vectors_error() {
            let vectors: Vec<Vec<f32>> = vec![];
            assert!(CachedGpuAccelerator::new(&vectors).is_err());
        }

        #[test]
        fn test_cached_dimension_mismatch_error() {
            let vectors = vec![vec![1.0, 0.0, 0.0]];
            let cached = CachedGpuAccelerator::new(&vectors).unwrap();
            // Wrong dimension query
            assert!(cached.batch_l2_distance(&[1.0, 2.0]).is_err());
            assert!(cached.batch_dot_product(&[1.0, 2.0]).is_err());
            assert!(cached.batch_cosine_distance(&[1.0, 2.0]).is_err());
        }

        #[test]
        fn test_cached_l2_distance() {
            let vectors = vec![
                vec![1.0, 0.0, 0.0, 0.0],  // distance from [1,0,0,0] = 0
                vec![0.0, 1.0, 0.0, 0.0],  // distance = 2
                vec![2.0, 0.0, 0.0, 0.0],  // distance = 1
            ];
            let cached = CachedGpuAccelerator::with_device(&vectors, Device::Cpu).unwrap();
            let query = vec![1.0, 0.0, 0.0, 0.0];
            let distances = cached.batch_l2_distance(&query).unwrap();

            assert_eq!(distances.len(), 3);
            assert!((distances[0] - 0.0).abs() < 1e-5);
            assert!((distances[1] - 2.0).abs() < 1e-5);
            assert!((distances[2] - 1.0).abs() < 1e-5);
        }

        #[test]
        fn test_cached_dot_product() {
            let vectors = vec![
                vec![1.0, 0.0, 0.0],  // dot with [1,2,3] = 1
                vec![0.0, 1.0, 0.0],  // dot = 2
                vec![0.0, 0.0, 1.0],  // dot = 3
            ];
            let cached = CachedGpuAccelerator::with_device(&vectors, Device::Cpu).unwrap();
            let query = vec![1.0, 2.0, 3.0];
            let dots = cached.batch_dot_product(&query).unwrap();

            assert_eq!(dots.len(), 3);
            assert!((dots[0] - 1.0).abs() < 1e-5);
            assert!((dots[1] - 2.0).abs() < 1e-5);
            assert!((dots[2] - 3.0).abs() < 1e-5);
        }

        #[test]
        fn test_cached_cosine_distance() {
            let vectors = vec![
                vec![1.0, 0.0, 0.0],   // cosine distance from [1,0,0] = 0
                vec![-1.0, 0.0, 0.0],  // cosine distance = 2
                vec![0.0, 1.0, 0.0],   // cosine distance = 1
            ];
            let cached = CachedGpuAccelerator::with_device(&vectors, Device::Cpu).unwrap();
            let query = vec![1.0, 0.0, 0.0];
            let distances = cached.batch_cosine_distance(&query).unwrap();

            assert_eq!(distances.len(), 3);
            assert!(distances[0].abs() < 0.01);
            assert!((distances[1] - 2.0).abs() < 0.01);
            assert!((distances[2] - 1.0).abs() < 0.01);
        }

        #[test]
        fn test_cached_knn_search() {
            let vectors = random_vectors(100, 8);
            let cached = CachedGpuAccelerator::with_device(&vectors, Device::Cpu).unwrap();
            let query = vec![0.0; 8];

            let results = cached.knn_search(&query, 5, DistanceMetric::L2Squared).unwrap();
            assert_eq!(results.len(), 5);

            // Results should be sorted by distance (ascending for L2)
            for i in 1..results.len() {
                assert!(results[i - 1].1 <= results[i].1);
            }
        }

        #[test]
        fn test_cached_repeated_queries() {
            // The whole point: multiple queries against the same cached vectors
            let vectors = random_vectors(500, 32);
            let cached = CachedGpuAccelerator::with_device(&vectors, Device::Cpu).unwrap();

            for _ in 0..10 {
                let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
                let distances = cached.batch_l2_distance(&query).unwrap();
                assert_eq!(distances.len(), 500);
            }
        }

        #[test]
        fn test_cached_matches_uncached() {
            // Verify cached and uncached produce the same results
            let vectors = random_vectors(200, 16);
            let query: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();

            let gpu = GpuAccelerator::new(GpuConfig::cpu_only()).unwrap();
            let uncached_l2 = gpu.batch_l2_distance(&query, &vectors).unwrap();

            let cached = CachedGpuAccelerator::with_device(&vectors, Device::Cpu).unwrap();
            let cached_l2 = cached.batch_l2_distance(&query).unwrap();

            assert_eq!(uncached_l2.len(), cached_l2.len());
            for (u, c) in uncached_l2.iter().zip(cached_l2.iter()) {
                assert!((u - c).abs() < 1e-4, "uncached={} cached={}", u, c);
            }
        }
    }
}
