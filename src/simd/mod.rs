//! SIMD-accelerated distance functions for vector similarity search.
//!
//! Provides optimized implementations using:
//! - AVX2 (256-bit, 8 floats at once) - ~4-8x speedup
//! - SSE4.1 (128-bit, 4 floats at once) - ~2-4x speedup
//! - Scalar fallback for unsupported CPUs
//!
//! Runtime feature detection ensures the best available implementation is used.

use std::sync::atomic::{AtomicU8, Ordering};

/// SIMD implementation level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SimdLevel {
    /// No SIMD, scalar operations only
    Scalar = 0,
    /// SSE4.1 (128-bit vectors)
    Sse41 = 1,
    /// AVX2 (256-bit vectors)
    Avx2 = 2,
    /// AVX-512 (512-bit vectors) - future
    Avx512 = 3,
}

impl SimdLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            SimdLevel::Scalar => "scalar",
            SimdLevel::Sse41 => "sse4.1",
            SimdLevel::Avx2 => "avx2",
            SimdLevel::Avx512 => "avx512",
        }
    }
}

/// Cached SIMD level (0 = not detected yet)
static SIMD_LEVEL: AtomicU8 = AtomicU8::new(255);

/// Detect the best available SIMD level at runtime.
#[inline]
pub fn detect_simd_level() -> SimdLevel {
    let cached = SIMD_LEVEL.load(Ordering::Relaxed);
    if cached != 255 {
        return match cached {
            0 => SimdLevel::Scalar,
            1 => SimdLevel::Sse41,
            2 => SimdLevel::Avx2,
            3 => SimdLevel::Avx512,
            _ => SimdLevel::Scalar,
        };
    }

    let level = detect_simd_level_impl();
    SIMD_LEVEL.store(level as u8, Ordering::Relaxed);
    level
}

#[cfg(target_arch = "x86_64")]
fn detect_simd_level_impl() -> SimdLevel {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        SimdLevel::Avx2
    } else if is_x86_feature_detected!("sse4.1") {
        SimdLevel::Sse41
    } else {
        SimdLevel::Scalar
    }
}

#[cfg(target_arch = "aarch64")]
fn detect_simd_level_impl() -> SimdLevel {
    // ARM NEON is always available on aarch64
    SimdLevel::Sse41 // Use SSE level as NEON equivalent
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_simd_level_impl() -> SimdLevel {
    SimdLevel::Scalar
}

/// Get SIMD statistics for diagnostics.
pub fn simd_info() -> SimdInfo {
    SimdInfo {
        level: detect_simd_level(),
        l2_impl: detect_simd_level().as_str(),
        dot_impl: detect_simd_level().as_str(),
    }
}

#[derive(Debug, Clone)]
pub struct SimdInfo {
    pub level: SimdLevel,
    pub l2_impl: &'static str,
    pub dot_impl: &'static str,
}

// =============================================================================
// L2 (Euclidean) Squared Distance
// =============================================================================

/// Compute L2 squared distance between two vectors.
/// Automatically uses the best available SIMD implementation.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector length mismatch");

    match detect_simd_level() {
        SimdLevel::Avx2 => l2_squared_avx2(a, b),
        SimdLevel::Sse41 => l2_squared_sse41(a, b),
        _ => l2_squared_scalar(a, b),
    }
}

/// Scalar implementation of L2 squared distance.
#[inline]
pub fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// SSE4.1 implementation of L2 squared distance.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn l2_squared_sse41(a: &[f32], b: &[f32]) -> f32 {
    if !is_x86_feature_detected!("sse4.1") {
        return l2_squared_scalar(a, b);
    }
    unsafe { l2_squared_sse41_impl(a, b) }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn l2_squared_sse41(a: &[f32], b: &[f32]) -> f32 {
    l2_squared_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn l2_squared_sse41_impl(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    // Process 4 floats at a time
    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let diff = _mm_sub_ps(va, vb);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        i += 4;
    }

    // Horizontal sum
    let mut result = horizontal_sum_sse(sum);

    // Handle remaining elements
    while i < len {
        let d = a[i] - b[i];
        result += d * d;
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn horizontal_sum_sse(v: std::arch::x86_64::__m128) -> f32 {
    use std::arch::x86_64::*;
    let shuf = _mm_movehdup_ps(v);
    let sums = _mm_add_ps(v, shuf);
    let shuf = _mm_movehl_ps(sums, sums);
    let sums = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(sums)
}

/// AVX2 implementation of L2 squared distance.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    if !is_x86_feature_detected!("avx2") {
        return l2_squared_sse41(a, b);
    }
    unsafe { l2_squared_avx2_impl(a, b) }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    l2_squared_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_squared_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 floats at a time
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        // FMA: sum = sum + diff * diff
        sum = _mm256_fmadd_ps(diff, diff, sum);
        i += 8;
    }

    // Horizontal sum (reduce 8 floats to 1)
    let mut result = horizontal_sum_avx(sum);

    // Handle remaining elements
    while i < len {
        let d = a[i] - b[i];
        result += d * d;
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // Extract high 128 bits and add to low 128 bits
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(low, high);
    // Now do SSE horizontal sum
    horizontal_sum_sse(sum128)
}

// =============================================================================
// Dot Product
// =============================================================================

/// Compute dot product of two vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector length mismatch");

    match detect_simd_level() {
        SimdLevel::Avx2 => dot_product_avx2(a, b),
        SimdLevel::Sse41 => dot_product_sse41(a, b),
        _ => dot_product_scalar(a, b),
    }
}

/// Scalar implementation of dot product.
#[inline]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot_product_sse41(a: &[f32], b: &[f32]) -> f32 {
    if !is_x86_feature_detected!("sse4.1") {
        return dot_product_scalar(a, b);
    }
    unsafe { dot_product_sse41_impl(a, b) }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn dot_product_sse41(a: &[f32], b: &[f32]) -> f32 {
    dot_product_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dot_product_sse41_impl(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
        i += 4;
    }

    let mut result = horizontal_sum_sse(sum);

    while i < len {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    if !is_x86_feature_detected!("avx2") {
        return dot_product_sse41(a, b);
    }
    unsafe { dot_product_avx2_impl(a, b) }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    dot_product_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_product_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_fmadd_ps(va, vb, sum);
        i += 8;
    }

    let mut result = horizontal_sum_avx(sum);

    while i < len {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

// =============================================================================
// Cosine Similarity
// =============================================================================

/// Compute cosine similarity between two vectors.
/// Returns 1 - cosine_similarity (distance form, lower is more similar).
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = dot_product(a, a).sqrt();
    let norm_b = dot_product(b, b).sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 1.0;
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Compute cosine similarity (not distance).
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_distance(a, b)
}

// =============================================================================
// Vector Normalization
// =============================================================================

/// Normalize a vector to unit length (in-place).
#[inline]
pub fn normalize_inplace(v: &mut [f32]) {
    let norm = dot_product(v, v).sqrt();
    if norm > 1e-10 {
        let inv_norm = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv_norm;
        }
    }
}

/// Normalize a vector to unit length (returns new vector).
#[inline]
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = dot_product(v, v).sqrt();
    if norm > 1e-10 {
        let inv_norm = 1.0 / norm;
        v.iter().map(|&x| x * inv_norm).collect()
    } else {
        v.to_vec()
    }
}

// =============================================================================
// Quantized Vector Distance (uint8)
// =============================================================================

/// Compute L2 squared distance between two uint8 quantized vectors.
/// Uses integer SIMD for faster computation.
#[inline]
pub fn l2_squared_u8(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len(), "Vector length mismatch");

    match detect_simd_level() {
        SimdLevel::Avx2 => l2_squared_u8_avx2(a, b),
        SimdLevel::Sse41 => l2_squared_u8_sse41(a, b),
        _ => l2_squared_u8_scalar(a, b),
    }
}

/// Scalar implementation for uint8 L2 distance.
#[inline]
pub fn l2_squared_u8_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = (x as i32) - (y as i32);
            (d * d) as u32
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn l2_squared_u8_sse41(a: &[u8], b: &[u8]) -> u32 {
    if !is_x86_feature_detected!("sse4.1") {
        return l2_squared_u8_scalar(a, b);
    }
    unsafe { l2_squared_u8_sse41_impl(a, b) }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn l2_squared_u8_sse41(a: &[u8], b: &[u8]) -> u32 {
    l2_squared_u8_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn l2_squared_u8_sse41_impl(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm_setzero_si128();
    let mut i = 0;

    // Process 16 bytes at a time using SAD (Sum of Absolute Differences)
    // Note: This computes |a-b|, not (a-b)^2, so we need a different approach
    // For squared distance, we'll use 8 bytes at a time with proper conversion

    while i + 8 <= len {
        // Load 8 bytes and zero-extend to 16-bit
        let va = _mm_cvtepu8_epi16(_mm_loadl_epi64(a.as_ptr().add(i) as *const _));
        let vb = _mm_cvtepu8_epi16(_mm_loadl_epi64(b.as_ptr().add(i) as *const _));

        // Compute difference (signed 16-bit)
        let diff = _mm_sub_epi16(va, vb);

        // Square: multiply low 4 and high 4 separately, then add
        let sq = _mm_madd_epi16(diff, diff);
        sum = _mm_add_epi32(sum, sq);

        i += 8;
    }

    // Horizontal sum of 4 i32
    let shuf = _mm_shuffle_epi32(sum, 0b11_10_11_10);
    let sum = _mm_add_epi32(sum, shuf);
    let shuf = _mm_shuffle_epi32(sum, 0b01_01_01_01);
    let sum = _mm_add_epi32(sum, shuf);
    let mut result = _mm_cvtsi128_si32(sum) as u32;

    // Handle remaining
    while i < len {
        let d = (a[i] as i32) - (b[i] as i32);
        result += (d * d) as u32;
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn l2_squared_u8_avx2(a: &[u8], b: &[u8]) -> u32 {
    if !is_x86_feature_detected!("avx2") {
        return l2_squared_u8_sse41(a, b);
    }
    unsafe { l2_squared_u8_avx2_impl(a, b) }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn l2_squared_u8_avx2(a: &[u8], b: &[u8]) -> u32 {
    l2_squared_u8_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_squared_u8_avx2_impl(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm256_setzero_si256();
    let mut i = 0;

    // Process 16 bytes at a time
    while i + 16 <= len {
        // Load 16 bytes and zero-extend to 16-bit (using 256-bit register)
        let va_128 = _mm_loadu_si128(a.as_ptr().add(i) as *const _);
        let vb_128 = _mm_loadu_si128(b.as_ptr().add(i) as *const _);

        let va = _mm256_cvtepu8_epi16(va_128);
        let vb = _mm256_cvtepu8_epi16(vb_128);

        let diff = _mm256_sub_epi16(va, vb);
        let sq = _mm256_madd_epi16(diff, diff);
        sum = _mm256_add_epi32(sum, sq);

        i += 16;
    }

    // Horizontal sum
    let high = _mm256_extracti128_si256(sum, 1);
    let low = _mm256_castsi256_si128(sum);
    let sum128 = _mm_add_epi32(low, high);

    let shuf = _mm_shuffle_epi32(sum128, 0b11_10_11_10);
    let sum128 = _mm_add_epi32(sum128, shuf);
    let shuf = _mm_shuffle_epi32(sum128, 0b01_01_01_01);
    let sum128 = _mm_add_epi32(sum128, shuf);
    let mut result = _mm_cvtsi128_si32(sum128) as u32;

    // Handle remaining
    while i < len {
        let d = (a[i] as i32) - (b[i] as i32);
        result += (d * d) as u32;
        i += 1;
    }

    result
}

// =============================================================================
// Dot Product for uint8 vectors
// =============================================================================

/// Compute dot product of two uint8 vectors.
#[inline]
pub fn dot_product_u8(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len(), "Vector length mismatch");

    match detect_simd_level() {
        SimdLevel::Avx2 => dot_product_u8_avx2(a, b),
        SimdLevel::Sse41 => dot_product_u8_sse41(a, b),
        _ => dot_product_u8_scalar(a, b),
    }
}

#[inline]
pub fn dot_product_u8_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as u32) * (y as u32))
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot_product_u8_sse41(a: &[u8], b: &[u8]) -> u32 {
    if !is_x86_feature_detected!("sse4.1") {
        return dot_product_u8_scalar(a, b);
    }
    unsafe { dot_product_u8_sse41_impl(a, b) }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn dot_product_u8_sse41(a: &[u8], b: &[u8]) -> u32 {
    dot_product_u8_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dot_product_u8_sse41_impl(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm_setzero_si128();
    let mut i = 0;

    while i + 8 <= len {
        let va = _mm_cvtepu8_epi16(_mm_loadl_epi64(a.as_ptr().add(i) as *const _));
        let vb = _mm_cvtepu8_epi16(_mm_loadl_epi64(b.as_ptr().add(i) as *const _));
        let prod = _mm_madd_epi16(va, vb);
        sum = _mm_add_epi32(sum, prod);
        i += 8;
    }

    // Horizontal sum
    let shuf = _mm_shuffle_epi32(sum, 0b11_10_11_10);
    let sum = _mm_add_epi32(sum, shuf);
    let shuf = _mm_shuffle_epi32(sum, 0b01_01_01_01);
    let sum = _mm_add_epi32(sum, shuf);
    let mut result = _mm_cvtsi128_si32(sum) as u32;

    while i < len {
        result += (a[i] as u32) * (b[i] as u32);
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot_product_u8_avx2(a: &[u8], b: &[u8]) -> u32 {
    if !is_x86_feature_detected!("avx2") {
        return dot_product_u8_sse41(a, b);
    }
    unsafe { dot_product_u8_avx2_impl(a, b) }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn dot_product_u8_avx2(a: &[u8], b: &[u8]) -> u32 {
    dot_product_u8_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_u8_avx2_impl(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm256_setzero_si256();
    let mut i = 0;

    while i + 16 <= len {
        let va_128 = _mm_loadu_si128(a.as_ptr().add(i) as *const _);
        let vb_128 = _mm_loadu_si128(b.as_ptr().add(i) as *const _);

        let va = _mm256_cvtepu8_epi16(va_128);
        let vb = _mm256_cvtepu8_epi16(vb_128);

        let prod = _mm256_madd_epi16(va, vb);
        sum = _mm256_add_epi32(sum, prod);

        i += 16;
    }

    // Horizontal sum
    let high = _mm256_extracti128_si256(sum, 1);
    let low = _mm256_castsi256_si128(sum);
    let sum128 = _mm_add_epi32(low, high);

    let shuf = _mm_shuffle_epi32(sum128, 0b11_10_11_10);
    let sum128 = _mm_add_epi32(sum128, shuf);
    let shuf = _mm_shuffle_epi32(sum128, 0b01_01_01_01);
    let sum128 = _mm_add_epi32(sum128, shuf);
    let mut result = _mm_cvtsi128_si32(sum128) as u32;

    while i < len {
        result += (a[i] as u32) * (b[i] as u32);
        i += 1;
    }

    result
}

// =============================================================================
// ARM NEON implementations (for Apple Silicon and ARM servers)
// =============================================================================

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    #[inline]
    pub unsafe fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut sum = vdupq_n_f32(0.0);
        let mut i = 0;

        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let diff = vsubq_f32(va, vb);
            sum = vfmaq_f32(sum, diff, diff);
            i += 4;
        }

        let mut result = vaddvq_f32(sum);

        while i < len {
            let d = a[i] - b[i];
            result += d * d;
            i += 1;
        }

        result
    }

    #[inline]
    pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut sum = vdupq_n_f32(0.0);
        let mut i = 0;

        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            sum = vfmaq_f32(sum, va, vb);
            i += 4;
        }

        let mut result = vaddvq_f32(sum);

        while i < len {
            result += a[i] * b[i];
            i += 1;
        }

        result
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    unsafe { neon::l2_squared_neon(a, b) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    unsafe { neon::dot_product_neon(a, b) }
}

// Override for aarch64 to use NEON
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn l2_squared_fast(a: &[f32], b: &[f32]) -> f32 {
    l2_squared_neon(a, b)
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dot_product_fast(a: &[f32], b: &[f32]) -> f32 {
    dot_product_neon(a, b)
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn l2_squared_fast(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b)
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn dot_product_fast(a: &[f32], b: &[f32]) -> f32 {
    dot_product(a, b)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn test_l2_squared() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let scalar = l2_squared_scalar(&a, &b);
        let result = l2_squared(&a, &b);

        assert!(approx_eq(scalar, 8.0)); // 8 * 1^2 = 8
        assert!(approx_eq(result, scalar));
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let scalar = dot_product_scalar(&a, &b);
        let result = dot_product(&a, &b);

        assert!(approx_eq(scalar, 40.0)); // 2+6+12+20 = 40
        assert!(approx_eq(result, scalar));
    }

    #[test]
    fn test_cosine() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];

        let sim = cosine_similarity(&a, &b);
        assert!(approx_eq(sim, 0.0)); // Orthogonal vectors

        let c = vec![1.0, 1.0, 0.0, 0.0];
        let sim2 = cosine_similarity(&a, &c);
        assert!(approx_eq(sim2, 0.7071068)); // ~0.707 = 1/sqrt(2)
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);

        assert!(approx_eq(n[0], 0.6)); // 3/5
        assert!(approx_eq(n[1], 0.8)); // 4/5

        let norm = dot_product(&n, &n).sqrt();
        assert!(approx_eq(norm, 1.0));
    }

    #[test]
    fn test_l2_squared_u8() {
        let a: Vec<u8> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let b: Vec<u8> = vec![11, 21, 31, 41, 51, 61, 71, 81];

        let scalar = l2_squared_u8_scalar(&a, &b);
        let result = l2_squared_u8(&a, &b);

        assert_eq!(scalar, 8); // 8 * 1^2 = 8
        assert_eq!(result, scalar);
    }

    #[test]
    fn test_dot_product_u8() {
        let a: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let b: Vec<u8> = vec![2, 3, 4, 5, 6, 7, 8, 9];

        let scalar = dot_product_u8_scalar(&a, &b);
        let result = dot_product_u8(&a, &b);

        // 2+6+12+20+30+42+56+72 = 240
        assert_eq!(scalar, 240);
        assert_eq!(result, scalar);
    }

    #[test]
    fn test_simd_detection() {
        let level = detect_simd_level();
        println!("Detected SIMD level: {:?}", level);
        assert!(level as u8 <= 3);
    }

    #[test]
    fn test_large_vectors() {
        // Test with 128-dim vectors (common embedding size)
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| i as f32 * 0.1 + 0.5).collect();

        let scalar = l2_squared_scalar(&a, &b);
        let simd = l2_squared(&a, &b);

        assert!(approx_eq(scalar, simd));
    }
}
