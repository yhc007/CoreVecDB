//! Benchmark: CPU vs Candle tensor ops for GPU distance and K-Means.
//!
//! Run with:
//!   cargo bench --bench candle_gpu_bench --features candle --no-default-features
//!
//! This benchmark uses std::time for measurement (no criterion dependency needed).

use std::time::Instant;
use vectordb::gpu::{GpuAccelerator, GpuConfig, GpuBackend, DistanceMetric};
#[cfg(feature = "candle")]
use vectordb::gpu::CachedGpuAccelerator;
use vectordb::ivfpq::{IvfPqConfig, IvfPqIndex, CoarseQuantizer};

fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

fn bench_fn<F: FnMut()>(name: &str, mut f: F, iterations: usize) {
    // Warmup
    f();

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iterations as u32;
    println!(
        "  {:<45} {:>8.2}ms  ({}x, total {:.1}s)",
        name,
        per_iter.as_secs_f64() * 1000.0,
        iterations,
        elapsed.as_secs_f64(),
    );
}

fn main() {
    println!("=== CoreVecDB Candle GPU Benchmark ===\n");

    #[cfg(feature = "candle")]
    println!("  candle feature: ENABLED");
    #[cfg(not(feature = "candle"))]
    println!("  candle feature: DISABLED (CPU-only baseline)");

    // Detect backend
    #[cfg(feature = "candle")]
    {
        let gpu = GpuAccelerator::new(GpuConfig {
            backend: GpuBackend::Auto,
            min_batch_size: 1,
            ..GpuConfig::default()
        }).unwrap();
        println!("  Selected backend: {:?}", gpu.backend());
        println!("  Capabilities: {:?}", gpu.capabilities().device_name);
    }
    println!();

    // ── Phase 4: Batch Distance Computation ──────────────────────────────

    println!("── Batch L2 Distance ──");
    for &(n, dim) in &[
        (1_000, 128), (10_000, 128), (10_000, 768),
        (50_000, 128), (100_000, 128), (100_000, 768),
    ] {
        println!("\n  n={}, dim={}", n, dim);
        let vectors = random_vectors(n, dim);
        let query = random_vectors(1, dim).into_iter().next().unwrap();
        let iters = match n {
            100_000 => 3,
            50_000 => 5,
            _ => 10,
        };

        // CPU (rayon SIMD)
        {
            let gpu = GpuAccelerator::new(GpuConfig::cpu_only()).unwrap();
            bench_fn("CPU (rayon + SIMD)", || {
                let _ = gpu.batch_l2_distance(&query, &vectors).unwrap();
            }, iters);
        }

        // Candle (Auto: Metal on macOS, else CPU tensor)
        #[cfg(feature = "candle")]
        {
            let gpu = GpuAccelerator::new(GpuConfig {
                backend: GpuBackend::Auto,
                min_batch_size: 1,
                ..GpuConfig::default()
            }).unwrap();
            let backend = gpu.backend();
            bench_fn(&format!("Candle ({:?})", backend), || {
                let _ = gpu.batch_l2_distance(&query, &vectors).unwrap();
            }, iters);
        }
    }

    // ── Cached GPU L2 ────────────────────────────────────────────────────
    #[cfg(feature = "candle")]
    {
        println!("\n── Batch L2 Distance (Cached GPU) ──");
        println!("  Vectors pre-loaded on GPU; only query transferred per search.\n");

        for &(n, dim) in &[
            (1_000, 128), (10_000, 128), (10_000, 768),
            (50_000, 128), (100_000, 128), (100_000, 768),
        ] {
            println!("  n={}, dim={}", n, dim);
            let vectors = random_vectors(n, dim);
            let query = random_vectors(1, dim).into_iter().next().unwrap();
            let iters = match n {
                100_000 => 3,
                50_000 => 5,
                _ => 10,
            };

            // CPU baseline
            {
                let gpu = GpuAccelerator::new(GpuConfig::cpu_only()).unwrap();
                bench_fn("CPU (rayon + SIMD)", || {
                    let _ = gpu.batch_l2_distance(&query, &vectors).unwrap();
                }, iters);
            }

            // Candle uncached
            {
                let gpu = GpuAccelerator::new(GpuConfig {
                    backend: GpuBackend::Auto,
                    min_batch_size: 1,
                    ..GpuConfig::default()
                }).unwrap();
                let backend = gpu.backend();
                bench_fn(&format!("Candle uncached ({:?})", backend), || {
                    let _ = gpu.batch_l2_distance(&query, &vectors).unwrap();
                }, iters);
            }

            // Candle cached
            {
                let cached = CachedGpuAccelerator::new(&vectors).unwrap();
                let device = format!("{:?}", cached.device());
                bench_fn(&format!("Candle CACHED ({})", device), || {
                    let _ = cached.batch_l2_distance(&query).unwrap();
                }, iters);
            }

            println!();
        }
    }

    println!("\n── Batch Dot Product ──");
    for &(n, dim) in &[(10_000, 128), (10_000, 768), (100_000, 128)] {
        println!("\n  n={}, dim={}", n, dim);
        let vectors = random_vectors(n, dim);
        let query = random_vectors(1, dim).into_iter().next().unwrap();
        let iters = if n >= 100_000 { 3 } else { 10 };

        {
            let gpu = GpuAccelerator::new(GpuConfig::cpu_only()).unwrap();
            bench_fn("CPU (rayon + SIMD)", || {
                let _ = gpu.batch_dot_product(&query, &vectors).unwrap();
            }, iters);
        }

        #[cfg(feature = "candle")]
        {
            let gpu = GpuAccelerator::new(GpuConfig {
                backend: GpuBackend::Auto,
                min_batch_size: 1,
                ..GpuConfig::default()
            }).unwrap();
            let backend = gpu.backend();
            bench_fn(&format!("Candle ({:?})", backend), || {
                let _ = gpu.batch_dot_product(&query, &vectors).unwrap();
            }, iters);
        }
    }

    // ── Cached GPU Dot Product ────────────────────────────────────────────
    #[cfg(feature = "candle")]
    {
        println!("\n── Batch Dot Product (Cached GPU) ──");
        for &(n, dim) in &[(10_000, 128), (10_000, 768), (100_000, 128)] {
            println!("\n  n={}, dim={}", n, dim);
            let vectors = random_vectors(n, dim);
            let query = random_vectors(1, dim).into_iter().next().unwrap();
            let iters = if n >= 100_000 { 3 } else { 10 };

            {
                let gpu = GpuAccelerator::new(GpuConfig::cpu_only()).unwrap();
                bench_fn("CPU (rayon + SIMD)", || {
                    let _ = gpu.batch_dot_product(&query, &vectors).unwrap();
                }, iters);
            }

            {
                let cached = CachedGpuAccelerator::new(&vectors).unwrap();
                let device = format!("{:?}", cached.device());
                bench_fn(&format!("Candle CACHED ({})", device), || {
                    let _ = cached.batch_dot_product(&query).unwrap();
                }, iters);
            }
        }
    }

    println!("\n── Batch Cosine Distance ──");
    for &(n, dim) in &[(10_000, 128), (10_000, 768), (100_000, 128)] {
        println!("\n  n={}, dim={}", n, dim);
        let vectors = random_vectors(n, dim);
        let query = random_vectors(1, dim).into_iter().next().unwrap();
        let iters = if n >= 100_000 { 3 } else { 10 };

        {
            let gpu = GpuAccelerator::new(GpuConfig::cpu_only()).unwrap();
            bench_fn("CPU (rayon + SIMD)", || {
                let _ = gpu.batch_cosine_distance(&query, &vectors).unwrap();
            }, iters);
        }

        #[cfg(feature = "candle")]
        {
            let gpu = GpuAccelerator::new(GpuConfig {
                backend: GpuBackend::Auto,
                min_batch_size: 1,
                ..GpuConfig::default()
            }).unwrap();
            let backend = gpu.backend();
            bench_fn(&format!("Candle ({:?})", backend), || {
                let _ = gpu.batch_cosine_distance(&query, &vectors).unwrap();
            }, iters);
        }
    }

    // ── Cached GPU Cosine Distance ─────────────────────────────────────
    #[cfg(feature = "candle")]
    {
        println!("\n── Batch Cosine Distance (Cached GPU) ──");
        for &(n, dim) in &[(10_000, 128), (10_000, 768), (100_000, 128)] {
            println!("\n  n={}, dim={}", n, dim);
            let vectors = random_vectors(n, dim);
            let query = random_vectors(1, dim).into_iter().next().unwrap();
            let iters = if n >= 100_000 { 3 } else { 10 };

            {
                let gpu = GpuAccelerator::new(GpuConfig::cpu_only()).unwrap();
                bench_fn("CPU (rayon + SIMD)", || {
                    let _ = gpu.batch_cosine_distance(&query, &vectors).unwrap();
                }, iters);
            }

            {
                let cached = CachedGpuAccelerator::new(&vectors).unwrap();
                let device = format!("{:?}", cached.device());
                bench_fn(&format!("Candle CACHED ({})", device), || {
                    let _ = cached.batch_cosine_distance(&query).unwrap();
                }, iters);
            }
        }
    }

    // ── Cached GPU K-NN Search ──────────────────────────────────────────
    #[cfg(feature = "candle")]
    {
        println!("\n── K-NN Search (Cached GPU) ──");
        for &(n, dim) in &[(10_000, 128), (100_000, 128)] {
            println!("\n  n={}, dim={}, k=10", n, dim);
            let vectors = random_vectors(n, dim);
            let query = random_vectors(1, dim).into_iter().next().unwrap();
            let iters = if n >= 100_000 { 3 } else { 10 };

            {
                let gpu = GpuAccelerator::new(GpuConfig::cpu_only()).unwrap();
                bench_fn("CPU knn_search (k=10)", || {
                    let _ = gpu.knn_search(&query, &vectors, 10, DistanceMetric::L2Squared).unwrap();
                }, iters);
            }

            {
                let cached = CachedGpuAccelerator::new(&vectors).unwrap();
                let device = format!("{:?}", cached.device());
                bench_fn(&format!("Cached knn_search (k=10, {})", device), || {
                    let _ = cached.knn_search(&query, 10, DistanceMetric::L2Squared).unwrap();
                }, iters);
            }
        }
    }

    // ── Phase 5: K-Means Training ────────────────────────────────────────

    println!("\n── K-Means Training (CoarseQuantizer) ──");
    for &(n, dim, nlist) in &[
        (10_000, 128, 64),
        (50_000, 128, 256),
        (100_000, 128, 256),
        (10_000, 768, 64),
    ] {
        println!("\n  n={}, dim={}, nlist={}", n, dim, nlist);
        let vectors = random_vectors(n, dim);
        let iters = match n {
            100_000 => 1,
            50_000 => 2,
            _ => 3,
        };

        bench_fn("K-Means train (10 iters)", || {
            let mut cq = CoarseQuantizer::new(nlist, dim);
            cq.train(&vectors, 10).unwrap();
        }, iters);
    }

    // ── IVF-PQ Full Pipeline ─────────────────────────────────────────────

    println!("\n── IVF-PQ Train + Search ──");
    for &(n, dim) in &[(10_000, 128), (50_000, 128)] {
        println!("\n  n={}, dim={}", n, dim);
        let vectors = random_vectors(n, dim);
        let query = random_vectors(1, dim).into_iter().next().unwrap();

        let config = IvfPqConfig {
            dim,
            nlist: 64,
            nprobe: 8,
            m: 8,
            ksub: 256,
            kmeans_iters: 10,
        };

        // Train
        let iters = if n >= 50_000 { 1 } else { 2 };
        bench_fn("IVF-PQ train", || {
            let idx = IvfPqIndex::new(config.clone()).unwrap();
            idx.train(&vectors).unwrap();
        }, iters);

        // Add + Search
        let idx = IvfPqIndex::new(config).unwrap();
        idx.train(&vectors).unwrap();
        for v in &vectors {
            idx.add(v).unwrap();
        }

        bench_fn("IVF-PQ search (k=10, nprobe=8)", || {
            let _ = idx.search(&query, 10).unwrap();
        }, 100);
    }

    println!("\n=== Done ===");
}
