// benches/hca_benchmarks.rs
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use criterion::measurement::WallTime;

use HashChainStream::{AnchorConfig, HashMode};
use HashChainStream::anchor_hca::AnchorHca;
use HashChainStream::direct_anchor_hca::DirectAnchorHca;
use HashChainStream::radix_hca::RadixHca;
use HashChainStream::concurrent_direct_anchor_hca::ConcurrentDirectAnchorHca;

use ahash::AHashMap;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/* --------------- utilities --------------- */

fn generate_test_keys(count: usize, key_length: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut keys = Vec::with_capacity(count);
    for i in 0..count {
        let random_part: String = (0..key_length.saturating_sub(10))
            .map(|_| {
                let chars = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
                chars[rng.gen_range(0..chars.len())] as char
            })
            .collect();
        keys.push(format!("{:010}{}", i, random_part));
    }
    keys
}

fn tune_group(group: &mut criterion::BenchmarkGroup<WallTime>, size: usize) {
    if size >= 10_000_000 {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(120));
        group.warm_up_time(Duration::from_secs(5));
    } else if size >= 1_000_000 {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(45));
        group.warm_up_time(Duration::from_secs(3));
    } else if size >= 500_000 {
        group.sample_size(20);
        group.measurement_time(Duration::from_secs(20));
        group.warm_up_time(Duration::from_secs(3));
    } else {
        group.sample_size(30);
        group.measurement_time(Duration::from_secs(10));
        group.warm_up_time(Duration::from_secs(2));
    }
}

fn pick_thread_counts(size: usize) -> Vec<usize> {
    // Avoid silly configs on tiny inputs; tune as you like
    let base = [2usize, 4, 8, 16, 32];
    if size < 10_000 {
        base.iter().copied().filter(|&t| t <= 4).collect()
    } else if size < 100_000 {
        base.iter().copied().filter(|&t| t <= 8).collect()
    } else {
        base.to_vec()
    }
}

/* -------------------- INSERTION (ST) -------------------- */
fn bench_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion");

    for size in [1_000, 10_000, 100_000, 500_000, 1_000_000, 10_000_000] {
        tune_group(&mut group, size);
        let keys = generate_test_keys(size, 20);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("anchor_hca", size), &keys, |b, keys| {
            b.iter(|| {
                let mut h = AnchorHca::with_default_config();
                for (i, k) in keys.iter().enumerate() {
                    h.insert(k.clone(), i).unwrap();
                }
                criterion::black_box(h);
            });
        });

        group.bench_with_input(BenchmarkId::new("direct_anchor_hca", size), &keys, |b, keys| {
            b.iter(|| {
                let mut h = DirectAnchorHca::with_default_config();
                for (i, k) in keys.iter().enumerate() {
                    h.insert(k.clone(), i).unwrap();
                }
                criterion::black_box(h);
            });
        });

        group.bench_with_input(BenchmarkId::new("radix_hca", size), &keys, |b, keys| {
            b.iter(|| {
                let mut h = RadixHca::<String, usize>::with_default_config();
                for (i, k) in keys.iter().enumerate() {
                    h.insert(k.clone(), i).unwrap();
                }
                criterion::black_box(h);
            });
        });

        // ConcurrentDirectAnchorHca single-thread "baseline"
        group.bench_with_input(
            BenchmarkId::new("concurrent_direct_anchor_hca_st", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let h = ConcurrentDirectAnchorHca::<String, usize>::with_default_config();
                    for (i, k) in keys.iter().enumerate() {
                        let _ = h.insert(k.clone(), i).unwrap();
                    }
                    criterion::black_box(h);
                });
            },
        );

        // std & ahash
        group.bench_with_input(BenchmarkId::new("std_hashmap", size), &keys, |b, keys| {
            b.iter(|| {
                let mut m = HashMap::new();
                for (i, k) in keys.iter().enumerate() {
                    m.insert(k.clone(), i);
                }
                criterion::black_box(m);
            });
        });

        group.bench_with_input(BenchmarkId::new("ahashmap", size), &keys, |b, keys| {
            b.iter(|| {
                let mut m = AHashMap::new();
                for (i, k) in keys.iter().enumerate() {
                    m.insert(k.clone(), i);
                }
                criterion::black_box(m);
            });
        });
    }

    group.finish();
}

/* -------------------- LOOKUP (ST) -------------------- */
fn bench_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup");

    for size in [1_000, 10_000, 100_000, 500_000, 1_000_000, 10_000_000] {
        tune_group(&mut group, size);
        let keys = generate_test_keys(size, 20);

        let mut anchor = AnchorHca::with_default_config();
        let mut direct = DirectAnchorHca::with_default_config();
        let mut radix = RadixHca::<String, usize>::with_default_config();
        let concurrent = ConcurrentDirectAnchorHca::<String, usize>::with_default_config();
        let mut std_map = HashMap::new();
        let mut ahash = AHashMap::new();

        for (i, k) in keys.iter().enumerate() {
            anchor.insert(k.clone(), i).unwrap();
            direct.insert(k.clone(), i).unwrap();
            radix.insert(k.clone(), i).unwrap();
            let _ = concurrent.insert(k.clone(), i).unwrap();
            std_map.insert(k.clone(), i);
            ahash.insert(k.clone(), i);
        }

        let mut lookup_keys = keys[0..(keys.len() / 2)].to_vec();
        lookup_keys.extend(generate_test_keys(keys.len() / 2, 20));
        group.throughput(Throughput::Elements(lookup_keys.len() as u64));

        group.bench_with_input(BenchmarkId::new("anchor_hca", size), &lookup_keys, |b, ks| {
            b.iter(|| for k in ks.iter() { criterion::black_box(anchor.get(k)); });
        });

        group.bench_with_input(BenchmarkId::new("direct_anchor_hca", size), &lookup_keys, |b, ks| {
            b.iter(|| for k in ks.iter() { criterion::black_box(direct.get(k)); });
        });

        group.bench_with_input(BenchmarkId::new("radix_hca", size), &lookup_keys, |b, ks| {
            b.iter(|| for k in ks.iter() { criterion::black_box(radix.get(k)); });
        });

        group.bench_with_input(
            BenchmarkId::new("concurrent_direct_anchor_hca_st", size),
            &lookup_keys,
            |b, ks| {
                b.iter(|| for k in ks.iter() { criterion::black_box(concurrent.get(k)); });
            },
        );

        group.bench_with_input(BenchmarkId::new("std_hashmap", size), &lookup_keys, |b, ks| {
            b.iter(|| for k in ks.iter() { criterion::black_box(std_map.get(k)); });
        });

        group.bench_with_input(BenchmarkId::new("ahashmap", size), &lookup_keys, |b, ks| {
            b.iter(|| for k in ks.iter() { criterion::black_box(ahash.get(k)); });
        });
    }

    group.finish();
}

/* -------------------- MIXED WORKLOAD (ST) -------------------- */
fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");

    for base_size in [1_000, 10_000, 100_000, 500_000, 1_000_000, 10_000_000] {
        tune_group(&mut group, base_size);
        let keys = generate_test_keys(base_size, 20);
        let operations = base_size;
        group.throughput(Throughput::Elements(operations as u64));

        // Anchor
        group.bench_with_input(BenchmarkId::new("anchor_hca_mixed", base_size), &operations, |b, &ops| {
            b.iter(|| {
                let mut h = AnchorHca::with_default_config();
                let mut rng = StdRng::seed_from_u64(123);
                for i in 0..ops {
                    match i % 4 {
                        0 | 1 => { let k = &keys[rng.gen_range(0..keys.len())]; h.insert(k.clone(), i).unwrap(); }
                        2 =>     { let k = &keys[rng.gen_range(0..keys.len())]; criterion::black_box(h.get(k)); }
                        3 =>     { let k = &keys[rng.gen_range(0..keys.len())]; criterion::black_box(h.remove(k)); }
                        _ => unreachable!(),
                    }
                }
                criterion::black_box(h);
            });
        });

        // Direct
        group.bench_with_input(BenchmarkId::new("direct_anchor_hca_mixed", base_size), &operations, |b, &ops| {
            b.iter(|| {
                let mut h = DirectAnchorHca::with_default_config();
                let mut rng = StdRng::seed_from_u64(123);
                for i in 0..ops {
                    match i % 4 {
                        0 | 1 => { let k = &keys[rng.gen_range(0..keys.len())]; h.insert(k.clone(), i).unwrap(); }
                        2 =>     { let k = &keys[rng.gen_range(0..keys.len())]; criterion::black_box(h.get(k)); }
                        3 =>     { let k = &keys[rng.gen_range(0..keys.len())]; criterion::black_box(h.remove(k)); }
                        _ => unreachable!(),
                    }
                }
                criterion::black_box(h);
            });
        });

        // Radix
        group.bench_with_input(BenchmarkId::new("radix_hca_mixed", base_size), &operations, |b, &ops| {
            b.iter(|| {
                let mut h = RadixHca::<String, usize>::with_default_config();
                let mut rng = StdRng::seed_from_u64(123);
                for i in 0..ops {
                    match i % 4 {
                        0 | 1 => { let k = &keys[rng.gen_range(0..keys.len())]; h.insert(k.clone(), i).unwrap(); }
                        2 =>     { let k = &keys[rng.gen_range(0..keys.len())]; criterion::black_box(h.get(k)); }
                        3 =>     { let k = &keys[rng.gen_range(0..keys.len())]; criterion::black_box(h.remove(k)); }
                        _ => unreachable!(),
                    }
                }
                criterion::black_box(h);
            });
        });

        // Concurrent (single-thread behavior)
        group.bench_with_input(
            BenchmarkId::new("concurrent_direct_anchor_hca_mixed_st", base_size),
            &operations,
            |b, &ops| {
                b.iter(|| {
                    let h = ConcurrentDirectAnchorHca::<String, usize>::with_default_config();
                    let mut rng = StdRng::seed_from_u64(123);
                    for i in 0..ops {
                        match i % 4 {
                            0 | 1 => { let k = &keys[rng.gen_range(0..keys.len())]; let _ = h.insert(k.clone(), i).unwrap(); }
                            2 =>     { let k = &keys[rng.gen_range(0..keys.len())]; criterion::black_box(h.get(k)); }
                            3 =>     { let k = &keys[rng.gen_range(0..keys.len())]; criterion::black_box(h.remove(k)); }
                            _ => unreachable!(),
                        }
                    }
                    criterion::black_box(h);
                });
            },
        );

        // Std
        group.bench_with_input(BenchmarkId::new("std_hashmap_mixed", base_size), &operations, |b, &ops| {
            b.iter(|| {
                let mut m = HashMap::new();
                let mut rng = StdRng::seed_from_u64(123);
                for i in 0..ops {
                    match i % 4 {
                        0 | 1 => { let k = &keys[rng.gen_range(0..keys.len())]; m.insert(k.clone(), i); }
                        2 =>     { let k = &keys[rng.gen_range(0..keys.len())]; criterion::black_box(m.get(k)); }
                        3 =>     { let k = &keys[rng.gen_range(0..keys.len())]; criterion::black_box(m.remove(k)); }
                        _ => unreachable!(),
                    }
                }
                criterion::black_box(m);
            });
        });
    }

    group.finish();
}

/* -------------------- GROWTH BEHAVIOR (ST) -------------------- */
fn bench_growth_behavior(c: &mut Criterion) {
    let mut group = c.benchmark_group("growth_behavior");

    for total_keys in [1_000, 10_000, 100_000, 500_000, 1_000_000, 10_000_000] {
        tune_group(&mut group, total_keys);

        // Anchor
        group.bench_with_input(
            BenchmarkId::new("anchor_hca_incremental_growth", total_keys),
            &total_keys,
            |b, &total| {
                b.iter(|| {
                    let mut h = AnchorHca::new(AnchorConfig {
                        s_max: 8,
                        default_depth: 1,
                        domain: b"bench".to_vec(),
                        hash_mode: HashMode::Fast,
                    });
                    let batch_count = 20usize;
                    let batch_size = (total / batch_count).max(1);
                    for batch in 0..batch_count {
                        let ks = generate_test_keys(batch_size, 15);
                        for (i, k) in ks.iter().enumerate() {
                            h.insert(k.clone(), batch * batch_size + i).unwrap();
                        }
                        if batch > 0 && batch % 5 == 0 {
                            for k in ks.iter().take(batch_size.min(100)) {
                                criterion::black_box(h.get(k));
                            }
                        }
                    }
                    criterion::black_box(h.stats());
                });
            },
        );

        // Direct
        group.bench_with_input(
            BenchmarkId::new("direct_anchor_hca_incremental_growth", total_keys),
            &total_keys,
            |b, &total| {
                b.iter(|| {
                    let mut h = DirectAnchorHca::with_default_config();
                    let batch_count = 20usize;
                    let batch_size = (total / batch_count).max(1);
                    for batch in 0..batch_count {
                        let ks = generate_test_keys(batch_size, 15);
                        for (i, k) in ks.iter().enumerate() {
                            h.insert(k.clone(), batch * batch_size + i).unwrap();
                        }
                        if batch > 0 && batch % 5 == 0 {
                            for k in ks.iter().take(batch_size.min(100)) {
                                criterion::black_box(h.get(k));
                            }
                        }
                    }
                    criterion::black_box(h.stats());
                });
            },
        );

        // Radix
        group.bench_with_input(
            BenchmarkId::new("radix_hca_incremental_growth", total_keys),
            &total_keys,
            |b, &total| {
                b.iter(|| {
                    let mut h = RadixHca::<String, usize>::with_default_config();
                    let batch_count = 20usize;
                    let batch_size = (total / batch_count).max(1);
                    for batch in 0..batch_count {
                        let ks = generate_test_keys(batch_size, 15);
                        for (i, k) in ks.iter().enumerate() {
                            h.insert(k.clone(), batch * batch_size + i).unwrap();
                        }
                        if batch > 0 && batch % 5 == 0 {
                            for k in ks.iter().take(batch_size.min(100)) {
                                criterion::black_box(h.get(k));
                            }
                        }
                    }
                    criterion::black_box(h.stats());
                });
            },
        );

        // Concurrent (ST growth)
        group.bench_with_input(
            BenchmarkId::new("concurrent_direct_anchor_hca_incremental_growth_st", total_keys),
            &total_keys,
            |b, &total| {
                b.iter(|| {
                    let h = ConcurrentDirectAnchorHca::<String, usize>::with_default_config();
                    let batch_count = 20usize;
                    let batch_size = (total / batch_count).max(1);
                    for batch in 0..batch_count {
                        let ks = generate_test_keys(batch_size, 15);
                        for (i, k) in ks.iter().enumerate() {
                            let _ = h.insert(k.clone(), batch * batch_size + i).unwrap();
                        }
                        if batch > 0 && batch % 5 == 0 {
                            for k in ks.iter().take(batch_size.min(100)) {
                                criterion::black_box(h.get(k));
                            }
                        }
                    }
                    criterion::black_box(());
                });
            },
        );
    }

    group.finish();
}

/* -------------------- RECOVERY (partial rebuild, ST) -------------------- */
fn bench_recovery_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("recovery");

    for size in [1_000, 10_000, 100_000, 500_000, 1_000_000, 10_000_000] {
        tune_group(&mut group, size);
        let keys = generate_test_keys(size, 20);
        let anchor_keys: Vec<_> = keys.iter().take(keys.len() / 10).cloned().collect();

        // Anchor partial
        group.bench_with_input(BenchmarkId::new("anchor_reconstruction", size), &anchor_keys, |b, ks| {
            b.iter(|| {
                let mut rebuilt = AnchorHca::with_default_config();
                for (i, k) in ks.iter().enumerate() {
                    rebuilt.insert(k.clone(), i).unwrap();
                }
                criterion::black_box(rebuilt.stats());
            });
        });

        // Direct partial
        group.bench_with_input(
            BenchmarkId::new("direct_anchor_reconstruction", size),
            &anchor_keys,
            |b, ks| {
                b.iter(|| {
                    let mut rebuilt = DirectAnchorHca::with_default_config();
                    for (i, k) in ks.iter().enumerate() {
                        rebuilt.insert(k.clone(), i).unwrap();
                    }
                    criterion::black_box(rebuilt.stats());
                });
            },
        );

        // Radix partial
        group.bench_with_input(
            BenchmarkId::new("radix_hca_reconstruction", size),
            &anchor_keys,
            |b, ks| {
                b.iter(|| {
                    let mut rebuilt = RadixHca::<String, usize>::with_default_config();
                    for (i, k) in ks.iter().enumerate() {
                        rebuilt.insert(k.clone(), i).unwrap();
                    }
                    criterion::black_box(rebuilt.stats());
                });
            },
        );

        // Concurrent partial (single-thread rebuild)
        group.bench_with_input(
            BenchmarkId::new("concurrent_direct_anchor_hca_reconstruction_st", size),
            &anchor_keys,
            |b, ks| {
                b.iter(|| {
                    let rebuilt = ConcurrentDirectAnchorHca::<String, usize>::with_default_config();
                    for (i, k) in ks.iter().enumerate() {
                        let _ = rebuilt.insert(k.clone(), i).unwrap();
                    }
                    criterion::black_box(());
                });
            },
        );

        // Full HashMap rebuild
        group.bench_with_input(BenchmarkId::new("full_hashmap_rebuild", size), &keys, |b, keys| {
            b.iter(|| {
                let mut rebuilt = HashMap::new();
                for (i, k) in keys.iter().enumerate() {
                    rebuilt.insert(k.clone(), i);
                }
                criterion::black_box(rebuilt.len());
            });
        });
    }

    group.finish();
}

/* -------- PARALLEL: concurrent_direct_anchor_hca -------- */

fn shard_ranges(len: usize, threads: usize) -> Vec<(usize, usize)> {
    let chunk = (len + threads - 1) / threads;
    (0..threads)
        .map(|i| {
            let start = i * chunk;
            let end = (start + chunk).min(len);
            (start, end)
        })
        .filter(|(s, e)| *s < *e)
        .collect()
}

/* -------- Parallel-only: INSERT -------- */
fn bench_par_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion");
    let thread_counts = [2usize, 4, 8, 16, 32];
    let sizes = [1_000usize, 10_000, 100_000, 500_000, 1_000_000, 10_000_000];

    for &threads in &thread_counts {
        for &size in &sizes {
            tune_group(&mut group, size);
            let id = BenchmarkId::new(
                format!("concurrent_direct_anchor_hca_par_t{}", threads),
                size,
            );
            group.throughput(Throughput::Elements(size as u64));

            // pre-generate keys once, outside timing
            let keys = generate_test_keys(size, 20);

            group.bench_with_input(id, &(threads, size), |b, &(t, _)| {
                b.iter(|| {
                    let h = std::sync::Arc::new(
                        ConcurrentDirectAnchorHca::<String, usize>::with_default_config(),
                    );
                    let ranges = shard_ranges(keys.len(), t);

                    thread::scope(|scope| {
                        for (start, end) in ranges.iter().copied() {
                            let h = std::sync::Arc::clone(&h);
                            let slice = &keys[start..end]; // borrow inside scope
                            scope.spawn(move || {
                                for (i, k) in slice.iter().enumerate() {
                                    let _ = h.insert(k.clone(), start + i).unwrap();
                                }
                            });
                        }
                    });

                    criterion::black_box(h.as_ref());
                });
            });
        }
    }

    group.finish();
}

/* -------- Parallel-only: LOOKUP -------- */
fn bench_par_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup");
    let thread_counts = [2usize, 4, 8, 16, 32];
    let sizes = [1_000usize, 10_000, 100_000];

    for &threads in &thread_counts {
        for &size in &sizes {
            tune_group(&mut group, size);
            let id = BenchmarkId::new(
                format!("concurrent_direct_anchor_hca_par_t{}", threads),
                size,
            );

            // build the map once (outside timing)
            let h = std::sync::Arc::new(
                ConcurrentDirectAnchorHca::<String, usize>::with_default_config(),
            );
            let keys = generate_test_keys(size, 20);

            // parallel insert to populate
            thread::scope(|scope| {
                for (start, end) in shard_ranges(keys.len(), threads) {
                    let h = std::sync::Arc::clone(&h);
                    let slice = &keys[start..end];
                    scope.spawn(move || {
                        for (i, k) in slice.iter().enumerate() {
                            let _ = h.insert(k.clone(), start + i).unwrap();
                        }
                    });
                }
            });

            // 50% hits, 50% misses
            let mut lookups = keys[..(keys.len() / 2)].to_vec();
            lookups.extend(generate_test_keys(keys.len() - lookups.len(), 20));
            group.throughput(Throughput::Elements(lookups.len() as u64));

            group.bench_with_input(id, &(threads, size), |b, &(t, _)| {
                b.iter(|| {
                    thread::scope(|scope| {
                        for (start, end) in shard_ranges(lookups.len(), t) {
                            let h = std::sync::Arc::clone(&h);
                            let slice = &lookups[start..end];
                            scope.spawn(move || {
                                for k in slice {
                                    criterion::black_box(h.get(k));
                                }
                            });
                        }
                    });
                });
            });
        }
    }

    group.finish();
}

/* -------- Parallel-only: MIXED -------- */
fn bench_par_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");
    let thread_counts = [2usize, 4, 8, 16, 32];
    let sizes = [1_000usize, 10_000, 100_000, 1_000_000, 10_000_000];

    for &threads in &thread_counts {
        for &ops in &sizes {
            tune_group(&mut group, ops);
            let id = BenchmarkId::new(
                format!("concurrent_direct_anchor_hca_mixed_par_t{}", threads),
                ops,
            );
            group.throughput(Throughput::Elements(ops as u64));

            // reusable keyset outside timing
            let keys = generate_test_keys(ops, 20);

            group.bench_with_input(id, &(threads, ops), |b, &(t, nops)| {
                b.iter(|| {
                    let h = std::sync::Arc::new(
                        ConcurrentDirectAnchorHca::<String, usize>::with_default_config(),
                    );

                    thread::scope(|scope| {
                        for (start, end) in shard_ranges(nops, t) {
                            let h = std::sync::Arc::clone(&h);
                            let slice = &keys[start..end];
                            scope.spawn(move || {
                                let mut rng = StdRng::seed_from_u64(123 + start as u64);
                                for i in 0..slice.len() {
                                    match i % 4 {
                                        0 | 1 => {
                                            let k = &slice[rng.gen_range(0..slice.len())];
                                            let _ = h.insert(k.clone(), start + i).unwrap();
                                        }
                                        2 => {
                                            let k = &slice[rng.gen_range(0..slice.len())];
                                            criterion::black_box(h.get(k));
                                        }
                                        3 => {
                                            let k = &slice[rng.gen_range(0..slice.len())];
                                            criterion::black_box(h.remove(k));
                                        }
                                        _ => unreachable!(),
                                    }
                                }
                            });
                        }
                    });

                    criterion::black_box(h.as_ref());
                });
            });
        }
    }

    group.finish();
}

/* -------- Parallel-only: GROWTH -------- */
fn bench_par_growth(c: &mut Criterion) {
    let mut group = c.benchmark_group("growth_behavior");
    let thread_counts = [2usize, 4, 8, 16, 32];
    let totals = [1_000usize, 10_000, 100_000, 1_000_000];

    for &threads in &thread_counts {
        for &total in &totals {
            tune_group(&mut group, total);
            let id = BenchmarkId::new(
                format!("concurrent_direct_anchor_hca_incremental_growth_par_t{}", threads),
                total,
            );
            group.throughput(Throughput::Elements(total as u64));

            group.bench_with_input(id, &(threads, total), |b, &(t, n)| {
                b.iter(|| {
                    let h = std::sync::Arc::new(
                        ConcurrentDirectAnchorHca::<String, usize>::with_default_config(),
                    );
                    let batch_count = 20usize;
                    let batch_size = (n / batch_count).max(1);

                    for b_ix in 0..batch_count {
                        let ks = generate_test_keys(batch_size, 15);

                        thread::scope(|scope| {
                            for (start, end) in shard_ranges(ks.len(), t) {
                                let h = std::sync::Arc::clone(&h);
                                let slice = &ks[start..end];
                                scope.spawn(move || {
                                    for (i, k) in slice.iter().enumerate() {
                                        let _ = h.insert(
                                            k.clone(),
                                            b_ix * batch_size + start + i,
                                        ).unwrap();
                                    }
                                });
                            }
                        });

                        if b_ix > 0 && b_ix % 5 == 0 {
                            let probe: Vec<_> =
                                ks.iter().take(batch_size.min(100)).cloned().collect();
                            thread::scope(|scope| {
                                for (start, end) in shard_ranges(probe.len(), t) {
                                    let h = std::sync::Arc::clone(&h);
                                    let slice = &probe[start..end];
                                    scope.spawn(move || {
                                        for k in slice {
                                            criterion::black_box(h.get(k));
                                        }
                                    });
                                }
                            });
                        }
                    }

                    criterion::black_box(h.as_ref());
                });
            });
        }
    }

    group.finish();
}

/* -------- Parallel-only: RECOVERY -------- */
fn bench_par_recovery(c: &mut Criterion) {
    let mut group = c.benchmark_group("recovery");
    let thread_counts = [2usize, 4, 8, 16, 32];
    let sizes = [1_000usize, 10_000, 100_000, 500_000, 1_000_000];

    for &threads in &thread_counts {
        for &size in &sizes {
            tune_group(&mut group, size);
            let id = BenchmarkId::new(
                format!("concurrent_direct_anchor_hca_reconstruction_par_t{}", threads),
                size,
            );

            // subset outside timing
            let keys = generate_test_keys(size, 20);
            let subset: Vec<_> = keys.into_iter().take(size / 10).collect();
            group.throughput(Throughput::Elements(subset.len() as u64));

            group.bench_with_input(id, &(threads, size), |b, &(t, _)| {
                b.iter(|| {
                    let rebuilt = std::sync::Arc::new(
                        ConcurrentDirectAnchorHca::<String, usize>::with_default_config(),
                    );

                    thread::scope(|scope| {
                        for (start, end) in shard_ranges(subset.len(), t) {
                            let h = std::sync::Arc::clone(&rebuilt);
                            let slice = &subset[start..end];
                            scope.spawn(move || {
                                for (i, k) in slice.iter().enumerate() {
                                    let _ = h.insert(k.clone(), start + i).unwrap();
                                }
                            });
                        }
                    });

                    criterion::black_box(rebuilt.as_ref());
                });
            });
        }
    }

    group.finish();
}

/* -------------------- groups -------------------- */

criterion_group!(
    benches,
    bench_insertion,
    bench_lookup,
    bench_mixed_workload,
    bench_growth_behavior,
    bench_recovery_scenarios,
    // parallel variants
    bench_par_insert,
    bench_par_lookup,
    bench_par_mixed,
    bench_par_growth,
    bench_par_recovery,
);
criterion_main!(benches);