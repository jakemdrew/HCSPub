use hash_chain_stream::HashMode;

// Simple map concrete types
use hash_chain_stream::{
    SimpleChainedHashMap_15,
    SimpleChainedHashMap_18,
    SimpleChainedHashMap_22,
    SimpleChainedHashMap_25,
    SimpleChainedHashMap_26,
};

// Chained/parallel map concrete types
use hash_chain_stream::{
    ChainedHashMap_15,
    ChainedHashMap_18,
    ChainedHashMap_22,
    ChainedHashMap_25,
    ChainedHashMap_26,
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use criterion::measurement::WallTime;
use criterion::BatchSize;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::thread;
use std::time::Duration;

use std::collections::HashMap;
use ahash::AHashMap;
use criterion::SamplingMode;

const BENCH_SIZES: &[usize] = &[10_000, 100_000, 1_000_000, 10_000_000, 25_000_000];

fn generate_test_keys(count: usize, key_length: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..count)
        .map(|_| {
            (0..key_length)
                .map(|_| {
                    let chars = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
                    chars[rng.gen_range(0..chars.len())] as char
                })
                .collect()
        })
        .collect()
}

fn tune_group(group: &mut criterion::BenchmarkGroup<WallTime>, size: usize) {
    group.sampling_mode(SamplingMode::Flat);
    
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

/* ------------------------------- Simple: insertion ------------------------------- */

fn bench_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion");

    for &size in BENCH_SIZES {
        tune_group(&mut group, size);
        let keys = generate_test_keys(size, 20);
        group.throughput(Throughput::Elements(size as u64));

        let capacity = (size as f64 * 1.25) as usize;

        if size <= 10_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map", size),
                &keys,
                |b, keys| {
                    b.iter_batched(
                        || {
                            let mut map = SimpleChainedHashMap_15::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast);
                            for (i, key) in keys.iter().enumerate() {
                                map.insert(key.clone(), i as u64);
                            }
                            map.clear_reuse();
                            map
                        },
                        |mut map| {
                            for (i, key) in keys.iter().enumerate() {
                                map.insert(key.clone(), i as u64);
                            }
                            criterion::black_box(map);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        } else if size <= 100_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map", size),
                &keys,
                |b, keys| {
                    b.iter_batched(
                        || {
                            let mut map = SimpleChainedHashMap_18::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast);
                            for (i, key) in keys.iter().enumerate() {
                                map.insert(key.clone(), i as u64);
                            }
                            map.clear_reuse();
                            map
                        },
                        |mut map| {
                            for (i, key) in keys.iter().enumerate() {
                                map.insert(key.clone(), i as u64);
                            }
                            criterion::black_box(map);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        } else if size <= 1_000_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map", size),
                &keys,
                |b, keys| {
                    b.iter_batched(
                        || {
                            let mut map = SimpleChainedHashMap_22::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast);
                            for (i, key) in keys.iter().enumerate() {
                                map.insert(key.clone(), i as u64);
                            }
                            map.clear_reuse();
                            map
                        },
                        |mut map| {
                            for (i, key) in keys.iter().enumerate() {
                                map.insert(key.clone(), i as u64);
                            }
                            criterion::black_box(map);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        } else if size <= 25_000_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map", size),
                &keys,
                |b, keys| {
                    b.iter_batched(
                        || {
                            let mut map = SimpleChainedHashMap_25::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast);
                            for (i, key) in keys.iter().enumerate() {
                                map.insert(key.clone(), i as u64);
                            }
                            map.clear_reuse();
                            map
                        },
                        |mut map| {
                            for (i, key) in keys.iter().enumerate() {
                                map.insert(key.clone(), i as u64);
                            }
                            criterion::black_box(map);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        } else {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map", size),
                &keys,
                |b, keys| {
                    b.iter_batched(
                        || {
                            let mut map = SimpleChainedHashMap_26::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast);
                            for (i, key) in keys.iter().enumerate() {
                                map.insert(key.clone(), i as u64);
                            }
                            map.clear_reuse();
                            map
                        },
                        |mut map| {
                            for (i, key) in keys.iter().enumerate() {
                                map.insert(key.clone(), i as u64);
                            }
                            criterion::black_box(map);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }

        /* baselines unchanged */
        group.bench_with_input(BenchmarkId::new("std_hashmap", size), &keys, |b, keys| {
            b.iter_batched(
                || {
                    let mut map: HashMap<String, usize> = HashMap::new();
                    for (i, key) in keys.iter().enumerate() {
                        map.insert(key.clone(), i);
                    }
                    map.clear();
                    map
                },
                |mut map| {
                    for (i, key) in keys.iter().enumerate() {
                        map.insert(key.clone(), i);
                    }
                    criterion::black_box(map);
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("ahashmap", size), &keys, |b, keys| {
            b.iter_batched(
                || {
                    let mut map: AHashMap<String, usize> = AHashMap::new();
                    for (i, key) in keys.iter().enumerate() {
                        map.insert(key.clone(), i);
                    }
                    map.clear();
                    map
                },
                |mut map| {
                    for (i, key) in keys.iter().enumerate() {
                        map.insert(key.clone(), i);
                    }
                    criterion::black_box(map);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/* -------------------------------- Simple: lookup -------------------------------- */

fn bench_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup");

    for &size in BENCH_SIZES {
        tune_group(&mut group, size);
        let keys = generate_test_keys(size, 20);

        // Build and populate ALL maps together BEFORE benchmarking (like v12)
        if size <= 10_000 {
            let mut simple_map = SimpleChainedHashMap_15::new_with_mode_and_domain(64, size, b"bench", HashMode::Fast);
            let mut std_map: HashMap<String, usize> = HashMap::with_capacity(size);
            let mut ahash_map: AHashMap<String, usize> = AHashMap::with_capacity(size);
            
            // Single warm-up loop for all maps
            for (i, key) in keys.iter().enumerate() {
                simple_map.insert(key.clone(), i as u64);
                std_map.insert(key.clone(), i);
                ahash_map.insert(key.clone(), i);
            }
            
            // Build lookup keys after population
            let mut lookup_keys = keys[0..(keys.len() / 2)].to_vec();
            lookup_keys.extend(generate_test_keys(keys.len() / 2, 20));
            group.throughput(Throughput::Elements(lookup_keys.len() as u64));

            // Now benchmark all three maps
            group.bench_with_input(BenchmarkId::new("simple_chained_hash_map", size), &lookup_keys, |b, ks| {
                b.iter(|| {
                    for k in ks.iter() {
                        criterion::black_box(simple_map.get(k));
                    }
                });
            });

            group.bench_with_input(BenchmarkId::new("std_hashmap", size), &lookup_keys, |b, ks| {
                b.iter(|| {
                    for k in ks.iter() {
                        criterion::black_box(std_map.get(k));
                    }
                });
            });

            group.bench_with_input(BenchmarkId::new("ahashmap", size), &lookup_keys, |b, ks| {
                b.iter(|| {
                    for k in ks.iter() {
                        criterion::black_box(ahash_map.get(k));
                    }
                });
            });

        } else if size <= 100_000 {
            let mut simple_map = SimpleChainedHashMap_18::new_with_mode_and_domain(64, size, b"bench", HashMode::Fast);
            let mut std_map: HashMap<String, usize> = HashMap::with_capacity(size);
            let mut ahash_map: AHashMap<String, usize> = AHashMap::with_capacity(size);
            
            for (i, key) in keys.iter().enumerate() {
                simple_map.insert(key.clone(), i as u64);
                std_map.insert(key.clone(), i);
                ahash_map.insert(key.clone(), i);
            }
            
            let mut lookup_keys = keys[0..(keys.len() / 2)].to_vec();
            lookup_keys.extend(generate_test_keys(keys.len() / 2, 20));
            group.throughput(Throughput::Elements(lookup_keys.len() as u64));

            group.bench_with_input(BenchmarkId::new("simple_chained_hash_map", size), &lookup_keys, |b, ks| {
                b.iter(|| {
                    for k in ks.iter() {
                        criterion::black_box(simple_map.get(k));
                    }
                });
            });

            group.bench_with_input(BenchmarkId::new("std_hashmap", size), &lookup_keys, |b, ks| {
                b.iter(|| {
                    for k in ks.iter() {
                        criterion::black_box(std_map.get(k));
                    }
                });
            });

            group.bench_with_input(BenchmarkId::new("ahashmap", size), &lookup_keys, |b, ks| {
                b.iter(|| {
                    for k in ks.iter() {
                        criterion::black_box(ahash_map.get(k));
                    }
                });
            });

        } else if size <= 1_000_000 {
            let mut simple_map = SimpleChainedHashMap_22::new_with_mode_and_domain(64, size, b"bench", HashMode::Fast);
            let mut std_map: HashMap<String, usize> = HashMap::with_capacity(size);
            let mut ahash_map: AHashMap<String, usize> = AHashMap::with_capacity(size);
            
            for (i, key) in keys.iter().enumerate() {
                simple_map.insert(key.clone(), i as u64);
                std_map.insert(key.clone(), i);
                ahash_map.insert(key.clone(), i);
            }
            
            let mut lookup_keys = keys[0..(keys.len() / 2)].to_vec();
            lookup_keys.extend(generate_test_keys(keys.len() / 2, 20));
            group.throughput(Throughput::Elements(lookup_keys.len() as u64));

            group.bench_with_input(BenchmarkId::new("simple_chained_hash_map", size), &lookup_keys, |b, ks| {
                b.iter(|| {
                    for k in ks.iter() {
                        criterion::black_box(simple_map.get(k));
                    }
                });
            });

            group.bench_with_input(BenchmarkId::new("std_hashmap", size), &lookup_keys, |b, ks| {
                b.iter(|| {
                    for k in ks.iter() {
                        criterion::black_box(std_map.get(k));
                    }
                });
            });

            group.bench_with_input(BenchmarkId::new("ahashmap", size), &lookup_keys, |b, ks| {
                b.iter(|| {
                    for k in ks.iter() {
                        criterion::black_box(ahash_map.get(k));
                    }
                });
            });

        } else if size <= 25_000_000 {
            let mut simple_map = SimpleChainedHashMap_25::new_with_mode_and_domain(64, size, b"bench", HashMode::Fast);
            let mut std_map: HashMap<String, usize> = HashMap::with_capacity(size);
            let mut ahash_map: AHashMap<String, usize> = AHashMap::with_capacity(size);
            
            for (i, key) in keys.iter().enumerate() {
                simple_map.insert(key.clone(), i as u64);
                std_map.insert(key.clone(), i);
                ahash_map.insert(key.clone(), i);
            }
            
            let mut lookup_keys = keys[0..(keys.len() / 2)].to_vec();
            lookup_keys.extend(generate_test_keys(keys.len() / 2, 20));
            group.throughput(Throughput::Elements(lookup_keys.len() as u64));

            group.bench_with_input(BenchmarkId::new("simple_chained_hash_map", size), &lookup_keys, |b, ks| {
                b.iter(|| {
                    for k in ks.iter() {
                        criterion::black_box(simple_map.get(k));
                    }
                });
            });

            group.bench_with_input(BenchmarkId::new("std_hashmap", size), &lookup_keys, |b, ks| {
                b.iter(|| {
                    for k in ks.iter() {
                        criterion::black_box(std_map.get(k));
                    }
                });
            });

            group.bench_with_input(BenchmarkId::new("ahashmap", size), &lookup_keys, |b, ks| {
                b.iter(|| {
                    for k in ks.iter() {
                        criterion::black_box(ahash_map.get(k));
                    }
                });
            });

        } else {
            let mut simple_map = SimpleChainedHashMap_26::new_with_mode_and_domain(64, size, b"bench", HashMode::Fast);
            let mut std_map: HashMap<String, usize> = HashMap::with_capacity(size);
            let mut ahash_map: AHashMap<String, usize> = AHashMap::with_capacity(size);
            
            for (i, key) in keys.iter().enumerate() {
                simple_map.insert(key.clone(), i as u64);
                std_map.insert(key.clone(), i);
                ahash_map.insert(key.clone(), i);
            }
            
            let mut lookup_keys = keys[0..(keys.len() / 2)].to_vec();
            lookup_keys.extend(generate_test_keys(keys.len() / 2, 20));
            group.throughput(Throughput::Elements(lookup_keys.len() as u64));

            group.bench_with_input(BenchmarkId::new("simple_chained_hash_map", size), &lookup_keys, |b, ks| {
                b.iter(|| {
                    for k in ks.iter() {
                        criterion::black_box(simple_map.get(k));
                    }
                });
            });

            group.bench_with_input(BenchmarkId::new("std_hashmap", size), &lookup_keys, |b, ks| {
                b.iter(|| {
                    for k in ks.iter() {
                        criterion::black_box(std_map.get(k));
                    }
                });
            });

            group.bench_with_input(BenchmarkId::new("ahashmap", size), &lookup_keys, |b, ks| {
                b.iter(|| {
                    for k in ks.iter() {
                        criterion::black_box(ahash_map.get(k));
                    }
                });
            });
        }
    }

    group.finish();
}


/* ------------------------------ Simple: mixed workload ------------------------------ */

fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");

    for &base_size in BENCH_SIZES {
        tune_group(&mut group, base_size);

        let keys = generate_test_keys(base_size, 20);
        let operations = base_size;
        let capacity = (base_size as f64 * 1.25) as usize;
        group.throughput(Throughput::Elements(operations as u64));

        if base_size <= 10_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map_mixed", base_size),
                &operations,
                |b, &ops| {
                    b.iter_batched(
                        || SimpleChainedHashMap_15::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast),
                        |mut map| {
                            let mut rng = StdRng::seed_from_u64(123);
                            for i in 0..ops {
                                match i % 4 {
                                    0 | 1 => {
                                        let key = &keys[rng.gen_range(0..keys.len())];
                                        map.insert(key.clone(), i as u64);
                                    }
                                    2 => {
                                        let key = &keys[rng.gen_range(0..keys.len())];
                                        criterion::black_box(map.get(key));
                                    }
                                    3 => {
                                        let key = &keys[rng.gen_range(0..keys.len())];
                                        criterion::black_box(map.remove(key));
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            criterion::black_box(map);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        } else if base_size <= 100_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map_mixed", base_size),
                &operations,
                |b, &ops| {
                    b.iter_batched(
                        || SimpleChainedHashMap_18::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast),
                        |mut map| {
                            let mut rng = StdRng::seed_from_u64(123);
                            for i in 0..ops {
                                match i % 4 {
                                    0 | 1 => {
                                        let key = &keys[rng.gen_range(0..keys.len())];
                                        map.insert(key.clone(), i as u64);
                                    }
                                    2 => {
                                        let key = &keys[rng.gen_range(0..keys.len())];
                                        criterion::black_box(map.get(key));
                                    }
                                    3 => {
                                        let key = &keys[rng.gen_range(0..keys.len())];
                                        criterion::black_box(map.remove(key));
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            criterion::black_box(map);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        } else if base_size <= 1_000_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map_mixed", base_size),
                &operations,
                |b, &ops| {
                    b.iter_batched(
                        || SimpleChainedHashMap_22::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast),
                        |mut map| {
                            let mut rng = StdRng::seed_from_u64(123);
                            for i in 0..ops {
                                match i % 4 {
                                    0 | 1 => {
                                        let key = &keys[rng.gen_range(0..keys.len())];
                                        map.insert(key.clone(), i as u64);
                                    }
                                    2 => {
                                        let key = &keys[rng.gen_range(0..keys.len())];
                                        criterion::black_box(map.get(key));
                                    }
                                    3 => {
                                        let key = &keys[rng.gen_range(0..keys.len())];
                                        criterion::black_box(map.remove(key));
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            criterion::black_box(map);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        } else if base_size <= 25_000_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map_mixed", base_size),
                &operations,
                |b, &ops| {
                    b.iter_batched(
                        || SimpleChainedHashMap_25::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast),
                        |mut map| {
                            let mut rng = StdRng::seed_from_u64(123);
                            for i in 0..ops {
                                match i % 4 {
                                    0 | 1 => {
                                        let key = &keys[rng.gen_range(0..keys.len())];
                                        map.insert(key.clone(), i as u64);
                                    }
                                    2 => {
                                        let key = &keys[rng.gen_range(0..keys.len())];
                                        criterion::black_box(map.get(key));
                                    }
                                    3 => {
                                        let key = &keys[rng.gen_range(0..keys.len())];
                                        criterion::black_box(map.remove(key));
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            criterion::black_box(map);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        } else {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map_mixed", base_size),
                &operations,
                |b, &ops| {
                    b.iter_batched(
                        || SimpleChainedHashMap_26::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast),
                        |mut map| {
                            let mut rng = StdRng::seed_from_u64(123);
                            for i in 0..ops {
                                match i % 4 {
                                    0 | 1 => {
                                        let key = &keys[rng.gen_range(0..keys.len())];
                                        map.insert(key.clone(), i as u64);
                                    }
                                    2 => {
                                        let key = &keys[rng.gen_range(0..keys.len())];
                                        criterion::black_box(map.get(key));
                                    }
                                    3 => {
                                        let key = &keys[rng.gen_range(0..keys.len())];
                                        criterion::black_box(map.remove(key));
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            criterion::black_box(map);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }

        /* baselines unchanged */
        group.bench_with_input(
            BenchmarkId::new("std_hashmap_mixed", base_size),
            &operations,
            |b, &ops| {
                b.iter_batched(
                    || HashMap::with_capacity(base_size),
                    |mut map| {
                        let mut rng = StdRng::seed_from_u64(123);
                        for i in 0..ops {
                            match i % 4 {
                                0 | 1 => {
                                    let key = &keys[rng.gen_range(0..keys.len())];
                                    map.insert(key.clone(), i);
                                }
                                2 => {
                                    let key = &keys[rng.gen_range(0..keys.len())];
                                    criterion::black_box(map.get(key));
                                }
                                3 => {
                                    let key = &keys[rng.gen_range(0..keys.len())];
                                    criterion::black_box(map.remove(key));
                                }
                                _ => unreachable!(),
                            }
                        }
                        criterion::black_box(map.len());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ahashmap_mixed", base_size),
            &operations,
            |b, &ops| {
                b.iter_batched(
                    || AHashMap::with_capacity(base_size),
                    |mut map| {
                        let mut rng = StdRng::seed_from_u64(123);
                        for i in 0..ops {
                            match i % 4 {
                                0 | 1 => {
                                    let key = &keys[rng.gen_range(0..keys.len())];
                                    map.insert(key.clone(), i);
                                }
                                2 => {
                                    let key = &keys[rng.gen_range(0..keys.len())];
                                    criterion::black_box(map.get(key));
                                }
                                3 => {
                                    let key = &keys[rng.gen_range(0..keys.len())];
                                    criterion::black_box(map.remove(key));
                                }
                                _ => unreachable!(),
                            }
                        }
                        criterion::black_box(map.len());
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/* ------------------------------ Simple: growth behavior ------------------------------ */

fn bench_growth_behavior(c: &mut Criterion) {
    let mut group = c.benchmark_group("growth_behavior");

    for &total in BENCH_SIZES {
        tune_group(&mut group, total);

        let capacity = (total as f64 * 1.25) as usize;

        if total <= 10_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map_incremental_growth", total),
                &total,
                |b, &_total| {
                    b.iter(|| {
                        let mut map = SimpleChainedHashMap_15::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast);
                        growth_fill_primitive(&mut map, total);
                        criterion::black_box(map);
                    });
                },
            );
        } else if total <= 100_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map_incremental_growth", total),
                &total,
                |b, &_total| {
                    b.iter(|| {
                        let mut map = SimpleChainedHashMap_18::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast);
                        growth_fill_primitive(&mut map, total);
                        criterion::black_box(map);
                    });
                },
            );
        } else if total <= 1_000_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map_incremental_growth", total),
                &total,
                |b, &_total| {
                    b.iter(|| {
                        let mut map = SimpleChainedHashMap_22::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast);
                        growth_fill_primitive(&mut map, total);
                        criterion::black_box(map);
                    });
                },
            );
        } else if total <= 25_000_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map_incremental_growth", total),
                &total,
                |b, &_total| {
                    b.iter(|| {
                        let mut map = SimpleChainedHashMap_25::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast);
                        growth_fill_primitive(&mut map, total);
                        criterion::black_box(map);
                    });
                },
            );
        } else {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map_incremental_growth", total),
                &total,
                |b, &_total| {
                    b.iter(|| {
                        let mut map = SimpleChainedHashMap_26::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast);
                        growth_fill_primitive(&mut map, total);
                        criterion::black_box(map);
                    });
                },
            );
        }

        /* baselines unchanged */
        group.bench_with_input(
            BenchmarkId::new("std_hashmap_incremental_growth", total),
            &total,
            |b, &total| {
                b.iter(|| {
                    let capacity = (total as f64 * 1.25) as usize;
                    let mut map: HashMap<String, usize> = HashMap::with_capacity(capacity);

                    let batch_count = 20usize;
                    let batch_size = (total / batch_count).max(1);

                    for batch in 0..batch_count {
                        let batch_keys = generate_test_keys(batch_size, 15);
                        for (i, k) in batch_keys.iter().enumerate() {
                            map.insert(k.clone(), batch * batch_size + i);
                        }

                        if batch > 0 && batch % 5 == 0 {
                            for k in batch_keys.iter().take(batch_size.min(100)) {
                                criterion::black_box(map.get(k));
                            }
                        }
                    }

                    criterion::black_box(map.len());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ahashmap_incremental_growth", total),
            &total,
            |b, &total| {
                b.iter(|| {
                    let capacity = (total as f64 * 1.25) as usize;
                    let mut map: AHashMap<String, usize> = AHashMap::with_capacity(capacity);

                    let batch_count = 20usize;
                    let batch_size = (total / batch_count).max(1);

                    for batch in 0..batch_count {
                        let batch_keys = generate_test_keys(batch_size, 15);
                        for (i, k) in batch_keys.iter().enumerate() {
                            map.insert(k.clone(), batch * batch_size + i);
                        }

                        if batch > 0 && batch % 5 == 0 {
                            for k in batch_keys.iter().take(batch_size.min(100)) {
                                criterion::black_box(map.get(k));
                            }
                        }
                    }

                    criterion::black_box(map.len());
                });
            },
        );
    }

    group.finish();
}

fn growth_fill_primitive<M>(map: &mut M, total: usize)
where
    M: SimpleInsertGetRemove,
{
    let batch_count = 20usize;
    let batch_size = (total / batch_count).max(1);

    for batch in 0..batch_count {
        let batch_keys = generate_test_keys(batch_size, 15);
        for (i, k) in batch_keys.iter().enumerate() {
            map.insert(k.clone(), (batch * batch_size + i) as u64);
        }

        if batch > 0 && batch % 5 == 0 {
            for k in batch_keys.iter().take(batch_size.min(100)) {
                criterion::black_box(map.get(k));
            }
        }
    }
}

/* ------------------------------- Simple: recovery ------------------------------- */

fn bench_recovery_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("recovery");

    for &size in BENCH_SIZES {
        tune_group(&mut group, size);
        let keys = generate_test_keys(size, 20);
        let anchor_keys: Vec<_> = keys.iter().take(keys.len() / 10).cloned().collect();

        let capacity = (anchor_keys.len() as f64 * 1.25) as usize;

        if size <= 10_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map_reconstruction", size),
                &anchor_keys,
                |b, anchor_keys| {
                    b.iter(|| {
                        let mut map = SimpleChainedHashMap_15::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast);
                        for (i, k) in anchor_keys.iter().enumerate() {
                            map.insert(k.clone(), i as u64);
                        }
                        criterion::black_box(map);
                    });
                },
            );
        } else if size <= 100_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map_reconstruction", size),
                &anchor_keys,
                |b, anchor_keys| {
                    b.iter(|| {
                        let mut map = SimpleChainedHashMap_18::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast);
                        for (i, k) in anchor_keys.iter().enumerate() {
                            map.insert(k.clone(), i as u64);
                        }
                        criterion::black_box(map);
                    });
                },
            );
        } else if size <= 1_000_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map_reconstruction", size),
                &anchor_keys,
                |b, anchor_keys| {
                    b.iter(|| {
                        let mut map = SimpleChainedHashMap_22::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast);
                        for (i, k) in anchor_keys.iter().enumerate() {
                            map.insert(k.clone(), i as u64);
                        }
                        criterion::black_box(map);
                    });
                },
            );
        } else if size <= 25_000_000 {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map_reconstruction", size),
                &anchor_keys,
                |b, anchor_keys| {
                    b.iter(|| {
                        let mut map = SimpleChainedHashMap_25::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast);
                        for (i, k) in anchor_keys.iter().enumerate() {
                            map.insert(k.clone(), i as u64);
                        }
                        criterion::black_box(map);
                    });
                },
            );
        } else {
            group.bench_with_input(
                BenchmarkId::new("simple_chained_hash_map_reconstruction", size),
                &anchor_keys,
                |b, anchor_keys| {
                    b.iter(|| {
                        let mut map = SimpleChainedHashMap_26::new_with_mode_and_domain(64, capacity, b"bench", HashMode::Fast);
                        for (i, k) in anchor_keys.iter().enumerate() {
                            map.insert(k.clone(), i as u64);
                        }
                        criterion::black_box(map);
                    });
                },
            );
        }

        // AHashMap full rebuild baseline
        group.bench_with_input(
            BenchmarkId::new("ahashmap_rebuild", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut rebuilt: AHashMap<String, usize> = AHashMap::with_capacity(keys.len());
                    for (i, k) in keys.iter().enumerate() {
                        rebuilt.insert(k.clone(), i);
                    }
                    criterion::black_box(rebuilt.len());
                });
            },
        );

        // std::HashMap full rebuild baseline (presized)
        group.bench_with_input(
            BenchmarkId::new("full_hashmap_rebuild", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut rebuilt_map: HashMap<String, usize> = HashMap::with_capacity(keys.len());
                    for (i, k) in keys.iter().enumerate() {
                        rebuilt_map.insert(k.clone(), i);
                    }
                    criterion::black_box(rebuilt_map.len());
                });
            },
        );
    }

    group.finish();
}

/* ------------------------------- Helpers ------------------------------- */

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

/// Parallel-only tuning. Keeps sample_size >= 10 (Criterion 0.5.1 minimum),
/// and scales measurement_time so large-N, high-thread runs don't underfill.
fn par_tune_group(group: &mut criterion::BenchmarkGroup<WallTime>, size: usize, threads: usize) {
    group.sampling_mode(SamplingMode::Flat);  // <-- Changed from Linear to Flat
    let sample_size = if size >= 100_000 { 10 } else { 25 };
    group.sample_size(sample_size);
    
    let base_secs = if size >= 25_000_000 {
        12
    } else if size >= 10_000_000 {
        10
    } else if size >= 1_000_000 {
        8
    } else if size >= 100_000 {
        6
    } else {
        4
    };
    
    let bump = match threads {
        t if t >= 16 => 6,
        t if t >= 8  => 3,
        _            => 0,
    };
    
    group.measurement_time(Duration::from_secs((base_secs + bump) as u64));
    group.warm_up_time(Duration::from_secs(2));
    group.nresamples(30); // Pragmatic for large-scale parallel benchmarks
}

#[inline]
fn par_batch_for_size(n: usize) -> BatchSize {
    if n >= 100_000 { BatchSize::LargeInput } else { BatchSize::SmallInput }
}

/* ------------------------------- Parallel: insertion ------------------------------- */

fn bench_par_insert_separated(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion");
    let thread_counts = [2usize, 4, 8, 16];

    for &size in BENCH_SIZES {
        // Generate inputs once per size
        let keys = generate_test_keys(size, 20);
        let items: Vec<(String, u64)> = keys
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, k)| (k, i as u64))
            .collect();

        group.throughput(Throughput::Elements(size as u64));
        let batch = par_batch_for_size(size);

        for &threads in &thread_counts {
            par_tune_group(&mut group, size, threads);

            let bench_id = BenchmarkId::new(
                format!("chained_hash_map_par_separated_t{}", threads),
                size,
            );

            if size <= 10_000 {
                group.bench_with_input(bench_id, &threads, |b, &t| {
                    b.iter_batched(
                        || ChainedHashMap_15::<String, u64>::with_capacity(size, b"bench", HashMode::Fast),
                        |map| {
                            map.par_insert_slice(&items, t);
                            criterion::black_box(map.len());
                        },
                        batch,
                    );
                });
            } else if size <= 100_000 {
                group.bench_with_input(bench_id, &threads, |b, &t| {
                    b.iter_batched(
                        || ChainedHashMap_18::<String, u64>::with_capacity(size, b"bench", HashMode::Fast),
                        |map| {
                            map.par_insert_slice(&items, t);
                            criterion::black_box(map.len());
                        },
                        batch,
                    );
                });
            } else if size <= 1_000_000 {
                group.bench_with_input(bench_id, &threads, |b, &t| {
                    b.iter_batched(
                        || ChainedHashMap_22::<String, u64>::with_capacity(size, b"bench", HashMode::Fast),
                        |map| {
                            map.par_insert_slice(&items, t);
                            criterion::black_box(map.len());
                        },
                        batch,
                    );
                });
            } else if size <= 25_000_000 {
                group.bench_with_input(bench_id, &threads, |b, &t| {
                    b.iter_batched(
                        || ChainedHashMap_25::<String, u64>::with_capacity(size, b"bench", HashMode::Fast),
                        |map| {
                            map.par_insert_slice(&items, t);
                            criterion::black_box(map.len());
                        },
                        batch,
                    );
                });
            } else {
                group.bench_with_input(bench_id, &threads, |b, &t| {
                    b.iter_batched(
                        || ChainedHashMap_26::<String, u64>::with_capacity(size, b"bench", HashMode::Fast),
                        |map| {
                            map.par_insert_slice(&items, t);
                            criterion::black_box(map.len());
                        },
                        batch,
                    );
                });
            }
        }
    }

    group.finish();
}

/* -------------------------------- Parallel: lookup -------------------------------- */

fn bench_par_lookup_separated(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup");
    let thread_counts = [2usize, 4, 8, 16];

    for &threads in &thread_counts {
        for &size in BENCH_SIZES {
            par_tune_group(&mut group, size, threads);

            // Build keys & items once per (threads, size)
            let keys = generate_test_keys(size, 20);
            let items: Vec<(String, u64)> = keys.iter().cloned()
                .enumerate()
                .map(|(i, k)| (k, i as u64))
                .collect();

            // Helper: safe plan builder (non-interleaved for simplicity)
            let build_plan = || {
                let hit_len = keys.len() / 2;
                let misses: Vec<String> = generate_test_keys(keys.len() - hit_len, 20);
                let mut plan: Vec<(bool, usize)> = (0..hit_len).map(|i| (true, i)).collect();
                plan.extend((0..misses.len()).map(|i| (false, i)));
                (plan, misses)
            };

            // Populate map + run for each const generic
            if size <= 10_000 {
                let map = ChainedHashMap_15::<String, u64>::with_capacity(size, b"bench", HashMode::Fast);
                map.par_insert_slice(&items, threads);

                let (plan, misses) = build_plan();
                group.throughput(Throughput::Elements(plan.len() as u64));

                let id = BenchmarkId::new(format!("chained_hash_map_par_separated_t{}", threads), size);
                group.bench_with_input(id, &threads, |b, &t| {
                    let ranges = shard_ranges(plan.len(), t);
                    b.iter(|| {
                        std::thread::scope(|scope| {
                            for (start, end) in ranges.iter().copied() {
                                let plan_slice = &plan[start..end];
                                let keys_ref = &keys;
                                let misses_ref = &misses;
                                let map_ref = &map;
                                scope.spawn(move || {
                                    for &(is_hit, idx) in plan_slice {
                                        let key_ref = if is_hit { &keys_ref[idx] } else { &misses_ref[idx] };
                                        criterion::black_box(map_ref.get(key_ref));
                                    }
                                });
                            }
                        });
                    });
                });
            } else if size <= 100_000 {
                let map = ChainedHashMap_18::<String, u64>::with_capacity(size, b"bench", HashMode::Fast);
                map.par_insert_slice(&items, threads);

                let (plan, misses) = build_plan();
                group.throughput(Throughput::Elements(plan.len() as u64));

                let id = BenchmarkId::new(format!("chained_hash_map_par_separated_t{}", threads), size);
                group.bench_with_input(id, &threads, |b, &t| {
                    let ranges = shard_ranges(plan.len(), t);
                    b.iter(|| {
                        std::thread::scope(|scope| {
                            for (start, end) in ranges.iter().copied() {
                                let plan_slice = &plan[start..end];
                                let keys_ref = &keys;
                                let misses_ref = &misses;
                                let map_ref = &map;
                                scope.spawn(move || {
                                    for &(is_hit, idx) in plan_slice {
                                        let key_ref = if is_hit { &keys_ref[idx] } else { &misses_ref[idx] };
                                        criterion::black_box(map_ref.get(key_ref));
                                    }
                                });
                            }
                        });
                    });
                });
            } else if size <= 1_000_000 {
                let map = ChainedHashMap_22::<String, u64>::with_capacity(size, b"bench", HashMode::Fast);
                map.par_insert_slice(&items, threads);

                let (plan, misses) = build_plan();
                group.throughput(Throughput::Elements(plan.len() as u64));

                let id = BenchmarkId::new(format!("chained_hash_map_par_separated_t{}", threads), size);
                group.bench_with_input(id, &threads, |b, &t| {
                    let ranges = shard_ranges(plan.len(), t);
                    b.iter(|| {
                        std::thread::scope(|scope| {
                            for (start, end) in ranges.iter().copied() {
                                let plan_slice = &plan[start..end];
                                let keys_ref = &keys;
                                let misses_ref = &misses;
                                let map_ref = &map;
                                scope.spawn(move || {
                                    for &(is_hit, idx) in plan_slice {
                                        let key_ref = if is_hit { &keys_ref[idx] } else { &misses_ref[idx] };
                                        criterion::black_box(map_ref.get(key_ref));
                                    }
                                });
                            }
                        });
                    });
                });
            } else if size <= 25_000_000 {
                let map = ChainedHashMap_25::<String, u64>::with_capacity(size, b"bench", HashMode::Fast);
                map.par_insert_slice(&items, threads);

                let (plan, misses) = build_plan();
                group.throughput(Throughput::Elements(plan.len() as u64));

                let id = BenchmarkId::new(format!("chained_hash_map_par_separated_t{}", threads), size);
                group.bench_with_input(id, &threads, |b, &t| {
                    let ranges = shard_ranges(plan.len(), t);
                    b.iter(|| {
                        std::thread::scope(|scope| {
                            for (start, end) in ranges.iter().copied() {
                                let plan_slice = &plan[start..end];
                                let keys_ref = &keys;
                                let misses_ref = &misses;
                                let map_ref = &map;
                                scope.spawn(move || {
                                    for &(is_hit, idx) in plan_slice {
                                        let key_ref = if is_hit { &keys_ref[idx] } else { &misses_ref[idx] };
                                        criterion::black_box(map_ref.get(key_ref));
                                    }
                                });
                            }
                        });
                    });
                });
            } else {
                let map = ChainedHashMap_26::<String, u64>::with_capacity(size, b"bench", HashMode::Fast);
                map.par_insert_slice(&items, threads);

                let (plan, misses) = build_plan();
                group.throughput(Throughput::Elements(plan.len() as u64));

                let id = BenchmarkId::new(format!("chained_hash_map_par_separated_t{}", threads), size);
                group.bench_with_input(id, &threads, |b, &t| {
                    let ranges = shard_ranges(plan.len(), t);
                    b.iter(|| {
                        std::thread::scope(|scope| {
                            for (start, end) in ranges.iter().copied() {
                                let plan_slice = &plan[start..end];
                                let keys_ref = &keys;
                                let misses_ref = &misses;
                                let map_ref = &map;
                                scope.spawn(move || {
                                    for &(is_hit, idx) in plan_slice {
                                        let key_ref = if is_hit { &keys_ref[idx] } else { &misses_ref[idx] };
                                        criterion::black_box(map_ref.get(key_ref));
                                    }
                                });
                            }
                        });
                    });
                });
            }
        }
    }

    group.finish();
}

/* ------------------------------- Parallel: mixed ------------------------------- */
fn bench_par_mixed_separated(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");
    let thread_counts = [2usize, 4, 8, 16];

    for &threads in &thread_counts {
        for &ops in BENCH_SIZES {
            par_tune_group(&mut group, ops, threads);

            // Hoist workload inputs once per (threads, ops)
            let keys = generate_test_keys(ops, 42);
//            let ranges = shard_ranges(ops, threads);
            group.throughput(Throughput::Elements(ops as u64));

            let id = BenchmarkId::new(
                format!("chained_hash_map_mixed_par_separated_t{}", threads),
                ops,
            );

            let batch = par_batch_for_size(ops);

            if ops <= 10_000 {
                group.bench_with_input(id, &threads, |b, &t| {
                    b.iter_batched(
                        || {
                            ChainedHashMap_15::<String, u64>::with_capacity(
                                ops,
                                b"bench",
                                HashMode::Fast,
                            )
                        },
                        |map| {
                            let ranges = shard_ranges(ops, t);
                            std::thread::scope(|scope| {
                                for (start, end) in ranges.iter().copied() {
                                    let slice = &keys[start..end];
                                    let map_ref = &map;
                                    scope.spawn(move || {
                                        let mut rng = StdRng::seed_from_u64(12345 + start as u64);
                                        for i in 0..slice.len() {
                                            match i % 4 {
                                                0 | 1 => {
                                                    let k = &slice[rng.gen_range(0..slice.len())];
                                                    let _ = map_ref.insert(
                                                        k.clone(),
                                                        (start + i) as u64,
                                                    );
                                                }
                                                2 => {
                                                    let k = &slice[rng.gen_range(0..slice.len())];
                                                    criterion::black_box(map_ref.get(k));
                                                }
                                                3 => {
                                                    let k = &slice[rng.gen_range(0..slice.len())];
                                                    criterion::black_box(map_ref.remove(k));
                                                }
                                                _ => unreachable!(),
                                            }
                                        }
                                    });
                                }
                            });
                            criterion::black_box(&map);
                        },
                        batch,
                    );
                });
            } else if ops <= 100_000 {
                group.bench_with_input(id, &threads, |b, &t| {
                    b.iter_batched(
                        || {
                            ChainedHashMap_18::<String, u64>::with_capacity(
                                ops,
                                b"bench",
                                HashMode::Fast,
                            )
                        },
                        |map| {
                            let ranges = shard_ranges(ops, t);
                            std::thread::scope(|scope| {
                                for (start, end) in ranges.iter().copied() {
                                    let slice = &keys[start..end];
                                    let map_ref = &map;
                                    scope.spawn(move || {
                                        let mut rng = StdRng::seed_from_u64(12345 + start as u64);
                                        for i in 0..slice.len() {
                                            match i % 4 {
                                                0 | 1 => {
                                                    let k = &slice[rng.gen_range(0..slice.len())];
                                                    let _ = map_ref.insert(
                                                        k.clone(),
                                                        (start + i) as u64,
                                                    );
                                                }
                                                2 => {
                                                    let k = &slice[rng.gen_range(0..slice.len())];
                                                    criterion::black_box(map_ref.get(k));
                                                }
                                                3 => {
                                                    let k = &slice[rng.gen_range(0..slice.len())];
                                                    criterion::black_box(map_ref.remove(k));
                                                }
                                                _ => unreachable!(),
                                            }
                                        }
                                    });
                                }
                            });
                            criterion::black_box(&map);
                        },
                        batch,
                    );
                });
            } else if ops <= 1_000_000 {
                group.bench_with_input(id, &threads, |b, &t| {
                    b.iter_batched(
                        || {
                            ChainedHashMap_22::<String, u64>::with_capacity(
                                ops,
                                b"bench",
                                HashMode::Fast,
                            )
                        },
                        |map| {
                            let ranges = shard_ranges(ops, t);
                            std::thread::scope(|scope| {
                                for (start, end) in ranges.iter().copied() {
                                    let slice = &keys[start..end];
                                    let map_ref = &map;
                                    scope.spawn(move || {
                                        let mut rng = StdRng::seed_from_u64(12345 + start as u64);
                                        for i in 0..slice.len() {
                                            match i % 4 {
                                                0 | 1 => {
                                                    let k = &slice[rng.gen_range(0..slice.len())];
                                                    let _ = map_ref.insert(
                                                        k.clone(),
                                                        (start + i) as u64,
                                                    );
                                                }
                                                2 => {
                                                    let k = &slice[rng.gen_range(0..slice.len())];
                                                    criterion::black_box(map_ref.get(k));
                                                }
                                                3 => {
                                                    let k = &slice[rng.gen_range(0..slice.len())];
                                                    criterion::black_box(map_ref.remove(k));
                                                }
                                                _ => unreachable!(),
                                            }
                                        }
                                    });
                                }
                            });
                            criterion::black_box(&map);
                        },
                        batch,
                    );
                });
            } else if ops <= 25_000_000 {
                group.bench_with_input(id, &threads, |b, &t| {
                    b.iter_batched(
                        || {
                            ChainedHashMap_25::<String, u64>::with_capacity(
                                ops,
                                b"bench",
                                HashMode::Fast,
                            )
                        },
                        |map| {
                            let ranges = shard_ranges(ops, t);
                            std::thread::scope(|scope| {
                                for (start, end) in ranges.iter().copied() {
                                    let slice = &keys[start..end];
                                    let map_ref = &map;
                                    scope.spawn(move || {
                                        let mut rng = StdRng::seed_from_u64(12345 + start as u64);
                                        for i in 0..slice.len() {
                                            match i % 4 {
                                                0 | 1 => {
                                                    let k = &slice[rng.gen_range(0..slice.len())];
                                                    let _ = map_ref.insert(
                                                        k.clone(),
                                                        (start + i) as u64,
                                                    );
                                                }
                                                2 => {
                                                    let k = &slice[rng.gen_range(0..slice.len())];
                                                    criterion::black_box(map_ref.get(k));
                                                }
                                                3 => {
                                                    let k = &slice[rng.gen_range(0..slice.len())];
                                                    criterion::black_box(map_ref.remove(k));
                                                }
                                                _ => unreachable!(),
                                            }
                                        }
                                    });
                                }
                            });
                            criterion::black_box(&map);
                        },
                        batch,
                    );
                });
            } else {
                group.bench_with_input(id, &threads, |b, &t| {
                    b.iter_batched(
                        || {
                            ChainedHashMap_26::<String, u64>::with_capacity(
                                ops,
                                b"bench",
                                HashMode::Fast,
                            )
                        },
                        |map| {
                            let ranges = shard_ranges(ops, t);
                            std::thread::scope(|scope| {
                                for (start, end) in ranges.iter().copied() {
                                    let slice = &keys[start..end];
                                    let map_ref = &map;
                                    scope.spawn(move || {
                                        let mut rng = StdRng::seed_from_u64(12345 + start as u64);
                                        for i in 0..slice.len() {
                                            match i % 4 {
                                                0 | 1 => {
                                                    let k = &slice[rng.gen_range(0..slice.len())];
                                                    let _ = map_ref.insert(
                                                        k.clone(),
                                                        (start + i) as u64,
                                                    );
                                                }
                                                2 => {
                                                    let k = &slice[rng.gen_range(0..slice.len())];
                                                    criterion::black_box(map_ref.get(k));
                                                }
                                                3 => {
                                                    let k = &slice[rng.gen_range(0..slice.len())];
                                                    criterion::black_box(map_ref.remove(k));
                                                }
                                                _ => unreachable!(),
                                            }
                                        }
                                    });
                                }
                            });
                            criterion::black_box(&map);
                        },
                        batch,
                    );
                });
            }
        }
    }

    group.finish();
}

/* -------------------- minimal trait to reuse growth_fill_primitive -------------------- */

trait SimpleInsertGetRemove {
    fn insert(&mut self, k: String, v: u64);
    fn get(&self, k: &String) -> Option<&u64>;
    fn remove(&mut self, k: &String) -> Option<u64>;
}
impl SimpleInsertGetRemove for SimpleChainedHashMap_15<String, u64> {
    fn insert(&mut self, k: String, v: u64) { let _ = Self::insert(self, k, v); }
    fn get(&self, k: &String) -> Option<&u64> { Self::get(self, k) }
    fn remove(&mut self, k: &String) -> Option<u64> { Self::remove(self, k) }
}
impl SimpleInsertGetRemove for SimpleChainedHashMap_18<String, u64> {
    fn insert(&mut self, k: String, v: u64) { let _ = Self::insert(self, k, v); }
    fn get(&self, k: &String) -> Option<&u64> { Self::get(self, k) }
    fn remove(&mut self, k: &String) -> Option<u64> { Self::remove(self, k) }
}
impl SimpleInsertGetRemove for SimpleChainedHashMap_22<String, u64> {
    fn insert(&mut self, k: String, v: u64) { let _ = Self::insert(self, k, v); }
    fn get(&self, k: &String) -> Option<&u64> { Self::get(self, k) }
    fn remove(&mut self, k: &String) -> Option<u64> { Self::remove(self, k) }
}
impl SimpleInsertGetRemove for SimpleChainedHashMap_25<String, u64> {
    fn insert(&mut self, k: String, v: u64) { let _ = Self::insert(self, k, v); }
    fn get(&self, k: &String) -> Option<&u64> { Self::get(self, k) }
    fn remove(&mut self, k: &String) -> Option<u64> { Self::remove(self, k) }
}
impl SimpleInsertGetRemove for SimpleChainedHashMap_26<String, u64> {
    fn insert(&mut self, k: String, v: u64) { let _ = Self::insert(self, k, v); }
    fn get(&self, k: &String) -> Option<&u64> { Self::get(self, k) }
    fn remove(&mut self, k: &String) -> Option<u64> { Self::remove(self, k) }
}

criterion_group!(
    benches,
    bench_lookup,
    bench_insertion,
    bench_mixed_workload,
    bench_growth_behavior,
    bench_recovery_scenarios,
    bench_par_insert_separated,
    bench_par_lookup_separated,
    bench_par_mixed_separated,
);
criterion_main!(benches);
