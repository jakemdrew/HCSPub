use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hash_chain_stream::{
    SimpleChainedHashMap, ChainedHashMap, HashMode,
};
use std::collections::HashMap;
use ahash::AHashMap;

// Helper to generate test keys
fn generate_keys(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("key_{:016x}", i))
        .collect()
}

// Benchmark SimpleChainedHashMap migration
fn bench_simple_migration(c: &mut Criterion) {
    let sizes = vec![10_000, 100_000, 1_000_000, 10_000_000];
    
    for &size in &sizes {
        let mut group = c.benchmark_group("migration/simple_chained_hash_map");
        group.throughput(Throughput::Elements(size as u64));
        group.sample_size(10);
        
        // Pre-generate keys and populate old map
        let keys = generate_keys(size);
        let mut old_map = SimpleChainedHashMap::<String, u64, 26>::new(64, size);
        for (i, key) in keys.iter().enumerate() {
            old_map.insert(key.clone(), i as u64);
        }
        
        // Benchmark migration to 2x capacity
        group.bench_with_input(
            BenchmarkId::new("2x_capacity", size),
            &size,
            |b, &_size| {
                b.iter(|| {
                    let new_map = SimpleChainedHashMap::<String, u64, 26>::from_map_with_capacity(
                        black_box(&old_map),
                        black_box(size * 2),
                        b"migration_bench"
                    );
                    black_box(new_map);
                });
            }
        );
        
        // Benchmark migration to 4x capacity
        group.bench_with_input(
            BenchmarkId::new("4x_capacity", size),
            &size,
            |b, &_size| {
                b.iter(|| {
                    let new_map = SimpleChainedHashMap::<String, u64, 26>::from_map_with_capacity(
                        black_box(&old_map),
                        black_box(size * 4),
                        b"migration_bench"
                    );
                    black_box(new_map);
                });
            }
        );
        
        group.finish();
    }
}

// Benchmark ChainedHashMap migration
fn bench_chained_migration(c: &mut Criterion) {
    let sizes = vec![10_000, 100_000, 1_000_000, 10_000_000];
    
    for &size in &sizes {
        let mut group = c.benchmark_group("migration/chained_hash_map");
        group.throughput(Throughput::Elements(size as u64));
        group.sample_size(10);
        
        // Pre-generate keys and populate old map
        let keys = generate_keys(size);
        let old_map = ChainedHashMap::<String, u64, 26>::with_capacity(
            size,
            b"migration_bench",
            HashMode::Fast
        );
        for (i, key) in keys.iter().enumerate() {
            old_map.insert(key.clone(), i as u64);
        }
        
        // Benchmark migration to 2x capacity
        group.bench_with_input(
            BenchmarkId::new("2x_capacity", size),
            &size,
            |b, &_size| {
                b.iter(|| {
                    let new_map = ChainedHashMap::<String, u64, 26>::from_map_with_capacity(
                        black_box(&old_map),
                        black_box(size * 2),
                        b"migration_bench"
                    );
                    black_box(new_map);
                });
            }
        );
        
        // Benchmark migration to 4x capacity
        group.bench_with_input(
            BenchmarkId::new("4x_capacity", size),
            &size,
            |b, &_size| {
                b.iter(|| {
                    let new_map = ChainedHashMap::<String, u64, 26>::from_map_with_capacity(
                        black_box(&old_map),
                        black_box(size * 4),
                        b"migration_bench"
                    );
                    black_box(new_map);
                });
            }
        );
        
        group.finish();
    }
}

// Benchmark migration with different shard counts (ChainedHashMap crossing thresholds)
fn bench_chained_migration_shard_change(c: &mut Criterion) {
    let mut group = c.benchmark_group("migration/chained_shard_change");
    group.sample_size(10);
    
    // 500K → 5M: crosses 1M threshold (128 shards → 256 shards)
    let size = 500_000;
    let keys = generate_keys(size);
    let old_map = ChainedHashMap::<String, u64, 26>::with_capacity(
        size,
        b"migration_bench",
        HashMode::Fast
    );
    for (i, key) in keys.iter().enumerate() {
        old_map.insert(key.clone(), i as u64);
    }
    
    group.throughput(Throughput::Elements(size as u64));
    group.bench_function("500k_to_5m", |b| {
        b.iter(|| {
            let new_map = ChainedHashMap::<String, u64, 26>::from_map_with_capacity(
                black_box(&old_map),
                black_box(5_000_000),
                b"migration_bench"
            );
            black_box(new_map);
        });
    });
    
    group.finish();
}

// Compare migration vs rebuild (including HashMap and AHashMap)
fn bench_migration_vs_rebuild(c: &mut Criterion) {
    let sizes = vec![100_000, 1_000_000];
    
    for &size in &sizes {
        let mut group = c.benchmark_group(format!("migration_vs_rebuild/{}", size));
        group.throughput(Throughput::Elements(size as u64));
        group.sample_size(10);
        
        let keys = generate_keys(size);
        
        // Setup SimpleChainedHashMap
        let mut old_map_simple = SimpleChainedHashMap::<String, u64, 26>::new(64, size);
        for (i, key) in keys.iter().enumerate() {
            old_map_simple.insert(key.clone(), i as u64);
        }
        
        // Setup ChainedHashMap
        let old_map_chained = ChainedHashMap::<String, u64, 26>::with_capacity(
            size,
            b"migration_bench",
            HashMode::Fast
        );
        for (i, key) in keys.iter().enumerate() {
            old_map_chained.insert(key.clone(), i as u64);
        }
        
        // Setup std::HashMap
        let mut old_hashmap = HashMap::new();
        for (i, key) in keys.iter().enumerate() {
            old_hashmap.insert(key.clone(), i as u64);
        }
        
        // Setup AHashMap
        let mut old_ahashmap = AHashMap::new();
        for (i, key) in keys.iter().enumerate() {
            old_ahashmap.insert(key.clone(), i as u64);
        }
        
        // SimpleChainedHashMap: Migration
        group.bench_function("simple_migration", |b| {
            b.iter(|| {
                let new_map = SimpleChainedHashMap::<String, u64, 26>::from_map_with_capacity(
                    black_box(&old_map_simple),
                    black_box(size * 2),
                    b"migration_bench"
                );
                black_box(new_map);
            });
        });
        
        // SimpleChainedHashMap: Rebuild (create new + insert all)
        group.bench_function("simple_rebuild", |b| {
            b.iter(|| {
                let mut new_map = SimpleChainedHashMap::<String, u64, 26>::new(64, size * 2);
                for (i, key) in keys.iter().enumerate() {
                    new_map.insert(key.clone(), i as u64);
                }
                black_box(new_map);
            });
        });
        
        // ChainedHashMap: Migration
        group.bench_function("chained_migration", |b| {
            b.iter(|| {
                let new_map = ChainedHashMap::<String, u64, 26>::from_map_with_capacity(
                    black_box(&old_map_chained),
                    black_box(size * 2),
                    b"migration_bench"
                );
                black_box(new_map);
            });
        });
        
        // ChainedHashMap: Rebuild
        group.bench_function("chained_rebuild", |b| {
            b.iter(|| {
                let new_map = ChainedHashMap::<String, u64, 26>::with_capacity(
                    size * 2,
                    b"migration_bench",
                    HashMode::Fast
                );
                for (i, key) in keys.iter().enumerate() {
                    new_map.insert(key.clone(), i as u64);
                }
                black_box(new_map);
            });
        });
        
        // std::HashMap: Rebuild (no migration support)
        group.bench_function("std_hashmap_rebuild", |b| {
            b.iter(|| {
                let mut new_map = HashMap::with_capacity(size * 2);
                for (k, v) in old_hashmap.iter() {
                    new_map.insert(k.clone(), *v);
                }
                black_box(new_map);
            });
        });
        
        // AHashMap: Rebuild (no migration support)
        group.bench_function("ahashmap_rebuild", |b| {
            b.iter(|| {
                let mut new_map = AHashMap::with_capacity(size * 2);
                for (k, v) in old_ahashmap.iter() {
                    new_map.insert(k.clone(), *v);
                }
                black_box(new_map);
            });
        });
        
        group.finish();
    }
}

// Direct comparison: All maps side-by-side
fn bench_all_maps_comparison(c: &mut Criterion) {
    let size = 1_000_000;
    let keys = generate_keys(size);
    
    let mut group = c.benchmark_group("all_maps_comparison");
    group.throughput(Throughput::Elements(size as u64));
    group.sample_size(10);
    
    // SimpleChainedHashMap with migration
    let mut old_simple = SimpleChainedHashMap::<String, u64, 26>::new(64, size);
    for (i, key) in keys.iter().enumerate() {
        old_simple.insert(key.clone(), i as u64);
    }
    group.bench_function("simple_chained_migration", |b| {
        b.iter(|| {
            SimpleChainedHashMap::<String, u64, 26>::from_map_with_capacity(
                black_box(&old_simple),
                black_box(size * 2),
                b"bench"
            )
        });
    });
    
    // ChainedHashMap with migration
    let old_chained = ChainedHashMap::<String, u64, 26>::with_capacity(size, b"bench", HashMode::Fast);
    for (i, key) in keys.iter().enumerate() {
        old_chained.insert(key.clone(), i as u64);
    }
    group.bench_function("chained_migration", |b| {
        b.iter(|| {
            ChainedHashMap::<String, u64, 26>::from_map_with_capacity(
                black_box(&old_chained),
                black_box(size * 2),
                b"bench"
            )
        });
    });
    
    // std::HashMap rebuild
    let mut old_std = HashMap::new();
    for (i, key) in keys.iter().enumerate() {
        old_std.insert(key.clone(), i as u64);
    }
    group.bench_function("std_hashmap_rebuild", |b| {
        b.iter(|| {
            let mut new_map = HashMap::with_capacity(size * 2);
            for (k, v) in old_std.iter() {
                new_map.insert(k.clone(), *v);
            }
            black_box(new_map);
        });
    });
    
    // AHashMap rebuild
    let mut old_ahash = AHashMap::new();
    for (i, key) in keys.iter().enumerate() {
        old_ahash.insert(key.clone(), i as u64);
    }
    group.bench_function("ahashmap_rebuild", |b| {
        b.iter(|| {
            let mut new_map = AHashMap::with_capacity(size * 2);
            for (k, v) in old_ahash.iter() {
                new_map.insert(k.clone(), *v);
            }
            black_box(new_map);
        });
    });
    
    group.finish();
}

criterion_group!(
    migration_benches,
    bench_simple_migration,
    bench_chained_migration,
    bench_chained_migration_shard_change,
    bench_migration_vs_rebuild,
    bench_all_maps_comparison
);

criterion_main!(migration_benches);