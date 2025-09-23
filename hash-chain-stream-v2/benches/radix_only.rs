use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use criterion::measurement::WallTime;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

use HashChainStream::radix_hca::RadixHca;   // import from crate root (important)
use HashChainStream::AnchorConfig;          // optional if you tweak config

fn gen_keys(n: usize, key_len: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let tail: String = (0..key_len.saturating_sub(10))
            .map(|_| {
                let a = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
                a[rng.gen_range(0..a.len())] as char
            })
            .collect();
        v.push(format!("{:010}{}", i, tail));
    }
    v
}

fn tune(group: &mut criterion::BenchmarkGroup<WallTime>, n: usize) {
    if n >= 10_000_000 {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(120));
        group.warm_up_time(Duration::from_secs(5));
    } else if n >= 1_000_000 {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(45));
        group.warm_up_time(Duration::from_secs(3));
    } else if n >= 500_000 {
        group.sample_size(20);
        group.measurement_time(Duration::from_secs(20));
        group.warm_up_time(Duration::from_secs(3));
    } else {
        group.sample_size(30);
        group.measurement_time(Duration::from_secs(10));
        group.warm_up_time(Duration::from_secs(2));
    }
}

fn bench_radix_hca(c: &mut Criterion) {
    let mut group = c.benchmark_group("radix_hca");

    for &n in &[1_000, 10_000, 100_000, 1_000_000, 10_000_000] {
        tune(&mut group, n);
        let keys = gen_keys(n, 20);
        group.throughput(Throughput::Elements(n as u64));

        // INSERT
        group.bench_with_input(BenchmarkId::new("insert", n), &keys, |b, keys| {
            b.iter(|| {
                let mut h = RadixHca::<String, usize>::with_default_config();
                for (i, k) in keys.iter().enumerate() {
                    h.insert(k.clone(), i).unwrap();
                }
                criterion::black_box(h);
            });
        });

        // LOOKUP (50/50 mix)
        let mut h = RadixHca::<String, usize>::with_default_config();
        for (i, k) in keys.iter().enumerate() {
            h.insert(k.clone(), i).unwrap();
        }
        let mut lookups = keys[..(keys.len() / 2)].to_vec();
        lookups.extend(gen_keys(keys.len() / 2, 20));
        group.throughput(Throughput::Elements(lookups.len() as u64));

        group.bench_with_input(BenchmarkId::new("lookup", n), &lookups, |b, ks| {
            b.iter(|| {
                for k in ks.iter() {
                    criterion::black_box(h.get(k));
                }
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_radix_hca);
criterion_main!(benches);
