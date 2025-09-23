use std::sync::RwLock;

use crate::common::AnchorConfig;
use crate::hashchain_stream::HashChainStream;
use crate::direct_anchor_hca::DirectAnchorHca;
use crate::HcaError;

/// A thread-safe, sharded wrapper around DirectAnchorHca.
/// - Shards are chosen by hashing the key (HashChainStream) and mod SHARDS.
/// - Each shard is a DirectAnchorHca protected by an RwLock.
/// - get() clones the value; insert/remove mirror the inner API.
///
/// NOTE: get() returns `Option<V>` (clone) to avoid holding a read-lock past the call.
/// If you need zero-clone returns, store `Arc<V>` at the call site or change V to Arc<V>.
pub struct ConcurrentDirectAnchorHca<K, V, const SHARDS: usize = 32>
where
    K: Clone + Default + PartialEq + AsRef<[u8]>,
    V: Clone + Default,
{
    config: AnchorConfig,
    shards: Vec<RwLock<DirectAnchorHca<K, V>>>,
}

impl<K, V, const SHARDS: usize> ConcurrentDirectAnchorHca<K, V, SHARDS>
where
    K: Clone + Default + PartialEq + AsRef<[u8]>,
    V: Clone + Default,
{
    #[inline]
    fn shard_index_with_config(key: &K, cfg: &AnchorConfig) -> usize {
        let mut s = HashChainStream::with_hash_mode(key.as_ref(), &cfg.domain, cfg.hash_mode);
        // Use 4 nibbles (16 bits) for the shard choice; modulo for any SHARDS
        (s.prefix(4) as usize) % SHARDS
    }

    #[inline]
    fn shard_index(&self, key: &K) -> usize {
        Self::shard_index_with_config(key, &self.config)
    }

    pub fn new(config: AnchorConfig) -> Self {
        let mut shards = Vec::with_capacity(SHARDS);
        for _ in 0..SHARDS {
            shards.push(RwLock::new(DirectAnchorHca::new(config.clone())));
        }
        Self { config, shards }
    }

    pub fn with_default_config() -> Self {
        Self::new(AnchorConfig::default())
    }

    /// Insert using only a single-shard write lock.
    /// Returns old value if key existed (same as inner).
    pub fn insert(&self, key: K, value: V) -> Result<Option<V>, HcaError> {
        let idx = self.shard_index(&key);
        let mut guard = self.shards[idx].write().unwrap();
        guard.insert(key, value)
    }

    /// Get using only a single-shard read lock.
    /// Returns a cloned value to avoid holding a lock after returning.
    pub fn get(&self, key: &K) -> Option<V> {
        let idx = self.shard_index(key);
        let guard = self.shards[idx].read().unwrap();
        guard.get(key).cloned()
    }

    /// Remove using a single-shard write lock.
    pub fn remove(&self, key: &K) -> Option<V> {
        let idx = self.shard_index(key);
        let mut guard = self.shards[idx].write().unwrap();
        guard.remove(key)
    }

    /// Aggregate stats across shards.
    pub fn stats(&self) -> crate::direct_anchor_hca::DirectAnchorStats {
        use crate::direct_anchor_hca::DirectAnchorStats;
        let mut total_anchors = 0usize;
        let mut total_entries = 0usize;
        let mut max_depth = 0usize;
        let mut min_depth = usize::MAX;

        for shard in &self.shards {
            let g = shard.read().unwrap();
            let s = g.stats();
            total_anchors += s.total_anchors;
            total_entries += s.total_entries;
            max_depth = max_depth.max(s.max_depth);
            min_depth = min_depth.min(s.min_depth);
        }
        if total_anchors == 0 {
            min_depth = 0;
        }

        DirectAnchorStats {
            total_anchors,
            total_entries,
            max_depth,
            min_depth,
            avg_entries_per_anchor: if total_anchors > 0 {
                total_entries as f64 / total_anchors as f64
            } else {
                0.0
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    // Helper: works whether get() returns Option<V> or Option<&V>
    fn got_equals<T: Clone + PartialEq>(got: Option<T>, expect: T) -> bool {
        match got {
            Some(v) => v == expect,
            None => false,
        }
    }

    #[test]
    fn single_thread_api_parity() {
        let mut h = ConcurrentDirectAnchorHca::<String, i32>::with_default_config();
        assert_eq!(h.insert("a".into(), 1).unwrap(), None);
        assert!(got_equals(h.get(&"a".into()), 1));
        assert_eq!(h.insert("a".into(), 2).unwrap(), Some(1));
        assert!(got_equals(h.get(&"a".into()), 2));
        assert_eq!(h.remove(&"a".into()), Some(2));
        assert!(h.get(&"a".into()).is_none());
    }

    #[test]
    fn parallel_inserts_all_present() {
        const N: usize = 8;       // threads
        const PER: usize = 5_000; // keys per thread
        let h = Arc::new(ConcurrentDirectAnchorHca::<String, usize>::with_default_config());
        let start = Arc::new(Barrier::new(N));

        // Insert ranges in parallel
        let mut handles = Vec::new();
        for t in 0..N {
            let h = Arc::clone(&h);
            let b = Arc::clone(&start);
            handles.push(thread::spawn(move || {
                b.wait();
                let begin = t * PER;
                let end = begin + PER;
                for i in begin..end {
                    let k = format!("k{:08}", i);
                    let _ = h.insert(k, i).unwrap();
                }
            }));
        }
        for j in handles {
            j.join().unwrap();
        }

        // Verify presence
        for i in 0..(N * PER) {
            let k = format!("k{:08}", i);
            let got = h.get(&k);
            assert!(got_equals(got, i), "missing key {}", k);
        }
    }

    #[test]
    fn parallel_mixed_smoke() {
        const N: usize = 8;
        const OPS: usize = 10_000;

        let h = Arc::new(ConcurrentDirectAnchorHca::<String, i32>::with_default_config());
        let start = Arc::new(Barrier::new(N));
        let mut handles = Vec::new();

        for t in 0..N {
            let h = Arc::clone(&h);
            let b = Arc::clone(&start);
            handles.push(thread::spawn(move || {
                b.wait();
                // deterministic but different regions to reduce contention
                let base = (t as i32) * 1_000_000;
                for i in 0..OPS {
                    match i % 4 {
                        0 | 1 => {
                            let k = format!("k{:08}", base + (i as i32));
                            let _ = h.insert(k, i as i32).unwrap();
                        }
                        2 => {
                            let k = format!("k{:08}", base + (i as i32));
                            let _ = h.get(&k);
                        }
                        3 => {
                            let k = format!("k{:08}", base + (i as i32));
                            let _ = h.remove(&k);
                        }
                        _ => unreachable!(),
                    }
                }
            }));
        }
        for j in handles {
            j.join().unwrap();
        }

        // Basic usability after stress
        let _ = h.insert("final".into(), 42).unwrap();
        assert!(got_equals(h.get(&"final".into()), 42));
        assert_eq!(h.remove(&"final".into()), Some(42));
    }
}
