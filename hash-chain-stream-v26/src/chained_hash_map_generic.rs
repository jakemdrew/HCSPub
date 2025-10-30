use parking_lot::Mutex;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;
use std::thread;

use crate::hashchain_stream::{HashChainStream, HashMode};

const MAX_PROBE_GROUPS: usize = 8;
const GROUP_WIDTH: usize = 16;
const STRIPES_PER_SHARD: usize = 16;
const DELETED_HASH: u32 = 0x8000_0000; // top-bit tombstone

#[derive(Debug, Clone)]
struct OptimizedSlot<K, V> {
    full_hash: u32,
    key: K,
    value: V,
}

impl<K, V> OptimizedSlot<K, V>
where
    K: Clone,
    V: Clone,
{
    #[inline(always)]
    fn new(full_hash: u32, key: K, value: V) -> Self {
        Self { full_hash, key, value }
    }
}

struct ShardData<K, V> {
    slots: Vec<Option<OptimizedSlot<K, V>>>,
    mask: usize,
    len: usize,            // live entries (excl. tombstones)
}

impl<K, V> ShardData<K, V> {
    fn with_capacity_pow2(cap: usize) -> Self {
        let cap = cap.max(GROUP_WIDTH).next_power_of_two();
        let slots = std::iter::repeat_with(|| None).take(cap).collect();

        Self { slots, mask: cap - 1, len: 0 }
    }

    #[inline] fn index(&self, h: u32) -> usize { (h as usize) & self.mask }
}

struct Shard<K, V> {
    data: Mutex<ShardData<K, V>>,
    stripe_locks: [Mutex<()>; STRIPES_PER_SHARD],
    mask: usize,  // Cached from ShardData to avoid extra lock
}

impl<K, V> Shard<K, V> {
    fn with_capacity_pow2(cap: usize) -> Self {
        let shard_data = ShardData::with_capacity_pow2(cap);
        let mask = shard_data.mask;
        let data = Mutex::new(shard_data);
        let stripe_locks: [Mutex<()>; STRIPES_PER_SHARD] = std::array::from_fn(|_| Mutex::new(()));
        Self { data, stripe_locks, mask }
    }

    #[inline]
    fn index(&self, h: u32) -> usize {
        (h as usize) & self.mask
    }

    #[inline]
    fn stripe_for_index(&self, idx: usize) -> usize {
        let capacity = self.mask + 1;
        (idx * STRIPES_PER_SHARD / capacity).min(STRIPES_PER_SHARD - 1)
    }
}

pub struct ChainedHashMap<K, V, const HASH_BITS: u32> {
    shard_bits: u32,
    shards: Vec<Arc<Shard<K, V>>>,
    hcs: HashChainStream,
    _kv: PhantomData<(K, V)>,
}

impl<K, V, const HASH_BITS: u32> ChainedHashMap<K, V, HASH_BITS>
where
    K: Eq + Hash + Send + Sync + Clone + AsRef<[u8]> + 'static,
    V: Send + Sync + Clone + 'static,
{
    // ---------------- constructors ----------------

    pub fn new(shard_count: usize, capacity: usize) -> Self {
        Self::new_with_mode_and_domain(shard_count, capacity, b"ChainedHashMap.v1", HashMode::Fast)
    }

    pub fn new_with_domain(shard_count: usize, capacity: usize, domain: &[u8]) -> Self {
        Self::new_with_mode_and_domain(shard_count, capacity, domain, HashMode::Fast)
    }

    pub fn with_capacity(capacity: usize, domain: &[u8], hash_mode: HashMode) -> Self {
        let shard_count = if capacity <= 100_000 { 64 }
                          else if capacity <= 1_000_000 { 128 }
                          else if capacity <= 10_000_000 { 256 }
                          else { 512 };
        Self::new_with_mode_and_domain(shard_count, capacity, domain, hash_mode)
    }

    pub fn new_with_mode_and_domain(
        shard_count: usize,
        capacity: usize,
        domain: &[u8],
        hash_mode: HashMode,
    ) -> Self {
        assert!(shard_count > 0);
        let shard_pow2 = shard_count.next_power_of_two();
        let shard_bits = shard_pow2.trailing_zeros();

        let keys_per_shard = (capacity + shard_pow2 - 1) / shard_pow2;
        // allocate ≈ 8/7 slots per key; clamp by HASH_BITS
        let target_slots = ((keys_per_shard as f64) * (8.0 / 6.0)).ceil() as usize;
        let max_bits_per_shard = (HASH_BITS - shard_bits).min(20);
        let per_shard = target_slots.next_power_of_two().min(1 << max_bits_per_shard);

        let shards = (0..shard_pow2)
            .map(|_| Arc::new(Shard::with_capacity_pow2(per_shard)))
            .collect();

        let hcs = HashChainStream::new(domain, hash_mode);
        Self { shard_bits, shards, hcs, _kv: PhantomData }
    }

    /// Migrate to larger capacity WITHOUT rehashing - PARALLEL
    pub fn from_map_with_capacity(old_map: &Self, new_capacity: usize, domain: &[u8]) -> Self {
        // Determine shard count for new map using same logic as with_capacity
        let new_shard_count = if new_capacity <= 100_000 { 64 }
                              else if new_capacity <= 1_000_000 { 128 }
                              else if new_capacity <= 10_000_000 { 256 }
                              else { 512 };
        
        let new_map = Self::new_with_mode_and_domain(
            new_shard_count,
            new_capacity,
            domain,
            old_map.hcs.hash_mode()
        );
        
        // Migrate each old shard in parallel
        thread::scope(|scope| {
            for shard_idx in 0..old_map.shards.len() {
                let old_shard = &old_map.shards[shard_idx];
                let new_map_ref = &new_map;
                
                scope.spawn(move || {
                    let old_data = old_shard.data.lock();
                    
                    for slot_opt in old_data.slots.iter() {
                        if let Some(slot) = slot_opt {
                            if (slot.full_hash & DELETED_HASH) == 0 {
                                // Use stored hash to find new shard (no rehashing!)
                                let full_hash = slot.full_hash;
                                let new_sidx = new_map_ref.shard_index_from_hash(full_hash);
                                let new_shard = &new_map_ref.shards[new_sidx];
                                
                                // Insert directly using stored hash
                                new_map_ref.insert_with_stripe_lock(
                                    full_hash,
                                    slot.key.clone(),
                                    slot.value.clone(),
                                    new_shard
                                );
                            }
                        }
                    }
                });
            }
        });
        
        new_map
    }
    // ---------------- helpers ----------------

    #[inline]
    fn shard_index_from_hash(&self, hash_trunc: u32) -> usize {
        if self.shard_bits == 0 { 0 } else { (hash_trunc >> (HASH_BITS - self.shard_bits)) as usize }
    }

    #[inline]
    fn sanitize_hash(h: u32) -> u32 { h & !DELETED_HASH }

    #[inline]
    fn shard_index(&self, key: &K) -> (u32, usize) {
        let raw = self.hcs.bits(key.as_ref(), 0, HASH_BITS as usize) as u32;
        let hash_trunc = Self::sanitize_hash(raw);
        (hash_trunc, self.shard_index_from_hash(hash_trunc))
    }

    #[inline]
    pub fn shard_count(&self) -> usize { 1usize << self.shard_bits }

    // ---------------- core ops (fixed-capacity, no-resize) ----------------

    fn insert_with_stripe_lock(
        &self,
        full_hash_in: u32,
        key: K,
        value: V,
        shard: &Shard<K, V>,
    ) -> Option<V> {
        let full_hash = Self::sanitize_hash(full_hash_in);

        // NO LOCK - use cached mask
        let start_pos = shard.index(full_hash);
        let stripe_idx = shard.stripe_for_index(start_pos);

        // Lock stripe, then data
        let _stripe_guard = shard.stripe_locks[stripe_idx].lock();
        let mut data = shard.data.lock();

        let mut pos = start_pos;
        let mut stride = 0usize;
        let mut first_tomb: Option<usize> = None;

        macro_rules! check_slot {
            ($i:expr) => {{
                let idx = (pos + $i) & data.mask;
                match &mut data.slots[idx] {
                    None => {
                    
                        let tgt = first_tomb.unwrap_or(idx);
                        data.slots[tgt] = Some(OptimizedSlot::new(full_hash, key, value));
                        data.len += 1;
                        return None;
                    }
                    Some(slot) => {
                        if (slot.full_hash & DELETED_HASH) != 0 {
                            if first_tomb.is_none() { first_tomb = Some(idx); }
                        } else if slot.full_hash == full_hash && slot.key == key {
                            let old = slot.value.clone();
                            slot.value = value;
                            return Some(old);
                        }
                    }
                }
            }};
        }

        for _ in 0..MAX_PROBE_GROUPS {
            check_slot!(0);  check_slot!(1);  check_slot!(2);  check_slot!(3);
            check_slot!(4);  check_slot!(5);  check_slot!(6);  check_slot!(7);
            check_slot!(8);  check_slot!(9);  check_slot!(10); check_slot!(11);
            check_slot!(12); check_slot!(13); check_slot!(14); check_slot!(15);

            stride += GROUP_WIDTH;
            pos = (pos + stride) & data.mask;
        }

        panic!(
            "Insert probe exhausted after {} groups (len={}, capacity={}). Increase initial capacity.",
            MAX_PROBE_GROUPS, data.len, data.slots.len()
        );
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let (full_hash, sidx) = self.shard_index(&key);
        self.insert_with_stripe_lock(full_hash, key, value, &self.shards[sidx])
    }

    #[inline]
    pub fn get(&self, key: &K) -> Option<V> {
        let (hash_trunc, sidx) = self.shard_index(key);
        let data = self.shards[sidx].data.lock();

        let mut pos = data.index(hash_trunc);
        let mut stride = 0usize;

        macro_rules! check_slot {
            ($i:expr) => {{
                let idx = (pos + $i) & data.mask;
                match &data.slots[idx] {
                    None => return None,
                    Some(slot) => {
                        if (slot.full_hash & DELETED_HASH) == 0
                            && slot.full_hash == hash_trunc
                            && slot.key == *key
                        {
                            return Some(slot.value.clone());
                        }
                    }
                }
            }};
        }

        for _ in 0..MAX_PROBE_GROUPS {
            check_slot!(0);  check_slot!(1);  check_slot!(2);  check_slot!(3);
            check_slot!(4);  check_slot!(5);  check_slot!(6);  check_slot!(7);
            check_slot!(8);  check_slot!(9);  check_slot!(10); check_slot!(11);
            check_slot!(12); check_slot!(13); check_slot!(14); check_slot!(15);

            stride += GROUP_WIDTH;
            pos = (pos + stride) & data.mask;
        }
        None
    }

    pub fn remove(&self, key: &K) -> Option<V> {
        let (hash_trunc, sidx) = self.shard_index(key);
        let shard = &self.shards[sidx];

        // NO LOCK - use cached mask
        let start_pos = shard.index(hash_trunc);
        let stripe_idx = shard.stripe_for_index(start_pos);

        // Lock stripe, then data
        let _stripe_guard = shard.stripe_locks[stripe_idx].lock();
        let mut data = shard.data.lock();

        let mut pos = start_pos;
        let mut stride = 0usize;

        macro_rules! check_slot {
            ($i:expr) => {{
                let idx = (pos + $i) & data.mask;
                match &mut data.slots[idx] {
                    None => return None,
                    Some(slot) => {
                        if (slot.full_hash & DELETED_HASH) == 0
                            && slot.full_hash == hash_trunc
                            && slot.key == *key
                        {
                            let old = slot.value.clone();
                            slot.full_hash |= DELETED_HASH; // mark tombstone
                            data.len -= 1;
                            return Some(old);
                        }
                    }
                }
            }};
        }

        for _ in 0..MAX_PROBE_GROUPS {
            check_slot!(0);  check_slot!(1);  check_slot!(2);  check_slot!(3);
            check_slot!(4);  check_slot!(5);  check_slot!(6);  check_slot!(7);
            check_slot!(8);  check_slot!(9);  check_slot!(10); check_slot!(11);
            check_slot!(12); check_slot!(13); check_slot!(14); check_slot!(15);

            stride += GROUP_WIDTH;
            pos = (pos + stride) & data.mask;
        }
        None
    }

    // ---------------- bulk helpers (unchanged behavior, no resize) ----------------

    pub fn par_insert<I>(&self, items: I, threads: usize)
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let mut items_vec: Vec<_> = items.into_iter().collect();
        if items_vec.is_empty() { return; }

        let workers = if threads == 0 {
            thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        } else { threads.max(1) };

        let chunk_size = (items_vec.len() + workers - 1) / workers;
        let mut owned_chunks = Vec::with_capacity(workers);
        while !items_vec.is_empty() {
            let take = chunk_size.min(items_vec.len());
            owned_chunks.push(items_vec.drain(..take).collect::<Vec<_>>());
        }

        thread::scope(|scope| {
            for chunk in owned_chunks {
                scope.spawn(move || {
                    for (k, v) in chunk {
                        let _ = self.insert(k, v);
                    }
                });
            }
        });
    }

    pub fn par_insert_slice(&self, items: &[(K, V)], threads: usize)
    where
        K: Clone + Send + Sync,
        V: Clone + Send + Sync,
    {
        if items.is_empty() { return; }

        let requested = if threads == 0 {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        } else { threads.max(1) };

        let workers = requested.min(items.len());
        let chunk_size = (items.len() + workers - 1) / workers;

        std::thread::scope(|scope| {
            for chunk_idx in 0..workers {
                let start = chunk_idx * chunk_size;
                let end = ((chunk_idx + 1) * chunk_size).min(items.len());
                let chunk = &items[start..end];

                scope.spawn(move || {
                    for (k, v) in chunk.iter().cloned() {
                        let _ = self.insert(k, v);
                    }
                });
            }
        });
    }

    // ---------------- size ----------------

    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.data.lock().len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.data.lock().len == 0)
    }
}