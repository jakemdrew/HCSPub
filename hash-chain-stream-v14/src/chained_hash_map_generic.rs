use parking_lot::Mutex;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;
use std::thread;

use crate::common::OptimizedSlot;
use crate::hashchain_stream::{HashChainStream, HashMode};

const MAX_PROBE_GROUPS: usize = 8;
const GROUP_WIDTH: usize = 16;
const STRIPES_PER_SHARD: usize = 16;

struct ShardData<K, V> {
    slots: Vec<Option<OptimizedSlot<K, V>>>,
    mask: usize,
    len: usize,
}

impl<K, V> ShardData<K, V> {
    fn with_capacity_pow2(cap: usize) -> Self {
        let cap = cap.max(GROUP_WIDTH).next_power_of_two();
        let slots = std::iter::repeat_with(|| None).take(cap).collect();
        Self { slots, mask: cap - 1, len: 0 }
    }

    #[inline]
    fn index(&self, h: u32) -> usize {
        (h as usize) & self.mask
    }

    #[inline]
    fn load_factor_exceeded(&self) -> bool {
        self.len * 8 >= self.slots.len() * 7
    }
}

struct Shard<K, V> {
    data: Mutex<ShardData<K, V>>,
    stripe_locks: Vec<Mutex<()>>,
}

impl<K, V> Shard<K, V> {
    fn with_capacity_pow2(cap: usize) -> Self {
        let data = Mutex::new(ShardData::with_capacity_pow2(cap));
        let stripe_locks = (0..STRIPES_PER_SHARD).map(|_| Mutex::new(())).collect();
        Self { data, stripe_locks }
    }

    #[inline]
    fn stripe_for_index(&self, idx: usize, capacity: usize) -> usize {
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
    pub fn new(shard_count: usize, capacity: usize) -> Self {
        Self::new_with_mode_and_domain(shard_count, capacity, b"ChainedHashMap.v1", HashMode::Fast)
    }

    pub fn new_with_domain(shard_count: usize, capacity: usize, domain: &[u8]) -> Self {
        Self::new_with_mode_and_domain(shard_count, capacity, domain, HashMode::Fast)
    }

    pub fn with_capacity(capacity: usize, domain: &[u8], hash_mode: HashMode) -> Self {
        let shard_count = if capacity <= 100_000 {
            32
        } else if capacity <= 1_000_000 {
            64
        } else if capacity <= 10_000_000 {
            128
        } else {
            256
        };
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
        let target_slots = (keys_per_shard * 20).max(256);

        let max_bits_per_shard = (HASH_BITS - shard_bits).min(20);
        let per_shard = target_slots.next_power_of_two().min(1 << max_bits_per_shard);

        let shards = (0..shard_pow2)
            .map(|_| Arc::new(Shard::with_capacity_pow2(per_shard)))
            .collect();

        let hcs = HashChainStream::new(domain, hash_mode);

        Self { shard_bits, shards, hcs, _kv: PhantomData }
    }

    #[inline]
    fn shard_index_from_hash(&self, hash_trunc: u32) -> usize {
        if self.shard_bits == 0 {
            0
        } else {
            (hash_trunc >> (HASH_BITS - self.shard_bits)) as usize
        }
    }

    #[inline]
    fn shard_index(&self, key: &K) -> (u32, usize) {
        let hash_trunc = self.hcs.bits(key.as_ref(), 0, HASH_BITS as usize) as u32;
        let sidx = self.shard_index_from_hash(hash_trunc);
        (hash_trunc, sidx)
    }

    #[inline]
    pub fn shard_count(&self) -> usize {
        1usize << self.shard_bits
    }

    fn insert_with_stripe_lock(&self, full_hash: u32, key: K, value: V, shard: &Shard<K, V>) -> Option<V> {
        loop {
            let cap_check = shard.data.lock().slots.len();
            let start_pos = shard.data.lock().index(full_hash);
            let stripe = shard.stripe_for_index(start_pos, cap_check);
            
            let _guard = shard.stripe_locks[stripe].lock();
            let mut data = shard.data.lock();
            
            if data.slots.len() != cap_check {
                drop(data);
                drop(_guard);
                continue;
            }
            
            let mut pos = start_pos;
            let mut stride = 0;

            for _ in 0..MAX_PROBE_GROUPS {
                // ONLY CHANGE: Unrolled inner loop
                macro_rules! check_slot {
                    ($i:expr) => {{
                        let idx = (pos + $i) & data.mask;
                        match &mut data.slots[idx] {
                            None => {
                                data.slots[idx] = Some(OptimizedSlot::new(full_hash, key, value));
                                data.len += 1;
                                if data.load_factor_exceeded() {
                                    drop(data);
                                    drop(_guard);
                                    self.resize_shard(shard);
                                }
                                return None;
                            }
                            Some(slot) => {
                                if slot.full_hash == full_hash && slot.key == key {
                                    let old_value = slot.value.clone();
                                    slot.value = value;
                                    return Some(old_value);
                                }
                            }
                        }
                    }};
                }

                check_slot!(0); check_slot!(1); check_slot!(2); check_slot!(3);
                check_slot!(4); check_slot!(5); check_slot!(6); check_slot!(7);
                check_slot!(8); check_slot!(9); check_slot!(10); check_slot!(11);
                check_slot!(12); check_slot!(13); check_slot!(14); check_slot!(15);

                stride += GROUP_WIDTH;
                pos = (pos + stride) & data.mask;
            }

            drop(data);
            drop(_guard);
            self.resize_shard(shard);
        }
    }

    fn resize_shard(&self, shard: &Shard<K, V>) {
        let _guards: Vec<_> = shard.stripe_locks.iter().map(|s| s.lock()).collect();
        let mut data = shard.data.lock();
        
        if !data.load_factor_exceeded() {
            return;
        }
        
        let old_slots = std::mem::take(&mut data.slots);
        let new_size = old_slots.len() * 2;

        data.slots = std::iter::repeat_with(|| None).take(new_size).collect();
        data.mask = new_size - 1;
        data.len = 0;

        for slot_opt in old_slots.into_iter().flatten() {
            let mut pos = data.index(slot_opt.full_hash);
            let mut stride = 0;

            'insert: for _ in 0..MAX_PROBE_GROUPS {
                for j in 0..GROUP_WIDTH {
                    let idx = (pos + j) & data.mask;
                    if data.slots[idx].is_none() {
                        data.slots[idx] = Some(slot_opt);
                        data.len += 1;
                        break 'insert;
                    }
                }
                stride += GROUP_WIDTH;
                pos = (pos + stride) & data.mask;
            }
        }
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let (full_hash, sidx) = self.shard_index(&key);
        self.insert_with_stripe_lock(full_hash, key, value, &self.shards[sidx])
    }

    pub fn par_insert<I>(&self, items: I, threads: usize)
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let mut items_vec: Vec<_> = items.into_iter().collect();
        
        if items_vec.is_empty() {
            return;
        }

        let workers = if threads == 0 {
            thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            threads.max(1)
        };

        let chunk_size = (items_vec.len() + workers - 1) / workers;
        let mut owned_chunks = Vec::with_capacity(workers);
        
        while !items_vec.is_empty() {
            let take = chunk_size.min(items_vec.len());
            let chunk: Vec<_> = items_vec.drain(..take).collect();
            owned_chunks.push(chunk);
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
    
    #[inline]
    pub fn get(&self, key: &K) -> Option<V> {
        let (hash_trunc, sidx) = self.shard_index(key);
        let data = self.shards[sidx].data.lock();

        let mut pos = data.index(hash_trunc);
        let mut stride = 0;

        for _ in 0..MAX_PROBE_GROUPS {
            // ONLY CHANGE: Unrolled inner loop
            macro_rules! check_slot {
                ($i:expr) => {{
                    let idx = (pos + $i) & data.mask;
                    match &data.slots[idx] {
                        None => return None,
                        Some(slot) => {
                            if slot.full_hash == hash_trunc && slot.key == *key {
                                return Some(slot.value.clone());
                            }
                        }
                    }
                }};
            }

            check_slot!(0); check_slot!(1); check_slot!(2); check_slot!(3);
            check_slot!(4); check_slot!(5); check_slot!(6); check_slot!(7);
            check_slot!(8); check_slot!(9); check_slot!(10); check_slot!(11);
            check_slot!(12); check_slot!(13); check_slot!(14); check_slot!(15);

            stride += GROUP_WIDTH;
            pos = (pos + stride) & data.mask;
        }
        None
    }
    
    pub fn par_insert_slice(&self, items: &[(K, V)], threads: usize)
    where
        K: Clone + Send + Sync,
        V: Clone + Send + Sync,
    {
        if items.is_empty() {
            return;
        }

        let requested = if threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            threads.max(1)
        };

        let workers = requested.min(items.len());
        let chunk_size = (items.len() + workers - 1) / workers;

        std::thread::scope(|scope| {
            for chunk_idx in 0..workers {
                let start = chunk_idx * chunk_size;
                let end = ((chunk_idx + 1) * chunk_size).min(items.len());
                let chunk = &items[start..end];

                scope.spawn(move || {
                    for (k, v) in chunk {
                        let _ = self.insert(k.clone(), v.clone());
                    }
                });
            }
        });
    }

    pub fn remove(&self, key: &K) -> Option<V> {
        let (hash_trunc, sidx) = self.shard_index(key);
        let shard = &self.shards[sidx];
        
        let cap_check = shard.data.lock().slots.len();
        let start_pos = shard.data.lock().index(hash_trunc);
        let stripe = shard.stripe_for_index(start_pos, cap_check);
        
        let _guard = shard.stripe_locks[stripe].lock();
        let mut data = shard.data.lock();

        let mut pos = start_pos;
        let mut stride = 0;

        for _ in 0..MAX_PROBE_GROUPS {
            // ONLY CHANGE: Unrolled inner loop
            macro_rules! check_slot {
                ($i:expr) => {{
                    let idx = (pos + $i) & data.mask;
                    match &data.slots[idx] {
                        None => return None,
                        Some(slot) => {
                            if slot.full_hash == hash_trunc && slot.key == *key {
                                let old_value = slot.value.clone();
                                data.slots[idx] = None;
                                data.len -= 1;
                                return Some(old_value);
                            }
                        }
                    }
                }};
            }

            check_slot!(0); check_slot!(1); check_slot!(2); check_slot!(3);
            check_slot!(4); check_slot!(5); check_slot!(6); check_slot!(7);
            check_slot!(8); check_slot!(9); check_slot!(10); check_slot!(11);
            check_slot!(12); check_slot!(13); check_slot!(14); check_slot!(15);

            stride += GROUP_WIDTH;
            pos = (pos + stride) & data.mask;
        }

        None
    }

    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.data.lock().len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.data.lock().len == 0)
    }
}