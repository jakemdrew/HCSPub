use std::marker::PhantomData;
use crate::hashchain_stream::{HashChainStream, HashMode};

const MAX_PROBE_GROUPS: usize = 8;
const GROUP_WIDTH: usize = 16;

// CHANGED: use top-bit as tombstone flag (mask, not a value)
const DELETED_HASH: u32 = 0x8000_0000;

#[derive(Debug)]
struct Shard<K, V> {
    slots: Vec<Option<OptimizedSlot<K, V>>>,
    mask: usize,
    len: usize,
}

impl<K, V> Shard<K, V> {
    fn with_capacity_pow2(cap: usize) -> Self {
        let cap = cap.max(GROUP_WIDTH).next_power_of_two();
        let slots = std::iter::repeat_with(|| None).take(cap).collect::<Vec<_>>();
        Self { slots, mask: cap - 1, len: 0 }
    }

    #[inline]
    fn index(&self, h: u32) -> usize { (h as usize) & self.mask }
}

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

pub struct SimpleChainedHashMap<K, V, const HASH_BITS: u32> {
    shard_bits: u32,
    shards: Vec<Shard<K, V>>,
    hcs: HashChainStream,
    _kv: PhantomData<(K, V)>,
}

impl<K, V, const HASH_BITS: u32> SimpleChainedHashMap<K, V, HASH_BITS>
where
    K: Eq + Clone + AsRef<[u8]> + 'static,
    V: Clone,
{
    pub fn new(shard_count: usize, capacity: usize) -> Self {
        Self::new_with_mode_and_domain(shard_count, capacity, b"SimpleChainedHashMap.v1", HashMode::Fast)
    }
    pub fn new_with_domain(shard_count: usize, capacity: usize, domain: &[u8]) -> Self {
        Self::new_with_mode_and_domain(shard_count, capacity, domain, HashMode::Fast)
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
        
        // Allocate for 7/8 load factor: need (keys * 8/7) slots
        let target_slots = ((keys_per_shard as f64) * (8.0 / 7.0)).ceil() as usize;
        
        let max_bits_per_shard = (HASH_BITS - shard_bits).min(20);
        let per_shard = target_slots.next_power_of_two().min(1 << max_bits_per_shard);

        let shards = (0..shard_pow2).map(|_| Shard::with_capacity_pow2(per_shard)).collect();
        let hcs = HashChainStream::new(domain, hash_mode);

        Self { shard_bits, shards, hcs, _kv: PhantomData }
    }

    /// Migrate to larger capacity WITHOUT rehashing
    pub fn from_map_with_capacity(old_map: &Self, new_capacity: usize, domain: &[u8]) -> Self {
        let old_shard_count = old_map.shards.len();
        
        let mut new_map = Self::new_with_mode_and_domain(
            old_shard_count,  // Keep same shard count
            new_capacity, 
            domain, 
            old_map.hcs.hash_mode()
        );
        
        // Migrate shard-to-shard (no cross-shard movement, no rehashing)
        for shard_idx in 0..old_map.shards.len() {
            let old_shard = &old_map.shards[shard_idx];
            let new_shard = &mut new_map.shards[shard_idx];
            
            for slot_opt in old_shard.slots.iter() {
                if let Some(slot) = slot_opt {
                    if (slot.full_hash & DELETED_HASH) == 0 {
                        // Direct insertion into same shard using stored hash
                        let mut pos = new_shard.index(slot.full_hash);
                        let mut stride = 0;
                        
                        'insert: for _ in 0..MAX_PROBE_GROUPS {
                            for j in 0..GROUP_WIDTH {
                                let idx = (pos + j) & new_shard.mask;
                                if new_shard.slots[idx].is_none() {
                                    new_shard.slots[idx] = Some(slot.clone());
                                    new_shard.len += 1;
                                    break 'insert;
                                }
                            }
                            stride += GROUP_WIDTH;
                            pos = (pos + stride) & new_shard.mask;
                        }
                    }
                }
            }
        }
        
        new_map
    }
    
    #[inline]
    fn shard_index(&self, h: u32) -> usize {
        if self.shard_bits == 0 { 0 } else { (h >> (HASH_BITS - self.shard_bits)) as usize }
    }

    // CHANGED: branchless; clear the tombstone flag from live hashes
    #[inline]
    fn sanitize_hash(h: u32) -> u32 {
        h & !DELETED_HASH
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let search_hash = Self::sanitize_hash(self.hcs.bits(key.as_ref(), 0, HASH_BITS as usize) as u32);
        let sidx = self.shard_index(search_hash);
        let shard = &mut self.shards[sidx];

        let mut pos = shard.index(search_hash);
        let mut stride = 0;
        let mut first_tomb: Option<usize> = None;

        for _ in 0..MAX_PROBE_GROUPS {
            macro_rules! check_slot {
                ($i:expr) => {{
                    let idx = (pos + $i) & shard.mask;
                    match &mut shard.slots[idx] {
                        None => {
                            let tgt = first_tomb.unwrap_or(idx);
                            if tgt == idx {
                                shard.slots[idx] = Some(OptimizedSlot::new(search_hash, key, value));
                            } else {
                                shard.slots[tgt] = Some(OptimizedSlot::new(search_hash, key, value));
                            }
                            shard.len += 1;
                            // REMOVED: if shard.load_factor_exceeded() { self.resize_shard(sidx); }
                            return None;
                        }
                        Some(slot) => {
                            if (slot.full_hash & DELETED_HASH) != 0 {
                                if first_tomb.is_none() { first_tomb = Some(idx); }
                            } else if slot.full_hash == search_hash && slot.key == key {
                                let old = slot.value.clone();
                                slot.value = value;
                                return Some(old);
                            }
                        }
                    }
                }};
            }

            check_slot!(0);  check_slot!(1);  check_slot!(2);  check_slot!(3);
            check_slot!(4);  check_slot!(5);  check_slot!(6);  check_slot!(7);
            check_slot!(8);  check_slot!(9);  check_slot!(10); check_slot!(11);
            check_slot!(12); check_slot!(13); check_slot!(14); check_slot!(15);

            stride += GROUP_WIDTH;
            pos = (pos + stride) & shard.mask;
        }

        // REMOVED: self.resize_shard(sidx);
        // REMOVED: self.insert(key, value)
        panic!("Insert exhausted {} probe groups; shard capacity exceeded", MAX_PROBE_GROUPS);
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        let search_hash = Self::sanitize_hash(self.hcs.bits(key.as_ref(), 0, HASH_BITS as usize) as u32);
        let sidx = self.shard_index(search_hash);
        let shard = &self.shards[sidx];

        let mut pos = shard.index(search_hash);
        let mut stride = 0;

        for _ in 0..MAX_PROBE_GROUPS {
            macro_rules! check_slot {
                ($i:expr) => {{
                    let idx = (pos + $i) & shard.mask;
                    match &shard.slots[idx] {
                        None => return None, // true empty terminates
                        Some(slot) => {
                            if (slot.full_hash & DELETED_HASH) == 0
                                && slot.full_hash == search_hash
                                && slot.key == *key
                            {
                                return Some(&slot.value);
                            }
                        }
                    }
                }};
            }

            check_slot!(0);  check_slot!(1);  check_slot!(2);  check_slot!(3);
            check_slot!(4);  check_slot!(5);  check_slot!(6);  check_slot!(7);
            check_slot!(8);  check_slot!(9);  check_slot!(10); check_slot!(11);
            check_slot!(12); check_slot!(13); check_slot!(14); check_slot!(15);

            stride += GROUP_WIDTH;
            pos = (pos + stride) & shard.mask;
        }

        None
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let search_hash = Self::sanitize_hash(self.hcs.bits(key.as_ref(), 0, HASH_BITS as usize) as u32);
        let sidx = self.shard_index(search_hash);
        let shard = &mut self.shards[sidx];

        let mut pos = shard.index(search_hash);
        let mut stride = 0;

        for _ in 0..MAX_PROBE_GROUPS {
            macro_rules! check_slot {
                ($i:expr) => {{
                    let idx = (pos + $i) & shard.mask;
                    match &mut shard.slots[idx] {
                        None => return None, // true empty terminates
                        Some(slot) => {
                            if (slot.full_hash & DELETED_HASH) == 0
                                && slot.full_hash == search_hash
                                && slot.key == *key
                            {
                                let old_value = slot.value.clone();
                                // mark tombstone
                                slot.full_hash = DELETED_HASH;
                                shard.len -= 1;
                                return Some(old_value);
                            }
                        }
                    }
                }};
            }

            check_slot!(0);  check_slot!(1);  check_slot!(2);  check_slot!(3);
            check_slot!(4);  check_slot!(5);  check_slot!(6);  check_slot!(7);
            check_slot!(8);  check_slot!(9);  check_slot!(10); check_slot!(11);
            check_slot!(12); check_slot!(13); check_slot!(14); check_slot!(15);

            stride += GROUP_WIDTH;
            pos = (pos + stride) & shard.mask;
        }

        None
    }

    // REMOVED: fn resize_shard(&mut self, sidx: usize) { ... }

    pub fn clear_reuse(&mut self) {
        for shard in &mut self.shards {
            shard.slots.fill(None);
            shard.len = 0;
        }
    }
}