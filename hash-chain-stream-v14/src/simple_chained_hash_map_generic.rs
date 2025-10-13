use std::marker::PhantomData;
use crate::common::OptimizedSlot;
use crate::hashchain_stream::{HashChainStream, HashMode};

const MAX_PROBE_GROUPS: usize = 8;
const GROUP_WIDTH: usize = 16;

#[derive(Debug)]
struct Shard<K, V> {
    slots: Vec<Option<OptimizedSlot<K, V>>>,
    mask: usize,
    len: usize,
}

impl<K, V> Shard<K, V> {
    fn with_capacity_pow2(cap: usize) -> Self {
        let cap = cap.max(GROUP_WIDTH).next_power_of_two();
        let slots = std::iter::repeat_with(|| None)
            .take(cap)
            .collect::<Vec<_>>();

        Self {
            slots,
            mask: cap - 1,
            len: 0,
        }
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
        let target_slots = (keys_per_shard * 20).max(256);

        let max_bits_per_shard = (HASH_BITS - shard_bits).min(20);
        let per_shard = target_slots
            .next_power_of_two()
            .min(1 << max_bits_per_shard);

        let shards = (0..shard_pow2)
            .map(|_| Shard::with_capacity_pow2(per_shard))
            .collect();

        let hcs = HashChainStream::new(domain, hash_mode);

        Self {
            shard_bits,
            shards,
            hcs,
            _kv: PhantomData,
        }
    }

    #[inline]
    fn shard_index(&self, h: u32) -> usize {
        if self.shard_bits == 0 {
            0
        } else {
            (h >> (HASH_BITS - self.shard_bits)) as usize
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash_trunc = self.hcs.bits(key.as_ref(), 0, HASH_BITS as usize) as u32;
        let sidx = self.shard_index(hash_trunc);
        let shard = &mut self.shards[sidx];

        let mut pos = shard.index(hash_trunc);
        let mut stride = 0;

        for _ in 0..MAX_PROBE_GROUPS {
            // ONLY CHANGE: Unrolled inner loop
            macro_rules! check_slot {
                ($i:expr) => {{
                    let idx = (pos + $i) & shard.mask;
                    match &mut shard.slots[idx] {
                        None => {
                            shard.slots[idx] = Some(OptimizedSlot::new(hash_trunc, key, value));
                            shard.len += 1;
                            if shard.load_factor_exceeded() {
                                self.resize_shard(sidx);
                            }
                            return None;
                        }
                        Some(slot) => {
                            if slot.full_hash == hash_trunc && slot.key == key {
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
            pos = (pos + stride) & shard.mask;
        }

        self.resize_shard(sidx);
        self.insert(key, value)
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        let hash_trunc = self.hcs.bits(key.as_ref(), 0, HASH_BITS as usize) as u32;
        let sidx = self.shard_index(hash_trunc);
        let shard = &self.shards[sidx];

        let mut pos = shard.index(hash_trunc);
        let mut stride = 0;

        for _ in 0..MAX_PROBE_GROUPS {
            // ONLY CHANGE: Unrolled inner loop
            macro_rules! check_slot {
                ($i:expr) => {{
                    let idx = (pos + $i) & shard.mask;
                    match &shard.slots[idx] {
                        None => return None,
                        Some(slot) => {
                            if slot.full_hash == hash_trunc && slot.key == *key {
                                return Some(&slot.value);
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
            pos = (pos + stride) & shard.mask;
        }

        None
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let hash_trunc = self.hcs.bits(key.as_ref(), 0, HASH_BITS as usize) as u32;
        let sidx = self.shard_index(hash_trunc);
        let shard = &mut self.shards[sidx];

        let mut pos = shard.index(hash_trunc);
        let mut stride = 0;

        for _ in 0..MAX_PROBE_GROUPS {
            // ONLY CHANGE: Unrolled inner loop
            macro_rules! check_slot {
                ($i:expr) => {{
                    let idx = (pos + $i) & shard.mask;
                    match &shard.slots[idx] {
                        None => return None,
                        Some(slot) => {
                            if slot.full_hash == hash_trunc && slot.key == *key {
                                let old_value = slot.value.clone();
                                shard.slots[idx] = None;
                                shard.len -= 1;
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
            pos = (pos + stride) & shard.mask;
        }

        None
    }

    fn resize_shard(&mut self, sidx: usize) {
        let shard = &mut self.shards[sidx];
        let old_slots = std::mem::take(&mut shard.slots);
        let new_size = old_slots.len() * 2;

        shard.slots = std::iter::repeat_with(|| None)
            .take(new_size)
            .collect::<Vec<_>>();
        shard.mask = new_size - 1;
        shard.len = 0;

        for slot_opt in old_slots.into_iter() {
            if let Some(slot) = slot_opt {
                let mut pos = shard.index(slot.full_hash);
                let mut stride = 0;

                'insert: for _ in 0..MAX_PROBE_GROUPS {
                    for j in 0..GROUP_WIDTH {
                        let idx = (pos + j) & shard.mask;
                        if shard.slots[idx].is_none() {
                            shard.slots[idx] = Some(slot);
                            shard.len += 1;
                            break 'insert;
                        }
                    }
                    stride += GROUP_WIDTH;
                    pos = (pos + stride) & shard.mask;
                }
            }
        }
    }
    
    pub fn clear_reuse(&mut self) {
        for shard in &mut self.shards {
            shard.slots.fill(None);
            shard.len = 0;
        }
    }
}