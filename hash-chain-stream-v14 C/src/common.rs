#![allow(dead_code)]
use crate::hashchain_stream::HashMode;

#[derive(Debug, Clone)]
pub struct AnchorConfig {
    pub s_max: usize,
    pub default_depth: usize,
    pub domain: Vec<u8>,
    pub hash_mode: HashMode,

    // NEW (shared across all classes)
    pub hash_bits: u32,   // unified HASH_BITS for full_hash truncation
    pub max_depth: usize, // unified depth cap for escalation
}

impl Default for AnchorConfig {
    fn default() -> Self {
        Self {
            s_max: 8,
            default_depth: 1,
            domain: b"anchor_hca".to_vec(),
            hash_mode: HashMode::Fast,
            hash_bits: 25,   // matches your 25M bench default
            max_depth: 16,   // your prior MAX_DEPTH_DEFAULT
        }
    }
}

/* ---------------------- Microtable (shared) ---------------------- */

pub type FullHashBits = u32;

#[derive(Debug, Clone)]
pub struct OptimizedSlot<K, V> {
    pub(crate) full_hash: u32, // exactly HASH_BITS of entropy (LSB-aligned)
    pub(crate) key: K,
    pub(crate) value: V,
}

impl<K, V> OptimizedSlot<K, V>
where
    K: Clone,
    V: Clone,
{
    #[inline(always)]
    pub(crate) fn new(full_hash: u32, key: K, value: V) -> Self {
        Self { full_hash, key, value }
    }
}