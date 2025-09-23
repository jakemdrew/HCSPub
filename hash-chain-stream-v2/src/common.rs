#![allow(dead_code)]
use crate::hashchain_stream::HashMode;

// Shared constants/types used by both modules
pub const MAX_DEPTH_DEFAULT: usize = 16;

#[derive(Debug, Clone)]
pub struct AnchorConfig {
    pub s_max: usize,          // Maximum entries per microtable (leaf capacity)
    pub default_depth: usize,  // Starting depth (nibble index) for anchoring
    pub domain: Vec<u8>,       // Domain separation tag
    pub hash_mode: HashMode,   // Hash function mode (Fast vs Cryptographic)
}

impl Default for AnchorConfig {
    fn default() -> Self {
        Self {
            s_max: 8,
            default_depth: 1,                 // clamp away from 0 by default
            domain: b"anchor_hca".to_vec(),
            hash_mode: HashMode::Fast,
        }
    }
}

// Slot is shared; keep internal to crate
#[derive(Debug, Clone)]
#[repr(C)]
pub(crate) struct OptimizedSlot<K, V> {
    pub(crate) fingerprint: u16,
    pub(crate) state: u8,   // 0=empty, 1=tombstone, 2=occupied
    pub(crate) _padding: u8,
    pub(crate) key: K,
    pub(crate) value: V,
}

impl<K, V> Default for OptimizedSlot<K, V>
where
    K: Default,
    V: Default,
{
    fn default() -> Self {
        Self {
            fingerprint: 0,
            state: 0,
            _padding: 0,
            key: K::default(),
            value: V::default(),
        }
    }
}
