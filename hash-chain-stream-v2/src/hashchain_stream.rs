// src/hashchain_stream.rs
#![allow(non_snake_case)]
use blake2::digest::Digest;
use blake2::Blake2b512;
use xxhash_rust::xxh3::{xxh3_64, Xxh3};

/// Fast non-cryptographic vs cryptographic mode.
#[derive(Debug, Clone, Copy)]
pub enum HashMode {
    /// xxh3-based. Blocks are 64 bits (8 bytes = 16 nibbles).
    Fast,
    /// BLAKE2b-512. Blocks are 512 bits (64 bytes = 128 nibbles).
    Cryptographic,
}

impl Default for HashMode {
    fn default() -> Self { HashMode::Fast }
}

/// Internal cached block storage (stack-allocated, no heap).
enum BlockCache {
    Fast  { index: usize, bytes: [u8; 8]  },
    Crypto{ index: usize, bytes: [u8; 64] },
}

/// Streaming, prefix-stable hash chain.
/// - Domain separation via `(domain, key)` framing/seed.
/// - Infinite stream via 64-bit block counter (big-endian).
/// - Nibbles are MSB-first within bytes and blocks.
pub struct HashChainStream {
    hash_mode: HashMode,

    // Fast mode seed derived once from b"HCSv1" || domain || key.
    fast_seed: Option<u64>,

    // Cryptographic mode base state after framed absorbs.
    base_blake: Option<Blake2b512>,

    // Array-backed cached block to avoid per-block allocations.
    block_cache: Option<BlockCache>,

    // Kept for debugging/inspection (not used on hot path).
    #[allow(dead_code)]
    key: Vec<u8>,
    #[allow(dead_code)]
    domain: Vec<u8>,
}

impl HashChainStream {
    /// Create a new stream (defaults to `HashMode::Fast`).
    pub fn new(key: &[u8], domain: &[u8]) -> Self {
        Self::with_hash_mode(key, domain, HashMode::default())
    }

    /// Create a new stream in cryptographic mode.
    pub fn new_cryptographic(key: &[u8], domain: &[u8]) -> Self {
        Self::with_hash_mode(key, domain, HashMode::Cryptographic)
    }

    /// Create with a specific mode. Builds a base so per-block work is tiny.
    pub fn with_hash_mode(key: &[u8], domain: &[u8], hash_mode: HashMode) -> Self {
        let key_vec = key.to_vec();
        let domain_vec = domain.to_vec();

        let (fast_seed, base_blake) = match hash_mode {
            HashMode::Fast => {
                // Derive a stable 64-bit seed from domain+key once.
                let mut mix = Vec::with_capacity(5 + domain_vec.len() + key_vec.len());
                mix.extend_from_slice(b"HCSv1");
                mix.extend_from_slice(&domain_vec);
                mix.extend_from_slice(&key_vec);
                (Some(xxh3_64(&mix)), None)
            }
            HashMode::Cryptographic => {
                // Frame input once: tag || len(domain) || domain || len(key) || key
                let mut st = Blake2b512::new();
                st.update(b"HCSv1");
                st.update(&u64::to_be_bytes(domain_vec.len() as u64));
                st.update(&domain_vec);
                st.update(&u64::to_be_bytes(key_vec.len() as u64));
                st.update(&key_vec);
                (None, Some(st))
            }
        };

        Self {
            hash_mode,
            fast_seed,
            base_blake,
            block_cache: None,
            key: key_vec,
            domain: domain_vec,
        }
    }

    /// Return the `index`-th 4-bit nibble (0..15), MSB-first within bytes/blocks.
    #[inline]
    pub fn nibble(&mut self, index: usize) -> u8 {
        match self.hash_mode {
            HashMode::Fast => {
                // 16 nibbles per 8-byte block
                let blk = self.get_block_fast(index / 16);
                let off = index % 16;
                let b = blk[off / 2];
                if (off & 1) == 0 { b >> 4 } else { b & 0x0F }
            }
            HashMode::Cryptographic => {
                // 128 nibbles per 64-byte block
                let blk = self.get_block_crypto(index / 128);
                let off = index % 128;
                let b = blk[off / 2];
                if (off & 1) == 0 { b >> 4 } else { b & 0x0F }
            }
        }
    }

    /// First `depth` nibbles as u64 (max 16 nibbles). MSB-first.
    #[inline]
    pub fn prefix(&mut self, depth: usize) -> u64 {
        assert!(depth <= 16, "Depth too large for u64 (max 16 nibbles)");
        match self.hash_mode {
            HashMode::Fast => {
                let blk = self.get_block_fast(0);
                let mut x = u64::from_be_bytes(blk);
                if depth < 16 { x >>= 64 - depth * 4; }
                x
            }
            HashMode::Cryptographic => {
                let blk = self.get_block_crypto(0);
                let mut bytes8 = [0u8; 8];
                bytes8.copy_from_slice(&blk[..8]);
                let mut x = u64::from_be_bytes(bytes8);
                if depth < 16 { x >>= 64 - depth * 4; }
                x
            }
        }
    }

    /// First `depth` nibbles as u128 (max 32 nibbles). MSB-first.
    #[inline]
    pub fn prefix_u128(&mut self, depth: usize) -> u128 {
        assert!(depth <= 32, "Depth too large for u128 (max 32 nibbles)");
        if depth <= 16 {
            self.prefix(depth) as u128
        } else {
            let mut acc = (self.prefix(16) as u128) << ((depth - 16) * 4);
            for i in 16..depth {
                acc |= (self.nibble(i) as u128) << ((depth - 1 - i) * 4);
            }
            acc
        }
    }

    /// Exactly `depth` nibbles as lowercase hex (no separators).
    #[inline]
    pub fn prefix_hex(&mut self, depth: usize) -> String {
        let mut s = String::with_capacity(depth);
        for i in 0..depth {
            let n = self.nibble(i) as u32;
            s.push(char::from_digit(n, 16).unwrap());
        }
        s
    }

    /// Convenient 16-bit fingerprint from the top 4 nibbles of block 0 (MSB-first).
    #[inline]
    pub fn fingerprint16(&mut self) -> u16 {
        let n0 = self.nibble(0) as u16;
        let n1 = self.nibble(1) as u16;
        let n2 = self.nibble(2) as u16;
        let n3 = self.nibble(3) as u16;
        (n0 << 12) | (n1 << 8) | (n2 << 4) | n3
    }

    // ---------------- internals ----------------

    /// Return 8-byte block by value (avoids borrow conflicts).
    #[inline]
    fn get_block_fast(&mut self, block_index: usize) -> [u8; 8] {
        if let Some(BlockCache::Fast { index, bytes }) = &self.block_cache {
            if *index == block_index {
                return *bytes; // copy 8 bytes
            }
        }
        let seed = self.fast_seed.expect("fast seed not initialized");
        let mut st = Xxh3::with_seed(seed);
        st.update(&(block_index as u64).to_be_bytes());
        let h = st.digest().to_be_bytes(); // [u8; 8]
        self.block_cache = Some(BlockCache::Fast { index: block_index, bytes: h });
        h
    }

    /// Return 64-byte block by value (avoids borrow conflicts).
    #[inline]
    fn get_block_crypto(&mut self, block_index: usize) -> [u8; 64] {
        if let Some(BlockCache::Crypto { index, bytes }) = &self.block_cache {
            if *index == block_index {
                return *bytes; // copy 64 bytes
            }
        }
        let mut st = self.base_blake.as_ref().expect("blake base not initialized").clone();
        st.update(&(block_index as u64).to_be_bytes());
        let digest = st.finalize();
        let mut bytes = [0u8; 64];
        bytes.copy_from_slice(digest.as_slice());
        self.block_cache = Some(BlockCache::Crypto { index: block_index, bytes });
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism_and_range() {
        let mut a = HashChainStream::new(b"key", b"domain");
        let mut b = HashChainStream::new(b"key", b"domain");
        for i in 0..200 {
            assert_eq!(a.nibble(i), b.nibble(i));
            assert!(a.nibble(i) <= 0xF);
        }
    }

    #[test]
    fn prefix_stability() {
        let mut s = HashChainStream::new(b"test_key", b"test_domain");
        let p4 = s.prefix(4);
        for i in 4..100 { let _ = s.nibble(i); }
        let p4_again = s.prefix(4);
        assert_eq!(p4, p4_again);
    }

    #[test]
    fn different_keys_domains_diverge() {
        let mut s1 = HashChainStream::new(b"key1", b"domain");
        let mut s2 = HashChainStream::new(b"key2", b"domain");
        let v1: Vec<u8> = (0..16).map(|i| s1.nibble(i)).collect();
        let v2: Vec<u8> = (0..16).map(|i| s2.nibble(i)).collect();
        assert_ne!(v1, v2);

        let mut s3 = HashChainStream::new(b"key", b"domain1");
        let mut s4 = HashChainStream::new(b"key", b"domain2");
        let w1: Vec<u8> = (0..16).map(|i| s3.nibble(i)).collect();
        let w2: Vec<u8> = (0..16).map(|i| s4.nibble(i)).collect();
        assert_ne!(w1, w2);
    }

    #[test]
    fn cryptographic_mode_basic() {
        let mut s = HashChainStream::new_cryptographic(b"k", b"d");
        let _ = s.nibble(0);
        let _ = s.nibble(127);
        let _ = s.nibble(128);
        let _p16 = s.prefix_u128(16);
        let hx = s.prefix_hex(20);
        assert_eq!(hx.len(), 20);
    }

    #[test]
    fn fingerprint16_consistency() {
        let mut s = HashChainStream::new(b"k", b"d");
        let fp1 = s.fingerprint16();
        let fp2 = s.fingerprint16();
        assert_eq!(fp1, fp2);
    }
}
