// src/hashchain_stream.rs
#![allow(non_snake_case)]
use blake2::digest::Digest;
use blake2::Blake2b512;
use xxhash_rust::xxh3::{xxh3_64_with_seed, Xxh3};

/// Fast non-cryptographic vs cryptographic mode.
#[derive(Debug, Clone, Copy)]
pub enum HashMode {
    /// xxh3-based. Blocks are 64 bits (8 bytes).
    Fast,
    /// BLAKE2b-512. Blocks are 512 bits (64 bytes).
    Cryptographic,
}

impl Default for HashMode {
    fn default() -> Self { HashMode::Fast }
}

/// Stateless hash chain stream providing infinite random-access bit extraction.
/// Domain is configured once at construction, keys are passed to methods.
pub struct HashChainStream {
    domain_seed: u64,
    domain_blake: Option<Blake2b512>,
    hash_mode: HashMode,
}

impl HashChainStream {
    /// Create a new stream with the given domain.
    pub fn new(domain: &[u8], hash_mode: HashMode) -> Self {
        match hash_mode {
            HashMode::Fast => {
                // domain_seed = xxh3_64("HCSv1" || domain) unseeded
                let mut st = Xxh3::new();
                st.update(b"HCSv1");
                st.update(domain);
                let domain_seed = st.digest();
                
                Self {
                    domain_seed,
                    domain_blake: None,
                    hash_mode,
                }
            }
            HashMode::Cryptographic => {
                let mut st = Blake2b512::new();
                st.update(b"HCSv1");
                st.update(&(domain.len() as u64).to_be_bytes());
                st.update(domain);
                
                Self {
                    domain_seed: 0,
                    domain_blake: Some(st),
                    hash_mode,
                }
            }
        }
    }

    /// Create with default Fast mode (for compatibility).
    pub fn with_hash_mode(key: &[u8], domain: &[u8], hash_mode: HashMode) -> Self {
        let _ = key; // key is unused in stateless design
        Self::new(domain, hash_mode)
    }
    
    #[inline]
    pub fn hash_mode(&self) -> HashMode {
        self.hash_mode  // Return the hash_mode field
    }

    /// Helper: compute block i in Fast mode (1 hash per block)
    #[inline]
    fn fast_block_be(&self, key: &[u8], block_index: usize) -> u64 {
        if block_index == 0 {
            xxh3_64_with_seed(key, self.domain_seed).to_be()
        } else {
            let mut hasher = Xxh3::with_seed(self.domain_seed);
            hasher.update(key);
            hasher.update(&(block_index as u64).to_be_bytes());
            hasher.digest().to_be()
        }
    }

    /// MSB-first bit extraction:
    /// - `bit_offset` 0 targets the block's most-significant bit.
    /// - `bit_count` in 1..=64; return value is a big-endian integer of those bits.
    #[inline]
    pub fn bits(&self, key: &[u8], bit_offset: usize, bit_count: usize) -> u64 {
        debug_assert!(bit_count > 0 && bit_count <= 64, "bit_count must be 1-64");
        
        match self.hash_mode {
            HashMode::Fast => {
                let block_index = bit_offset >> 6;  // / 64
                let offset_in_block = bit_offset & 63;  // % 64
                
                let block_hash = self.fast_block_be(key, block_index);
                
                // Fast path: full block at offset 0
                if bit_count == 64 && offset_in_block == 0 {
                    return block_hash;
                }
                
                if offset_in_block + bit_count <= 64 {
                    // Fits in one block
                    extract_bits_u64_be(block_hash, offset_in_block, bit_count)
                } else {
                    // Spans two blocks
                    let next_index = block_index.checked_add(1)
                        .expect("bit_offset overflow into next block");
                    let next_block_hash = self.fast_block_be(key, next_index);
                    
                    let bits_from_block0 = 64 - offset_in_block;
                    let bits_from_block1 = bit_count - bits_from_block0;
                    
                    let part0 = extract_bits_u64_be(block_hash, offset_in_block, bits_from_block0);
                    let part1 = extract_bits_u64_be(next_block_hash, 0, bits_from_block1);
                    
                    (part0 << bits_from_block1) | part1
                }
            }
            HashMode::Cryptographic => {
                let key_seed_state = compute_key_seed_crypto(
                    self.domain_blake.as_ref().expect("blake not initialized"), 
                    key
                );
                let block_index = bit_offset / 512;
                let offset_in_block = bit_offset % 512;
                
                if offset_in_block + bit_count <= 512 {
                    let block = compute_block_crypto_bytes(&key_seed_state, block_index);
                    extract_bits_from_bytes(&block, offset_in_block, bit_count)
                } else {
                    let block0 = compute_block_crypto_bytes(&key_seed_state, block_index);
                    let block1 = compute_block_crypto_bytes(&key_seed_state, block_index + 1);
                    
                    let bits_from_block0 = 512 - offset_in_block;
                    let part0 = extract_bits_from_bytes(&block0, offset_in_block, bits_from_block0);
                    let bits_from_block1 = bit_count - bits_from_block0;
                    let part1 = extract_bits_from_bytes(&block1, 0, bits_from_block1);
                    (part0 << bits_from_block1) | part1
                }
            }
        }
    }  
    
    /// Convenience: extract first N bits (for hash tables, DirectAnchorHca)
    #[inline]
    pub fn nibbles(&self, key: &[u8], bit_count: usize) -> u64 {
        self.bits(key, 0, bit_count)
    }
    
    /// Convenience: first N nibbles as prefix (for compatibility)
    #[inline]
    pub fn prefix(&self, key: &[u8], depth: usize) -> u64 {
        debug_assert!(depth <= 16, "Depth too large for u64 (max 16 nibbles)");
        self.bits(key, 0, depth * 4)
    }
    
    /// Return the Nth 4-bit nibble (0..15), MSB-first
    #[inline]
    pub fn nibble(&self, key: &[u8], index: usize) -> u8 {
        let bit_offset = index * 4;
        (self.bits(key, bit_offset, 4) & 0x0F) as u8
    }
}

// Fast mode helpers
#[allow(dead_code)]
#[inline]
fn compute_key_seed_fast(domain_seed: u64, key: &[u8]) -> u64 {
    xxh3_64_with_seed(key, domain_seed)
}

#[allow(dead_code)]
#[inline]
fn compute_block_fast_u64(key_seed: u64, block_index: usize) -> u64 {
    let counter = (block_index as u64).to_be_bytes();
    xxh3_64_with_seed(&counter, key_seed).to_be()
}

// Crypto mode helpers

fn compute_key_seed_crypto(base: &Blake2b512, key: &[u8]) -> Blake2b512 {
    let mut st = base.clone();
    st.update(&(key.len() as u64).to_be_bytes());
    st.update(key);
    st
}

fn compute_block_crypto_bytes(key_state: &Blake2b512, block_index: usize) -> [u8; 64] {
    let mut st = key_state.clone();
    st.update(&(block_index as u64).to_be_bytes());
    let digest = st.finalize();
    let mut bytes = [0u8; 64];
    bytes.copy_from_slice(&digest);
    bytes
}

// Extraction helpers

#[inline]
fn extract_bits_u64_be(block_be_u64: u64, offset: usize, count: usize) -> u64 {
    (block_be_u64 << offset) >> (64 - count)
}

fn extract_bits_from_bytes(bytes: &[u8], bit_offset: usize, bit_count: usize) -> u64 {
    debug_assert!(bit_count <= 64);
    
    let byte_offset = bit_offset >> 3;  // / 8
    let bit_offset_in_byte = bit_offset & 7;  // % 8
    
    let mut result: u64 = 0;
    let mut bits_collected = 0;
    
    for i in 0..9 {
        if bits_collected >= bit_count {
            break;
        }
        if byte_offset + i >= bytes.len() {
            break;
        }
        
        let byte = bytes[byte_offset + i];
        let available = 8 - if i == 0 { bit_offset_in_byte } else { 0 };
        let needed = bit_count - bits_collected;
        let take = available.min(needed);
        
        let shifted: u16 = if i == 0 {
            (((byte as u16) << bit_offset_in_byte) >> (8 - take)) & ((1 << take) - 1)
        } else {
            ((byte as u16) >> (8 - take)) & ((1 << take) - 1)
        };
        
        result = (result << take) | shifted as u64;
        bits_collected += take;
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_nibbles() {
        let hcs = HashChainStream::new(b"test_domain", HashMode::Fast);
        let n1 = hcs.nibbles(b"key1", 20);
        let n2 = hcs.nibbles(b"key1", 20);
        assert_eq!(n1, n2);
    }
    
    #[test]
    fn test_different_keys() {
        let hcs = HashChainStream::new(b"domain", HashMode::Fast);
        let n1 = hcs.nibbles(b"key1", 20);
        let n2 = hcs.nibbles(b"key2", 20);
        assert_ne!(n1, n2);
    }
    
    #[test]
    fn test_random_access() {
        let hcs = HashChainStream::new(b"domain", HashMode::Fast);
        let bits = hcs.bits(b"key", 100, 20);
        assert!(bits < (1 << 20));
    }
    
    #[test]
    fn test_prefix_stability() {
        let hcs = HashChainStream::new(b"test_domain", HashMode::Fast);
        let p4 = hcs.prefix(b"test_key", 4);
        let p4_again = hcs.prefix(b"test_key", 4);
        assert_eq!(p4, p4_again);
    }
    
    #[test]
    fn test_cryptographic_mode() {
        let hcs = HashChainStream::new(b"domain", HashMode::Cryptographic);
        let n1 = hcs.nibbles(b"key", 64);
        let n2 = hcs.nibbles(b"key", 64);
        assert_eq!(n1, n2);
    }
}