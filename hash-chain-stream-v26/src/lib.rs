/* -------- Public modules -------- */
pub mod hashchain_stream;

/* -------- Generic SimpleChainedHashMap -------- */
mod simple_chained_hash_map_generic;
pub use simple_chained_hash_map_generic::SimpleChainedHashMap;

// Type aliases for convenience
pub type SimpleChainedHashMap_15<K, V> = SimpleChainedHashMap<K, V, 15>;
pub type SimpleChainedHashMap_18<K, V> = SimpleChainedHashMap<K, V, 18>;
pub type SimpleChainedHashMap_22<K, V> = SimpleChainedHashMap<K, V, 22>;
pub type SimpleChainedHashMap_25<K, V> = SimpleChainedHashMap<K, V, 25>;
pub type SimpleChainedHashMap_26<K, V> = SimpleChainedHashMap<K, V, 26>;

/* -------- Generic ChainedHashMap -------- */
mod chained_hash_map_generic;
pub use chained_hash_map_generic::ChainedHashMap;

// Type aliases for convenience
pub type ChainedHashMap_15<K, V> = ChainedHashMap<K, V, 15>;
pub type ChainedHashMap_18<K, V> = ChainedHashMap<K, V, 18>;
pub type ChainedHashMap_22<K, V> = ChainedHashMap<K, V, 22>;
pub type ChainedHashMap_25<K, V> = ChainedHashMap<K, V, 25>;
pub type ChainedHashMap_26<K, V> = ChainedHashMap<K, V, 26>;

/* -------- Re-exports -------- */
pub use hashchain_stream::{HashChainStream, HashMode};

/* -------- Error type -------- */
#[derive(Debug, Clone, PartialEq)]
pub enum HcaError {
    NotFound,
    KeyExists,
    InvalidDepth,
    CorruptedData(String),
}

impl std::fmt::Display for HcaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HcaError::NotFound => write!(f, "Key not found"),
            HcaError::KeyExists => write!(f, "Key already exists"),
            HcaError::InvalidDepth => write!(f, "Invalid depth specified"),
            HcaError::CorruptedData(msg) => write!(f, "Corrupted data: {}", msg),
        }
    }
}

impl std::error::Error for HcaError {}