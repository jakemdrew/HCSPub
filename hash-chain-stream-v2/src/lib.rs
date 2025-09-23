#![allow(non_snake_case)]
pub mod hashchain_stream;
pub mod common;
pub mod anchor_hca;
pub mod direct_anchor_hca;

pub use hashchain_stream::{HashChainStream, HashMode}; // <-- single re-export
pub use common::AnchorConfig;
pub use anchor_hca::{AnchorHca, AnchorHcaStats};
pub use direct_anchor_hca::{DirectAnchorHca, DirectAnchorStats};

pub mod concurrent_direct_anchor_hca;
pub use concurrent_direct_anchor_hca::ConcurrentDirectAnchorHca;

pub mod radix_hca;
pub use radix_hca::{RadixHca, RadixAnchorStats};

// Re-export commonly used types (optional)
pub use std::collections::HashMap;

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
