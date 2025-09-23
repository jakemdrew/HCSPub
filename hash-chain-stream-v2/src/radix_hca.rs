// src/radix_hca.rs
#![allow(dead_code)]

use crate::anchor_hca::OptimizedMicrotable;
use crate::common::AnchorConfig;
use crate::hashchain_stream::{HashChainStream, HashMode};
use crate::HcaError;

const RADIX_MAX_DEPTH: usize = 16;
const ARENA_PREALLOC: usize = 32_768; // fewer reallocations on growth

type NodeId = usize;

#[derive(Debug)]
enum Node<K, V> {
    Branch {
        depth: usize,
        children: [Option<NodeId>; 16], // fixed fanout by nibble
    },
    Leaf {
        depth: usize,
        mt: OptimizedMicrotable<K, V>,
    },
}

#[derive(Debug, Clone)]
pub struct RadixStats {
    pub nodes: usize,
    pub branch_nodes: usize,
    pub leaf_nodes: usize,
}

pub type RadixAnchorStats = RadixStats;

pub struct RadixHca<K, V>
where
    K: AsRef<[u8]> + Default + Clone + PartialEq,
    V: Default + Clone,
{
    config: AnchorConfig,
    arena: Vec<Node<K, V>>,
    root: NodeId,
}

impl<K, V> RadixHca<K, V>
where
    K: AsRef<[u8]> + Default + Clone + PartialEq,
    V: Default + Clone,
{
    #[inline]
    pub fn new(config: AnchorConfig) -> Self {
        let mut arena = Vec::with_capacity(ARENA_PREALLOC);
        let root = arena.len();
        arena.push(Node::Leaf {
            depth: 0,
            mt: OptimizedMicrotable::new(config.s_max),
        });
        Self { config, arena, root }
    }

    #[inline]
    pub fn with_default_config() -> Self {
        Self::new(AnchorConfig::default())
    }

    /// Optimized defaults for benches: s_max = 16, HashMode::Fast
    #[inline]
    pub fn with_optimized_config() -> Self {
        let mut cfg = AnchorConfig::default();
        cfg.s_max = 16;
        cfg.hash_mode = HashMode::Fast;
        Self::new(cfg)
    }

    #[inline]
    pub fn stats(&self) -> RadixStats {
        let mut branch_nodes = 0usize;
        let mut leaf_nodes = 0usize;
        for n in &self.arena {
            match n {
                Node::Branch { .. } => branch_nodes += 1,
                Node::Leaf { .. } => leaf_nodes += 1,
            }
        }
        RadixStats {
            nodes: self.arena.len(),
            branch_nodes,
            leaf_nodes,
        }
    }

    // ---------- public API ----------

    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>, HcaError> {
        let mut s = HashChainStream::with_hash_mode(
            key.as_ref(),
            &self.config.domain,
            self.config.hash_mode,
        );
        let fp = Self::compute_fingerprint(&mut s);
        self.insert_inner(key, value, fp, &mut s)
    }

    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        let mut s = HashChainStream::with_hash_mode(
            key.as_ref(),
            &self.config.domain,
            self.config.hash_mode,
        );
        let fp = Self::compute_fingerprint(&mut s);
        self.get_inner(key, fp, &mut s)
    }

    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let mut s = HashChainStream::with_hash_mode(
            key.as_ref(),
            &self.config.domain,
            self.config.hash_mode,
        );
        let fp = Self::compute_fingerprint(&mut s);
        self.remove_inner(key, fp, &mut s)
    }

    // ---------- hot paths ----------

    fn insert_inner(
        &mut self,
        key: K,
        value: V,
        fp: u16,
        s: &mut HashChainStream,
    ) -> Result<Option<V>, HcaError> {
        let mut cur: NodeId = self.root;
        let mut depth = 0usize;

        loop {
            match &mut self.arena[cur] {
                Node::Leaf { depth: leaf_depth, mt } => {
                    debug_assert_eq!(*leaf_depth, depth);

                    // Overwrite w/o cloning the old value
                    if let Some(i) = mt.find_slot(fp, &key) {
                        let old = core::mem::replace(&mut mt.slots[i].value, value);
                        return Ok(Some(old));
                    }
                    // Fast path: insert into a free slot
                    if let Some(e) = mt.find_empty_slot() {
                        if mt.insert_at(e, fp, key, value) {
                            return Ok(None);
                        } else {
                            return Err(HcaError::CorruptedData("insert_at failed".into()));
                        }
                    }

                    // Need to split
                    if depth + 1 >= RADIX_MAX_DEPTH {
                        return Err(HcaError::CorruptedData(
                            "max depth reached during split".into(),
                        ));
                    }

                    // Replace leaf with branch; move out the old microtable
                    let full_mt = match core::mem::replace(
                        &mut self.arena[cur],
                        Node::Branch { depth, children: [None; 16] },
                    ) {
                        Node::Leaf { depth: d, mt } => {
                            debug_assert_eq!(d, depth);
                            mt
                        }
                        _ => unreachable!(),
                    };

                    // Redistribute entries under the new branch
                    for slot in full_mt.slots.into_iter() {
                        if slot.state == 0 {
                            continue;
                        }
                        let k = slot.key;   // move
                        let v = slot.value; // move

                        // Use fp to avoid rehash on shallow depths, else nibble()
                        let nib = Self::fp_nibble(slot.fingerprint, depth)
                            .unwrap_or_else(|| {
                                let mut s2 = HashChainStream::with_hash_mode(
                                    k.as_ref(),
                                    &self.config.domain,
                                    self.config.hash_mode,
                                );
                                s2.nibble(depth) as u8
                            });

                        let child = self.ensure_child_leaf(cur, depth, nib);
                        match &mut self.arena[child] {
                            Node::Leaf { mt, .. } => {
                                if let Some(e) = mt.find_empty_slot() {
                                    let _ = mt.insert_at(e, slot.fingerprint, k, v);
                                } else {
                                    let _ = self.insert_at_node(child, depth + 1, k, v, slot.fingerprint)?;
                                }
                            }
                            _ => unreachable!(),
                        }
                    }

                    // Insert the new (key,value)
                    let nib = Self::fp_nibble(fp, depth).unwrap_or_else(|| s.nibble(depth) as u8);
                    let child = self.ensure_child_leaf(cur, depth, nib);
                    return self.insert_at_node(child, depth + 1, key, value, fp);
                }

                Node::Branch { depth: branch_depth, .. } => {
                    debug_assert_eq!(*branch_depth, depth);
                    let nib = Self::fp_nibble(fp, depth).unwrap_or_else(|| s.nibble(depth) as u8);
                    cur = self.ensure_child_leaf(cur, depth, nib);
                    depth += 1;
                    if depth >= RADIX_MAX_DEPTH {
                        return Err(HcaError::CorruptedData(
                            "max depth reached in branch walk".into(),
                        ));
                    }
                }
            }
        }
    }

    fn get_inner<'a>(&'a self, key: &K, fp: u16, s: &mut HashChainStream) -> Option<&'a V> {
        let mut cur = self.root;
        let mut depth = 0usize;

        loop {
            match &self.arena[cur] {
                Node::Leaf { depth: leaf_depth, mt } => {
                    debug_assert_eq!(*leaf_depth, depth);
                    let i = mt.find_slot(fp, key)?;
                    return Some(&mt.slots[i].value);
                }
                Node::Branch { depth: branch_depth, children } => {
                    debug_assert_eq!(*branch_depth, depth);
                    let nib = Self::fp_nibble(fp, depth).unwrap_or_else(|| s.nibble(depth) as u8);
                    if let Some(next) = children[nib as usize] {
                        cur = next;
                        depth += 1;
                        if depth >= RADIX_MAX_DEPTH { return None; }
                    } else {
                        return None;
                    }
                }
            }
        }
    }

    fn remove_inner(&mut self, key: &K, fp: u16, s: &mut HashChainStream) -> Option<V> {
        let mut cur = self.root;
        let mut depth = 0usize;

        loop {
            match &mut self.arena[cur] {
                Node::Leaf { depth: leaf_depth, mt } => {
                    debug_assert_eq!(*leaf_depth, depth);
                    let i = mt.find_slot(fp, key)?;
                    return mt.remove_at(i);
                }
                Node::Branch { depth: branch_depth, children } => {
                    debug_assert_eq!(*branch_depth, depth);
                    let nib = Self::fp_nibble(fp, depth).unwrap_or_else(|| s.nibble(depth) as u8);
                    if let Some(next) = children[nib as usize] {
                        cur = next;
                        depth += 1;
                        if depth >= RADIX_MAX_DEPTH { return None; }
                    } else {
                        return None;
                    }
                }
            }
        }
    }

    // ---------- helpers ----------

    #[inline]
    fn push_leaf(&mut self, depth: usize) -> NodeId {
        let id = self.arena.len();
        self.arena.push(Node::Leaf {
            depth,
            mt: OptimizedMicrotable::new(self.config.s_max),
        });
        id
    }

    #[inline]
    fn ensure_child_leaf(&mut self, branch_idx: NodeId, depth: usize, nib: u8) -> NodeId {
        let need_new = match &self.arena[branch_idx] {
            Node::Branch { children, .. } => children[nib as usize].is_none(),
            _ => unreachable!("ensure_child_leaf on non-branch"),
        };

        if need_new {
            let new_id = self.push_leaf(depth + 1);
            if let Node::Branch { children, .. } = &mut self.arena[branch_idx] {
                children[nib as usize] = Some(new_id);
                new_id
            } else {
                unreachable!()
            }
        } else if let Node::Branch { children, .. } = &self.arena[branch_idx] {
            children[nib as usize].unwrap()
        } else {
            unreachable!()
        }
    }

    #[inline(always)]
    fn compute_fingerprint(s: &mut HashChainStream) -> u16 {
        // First 4 nibbles packed to u16; enough to guide top-level routing
        let mut fp: u16 = 0;
        fp = (fp << 4) | (s.nibble(0) as u16);
        fp = (fp << 4) | (s.nibble(1) as u16);
        fp = (fp << 4) | (s.nibble(2) as u16);
        fp = (fp << 4) | (s.nibble(3) as u16);
        fp
    }

    #[inline(always)]
    fn fp_nibble(fp: u16, depth: usize) -> Option<u8> {
        if depth <= 3 {
            let shift = (3 - depth) * 4;
            Some(((fp >> shift) & 0xF) as u8)
        } else {
            None
        }
    }

    #[inline]
    fn insert_at_node(
        &mut self,
        node: NodeId,
        depth: usize,
        key: K,
        value: V,
        fp: u16,
    ) -> Result<Option<V>, HcaError> {
        match &mut self.arena[node] {
            Node::Leaf { depth: leaf_depth, mt } => {
                debug_assert_eq!(*leaf_depth, depth);

                if let Some(i) = mt.find_slot(fp, &key) {
                    let old = core::mem::replace(&mut mt.slots[i].value, value);
                    return Ok(Some(old));
                }
                if let Some(e) = mt.find_empty_slot() {
                    if mt.insert_at(e, fp, key, value) {
                        return Ok(None);
                    } else {
                        return Err(HcaError::CorruptedData("insert_at failed".into()));
                    }
                }

                if depth + 1 >= RADIX_MAX_DEPTH {
                    return Err(HcaError::CorruptedData(
                        "max depth reached during child split".into(),
                    ));
                }

                // Turn leaf into branch; take old mt out
                let full_mt = match core::mem::replace(
                    &mut self.arena[node],
                    Node::Branch { depth, children: [None; 16] },
                ) {
                    Node::Leaf { depth: d, mt } => {
                        debug_assert_eq!(d, depth);
                        mt
                    }
                    _ => unreachable!(),
                };

                for slot in full_mt.slots.into_iter() {
                    if slot.state == 0 { continue; }
                    let k = slot.key;
                    let v = slot.value;

                    let nib = Self::fp_nibble(slot.fingerprint, depth)
                        .unwrap_or_else(|| {
                            let mut s2 = HashChainStream::with_hash_mode(
                                k.as_ref(),
                                &self.config.domain,
                                self.config.hash_mode,
                            );
                            s2.nibble(depth) as u8
                        });

                    let child = self.ensure_child_leaf(node, depth, nib);
                    match &mut self.arena[child] {
                        Node::Leaf { mt, .. } => {
                            if let Some(e) = mt.find_empty_slot() {
                                let _ = mt.insert_at(e, slot.fingerprint, k, v);
                            } else {
                                let _ = self.insert_at_node(child, depth + 1, k, v, slot.fingerprint)?;
                            }
                        }
                        _ => unreachable!(),
                    }
                }

                let nib = Self::fp_nibble(fp, depth).unwrap_or_else(|| {
                    let mut s3 = HashChainStream::with_hash_mode(
                        key.as_ref(), &self.config.domain, self.config.hash_mode,
                    );
                    s3.nibble(depth) as u8
                });
                let child = self.ensure_child_leaf(node, depth, nib);
                self.insert_at_node(child, depth + 1, key, value, fp)
            }

            Node::Branch { .. } => {
                let nib = Self::fp_nibble(fp, depth).unwrap_or_else(|| {
                    let mut s2 = HashChainStream::with_hash_mode(
                        key.as_ref(), &self.config.domain, self.config.hash_mode,
                    );
                    s2.nibble(depth) as u8
                });
                let child = self.ensure_child_leaf(node, depth, nib);
                self.insert_at_node(child, depth + 1, key, value, fp)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insert_get_remove() {
        let mut h = RadixHca::<String, i32>::with_default_config();
        assert_eq!(h.insert("a".into(), 1).unwrap(), None);
        assert_eq!(h.insert("b".into(), 2).unwrap(), None);
        assert_eq!(h.get(&"a".into()), Some(&1));
        assert_eq!(h.get(&"b".into()), Some(&2));
        assert_eq!(h.remove(&"a".into()), Some(1));
        assert_eq!(h.get(&"a".into()), None);
    }

    #[test]
    fn remove_nonexistent_is_none() {
        let mut h = RadixHca::<String, i32>::with_default_config();
        assert_eq!(h.remove(&"zzz".into()), None);
        assert_eq!(h.get(&"zzz".into()), None);
    }

    #[test]
    fn update_existing() {
        let mut h = RadixHca::<String, i32>::with_default_config();
        assert_eq!(h.insert("x".into(), 10).unwrap(), None);
        assert_eq!(h.insert("x".into(), 11).unwrap(), Some(10));
        assert_eq!(h.get(&"x".into()), Some(&11));
    }

    #[test]
    fn many_inserts_then_lookups() {
        let mut h = RadixHca::<String, usize>::with_optimized_config();
        for i in 0..10_000 {
            let k = format!("k{:06}", i);
            assert_eq!(h.insert(k.clone(), i).unwrap(), None);
        }
        for i in (0..10_000).step_by(97) {
            let k = format!("k{:06}", i);
            assert_eq!(h.get(&k), Some(&i));
        }
    }

    #[test]
    fn forces_leaf_split_then_lookup() {
        let mut cfg = AnchorConfig::default();
        cfg.s_max = 4;
        let mut h = RadixHca::<String, i32>::new(cfg);

        for i in 0..20 {
            let k = format!("key{:02}", i);
            assert_eq!(h.insert(k.clone(), i).unwrap(), None);
        }
        for i in 0..20 {
            let k = format!("key{:02}", i);
            assert_eq!(h.get(&k), Some(&i));
        }
    }

    #[test]
    fn overwrite_after_split() {
        let mut cfg = AnchorConfig::default();
        cfg.s_max = 4;
        let mut h = RadixHca::<String, i32>::new(cfg);

        for i in 0..64 {
            let k = format!("k{:03}", i);
            assert_eq!(h.insert(k.clone(), i).unwrap(), None);
        }
        for i in 0..64 {
            let k = format!("k{:03}", i);
            assert_eq!(h.insert(k.clone(), i + 1000).unwrap(), Some(i));
        }
        for i in 0..64 {
            let k = format!("k{:03}", i);
            assert_eq!(h.get(&k), Some(&(i + 1000)));
        }
    }
}