use crate::hashchain_stream::HashChainStream;
use crate::common::AnchorConfig;
use crate::anchor_hca::{OptimizedMicrotable, MAX_DEPTH_DEFAULT};
use crate::HcaError;

pub struct DirectAnchorHca<K, V>
where
    K: Default + Clone + PartialEq + AsRef<[u8]>,
    V: Default + Clone,
{
    config: AnchorConfig,
    depth_maps: Vec<Vec<Option<OptimizedMicrotable<K, V>>>>,
    max_prefix_bits: usize, // mask to keep directories bounded
    max_depth: usize,
}

impl<K, V> DirectAnchorHca<K, V>
where
    K: Clone + Default + PartialEq + AsRef<[u8]>,
    V: Clone + Default,
{
    pub fn new(mut config: AnchorConfig) -> Self {
        if config.default_depth == 0 {
            config.default_depth = 1;
        }
        Self {
            config,
            depth_maps: Vec::new(),
            max_prefix_bits: 20, // cap addressable buckets per depth
            max_depth: MAX_DEPTH_DEFAULT,
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(AnchorConfig::default())
    }

    #[inline(always)]
    fn compute_fingerprint(stream: &mut HashChainStream) -> u16 {
        let a = stream.nibble(0) as u16;
        let b = stream.nibble(1) as u16;
        let c = stream.nibble(2) as u16;
        (a << 8) | (b << 4) | c
    }

    #[inline(always)]
    fn idx(&self, prefix: u64) -> usize {
        (prefix as usize) & ((1usize << self.max_prefix_bits) - 1)
    }

    fn get_or_create_microtable(&mut self, depth: usize, prefix: u64) -> &mut OptimizedMicrotable<K, V> {
        let idx = self.idx(prefix);
        if depth >= self.depth_maps.len() {
            self.depth_maps.resize_with(depth + 1, Vec::new);
        }
        let dmap = &mut self.depth_maps[depth];

        if idx >= dmap.len() {
            dmap.resize_with(idx + 1, || None);
        }
        if dmap[idx].is_none() {
            let cap = self.config.s_max;
            dmap[idx] = Some(OptimizedMicrotable::new(cap));
        }
        dmap[idx].as_mut().unwrap()
    }

    fn get_microtable(&self, depth: usize, prefix: u64) -> Option<&OptimizedMicrotable<K, V>> {
        if depth >= self.depth_maps.len() {
            return None;
        }
        let idx = self.idx(prefix);
        let dmap = &self.depth_maps[depth];
        if idx >= dmap.len() {
            return None;
        }
        dmap[idx].as_ref()
    }

    fn get_microtable_mut(&mut self, depth: usize, prefix: u64) -> Option<&mut OptimizedMicrotable<K, V>> {
        if depth >= self.depth_maps.len() {
            return None;
        }
        let idx = self.idx(prefix);
        let dmap = &mut self.depth_maps[depth];
        if idx >= dmap.len() {
            return None;
        }
        dmap[idx].as_mut()
    }

    fn find_insertion_location(&self, stream: &mut HashChainStream) -> (usize, u64) {
        let mut prefix: u64 = 0;
        for d in 0..self.max_depth {
            prefix = (prefix << 4) | (stream.nibble(d) as u64);
            if d < self.config.default_depth { continue; }
            match self.get_microtable(d, prefix) {
                Some(mt) if !mt.is_full() => return (d, prefix),
                None => return (d, prefix),
                _ => {}
            }
        }
        (self.max_depth - 1, prefix)
    }

    fn find_lookup_location(&self, stream: &mut HashChainStream) -> Option<(usize, u64)> {
        let mut prefix: u64 = 0;
        for d in 0..self.max_depth {
            prefix = (prefix << 4) | (stream.nibble(d) as u64);
            if d < self.config.default_depth { continue; }
            if self.get_microtable(d, prefix).is_some() {
                return Some((d, prefix));
            }
        }
        None
    }

    /* -------------------------- API -------------------------- */

    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>, HcaError> {
        let mut s = HashChainStream::with_hash_mode(key.as_ref(), &self.config.domain, self.config.hash_mode);
        let fp = Self::compute_fingerprint(&mut s);

        let (depth, prefix) = self.find_insertion_location(&mut s);
        let mt = self.get_or_create_microtable(depth, prefix);

        // update existing
        if let Some(i) = mt.find_slot(fp, &key) {
            let old = mt.slots[i].value.clone();
            mt.slots[i].value = value;
            return Ok(Some(old));
        }

        // normal insert
        if let Some(e) = mt.find_empty_slot() {
            if mt.insert_at(e, fp, key, value) {
                return Ok(None);
            }
            return Err(HcaError::CorruptedData("Insert failed".to_string()));
        }

        // escalate deeper (cold path)
        for fd in (depth + 1)..self.max_depth {
            let fprefix = s.prefix(fd);
            let fmt = self.get_or_create_microtable(fd, fprefix);
            if !fmt.is_full() {
                if let Some(e) = fmt.find_empty_slot() {
                    if fmt.insert_at(e, fp, key.clone(), value.clone()) {
                        return Ok(None);
                    }
                }
            }
        }

        Err(HcaError::CorruptedData("All depth levels exhausted".to_string()))
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        let mut s = HashChainStream::with_hash_mode(key.as_ref(), &self.config.domain, self.config.hash_mode);
        let fp = Self::compute_fingerprint(&mut s);
        let (d, p) = self.find_lookup_location(&mut s)?;
        let mt = self.get_microtable(d, p)?;
        let i = mt.find_slot(fp, key)?;
        Some(&mt.slots[i].value)
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let mut s = HashChainStream::with_hash_mode(key.as_ref(), &self.config.domain, self.config.hash_mode);
        let fp = Self::compute_fingerprint(&mut s);
        let (d, p) = self.find_lookup_location(&mut s)?;
        let mt = self.get_microtable_mut(d, p)?;
        let i = mt.find_slot(fp, key)?;
        mt.remove_at(i)
    }

    pub fn stats(&self) -> DirectAnchorStats {
        let mut total_anchors = 0;
        let mut total_entries = 0;
        let mut max_depth = 0;
        let mut min_depth = usize::MAX;

        for (depth, dmap) in self.depth_maps.iter().enumerate() {
            for mt in dmap {
                if let Some(mt) = mt {
                    total_anchors += 1;
                    total_entries += mt.occupied_count;
                    max_depth = max_depth.max(depth);
                    min_depth = min_depth.min(depth);
                }
            }
        }
        if total_anchors == 0 { min_depth = 0; }

        DirectAnchorStats {
            total_anchors,
            total_entries,
            max_depth,
            min_depth,
            avg_entries_per_anchor: if total_anchors > 0 {
                total_entries as f64 / total_anchors as f64
            } else { 0.0 },
        }
    }
}

#[derive(Debug, Clone)]
pub struct DirectAnchorStats {
    pub total_anchors: usize,
    pub total_entries: usize,
    pub max_depth: usize,
    pub min_depth: usize,
    pub avg_entries_per_anchor: f64,
}
