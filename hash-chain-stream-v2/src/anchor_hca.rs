use crate::hashchain_stream::HashChainStream;
use crate::common::AnchorConfig;
use crate::HcaError;

// One shared depth limit for both implementations.
pub(crate) const MAX_DEPTH_DEFAULT: usize = 16;

/* ---------------------- Microtable (shared) ---------------------- */

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

#[derive(Debug, Clone)]
pub(crate) struct OptimizedMicrotable<K, V> {
    pub(crate) slots: Vec<OptimizedSlot<K, V>>,
    pub(crate) occupied_count: usize,
    pub(crate) capacity: usize,
}

impl<K, V> OptimizedMicrotable<K, V>
where
    K: Default + Clone + PartialEq,
    V: Default + Clone,
{
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            slots: vec![OptimizedSlot::default(); capacity],
            occupied_count: 0,
            capacity,
        }
    }

    #[inline(always)]
    pub(crate) fn is_full(&self) -> bool {
        self.occupied_count >= self.capacity
    }

    #[allow(dead_code)]
    pub(crate) fn reset(&mut self) {
        for slot in &mut self.slots {
            slot.state = 0;
        }
        self.occupied_count = 0;
    }

    #[inline(always)]
    pub(crate) fn find_slot(&self, fp: u16, key: &K) -> Option<usize> {
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.state != 2 { continue; }
            if slot.fingerprint == fp && slot.key == *key {
                return Some(i);
            }
        }
        None
    }

    #[inline(always)]
    pub(crate) fn find_empty_slot(&self) -> Option<usize> {
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.state == 0 { return Some(i); }
        }
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.state == 1 { return Some(i); }
        }
        None
    }

    #[inline(always)]
    pub(crate) fn insert_at(&mut self, idx: usize, fp: u16, key: K, value: V) -> bool {
        if idx < self.slots.len() && (self.slots[idx].state == 0 || self.slots[idx].state == 1) {
            let was_not_occupied = self.slots[idx].state != 2;
            self.slots[idx] = OptimizedSlot { fingerprint: fp, state: 2, _padding: 0, key, value };
            if was_not_occupied {
                self.occupied_count += 1;
            }
            true
        } else {
            false
        }
    }

    #[inline(always)]
    pub(crate) fn remove_at(&mut self, idx: usize) -> Option<V> {
        if idx < self.slots.len() && self.slots[idx].state == 2 {
            self.slots[idx].state = 1; // tombstone
            self.occupied_count -= 1;
            Some(self.slots[idx].value.clone())
        } else {
            None
        }
    }
}

/* --------------- AnchorHca using paged directory ---------------- */

const PAGE_BITS: usize = 14; // 16k microtables per page; tweak as you like
const PAGE_SIZE: usize = 1 << PAGE_BITS;
const PAGE_MASK: u64 = (PAGE_SIZE as u64) - 1;

#[derive(Debug, Clone)]
struct Page<K, V> {
    cells: Vec<Option<OptimizedMicrotable<K, V>>>, // len = PAGE_SIZE
}
impl<K, V> Page<K, V> {
    #[inline]
    fn new() -> Self {
        // Avoid `vec![None; PAGE_SIZE]` which requires `Option<T>: Clone`
        Self { cells: (0..PAGE_SIZE).map(|_| None).collect() }
    }
}

#[derive(Debug, Clone)]
struct PageEntry<K, V> {
    hi: u64,
    page: Page<K, V>,
}

#[derive(Debug, Clone)]
struct Layer<K, V> {
    pages: Vec<PageEntry<K, V>>, // sorted by hi
}
impl<K, V> Layer<K, V>
where
    K: Default + Clone + PartialEq,
    V: Default + Clone,
{
    #[inline]
    fn new() -> Self { Self { pages: Vec::new() } }

    #[inline]
    fn get_page_index(&self, hi: u64) -> Result<usize, usize> {
        self.pages.binary_search_by(|e| e.hi.cmp(&hi))
    }

    #[inline]
    fn get_or_create_page(&mut self, hi: u64) -> &mut Page<K, V> {
        match self.get_page_index(hi) {
            Ok(i) => &mut self.pages[i].page,
            Err(pos) => {
                self.pages.insert(pos, PageEntry { hi, page: Page::new() });
                &mut self.pages[pos].page
            }
        }
    }

    #[inline]
    fn get(&self, hi: u64, lo: usize) -> Option<&OptimizedMicrotable<K, V>> {
        let i = self.get_page_index(hi).ok()?;
        self.pages[i].page.cells.get(lo)?.as_ref()
    }

    #[inline]
    fn get_mut(&mut self, hi: u64, lo: usize) -> Option<&mut OptimizedMicrotable<K, V>> {
        let i = self.get_page_index(hi).ok()?;
        self.pages[i].page.cells.get_mut(lo)?.as_mut()
    }

    #[inline]
    fn get_or_create(&mut self, hi: u64, lo: usize, cap: usize) -> &mut OptimizedMicrotable<K, V> {
        let page = self.get_or_create_page(hi);
        if page.cells[lo].is_none() {
            page.cells[lo] = Some(OptimizedMicrotable::new(cap));
        }
        page.cells[lo].as_mut().unwrap()
    }
}

pub struct AnchorHca<K, V>
where
    K: Default + Clone + PartialEq + AsRef<[u8]>,
    V: Default + Clone,
{
    config: AnchorConfig,
    layers: Vec<Layer<K, V>>,
    max_depth: usize,
}

impl<K, V> AnchorHca<K, V>
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
            layers: Vec::new(),
            max_depth: MAX_DEPTH_DEFAULT,
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(AnchorConfig::default())
    }

    #[inline(always)]
    fn compute_fingerprint_from_stream(stream: &mut HashChainStream) -> u16 {
        let a = stream.nibble(0) as u16;
        let b = stream.nibble(1) as u16;
        let c = stream.nibble(2) as u16;
        (a << 8) | (b << 4) | c
    }

    #[inline]
    fn layer_mut(&mut self, depth: usize) -> &mut Layer<K, V> {
        if depth >= self.layers.len() {
            self.layers.resize_with(depth + 1, || Layer::new());
        }
        &mut self.layers[depth]
    }

    #[inline]
    fn layer(&self, depth: usize) -> Option<&Layer<K, V>> {
        self.layers.get(depth)
    }

    #[inline]
    fn get_or_create_microtable(&mut self, depth: usize, prefix: u64) -> &mut OptimizedMicrotable<K, V> {
        let cap = self.config.s_max; // read before borrow
        let hi = prefix >> PAGE_BITS;
        let lo = (prefix & PAGE_MASK) as usize;
        self.layer_mut(depth).get_or_create(hi, lo, cap)
    }

    #[inline]
    fn get_microtable(&self, depth: usize, prefix: u64) -> Option<&OptimizedMicrotable<K, V>> {
        let hi = prefix >> PAGE_BITS;
        let lo = (prefix & PAGE_MASK) as usize;
        self.layer(depth)?.get(hi, lo)
    }

    #[inline]
    fn get_microtable_mut(&mut self, depth: usize, prefix: u64) -> Option<&mut OptimizedMicrotable<K, V>> {
        let hi = prefix >> PAGE_BITS;
        let lo = (prefix & PAGE_MASK) as usize;
        self.layers.get_mut(depth)?.get_mut(hi, lo)
    }

    #[inline]
    fn find_insertion_location(&self, stream: &mut HashChainStream) -> (usize, u64) {
        for depth in self.config.default_depth..self.max_depth {
            let prefix = stream.prefix(depth);
            match self.get_microtable(depth, prefix) {
                Some(mt) if !mt.is_full() => return (depth, prefix),
                None => return (depth, prefix),
                _ => {}
            }
        }
        (self.max_depth - 1, stream.prefix(self.max_depth - 1))
    }

    #[inline]
    fn find_lookup_location(&self, stream: &mut HashChainStream) -> Option<(usize, u64)> {
        for depth in self.config.default_depth..self.max_depth {
            let prefix = stream.prefix(depth);
            if self.get_microtable(depth, prefix).is_some() {
                return Some((depth, prefix));
            }
        }
        None
    }

    /* -------------------------- API -------------------------- */

    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>, HcaError> {
        let mut s = HashChainStream::with_hash_mode(key.as_ref(), &self.config.domain, self.config.hash_mode);
        let fp = Self::compute_fingerprint_from_stream(&mut s);

        let (depth, prefix) = self.find_insertion_location(&mut s);
        let mt = self.get_or_create_microtable(depth, prefix);

        // update existing
        if let Some(i) = mt.find_slot(fp, &key) {
            let old = mt.slots[i].value.clone();
            mt.slots[i].value = value;
            return Ok(Some(old));
        }

        // normal insert into chosen microtable
        if let Some(e) = mt.find_empty_slot() {
            if mt.insert_at(e, fp, key, value) {
                return Ok(None);
            }
            return Err(HcaError::CorruptedData("Insert failed".to_string()));
        }

        // If it's full, try deeper (cold path).
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
        let fp = Self::compute_fingerprint_from_stream(&mut s);
        let (d, p) = self.find_lookup_location(&mut s)?;
        let mt = self.get_microtable(d, p)?;
        let i = mt.find_slot(fp, key)?;
        Some(&mt.slots[i].value)
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let mut s = HashChainStream::with_hash_mode(key.as_ref(), &self.config.domain, self.config.hash_mode);
        let fp = Self::compute_fingerprint_from_stream(&mut s);
        let (d, p) = self.find_lookup_location(&mut s)?;
        let mt = self.get_microtable_mut(d, p)?;
        let i = mt.find_slot(fp, key)?;
        mt.remove_at(i)
    }

    pub fn stats(&self) -> AnchorHcaStats {
        let mut total_anchors = 0usize;
        let mut total_entries = 0usize;
        let mut max_depth = 0usize;
        let mut min_depth = usize::MAX;

        for (depth, layer) in self.layers.iter().enumerate() {
            for entry in &layer.pages {
                for cell in &entry.page.cells {
                    if let Some(mt) = cell {
                        total_anchors += 1;
                        total_entries += mt.occupied_count;
                        max_depth = max_depth.max(depth);
                        min_depth = min_depth.min(depth);
                    }
                }
            }
        }
        if total_anchors == 0 {
            min_depth = 0;
        }

        AnchorHcaStats {
            total_anchors,
            total_entries,
            max_depth,
            min_depth,
            avg_entries_per_anchor: if total_anchors > 0 {
                total_entries as f64 / total_anchors as f64
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnchorHcaStats {
    pub total_anchors: usize,
    pub total_entries: usize,
    pub max_depth: usize,
    pub min_depth: usize,
    pub avg_entries_per_anchor: f64,
}