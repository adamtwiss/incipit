// Transposition table: 12-byte entries in 5-entry buckets (one cache line),
// a 32-bit key check, depth- and age-aware replacement. Lockless: safe to
// probe and store from several threads at once (see `Bucket`).
use crate::params::{on, tp, P};
use crate::position::Move;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering::Relaxed};

pub const BOUND_NONE: u8 = 0;
pub const BOUND_UPPER: u8 = 1;
pub const BOUND_LOWER: u8 = 2;
pub const BOUND_EXACT: u8 = 3;

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Entry {
    pub key: u64,
    pub mv: Move,
    pub score: i16,
    pub eval: i16,
    pub depth: u8,
    pub bound: u8,
    /// Stored from a PV node, or from a node whose entry already said so.
    pub pv: bool,
}

impl Entry {
    #[inline(always)]
    fn pack(&self) -> u64 {
        self.mv as u64
            | (self.score as u16 as u64) << 16
            | (self.eval as u16 as u64) << 32
            | (self.depth as u64 & 127) << 48
            | (self.pv as u64) << 55
            | (self.bound as u64) << 56
    }
    #[inline(always)]
    fn unpack(key: u64, d: u64) -> Entry {
        Entry {
            key,
            mv: d as u16 as Move,
            score: (d >> 16) as u16 as i16,
            eval: (d >> 32) as u16 as i16,
            depth: (d >> 48) as u8 & 127,
            bound: (d >> 56) as u8,
            pv: d >> 55 & 1 != 0,
        }
    }
}

/// Entries per bucket: five fill one 64-byte cache line.
const WAYS: usize = 5;

/// A bucket: each entry as its packed data word and a 32-bit check, the
/// key's low 32 bits XOR-ed with a hash of the data (`fold`). The bucket
/// index comes from the key's high bits, so index and check together test
/// about 32 + log2(buckets) bits of the key: another position passes with
/// odds of about 2^-32 per entry (its move is validated before it is
/// played). Threads read and write the words without locks, so a reader can
/// see the data of one write and the check of another (a torn entry); the
/// check then fails, except with odds of about 2^-32, and a torn entry reads
/// as a miss. No word orders any other memory, so Relaxed is enough
/// throughout: plain loads and stores on x86 and ARM.
#[derive(Default)]
#[repr(C, align(64))]
struct Bucket {
    data: [AtomicU64; WAYS],
    check: [AtomicU32; WAYS],
}

const _: () = assert!(std::mem::size_of::<Bucket>() == 64);

/// The data's share of the check: the high half of a multiplicative hash,
/// so any change to the data changes it (except with odds of about 2^-32),
/// where a plain fold of the two halves would miss some structured changes.
#[inline(always)]
fn fold(d: u64) -> u32 {
    (d.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as u32
}

impl Bucket {
    /// Entry `i`: the low 32 bits of the key it was written for, if it is
    /// intact, and its data.
    #[inline(always)]
    fn read(&self, i: usize) -> (u32, u64) {
        let d = self.data[i].load(Relaxed);
        (self.check[i].load(Relaxed) ^ fold(d), d)
    }
    #[inline(always)]
    fn write(&self, i: usize, e: &Entry) {
        let d = e.pack();
        self.check[i].store(e.key as u32 ^ fold(d), Relaxed);
        self.data[i].store(d, Relaxed);
    }
}

/// Buckets of 5 entries. An entry's `bound` byte also holds the generation
/// (search number) it was written in: bits 0-1 the bound, bits 2-7 the
/// generation; the depth byte holds the depth (7 bits, depth < MAX_PLY)
/// and the was-PV flag (bit 7). Replacement prefers, in a bucket: the same key; else an empty
/// entry; else the entry with the lowest depth - 8 * age (old and shallow
/// first), so quiescence stores no longer evict deep or PV entries.
pub struct TT {
    table: Table,
    gen: AtomicU8,
}

/// The buckets: zeroed memory straight from the allocator, 64-byte aligned
/// by hand. A zeroed request at the type's own alignment (64) is served
/// by clearing the memory, which touches every page; one at a plain
/// alignment is a calloc, whose large allocations come from the OS as
/// untouched zero pages, so a big table costs only the memory the search
/// goes on to use. An all-zero bucket is five empty entries.
struct Table {
    base: *mut u8,
    layout: std::alloc::Layout,
    buckets: *mut Bucket,
    len: usize,
}

// The allocation is owned exclusively, like a Box<[Bucket]>.
unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Table {
    fn new(len: usize) -> Option<Table> {
        let align = std::mem::align_of::<Bucket>();
        let layout =
            std::alloc::Layout::from_size_align(len.checked_mul(std::mem::size_of::<Bucket>())? + align, 16).ok()?;
        // SAFETY: the layout has a non-zero size.
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        if base.is_null() {
            return None;
        }
        // SAFETY: the extra `align` bytes leave room for the offset.
        let buckets = unsafe { base.add(base.align_offset(align)) } as *mut Bucket;
        Some(Table { base, layout, buckets, len })
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.base, self.layout) }
    }
}

impl std::ops::Deref for Table {
    type Target = [Bucket];
    fn deref(&self) -> &[Bucket] {
        // SAFETY: `buckets` is aligned, zero-initialised (valid) and `len` long.
        unsafe { std::slice::from_raw_parts(self.buckets, self.len) }
    }
}

impl std::ops::DerefMut for Table {
    fn deref_mut(&mut self) -> &mut [Bucket] {
        unsafe { std::slice::from_raw_parts_mut(self.buckets, self.len) }
    }
}

impl TT {
    pub fn new(mb: usize) -> TT {
        Self::try_new(mb).unwrap_or_else(|| {
            eprintln!("error: cannot allocate a {} MB hash table", mb);
            std::process::exit(1)
        })
    }
    /// As `new`, but None instead of aborting when the memory isn't there
    /// (see `Table` for why a big table is cheap until used).
    pub fn try_new(mb: usize) -> Option<TT> {
        let n = ((mb.max(1) * 1024 * 1024) / std::mem::size_of::<Bucket>()).max(1);
        Some(TT { table: Table::new(n)?, gen: AtomicU8::new(0) })
    }
    /// A table of exactly `n` buckets (tests).
    #[cfg(test)]
    pub fn with_buckets(n: usize) -> TT {
        TT { table: Table::new(n.max(1)).expect("table allocation"), gen: AtomicU8::new(0) }
    }
    /// Empties the table. `&mut self` means no other thread is using it,
    /// so it can be zeroed as plain memory (an all-zero entry is empty).
    pub fn clear(&mut self) {
        // SAFETY: all-zero buckets are valid (and empty).
        unsafe { std::ptr::write_bytes(self.table.as_mut_ptr(), 0, self.table.len()) };
        *self.gen.get_mut() = 0;
    }
    /// Starts a new search: entries written before it age by one.
    pub fn new_search(&self) {
        self.gen.store((self.gen.load(Relaxed) + 1) & 63, Relaxed);
    }
    #[inline(always)]
    fn idx(&self, key: u64) -> usize {
        ((key as u128 * self.table.len() as u128) >> 64) as usize
    }
    #[inline(always)]
    pub fn prefetch(&self, key: u64) {
        let p = unsafe { self.table.as_ptr().add(self.idx(key)) } as *const i8;
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(p, std::arch::x86_64::_MM_HINT_T0)
        };
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags, readonly))
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let _ = p;
    }
    #[inline(always)]
    pub fn probe(&self, key: u64) -> Option<Entry> {
        let b = unsafe { self.table.get_unchecked(self.idx(key)) };
        for i in 0..WAYS {
            let (k, d) = b.read(i);
            if k == key as u32 && (d >> 56) as u8 & 3 != BOUND_NONE {
                let e = Entry::unpack(key, d);
                return Some(Entry { bound: e.bound & 3, ..e });
            }
        }
        None
    }
    #[inline(always)]
    pub fn store(&self, key: u64, mv: Move, score: i32, eval: i32, depth: i32, bound: u8, pv: bool) {
        let gen = self.gen.load(Relaxed);
        let b = unsafe { self.table.get_unchecked(self.idx(key)) };
        // Only the bound byte (bits 56-63: bound and generation) and depth
        // (bits 48-55) of a slot matter for replacement, so the packed
        // words are read directly rather than unpacked.
        let age = |bd: u8| (gen.wrapping_sub(bd >> 2) & 63) as i32;
        let mut slot = 0;
        let mut old = 0u64;
        let mut found = false;
        let mut worst = i32::MAX;
        for k in 0..WAYS {
            let (ek, d) = b.read(k);
            let bd = (d >> 56) as u8;
            if ek == key as u32 && bd & 3 != BOUND_NONE {
                slot = k;
                old = d;
                found = true;
                break;
            }
            // A was-PV entry (bit 55) counts TtPvKeep plies deeper than it
            // is, so it survives replacement longer.
            let q = if bd & 3 == BOUND_NONE {
                i32::MIN
            } else {
                ((d >> 48) as u8 & 127) as i32 - 8 * age(bd)
                    + if on(P::UseTtPvKeep) && (d >> 55) & 1 != 0 { tp(P::TtPvKeep) } else { 0 }
            };
            if q < worst {
                worst = q;
                slot = k;
            }
        }
        // Same position: keep the deeper result unless this one is exact,
        // nearly as deep, or the old one is from an earlier search.
        if found
            && !(bound == BOUND_EXACT || depth + 3 >= ((old >> 48) as u8 & 127) as i32 || age((old >> 56) as u8) != 0)
        {
            return;
        }
        let mv = if mv == 0 && found { old as u16 as Move } else { mv };
        b.write(
            slot,
            &Entry {
                key,
                mv,
                score: score as i16,
                eval: eval as i16,
                depth: depth.clamp(0, 127) as u8,
                bound: bound | (gen << 2),
                pv,
            },
        );
    }
    /// Permille of entries in use from the current search (first 1000).
    pub fn hashfull(&self) -> usize {
        let gen = self.gen.load(Relaxed);
        self.table
            .iter()
            .take(1000 / WAYS)
            .flat_map(|b| b.data.iter())
            .map(|d| (d.load(Relaxed) >> 56) as u8)
            .filter(|&bd| bd & 3 != BOUND_NONE && bd >> 2 == gen)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// splitmix64: the test's random numbers.
    fn mix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// What writer `t` stores for `key` on its `n`th write: the move and
    /// eval are a function of (key, score), so a probe can check that every
    /// field it sees came from one write of that key.
    fn fields(key: u64, t: u64, n: u64) -> (Move, i32, i32, i32, u8) {
        let score = (mix(t << 32 | n) % 20000) as i32 - 10000;
        let h = mix(key ^ score as u64);
        let mv = (h as u16 | 1) as Move;
        let eval = (h >> 16) as u16 as i16 as i32;
        (mv, score, eval, (h >> 32) as i32 & 63, 1 + ((h >> 40) % 3) as u8)
    }

    #[test]
    fn entry_roundtrip() {
        let e = Entry {
            key: 0x0123_4567_89AB_CDEF,
            mv: 0xBEEF,
            score: -31000,
            eval: 32000,
            depth: 100,
            bound: 0xFF,
            pv: true,
        };
        assert_eq!(Entry::unpack(e.key, e.pack()), e);
        let e = Entry { depth: 127, pv: false, ..e };
        assert_eq!(Entry::unpack(e.key, e.pack()), e);
    }

    /// Eight threads hammer a 4-bucket table with 64 keys: every probe hit
    /// must be a whole entry some thread wrote for that key, never a mix of
    /// two writes.
    #[test]
    fn concurrent_no_torn_entries() {
        let tt = Arc::new(TT::with_buckets(4));
        let keys: Arc<Vec<u64>> = Arc::new((0..64).map(|i| mix(i + 1000)).collect());
        let threads: Vec<_> = (0..8u64)
            .map(|t| {
                let (tt, keys) = (tt.clone(), keys.clone());
                std::thread::spawn(move || {
                    let (mut hits, mut r) = (0u64, mix(t));
                    for n in 0..400_000u64 {
                        r = mix(r);
                        let key = keys[(r % keys.len() as u64) as usize];
                        if r >> 32 & 1 == 0 {
                            let (mv, score, eval, depth, bound) = fields(key, t, n);
                            tt.store(key, mv, score, eval, depth, bound, n & 1 == 0);
                        } else if let Some(e) = tt.probe(key) {
                            hits += 1;
                            let h = mix(key ^ e.score as i32 as u64);
                            assert_eq!(e.key, key);
                            assert_eq!(e.mv, (h as u16 | 1) as Move, "torn entry");
                            assert_eq!(e.eval, (h >> 16) as u16 as i16, "torn entry");
                            assert_eq!(e.depth as u64, (h >> 32) & 63, "torn entry");
                            assert_eq!(e.bound as u64, 1 + (h >> 40) % 3, "torn entry");
                        }
                    }
                    hits
                })
            })
            .collect();
        let hits: u64 = threads.into_iter().map(|h| h.join().unwrap()).sum();
        assert!(hits > 100_000, "too few hits to test anything: {hits}");
    }
}
