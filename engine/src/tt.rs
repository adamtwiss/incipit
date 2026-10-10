// Transposition table: 16-byte entries in 4-entry buckets (one cache line),
// full key verification, depth- and age-aware replacement. Lockless: safe
// to probe and store from several threads at once (see `Slot`).
use crate::position::Move;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};

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
}

impl Entry {
    #[inline(always)]
    fn pack(&self) -> u64 {
        self.mv as u64
            | (self.score as u16 as u64) << 16
            | (self.eval as u16 as u64) << 32
            | (self.depth as u64) << 48
            | (self.bound as u64) << 56
    }
    #[inline(always)]
    fn unpack(key: u64, d: u64) -> Entry {
        Entry {
            key,
            mv: d as u16 as Move,
            score: (d >> 16) as u16 as i16,
            eval: (d >> 32) as u16 as i16,
            depth: (d >> 48) as u8,
            bound: (d >> 56) as u8,
        }
    }
}

/// One entry as two 64-bit words: the packed data, and the key XOR-ed with
/// the data. Threads read and write the words without locks, so a reader
/// can see one word of one write and the other word of another (a torn
/// entry); `key ^ data` then fails the key check (except with odds of about
/// 2^-64, like any key collision), so a torn entry reads as a miss. No word
/// orders any other memory, so Relaxed is enough throughout: plain loads and
/// stores on x86 and ARM.
#[derive(Default)]
struct Slot {
    check: AtomicU64,
    data: AtomicU64,
}

impl Slot {
    /// The entry, if this slot holds `key` intact.
    #[inline(always)]
    fn read(&self) -> (u64, u64) {
        let d = self.data.load(Relaxed);
        (self.check.load(Relaxed) ^ d, d)
    }
    #[inline(always)]
    fn write(&self, e: &Entry) {
        let d = e.pack();
        self.check.store(e.key ^ d, Relaxed);
        self.data.store(d, Relaxed);
    }
}

/// Entries per bucket: one 64-byte cache line.
const WAYS: usize = 4;

#[derive(Default)]
#[repr(C, align(64))]
struct Bucket([Slot; WAYS]);

/// Buckets of 4 entries. An entry's `bound` byte also holds the generation
/// (search number) it was written in: bits 0-1 the bound, bits 2-7 the
/// generation. Replacement prefers, in a bucket: the same key; else an empty
/// entry; else the entry with the lowest depth - 8 * age (old and shallow
/// first), so quiescence stores no longer evict deep or PV entries.
pub struct TT {
    table: Box<[Bucket]>,
    gen: AtomicU8,
}

impl TT {
    pub fn new(mb: usize) -> TT {
        Self::with_buckets((mb.max(1) * 1024 * 1024) / std::mem::size_of::<Bucket>())
    }
    /// As `new`, but None instead of aborting when the memory isn't there.
    pub fn try_new(mb: usize) -> Option<TT> {
        let n = ((mb.max(1) * 1024 * 1024) / std::mem::size_of::<Bucket>()).max(1);
        let mut table = Vec::new();
        table.try_reserve_exact(n).ok()?;
        table.resize_with(n, Bucket::default);
        Some(TT { table: table.into_boxed_slice(), gen: AtomicU8::new(0) })
    }
    pub fn with_buckets(n: usize) -> TT {
        TT { table: (0..n.max(1)).map(|_| Bucket::default()).collect(), gen: AtomicU8::new(0) }
    }
    /// Empties the table. `&mut self` means no other thread is using it,
    /// so it can be zeroed as plain memory (an all-zero `Slot` is two zero
    /// atomics: an empty entry).
    pub fn clear(&mut self) {
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
        for s in b.0.iter() {
            let (k, d) = s.read();
            if k == key && (d >> 56) as u8 & 3 != BOUND_NONE {
                let e = Entry::unpack(k, d);
                return Some(Entry { bound: e.bound & 3, ..e });
            }
        }
        None
    }
    #[inline(always)]
    pub fn store(&self, key: u64, mv: Move, score: i32, eval: i32, depth: i32, bound: u8) {
        let gen = self.gen.load(Relaxed);
        let b = unsafe { self.table.get_unchecked(self.idx(key)) };
        let age = |e: &Entry| (gen.wrapping_sub(e.bound >> 2) & 63) as i32;
        let mut slot = 0;
        let mut old = Entry::default();
        let mut found = false;
        let mut worst = i32::MAX;
        for (k, s) in b.0.iter().enumerate() {
            let (ek, d) = s.read();
            let e = Entry::unpack(ek, d);
            if ek == key && e.bound & 3 != BOUND_NONE {
                slot = k;
                old = e;
                found = true;
                break;
            }
            let q = if e.bound & 3 == BOUND_NONE { i32::MIN } else { e.depth as i32 - 8 * age(&e) };
            if q < worst {
                worst = q;
                slot = k;
            }
        }
        // Same position: keep the deeper result unless this one is exact,
        // nearly as deep, or the old one is from an earlier search.
        if found && !(bound == BOUND_EXACT || depth + 3 >= old.depth as i32 || age(&old) != 0) {
            return;
        }
        let mv = if mv == 0 && found { old.mv } else { mv };
        b.0[slot].write(&Entry {
            key,
            mv,
            score: score as i16,
            eval: eval as i16,
            depth: depth.clamp(0, 255) as u8,
            bound: bound | (gen << 2),
        });
    }
    /// Permille of entries in use from the current search (first 1000).
    pub fn hashfull(&self) -> usize {
        let gen = self.gen.load(Relaxed);
        self.table
            .iter()
            .take(1000 / WAYS)
            .flat_map(|b| b.0.iter())
            .map(|s| (s.read().1 >> 56) as u8)
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
        let e = Entry { key: 0x0123_4567_89AB_CDEF, mv: 0xBEEF, score: -31000, eval: 32000, depth: 200, bound: 0xFF };
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
                            tt.store(key, mv, score, eval, depth, bound);
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
