// Transposition table: 16-byte entries, full key verification.
use crate::position::Move;

pub const BOUND_NONE: u8 = 0;
pub const BOUND_UPPER: u8 = 1;
pub const BOUND_LOWER: u8 = 2;
pub const BOUND_EXACT: u8 = 3;

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Entry {
    pub key: u64,
    pub mv: Move,
    pub score: i16,
    pub eval: i16,
    pub depth: u8,
    pub bound: u8,
}

pub struct TT {
    table: Vec<Entry>,
}

impl TT {
    pub fn new(mb: usize) -> TT {
        let n = (mb.max(1) * 1024 * 1024) / std::mem::size_of::<Entry>();
        TT { table: vec![Entry::default(); n] }
    }
    pub fn clear(&mut self) {
        for e in self.table.iter_mut() {
            *e = Entry::default();
        }
    }
    #[inline(always)]
    fn idx(&self, key: u64) -> usize {
        ((key as u128 * self.table.len() as u128) >> 64) as usize
    }
    #[inline(always)]
    pub fn prefetch(&self, key: u64) {
        unsafe {
            let p = self.table.as_ptr().add(self.idx(key)) as *const i8;
            std::arch::x86_64::_mm_prefetch(p, std::arch::x86_64::_MM_HINT_T0);
        }
    }
    #[inline(always)]
    pub fn probe(&self, key: u64) -> Option<Entry> {
        let e = unsafe { *self.table.get_unchecked(self.idx(key)) };
        if e.key == key && e.bound != BOUND_NONE {
            Some(e)
        } else {
            None
        }
    }
    #[inline(always)]
    pub fn store(&mut self, key: u64, mv: Move, score: i32, eval: i32, depth: i32, bound: u8) {
        let i = self.idx(key);
        let e = unsafe { self.table.get_unchecked_mut(i) };
        if e.key != key || bound == BOUND_EXACT || depth + 3 >= e.depth as i32 {
            let mv = if mv == 0 && e.key == key { e.mv } else { mv };
            *e = Entry {
                key,
                mv,
                score: score as i16,
                eval: eval as i16,
                depth: depth.clamp(0, 255) as u8,
                bound,
            };
        }
    }
    pub fn hashfull(&self) -> usize {
        self.table.iter().take(1000).filter(|e| e.bound != BOUND_NONE).count()
    }
}
