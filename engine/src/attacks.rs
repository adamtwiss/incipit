// Attack tables: leapers + PEXT-indexed sliders, zobrist keys.
use std::arch::x86_64::{_pdep_u64, _pext_u64};

pub static mut KNIGHT: [u64; 64] = [0; 64];
pub static mut KING: [u64; 64] = [0; 64];
pub static mut PAWN_ATT: [[u64; 64]; 2] = [[0; 64]; 2];
pub static mut BETWEEN: [[u64; 64]; 64] = [[0; 64]; 64];
pub static mut LINE: [[u64; 64]; 64] = [[0; 64]; 64];
static mut R_MASK: [u64; 64] = [0; 64];
static mut B_MASK: [u64; 64] = [0; 64];
static mut R_OFF: [usize; 64] = [0; 64];
static mut B_OFF: [usize; 64] = [0; 64];
static mut SLIDE: [u64; 107648] = [0; 107648];

pub static mut ZOB_PIECE: [[u64; 64]; 12] = [[0; 64]; 12];
pub static mut ZOB_CASTLE: [u64; 16] = [0; 16];
pub static mut ZOB_EP: [u64; 8] = [0; 8];
pub static mut ZOB_STM: u64 = 0;

pub const FILE_A: u64 = 0x0101010101010101;
pub const FILE_H: u64 = FILE_A << 7;
pub const RANK_1: u64 = 0xff;
pub const RANK_3: u64 = 0xff << 16;
pub const RANK_6: u64 = 0xff << 40;
pub const RANK_8: u64 = 0xff << 56;

const ROOK_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

fn slide(sq: usize, occ: u64, dirs: &[(i32, i32)]) -> u64 {
    let mut a = 0u64;
    let (r0, f0) = ((sq / 8) as i32, (sq % 8) as i32);
    for &(dr, df) in dirs {
        let (mut r, mut f) = (r0 + dr, f0 + df);
        while (0..8).contains(&r) && (0..8).contains(&f) {
            let s = (r * 8 + f) as usize;
            a |= 1u64 << s;
            if occ & (1u64 << s) != 0 {
                break;
            }
            r += dr;
            f += df;
        }
    }
    a
}

fn slide_mask(sq: usize, dirs: &[(i32, i32)]) -> u64 {
    let mut a = 0u64;
    let (r0, f0) = ((sq / 8) as i32, (sq % 8) as i32);
    for &(dr, df) in dirs {
        let (mut r, mut f) = (r0 + dr, f0 + df);
        while (0..8).contains(&(r + dr)) && (0..8).contains(&(f + df)) {
            a |= 1u64 << (r * 8 + f);
            r += dr;
            f += df;
        }
    }
    a
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

pub fn init() {
    unsafe {
        for sq in 0..64usize {
            let (r, f) = ((sq / 8) as i32, (sq % 8) as i32);
            let mut n = 0u64;
            for (dr, df) in [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)] {
                let (rr, ff) = (r + dr, f + df);
                if (0..8).contains(&rr) && (0..8).contains(&ff) {
                    n |= 1u64 << (rr * 8 + ff);
                }
            }
            KNIGHT[sq] = n;
            let mut k = 0u64;
            for dr in -1..=1 {
                for df in -1..=1 {
                    if dr == 0 && df == 0 {
                        continue;
                    }
                    let (rr, ff) = (r + dr, f + df);
                    if (0..8).contains(&rr) && (0..8).contains(&ff) {
                        k |= 1u64 << (rr * 8 + ff);
                    }
                }
            }
            KING[sq] = k;
            let b = 1u64 << sq;
            PAWN_ATT[0][sq] = ((b & !FILE_A) << 7) | ((b & !FILE_H) << 9);
            PAWN_ATT[1][sq] = ((b & !FILE_A) >> 9) | ((b & !FILE_H) >> 7);
        }
        let mut off = 0usize;
        for sq in 0..64usize {
            R_MASK[sq] = slide_mask(sq, &ROOK_DIRS);
            R_OFF[sq] = off;
            let bits = R_MASK[sq].count_ones();
            for i in 0..(1u64 << bits) {
                let occ = _pdep_u64(i, R_MASK[sq]);
                SLIDE[off + i as usize] = slide(sq, occ, &ROOK_DIRS);
            }
            off += 1 << bits;
        }
        for sq in 0..64usize {
            B_MASK[sq] = slide_mask(sq, &BISHOP_DIRS);
            B_OFF[sq] = off;
            let bits = B_MASK[sq].count_ones();
            for i in 0..(1u64 << bits) {
                let occ = _pdep_u64(i, B_MASK[sq]);
                SLIDE[off + i as usize] = slide(sq, occ, &BISHOP_DIRS);
            }
            off += 1 << bits;
        }
        assert!(off == 107648);
        for a in 0..64usize {
            for b in 0..64usize {
                if a == b {
                    continue;
                }
                let ba = 1u64 << b;
                if slide(a, 0, &ROOK_DIRS) & ba != 0 {
                    BETWEEN[a][b] = slide(a, ba, &ROOK_DIRS) & slide(b, 1u64 << a, &ROOK_DIRS);
                    LINE[a][b] = (slide(a, 0, &ROOK_DIRS) & slide(b, 0, &ROOK_DIRS)) | (1u64 << a) | ba;
                } else if slide(a, 0, &BISHOP_DIRS) & ba != 0 {
                    BETWEEN[a][b] = slide(a, ba, &BISHOP_DIRS) & slide(b, 1u64 << a, &BISHOP_DIRS);
                    LINE[a][b] = (slide(a, 0, &BISHOP_DIRS) & slide(b, 0, &BISHOP_DIRS)) | (1u64 << a) | ba;
                }
            }
        }
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for p in 0..12 {
            for s in 0..64 {
                ZOB_PIECE[p][s] = rng.next();
            }
        }
        for c in 0..16 {
            ZOB_CASTLE[c] = 0;
        }
        let ck = [rng.next(), rng.next(), rng.next(), rng.next()];
        for c in 0..16 {
            for i in 0..4 {
                if c & (1 << i) != 0 {
                    ZOB_CASTLE[c] ^= ck[i];
                }
            }
        }
        for f in 0..8 {
            ZOB_EP[f] = rng.next();
        }
        ZOB_STM = rng.next();
    }
}

#[inline(always)]
pub fn rook_attacks(sq: usize, occ: u64) -> u64 {
    unsafe { *SLIDE.get_unchecked(R_OFF[sq] + _pext_u64(occ, R_MASK[sq]) as usize) }
}
#[inline(always)]
pub fn bishop_attacks(sq: usize, occ: u64) -> u64 {
    unsafe { *SLIDE.get_unchecked(B_OFF[sq] + _pext_u64(occ, B_MASK[sq]) as usize) }
}
#[inline(always)]
pub fn knight_attacks(sq: usize) -> u64 {
    unsafe { KNIGHT[sq] }
}
#[inline(always)]
pub fn king_attacks(sq: usize) -> u64 {
    unsafe { KING[sq] }
}
#[inline(always)]
pub fn pawn_attacks(c: usize, sq: usize) -> u64 {
    unsafe { PAWN_ATT[c][sq] }
}
#[inline(always)]
pub fn between(a: usize, b: usize) -> u64 {
    unsafe { BETWEEN[a][b] }
}
#[inline(always)]
pub fn line(a: usize, b: usize) -> u64 {
    unsafe { LINE[a][b] }
}
#[inline(always)]
pub fn zob_piece(pc: usize, sq: usize) -> u64 {
    unsafe { ZOB_PIECE[pc][sq] }
}
#[inline(always)]
pub fn zob_castle(c: u8) -> u64 {
    unsafe { ZOB_CASTLE[c as usize] }
}
#[inline(always)]
pub fn zob_ep(f: usize) -> u64 {
    unsafe { ZOB_EP[f] }
}
#[inline(always)]
pub fn zob_stm() -> u64 {
    unsafe { ZOB_STM }
}

#[inline(always)]
pub fn lsb(b: u64) -> usize {
    b.trailing_zeros() as usize
}

pub struct Bits(pub u64);
impl Iterator for Bits {
    type Item = usize;
    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            let s = self.0.trailing_zeros() as usize;
            self.0 &= self.0 - 1;
            Some(s)
        }
    }
}
