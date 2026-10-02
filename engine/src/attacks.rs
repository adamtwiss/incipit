// Attack tables: leapers, table-indexed sliders, zobrist keys.
//
// Sliders share one table holding a block of 2^bits entries per square, where
// bits is the number of squares in the square's occupancy mask. With BMI2 the
// block is indexed by PEXT of the occupancy; elsewhere (e.g. arm64) by a magic
// multiply. The order within a block differs between the two, but the attack
// sets are the same.
#[cfg(target_feature = "bmi2")]
use std::arch::x86_64::_pext_u64;
use std::ptr::{addr_of, addr_of_mut};

pub static mut KNIGHT: [u64; 64] = [0; 64];
pub static mut KING: [u64; 64] = [0; 64];
pub static mut PAWN_ATT: [[u64; 64]; 2] = [[0; 64]; 2];
pub static mut BETWEEN: [[u64; 64]; 64] = [[0; 64]; 64];
pub static mut LINE: [[u64; 64]; 64] = [[0; 64]; 64];
static mut R_MASK: [u64; 64] = [0; 64];
static mut B_MASK: [u64; 64] = [0; 64];
static mut R_OFF: [usize; 64] = [0; 64];
static mut B_OFF: [usize; 64] = [0; 64];
// Shifts (64 - bits) for the magic index; unused with BMI2.
static mut R_SHIFT: [u8; 64] = [0; 64];
static mut B_SHIFT: [u8; 64] = [0; 64];
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

// Magic multipliers for the non-BMI2 index, one per square, found by the
// `find_magics` test below (Incipit's own random search); baked in so that
// startup doesn't repeat the search. Unused with BMI2.
const R_MAGIC: [u64; 64] = [
    0x0280041040008021,
    0x0440042000401000,
    0x0080086002d00080,
    0x4080041002080080,
    0x2700080004100700,
    0x510001003400a802,
    0x0080088042002500,
    0x0200020484002443,
    0x0208800040008420,
    0x3048401005200040,
    0x0400801004802000,
    0x8002004200082010,
    0x1002000812000420,
    0x0244800200040080,
    0x000100a10042000c,
    0x2843000880510002,
    0x0280208000400080,
    0x1004484001a01002,
    0x2002410010200100,
    0x0401010008100060,
    0x0201808004022800,
    0x040081800a002400,
    0x8200040010030208,
    0x000012000048a104,
    0xc011843080044000,
    0x2200400080806000,
    0x000a150100412000,
    0x0001890100100024,
    0x2820050100080010,
    0x00000200802c0080,
    0x004b490400082210,
    0x002000820000d104,
    0x0104400080800030,
    0x2410016002404000,
    0x04a0021000802180,
    0x0088080280801000,
    0x4108080080800c00,
    0x0290020080800400,
    0x0802000c02000148,
    0x4300628042000104,
    0x0000450080030020,
    0x480060095000c001,
    0x08200100c1310020,
    0x0120700008008080,
    0x0201004800050030,
    0x1242009810020004,
    0x1063020108540090,
    0x0000024311a20004,
    0x0004400080042080,
    0x0000200840100240,
    0x0900e00100124100,
    0x0100080080100080,
    0x0989080080140080,
    0x0208120080040080,
    0x2204820830010400,
    0x2012040105804600,
    0x8240130123820042,
    0x0408400104802011,
    0x4044200211000941,
    0x0800100049000421,
    0x1122012030940802,
    0x201100081400d201,
    0x0040501208408504,
    0x000112270084004a,
];
const B_MAGIC: [u64; 64] = [
    0x1020182080888e00,
    0x0402088a04004082,
    0x000c190202010400,
    0x8094040184008000,
    0x0084042001050040,
    0x0082411040040820,
    0x0245028220200028,
    0x0002004402051101,
    0x404020041058810c,
    0x80000a0408020840,
    0x0040100100650a20,
    0x2220881610402210,
    0x82041c0422200060,
    0x008080900c200400,
    0x004000c602202004,
    0x0020004208030801,
    0x002002124c411800,
    0x3010002024888080,
    0x00080010002025a0,
    0x0008000404123101,
    0x8124000206110010,
    0x01041022010c0104,
    0x5006020402020200,
    0x0080200a03110808,
    0x905040c0240d0402,
    0x0881200004089212,
    0x080048000c042400,
    0x29a1080184004010,
    0x0224840100802000,
    0x0010006001008809,
    0x4004010004210500,
    0x0000808803240401,
    0x1012200500101000,
    0x1068481868422600,
    0x0004084800040821,
    0x2000420084080080,
    0x20a0450040040040,
    0x0004110200040888,
    0x01410a0182040c25,
    0x4008940100414130,
    0x000210108800c451,
    0x200888015000880a,
    0x0018501090040800,
    0x0008430c24000800,
    0x0800510122000c00,
    0x005020480c200040,
    0x04a0240114480600,
    0x0010188109010040,
    0x4183008220200080,
    0x80c1444208200020,
    0x0002390088040442,
    0x0500000142020108,
    0x0980204008221411,
    0x0001600202620000,
    0x22a00204010c0070,
    0x200808080640c007,
    0x0003008550080400,
    0x0000006608040410,
    0x08288002aa111010,
    0x1200000004842400,
    0x0000080070211880,
    0x09401804100e0600,
    0x1200092230222602,
    0x07d4491003010104,
];

/// Index of occupancy `occ` within a square's block of the slider table.
#[cfg(target_feature = "bmi2")]
#[inline(always)]
fn index(occ: u64, mask: u64, _magic: u64, _shift: u8) -> usize {
    unsafe { _pext_u64(occ, mask) as usize }
}

#[cfg(not(target_feature = "bmi2"))]
#[inline(always)]
fn index(occ: u64, mask: u64, magic: u64, shift: u8) -> usize {
    ((occ & mask).wrapping_mul(magic) >> shift) as usize
}

/// Fills square sq's block of SLIDE at `off` with the attacks for every
/// subset of its mask, and returns (mask, shift). Attack sets are never
/// empty, so a non-zero slot holding different attacks means the index
/// collides (a bad magic).
unsafe fn init_slider(sq: usize, dirs: &[(i32, i32)], off: usize, magic: u64) -> (u64, u8) {
    let mask = slide_mask(sq, dirs);
    let shift = (64 - mask.count_ones()) as u8;
    let block = addr_of_mut!(SLIDE).cast::<u64>().add(off);
    let mut occ = 0u64;
    loop {
        let a = slide(sq, occ, dirs);
        let slot = block.add(index(occ, mask, magic, shift));
        assert!(*slot == 0 || *slot == a, "slider magic for square {} collides", sq);
        *slot = a;
        occ = occ.wrapping_sub(mask) & mask;
        if occ == 0 {
            return (mask, shift);
        }
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
            R_OFF[sq] = off;
            (R_MASK[sq], R_SHIFT[sq]) = init_slider(sq, &ROOK_DIRS, off, R_MAGIC[sq]);
            off += 1 << R_MASK[sq].count_ones();
        }
        for sq in 0..64usize {
            B_OFF[sq] = off;
            (B_MASK[sq], B_SHIFT[sq]) = init_slider(sq, &BISHOP_DIRS, off, B_MAGIC[sq]);
            off += 1 << B_MASK[sq].count_ones();
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
    unsafe { *addr_of!(SLIDE).cast::<u64>().add(R_OFF[sq] + index(occ, R_MASK[sq], R_MAGIC[sq], R_SHIFT[sq])) }
}
#[inline(always)]
pub fn bishop_attacks(sq: usize, occ: u64) -> u64 {
    unsafe { *addr_of!(SLIDE).cast::<u64>().add(B_OFF[sq] + index(occ, B_MASK[sq], B_MAGIC[sq], B_SHIFT[sq])) }
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
#[allow(dead_code)]
#[inline(always)]
pub fn between(a: usize, b: usize) -> u64 {
    unsafe { BETWEEN[a][b] }
}
#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Random search for a magic that fits square sq's occupancy subsets into
    /// 2^bits slots: sparse random candidates are tried until every subset
    /// lands in its own slot or one already holding the same attacks.
    fn find_magic(sq: usize, dirs: &[(i32, i32)], rng: &mut Rng) -> u64 {
        let mask = slide_mask(sq, dirs);
        let bits = mask.count_ones();
        let shift = (64 - bits) as u8;
        let mut table = vec![0u64; 1 << bits];
        'next: loop {
            let magic = rng.next() & rng.next() & rng.next();
            table.fill(0);
            let mut occ = 0u64;
            loop {
                let a = slide(sq, occ, dirs);
                let slot = &mut table[((occ & mask).wrapping_mul(magic) >> shift) as usize];
                if *slot != 0 && *slot != a {
                    continue 'next;
                }
                *slot = a;
                occ = occ.wrapping_sub(mask) & mask;
                if occ == 0 {
                    return magic;
                }
            }
        }
    }

    /// Prints R_MAGIC and B_MAGIC. Regenerate them (e.g. after changing the
    /// masks) with `cargo test --release find_magics -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn find_magics() {
        let mut rng = Rng(0xD1B54A32D192ED03);
        for (name, dirs) in [("R_MAGIC", &ROOK_DIRS), ("B_MAGIC", &BISHOP_DIRS)] {
            println!("const {}: [u64; 64] = [", name);
            for sq in 0..64 {
                println!("    0x{:016x},", find_magic(sq, dirs, &mut rng));
            }
            println!("];");
        }
    }

    /// Table lookups agree with a direct ray walk for random occupancies
    /// (checks the PEXT or magic indexing, whichever the build uses).
    #[test]
    fn slider_tables() {
        init();
        let mut rng = Rng(0x2545F4914F6CDD1D);
        for sq in 0..64 {
            for _ in 0..500 {
                let occ = rng.next() & rng.next();
                assert_eq!(rook_attacks(sq, occ), slide(sq, occ, &ROOK_DIRS), "rook sq {} occ {:x}", sq, occ);
                assert_eq!(bishop_attacks(sq, occ), slide(sq, occ, &BISHOP_DIRS), "bishop sq {} occ {:x}", sq, occ);
            }
        }
    }
}
