// NNUE inference: (768 -> H)x2, SCReLU, 8 material output buckets.
use crate::attacks::Bits;
use crate::position::*;

pub const H: usize = 512;
const NB: usize = 8;
pub const NKB: usize = 1;
pub const MIRROR: bool = true;

#[inline(always)]
fn kb_of(ks: usize) -> usize {
    if NKB == 1 {
        return 0;
    }
    let f = ks & 7;
    let r = ks >> 3;
    let ff = if f >= 4 { 7 - f } else { f };
    match r {
        0 => ff / 2,
        1 => 2 + ff / 2,
        2 => 4,
        3 => 5,
        4 | 5 => 6,
        _ => 7,
    }
}

/// (bucket offset, square xor) for perspective p given its king square.
#[inline(always)]
fn kinfo(p: usize, ksq: usize) -> (usize, usize) {
    let ks = if p == WHITE { ksq } else { ksq ^ 56 };
    let x = if (NKB > 1 || MIRROR) && (ks & 7) >= 4 { 7 } else { 0 };
    (kb_of(ks) * 768, x)
}
const QA: i32 = 255;
const QB: i32 = 64;

static NET_BYTES: &[u8] = include_bytes!("../net.nnue");

#[repr(C, align(64))]
pub struct Network {
    ftw: [[i16; H]; 768 * NKB],
    ftb: [i16; H],
    ow: [[i16; 2 * H]; NB],
    ob: [i32; NB],
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Acc {
    pub v: [[i16; H]; 2],
}

static mut NET: *const Network = std::ptr::null();

pub fn init() {
    let expect = 768 * NKB * H * 2 + H * 2 + NB * 2 * H * 2 + NB * 4;
    assert_eq!(NET_BYTES.len(), expect, "net size mismatch");
    let mut net: Box<Network> = unsafe {
        let layout = std::alloc::Layout::new::<Network>();
        let p = std::alloc::alloc_zeroed(layout) as *mut Network;
        Box::from_raw(p)
    };
    let mut o = 0;
    let rd16 = |o: &mut usize| -> i16 {
        let v = i16::from_le_bytes([NET_BYTES[*o], NET_BYTES[*o + 1]]);
        *o += 2;
        v
    };
    for f in 0..768 * NKB {
        for j in 0..H {
            net.ftw[f][j] = rd16(&mut o);
        }
    }
    for j in 0..H {
        net.ftb[j] = rd16(&mut o);
    }
    for b in 0..NB {
        for j in 0..2 * H {
            net.ow[b][j] = rd16(&mut o);
        }
    }
    for b in 0..NB {
        net.ob[b] = i32::from_le_bytes(NET_BYTES[o..o + 4].try_into().unwrap());
        o += 4;
    }
    unsafe {
        NET = Box::into_raw(net);
    }
}

#[inline(always)]
fn net() -> &'static Network {
    unsafe { &*NET }
}

#[inline(always)]
fn feat(persp: usize, ki: (usize, usize), pc: u8, sq: usize) -> usize {
    let c = pc_color(pc);
    let pt = pc_type(pc);
    if persp == WHITE {
        ki.0 + c * 384 + pt * 64 + (sq ^ ki.1)
    } else {
        ki.0 + (c ^ 1) * 384 + pt * 64 + (sq ^ 56 ^ ki.1)
    }
}

impl Acc {
    pub fn refresh(&mut self, pos: &Position) {
        for p in 0..2 {
            self.refresh_persp(p, pos);
        }
    }

    fn refresh_persp(&mut self, p: usize, pos: &Position) {
        let n = net();
        let ki = kinfo(p, pos.king_sq(p));
        self.v[p] = n.ftb;
        for sq in Bits(pos.occ()) {
            let f = feat(p, ki, pos.board[sq], sq);
            let row = &n.ftw[f];
            for j in 0..H {
                self.v[p][j] = self.v[p][j].wrapping_add(row[j]);
            }
        }
    }

    /// Compute accumulator of `child` after move m made in parent position `pos`.
    #[inline]
    pub fn update_from(&mut self, parent: &Acc, pos: &Position, child: &Position, m: Move) {
        let n = net();
        let from = mfrom(m);
        let to = mto(m);
        let flag = mflag(m);
        let pc = pos.board[from];
        let us = pos.stm;
        let newpc = if flag & 8 != 0 { make_pc(promo_pt(m), us) } else { pc };
        let mut adds: [(u8, usize); 2] = [(newpc, to), (0, 64)];
        let mut subs: [(u8, usize); 2] = [(pc, from), (0, 64)];
        let mut na = 1;
        let mut ns = 1;
        if flag == F_EP {
            subs[1] = (make_pc(PAWN, us ^ 1), to ^ 8);
            ns = 2;
        } else if pos.board[to] != NONE_PC {
            subs[1] = (pos.board[to], to);
            ns = 2;
        } else if flag == F_KCASTLE {
            subs[1] = (make_pc(ROOK, us), to + 1);
            adds[1] = (make_pc(ROOK, us), to - 1);
            ns = 2;
            na = 2;
        } else if flag == F_QCASTLE {
            subs[1] = (make_pc(ROOK, us), to - 2);
            adds[1] = (make_pc(ROOK, us), to + 1);
            ns = 2;
            na = 2;
        }
        for p in 0..2 {
            let ki = kinfo(p, pos.king_sq(p));
            if (NKB > 1 || MIRROR) && p == us && pc_type(pc) == KING && kinfo(p, to) != ki {
                self.refresh_persp(p, child);
                continue;
            }
            let a0 = &n.ftw[feat(p, ki, adds[0].0, adds[0].1)];
            let s0 = &n.ftw[feat(p, ki, subs[0].0, subs[0].1)];
            let src = &parent.v[p];
            let dst = &mut self.v[p];
            if ns == 1 {
                for j in 0..H {
                    dst[j] = src[j].wrapping_add(a0[j]).wrapping_sub(s0[j]);
                }
            } else if na == 1 {
                let s1 = &n.ftw[feat(p, ki, subs[1].0, subs[1].1)];
                for j in 0..H {
                    dst[j] = src[j].wrapping_add(a0[j]).wrapping_sub(s0[j]).wrapping_sub(s1[j]);
                }
            } else {
                let s1 = &n.ftw[feat(p, ki, subs[1].0, subs[1].1)];
                let a1 = &n.ftw[feat(p, ki, adds[1].0, adds[1].1)];
                for j in 0..H {
                    dst[j] = src[j].wrapping_add(a0[j]).wrapping_sub(s0[j]).wrapping_add(a1[j]).wrapping_sub(s1[j]);
                }
            }
        }
    }
}

#[inline]
pub fn evaluate(acc: &Acc, pos: &Position) -> i32 {
    let n = net();
    let bucket = ((pos.occ().count_ones() as usize - 2) / 4).min(NB - 1);
    let us = &acc.v[pos.stm];
    let them = &acc.v[pos.stm ^ 1];
    let w = &n.ow[bucket];
    let sum = unsafe { screlu_dot(us, &w[..H]) + screlu_dot(them, &w[H..]) };
    let out = (sum / QA + n.ob[bucket]) * 400 / (QA * QB);
    out
}

#[inline(always)]
unsafe fn screlu_dot(a: &[i16; H], w: &[i16]) -> i32 {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(QA as i16);
    let mut s0 = _mm256_setzero_si256();
    let mut s1 = _mm256_setzero_si256();
    let mut i = 0;
    while i < H {
        let x0 = _mm256_load_si256(a.as_ptr().add(i) as *const __m256i);
        let x1 = _mm256_load_si256(a.as_ptr().add(i + 16) as *const __m256i);
        let c0 = _mm256_min_epi16(_mm256_max_epi16(x0, zero), qa);
        let c1 = _mm256_min_epi16(_mm256_max_epi16(x1, zero), qa);
        let w0 = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
        let w1 = _mm256_loadu_si256(w.as_ptr().add(i + 16) as *const __m256i);
        s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_mullo_epi16(c0, w0), c0));
        s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_mullo_epi16(c1, w1), c1));
        i += 32;
    }
    let s = _mm256_add_epi32(s0, s1);
    let lo = _mm256_castsi256_si128(s);
    let hi = _mm256_extracti128_si256(s, 1);
    let x = _mm_add_epi32(lo, hi);
    let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0b01_00_11_10));
    let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0b10_11_00_01));
    _mm_cvtsi128_si32(x)
}
