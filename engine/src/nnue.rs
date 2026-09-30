// NNUE inference. The architecture is read from the network's header (see
// docs/net-format.md): (768 x king buckets -> H) x 2 perspectives, with
// optional horizontal mirroring, a SCReLU or CReLU activation, and a
// material-count output-bucketed output layer.
use crate::attacks::Bits;
use crate::position::*;

/// Largest hidden size the accumulators can hold.
pub const MAX_H: usize = 2048;

const MAGIC: &[u8; 8] = b"INCPTNET";
const TAG_INPUTS: u16 = 0x0001;
const TAG_KING_BUCKETS: u16 = 0x0002;
const TAG_FT: u16 = 0x0003;
const TAG_OUTPUT_BUCKETS: u16 = 0x0004;
const TAG_LAYER: u16 = 0x0005;
const TAG_QUANT: u16 = 0x0006;
const OPTIONAL: u16 = 0x8000;
const INPUT_PSQ768: u16 = 1;
const ACT_NONE: u8 = 0;
const ACT_CRELU: u8 = 2;
const ACT_SCRELU: u8 = 3;
const ACT_PAIRWISE: u8 = 4;
const TYPE_I16: u8 = 2;
const TYPE_I32: u8 = 3;
const OUTPUT_MATERIAL: u8 = 1;

// Path chosen by build.rs (EVALFILE, or the net named in net.txt).
static NET_BYTES: &[u8] = include_bytes!(env!("INCIPIT_NET"));

pub struct Network {
    h: usize,
    mirror: bool,
    refresh_on_king_move: bool,
    king_bucket: [u8; 64],
    bucket_of: [u8; 33], // output bucket by piece count
    qa: i32,
    qb: i32,
    scale: i32,
    ftw: Aligned, // [king bucket][feature][h]
    ftb: Aligned, // [h]
    ow: Aligned,  // [output bucket][2h], side to move first
    ob: Vec<i32>,  // [output bucket]
    eval: EvalFn,  // chosen at load for this hidden size and activation
}

type EvalFn = fn(&Network, &Acc, &Position) -> i32;

/// Both perspectives' accumulators, packed: white's at [0, h), black's at
/// [h, 2h), so the part in use is contiguous whatever the network's size.
/// An i16 buffer aligned to 64 bytes, so a row of weights (h, a multiple of
/// 32) never straddles cache lines in the SIMD loops.
struct Aligned {
    ptr: *mut i16,
    len: usize,
}

impl Aligned {
    fn layout(len: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(len.max(1) * 2, 64).unwrap()
    }

    fn from(values: impl ExactSizeIterator<Item = i16>) -> Aligned {
        let len = values.len();
        let ptr = unsafe { std::alloc::alloc(Self::layout(len)) } as *mut i16;
        assert!(!ptr.is_null(), "out of memory loading the network");
        for (i, v) in values.enumerate() {
            unsafe { ptr.add(i).write(v) };
        }
        Aligned { ptr, len }
    }
}

impl std::ops::Deref for Aligned {
    type Target = [i16];
    fn deref(&self) -> &[i16] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr as *mut u8, Self::layout(self.len)) }
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Acc {
    pub v: [i16; 2 * MAX_H],
}

static mut NET: *const Network = std::ptr::null();

pub fn init() {
    match load(NET_BYTES) {
        Ok(net) => unsafe { NET = Box::into_raw(Box::new(net)) },
        Err(e) => {
            eprintln!("error: embedded network: {}", e);
            std::process::exit(1);
        }
    }
}

/// Parses and validates a network file.
pub fn load(d: &[u8]) -> Result<Network, String> {
    let u16_at = |o: usize| u16::from_le_bytes([d[o], d[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]);
    if d.len() < 16 || &d[0..8] != MAGIC {
        return Err("not an Incipit network (bad magic); old headerless nets must be converted with `datatools net raw`".into());
    }
    if u16_at(8) != 1 {
        return Err(format!("unsupported format version {}", u16_at(8)));
    }
    let header = u32_at(12) as usize;
    if header > d.len() {
        return Err("header size is beyond the end of the file".into());
    }

    let mut inputs = None;
    let mut kb = None;
    let mut ft = None;
    let mut outb = None;
    let mut layers = Vec::new();
    let mut quant = None;
    let mut o = 16;
    while o + 8 <= header {
        let (tag, len) = (u16_at(o), u32_at(o + 4) as usize);
        if tag == 0 {
            break;
        }
        if o + 8 + len > header {
            return Err(format!("field 0x{:04x} overruns the header", tag));
        }
        let p = &d[o + 8..o + 8 + len];
        let need = |n: usize| if len < n { Err(format!("field 0x{:04x} is too short", tag)) } else { Ok(()) };
        let p32 = |i: usize| u32::from_le_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]]);
        match tag {
            TAG_INPUTS => {
                need(2)?;
                let n = u16::from_le_bytes([p[0], p[1]]) as usize;
                need(2 + 8 * n)?;
                inputs = Some((0..n).map(|i| (u16::from_le_bytes([p[2 + 8 * i], p[3 + 8 * i]]), p32(6 + 8 * i))).collect::<Vec<_>>());
            }
            TAG_KING_BUCKETS => {
                need(68)?;
                kb = Some((p[0] as usize, p[1] != 0, <[u8; 64]>::try_from(&p[4..68]).unwrap()));
            }
            TAG_FT => {
                need(8)?;
                ft = Some((p32(0) as usize, p[4], p[5], p[6]));
            }
            TAG_OUTPUT_BUCKETS => {
                need(2)?;
                outb = Some((p[0], p[1] as usize));
            }
            TAG_LAYER => {
                need(12)?;
                layers.push((p32(0) as usize, p32(4) as usize, p[8], p[9], p[10], p[11]));
            }
            TAG_QUANT => {
                need(12)?;
                quant = Some((p32(0) as i32, p32(4) as i32, p32(8) as i32));
            }
            t if t & OPTIONAL != 0 => {}
            t => return Err(format!("unknown required field 0x{:04x}", t)),
        }
        o += 8 + len;
    }

    let missing = |name: &str| format!("missing {} field", name);
    let inputs = inputs.ok_or_else(|| missing("INPUTS"))?;
    if inputs != [(INPUT_PSQ768, 768)] {
        return Err(format!("unsupported inputs {:?} (only a single 768 piece-square set)", inputs));
    }
    let (nkb, mirror, king_bucket) = kb.ok_or_else(|| missing("KING_BUCKETS"))?;
    if nkb == 0 || king_bucket.iter().any(|&b| b as usize >= nkb) {
        return Err(format!("king bucket table doesn't match {} bucket(s)", nkb));
    }
    let (h, act, wt, bt) = ft.ok_or_else(|| missing("FEATURE_TRANSFORMER"))?;
    if h == 0 || h > MAX_H || h % 32 != 0 {
        return Err(format!("hidden size {} unsupported (must be a multiple of 32, at most {})", h, MAX_H));
    }
    if act != ACT_SCRELU && act != ACT_CRELU && act != ACT_PAIRWISE {
        return Err(format!("unsupported feature transformer activation {}", act));
    }
    // Pairwise halves each perspective; each half must still be a multiple of 32.
    if act == ACT_PAIRWISE && h % 64 != 0 {
        return Err(format!("pairwise needs a hidden size that is a multiple of 64, got {}", h));
    }
    let out_in = if act == ACT_PAIRWISE { h } else { 2 * h };
    if wt != TYPE_I16 || bt != TYPE_I16 {
        return Err("feature transformer weights and biases must be i16".into());
    }
    let (scheme, buckets) = outb.ok_or_else(|| missing("OUTPUT_BUCKETS"))?;
    if scheme != OUTPUT_MATERIAL || buckets == 0 || buckets > 32 {
        return Err(format!("unsupported output buckets (scheme {}, {} buckets)", scheme, buckets));
    }
    let &[(lin, lout, lact, lwt, lbt, lflags)] = layers.as_slice() else {
        return Err(format!("{} output layers; only one is supported", layers.len()));
    };
    if lin != out_in || lout != 1 || lact != ACT_NONE || lwt != TYPE_I16 || !(lbt == TYPE_I16 || lbt == TYPE_I32) || lflags & 1 == 0 {
        return Err(format!("unsupported output layer (need {} -> 1, no activation, i16 weights, per output bucket)", out_in));
    }
    let (qa, qb, scale) = quant.ok_or_else(|| missing("QUANTISATION"))?;

    let bias_size = if lbt == TYPE_I32 { 4 } else { 2 };
    let expect = 2 * (nkb * 768 * h + h + buckets * out_in) + bias_size * buckets;
    if d.len() - header != expect {
        return Err(format!("weights are {} bytes, but the header describes {}", d.len() - header, expect));
    }
    let mut o = header;
    let mut i16s = |n: usize| {
        let v = Aligned::from(d[o..o + 2 * n].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])));
        o += 2 * n;
        v
    };
    let ftw = i16s(nkb * 768 * h);
    let ftb = i16s(h);
    let ow = i16s(buckets * out_in);
    let ob = if bias_size == 2 {
        i16s(buckets).iter().map(|&b| i32::from(b)).collect()
    } else {
        d[o..].chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    Ok(Network {
        h,
        mirror,
        refresh_on_king_move: nkb > 1 || mirror,
        king_bucket,
        eval: select_eval(h, act),
        bucket_of: std::array::from_fn(|pieces| (pieces.saturating_sub(2) / 32usize.div_ceil(buckets)).min(buckets - 1) as u8),
        qa,
        qb,
        scale,
        ftw,
        ftb,
        ow,
        ob,
    })
}

#[inline(always)]
fn net() -> &'static Network {
    unsafe { &*NET }
}

/// (feature offset of the king bucket, square xor) for perspective p.
#[inline(always)]
fn kinfo(n: &Network, p: usize, ksq: usize) -> (usize, usize) {
    let ks = if p == WHITE { ksq } else { ksq ^ 56 };
    let x = if n.mirror && (ks & 7) >= 4 { 7 } else { 0 };
    (n.king_bucket[ks] as usize * 768, x)
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

/// Calls `$f::<H>($args)` with H the loaded network's hidden size as a
/// constant for common sizes (0 = read it at runtime). Everything sized by H
/// (row and accumulator offsets, loop lengths) then compiles to constants and
/// fully unrolled loops; the size is matched once per call.
macro_rules! with_h {
    ($h:expr, $f:ident($($arg:expr),*)) => {
        match $h {
            128 => $f::<128>($($arg),*),
            256 => $f::<256>($($arg),*),
            512 => $f::<512>($($arg),*),
            768 => $f::<768>($($arg),*),
            1024 => $f::<1024>($($arg),*),
            1536 => $f::<1536>($($arg),*),
            2048 => $f::<2048>($($arg),*),
            _ => $f::<0>($($arg),*),
        }
    };
}

#[inline(always)]
fn hidden<const H: usize>(n: &Network) -> usize {
    if H > 0 { H } else { n.h }
}

/// Feature-transformer row `f` (length h).
#[inline(always)]
fn row(n: &Network, f: usize, h: usize) -> &[i16] {
    // Safe: load() checks ftw holds king buckets * 768 rows of h, and feat()
    // stays below that.
    unsafe { n.ftw.get_unchecked(f * h..(f + 1) * h) }
}

impl Acc {
    pub fn new() -> Self {
        Acc { v: [0; 2 * MAX_H] }
    }

    // Safe: p < 2 and load() checks h <= MAX_H.
    #[inline(always)]
    fn side(&self, p: usize, h: usize) -> &[i16] {
        unsafe { self.v.get_unchecked(p * h..(p + 1) * h) }
    }

    #[inline(always)]
    fn side_mut(&mut self, p: usize, h: usize) -> &mut [i16] {
        unsafe { self.v.get_unchecked_mut(p * h..(p + 1) * h) }
    }

    /// Copies only the part of `other` the loaded network uses.
    #[inline(always)]
    pub fn copy_from(&mut self, other: &Acc) {
        let h2 = 2 * net().h;
        self.v[..h2].copy_from_slice(&other.v[..h2]);
    }

    pub fn refresh(&mut self, pos: &Position) {
        let n = net();
        for p in 0..2 {
            with_h!(n.h, refresh_persp(self, n, p, pos));
        }
    }

    /// Compute accumulator of `child` after move m made in parent position `pos`.
    #[inline]
    pub fn update_from(&mut self, parent: &Acc, pos: &Position, child: &Position, m: Move) {
        let n = net();
        with_h!(n.h, update(self, n, parent, pos, child, m));
    }
}

#[inline(never)]
fn refresh_persp<const H: usize>(acc: &mut Acc, n: &Network, p: usize, pos: &Position) {
    let h = hidden::<H>(n);
    let ki = kinfo(n, p, pos.king_sq(p));
    let v = acc.side_mut(p, h);
    v.copy_from_slice(&n.ftb[..h]);
    for sq in Bits(pos.occ()) {
        let r = row(n, feat(p, ki, pos.board[sq], sq), h);
        for (x, &w) in v.iter_mut().zip(r) {
            *x = x.wrapping_add(w);
        }
    }
}

#[inline(never)]
fn update<const H: usize>(acc: &mut Acc, n: &Network, parent: &Acc, pos: &Position, child: &Position, m: Move) {
    let h = hidden::<H>(n);
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
        let ki = kinfo(n, p, pos.king_sq(p));
        if n.refresh_on_king_move && p == us && pc_type(pc) == KING && kinfo(n, p, to) != ki {
            refresh_persp::<H>(acc, n, p, child);
            continue;
        }
        let a0 = row(n, feat(p, ki, adds[0].0, adds[0].1), h);
        let s0 = row(n, feat(p, ki, subs[0].0, subs[0].1), h);
        let src = parent.side(p, h);
        let dst = acc.side_mut(p, h);
        if ns == 1 {
            add_sub(dst, src, [a0], [s0]);
        } else if na == 1 {
            let s1 = row(n, feat(p, ki, subs[1].0, subs[1].1), h);
            add_sub(dst, src, [a0], [s0, s1]);
        } else {
            let s1 = row(n, feat(p, ki, subs[1].0, subs[1].1), h);
            let a1 = row(n, feat(p, ki, adds[1].0, adds[1].1), h);
            add_sub(dst, src, [a0, a1], [s0, s1]);
        }
    }
}

/// dst = src + Σ adds − Σ subs. Plain Rust over equal-length slices, so the
/// compiler vectorises at the widest width the build target allows
/// (AVX-512 where available) and fully unrolls when the length is constant.
#[inline(always)]
fn add_sub<const NA: usize, const NS: usize>(dst: &mut [i16], src: &[i16], adds: [&[i16]; NA], subs: [&[i16]; NS]) {
    let h = dst.len();
    let src = &src[..h];
    let adds = adds.map(|a| &a[..h]);
    let subs = subs.map(|s| &s[..h]);
    for i in 0..h {
        let mut x = src[i];
        for a in &adds {
            x = x.wrapping_add(a[i]);
        }
        for s in &subs {
            x = x.wrapping_sub(s[i]);
        }
        dst[i] = x;
    }
}

#[inline]
pub fn evaluate(acc: &Acc, pos: &Position) -> i32 {
    let n = net();
    (n.eval)(n, acc, pos)
}

/// Picks the eval specialised for this hidden size (a constant for common
/// sizes, 0 = runtime) and activation, so evaluate() itself has no branches.
fn select_eval(h: usize, act: u8) -> EvalFn {
    macro_rules! for_act {
        ($h:literal) => {
            match act {
                ACT_PAIRWISE => eval_n::<$h, ACT_PAIRWISE> as EvalFn,
                ACT_CRELU => eval_n::<$h, ACT_CRELU> as EvalFn,
                _ => eval_n::<$h, ACT_SCRELU> as EvalFn,
            }
        };
    }
    match h {
        128 => for_act!(128),
        256 => for_act!(256),
        512 => for_act!(512),
        768 => for_act!(768),
        1024 => for_act!(1024),
        1536 => for_act!(1536),
        2048 => for_act!(2048),
        _ => for_act!(0),
    }
}

fn eval_n<const H: usize, const ACT: u8>(n: &Network, acc: &Acc, pos: &Position) -> i32 {
    let h = hidden::<H>(n);
    let out_in = if ACT == ACT_PAIRWISE { h } else { 2 * h };
    let bucket = n.bucket_of[pos.occ().count_ones() as usize] as usize;
    let us = acc.side(pos.stm, h);
    let them = acc.side(pos.stm ^ 1, h);
    let w = &n.ow[bucket * out_in..(bucket + 1) * out_in];
    // The common quantisation gets constant divisors (cheap multiplies instead
    // of hardware divides); the results are identical.
    let (qa, qb) = if n.qa == 255 && n.qb == 64 { (255, 64) } else { (n.qa, n.qb) };
    if ACT == ACT_CRELU {
        let sum = unsafe { dot::<false>(us, &w[..h], qa) + dot::<false>(them, &w[h..], qa) };
        return (sum + n.ob[bucket]) * n.scale / (qa * qb);
    }
    // SCReLU and pairwise share the QA^2 scale, so the output formula is the same.
    let sum = if ACT == ACT_PAIRWISE {
        let half = h / 2;
        unsafe { pairwise_dot(us, &w[..half], qa) + pairwise_dot(them, &w[half..h], qa) }
    } else {
        unsafe { dot::<true>(us, &w[..h], qa) + dot::<true>(them, &w[h..], qa) }
    };
    if qa == 255 && qb == 64 {
        (sum / 255 + n.ob[bucket]) * n.scale / (255 * 64)
    } else {
        (sum / qa + n.ob[bucket]) * n.scale / (qa * qb)
    }
}

#[inline(always)]
unsafe fn hsum(s: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castsi256_si128(s);
    let hi = _mm256_extracti128_si256(s, 1);
    let x = _mm_add_epi32(lo, hi);
    let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0b01_00_11_10));
    let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0b10_11_00_01));
    _mm_cvtsi128_si32(x)
}

/// SCReLU: Σ clamp(a, 0, qa)² · w; CReLU: Σ clamp(a, 0, qa) · w, over
/// a.len() values (a multiple of 32). `a` is 64-byte aligned.
#[inline(always)]
unsafe fn dot<const SCRELU: bool>(a: &[i16], w: &[i16], qa: i32) -> i32 {
    use std::arch::x86_64::*;
    let h = a.len();
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(qa as i16);
    let mut s0 = _mm256_setzero_si256();
    let mut s1 = _mm256_setzero_si256();
    let mut i = 0;
    while i < h {
        let x0 = _mm256_load_si256(a.as_ptr().add(i) as *const __m256i);
        let x1 = _mm256_load_si256(a.as_ptr().add(i + 16) as *const __m256i);
        let c0 = _mm256_min_epi16(_mm256_max_epi16(x0, zero), qa);
        let c1 = _mm256_min_epi16(_mm256_max_epi16(x1, zero), qa);
        let w0 = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
        let w1 = _mm256_loadu_si256(w.as_ptr().add(i + 16) as *const __m256i);
        if SCRELU {
            s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_mullo_epi16(c0, w0), c0));
            s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_mullo_epi16(c1, w1), c1));
        } else {
            s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(c0, w0));
            s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(c1, w1));
        }
        i += 32;
    }
    hsum(_mm256_add_epi32(s0, s1))
}

/// Pairwise: Σ_{j < h/2} clamp(a[j], 0, qa) · clamp(a[j + h/2], 0, qa) · w[j],
/// with h = a.len() a multiple of 64 and `a` 64-byte aligned (so both halves are).
#[inline(always)]
unsafe fn pairwise_dot(a: &[i16], w: &[i16], qa: i32) -> i32 {
    use std::arch::x86_64::*;
    let half = a.len() / 2;
    let (lo, hi) = (a.as_ptr(), a.as_ptr().add(half));
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(qa as i16);
    let mut s0 = _mm256_setzero_si256();
    let mut s1 = _mm256_setzero_si256();
    let mut i = 0;
    while i < half {
        let l0 = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256(lo.add(i) as *const __m256i), zero), qa);
        let l1 = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256(lo.add(i + 16) as *const __m256i), zero), qa);
        let h0 = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256(hi.add(i) as *const __m256i), zero), qa);
        let h1 = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256(hi.add(i + 16) as *const __m256i), zero), qa);
        let w0 = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
        let w1 = _mm256_loadu_si256(w.as_ptr().add(i + 16) as *const __m256i);
        // (lo * w) fits i16 as |w| <= 128; madd then multiplies by hi and pairs up into i32.
        s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_mullo_epi16(l0, w0), h0));
        s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_mullo_epi16(l1, w1), h1));
        i += 32;
    }
    hsum(_mm256_add_epi32(s0, s1))
}
