// NNUE inference. The architecture is read from the network's header (see
// docs/net-format.md): (768 x king buckets -> H) x 2 perspectives, with
// optional horizontal mirroring, a SCReLU or CReLU activation, and a
// material-count output-bucketed output layer, or one hidden layer of
// L1_SIZE neurons (int8) before it.
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
const TAG_LAYER_QUANT: u16 = 0x0007;
const OPTIONAL: u16 = 0x8000;
const INPUT_PSQ768: u16 = 1;
const ACT_NONE: u8 = 0;
const ACT_CRELU: u8 = 2;
const ACT_SCRELU: u8 = 3;
const TYPE_I8: u8 = 1;
const TYPE_I16: u8 = 2;
const TYPE_I32: u8 = 3;
const TYPE_F32: u8 = 4;

/// Hidden-layer width the engine supports (one zmm / two ymm of i32 outputs).
pub const L1_SIZE: usize = 16;
const OUTPUT_MATERIAL: u8 = 1;

// Path chosen by build.rs (EVALFILE, or the net named in net.txt).
static NET_BYTES: &[u8] = include_bytes!(env!("INCIPIT_NET"));

pub struct Network {
    h: usize,
    nkb: usize, // king buckets
    mirror: bool,
    refresh_on_king_move: bool,
    king_bucket: [u8; 64],
    screlu: bool,
    bucket_of: [u8; 33], // output bucket by piece count
    qa: i32,
    qb: i32,
    scale: i32,
    ftw: Aligned, // [king bucket][feature][h]
    ftb: Aligned, // [h]
    ow: Aligned,  // [output bucket][2h], side to move first
    ob: Vec<i32>,  // [output bucket]
    l1: Option<Hidden>,
}

/// One hidden layer between the feature transformer and the output: the FT
/// outputs become u8 inputs (clamp(a, 0, QA)^2 >> shift), times int8 weights,
/// then a float SCReLU and a float output layer, all per output bucket.
struct Hidden {
    shift: u32,
    in_scale: f32,          // QA^2 / 2^shift: one input unit in real terms
    w_scale: f32,           // int8 weight quantisation
    w1: AlignedI8,          // [bucket][input / 4][L1_SIZE][4], the kernels' layout
    b1: Vec<[f32; L1_SIZE]>, // [bucket]
    shared: bool,           // one hidden layer for all buckets (w1, b1 have one entry)
    w2: Vec<[f32; L1_SIZE]>, // [bucket]
    b2: Vec<f32>,           // [bucket]
}

/// A 64-byte-aligned i8 buffer.
struct AlignedI8 {
    ptr: *mut i8,
    len: usize,
}

impl AlignedI8 {
    fn layout(len: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(len.max(1), 64).unwrap()
    }
    fn zeroed(len: usize) -> AlignedI8 {
        let ptr = unsafe { std::alloc::alloc_zeroed(Self::layout(len)) } as *mut i8;
        assert!(!ptr.is_null(), "out of memory loading the network");
        AlignedI8 { ptr, len }
    }
}

impl std::ops::Deref for AlignedI8 {
    type Target = [i8];
    fn deref(&self) -> &[i8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl std::ops::DerefMut for AlignedI8 {
    fn deref_mut(&mut self) -> &mut [i8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedI8 {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr as *mut u8, Self::layout(self.len)) }
    }
}

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
    let mut layer_quant: Vec<(u32, i32)> = Vec::new();
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
            TAG_LAYER_QUANT => {
                layer_quant = (0..len / 8).map(|i| (p32(8 * i), p32(8 * i + 4) as i32)).collect();
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
    if act != ACT_SCRELU && act != ACT_CRELU {
        return Err(format!("unsupported feature transformer activation {}", act));
    }
    if wt != TYPE_I16 || bt != TYPE_I16 {
        return Err("feature transformer weights and biases must be i16".into());
    }
    let (scheme, buckets) = outb.ok_or_else(|| missing("OUTPUT_BUCKETS"))?;
    if scheme != OUTPUT_MATERIAL || buckets == 0 || buckets > 32 {
        return Err(format!("unsupported output buckets (scheme {}, {} buckets)", scheme, buckets));
    }
    let (qa, qb, scale) = quant.ok_or_else(|| missing("QUANTISATION"))?;
    if layers.len() == 2 {
        return load_hidden(d, header, h, nkb, mirror, king_bucket, act, buckets, qa, qb, scale, &layers, &layer_quant);
    }
    let &[(lin, lout, lact, lwt, lbt, lflags)] = layers.as_slice() else {
        return Err(format!("{} layers after the feature transformer; one or two are supported", layers.len()));
    };
    if lin != 2 * h || lout != 1 || lact != ACT_NONE || lwt != TYPE_I16 || !(lbt == TYPE_I16 || lbt == TYPE_I32) || lflags & 1 == 0 {
        return Err("unsupported output layer (need 2H -> 1, no activation, i16 weights, per output bucket)".into());
    }

    let bias_size = if lbt == TYPE_I32 { 4 } else { 2 };
    let expect = 2 * (nkb * 768 * h + h + buckets * 2 * h) + bias_size * buckets;
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
    let ow = i16s(buckets * 2 * h);
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
        screlu: act == ACT_SCRELU,
        nkb,
        bucket_of: std::array::from_fn(|pieces| (pieces.saturating_sub(2) / 32usize.div_ceil(buckets)).min(buckets - 1) as u8),
        qa,
        qb,
        scale,
        ftw,
        ftb,
        ow,
        ob,
        l1: None,
    })
}

/// Loads a network with one hidden layer: FT -> L1_SIZE (SCReLU, i8 weights,
/// f32 biases) -> 1 (f32), both per output bucket.
#[allow(clippy::too_many_arguments)]
fn load_hidden(
    d: &[u8],
    header: usize,
    h: usize,
    nkb: usize,
    mirror: bool,
    king_bucket: [u8; 64],
    act: u8,
    buckets: usize,
    qa: i32,
    qb: i32,
    scale: i32,
    layers: &[(usize, usize, u8, u8, u8, u8)],
    layer_quant: &[(u32, i32)],
) -> Result<Network, String> {
    let (l1, l2) = (layers[0], layers[1]);
    if act != ACT_SCRELU {
        return Err("a hidden layer needs a SCReLU feature transformer".into());
    }
    let shared = l1.5 & 1 == 0;
    if (l1.0, l1.1, l1.2, l1.3, l1.4) != (2 * h, L1_SIZE, ACT_SCRELU, TYPE_I8, TYPE_F32) {
        return Err(format!("unsupported hidden layer {:?} (need 2H -> {} SCReLU, i8 weights, f32 biases)", l1, L1_SIZE));
    }
    let nb1 = if shared { 1 } else { buckets };
    if l2 != (L1_SIZE, 1, ACT_NONE, TYPE_F32, TYPE_F32, 1) {
        return Err(format!("unsupported final layer {:?} (need {} -> 1, f32, per bucket)", l2, L1_SIZE));
    }
    let &[(shift, w_scale), _] = layer_quant else {
        return Err("a hidden layer needs a LAYER_QUANT field for both layers".into());
    };
    if shift > 30 || w_scale <= 0 || ((qa as i64 * qa as i64) >> shift) > 127 {
        return Err(format!("hidden layer quantisation (shift {}, weight scale {}) doesn't fit u8 0..127 inputs", shift, w_scale));
    }
    let n_ftw = nkb * 768 * h;
    let expect = 2 * (n_ftw + h) + nb1 * L1_SIZE * 2 * h + 4 * (nb1 * L1_SIZE + buckets * (L1_SIZE + 1));
    if d.len() - header != expect {
        return Err(format!("weights are {} bytes, but the header describes {}", d.len() - header, expect));
    }
    let mut o = header;
    let mut i16s = |n: usize| {
        let v = Aligned::from(d[o..o + 2 * n].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])));
        o += 2 * n;
        v
    };
    let ftw = i16s(n_ftw);
    let ftb = i16s(h);
    // File: [bucket][out][in]. Kernels want each group of 4 inputs for all
    // outputs together: [bucket][in / 4][out][in % 4].
    let mut w1 = AlignedI8::zeroed(nb1 * L1_SIZE * 2 * h);
    for b in 0..nb1 {
        for n in 0..L1_SIZE {
            for i in 0..2 * h {
                let v = d[o + (b * L1_SIZE + n) * 2 * h + i] as i8;
                w1[b * L1_SIZE * 2 * h + (i / 4) * L1_SIZE * 4 + n * 4 + i % 4] = v;
            }
        }
    }
    o += nb1 * L1_SIZE * 2 * h;
    let mut f32s = |n: usize| {
        let v: Vec<f32> = d[o..o + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        o += 4 * n;
        v
    };
    let rows = |v: Vec<f32>| v.chunks_exact(L1_SIZE).map(|c| <[f32; L1_SIZE]>::try_from(c).unwrap()).collect::<Vec<_>>();
    let b1 = rows(f32s(nb1 * L1_SIZE));
    let w2 = rows(f32s(buckets * L1_SIZE));
    let b2 = f32s(buckets);
    Ok(Network {
        h,
        mirror,
        refresh_on_king_move: nkb > 1 || mirror,
        king_bucket,
        screlu: true,
        nkb,
        bucket_of: std::array::from_fn(|pieces| (pieces.saturating_sub(2) / 32usize.div_ceil(buckets)).min(buckets - 1) as u8),
        qa,
        qb,
        scale,
        ftw,
        ftb,
        ow: Aligned::from(std::iter::empty()),
        ob: Vec::new(),
        l1: Some(Hidden {
            shift,
            in_scale: (qa as f32 * qa as f32) / (1u64 << shift) as f32,
            w_scale: w_scale as f32,
            w1,
            b1,
            shared,
            w2,
            b2,
        }),
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
    /// A king move that changes the king's bucket or mirror state rebuilds
    /// that perspective via `cache`.
    #[inline]
    pub fn update_from(&mut self, parent: &Acc, pos: &Position, child: &Position, m: Move, cache: &mut RefreshCache) {
        let n = net();
        with_h!(n.h, update(self, n, parent, pos, child, m, cache));
    }
}

/// Accumulator refresh cache: for each (perspective, king bucket, mirror
/// state), an accumulator and the piece bitboards it currently represents.
/// Rebuilding a perspective after its king changes bucket then only adds and
/// subtracts the pieces that differ from that entry, instead of every piece.
pub struct RefreshCache {
    entries: Vec<CacheEntry>,
}

#[repr(C, align(64))]
struct CacheEntry {
    acc: [i16; MAX_H],
    bb: [u64; 12], // by piece code
}

impl RefreshCache {
    /// A cache for the loaded network: every entry is the bias with no pieces.
    pub fn new() -> Self {
        let n = net();
        let entries = (0..2 * n.nkb * 2)
            .map(|_| {
                let mut e = CacheEntry { acc: [0; MAX_H], bb: [0; 12] };
                e.acc[..n.h].copy_from_slice(&n.ftb);
                e
            })
            .collect();
        RefreshCache { entries }
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

/// Rebuilds perspective p of `acc` for `pos` through the cache entry for p's
/// king bucket and mirror state.
#[inline(never)]
fn refresh_cached<const H: usize>(acc: &mut Acc, n: &Network, p: usize, pos: &Position, cache: &mut RefreshCache) {
    let h = hidden::<H>(n);
    let ki = kinfo(n, p, pos.king_sq(p));
    let idx = (p * n.nkb + ki.0 / 768) * 2 + (ki.1 != 0) as usize;
    let e = &mut cache.entries[idx];
    let v = &mut e.acc[..h];
    for c in 0..2 {
        for pt in 0..6 {
            let pc = make_pc(pt, c);
            let cur = pos.pieces[pt] & pos.colors[c];
            let old = e.bb[pc as usize];
            for sq in Bits(cur & !old) {
                for (x, &w) in v.iter_mut().zip(row(n, feat(p, ki, pc, sq), h)) {
                    *x = x.wrapping_add(w);
                }
            }
            for sq in Bits(old & !cur) {
                for (x, &w) in v.iter_mut().zip(row(n, feat(p, ki, pc, sq), h)) {
                    *x = x.wrapping_sub(w);
                }
            }
            e.bb[pc as usize] = cur;
        }
    }
    acc.side_mut(p, h).copy_from_slice(v);
}

#[inline(never)]
fn update<const H: usize>(acc: &mut Acc, n: &Network, parent: &Acc, pos: &Position, child: &Position, m: Move, cache: &mut RefreshCache) {
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
    if is_castle(m) {
        // Checked first: in Chess960 the king's destination may hold the rook.
        let q = flag == F_QCASTLE;
        subs[1] = (make_pc(ROOK, us), pos.rook_sq[castle_right(us, q)] as usize);
        adds[1] = (make_pc(ROOK, us), if q { to + 1 } else { to - 1 });
        ns = 2;
        na = 2;
    } else if flag == F_EP {
        subs[1] = (make_pc(PAWN, us ^ 1), to ^ 8);
        ns = 2;
    } else if pos.board[to] != NONE_PC {
        subs[1] = (pos.board[to], to);
        ns = 2;
    }
    for p in 0..2 {
        let ki = kinfo(n, p, pos.king_sq(p));
        if n.refresh_on_king_move && p == us && pc_type(pc) == KING && kinfo(n, p, to) != ki {
            refresh_cached::<H>(acc, n, p, child, cache);
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
    match &n.l1 {
        None => with_h!(n.h, eval_n(n, acc, pos)),
        Some(l1) => with_h!(n.h, eval_hidden(n, l1, acc, pos)),
    }
}

/// Network with one hidden layer: u8 inputs from both perspectives (side to
/// move first), an int8 matrix product per output bucket, then float SCReLU
/// and the float output layer.
#[inline(never)]
fn eval_hidden<const H: usize>(n: &Network, l1: &Hidden, acc: &Acc, pos: &Position) -> i32 {
    let h = hidden::<H>(n);
    let bucket = n.bucket_of[pos.occ().count_ones() as usize] as usize;
    #[repr(C, align(64))]
    struct Inputs([u8; 2 * MAX_H]);
    // Left uninitialised: to_u8 writes the 2h bytes used (zeroing 4 KB per
    // eval showed up in profiles).
    #[allow(invalid_value, clippy::uninit_assumed_init)]
    let mut x: Inputs = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    to_u8(acc.side(pos.stm, h), &mut x.0[..h], n.qa, l1.shift);
    to_u8(acc.side(pos.stm ^ 1, h), &mut x.0[h..2 * h], n.qa, l1.shift);
    // A shared hidden layer has one block for all buckets (branch-free).
    let b1i = bucket & !(l1.shared as usize).wrapping_neg();
    let w = &l1.w1[b1i * L1_SIZE * 2 * h..(b1i + 1) * L1_SIZE * 2 * h];
    let z = l1_matmul(&x.0[..2 * h], w);
    let k = 1.0 / (l1.in_scale * l1.w_scale);
    let (b1, w2) = (&l1.b1[b1i], &l1.w2[bucket]);
    let mut out = l1.b2[bucket];
    for i in 0..L1_SIZE {
        let v = (z[i] as f32 * k + b1[i]).clamp(0.0, 1.0);
        out += v * v * w2[i];
    }
    (out * n.scale as f32) as i32
}

/// clamp(a, 0, qa)^2 >> shift as u8 (load() checks it fits 0..127). The
/// usual QA 255 / shift 9 has SIMD versions; a.len() is a multiple of 32.
#[inline(always)]
fn to_u8(a: &[i16], x: &mut [u8], qa: i32, shift: u32) {
    if qa == 255 && shift == 9 {
        unsafe { to_u8_255_9(a, x) }
    } else {
        to_u8_scalar(a, x, qa, shift)
    }
}

fn to_u8_scalar(a: &[i16], x: &mut [u8], qa: i32, shift: u32) {
    for (o, &v) in x.iter_mut().zip(a) {
        let c = (v as i32).clamp(0, qa);
        *o = ((c * c) >> shift) as u8;
    }
}

/// QA 255, shift 9: (c * c) >> 9 = mulhi_u16(c << 7, c) (c << 7 fits u16).
#[cfg(all(avx512_intrinsics, target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn to_u8_255_9(a: &[i16], x: &mut [u8]) {
    use std::arch::x86_64::*;
    let zero = _mm512_setzero_si512();
    let qa = _mm512_set1_epi16(255);
    // packus works within 128-bit lanes; this puts the 64-bit pieces back in order.
    let order = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    for i in (0..a.len()).step_by(64) {
        let sq = |o: usize| {
            let c = _mm512_min_epi16(_mm512_max_epi16(_mm512_loadu_si512(a.as_ptr().add(o) as *const __m512i), zero), qa);
            _mm512_mulhi_epu16(_mm512_slli_epi16(c, 7), c)
        };
        let (lo, hi) = (sq(i), if i + 32 < a.len() { sq(i + 32) } else { zero });
        let p = _mm512_permutexvar_epi64(order, _mm512_packus_epi16(lo, hi));
        if i + 64 <= a.len() {
            _mm512_storeu_si512(x.as_mut_ptr().add(i) as *mut __m512i, p);
        } else {
            _mm256_storeu_si256(x.as_mut_ptr().add(i) as *mut __m256i, _mm512_castsi512_si256(p));
        }
    }
}

#[cfg(not(all(avx512_intrinsics, target_feature = "avx512bw")))]
#[inline(always)]
unsafe fn to_u8_255_9(a: &[i16], x: &mut [u8]) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(255);
    for i in (0..a.len()).step_by(32) {
        let sq = |o: usize| {
            let c = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(a.as_ptr().add(o) as *const __m256i), zero), qa);
            _mm256_mulhi_epu16(_mm256_slli_epi16(c, 7), c)
        };
        // packus interleaves the 128-bit lanes; permute4x64 restores the order.
        let p = _mm256_permute4x64_epi64(_mm256_packus_epi16(sq(i), sq(i + 16)), 0b11_01_10_00);
        _mm256_storeu_si256(x.as_mut_ptr().add(i) as *mut __m256i, p);
    }
}

/// For each 8-bit mask, the positions of its set bits (unused slots 0): lets
/// the kernels list non-zero input groups without a branch per group.
static NZ_TABLE: [[u16; 8]; 256] = {
    let mut t = [[0u16; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let (mut k, mut b) = (0, 0);
        while b < 8 {
            if m & (1 << b) != 0 {
                t[m][k] = b as u16;
                k += 1;
            }
            b += 1;
        }
        m += 1;
    }
    t
};

/// Appends the indices (base + bit) of the set bits of the 8-bit mask m to
/// nz at count, with one 16-byte store; returns the new count. nz needs 8
/// slots of slack past the last real entry.
#[inline(always)]
unsafe fn push_nz(nz: &mut [u16], count: usize, m: u32, base: u16) -> usize {
    use std::arch::x86_64::*;
    let idx = _mm_add_epi16(_mm_loadu_si128(NZ_TABLE.get_unchecked(m as usize).as_ptr() as *const __m128i), _mm_set1_epi16(base as i16));
    _mm_storeu_si128(nz.as_mut_ptr().add(count) as *mut __m128i, idx);
    count + m.count_ones() as usize
}

/// z[n] = sum_i x[i] * w[n][i] for the L1_SIZE outputs, with w in the
/// [in / 4][out][in % 4] layout. x.len() is a multiple of 64.
#[inline(always)]
fn l1_matmul(x: &[u8], w: &[i8]) -> [i32; L1_SIZE] {
    unsafe { l1_matmul_simd(x, w) }
}

/// AVX-512: one register holds all 16 outputs; each group of 4 inputs is a
/// broadcast and one VNNI dpbusd (or maddubs + madd without VNNI). Inputs are
/// at most 127 and weights within +-127, so maddubs's i16 sums can't overflow.
#[cfg(all(avx512_intrinsics, target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn l1_matmul_simd(x: &[u8], w: &[i8]) -> [i32; L1_SIZE] {
    use std::arch::x86_64::*;
    let (xp, wp) = (x.as_ptr() as *const i32, w.as_ptr());
    let mut s0 = _mm512_setzero_si512();
    let mut s1 = _mm512_setzero_si512();
    #[inline(always)]
    unsafe fn step(s: __m512i, xg: i32, wg: *const i8) -> __m512i {
        let xb = _mm512_set1_epi32(xg);
        let wv = _mm512_load_si512(wg as *const __m512i);
        #[cfg(target_feature = "avx512vnni")]
        {
            _mm512_dpbusd_epi32(s, xb, wv)
        }
        #[cfg(not(target_feature = "avx512vnni"))]
        {
            _mm512_add_epi32(s, _mm512_madd_epi16(_mm512_maddubs_epi16(xb, wv), _mm512_set1_epi16(1)))
        }
    }
    // SCReLU leaves most inputs at zero: list the groups of 4 with any
    // non-zero input and skip the rest (their weights aren't even loaded).
    #[allow(invalid_value, clippy::uninit_assumed_init)]
    let mut nz: [u16; 2 * MAX_H / 4 + 8] = std::mem::MaybeUninit::uninit().assume_init();
    let mut count = 0;
    for c in (0..x.len()).step_by(64) {
        let v = _mm512_loadu_si512(x.as_ptr().add(c) as *const __m512i);
        let m = _mm512_test_epi32_mask(v, v) as u32;
        count = push_nz(&mut nz, count, m & 0xff, (c / 4) as u16);
        count = push_nz(&mut nz, count, m >> 8, (c / 4) as u16 + 8);
    }
    // Four accumulators so consecutive dpbusds don't wait on each other.
    let mut s2 = _mm512_setzero_si512();
    let mut s3 = _mm512_setzero_si512();
    let g = |i: usize| *nz.get_unchecked(i) as usize;
    let mut i = 0;
    while i + 3 < count {
        let (g0, g1, g2, g3) = (g(i), g(i + 1), g(i + 2), g(i + 3));
        s0 = step(s0, xp.add(g0).read_unaligned(), wp.add(g0 * 64));
        s1 = step(s1, xp.add(g1).read_unaligned(), wp.add(g1 * 64));
        s2 = step(s2, xp.add(g2).read_unaligned(), wp.add(g2 * 64));
        s3 = step(s3, xp.add(g3).read_unaligned(), wp.add(g3 * 64));
        i += 4;
    }
    while i < count {
        let g0 = g(i);
        s0 = step(s0, xp.add(g0).read_unaligned(), wp.add(g0 * 64));
        i += 1;
    }
    let mut z = [0i32; L1_SIZE];
    _mm512_storeu_si512(z.as_mut_ptr() as *mut __m512i, _mm512_add_epi32(_mm512_add_epi32(s0, s1), _mm512_add_epi32(s2, s3)));
    z
}

/// AVX2: two registers of 8 outputs; per group of 4 inputs, maddubs + madd.
#[cfg(not(all(avx512_intrinsics, target_feature = "avx512bw")))]
#[inline(always)]
unsafe fn l1_matmul_simd(x: &[u8], w: &[i8]) -> [i32; L1_SIZE] {
    use std::arch::x86_64::*;
    let (xp, wp) = (x.as_ptr() as *const i32, w.as_ptr());
    let ones = _mm256_set1_epi16(1);
    let mut a = [_mm256_setzero_si256(); 4];
    // Skip groups of 4 inputs that are all zero (see the AVX-512 version).
    #[allow(invalid_value, clippy::uninit_assumed_init)]
    let mut nz: [u16; 2 * MAX_H / 4 + 8] = std::mem::MaybeUninit::uninit().assume_init();
    let mut count = 0;
    let zero = _mm256_setzero_si256();
    for c in (0..x.len()).step_by(32) {
        let v = _mm256_loadu_si256(x.as_ptr().add(c) as *const __m256i);
        let m = !(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, zero))) as u32) & 0xff;
        count = push_nz(&mut nz, count, m, (c / 4) as u16);
    }
    let mut step = |k: usize, g: usize| {
        let xb = _mm256_set1_epi32(xp.add(g).read_unaligned());
        let w0 = _mm256_load_si256(wp.add(g * 64) as *const __m256i);
        let w1 = _mm256_load_si256(wp.add(g * 64 + 32) as *const __m256i);
        a[2 * k] = _mm256_add_epi32(a[2 * k], _mm256_madd_epi16(_mm256_maddubs_epi16(xb, w0), ones));
        a[2 * k + 1] = _mm256_add_epi32(a[2 * k + 1], _mm256_madd_epi16(_mm256_maddubs_epi16(xb, w1), ones));
    };
    let mut i = 0;
    while i + 1 < count {
        step(0, *nz.get_unchecked(i) as usize);
        step(1, *nz.get_unchecked(i + 1) as usize);
        i += 2;
    }
    if i < count {
        step(0, *nz.get_unchecked(i) as usize);
    }
    let mut z = [0i32; L1_SIZE];
    _mm256_storeu_si256(z.as_mut_ptr() as *mut __m256i, _mm256_add_epi32(a[0], a[2]));
    _mm256_storeu_si256(z.as_mut_ptr().add(8) as *mut __m256i, _mm256_add_epi32(a[1], a[3]));
    z
}

/// Scalar reference for l1_matmul (used by `l1check`).
pub fn l1_matmul_scalar(x: &[u8], w: &[i8]) -> [i32; L1_SIZE] {
    let mut z = [0i32; L1_SIZE];
    for (g, xs) in x.chunks_exact(4).enumerate() {
        let wg = &w[g * L1_SIZE * 4..(g + 1) * L1_SIZE * 4];
        for n in 0..L1_SIZE {
            for j in 0..4 {
                z[n] += xs[j] as i32 * wg[n * 4 + j] as i32;
            }
        }
    }
    z
}

#[inline(never)]
fn eval_n<const H: usize>(n: &Network, acc: &Acc, pos: &Position) -> i32 {
    let h = hidden::<H>(n);
    let bucket = n.bucket_of[pos.occ().count_ones() as usize] as usize;
    let us = acc.side(pos.stm, h);
    let them = acc.side(pos.stm ^ 1, h);
    let w = &n.ow[bucket * 2 * h..(bucket + 1) * 2 * h];
    // The common quantisation gets constant divisors (cheap multiplies instead
    // of hardware divides); the results are identical.
    let (qa, qb) = if n.qa == 255 && n.qb == 64 { (255, 64) } else { (n.qa, n.qb) };
    if n.screlu {
        let sum = unsafe { dot::<true>(us, &w[..h], qa) + dot::<true>(them, &w[h..], qa) };
        if qa == 255 && qb == 64 {
            (sum / 255 + n.ob[bucket]) * n.scale / (255 * 64)
        } else {
            (sum / qa + n.ob[bucket]) * n.scale / (qa * qb)
        }
    } else {
        let sum = unsafe { dot::<false>(us, &w[..h], qa) + dot::<false>(them, &w[h..], qa) };
        (sum + n.ob[bucket]) * n.scale / (qa * qb)
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

/// Checks the SIMD hidden-layer kernels against the scalar versions on random
/// accumulators and weights. Returns the number of mismatches.
pub fn l1check(trials: usize) -> usize {
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut bad = 0;
    for &h in &[128usize, 256, 512, 768, 1024, 2048] {
        for _ in 0..trials {
            let a: Vec<i16> = (0..2 * h).map(|_| (rnd() % 700) as i16 - 200).collect();
            let mut x1 = vec![0u8; 2 * h];
            let mut x2 = vec![0u8; 2 * h];
            to_u8(&a, &mut x1, 255, 9);
            to_u8_scalar(&a, &mut x2, 255, 9);
            if x1 != x2 {
                bad += 1;
                continue;
            }
            let mut w = AlignedI8::zeroed(L1_SIZE * 2 * h);
            for v in w.iter_mut() {
                *v = ((rnd() % 255) as i32 - 127) as i8;
            }
            if l1_matmul(&x1, &w) != l1_matmul_scalar(&x1, &w) {
                bad += 1;
            }
        }
    }
    bad
}
