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
/// Pairwise CReLU: clamp to [0, QA], then each perspective's first half times
/// its second half (h/2 values per perspective; same QA^2 scale as SCReLU).
const ACT_PAIRWISE: u8 = 4;
const TYPE_I8: u8 = 1;
const TYPE_I16: u8 = 2;
const TYPE_I32: u8 = 3;
const TYPE_F32: u8 = 4;

/// Hidden-layer width the engine supports (one zmm / two ymm of i32 outputs).
pub const L1_SIZE: usize = 16;
/// Most neurons in the optional second hidden layer.
pub const L2_MAX: usize = 32;
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
    ob: Vec<i32>, // [output bucket]
    l1: Option<Hidden>,
}

/// One hidden layer between the feature transformer and the output: the FT
/// outputs become u8 inputs (clamp(a, 0, QA)^2 >> shift), times int8 weights,
/// then a float SCReLU and a float output layer, all per output bucket.
struct Hidden {
    n: usize, // neurons: 8 or 16 (b1 and w2 rows padded to L1_SIZE with zeros)
    shift: u32,
    in_scale: f32,           // QA^2 / 2^shift: one input unit in real terms
    w_scale: f32,            // int8 weight quantisation
    w1: AlignedI8,           // [bucket][input / 4][n][4], the kernels' layout
    b1: Vec<[f32; L1_SIZE]>, // [bucket]
    shared: bool,            // one hidden layer for all buckets (w1, b1 have one entry)
    pw: bool,                // pairwise FT: the hidden layer has h inputs, not 2h
    w2: Vec<[f32; L1_SIZE]>, // output layer [bucket] (no second hidden layer)
    b2: Vec<f32>,            // output bias [bucket]
    // Optional second hidden layer (f32, SCReLU, per bucket): n2 neurons
    // (0 = none). wm is [bucket][input][n2] (input-major, so the product
    // vectorises over the outputs), bm [bucket][n2]; the output layer is then
    // wo [bucket][n2] with bias b2.
    n2: usize,
    wm: Vec<[[f32; L2_MAX]; 2 * L1_SIZE]>, // [bucket][input][l2]; 2L inputs when dual
    dual: bool,                            // the second hidden layer sees SCReLU and CReLU of the first (2L inputs)
    bm: Vec<[f32; L2_MAX]>,
    wo: Vec<[f32; L2_MAX]>,
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

/// Loads the embedded network once (repeat calls, e.g. from parallel tests,
/// do nothing).
pub fn init() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(init_net);
}

fn init_net() {
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
        return Err(
            "not an Incipit network (bad magic); old headerless nets must be converted with `datatools net raw`".into(),
        );
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
                inputs = Some(
                    (0..n)
                        .map(|i| (u16::from_le_bytes([p[2 + 8 * i], p[3 + 8 * i]]), p32(6 + 8 * i)))
                        .collect::<Vec<_>>(),
                );
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
    // Pairwise is only supported with a hidden layer (load_hidden checks).
    if act != ACT_SCRELU && act != ACT_CRELU && act != ACT_PAIRWISE {
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
    // The kernels clamp to qa as an i16, and the output divides by qa and qb.
    if !(1..=i16::MAX as i32).contains(&qa) || !(1..=i16::MAX as i32).contains(&qb) || scale <= 0 {
        return Err(format!("unsupported quantisation (QA {}, QB {}, scale {})", qa, qb, scale));
    }
    if act == ACT_PAIRWISE && !(layers.len() == 2 || layers.len() == 3) {
        return Err("a pairwise feature transformer needs a hidden layer".into());
    }
    if layers.len() == 2 || layers.len() == 3 {
        return load_hidden(d, header, h, nkb, mirror, king_bucket, act, buckets, qa, qb, scale, &layers, &layer_quant);
    }
    let &[(lin, lout, lact, lwt, lbt, lflags)] = layers.as_slice() else {
        return Err(format!("{} layers after the feature transformer; one to three are supported", layers.len()));
    };
    if lin != 2 * h
        || lout != 1
        || lact != ACT_NONE
        || lwt != TYPE_I16
        || !(lbt == TYPE_I16 || lbt == TYPE_I32)
        || lflags & 1 == 0
    {
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
        bucket_of: std::array::from_fn(|pieces| {
            (pieces.saturating_sub(2) / 32usize.div_ceil(buckets)).min(buckets - 1) as u8
        }),
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
    let (l1, l2) = (layers[0], layers[layers.len() - 1]);
    // A middle layer: the second hidden layer.
    let mid = (layers.len() == 3).then(|| layers[1]);
    if act != ACT_SCRELU && act != ACT_PAIRWISE {
        return Err("a hidden layer needs a SCReLU or pairwise feature transformer".into());
    }
    let pw = act == ACT_PAIRWISE;
    if pw && h % 128 != 0 {
        return Err(format!("pairwise needs a hidden size that is a multiple of 128, got {}", h));
    }
    // Hidden-layer inputs: both perspectives, h each (SCReLU) or h/2 each (pairwise).
    let inl = if pw { h } else { 2 * h };
    let shared = l1.5 & 1 == 0;
    let ln = l1.1;
    if (l1.0, l1.2, l1.3, l1.4) != (inl, ACT_SCRELU, TYPE_I8, TYPE_F32) || (ln != 8 && ln != 16) {
        return Err(format!(
            "unsupported hidden layer {:?} (need {} -> 8 or 16 SCReLU, i8 weights, f32 biases)",
            l1, inl
        ));
    }
    let nb1 = if shared { 1 } else { buckets };
    let n2 = mid.map_or(0, |m| m.1);
    if let Some(m) = mid {
        if (m.0 != ln && m.0 != 2 * ln)
            || (m.1, m.2, m.3, m.4, m.5) != (n2, ACT_SCRELU, TYPE_F32, TYPE_F32, 1)
            || n2 == 0
            || n2 > L2_MAX
        {
            return Err(format!(
                "unsupported second hidden layer {:?} (need {} -> 1..{} SCReLU, f32, per bucket)",
                m, ln, L2_MAX
            ));
        }
    }
    if pw && ln != 16 {
        return Err(format!("pairwise hidden nets need a 16-neuron first hidden layer, got {}", ln));
    }
    // The two-hidden-layer kernels exist for 16 -> 16 and 16 -> 32 only
    // (evaluate dispatches on these).
    if n2 > 0 && (ln != 16 || (n2 != 16 && n2 != 32)) {
        return Err(format!("a second hidden layer needs 16 -> 16 or 16 -> 32 neurons, got {} -> {}", ln, n2));
    }
    // Dual activation: the second hidden layer takes SCReLU then CReLU of the
    // first layer's outputs (2L inputs).
    let dual = mid.is_some_and(|m| m.0 == 2 * ln);
    let in2 = if dual { 2 * ln } else { ln };
    let last_in = if n2 > 0 { n2 } else { ln };
    if l2 != (last_in, 1, ACT_NONE, TYPE_F32, TYPE_F32, 1) {
        return Err(format!("unsupported final layer {:?} (need {} -> 1, f32, per bucket)", l2, last_in));
    }
    let &[(shift, w_scale), ..] = layer_quant else {
        return Err("a hidden layer needs a LAYER_QUANT field".into());
    };
    if shift > 30 || w_scale <= 0 || ((qa as i64 * qa as i64) >> shift) > 127 {
        return Err(format!(
            "hidden layer quantisation (shift {}, weight scale {}) doesn't fit u8 0..127 inputs",
            shift, w_scale
        ));
    }
    let n_ftw = nkb * 768 * h;
    let expect = 2 * (n_ftw + h) + nb1 * ln * inl + 4 * (nb1 * ln + buckets * n2 * (in2 + 1) + buckets * (last_in + 1));
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
    let mut w1 = AlignedI8::zeroed(nb1 * ln * inl);
    for b in 0..nb1 {
        for n in 0..ln {
            for i in 0..inl {
                let v = d[o + (b * ln + n) * inl + i] as i8;
                w1[b * ln * inl + (i / 4) * ln * 4 + n * 4 + i % 4] = v;
            }
        }
    }
    o += nb1 * ln * inl;
    let mut f32s = |n: usize| {
        let v: Vec<f32> =
            d[o..o + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        o += 4 * n;
        v
    };
    let rows = |v: Vec<f32>| {
        v.chunks_exact(ln)
            .map(|c| {
                let mut r = [0f32; L1_SIZE];
                r[..ln].copy_from_slice(c);
                r
            })
            .collect::<Vec<_>>()
    };
    let b1 = rows(f32s(nb1 * ln));
    let (mut wm, mut bm, mut wo) = (Vec::new(), Vec::new(), Vec::new());
    let w2;
    if n2 > 0 {
        let wmf = f32s(buckets * n2 * in2);
        let bmf = f32s(buckets * n2);
        let wof = f32s(buckets * n2);
        for b in 0..buckets {
            let mut m = [[0f32; L2_MAX]; 2 * L1_SIZE];
            let (mut mb, mut ow) = ([0f32; L2_MAX], [0f32; L2_MAX]);
            for j in 0..n2 {
                for i in 0..in2 {
                    m[i][j] = wmf[(b * n2 + j) * in2 + i];
                }
                mb[j] = bmf[b * n2 + j];
                ow[j] = wof[b * n2 + j];
            }
            wm.push(m);
            bm.push(mb);
            wo.push(ow);
        }
        w2 = vec![[0f32; L1_SIZE]; buckets];
    } else {
        w2 = rows(f32s(buckets * ln));
    }
    let b2 = f32s(buckets);
    Ok(Network {
        h,
        mirror,
        refresh_on_king_move: nkb > 1 || mirror,
        king_bucket,
        screlu: true,
        nkb,
        bucket_of: std::array::from_fn(|pieces| {
            (pieces.saturating_sub(2) / 32usize.div_ceil(buckets)).min(buckets - 1) as u8
        }),
        qa,
        qb,
        scale,
        ftw,
        ftb,
        ow: Aligned::from(std::iter::empty()),
        ob: Vec::new(),
        l1: Some(Hidden {
            n: ln,
            shift,
            in_scale: (qa as f32 * qa as f32) / (1u64 << shift) as f32,
            w_scale: w_scale as f32,
            w1,
            b1,
            shared,
            pw,
            w2,
            b2,
            n2,
            wm,
            dual,
            bm,
            wo,
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
    if H > 0 {
        H
    } else {
        n.h
    }
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
/// king bucket and mirror state: the entry's rows plus the pieces added and
/// minus those removed since it was last used, written back to the entry and
/// to acc.
#[inline(never)]
fn refresh_cached<const H: usize>(acc: &mut Acc, n: &Network, p: usize, pos: &Position, cache: &mut RefreshCache) {
    let h = hidden::<H>(n);
    let ki = kinfo(n, p, pos.king_sq(p));
    let idx = (p * n.nkb + ki.0 / 768) * 2 + (ki.1 != 0) as usize;
    let e = &mut cache.entries[idx];
    // Feature rows to add and subtract (at most 32 pieces each way).
    let mut adds = [0usize; 32];
    let mut subs = [0usize; 32];
    let (mut na, mut ns) = (0, 0);
    for c in 0..2 {
        for pt in 0..6 {
            let pc = make_pc(pt, c);
            let cur = pos.pieces[pt] & pos.colors[c];
            let old = e.bb[pc as usize];
            for sq in Bits(cur & !old) {
                adds[na] = feat(p, ki, pc, sq);
                na += 1;
            }
            for sq in Bits(old & !cur) {
                subs[ns] = feat(p, ki, pc, sq);
                ns += 1;
            }
            e.bb[pc as usize] = cur;
        }
    }
    apply_rows(&mut e.acc[..h], acc.side_mut(p, h), n, h, &adds[..na], &subs[..ns]);
}

/// v += the rows `adds` - the rows `subs`, then out = v. One pass per row.
/// (Wrapping i16 arithmetic: the order of the rows doesn't matter.)
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn apply_rows(v: &mut [i16], out: &mut [i16], n: &Network, h: usize, adds: &[usize], subs: &[usize]) {
    for &f in adds {
        for (x, &w) in v.iter_mut().zip(row(n, f, h)) {
            *x = x.wrapping_add(w);
        }
    }
    for &f in subs {
        for (x, &w) in v.iter_mut().zip(row(n, f, h)) {
            *x = x.wrapping_sub(w);
        }
    }
    out.copy_from_slice(v);
}

/// NEON version of apply_rows: one pass over v, 128 values at a time held in
/// sixteen registers while every row is applied, so v is read and written
/// once instead of once per row. h is a multiple of 32; a 32-value tail
/// step handles sizes that aren't multiples of 128.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn apply_rows(v: &mut [i16], out: &mut [i16], n: &Network, h: usize, adds: &[usize], subs: &[usize]) {
    #[inline(always)]
    unsafe fn chunk<const K: usize>(
        v: *mut i16,
        out: *mut i16,
        w: *const i16,
        h: usize,
        adds: &[usize],
        subs: &[usize],
    ) {
        use std::arch::aarch64::*;
        let mut x = [vdupq_n_s16(0); K];
        for k in 0..K {
            x[k] = vld1q_s16(v.add(8 * k));
        }
        for &f in adds {
            let r = w.add(f * h);
            for k in 0..K {
                x[k] = vaddq_s16(x[k], vld1q_s16(r.add(8 * k)));
            }
        }
        for &f in subs {
            let r = w.add(f * h);
            for k in 0..K {
                x[k] = vsubq_s16(x[k], vld1q_s16(r.add(8 * k)));
            }
        }
        for k in 0..K {
            vst1q_s16(v.add(8 * k), x[k]);
            vst1q_s16(out.add(8 * k), x[k]);
        }
    }
    // Safe: v and out hold h values, and load() checks every feature row
    // (f * h .. f * h + h) is inside ftw.
    unsafe {
        let w = n.ftw.as_ptr();
        let mut o = 0;
        while o + 128 <= h {
            chunk::<16>(v.as_mut_ptr().add(o), out.as_mut_ptr().add(o), w.add(o), h, adds, subs);
            o += 128;
        }
        while o < h {
            chunk::<4>(v.as_mut_ptr().add(o), out.as_mut_ptr().add(o), w.add(o), h, adds, subs);
            o += 32;
        }
    }
}

#[inline(never)]
fn update<const H: usize>(
    acc: &mut Acc,
    n: &Network,
    parent: &Acc,
    pos: &Position,
    child: &Position,
    m: Move,
    cache: &mut RefreshCache,
) {
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

/// Working memory for the hidden-layer evals, owned by the caller and reused:
/// zeroing it per eval showed up in profiles, and leaving it uninitialised
/// isn't sound.
#[repr(C, align(64))]
pub struct Scratch {
    /// Hidden-layer inputs; to_u8 writes the 2h bytes used.
    x: Inputs,
    /// Indices of the 4-input groups with any non-zero input (+ 8 of slack).
    nz: [u16; 2 * MAX_H / 4 + 8],
}

#[repr(C, align(64))]
struct Inputs([u8; 2 * MAX_H]);

impl Scratch {
    pub fn new() -> Box<Scratch> {
        Box::new(Scratch { x: Inputs([0; 2 * MAX_H]), nz: [0; 2 * MAX_H / 4 + 8] })
    }
}

#[inline]
pub fn evaluate(acc: &Acc, pos: &Position, s: &mut Scratch) -> i32 {
    let n = net();
    match &n.l1 {
        None => with_h!(n.h, eval_n(n, acc, pos)),
        Some(l1) if l1.pw && l1.dual && l1.n2 == 32 => with_h!(n.h, eval_hidden_pw16x32d(n, l1, acc, pos, s)),
        Some(l1) if l1.pw && l1.dual => with_h!(n.h, eval_hidden_pw16x16d(n, l1, acc, pos, s)),
        Some(l1) if l1.pw && l1.n2 == 32 => with_h!(n.h, eval_hidden_pw16x32(n, l1, acc, pos, s)),
        Some(l1) if l1.pw && l1.n2 == 16 => with_h!(n.h, eval_hidden_pw16x16(n, l1, acc, pos, s)),
        Some(l1) if l1.pw => with_h!(n.h, eval_hidden_pw16(n, l1, acc, pos, s)),
        Some(l1) if l1.dual && l1.n2 == 32 => with_h!(n.h, eval_hidden16x32d(n, l1, acc, pos, s)),
        Some(l1) if l1.dual => with_h!(n.h, eval_hidden16x16d(n, l1, acc, pos, s)),
        Some(l1) if l1.n2 == 32 => with_h!(n.h, eval_hidden16x32(n, l1, acc, pos, s)),
        Some(l1) if l1.n2 == 16 => with_h!(n.h, eval_hidden16x16(n, l1, acc, pos, s)),
        Some(l1) if l1.n == 8 => with_h!(n.h, eval_hidden8(n, l1, acc, pos, s)),
        Some(l1) => with_h!(n.h, eval_hidden16(n, l1, acc, pos, s)),
    }
}

/// Network with one hidden layer: u8 inputs from both perspectives (side to
/// move first), an int8 matrix product per output bucket, then float SCReLU
/// and the float output layer.
#[inline(never)]
fn eval_hidden16<const H: usize>(n: &Network, l1: &Hidden, acc: &Acc, pos: &Position, s: &mut Scratch) -> i32 {
    eval_hidden::<H, 16, 0, false, false>(n, l1, acc, pos, s)
}

#[inline(never)]
fn eval_hidden8<const H: usize>(n: &Network, l1: &Hidden, acc: &Acc, pos: &Position, s: &mut Scratch) -> i32 {
    eval_hidden::<H, 8, 0, false, false>(n, l1, acc, pos, s)
}

#[inline(never)]
fn eval_hidden16x16d<const H: usize>(n: &Network, l1: &Hidden, acc: &Acc, pos: &Position, s: &mut Scratch) -> i32 {
    eval_hidden::<H, 16, 16, false, true>(n, l1, acc, pos, s)
}

#[inline(never)]
fn eval_hidden16x32d<const H: usize>(n: &Network, l1: &Hidden, acc: &Acc, pos: &Position, s: &mut Scratch) -> i32 {
    eval_hidden::<H, 16, 32, false, true>(n, l1, acc, pos, s)
}

#[inline(never)]
fn eval_hidden_pw16<const H: usize>(n: &Network, l1: &Hidden, acc: &Acc, pos: &Position, s: &mut Scratch) -> i32 {
    eval_hidden::<H, 16, 0, true, false>(n, l1, acc, pos, s)
}

#[inline(never)]
fn eval_hidden_pw16x16<const H: usize>(n: &Network, l1: &Hidden, acc: &Acc, pos: &Position, s: &mut Scratch) -> i32 {
    eval_hidden::<H, 16, 16, true, false>(n, l1, acc, pos, s)
}

#[inline(never)]
fn eval_hidden_pw16x32<const H: usize>(n: &Network, l1: &Hidden, acc: &Acc, pos: &Position, s: &mut Scratch) -> i32 {
    eval_hidden::<H, 16, 32, true, false>(n, l1, acc, pos, s)
}

#[inline(never)]
fn eval_hidden_pw16x16d<const H: usize>(n: &Network, l1: &Hidden, acc: &Acc, pos: &Position, s: &mut Scratch) -> i32 {
    eval_hidden::<H, 16, 16, true, true>(n, l1, acc, pos, s)
}

#[inline(never)]
fn eval_hidden_pw16x32d<const H: usize>(n: &Network, l1: &Hidden, acc: &Acc, pos: &Position, s: &mut Scratch) -> i32 {
    eval_hidden::<H, 16, 32, true, true>(n, l1, acc, pos, s)
}

#[inline(never)]
fn eval_hidden16x16<const H: usize>(n: &Network, l1: &Hidden, acc: &Acc, pos: &Position, s: &mut Scratch) -> i32 {
    eval_hidden::<H, 16, 16, false, false>(n, l1, acc, pos, s)
}

#[inline(never)]
fn eval_hidden16x32<const H: usize>(n: &Network, l1: &Hidden, acc: &Acc, pos: &Position, s: &mut Scratch) -> i32 {
    eval_hidden::<H, 16, 32, false, false>(n, l1, acc, pos, s)
}

#[inline(always)]
fn eval_hidden<const H: usize, const L: usize, const L2: usize, const PW: bool, const DUAL: bool>(
    n: &Network,
    l1: &Hidden,
    acc: &Acc,
    pos: &Position,
    s: &mut Scratch,
) -> i32 {
    let h = hidden::<H>(n);
    // Hidden-layer inputs: h per perspective (SCReLU) or h/2 (pairwise).
    let inl = if PW { h } else { 2 * h };
    let bucket = n.bucket_of[pos.occ().count_ones() as usize] as usize;
    let Scratch { x, nz } = s;
    let nz = &mut nz[..];
    let count = if PW {
        let (us, them) = (acc.side(pos.stm, h), acc.side(pos.stm ^ 1, h));
        if n.qa == 255 && l1.shift == 9 {
            unsafe { to_u8_pw_nz_255_9(us, them, &mut x.0[..h], nz) }
        } else {
            to_u8_pw_scalar(us, &mut x.0[..h / 2], n.qa, l1.shift);
            to_u8_pw_scalar(them, &mut x.0[h / 2..h], n.qa, l1.shift);
            unsafe { scan_nz(&x.0[..h], nz) }
        }
    } else if n.qa == 255 && l1.shift == 9 {
        // Converts and lists the non-zero groups in one pass.
        unsafe { to_u8_nz_255_9(acc.side(pos.stm, h), acc.side(pos.stm ^ 1, h), &mut x.0[..2 * h], nz) }
    } else {
        to_u8_scalar(acc.side(pos.stm, h), &mut x.0[..h], n.qa, l1.shift);
        to_u8_scalar(acc.side(pos.stm ^ 1, h), &mut x.0[h..2 * h], n.qa, l1.shift);
        unsafe { scan_nz(&x.0[..2 * h], nz) }
    };
    // A shared hidden layer has one block for all buckets (branch-free).
    let b1i = bucket & !(l1.shared as usize).wrapping_neg();
    let w = &l1.w1[b1i * L * inl..(b1i + 1) * L * inl];
    let z = unsafe { l1_product::<L>(&x.0[..inl], &nz[..count + 8], count, w) };
    let k = 1.0 / (l1.in_scale * l1.w_scale);
    // The float layers, 8 lanes at a time with AVX2 (every build has it; the
    // x86-64-v3 baseline). Separate multiply and add (no FMA) and a fixed
    // reduction order: every ISA gives the same result.
    let out = unsafe { hidden_float::<L, L2, DUAL>(l1, &z, k, b1i, bucket) };
    (out * n.scale as f32) as i32
}

/// After the int8 product z: SCReLU of the first hidden layer (L neurons),
/// then either the output layer, or the second hidden layer (L2 neurons,
/// SCReLU) and the output layer. Returns the output before scaling.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hidden_float<const L: usize, const L2: usize, const DUAL: bool>(
    l1: &Hidden,
    z: &[i32; L1_SIZE],
    k: f32,
    b1i: usize,
    bucket: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let (zero, one, kv) = (_mm256_setzero_ps(), _mm256_set1_ps(1.0), _mm256_set1_ps(k));
    let screlu = |v: __m256| {
        let c = _mm256_min_ps(_mm256_max_ps(v, zero), one);
        _mm256_mul_ps(c, c)
    };
    // First hidden layer activations, 8 at a time.
    let b1 = &l1.b1[b1i];
    let mut v1 = [zero; L1_SIZE / 8];
    let mut c1 = [zero; L1_SIZE / 8]; // CReLU of the same pre-activations (dual)
    for c in 0..L / 8 {
        let zi = _mm256_loadu_si256(z.as_ptr().add(8 * c) as *const __m256i);
        let pre = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(zi), kv), _mm256_loadu_ps(b1.as_ptr().add(8 * c)));
        v1[c] = screlu(pre);
        if DUAL {
            c1[c] = _mm256_min_ps(_mm256_max_ps(pre, zero), one);
        }
    }
    let mut acc = zero;
    if L2 == 0 {
        let w2 = &l1.w2[bucket];
        for c in 0..L / 8 {
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v1[c], _mm256_loadu_ps(w2.as_ptr().add(8 * c))));
        }
    } else {
        let (wm, bm, wo) = (&l1.wm[bucket], &l1.bm[bucket], &l1.wo[bucket]);
        // Second-layer inputs: SCReLU (L), then CReLU (L) when dual.
        let mut a1 = [0f32; 2 * L1_SIZE];
        for c in 0..L / 8 {
            _mm256_storeu_ps(a1.as_mut_ptr().add(8 * c), v1[c]);
            if DUAL {
                _mm256_storeu_ps(a1.as_mut_ptr().add(L + 8 * c), c1[c]);
            }
        }
        let mut u = [zero; L2_MAX / 8];
        for c in 0..L2 / 8 {
            u[c] = _mm256_loadu_ps(bm.as_ptr().add(8 * c));
        }
        for i in 0..if DUAL { 2 * L } else { L } {
            let x = _mm256_set1_ps(a1[i]);
            for c in 0..L2 / 8 {
                u[c] = _mm256_add_ps(u[c], _mm256_mul_ps(x, _mm256_loadu_ps(wm[i].as_ptr().add(8 * c))));
            }
        }
        for c in 0..L2 / 8 {
            acc = _mm256_add_ps(acc, _mm256_mul_ps(screlu(u[c]), _mm256_loadu_ps(wo.as_ptr().add(8 * c))));
        }
    }
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    l1.b2[bucket] + ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]))
}

/// NEON version of hidden_float: the AVX2 version's 8 lanes as two 4-lane
/// halves, with the same operations in the same order per lane (separate
/// multiply and add, never fused) and the same final reduction. Vectors of 4
/// outputs alternate between the two accumulator halves, so each lane sums
/// the same outputs in the same order as in the 8-lane version. maxnm/minnm
/// clamp as max/min do (finite inputs; -0 and +0 both clamp to +0).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn hidden_float<const L: usize, const L2: usize, const DUAL: bool>(
    l1: &Hidden,
    z: &[i32; L1_SIZE],
    k: f32,
    b1i: usize,
    bucket: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let (zero, one, kv) = (vdupq_n_f32(0.0), vdupq_n_f32(1.0), vdupq_n_f32(k));
    let clamp = |v: float32x4_t| vminnmq_f32(vmaxnmq_f32(v, zero), one);
    let screlu = |v: float32x4_t| {
        let c = clamp(v);
        vmulq_f32(c, c)
    };
    // First hidden layer activations, 4 at a time.
    let b1 = &l1.b1[b1i];
    let mut v1 = [zero; L1_SIZE / 4];
    let mut c1 = [zero; L1_SIZE / 4]; // CReLU of the same pre-activations (dual)
    for c in 0..L / 4 {
        let zi = vld1q_s32(z.as_ptr().add(4 * c));
        let pre = vaddq_f32(vmulq_f32(vcvtq_f32_s32(zi), kv), vld1q_f32(b1.as_ptr().add(4 * c)));
        v1[c] = screlu(pre);
        if DUAL {
            c1[c] = clamp(pre);
        }
    }
    // acc[0] is the 8-lane accumulator's lanes 0-3, acc[1] lanes 4-7.
    let mut acc = [zero; 2];
    if L2 == 0 {
        let w2 = &l1.w2[bucket];
        for c in 0..L / 4 {
            acc[c % 2] = vaddq_f32(acc[c % 2], vmulq_f32(v1[c], vld1q_f32(w2.as_ptr().add(4 * c))));
        }
    } else {
        let (wm, bm, wo) = (&l1.wm[bucket], &l1.bm[bucket], &l1.wo[bucket]);
        // Second-layer inputs: SCReLU (L), then CReLU (L) when dual.
        let mut a1 = [0f32; 2 * L1_SIZE];
        for c in 0..L / 4 {
            vst1q_f32(a1.as_mut_ptr().add(4 * c), v1[c]);
            if DUAL {
                vst1q_f32(a1.as_mut_ptr().add(L + 4 * c), c1[c]);
            }
        }
        let mut u = [zero; L2_MAX / 4];
        for c in 0..L2 / 4 {
            u[c] = vld1q_f32(bm.as_ptr().add(4 * c));
        }
        for i in 0..if DUAL { 2 * L } else { L } {
            let x = vdupq_n_f32(a1[i]);
            for c in 0..L2 / 4 {
                u[c] = vaddq_f32(u[c], vmulq_f32(x, vld1q_f32(wm[i].as_ptr().add(4 * c))));
            }
        }
        for c in 0..L2 / 4 {
            acc[c % 2] = vaddq_f32(acc[c % 2], vmulq_f32(screlu(u[c]), vld1q_f32(wo.as_ptr().add(4 * c))));
        }
    }
    // (l0 + l4, l1 + l5, l2 + l6, l3 + l7), then the same sums as the x86 version.
    let s = vaddq_f32(acc[0], acc[1]);
    let p = vpaddq_f32(s, s);
    l1.b2[bucket] + vgetq_lane_f32(p, 0) + vgetq_lane_f32(p, 1)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
unsafe fn hidden_float<const L: usize, const L2: usize, const DUAL: bool>(
    l1: &Hidden,
    z: &[i32; L1_SIZE],
    k: f32,
    b1i: usize,
    bucket: usize,
) -> f32 {
    hidden_float_portable::<L, L2, DUAL>(l1, z, k, b1i, bucket)
}

/// Portable version of hidden_float with the same operations in the same
/// order per lane (8 lanes; separate multiply and add; the same final
/// reduction), so it gives the same result as the SIMD versions (l1check
/// compares them).
#[inline(always)]
fn hidden_float_portable<const L: usize, const L2: usize, const DUAL: bool>(
    l1: &Hidden,
    z: &[i32; L1_SIZE],
    k: f32,
    b1i: usize,
    bucket: usize,
) -> f32 {
    let screlu = |v: f32| {
        let c = v.max(0.0).min(1.0);
        c * c
    };
    let b1 = &l1.b1[b1i];
    // Second-layer inputs: SCReLU (L), then CReLU (L) when dual.
    let mut v1 = [0f32; 2 * L1_SIZE];
    for i in 0..L {
        let pre = z[i] as f32 * k + b1[i];
        v1[i] = screlu(pre);
        if DUAL {
            v1[L + i] = pre.max(0.0).min(1.0);
        }
    }
    let mut acc = [0f32; 8];
    if L2 == 0 {
        let w2 = &l1.w2[bucket];
        for i in 0..L {
            acc[i % 8] = acc[i % 8] + v1[i] * w2[i];
        }
    } else {
        let (wm, bm, wo) = (&l1.wm[bucket], &l1.bm[bucket], &l1.wo[bucket]);
        let mut u = [0f32; L2_MAX];
        u[..L2].copy_from_slice(&bm[..L2]);
        for i in 0..if DUAL { 2 * L } else { L } {
            let x = v1[i];
            for j in 0..L2 {
                u[j] = u[j] + x * wm[i][j];
            }
        }
        for j in 0..L2 {
            acc[j % 8] = acc[j % 8] + screlu(u[j]) * wo[j];
        }
    }
    l1.b2[bucket] + ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
}

/// Portable kernels (targets without SIMD versions): the same results as the
/// SIMD ones.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
unsafe fn to_u8_255_9(a: &[i16], x: &mut [u8]) {
    to_u8_scalar(a, x, 255, 9)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
unsafe fn to_u8_pw_nz_255_9(a0: &[i16], a1: &[i16], x: &mut [u8], nz: &mut [u16]) -> usize {
    let half = a0.len() / 2;
    to_u8_pw_scalar(a0, &mut x[..half], 255, 9);
    to_u8_pw_scalar(a1, &mut x[half..2 * half], 255, 9);
    scan_nz(&x[..2 * half], nz)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
unsafe fn to_u8_nz_255_9(a0: &[i16], a1: &[i16], x: &mut [u8], nz: &mut [u16]) -> usize {
    let h = a0.len();
    to_u8_scalar(a0, &mut x[..h], 255, 9);
    to_u8_scalar(a1, &mut x[h..2 * h], 255, 9);
    scan_nz(&x[..2 * h], nz)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
unsafe fn scan_nz(x: &[u8], nz: &mut [u16]) -> usize {
    scan_nz_scalar(x, nz)
}

/// Portable scan_nz (the reference l1check compares the SIMD scans against).
fn scan_nz_scalar(x: &[u8], nz: &mut [u16]) -> usize {
    let mut count = 0;
    for (g, c) in x.chunks_exact(4).enumerate() {
        nz[count] = g as u16;
        count += (c != [0, 0, 0, 0]) as usize;
    }
    count
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
unsafe fn l1_product_portable(x: &[u8], nz: &[u16], count: usize, w: &[i8], l: usize) -> [i32; L1_SIZE] {
    let mut z = [0i32; L1_SIZE];
    for &g in &nz[..count] {
        let g = g as usize;
        let wg = &w[g * l * 4..(g + 1) * l * 4];
        for n in 0..l {
            for j in 0..4 {
                z[n] += x[4 * g + j] as i32 * wg[n * 4 + j] as i32;
            }
        }
    }
    z
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
unsafe fn l1_product16(x: &[u8], nz: &[u16], count: usize, w: &[i8]) -> [i32; L1_SIZE] {
    l1_product_portable(x, nz, count, w, 16)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
unsafe fn l1_product8(x: &[u8], nz: &[u16], count: usize, w: &[i8]) -> [i32; L1_SIZE] {
    l1_product_portable(x, nz, count, w, 8)
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

/// Pairwise: x[i] = (clamp(a[i]) * clamp(a[i + h/2])) >> shift, h/2 outputs.
fn to_u8_pw_scalar(a: &[i16], x: &mut [u8], qa: i32, shift: u32) {
    let half = a.len() / 2;
    for i in 0..half {
        let (c1, c2) = ((a[i] as i32).clamp(0, qa), (a[half + i] as i32).clamp(0, qa));
        x[i] = ((c1 * c2) >> shift) as u8;
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
            let c =
                _mm512_min_epi16(_mm512_max_epi16(_mm512_loadu_si512(a.as_ptr().add(o) as *const __m512i), zero), qa);
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

/// to_u8 (QA 255, shift 9) of both perspectives into x (a0 then a1), listing
/// the non-zero 4-input groups in nz as each 64-byte block is produced (no
/// second pass over x). Returns the number of groups listed.
#[cfg(all(avx512_intrinsics, target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn to_u8_nz_255_9(a0: &[i16], a1: &[i16], x: &mut [u8], nz: &mut [u16]) -> usize {
    use std::arch::x86_64::*;
    let zero = _mm512_setzero_si512();
    let qa = _mm512_set1_epi16(255);
    let order = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    let mut count = 0;
    for (half, a) in [a0, a1].into_iter().enumerate() {
        let base = half * a0.len();
        for i in (0..a.len()).step_by(64) {
            let sq = |o: usize| {
                let c = _mm512_min_epi16(
                    _mm512_max_epi16(_mm512_loadu_si512(a.as_ptr().add(o) as *const __m512i), zero),
                    qa,
                );
                _mm512_mulhi_epu16(_mm512_slli_epi16(c, 7), c)
            };
            let (lo, hi) = (sq(i), if i + 32 < a.len() { sq(i + 32) } else { zero });
            let p = _mm512_permutexvar_epi64(order, _mm512_packus_epi16(lo, hi));
            let o = base + i;
            if i + 64 <= a.len() {
                _mm512_storeu_si512(x.as_mut_ptr().add(o) as *mut __m512i, p);
            } else {
                _mm256_storeu_si256(x.as_mut_ptr().add(o) as *mut __m256i, _mm512_castsi512_si256(p));
            }
            // The upper half of p is zero past the end of a.
            let m = _mm512_test_epi32_mask(p, p) as u32;
            count = push_nz(nz, count, m & 0xff, (o / 4) as u16);
            count = push_nz(nz, count, m >> 8, (o / 4) as u16 + 8);
        }
    }
    count
}

/// Pairwise version of to_u8_nz_255_9: per perspective, the first half times
/// the second half, (c1 * c2) >> 9 = mulhi_u16(c1 << 7, c2); h/2 outputs each
/// (h/2 is a multiple of 64).
#[cfg(all(avx512_intrinsics, target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn to_u8_pw_nz_255_9(a0: &[i16], a1: &[i16], x: &mut [u8], nz: &mut [u16]) -> usize {
    use std::arch::x86_64::*;
    let zero = _mm512_setzero_si512();
    let qa = _mm512_set1_epi16(255);
    let order = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    let half = a0.len() / 2;
    let mut count = 0;
    for (p, a) in [a0, a1].into_iter().enumerate() {
        let base = p * half;
        let clamp = |o: usize| {
            _mm512_min_epi16(_mm512_max_epi16(_mm512_loadu_si512(a.as_ptr().add(o) as *const __m512i), zero), qa)
        };
        for i in (0..half).step_by(64) {
            let prod = |o: usize| _mm512_mulhi_epu16(_mm512_slli_epi16(clamp(o), 7), clamp(half + o));
            let q = _mm512_permutexvar_epi64(order, _mm512_packus_epi16(prod(i), prod(i + 32)));
            let o = base + i;
            _mm512_storeu_si512(x.as_mut_ptr().add(o) as *mut __m512i, q);
            let m = _mm512_test_epi32_mask(q, q) as u32;
            count = push_nz(nz, count, m & 0xff, (o / 4) as u16);
            count = push_nz(nz, count, m >> 8, (o / 4) as u16 + 8);
        }
    }
    count
}

/// AVX2 version of to_u8_pw_nz_255_9.
#[cfg(all(target_arch = "x86_64", not(all(avx512_intrinsics, target_feature = "avx512bw"))))]
#[inline(always)]
unsafe fn to_u8_pw_nz_255_9(a0: &[i16], a1: &[i16], x: &mut [u8], nz: &mut [u16]) -> usize {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(255);
    let half = a0.len() / 2;
    let mut count = 0;
    for (p, a) in [a0, a1].into_iter().enumerate() {
        let base = p * half;
        let clamp = |o: usize| {
            _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(a.as_ptr().add(o) as *const __m256i), zero), qa)
        };
        for i in (0..half).step_by(32) {
            let prod = |o: usize| _mm256_mulhi_epu16(_mm256_slli_epi16(clamp(o), 7), clamp(half + o));
            let q = _mm256_permute4x64_epi64(_mm256_packus_epi16(prod(i), prod(i + 16)), 0b11_01_10_00);
            let o = base + i;
            _mm256_storeu_si256(x.as_mut_ptr().add(o) as *mut __m256i, q);
            let m = !(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(q, zero))) as u32) & 0xff;
            count = push_nz(nz, count, m, (o / 4) as u16);
        }
    }
    count
}

/// AVX2 version of to_u8_nz_255_9.
#[cfg(all(target_arch = "x86_64", not(all(avx512_intrinsics, target_feature = "avx512bw"))))]
#[inline(always)]
unsafe fn to_u8_nz_255_9(a0: &[i16], a1: &[i16], x: &mut [u8], nz: &mut [u16]) -> usize {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(255);
    let mut count = 0;
    for (half, a) in [a0, a1].into_iter().enumerate() {
        let base = half * a0.len();
        for i in (0..a.len()).step_by(32) {
            let sq = |o: usize| {
                let c = _mm256_min_epi16(
                    _mm256_max_epi16(_mm256_loadu_si256(a.as_ptr().add(o) as *const __m256i), zero),
                    qa,
                );
                _mm256_mulhi_epu16(_mm256_slli_epi16(c, 7), c)
            };
            let p = _mm256_permute4x64_epi64(_mm256_packus_epi16(sq(i), sq(i + 16)), 0b11_01_10_00);
            let o = base + i;
            _mm256_storeu_si256(x.as_mut_ptr().add(o) as *mut __m256i, p);
            let m = !(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(p, zero))) as u32) & 0xff;
            count = push_nz(nz, count, m, (o / 4) as u16);
        }
    }
    count
}

#[cfg(all(target_arch = "x86_64", not(all(avx512_intrinsics, target_feature = "avx512bw"))))]
#[inline(always)]
unsafe fn to_u8_255_9(a: &[i16], x: &mut [u8]) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(255);
    for i in (0..a.len()).step_by(32) {
        let sq = |o: usize| {
            let c =
                _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(a.as_ptr().add(o) as *const __m256i), zero), qa);
            _mm256_mulhi_epu16(_mm256_slli_epi16(c, 7), c)
        };
        // packus interleaves the 128-bit lanes; permute4x64 restores the order.
        let p = _mm256_permute4x64_epi64(_mm256_packus_epi16(sq(i), sq(i + 16)), 0b11_01_10_00);
        _mm256_storeu_si256(x.as_mut_ptr().add(i) as *mut __m256i, p);
    }
}

/// For each 8-bit mask, the positions of its set bits (unused slots 0): lets
/// the kernels list non-zero input groups without a branch per group.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
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
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn push_nz(nz: &mut [u16], count: usize, m: u32, base: u16) -> usize {
    use std::arch::x86_64::*;
    let idx = _mm_add_epi16(
        _mm_loadu_si128(NZ_TABLE.get_unchecked(m as usize).as_ptr() as *const __m128i),
        _mm_set1_epi16(base as i16),
    );
    _mm_storeu_si128(nz.as_mut_ptr().add(count) as *mut __m128i, idx);
    count + m.count_ones() as usize
}

/// z[n] = sum_i x[i] * w[n][i] for the L1_SIZE outputs, with w in the
/// [in / 4][out][in % 4] layout. x.len() is a multiple of 64. (Scan, then
/// product; eval_hidden fuses the scan into the u8 conversion instead.)
#[inline(always)]
fn l1_matmul<const L: usize>(x: &[u8], w: &[i8]) -> [i32; L1_SIZE] {
    let mut nz = [0u16; 2 * MAX_H / 4 + 8];
    unsafe {
        let count = scan_nz(x, &mut nz);
        l1_product::<L>(x, &nz[..count + 8], count, w)
    }
}

/// Lists the 4-input groups of x with any non-zero input; returns the count.
#[cfg(all(avx512_intrinsics, target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn scan_nz(x: &[u8], nz: &mut [u16]) -> usize {
    use std::arch::x86_64::*;
    let mut count = 0;
    for c in (0..x.len()).step_by(64) {
        let v = _mm512_loadu_si512(x.as_ptr().add(c) as *const __m512i);
        let m = _mm512_test_epi32_mask(v, v) as u32;
        count = push_nz(nz, count, m & 0xff, (c / 4) as u16);
        count = push_nz(nz, count, m >> 8, (c / 4) as u16 + 8);
    }
    count
}

#[cfg(all(target_arch = "x86_64", not(all(avx512_intrinsics, target_feature = "avx512bw"))))]
#[inline(always)]
unsafe fn scan_nz(x: &[u8], nz: &mut [u16]) -> usize {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let mut count = 0;
    for c in (0..x.len()).step_by(32) {
        let v = _mm256_loadu_si256(x.as_ptr().add(c) as *const __m256i);
        let m = !(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, zero))) as u32) & 0xff;
        count = push_nz(nz, count, m, (c / 4) as u16);
    }
    count
}

/// AVX-512: one register holds all 16 outputs; each group of 4 inputs is a
/// broadcast and one VNNI dpbusd (or maddubs + madd without VNNI). Inputs are
/// at most 127 and weights within +-127, so maddubs's i16 sums can't overflow.
#[cfg(all(avx512_intrinsics, target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn l1_product16(x: &[u8], nz: &[u16], count: usize, w: &[i8]) -> [i32; L1_SIZE] {
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
    // SCReLU leaves most inputs at zero: only the listed groups of 4 with any
    // non-zero input are used (the others' weights aren't even loaded).
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
    _mm512_storeu_si512(
        z.as_mut_ptr() as *mut __m512i,
        _mm512_add_epi32(_mm512_add_epi32(s0, s1), _mm512_add_epi32(s2, s3)),
    );
    z
}

/// AVX2: two registers of 8 outputs; per group of 4 inputs, maddubs + madd.
/// acc + Σ (u8 x) · (i8 w) over each 4-byte group, per 32-bit lane: one
/// vpdpbusd with AVX-VNNI (VEX, e.g. Alder Lake and later without AVX-512),
/// else maddubs + madd + add. Same result either way: x <= 127 and |w| <= 127
/// keep maddubs's i16 pair sums (at most 32258) from saturating.
#[cfg(all(target_arch = "x86_64", not(all(avx512_intrinsics, target_feature = "avx512bw"))))]
#[inline(always)]
unsafe fn dpbusd256(
    acc: std::arch::x86_64::__m256i,
    x: std::arch::x86_64::__m256i,
    w: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    #[cfg(target_feature = "avxvnni")]
    {
        _mm256_dpbusd_avx_epi32(acc, x, w)
    }
    #[cfg(not(target_feature = "avxvnni"))]
    {
        _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), _mm256_set1_epi16(1)))
    }
}

#[cfg(all(target_arch = "x86_64", not(all(avx512_intrinsics, target_feature = "avx512bw"))))]
#[inline(always)]
unsafe fn l1_product16(x: &[u8], nz: &[u16], count: usize, w: &[i8]) -> [i32; L1_SIZE] {
    use std::arch::x86_64::*;
    let (xp, wp) = (x.as_ptr() as *const i32, w.as_ptr());
    let mut a = [_mm256_setzero_si256(); 4];
    // Only the listed non-zero groups (see the AVX-512 version).
    let mut step = |k: usize, g: usize| {
        let xb = _mm256_set1_epi32(xp.add(g).read_unaligned());
        let w0 = _mm256_load_si256(wp.add(g * 64) as *const __m256i);
        let w1 = _mm256_load_si256(wp.add(g * 64 + 32) as *const __m256i);
        a[2 * k] = dpbusd256(a[2 * k], xb, w0);
        a[2 * k + 1] = dpbusd256(a[2 * k + 1], xb, w1);
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

/// The hidden-layer product for L (8 or 16) outputs; outputs past L are 0.
#[inline(always)]
unsafe fn l1_product<const L: usize>(x: &[u8], nz: &[u16], count: usize, w: &[i8]) -> [i32; L1_SIZE] {
    if L == 8 {
        l1_product8(x, nz, count, w)
    } else {
        l1_product16(x, nz, count, w)
    }
}

/// AVX-512, 8 outputs: a group's weight row is 32 bytes, so two listed groups
/// share one register (one in each half) and one VNNI dpbusd.
#[cfg(all(avx512_intrinsics, target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn l1_product8(x: &[u8], nz: &[u16], count: usize, w: &[i8]) -> [i32; L1_SIZE] {
    use std::arch::x86_64::*;
    let (xp, wp) = (x.as_ptr() as *const i32, w.as_ptr());
    #[inline(always)]
    unsafe fn step(s: __m512i, xa: i32, xb: i32, wa: *const i8, wb: *const i8) -> __m512i {
        let xv = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi32(xa)), _mm256_set1_epi32(xb), 1);
        let wv = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_load_si256(wa as *const __m256i)),
            _mm256_load_si256(wb as *const __m256i),
            1,
        );
        #[cfg(target_feature = "avx512vnni")]
        {
            _mm512_dpbusd_epi32(s, xv, wv)
        }
        #[cfg(not(target_feature = "avx512vnni"))]
        {
            _mm512_add_epi32(s, _mm512_madd_epi16(_mm512_maddubs_epi16(xv, wv), _mm512_set1_epi16(1)))
        }
    }
    let g = |i: usize| *nz.get_unchecked(i) as usize;
    let mut s0 = _mm512_setzero_si512();
    let mut s1 = _mm512_setzero_si512();
    let mut i = 0;
    while i + 3 < count {
        let (g0, g1, g2, g3) = (g(i), g(i + 1), g(i + 2), g(i + 3));
        s0 = step(s0, xp.add(g0).read_unaligned(), xp.add(g1).read_unaligned(), wp.add(g0 * 32), wp.add(g1 * 32));
        s1 = step(s1, xp.add(g2).read_unaligned(), xp.add(g3).read_unaligned(), wp.add(g2 * 32), wp.add(g3 * 32));
        i += 4;
    }
    while i < count {
        // A lone group: the other half multiplies zero inputs.
        let g0 = g(i);
        s0 = step(s0, xp.add(g0).read_unaligned(), 0, wp.add(g0 * 32), wp.add(g0 * 32));
        i += 1;
    }
    let s = _mm512_add_epi32(s0, s1);
    let r = _mm256_add_epi32(_mm512_castsi512_si256(s), _mm512_extracti64x4_epi64(s, 1));
    let mut z = [0i32; L1_SIZE];
    _mm256_storeu_si256(z.as_mut_ptr() as *mut __m256i, r);
    z
}

/// AVX2, 8 outputs: one register per group; maddubs + madd.
#[cfg(all(target_arch = "x86_64", not(all(avx512_intrinsics, target_feature = "avx512bw"))))]
#[inline(always)]
unsafe fn l1_product8(x: &[u8], nz: &[u16], count: usize, w: &[i8]) -> [i32; L1_SIZE] {
    use std::arch::x86_64::*;
    let (xp, wp) = (x.as_ptr() as *const i32, w.as_ptr());
    let mut a = [_mm256_setzero_si256(); 4];
    let mut step = |k: usize, g: usize| {
        let xb = _mm256_set1_epi32(xp.add(g).read_unaligned());
        let wv = _mm256_load_si256(wp.add(g * 32) as *const __m256i);
        a[k] = dpbusd256(a[k], xb, wv);
    };
    let mut i = 0;
    while i + 3 < count {
        for k in 0..4 {
            step(k, *nz.get_unchecked(i + k) as usize);
        }
        i += 4;
    }
    while i < count {
        step(0, *nz.get_unchecked(i) as usize);
        i += 1;
    }
    let r = _mm256_add_epi32(_mm256_add_epi32(a[0], a[1]), _mm256_add_epi32(a[2], a[3]));
    let mut z = [0i32; L1_SIZE];
    _mm256_storeu_si256(z.as_mut_ptr() as *mut __m256i, r);
    z
}

/// Scalar reference for the hidden-layer product with `l` outputs (used by
/// `l1check` and the diagnostics).
pub fn l1_matmul_scalar(x: &[u8], w: &[i8], l: usize) -> [i32; L1_SIZE] {
    let mut z = [0i32; L1_SIZE];
    for (g, xs) in x.chunks_exact(4).enumerate() {
        let wg = &w[g * l * 4..(g + 1) * l * 4];
        for n in 0..l {
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

/// Reference implementation of `dot` (below), also the fallback on targets
/// without a SIMD version.
#[cfg_attr(not(test), allow(dead_code))]
fn dot_scalar<const SCRELU: bool>(a: &[i16], w: &[i16], qa: i32) -> i32 {
    let mut s = 0i32;
    for (&x, &w) in a.iter().zip(w) {
        let c = x.clamp(0, qa as i16);
        let m = if SCRELU { c.wrapping_mul(w) } else { w };
        s = s.wrapping_add(c as i32 * m as i32);
    }
    s
}

/// SCReLU: Σ clamp(a, 0, qa)² · w; CReLU: Σ clamp(a, 0, qa) · w, over
/// a.len() values (a multiple of 32). `a` is 64-byte aligned.
///
/// The arithmetic is wrapping throughout: for SCReLU, clamp(a) · w is
/// truncated to i16 before the second multiply, and the i32 sums wrap, so
/// every implementation gives the same result bit for bit (see the test).
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
unsafe fn dot<const SCRELU: bool>(a: &[i16], w: &[i16], qa: i32) -> i32 {
    dot_scalar::<SCRELU>(a, w, qa)
}

#[cfg(target_arch = "x86_64")]
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

/// AVX2 version: two 16-lane accumulators per 32-wide step.
#[cfg(target_arch = "x86_64")]
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
            // Fused conversion + scan (two halves) vs conversion then scan.
            let mut x3 = vec![0u8; 2 * h];
            let mut nz1 = vec![0u16; 2 * h / 4 + 8];
            let mut nz2 = vec![0u16; 2 * h / 4 + 8];
            let c1 = unsafe { to_u8_nz_255_9(&a[..h], &a[h..], &mut x3, &mut nz1) };
            let c2 = scan_nz_scalar(&x2, &mut nz2);
            if x3 != x2 || c1 != c2 || nz1[..c1] != nz2[..c2] {
                bad += 1;
                continue;
            }
            let c3 = unsafe { scan_nz(&x2, &mut nz1) };
            if c3 != c2 || nz1[..c3] != nz2[..c2] {
                bad += 1;
                continue;
            }
            // Pairwise conversion + scan vs scalar (h/2 per perspective).
            if h % 128 == 0 {
                let mut p1 = vec![0u8; h];
                let mut p2 = vec![0u8; h];
                let mut pz1 = vec![0u16; h / 4 + 8];
                let mut pz2 = vec![0u16; h / 4 + 8];
                let d1 = unsafe { to_u8_pw_nz_255_9(&a[..h], &a[h..], &mut p1, &mut pz1) };
                to_u8_pw_scalar(&a[..h], &mut p2[..h / 2], 255, 9);
                to_u8_pw_scalar(&a[h..], &mut p2[h / 2..], 255, 9);
                let d2 = scan_nz_scalar(&p2, &mut pz2);
                if p1 != p2 || d1 != d2 || pz1[..d1] != pz2[..d2] {
                    bad += 1;
                    continue;
                }
            }
            let mut w = AlignedI8::zeroed(L1_SIZE * 2 * h);
            for v in w.iter_mut() {
                *v = ((rnd() % 255) as i32 - 127) as i8;
            }
            if l1_matmul::<16>(&x1, &w) != l1_matmul_scalar(&x1, &w, 16) {
                bad += 1;
            }
            if l1_matmul::<8>(&x1, &w[..8 * 2 * h]) != l1_matmul_scalar(&x1, &w[..8 * 2 * h], 8) {
                bad += 1;
            }
            // Sparse inputs, as in real positions (few non-zero groups, odd counts).
            let keep = rnd() % 64;
            let xs: Vec<u8> = x1.iter().map(|&v| if rnd() % 64 < keep { v } else { 0 }).collect();
            if l1_matmul::<16>(&xs, &w) != l1_matmul_scalar(&xs, &w, 16) {
                bad += 1;
            }
            if l1_matmul::<8>(&xs, &w[..8 * 2 * h]) != l1_matmul_scalar(&xs, &w[..8 * 2 * h], 8) {
                bad += 1;
            }
        }
    }
    // The float layers against the portable version, bit for bit, for each
    // layer shape eval_hidden uses. Pre-activations span below 0, 0..1 and
    // above 1, so every clamp region is used.
    let mut rf = || ((rnd() % 2001) as f32 - 1000.0) / 1000.0;
    for _ in 0..trials {
        let l1 = Hidden {
            n: L1_SIZE,
            shift: 9,
            in_scale: 1.0,
            w_scale: 1.0,
            w1: AlignedI8::zeroed(0),
            b1: vec![std::array::from_fn(|_| rf())],
            shared: false,
            pw: false,
            w2: vec![std::array::from_fn(|_| rf())],
            b2: vec![rf()],
            n2: 0,
            wm: vec![std::array::from_fn(|_| std::array::from_fn(|_| rf()))],
            dual: false,
            bm: vec![std::array::from_fn(|_| rf())],
            wo: vec![std::array::from_fn(|_| rf())],
        };
        let z: [i32; L1_SIZE] = std::array::from_fn(|_| (rf() * 40000.0) as i32);
        let k = 1.0 / (12345.0 + rf() * 1000.0);
        let pairs = unsafe {
            [
                (hidden_float::<8, 0, false>(&l1, &z, k, 0, 0), hidden_float_portable::<8, 0, false>(&l1, &z, k, 0, 0)),
                (
                    hidden_float::<16, 0, false>(&l1, &z, k, 0, 0),
                    hidden_float_portable::<16, 0, false>(&l1, &z, k, 0, 0),
                ),
                (
                    hidden_float::<16, 16, false>(&l1, &z, k, 0, 0),
                    hidden_float_portable::<16, 16, false>(&l1, &z, k, 0, 0),
                ),
                (
                    hidden_float::<16, 32, false>(&l1, &z, k, 0, 0),
                    hidden_float_portable::<16, 32, false>(&l1, &z, k, 0, 0),
                ),
                (
                    hidden_float::<16, 16, true>(&l1, &z, k, 0, 0),
                    hidden_float_portable::<16, 16, true>(&l1, &z, k, 0, 0),
                ),
                (
                    hidden_float::<16, 32, true>(&l1, &z, k, 0, 0),
                    hidden_float_portable::<16, 32, true>(&l1, &z, k, 0, 0),
                ),
            ]
        };
        bad += pairs.iter().filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    }
    bad
}

/// Hidden-layer diagnostics over a FEN file with the network in `path`: per
/// output bucket and neuron, how often the neuron is active (pre-activation
/// > 0) and saturated (>= 1), its bias, and the share of int8 weights at the
/// clip. Replaces the embedded network for the rest of the process.
pub fn l1stats(path: &str, fens: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
    let n = load(&bytes)?;
    unsafe { NET = Box::into_raw(Box::new(n)) };
    let n = net();
    let l1 = n.l1.as_ref().ok_or("network has no hidden layer")?;
    if l1.pw {
        return Err("not supported for pairwise nets yet".into());
    }
    let h = n.h;
    let nb = l1.b2.len();
    let mut cnt = vec![0u64; nb];
    let mut act = vec![[0u64; L1_SIZE]; nb];
    let mut sat = vec![[0u64; L1_SIZE]; nb];
    let mut maxv = vec![[f32::MIN; L1_SIZE]; nb];
    let text = std::fs::read_to_string(fens).map_err(|e| format!("{}: {}", fens, e))?;
    let mut acc = Acc::new();
    for line in text.lines() {
        let fen = line.split(['|', ';']).next().unwrap_or("").trim();
        let Some(pos) = crate::position::Position::from_fen(fen) else { continue };
        if (pos.occ().count_ones() as usize) >= n.bucket_of.len() {
            continue;
        }
        acc.refresh(&pos);
        let bucket = n.bucket_of[pos.occ().count_ones() as usize] as usize;
        let mut x = vec![0u8; 2 * h];
        to_u8_scalar(acc.side(pos.stm, h), &mut x[..h], n.qa, l1.shift);
        to_u8_scalar(acc.side(pos.stm ^ 1, h), &mut x[h..], n.qa, l1.shift);
        let b1i = if l1.shared { 0 } else { bucket };
        let ln = l1.n;
        let w = &l1.w1[b1i * ln * 2 * h..(b1i + 1) * ln * 2 * h];
        let z = l1_matmul_scalar(&x, w, ln);
        let k = 1.0 / (l1.in_scale * l1.w_scale);
        cnt[bucket] += 1;
        for i in 0..l1.n {
            let v = z[i] as f32 * k + l1.b1[b1i][i];
            act[bucket][i] += (v > 0.0) as u64;
            sat[bucket][i] += (v >= 1.0) as u64;
            maxv[bucket][i] = maxv[bucket][i].max(v);
        }
    }
    for b in 0..nb {
        let b1i = if l1.shared { 0 } else { b };
        let w = &l1.w1[b1i * l1.n * 2 * h..(b1i + 1) * l1.n * 2 * h];
        let clip = w.iter().filter(|&&v| v.unsigned_abs() >= 126).count();
        let c = cnt[b].max(1) as f64;
        let dead = (0..l1.n).filter(|&i| act[b][i] == 0).count();
        let always = (0..l1.n).filter(|&i| sat[b][i] == cnt[b] && cnt[b] > 0).count();
        print!(
            "bucket {} positions {} dead {} always-saturated {} clipped {:.2}% | active%:",
            b,
            cnt[b],
            dead,
            always,
            100.0 * clip as f64 / w.len() as f64
        );
        for i in 0..l1.n {
            print!(" {:.0}", 100.0 * act[b][i] as f64 / c);
        }
        print!(" | max:");
        for i in 0..l1.n {
            print!(" {:.2}", maxv[b][i]);
        }
        print!(" | b1:");
        for i in 0..l1.n {
            print!(" {:.2}", l1.b1[b1i][i]);
        }
        println!();
    }
    Ok(())
}

/// FT neuron order for a net with a hidden layer, from activity on a FEN file:
/// counts how often each neuron's u8 input is non-zero (both perspectives),
/// prints the share of non-zero 4-input groups now and after sorting neurons
/// by activity (most active first), and writes the order to `out` for the
/// converter's --permute. Replaces the embedded network.
pub fn l1perm(path: &str, fens: &str, out: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
    let n = load(&bytes)?;
    unsafe { NET = Box::into_raw(Box::new(n)) };
    let n = net();
    let l1 = n.l1.as_ref().ok_or("network has no hidden layer")?;
    // Hidden-layer inputs per perspective: h neurons (SCReLU) or h/2 products
    // of neuron pairs j, j + h/2 (pairwise; the order then moves whole pairs).
    let h = if l1.pw { n.h / 2 } else { n.h };
    let text = std::fs::read_to_string(fens).map_err(|e| format!("{}: {}", fens, e))?;
    let mut inputs: Vec<Vec<u8>> = Vec::new();
    let mut acc = Acc::new();
    for line in text.lines() {
        let fen = line.split(['|', ';']).next().unwrap_or("").trim();
        let Some(pos) = crate::position::Position::from_fen(fen) else { continue };
        acc.refresh(&pos);
        let mut x = vec![0u8; 2 * h];
        if l1.pw {
            to_u8_pw_scalar(acc.side(pos.stm, n.h), &mut x[..h], n.qa, l1.shift);
            to_u8_pw_scalar(acc.side(pos.stm ^ 1, n.h), &mut x[h..], n.qa, l1.shift);
        } else {
            to_u8_scalar(acc.side(pos.stm, h), &mut x[..h], n.qa, l1.shift);
            to_u8_scalar(acc.side(pos.stm ^ 1, h), &mut x[h..], n.qa, l1.shift);
        }
        inputs.push(x);
    }
    if inputs.len() < 2 {
        return Err("too few positions".into());
    }
    // Build the order on the even positions, measure it on the odd ones.
    let test: Vec<Vec<u8>> = inputs.iter().skip(1).step_by(2).cloned().collect();
    let inputs: Vec<Vec<u8>> = inputs.into_iter().step_by(2).collect();
    let mut active = vec![0u64; h];
    for x in &inputs {
        for j in 0..h {
            active[j] += (x[j] != 0) as u64 + (x[h + j] != 0) as u64;
        }
    }
    let mut sorted: Vec<usize> = (0..h).collect();
    sorted.sort_by_key(|&j| std::cmp::Reverse(active[j]));
    // Greedy co-activation grouping: bitsets of the samples (position x
    // perspective) where each neuron is active; each group starts from the
    // most active unassigned neuron and adds, three times, the neuron most
    // similar (Jaccard) to the group so far, so neurons that fire together
    // share a group.
    let samples = 2 * inputs.len();
    let words = samples.div_ceil(64);
    let mut bits = vec![0u64; h * words];
    for (k, x) in inputs.iter().enumerate() {
        for side in 0..2 {
            let sm = 2 * k + side;
            for j in 0..h {
                if x[side * h + j] != 0 {
                    bits[j * words + sm / 64] |= 1 << (sm % 64);
                }
            }
        }
    }
    let mut used = vec![false; h];
    let mut perm = Vec::with_capacity(h);
    for &seed in &sorted {
        if used[seed] {
            continue;
        }
        used[seed] = true;
        perm.push(seed);
        let mut union: Vec<u64> = bits[seed * words..(seed + 1) * words].to_vec();
        for _ in 0..3 {
            // Jaccard similarity of the neuron's active samples with the
            // group's: |A & U| / |A | U|.
            let un: u64 = union.iter().map(|u| u.count_ones() as u64).sum();
            let mut best = (-1.0f64, usize::MAX);
            for j in 0..h {
                if used[j] {
                    continue;
                }
                let b = &bits[j * words..(j + 1) * words];
                let add: u64 = b.iter().zip(&union).map(|(a, u)| (a & !u).count_ones() as u64).sum();
                let both = active[j] - add;
                let sim = both as f64 / (un + add).max(1) as f64;
                if sim > best.0 {
                    best = (sim, j);
                }
            }
            let j = best.1;
            used[j] = true;
            perm.push(j);
            for (u, a) in union.iter_mut().zip(&bits[j * words..(j + 1) * words]) {
                *u |= a;
            }
        }
    }
    // Share of 4-input groups with any non-zero input, for an order.
    let nz_share = |order: &[usize]| -> f64 {
        let mut nz = 0u64;
        for x in &test {
            for side in [0, h] {
                for g in order.chunks_exact(4) {
                    nz += g.iter().any(|&j| x[side + j] != 0) as u64;
                }
            }
        }
        nz as f64 / (test.len() * 2 * h / 4) as f64
    };
    let ident: Vec<usize> = (0..h).collect();
    let inputs_nz = active.iter().sum::<u64>() as f64 / (inputs.len() * 2 * h) as f64;
    println!(
        "positions {} (order) + {} (measured); non-zero inputs {:.1}%; non-zero 4-groups: current order {:.1}%, sorted by activity {:.1}%, co-activation groups {:.1}%",
        inputs.len(),
        test.len(),
        100.0 * inputs_nz,
        100.0 * nz_share(&ident),
        100.0 * nz_share(&sorted),
        100.0 * nz_share(&perm)
    );
    let dead = active.iter().filter(|&&a| a == 0).count();
    println!("FT neurons never active: {} of {}", dead, h);
    let s: Vec<String> = perm.iter().map(|i| i.to_string()).collect();
    std::fs::write(out, s.join(" ") + "\n").map_err(|e| format!("{}: {}", out, e))?;
    Ok(())
}

/// NEON: appends the indices (base + bit) of the set bits of mask mc (an
/// nz_mask8 result: mask in bits 0-7, its count in bits 8-11) to nz at
/// count, with one 16-byte store (as the x86 push_nz).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn push_nz(nz: &mut [u16], count: usize, mc: u32, base: u16) -> usize {
    use std::arch::aarch64::*;
    let idx = vaddq_u16(vld1q_u16(NZ_TABLE.get_unchecked((mc & 0xff) as usize).as_ptr()), vdupq_n_u16(base));
    vst1q_u16(nz.as_mut_ptr().add(count), idx);
    count + (mc >> 8) as usize
}

/// For the eight 4-byte groups of q0:q1 (32 bytes): bit k set if group k has
/// a non-zero byte (bits 0-7), and the number of such groups (bits 8-11).
/// Each group's all-ones/zero test, narrowed to 16 bits, is weighted by
/// 0x100 | its bit and summed, so one addv gives both and the count needs no
/// popcount (a slow vector-unit round trip on aarch64 without CSSC).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn nz_mask8(q0: std::arch::aarch64::uint8x16_t, q1: std::arch::aarch64::uint8x16_t) -> u32 {
    use std::arch::aarch64::*;
    const BITS: [u16; 8] = [0x101, 0x102, 0x104, 0x108, 0x110, 0x120, 0x140, 0x180];
    let (g0, g1) = (vreinterpretq_u32_u8(q0), vreinterpretq_u32_u8(q1));
    let t = vcombine_u16(vmovn_u32(vtstq_u32(g0, g0)), vmovn_u32(vtstq_u32(g1, g1)));
    vaddvq_u16(vandq_u16(t, vld1q_u16(BITS.as_ptr()))) as u32
}

/// NEON: clamp(a[o..o + 16], 0, 255) as u8 (vqmovun saturates i16 to 0..255).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn clamp_u8(ap: *const i16) -> std::arch::aarch64::uint8x16_t {
    use std::arch::aarch64::*;
    vqmovun_high_s16(vqmovun_s16(vld1q_s16(ap)), vld1q_s16(ap.add(8)))
}

/// NEON: (c1 * c2) >> 9 per byte. The u8 x u8 product is exact in u16 (at
/// most 65025); uzp2 takes the products' high bytes (>> 8) and a shift by 1
/// completes the >> 9.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mul_255_9(
    c1: std::arch::aarch64::uint8x16_t,
    c2: std::arch::aarch64::uint8x16_t,
) -> std::arch::aarch64::uint8x16_t {
    use std::arch::aarch64::*;
    let (lo, hi) = (vmull_u8(vget_low_u8(c1), vget_low_u8(c2)), vmull_high_u8(c1, c2));
    vshrq_n_u8(vuzp2q_u8(vreinterpretq_u8_u16(lo), vreinterpretq_u8_u16(hi)), 1)
}

/// NEON version of to_u8_pw_nz_255_9: per perspective, 32 outputs at a time,
/// clamp(a[i]) * clamp(a[half + i]) >> 9. half is a multiple of 32.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn to_u8_pw_nz_255_9(a0: &[i16], a1: &[i16], x: &mut [u8], nz: &mut [u16]) -> usize {
    use std::arch::aarch64::*;
    let half = a0.len() / 2;
    let mut count = 0;
    for (p, a) in [a0, a1].into_iter().enumerate() {
        let base = p * half;
        let ap = a.as_ptr();
        let prod = |o: usize| mul_255_9(clamp_u8(ap.add(o)), clamp_u8(ap.add(half + o)));
        for i in (0..half).step_by(32) {
            let (q0, q1) = (prod(i), prod(i + 16));
            let o = base + i;
            vst1q_u8(x.as_mut_ptr().add(o), q0);
            vst1q_u8(x.as_mut_ptr().add(o + 16), q1);
            count = push_nz(nz, count, nz_mask8(q0, q1), (o / 4) as u16);
        }
    }
    count
}

/// NEON version of to_u8_nz_255_9 (a.len() a multiple of 32).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn to_u8_nz_255_9(a0: &[i16], a1: &[i16], x: &mut [u8], nz: &mut [u16]) -> usize {
    use std::arch::aarch64::*;
    let mut count = 0;
    for (half, a) in [a0, a1].into_iter().enumerate() {
        let base = half * a0.len();
        let sq = |o: usize| {
            let c = clamp_u8(a.as_ptr().add(o));
            mul_255_9(c, c)
        };
        for i in (0..a.len()).step_by(32) {
            let (q0, q1) = (sq(i), sq(i + 16));
            let o = base + i;
            vst1q_u8(x.as_mut_ptr().add(o), q0);
            vst1q_u8(x.as_mut_ptr().add(o + 16), q1);
            count = push_nz(nz, count, nz_mask8(q0, q1), (o / 4) as u16);
        }
    }
    count
}

/// NEON version of to_u8_255_9 (a.len() a multiple of 32).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn to_u8_255_9(a: &[i16], x: &mut [u8]) {
    use std::arch::aarch64::*;
    for i in (0..a.len()).step_by(16) {
        let c = clamp_u8(a.as_ptr().add(i));
        vst1q_u8(x.as_mut_ptr().add(i), mul_255_9(c, c));
    }
}

/// NEON version of scan_nz (x.len() a multiple of 32).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn scan_nz(x: &[u8], nz: &mut [u16]) -> usize {
    use std::arch::aarch64::*;
    let mut count = 0;
    for c in (0..x.len()).step_by(32) {
        let (q0, q1) = (vld1q_u8(x.as_ptr().add(c)), vld1q_u8(x.as_ptr().add(c + 16)));
        count = push_nz(nz, count, nz_mask8(q0, q1), (c / 4) as u16);
    }
    count
}

/// acc + Σ x · w over each 4-byte group, per 32-bit lane: one sdot with the
/// dot-product extension (every Apple and recent Arm core), else widening
/// multiplies and pairwise adds. The u8 inputs are at most 127, so they are
/// exact as i8 and the signed dot product gives the u8 x i8 result.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot4(
    acc: std::arch::aarch64::int32x4_t,
    x: std::arch::aarch64::int8x16_t,
    w: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;
    #[cfg(target_feature = "dotprod")]
    {
        vdotq_s32(acc, x, w)
    }
    #[cfg(not(target_feature = "dotprod"))]
    {
        // i8 x i8 products fit i16; pairs, then pairs of pairs, to i32.
        let lo = vpaddlq_s16(vmull_s8(vget_low_s8(x), vget_low_s8(w)));
        let hi = vpaddlq_s16(vmull_high_s8(x, w));
        vaddq_s32(acc, vpaddq_s32(lo, hi))
    }
}

/// NEON, 16 outputs: a group's 64-byte weight row is four vectors of 4
/// outputs; per listed group, a broadcast of its 4 inputs and four dot4s.
/// Two groups per step into separate accumulators (eight chains).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn l1_product16(x: &[u8], nz: &[u16], count: usize, w: &[i8]) -> [i32; L1_SIZE] {
    use std::arch::aarch64::*;
    let (xp, wp) = (x.as_ptr() as *const i32, w.as_ptr());
    let mut a = [vdupq_n_s32(0); 8];
    let mut step = |k: usize, g: usize| {
        let xb = vreinterpretq_s8_s32(vdupq_n_s32(xp.add(g).read_unaligned()));
        let wv = vld1q_s8_x4(wp.add(g * 64));
        a[4 * k] = dot4(a[4 * k], xb, wv.0);
        a[4 * k + 1] = dot4(a[4 * k + 1], xb, wv.1);
        a[4 * k + 2] = dot4(a[4 * k + 2], xb, wv.2);
        a[4 * k + 3] = dot4(a[4 * k + 3], xb, wv.3);
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
    for j in 0..4 {
        vst1q_s32(z.as_mut_ptr().add(4 * j), vaddq_s32(a[j], a[4 + j]));
    }
    z
}

/// NEON, 8 outputs: a group's 32-byte weight row is two vectors of 4
/// outputs; four groups per step into separate accumulators (eight chains).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn l1_product8(x: &[u8], nz: &[u16], count: usize, w: &[i8]) -> [i32; L1_SIZE] {
    use std::arch::aarch64::*;
    let (xp, wp) = (x.as_ptr() as *const i32, w.as_ptr());
    let mut a = [vdupq_n_s32(0); 8];
    let mut step = |k: usize, g: usize| {
        let xb = vreinterpretq_s8_s32(vdupq_n_s32(xp.add(g).read_unaligned()));
        let wv = vld1q_s8_x2(wp.add(g * 32));
        a[2 * k] = dot4(a[2 * k], xb, wv.0);
        a[2 * k + 1] = dot4(a[2 * k + 1], xb, wv.1);
    };
    let mut i = 0;
    while i + 3 < count {
        for k in 0..4 {
            step(k, *nz.get_unchecked(i + k) as usize);
        }
        i += 4;
    }
    while i < count {
        step(0, *nz.get_unchecked(i) as usize);
        i += 1;
    }
    let mut z = [0i32; L1_SIZE];
    for j in 0..2 {
        vst1q_s32(z.as_mut_ptr().add(4 * j), vaddq_s32(vaddq_s32(a[j], a[2 + j]), vaddq_s32(a[4 + j], a[6 + j])));
    }
    z
}

/// NEON version: four 8-lane vectors per 32-wide step, each with its own
/// pair of i32 accumulators (low and high halves of the widening multiply),
/// so the eight multiply-accumulate chains are independent.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot<const SCRELU: bool>(a: &[i16], w: &[i16], qa: i32) -> i32 {
    use std::arch::aarch64::*;
    let h = a.len();
    let zero = vdupq_n_s16(0);
    let qa = vdupq_n_s16(qa as i16);
    let mut s = [vdupq_n_s32(0); 8];
    let mut i = 0;
    while i < h {
        for k in 0..4 {
            let x = vld1q_s16(a.as_ptr().add(i + 8 * k));
            let wv = vld1q_s16(w.as_ptr().add(i + 8 * k));
            let c = vminq_s16(vmaxq_s16(x, zero), qa);
            let m = if SCRELU { vmulq_s16(c, wv) } else { wv };
            s[2 * k] = vmlal_s16(s[2 * k], vget_low_s16(c), vget_low_s16(m));
            s[2 * k + 1] = vmlal_high_s16(s[2 * k + 1], c, m);
        }
        i += 32;
    }
    let s = vaddq_s32(
        vaddq_s32(vaddq_s32(s[0], s[1]), vaddq_s32(s[2], s[3])),
        vaddq_s32(vaddq_s32(s[4], s[5]), vaddq_s32(s[6], s[7])),
    );
    vaddvq_s32(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SIMD `dot` matches the scalar reference, including where the
    /// 16-bit products and 32-bit sums wrap.
    /// The hidden-layer kernels (u8 conversion, fused conversion + non-zero
    /// scan, pairwise conversion, sparse int8 products) against their scalar
    /// versions on random inputs.
    #[test]
    fn hidden_kernels_match_scalar() {
        assert_eq!(l1check(20), 0);
    }

    #[test]
    fn dot_matches_scalar() {
        let mut r = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            r
        };
        for h in [32usize, 256, 1024, 2048] {
            for _ in 0..20 {
                let a = Aligned::from((0..h).map(|_| next() as i16));
                let w: Vec<i16> = (0..h).map(|_| next() as i16).collect();
                for qa in [255, 181, 127, 64] {
                    assert_eq!(
                        unsafe { dot::<true>(&a, &w, qa) },
                        dot_scalar::<true>(&a, &w, qa),
                        "screlu h={} qa={}",
                        h,
                        qa
                    );
                    assert_eq!(
                        unsafe { dot::<false>(&a, &w, qa) },
                        dot_scalar::<false>(&a, &w, qa),
                        "crelu h={} qa={}",
                        h,
                        qa
                    );
                }
            }
        }
    }
}
