// Incipit network format (docs/net-format.md): writer and reader.
//
// `convert` builds a network file from quantised weights in one of two source
// layouts, both [l0w][l0b][l1w][l1b] with l0w as [bucket][feature][hidden] and
// l1w as [output bucket][2 * hidden]:
//   * bullet: Bullet's quantised.bin (l1b as i16, padded to 64 bytes);
//   * raw:    the engine's previous headerless format (l1b as i32).
use std::fmt::Write as _;

pub const MAGIC: &[u8; 8] = b"INCPTNET";
pub const VERSION: u16 = 1;

pub const TAG_INPUTS: u16 = 0x0001;
pub const TAG_KING_BUCKETS: u16 = 0x0002;
pub const TAG_FT: u16 = 0x0003;
pub const TAG_OUTPUT_BUCKETS: u16 = 0x0004;
pub const TAG_LAYER: u16 = 0x0005;
pub const TAG_QUANT: u16 = 0x0006;
pub const TAG_LAYER_QUANT: u16 = 0x0007;
pub const TAG_DESCRIPTION: u16 = 0x8001;

pub const INPUT_PSQ768: u16 = 1;
pub const ACT_NONE: u8 = 0;
pub const ACT_CRELU: u8 = 2;
pub const ACT_SCRELU: u8 = 3;
/// Pairwise CReLU FT: per perspective, first half times second half (h/2 values).
pub const ACT_PAIRWISE: u8 = 4;
pub const TYPE_I8: u8 = 1;
pub const TYPE_I16: u8 = 2;
pub const TYPE_I32: u8 = 3;
pub const TYPE_F32: u8 = 4;
pub const OUTPUT_MATERIAL: u8 = 1;

#[derive(Clone, Debug)]
pub struct Arch {
    pub hidden: usize,
    pub king_buckets: [u8; 64],
    pub mirror: bool,
    pub activation: u8,
    pub output_buckets: usize,
    pub qa: i32,
    pub qb: i32,
    pub scale: i32,
    pub description: String,
    /// Neurons in one hidden layer after the FT (0 = none).
    pub l1: usize,
    /// The hidden layer is shared by all output buckets (only the final
    /// layer is per bucket).
    pub l1_shared: bool,
    /// The second hidden layer takes SCReLU then CReLU of the first (2 * l1 inputs).
    pub l1_dual: bool,
    /// The hidden layer's last neuron is a linear skip: no activation, added
    /// to the output; the second hidden layer sees the other l1 - 1.
    pub l1_skip: bool,
    /// FT neuron order for a net with a hidden layer: new neuron k is old
    /// neuron perm[k] (empty = unchanged). Reordering the FT columns and the
    /// matching hidden-layer inputs together leaves every eval unchanged; the
    /// engine's sparse product is faster when rarely active neurons share
    /// 4-neuron groups.
    pub perm: Vec<usize>,
    /// Hidden-layer input shift: u8 input = clamp(a, 0, QA)^2 >> shift
    /// (0 = the smallest shift that fits 0..127).
    pub l1_shift: u32,
    /// Neurons in a second hidden layer (f32, SCReLU, per bucket; 0 = none).
    pub l2: usize,
}

impl Arch {
    pub fn num_king_buckets(&self) -> usize {
        *self.king_buckets.iter().max().unwrap() as usize + 1
    }
}

fn field(out: &mut Vec<u8>, tag: u16, payload: &[u8]) {
    out.extend_from_slice(&tag.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
}

/// Serialises a network: header, then weights in the documented order.
/// `ftw`, `ftb`, `ow` are i16; `ob` is i32 (one per output bucket).
pub fn write(arch: &Arch, ftw: &[i16], ftb: &[i16], ow: &[i16], ob: &[i32]) -> Vec<u8> {
    let (h, nb) = (arch.hidden, arch.output_buckets);
    assert_eq!(ow.len(), nb * 2 * h);
    assert_eq!(ob.len(), nb);
    let mut p = ((2 * h) as u32).to_le_bytes().to_vec();
    p.extend_from_slice(&1u32.to_le_bytes());
    p.extend_from_slice(&[ACT_NONE, TYPE_I16, TYPE_I32, 1]);
    let mut weights = Vec::new();
    for v in ow {
        weights.extend_from_slice(&v.to_le_bytes());
    }
    for v in ob {
        weights.extend_from_slice(&v.to_le_bytes());
    }
    assemble(arch, ftw, ftb, &[p], None, &weights)
}

/// Input shift for the hidden layer's u8 inputs: the smallest s with
/// QA^2 >> s <= 127, so the int8 kernels can't overflow.
pub fn hidden_shift(qa: i32) -> u32 {
    let sq = qa as i64 * qa as i64;
    (0..32).find(|&s| (sq >> s) <= 127).unwrap()
}

/// Serialises a network with one hidden layer: FT -> l1 (SCReLU, i8 weights
/// at QB, f32 biases; per output bucket unless `l1_shared`) -> 1 (f32, per
/// bucket). `w1` is [bucket][out][2h] (or [out][2h]); `b1` is [bucket][out]
/// (or [out]); `w2` is [bucket][out]; `b2` is [bucket].
/// With a second hidden layer (`arch.l2` > 0), `mid` holds its weights
/// [bucket][l2][l1] and biases [bucket][l2], and `w2` is [bucket][l2].
#[allow(clippy::too_many_arguments)]
pub fn write_hidden(
    arch: &Arch,
    ftw: &[i16],
    ftb: &[i16],
    w1: &[i8],
    b1: &[f32],
    mid: Option<(&[f32], &[f32])>,
    w2: &[f32],
    b2: &[f32],
) -> Vec<u8> {
    let (h, nb, l1, l2) = (arch.hidden, arch.output_buckets, arch.l1, arch.l2);
    let nb1 = if arch.l1_shared { 1 } else { nb };
    let last = if l2 > 0 { l2 } else { l1 };
    // Hidden-layer inputs: 2h (SCReLU) or h (pairwise).
    let inl = if arch.activation == ACT_PAIRWISE { h } else { 2 * h };
    assert_eq!(w1.len(), nb1 * l1 * inl);
    assert_eq!(b1.len(), nb1 * l1);
    assert_eq!(w2.len(), nb * last);
    assert_eq!(b2.len(), nb);
    assert_eq!(mid.is_some(), l2 > 0);
    let mut p1 = (inl as u32).to_le_bytes().to_vec();
    p1.extend_from_slice(&(l1 as u32).to_le_bytes());
    p1.extend_from_slice(&[ACT_SCRELU, TYPE_I8, TYPE_F32, !arch.l1_shared as u8 | (arch.l1_skip as u8) << 1]);
    let in2 = if arch.l1_dual { 2 * l1 } else if arch.l1_skip { l1 - 1 } else { l1 };
    let mut pm = (in2 as u32).to_le_bytes().to_vec();
    pm.extend_from_slice(&(l2 as u32).to_le_bytes());
    pm.extend_from_slice(&[ACT_SCRELU, TYPE_F32, TYPE_F32, 1]);
    let mut p2 = (last as u32).to_le_bytes().to_vec();
    p2.extend_from_slice(&1u32.to_le_bytes());
    p2.extend_from_slice(&[ACT_NONE, TYPE_F32, TYPE_F32, 1]);
    let mut lq = Vec::new();
    let shift = if arch.l1_shift > 0 { arch.l1_shift } else { hidden_shift(arch.qa) };
    for v in [shift as i32, arch.qb, 0, 0] {
        lq.extend_from_slice(&v.to_le_bytes());
    }
    let mut weights: Vec<u8> = w1.iter().map(|&v| v as u8).collect();
    let (wm, bm): (&[f32], &[f32]) = mid.unwrap_or((&[], &[]));
    for v in b1.iter().chain(wm).chain(bm).chain(w2).chain(b2) {
        weights.extend_from_slice(&v.to_le_bytes());
    }
    if l2 > 0 {
        for v in [0i32, 0] {
            lq.extend_from_slice(&v.to_le_bytes());
        }
        assemble(arch, ftw, ftb, &[p1, pm, p2], Some(&lq), &weights)
    } else {
        assemble(arch, ftw, ftb, &[p1, p2], Some(&lq), &weights)
    }
}

/// Header plus FT weights, then the given layer records and their weights.
fn assemble(arch: &Arch, ftw: &[i16], ftb: &[i16], layers: &[Vec<u8>], layer_quant: Option<&[u8]>, weights: &[u8]) -> Vec<u8> {
    let (h, nkb, nb) = (arch.hidden, arch.num_king_buckets(), arch.output_buckets);
    assert_eq!(ftw.len(), nkb * 768 * h);
    assert_eq!(ftb.len(), h);

    let mut fields = Vec::new();
    let mut p = Vec::new();
    p.extend_from_slice(&1u16.to_le_bytes());
    p.extend_from_slice(&INPUT_PSQ768.to_le_bytes());
    p.extend_from_slice(&0u16.to_le_bytes());
    p.extend_from_slice(&768u32.to_le_bytes());
    field(&mut fields, TAG_INPUTS, &p);

    let mut p = vec![nkb as u8, arch.mirror as u8, 0, 0];
    p.extend_from_slice(&arch.king_buckets);
    field(&mut fields, TAG_KING_BUCKETS, &p);

    let mut p = (h as u32).to_le_bytes().to_vec();
    p.extend_from_slice(&[arch.activation, TYPE_I16, TYPE_I16, 0]);
    field(&mut fields, TAG_FT, &p);

    field(&mut fields, TAG_OUTPUT_BUCKETS, &[OUTPUT_MATERIAL, nb as u8, 0, 0]);

    for p in layers {
        field(&mut fields, TAG_LAYER, p);
    }
    if let Some(lq) = layer_quant {
        field(&mut fields, TAG_LAYER_QUANT, lq);
    }

    let mut p = Vec::new();
    for v in [arch.qa, arch.qb, arch.scale] {
        p.extend_from_slice(&v.to_le_bytes());
    }
    field(&mut fields, TAG_QUANT, &p);

    if !arch.description.is_empty() {
        field(&mut fields, TAG_DESCRIPTION, arch.description.as_bytes());
    }
    fields.extend_from_slice(&[0; 8]); // end tag

    let header_size = (16 + fields.len()).div_ceil(64) * 64;
    let mut out = Vec::with_capacity(header_size + 2 * (ftw.len() + ftb.len()) + weights.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(header_size as u32).to_le_bytes());
    out.extend_from_slice(&fields);
    out.resize(header_size, 0);
    for v in ftw.iter().chain(ftb) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(weights);
    out
}

fn i16s(bytes: &[u8]) -> Vec<i16> {
    bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

/// Reads quantised weights in `source` layout ("bullet" or "raw") and returns
/// the new-format file.
pub fn convert(arch: &Arch, source: &str, data: &[u8]) -> Result<Vec<u8>, String> {
    if arch.l1 > 0 {
        return convert_hidden(arch, source, data);
    }
    let (h, nkb, nb) = (arch.hidden, arch.num_king_buckets(), arch.output_buckets);
    let n_ftw = nkb * 768 * h;
    let n_ow = nb * 2 * h;
    let bias_size = match source {
        "bullet" => 2,
        "raw" => 4,
        _ => return Err(format!("unknown source layout {:?} (use bullet or raw)", source)),
    };
    let expect = 2 * (n_ftw + h + n_ow) + bias_size * nb;
    let ok = match source {
        // Bullet pads to a multiple of 64 bytes with the repeating text "bullet".
        "bullet" => {
            data.len() == expect.div_ceil(64) * 64
                && (data[expect..].iter().all(|&b| b == 0)
                    || data[expect..].iter().zip(b"bullet".iter().cycle()).all(|(a, b)| a == b))
        }
        _ => data.len() == expect,
    };
    if !ok {
        return Err(format!(
            "input is {} bytes, but hidden {}, {} king bucket(s) and {} output buckets need {} ({} layout)",
            data.len(), h, nkb, nb, expect, source
        ));
    }
    let mut o = 0;
    let mut take = |n: usize| {
        let s = &data[o..o + n];
        o += n;
        s
    };
    let ftw = i16s(take(2 * n_ftw));
    let ftb = i16s(take(2 * h));
    let ow = i16s(take(2 * n_ow));
    let ob: Vec<i32> = if bias_size == 2 {
        i16s(take(2 * nb)).into_iter().map(i32::from).collect()
    } else {
        take(4 * nb).chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
    };
    Ok(write(arch, &ftw, &ftb, &ow, &ob))
}

/// Bullet's quantised.bin for a net with one hidden layer: [l0w i16][l0b i16]
/// [l1w i8, [bucket * l1][2h]][l1b f32][l2w f32, [bucket][l1]][l2b f32], padded
/// to 64 bytes.
fn convert_hidden(arch: &Arch, source: &str, data: &[u8]) -> Result<Vec<u8>, String> {
    if source != "bullet" {
        return Err("--l1 needs a bullet source".into());
    }
    let (h, nkb, nb, l1) = (arch.hidden, arch.num_king_buckets(), arch.output_buckets, arch.l1);
    let nb1 = if arch.l1_shared { 1 } else { nb };
    let n_ftw = nkb * 768 * h;
    let l2 = arch.l2;
    let last = if l2 > 0 { l2 } else { l1 };
    let pw = arch.activation == ACT_PAIRWISE;
    if pw && h % 128 != 0 {
        return Err(format!("pairwise needs a hidden size that is a multiple of 128, got {}", h));
    }
    let inl = if pw { h } else { 2 * h };
    if arch.l1_skip && (l2 == 0 || arch.l1_dual) {
        return Err("--l1-skip needs --l2 and no --l1-dual".into());
    }
    let in2 = if arch.l1_dual { 2 * l1 } else if arch.l1_skip { l1 - 1 } else { l1 };
    let expect = 2 * (n_ftw + h) + nb1 * l1 * inl + 4 * (nb1 * l1 + nb * l2 * (in2 + 1) + nb * last + nb);
    let ok = data.len() == expect.div_ceil(64) * 64
        && (data[expect..].iter().all(|&b| b == 0) || data[expect..].iter().zip(b"bullet".iter().cycle()).all(|(a, b)| a == b));
    if !ok {
        return Err(format!(
            "input is {} bytes, but hidden {}, {} king bucket(s), {} output buckets and l1 {} need {}",
            data.len(), h, nkb, nb, l1, expect
        ));
    }
    let mut o = 0;
    let mut take = |n: usize| {
        let s = &data[o..o + n];
        o += n;
        s
    };
    let ftw = i16s(take(2 * n_ftw));
    let ftb = i16s(take(2 * h));
    let w1: Vec<i8> = take(nb1 * l1 * inl).iter().map(|&b| b as i8).collect();
    let f32s = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
    let b1 = f32s(take(4 * nb1 * l1));
    let wm = f32s(take(4 * nb * l2 * in2));
    let bm = f32s(take(4 * nb * l2));
    let w2 = f32s(take(4 * nb * last));
    let b2 = f32s(take(4 * nb));
    let mid = (l2 > 0).then_some((&wm[..], &bm[..]));
    if arch.perm.is_empty() {
        return Ok(write_hidden(arch, &ftw, &ftb, &w1, &b1, mid, &w2, &b2));
    }
    // The order covers the hidden layer's inputs per perspective: h neurons, or
    // for pairwise nets h/2 pairs (neurons j and j + h/2 move together).
    let p = &arch.perm;
    let u = if pw { h / 2 } else { h };
    let mut seen = vec![false; u];
    if p.len() != u || !p.iter().all(|&i| i < u && !std::mem::replace(&mut seen[i], true)) {
        return Err(format!("--permute: need a permutation of 0..{}", u));
    }
    let full: Vec<usize> = if pw { p.iter().copied().chain(p.iter().map(|&i| i + u)).collect() } else { p.clone() };
    let ftw: Vec<i16> = ftw.chunks_exact(h).flat_map(|row| full.iter().map(move |&i| row[i])).collect();
    let ftb: Vec<i16> = full.iter().map(|&i| ftb[i]).collect();
    let w1: Vec<i8> = w1
        .chunks_exact(2 * u)
        .flat_map(|row| p.iter().map(move |&i| row[i]).chain(p.iter().map(move |&i| row[u + i])))
        .collect();
    Ok(write_hidden(arch, &ftw, &ftb, &w1, &b1, mid, &w2, &b2))
}

/// Describes a network file's header (and checks its size), for `net-info`.
pub fn describe(data: &[u8]) -> Result<String, String> {
    if data.len() < 16 || &data[0..8] != MAGIC {
        return Err("not an Incipit network (bad magic)".into());
    }
    let u16_at = |o: usize| u16::from_le_bytes([data[o], data[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
    let header_size = u32_at(12) as usize;
    let mut s = format!("version {}, header {} bytes, weights {} bytes\n", u16_at(8), header_size, data.len() - header_size);
    let mut o = 16;
    while o + 8 <= header_size {
        let (tag, len) = (u16_at(o), u32_at(o + 4) as usize);
        if tag == 0 {
            break;
        }
        let p = &data[o + 8..o + 8 + len];
        let p32 = |i: usize| u32::from_le_bytes(p[i..i + 4].try_into().unwrap());
        match tag {
            TAG_INPUTS => {
                let _ = writeln!(s, "inputs: {} set(s), first kind {} with {} features", u16::from_le_bytes([p[0], p[1]]), u16::from_le_bytes([p[2], p[3]]), p32(6));
            }
            TAG_KING_BUCKETS => {
                let _ = writeln!(s, "king buckets: {}, mirrored: {}, table {:?}", p[0], p[1] != 0, &p[4..68]);
            }
            TAG_FT => {
                let _ = writeln!(s, "feature transformer: hidden {}, activation {}, weight type {}, bias type {}", p32(0), p[4], p[5], p[6]);
            }
            TAG_OUTPUT_BUCKETS => {
                let _ = writeln!(s, "output buckets: scheme {}, {} buckets", p[0], p[1]);
            }
            TAG_LAYER => {
                let _ = writeln!(s, "layer: {} -> {}, activation {}, weight type {}, bias type {}, flags {}", p32(0), p32(4), p[8], p[9], p[10], p[11]);
            }
            TAG_QUANT => {
                let _ = writeln!(s, "quantisation: QA {}, QB {}, scale {}", p32(0) as i32, p32(4) as i32, p32(8) as i32);
            }
            TAG_LAYER_QUANT => {
                let per: Vec<String> = (0..len / 8).map(|i| format!("(input shift {}, weight scale {})", p32(8 * i), p32(8 * i + 4) as i32)).collect();
                let _ = writeln!(s, "layer quantisation: {}", per.join(", "));
            }
            TAG_DESCRIPTION => {
                let _ = writeln!(s, "description: {}", String::from_utf8_lossy(p));
            }
            other => {
                let _ = writeln!(s, "field 0x{:04x}: {} bytes", other, len);
            }
        }
        o += 8 + len;
    }
    Ok(s)
}
