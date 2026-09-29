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
pub const TAG_DESCRIPTION: u16 = 0x8001;

pub const INPUT_PSQ768: u16 = 1;
pub const ACT_NONE: u8 = 0;
pub const ACT_CRELU: u8 = 2;
pub const ACT_SCRELU: u8 = 3;
pub const TYPE_I16: u8 = 2;
pub const TYPE_I32: u8 = 3;
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
    let (h, nkb, nb) = (arch.hidden, arch.num_king_buckets(), arch.output_buckets);
    assert_eq!(ftw.len(), nkb * 768 * h);
    assert_eq!(ftb.len(), h);
    assert_eq!(ow.len(), nb * 2 * h);
    assert_eq!(ob.len(), nb);

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

    let mut p = ((2 * h) as u32).to_le_bytes().to_vec();
    p.extend_from_slice(&1u32.to_le_bytes());
    p.extend_from_slice(&[ACT_NONE, TYPE_I16, TYPE_I32, 1]);
    field(&mut fields, TAG_LAYER, &p);

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
    let mut out = Vec::with_capacity(header_size + 2 * (ftw.len() + ftb.len() + ow.len()) + 4 * ob.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(header_size as u32).to_le_bytes());
    out.extend_from_slice(&fields);
    out.resize(header_size, 0);
    for v in ftw.iter().chain(ftb).chain(ow) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in ob {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn i16s(bytes: &[u8]) -> Vec<i16> {
    bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

/// Reads quantised weights in `source` layout ("bullet" or "raw") and returns
/// the new-format file.
pub fn convert(arch: &Arch, source: &str, data: &[u8]) -> Result<Vec<u8>, String> {
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
