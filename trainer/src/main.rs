// Minimal multi-threaded NNUE trainer (std only).
// Net: (768 -> H) x2 perspectives, SCReLU, -> 1 output with NB material buckets.
use std::io::{Read, Write};
use std::time::Instant;

const NI: usize = 768;
static mut NKB: usize = 1; // king buckets (1 = none)
static mut MIRROR: bool = false;

fn kb_of(ks: usize) -> usize {
    // ks: perspective-relative king square
    unsafe {
        if NKB == 1 {
            return 0;
        }
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
const NB: usize = 8;
const QA: f32 = 255.0;
const QB: f32 = 64.0;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

struct Layout {
    h: usize,
    ftw: usize,
    ftb: usize,
    ow: usize,
    ob: usize,
    ftf: usize, // factor weights (768*h) if NKB>1, placed at end
    total: usize,
}
impl Layout {
    fn new(h: usize) -> Layout {
        let ftw = 0;
        let ftb = ftw + NI * unsafe { NKB } * h;
        let ow = ftb + h;
        let ob = ow + NB * 2 * h;
        let ftf = ob + NB;
        let total = if unsafe { NKB } > 1 { ftf + NI * h } else { ftf };
        Layout { h, ftw, ftb, ow, ob, ftf, total }
    }
}

/// A record decode can handle: 2 to 32 pieces (the packed board holds 32),
/// piece codes 0-11 with exactly one king of each colour, result 0-2 and
/// side to move 0-1.
fn valid(r: &[u8; 32]) -> bool {
    let n = u64::from_le_bytes(r[0..8].try_into().unwrap()).count_ones() as usize;
    if !(2..=32).contains(&n) || r[26] > 2 || r[27] > 1 {
        return false;
    }
    let pc = |i: usize| (r[8 + i / 2] >> ((i & 1) * 4)) & 15;
    (0..n).all(|i| pc(i) < 12)
        && (0..n).filter(|&i| pc(i) == 10).count() == 1
        && (0..n).filter(|&i| pc(i) == 11).count() == 1
}

#[inline(always)]
fn decode(r: &[u8; 32], fw: &mut [usize; 32], fb: &mut [usize; 32]) -> (usize, usize, f32, f32, usize) {
    let occ = u64::from_le_bytes(r[0..8].try_into().unwrap());
    let mut o = occ;
    let mut i = 0;
    let mut sqs = [0usize; 32];
    let mut pcs = [0usize; 32];
    let mut wk = 0;
    let mut bk = 0;
    while o != 0 {
        let sq = o.trailing_zeros() as usize;
        o &= o - 1;
        let pc = ((r[8 + i / 2] >> ((i & 1) * 4)) & 15) as usize;
        if pc == 10 {
            wk = sq;
        } else if pc == 11 {
            bk = sq;
        }
        sqs[i] = sq;
        pcs[i] = pc;
        i += 1;
    }
    let nkb = unsafe { NKB };
    let (mw, mb) = if nkb > 1 || unsafe { MIRROR } { ((wk & 7) >= 4, (bk & 7) >= 4) } else { (false, false) };
    let bw = kb_of(wk) * 768;
    let bb = kb_of(bk ^ 56) * 768;
    let xw = if mw { 7 } else { 0 };
    let xb = if mb { 7 } else { 0 };
    for k in 0..i {
        let (sq, pc) = (sqs[k], pcs[k]);
        let pt = pc >> 1;
        let c = pc & 1;
        fw[k] = bw + c * 384 + pt * 64 + (sq ^ xw);
        fb[k] = bb + (c ^ 1) * 384 + pt * 64 + (sq ^ 56 ^ xb);
    }
    let score = i16::from_le_bytes([r[24], r[25]]) as f32;
    let result = r[26] as f32 / 2.0;
    let stm = r[27] as usize;
    let bucket = ((i - 2) / 4).min(NB - 1);
    let (s, res) = if stm == 0 { (score, result) } else { (-score, 1.0 - result) };
    (i, stm, s, res, bucket)
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// forward (+ optional backward into g). returns summed loss.
fn process(data: &[[u8; 32]], idx: &[u32], w: &[f32], g: Option<&mut [f32]>, l: &Layout, lambda: f32) -> f64 {
    let h = l.h;
    let mut fw = [0usize; 32];
    let mut fb = [0usize; 32];
    let mut acc_w = vec![0f32; h];
    let mut acc_b = vec![0f32; h];
    let mut d_us = vec![0f32; h];
    let mut d_th = vec![0f32; h];
    let mut loss = 0f64;
    let mut g = g;
    let ftb = &w[l.ftb..l.ftb + h];
    let fact = l.total > l.ftf;
    for &ix in idx {
        let (n, stm, score, res, bucket) = decode(&data[ix as usize], &mut fw, &mut fb);
        acc_w.copy_from_slice(ftb);
        acc_b.copy_from_slice(ftb);
        for k in 0..n {
            let rw = &w[l.ftw + fw[k] * h..l.ftw + fw[k] * h + h];
            for j in 0..h {
                acc_w[j] += rw[j];
            }
            let rb = &w[l.ftw + fb[k] * h..l.ftw + fb[k] * h + h];
            for j in 0..h {
                acc_b[j] += rb[j];
            }
            if fact {
                let o = l.ftf + (fw[k] % NI) * h;
                let rw = &w[o..o + h];
                for j in 0..h {
                    acc_w[j] += rw[j];
                }
                let o = l.ftf + (fb[k] % NI) * h;
                let rb = &w[o..o + h];
                for j in 0..h {
                    acc_b[j] += rb[j];
                }
            }
        }
        let (us, th, fus, fth) = if stm == 0 { (&acc_w, &acc_b, &fw, &fb) } else { (&acc_b, &acc_w, &fb, &fw) };
        let ow = &w[l.ow + bucket * 2 * h..l.ow + bucket * 2 * h + 2 * h];
        let mut y = w[l.ob + bucket];
        for j in 0..h {
            let a = us[j].clamp(0.0, 1.0);
            let b = th[j].clamp(0.0, 1.0);
            y += a * a * ow[j] + b * b * ow[h + j];
        }
        let p = sigmoid(y);
        let t = lambda * sigmoid(score / 400.0) + (1.0 - lambda) * res;
        let e = p - t;
        loss += (e * e) as f64;
        if let Some(g) = g.as_deref_mut() {
            let dy = 2.0 * e * p * (1.0 - p);
            g[l.ob + bucket] += dy;
            {
                let gow = &mut g[l.ow + bucket * 2 * h..l.ow + bucket * 2 * h + 2 * h];
                for j in 0..h {
                    let a = us[j].clamp(0.0, 1.0);
                    let b = th[j].clamp(0.0, 1.0);
                    gow[j] += dy * a * a;
                    gow[h + j] += dy * b * b;
                    d_us[j] = if us[j] > 0.0 && us[j] < 1.0 { dy * ow[j] * 2.0 * us[j] } else { 0.0 };
                    d_th[j] = if th[j] > 0.0 && th[j] < 1.0 { dy * ow[h + j] * 2.0 * th[j] } else { 0.0 };
                }
            }
            {
                let gfb = &mut g[l.ftb..l.ftb + h];
                for j in 0..h {
                    gfb[j] += d_us[j] + d_th[j];
                }
            }
            for k in 0..n {
                let o = l.ftw + fus[k] * h;
                let r = &mut g[o..o + h];
                for j in 0..h {
                    r[j] += d_us[j];
                }
                let o = l.ftw + fth[k] * h;
                let r = &mut g[o..o + h];
                for j in 0..h {
                    r[j] += d_th[j];
                }
                if fact {
                    let o = l.ftf + (fus[k] % NI) * h;
                    let r = &mut g[o..o + h];
                    for j in 0..h {
                        r[j] += d_us[j];
                    }
                    let o = l.ftf + (fth[k] % NI) * h;
                    let r = &mut g[o..o + h];
                    for j in 0..h {
                        r[j] += d_th[j];
                    }
                }
            }
        }
    }
    loss
}

fn save_quant(path: &str, w: &[f32], l: &Layout) {
    let mut out: Vec<u8> = Vec::new();
    let q16 = |x: f32, s: f32| -> [u8; 2] { ((x * s).round().clamp(-32767.0, 32767.0) as i16).to_le_bytes() };
    for i in 0..NI * unsafe { NKB } * l.h {
        let f = if l.total > l.ftf { w[l.ftf + ((i / l.h) % NI) * l.h + i % l.h] } else { 0.0 };
        out.extend_from_slice(&q16(w[l.ftw + i] + f, QA));
    }
    for i in 0..l.h {
        out.extend_from_slice(&q16(w[l.ftb + i], QA));
    }
    for i in 0..NB * 2 * l.h {
        out.extend_from_slice(&q16(w[l.ow + i], QB));
    }
    for i in 0..NB {
        out.extend_from_slice(&((w[l.ob + i] * QA * QB).round() as i32).to_le_bytes());
    }
    std::fs::write(path, out).unwrap();
}

fn save_float(path: &str, w: &[f32]) {
    let mut out: Vec<u8> = Vec::with_capacity(w.len() * 4);
    for x in w {
        out.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, out).unwrap();
}

fn load_float(path: &str, n: usize) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    assert_eq!(b.len(), n * 4);
    b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn main() {
    // trainer <out_prefix> <hidden> <epochs> <lr> <lambda> <threads> [--init float.bin] files...
    let args: Vec<String> = std::env::args().collect();
    let out = args[1].clone();
    let h: usize = args[2].parse().unwrap();
    let epochs: usize = args[3].parse().unwrap();
    let lr0: f32 = args[4].parse().unwrap();
    let lambda: f32 = args[5].parse().unwrap();
    let threads: usize = args[6].parse().unwrap();
    let mut files = Vec::new();
    let mut init: Option<String> = None;
    let mut valfile: Option<String> = None;
    let mut cosine = false;
    let mut i = 7;
    while i < args.len() {
        if args[i] == "--cosine" {
            cosine = true;
            i += 1;
        } else if args[i] == "--mirror" {
            unsafe {
                MIRROR = true;
            }
            i += 1;
        } else if args[i] == "--valfile" {
            valfile = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--kb" {
            unsafe {
                NKB = args[i + 1].parse().unwrap();
                // kb_of is the 8-bucket table; other counts would index past the FT.
                if NKB != 1 && NKB != 8 {
                    eprintln!("--kb must be 1 or 8 (the king bucket table has 8 buckets)");
                    std::process::exit(1);
                }
            }
            i += 2;
        } else if args[i] == "--init" {
            init = Some(args[i + 1].clone());
            i += 2;
        } else {
            files.push(args[i].clone());
            i += 1;
        }
    }
    let l = Layout::new(h);
    // load data
    let t0 = Instant::now();
    let mut data: Vec<[u8; 32]> = Vec::new();
    for f in &files {
        let mut b = Vec::new();
        std::fs::File::open(f).unwrap().read_to_end(&mut b).unwrap();
        let n = b.len() / 32;
        let before = data.len();
        data.reserve(n);
        data.extend(b.chunks_exact(32).map(|c| <[u8; 32]>::try_from(c).unwrap()).filter(valid));
        if data.len() - before < n {
            println!("{}: skipped {} corrupt records", f, n - (data.len() - before));
        }
    }
    println!("loaded {} positions in {:?}", data.len(), t0.elapsed());
    let mut rng = Rng(0x1234567);
    let mut idx: Vec<u32> = (0..data.len() as u32).collect();
    for k in (1..idx.len()).rev() {
        let j = (rng.next() % (k as u64 + 1)) as usize;
        idx.swap(k, j);
    }
    let (val, mut train): (Vec<u32>, Vec<u32>) = if let Some(vf) = &valfile {
        let n0 = data.len();
        let b = std::fs::read(vf).unwrap();
        data.extend(b.chunks_exact(32).map(|c| <[u8; 32]>::try_from(c).unwrap()).filter(valid));
        let mut v: Vec<u32> = (n0 as u32..data.len() as u32).collect();
        v.truncate(500_000);
        println!("val from {}: {}", vf, v.len());
        (v, idx)
    } else {
        let nval = (idx.len() / 100).min(500_000);
        (idx[..nval].to_vec(), idx[nval..].to_vec())
    };

    let mut w = vec![0f32; l.total];
    if let Some(p) = &init {
        w = load_float(p, l.total);
        println!("init from {}", p);
    } else {
        for k in 0..NI * unsafe { NKB } * h {
            w[l.ftw + k] = if l.total > l.ftf { 0.0 } else { (rng.f() * 2.0 - 1.0) * 0.1 };
        }
        for k in l.ftf..l.total {
            w[k] = (rng.f() * 2.0 - 1.0) * 0.1;
        }
        let a = 1.0 / ((2 * h) as f32).sqrt();
        for k in 0..NB * 2 * h {
            w[l.ow + k] = (rng.f() * 2.0 - 1.0) * a;
        }
    }
    let mut m = vec![0f32; l.total];
    let mut v = vec![0f32; l.total];
    let bs = 16384usize;
    if train.len() < bs || val.is_empty() {
        eprintln!("need at least {} training positions and a non-empty validation set", bs);
        std::process::exit(1);
    }
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let mut step = 0i32;
    let mut grads: Vec<Vec<f32>> = (0..threads).map(|_| vec![0f32; l.total]).collect();
    let mut g = vec![0f32; l.total];
    let clip = 1.98f32;
    if epochs == 0 {
        // validation-only mode: report loss of the --init weights on the validation set
        let chunk = (val.len() + threads - 1) / threads;
        let vl: f64 = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    let (data, val, w, l) = (&data, &val, &w, &l);
                    s.spawn(move || {
                        let lo = (t * chunk).min(val.len());
                        let hi = ((t + 1) * chunk).min(val.len());
                        process(data, &val[lo..hi], w, None, l, lambda)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).sum()
        });
        println!("val {:.6}", vl / val.len() as f64);
        return;
    }
    for ep in 0..epochs {
        let te = Instant::now();
        // shuffle training set
        for k in (1..train.len()).rev() {
            let j = (rng.next() % (k as u64 + 1)) as usize;
            train.swap(k, j);
        }
        // lr schedule: cosine-ish step
        let frac = ep as f32 / epochs as f32;
        let lr_step = if frac < 0.6 {
            lr0
        } else if frac < 0.85 {
            lr0 * 0.3
        } else {
            lr0 * 0.09
        };
        let mut tl = 0f64;
        let nbatches = train.len() / bs;
        let mut last_lr = lr_step;
        for bi in 0..nbatches {
            let lr = if cosine {
                let t = (ep as f32 + bi as f32 / nbatches as f32) / epochs as f32;
                let lmin = lr0 * 0.01;
                lmin + 0.5 * (lr0 - lmin) * (1.0 + (std::f32::consts::PI * t).cos())
            } else {
                lr_step
            };
            last_lr = lr;
            let batch = &train[bi * bs..(bi + 1) * bs];
            let chunk = (bs + threads - 1) / threads;
            let wr = &w;
            let lr_ = &l;
            let losses: Vec<f64> = std::thread::scope(|s| {
                let hs: Vec<_> = grads
                    .iter_mut()
                    .enumerate()
                    .map(|(t, gt)| {
                        let data = &data;
                        s.spawn(move || {
                            gt.iter_mut().for_each(|x| *x = 0.0);
                            let lo = (t * chunk).min(bs);
                            let hi = ((t + 1) * chunk).min(bs);
                            process(data, &batch[lo..hi], wr, Some(gt), lr_, lambda)
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            tl += losses.iter().sum::<f64>();
            // reduce + adam, parallel over parameter chunks
            step += 1;
            let bc1 = 1.0 - b1.powi(step);
            let bc2 = 1.0 - b2.powi(step);
            let pchunk = (l.total + threads - 1) / threads;
            let gr = &grads;
            std::thread::scope(|s| {
                let mut wrest: &mut [f32] = &mut w;
                let mut mrest: &mut [f32] = &mut m;
                let mut vrest: &mut [f32] = &mut v;
                let mut grest: &mut [f32] = &mut g;
                let mut off = 0;
                while !wrest.is_empty() {
                    let n = pchunk.min(wrest.len());
                    let (wc, wr2) = wrest.split_at_mut(n);
                    let (mc, mr2) = mrest.split_at_mut(n);
                    let (vc, vr2) = vrest.split_at_mut(n);
                    let (gc, gr2) = grest.split_at_mut(n);
                    wrest = wr2;
                    mrest = mr2;
                    vrest = vr2;
                    grest = gr2;
                    let o = off;
                    off += n;
                    s.spawn(move || {
                        for k in 0..n {
                            let mut sum = 0f32;
                            for t in 0..gr.len() {
                                sum += gr[t][o + k];
                            }
                            gc[k] = sum / bs as f32;
                        }
                        for k in 0..n {
                            let gk = gc[k];
                            mc[k] = b1 * mc[k] + (1.0 - b1) * gk;
                            vc[k] = b2 * vc[k] + (1.0 - b2) * gk * gk;
                            let upd = (mc[k] / bc1) / ((vc[k] / bc2).sqrt() + eps);
                            wc[k] = (wc[k] - lr * upd).clamp(-clip, clip);
                        }
                    });
                }
            });
            if bi % 500 == 0 && bi > 0 {
                println!("  ep {} batch {}/{} loss {:.6}", ep, bi, nbatches, tl / ((bi + 1) * bs) as f64);
            }
        }
        // validation
        let chunk = (val.len() + threads - 1) / threads;
        let vl: f64 = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    let (data, val, w, l) = (&data, &val, &w, &l);
                    s.spawn(move || {
                        let lo = (t * chunk).min(val.len());
                        let hi = ((t + 1) * chunk).min(val.len());
                        process(data, &val[lo..hi], w, None, l, lambda)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).sum()
        });
        println!(
            "epoch {} lr {} train {:.6} val {:.6} time {:?}",
            ep,
            last_lr,
            tl / (nbatches * bs) as f64,
            vl / val.len() as f64,
            te.elapsed()
        );
        std::io::stdout().flush().unwrap();
        save_float(&format!("{}.f32", out), &w);
        save_quant(&format!("{}.nnue", out), &w, &l);
    }
}
