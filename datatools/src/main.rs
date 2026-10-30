// Data tools for Incipit: converts OpenBench datagen PGNs and the engine's old
// datagen `.bin` files to viriformat, summarises viriformat files, and
// converts trained networks to Incipit's network format (docs/net-format.md).
//
// The board code is the engine's own, included directly so this tool needs
// neither the engine's net nor any dependency.
#[allow(dead_code)]
#[path = "../../engine/src/attacks.rs"]
mod attacks;
#[allow(dead_code)]
#[path = "../../engine/src/position.rs"]
mod position;
#[allow(dead_code)]
#[path = "../../engine/src/san.rs"]
mod san;

/// The engine's search switches read by the shared engine code above
/// (position.rs: pin-aware SEE). Datatools uses the engine defaults.
mod params {
    pub enum P {
        UsePinSee,
    }
    pub fn on(_: P) -> bool {
        true
    }
}
#[allow(dead_code)]
#[path = "../../engine/src/pawnpairs.rs"]
mod pawnpairs;
#[allow(dead_code)]
#[path = "../../engine/src/tb.rs"]
mod tb;
#[allow(dead_code)]
#[path = "../../engine/src/threats.rs"]
mod threats;

mod netfmt;
mod oldbin;
mod pgn;
#[cfg(test)]
mod tests;
mod viri;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const USAGE: &str = "\
usage:
  datatools pgn <out.vf> [--tb-label CP] <in.pgn | -> ...
                                              OpenBench/fastchess PGNs -> viriformat (--tb-label: tablebase
                                              win/loss scores become +-CP labels instead of unscored)
                                              (- reads stdin, e.g. bzcat *.pgn.bz2 | ...)
  datatools bin <out_dir> <in.bin> ...        old datagen .bin -> viriformat, one
                                              <out_dir>/<name>.vf per input, in parallel
  datatools stats <file.vf> ...               count games and positions
  datatools heldout <every> <in.vf> ...       every Nth training-eligible position (quiet, not in check, ply >= 16,
                                              scored) as FEN, game result for the side to move, score (TSV)
  datatools plystats <in.vf> ...              positions per ply and how many are distinct (by hash),
                                              to see where the ply filter removes duplicates
  datatools structstats <in.vf> ...           king safety and pawn-structure rates by game phase (ply bands),
                                              to compare our self-play with normal games
  datatools tbstats <tb_dir> <in.vf> ...      games reaching the Syzygy tables: game result vs tablebase result
  datatools tbrelabel <tb_dir> <out_dir> [--fix-result] [--score CP | --score-dtz LO HI PER_PAWN | --score-keep CP] <in.vf> ...
                                              relabel with the Syzygy tables: --fix-result sets each game's
                                              result to the table result at its first table position with a
                                              fresh fifty-move clock (as tbstats); --score CP sets the score of
                                              every scored table position to +-CP (win/loss, side to move) or 0
                                              (draw); --score-dtz grades wins from HI at DTZ 0 to LO at DTZ 100
                                              plies, plus PER_PAWN per pawn of material lead; --score-keep CP
                                              keeps the search score when it shows the win by >= CP, else +-CP
  datatools wdlstats <out.csv> <in.vf> ...    win/draw/loss counts by (material, eval), side to move,
                                              for fitting a WDL model (from ply 16, not in check, scored)
  datatools wdlfit <stats.csv>                 fit the WDL model to wdlstats output (maximum likelihood)
  datatools rescale <factor> <out_dir> <in.vf> ...
                                              multiply every score by <factor> (unscored positions
                                              unchanged), writing <out_dir>/<same name> per input
  datatools calibrate <a0,a1,a2,a3> <b0,b1,b2,b3> <out_dir> <in.vf> ...
                                              replace each score x by 400 logit(E(x)), E the expected
                                              score under the set's WDL model (wdlfit's a and b), so
                                              sigmoid(score / 400) is the label's calibrated
                                              probability; unscored positions unchanged
                                              (writes <out_dir>/CALIBRATED, which incipit-train-v2 requires)
  datatools bishopstats <out.csv> <in.vf> ... one bishop a side: opposite- vs same-coloured, score and result
                                              by pawn lead (side to move), to check how nets score OCB endings
  datatools matstats <out.csv> <in.vf> ...    stored score and result by material lead, side to move,
                                              to check labels in decided positions (quiet moves only)
  datatools net <source> <in> <out_dir> [options]
                                              convert a network to Incipit's format,
                                              writing <out_dir>/net-XXXXXXXX.nnue
      <source>: bullet (Bullet quantised.bin) or raw (the engine's old format)
      --hidden N           hidden size (required)
      --king-buckets LIST  32 (mirrored half) or 64 comma-separated bucket numbers;
                           default: one bucket
      --mirror             features mirrored when the king is on files e-h
      --output-buckets N   material-count output buckets (default 8)
      --activation A       screlu (default), crelu or pairwise (with --l1: pairwise FT)
      --qa N --qb N --scale N   quantisation (defaults 255, 64, 400); with --l1,
                           QB is the hidden layer's int8 weight scale, which
                           the trainer always saves at 128 (the default then)
      --l1 N               one hidden layer of N neurons after the FT (bullet only)
      --l1-shared          the hidden layer is shared by all output buckets
      --l1-dual            the second hidden layer takes SCReLU and CReLU of the first (needs --l2)
      --l1-lin N           the first N hidden neurons' pre-activations also feed the output (trainer l1lin=N)
      --pawn-pairs         1488 pawn-pair input rows after the king buckets' rows (trainer pp=1)
      --pawn-pair-window N pawn pairs in window N: 0 = 1488 rows (as --pawn-pairs), 1 = 928
                           (same file any distance, adjacent files within one rank), 2 = 608
                           (within one rank) (trainer pp=1 ppw=N)
      --threats V          threat inputs of variant V (0, 1, 2; trainer thr=V): i8 rows after all layers
      --l1-skip            the hidden layer's last neuron is a linear skip to the output (trainer l1skip=1; needs --l2)
      --description TEXT   training run, data and settings
      --permute FILE       FT neuron order (hidden-layer nets; from the engine's l1perm)
  datatools net-info <file.nnue> ...          show a network's header
  datatools fens <count> <seed> <file.vf> ... sample undecided positions across game phases
  datatools pgnmoves <in.pgn> ...          per move: game, ply, mover, FEN before, UCI move, comment (TSV)
                                              and search-stressing kinds (for the bench set)";

fn main() {
    attacks::init();
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("pgn") if args.len() >= 4 => cmd_pgn(&args[2], &args[3..]),
        Some("bin") if args.len() >= 4 => cmd_bin(&args[2], &args[3..]),
        Some("stats") if args.len() >= 3 => cmd_stats(&args[2..]),
        Some("heldout") if args.len() >= 4 => cmd_heldout(&args[2], &args[3..]),
        Some("plystats") if args.len() >= 3 => cmd_plystats(&args[2..]),
        Some("structstats") if args.len() >= 3 => cmd_structstats(&args[2..]),
        Some("tbstats") if args.len() >= 4 => cmd_tbstats(&args[2], &args[3..]),
        Some("tbrelabel") if args.len() >= 5 => cmd_tbrelabel(&args[2], &args[3], &args[4..]),
        Some("wdlstats") if args.len() >= 4 => cmd_wdlstats(&args[2], &args[3..]),
        Some("piecestats") if args.len() >= 3 => cmd_piecestats(&args[2..]),
        Some("threatstats") if args.len() >= 4 => cmd_threatstats(&args[2], &args[3..]),
        Some("matstats") if args.len() >= 4 => cmd_matstats(&args[2], &args[3..]),
        Some("bishopstats") if args.len() >= 4 => cmd_bishopstats(&args[2], &args[3..]),
        Some("wdlfit") if args.len() >= 3 => cmd_wdlfit(&args[2]),
        Some("rescale") if args.len() >= 5 => cmd_rescale(&args[2], &args[3], &args[4..]),
        Some("calibrate") if args.len() >= 6 => cmd_calibrate(&args[2], &args[3], &args[4], &args[5..]),
        Some("net") if args.len() >= 5 => cmd_net(&args[2], &args[3], &args[4], &args[5..]),
        Some("net-info") if args.len() >= 3 => cmd_net_info(&args[2..]),
        Some("fens") if args.len() >= 5 => cmd_fens(&args[2], &args[3], &args[4..]),
        Some("pgnmoves") if args.len() >= 3 => cmd_pgnmoves(&args[2..]),
        _ => Err(USAGE.to_string()),
    };
    if let Err(e) = result {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}

fn open_input(path: &str) -> Result<Box<dyn std::io::BufRead>, String> {
    if path == "-" {
        Ok(Box::new(std::io::BufReader::with_capacity(1 << 20, std::io::stdin())))
    } else {
        let f = File::open(path).map_err(|e| format!("{}: {}", path, e))?;
        Ok(Box::new(std::io::BufReader::with_capacity(1 << 20, f)))
    }
}

fn cmd_pgn(out: &str, args: &[String]) -> Result<(), String> {
    // --tb-label CP: tablebase scores become +-CP labels (else unscored).
    let mut inputs: Vec<String> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--tb-label" {
            let cp: i32 = it.next().and_then(|v| v.parse().ok()).ok_or("--tb-label needs a cp value")?;
            if !(1..19000).contains(&cp) {
                return Err(format!("--tb-label {} out of range 1..19000", cp));
            }
            pgn::TB_LABEL.store(cp, std::sync::atomic::Ordering::Relaxed);
        } else {
            inputs.push(a.clone());
        }
    }
    let inputs = &inputs;
    let mut w = BufWriter::new(File::create(out).map_err(|e| format!("{}: {}", out, e))?);
    let (mut games, mut moves, mut skipped) = (0u64, 0u64, 0u64);
    for input in inputs {
        for g in pgn::Games::new(open_input(input)?) {
            let g = g.map_err(|e| format!("{}: {}", input, e))?;
            match pgn::to_game(&g) {
                Ok(game) => {
                    game.write(&mut w).map_err(|e| e.to_string())?;
                    games += 1;
                    moves += game.moves.len() as u64;
                }
                Err(reason) => {
                    skipped += 1;
                    if skipped <= 10 {
                        eprintln!("skipping game: {}", reason);
                    }
                }
            }
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    println!("{}: {} games, {} positions ({} games skipped)", out, games, moves, skipped);
    Ok(())
}

fn cmd_bin(out_dir: &str, inputs: &[String]) -> Result<(), String> {
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {}", out_dir, e))?;
    let next = AtomicUsize::new(0);
    let total = Mutex::new(oldbin::Stats::default());
    let errors = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(inputs.len());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(input) = inputs.get(i) else { break };
                let stem = Path::new(input).file_stem().unwrap_or_default().to_string_lossy();
                let out = Path::new(out_dir).join(format!("{}.vf", stem));
                let run = || -> std::io::Result<oldbin::Stats> {
                    let data = std::fs::read(input)?;
                    let mut w = BufWriter::new(File::create(&out)?);
                    let mut stats = oldbin::Stats::default();
                    let mut err = Ok(());
                    oldbin::rebuild(&data, &mut stats, |g| {
                        if err.is_ok() {
                            err = g.write(&mut w);
                        }
                    });
                    err?;
                    w.flush()?;
                    Ok(stats)
                };
                match run() {
                    Ok(st) => {
                        println!(
                            "{} -> {}: {} records, {} games, {} positions",
                            input,
                            out.display(),
                            st.records,
                            st.games,
                            st.moves
                        );
                        let mut t = total.lock().unwrap();
                        t.records += st.records;
                        t.games += st.games;
                        t.moves += st.moves;
                        t.unscored += st.unscored;
                        t.dropped += st.dropped;
                        t.restarts += st.restarts;
                        for (a, b) in t.gaps.iter_mut().zip(st.gaps) {
                            *a += b;
                        }
                    }
                    Err(e) => errors.lock().unwrap().push(format!("{}: {}", input, e)),
                }
            });
        }
    });
    let t = total.into_inner().unwrap();
    let scored = t.moves - t.unscored;
    println!(
        "total: {} records -> {} games, {} positions ({} scored = {:.1}% of records, {} unscored bridging moves)",
        t.records,
        t.games,
        t.moves,
        scored,
        100.0 * scored as f64 / t.records.max(1) as f64,
        t.unscored
    );
    let bridges: Vec<String> = (1..=oldbin::MAX_GAP).map(|d| format!("{}: {}", d, t.gaps[d])).collect();
    println!("bridge lengths (plies): {}", bridges.join(", "));
    println!("chain restarts within a game: {}; records without a move: {}", t.restarts, t.dropped);
    let errors = errors.into_inner().unwrap();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}

/// <out_dir>/<file name> for each input, refusing two inputs with the same
/// file name (one would silently overwrite the other) and an output that is
/// its own input.
fn output_paths(out_dir: &str, inputs: &[String]) -> Result<Vec<std::path::PathBuf>, String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(inputs.len());
    for input in inputs {
        let name = Path::new(input).file_name().ok_or_else(|| format!("{}: no file name", input))?;
        if !seen.insert(name.to_os_string()) {
            return Err(format!("two inputs are named {}: their outputs would collide", name.to_string_lossy()));
        }
        let dest = Path::new(out_dir).join(name);
        if let (Ok(a), Ok(b)) = (std::fs::canonicalize(input), std::fs::canonicalize(&dest)) {
            if a == b {
                return Err(format!("{}: the output would overwrite the input", input));
            }
        }
        out.push(dest);
    }
    Ok(out)
}

fn cmd_rescale(factor: &str, out_dir: &str, inputs: &[String]) -> Result<(), String> {
    let factor: f64 = factor.parse().map_err(|_| format!("bad factor {}", factor))?;
    if !(factor > 0.0 && factor.is_finite()) {
        return Err(format!("factor must be positive, got {}", factor));
    }
    let dests = output_paths(out_dir, inputs)?;
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {}", out_dir, e))?;
    for (input, dest) in inputs.iter().zip(&dests) {
        let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
        let out = viri::rescale(&data, factor).map_err(|e| format!("{}: {}", input, e))?;
        std::fs::write(dest, out).map_err(|e| format!("{}: {}", dest.display(), e))?;
    }
    println!("rescaled {} files by {} into {}", inputs.len(), factor, out_dir);
    Ok(())
}

/// Label calibration against outcomes (cal2): each scored position's label x
/// (side to move) becomes 400 * logit(E), where E = (1 + P(win) - P(loss)) / 2
/// under the set's own WDL model (wdlfit: P(win) = sigmoid((x - a) / b),
/// P(loss) = sigmoid((-x - a) / b), a and b cubic in material / 58). The
/// trainer's target sigmoid(score / 400) is then the label's probability of
/// the outcome as the set's own games measured it, whatever eval scale the
/// labelling net had.
fn cmd_calibrate(a: &str, b: &str, out_dir: &str, inputs: &[String]) -> Result<(), String> {
    let coef = |s: &str| -> Result<[f64; 4], String> {
        let v: Vec<f64> = s
            .split(',')
            .map(|x| x.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|_| format!("bad coefficients {}", s))?;
        v.try_into().map_err(|_| format!("need 4 coefficients, got {}", s))
    };
    let (ca, cb) = (coef(a)?, coef(b)?);
    let poly = |c: &[f64; 4], m: f64| ((c[0] * m + c[1]) * m + c[2]) * m + c[3];
    // Reject a model that isn't finite over the material range (NaN would
    // otherwise turn every label into 0 without an error).
    if ca.iter().chain(&cb).any(|c| !c.is_finite())
        || (17..=78).any(|mat| {
            let m = mat as f64 / 58.0;
            !poly(&ca, m).is_finite() || !poly(&cb, m).is_finite()
        })
    {
        return Err("calibrate: the coefficients must be finite (and give finite a, b for material 17..78)".into());
    }
    let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
    let map = |pos: &position::Position, s: i16| -> i16 {
        if s == viri::NO_SCORE {
            return s;
        }
        let m = (material(pos) as f64).clamp(17.0, 78.0) / 58.0;
        let (am, bm) = (poly(&ca, m), poly(&cb, m).max(1.0));
        let x = if pos.stm == position::WHITE { s as f64 } else { -(s as f64) };
        let e = (0.5 + 0.5 * (sig((x - am) / bm) - sig((-x - am) / bm))).clamp(1e-4, 1.0 - 1e-4);
        let y = (400.0 * (e / (1.0 - e)).ln()).round() as i16;
        if pos.stm == position::WHITE {
            y
        } else {
            -y
        }
    };
    let dests = output_paths(out_dir, inputs)?;
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {}", out_dir, e))?;
    let next = AtomicUsize::new(0);
    let errors = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(inputs.len());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(input) = inputs.get(i) else { break };
                let dest = &dests[i];
                let run = || -> Result<(), String> {
                    let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
                    let out = viri::map_scores(&data, map).map_err(|e| format!("{}: {}", input, e))?;
                    std::fs::write(dest, out).map_err(|e| format!("{}: {}", dest.display(), e))
                };
                if let Err(e) = run() {
                    errors.lock().unwrap().push(e);
                }
            });
        }
    });
    let errors = errors.into_inner().unwrap();
    if let Some(e) = errors.first() {
        return Err(format!("{} files failed, first: {}", errors.len(), e));
    }
    // Marker the trainer checks before training on a directory: these scores
    // are calibrated (cal2), and with which WDL model.
    let marker = std::path::Path::new(out_dir).join("CALIBRATED");
    std::fs::write(&marker, format!("cal2\na {}\nb {}\n", a, b)).map_err(|e| format!("{}: {}", marker.display(), e))?;
    println!("calibrated {} files into {}", inputs.len(), out_dir);
    Ok(())
}

fn cmd_stats(inputs: &[String]) -> Result<(), String> {
    for input in inputs {
        let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
        let (games, moves, unscored) = viri::count(&data).map_err(|e| format!("{}: {}", input, e))?;
        println!("{}: {} games, {} positions ({} unscored)", input, games, moves, unscored);
    }
    Ok(())
}

/// For each game that reaches a position the tablebases cover, compares the
/// game's result with the tablebase result of the first such position.
/// Counts game results by (material, eval) from the side to move's point of
/// view, for fitting a win/draw/loss model of our own eval. Material is
/// 1/3/3/5/9 for P/N/B/R/Q over both sides (the definition viriformat's WDL
/// filter uses); evals are bucketed to 5 cp and limited to +-2000; positions
/// before ply 16, in check or unscored are skipped, as in training.
/// Histogram of piece counts (all pieces, kings included) over the positions
/// training uses (scored, not in check, ply >= 16), for piece-count filtering.
fn cmd_piecestats(inputs: &[String]) -> Result<(), String> {
    let mut counts = [0u64; 33];
    for input in inputs {
        let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
        viri::for_each_position(&data, |pos, _, score, _| {
            let ply = 2 * (pos.fullmove as i32 - 1) + (pos.stm != position::WHITE) as i32;
            if score == viri::NO_SCORE || pos.checkers != 0 || ply < 16 {
                return;
            }
            counts[(pos.occ().count_ones() as usize).min(32)] += 1;
        })?;
    }
    let total: u64 = counts.iter().sum();
    println!("pieces,positions,share");
    for (n, &c) in counts.iter().enumerate() {
        if c > 0 {
            println!("{},{},{:.4}", n, c, c as f64 / total as f64);
        }
    }
    Ok(())
}

/// Threat inputs (engine/src/threats.rs) over the positions training uses
/// (scored, not in check, ply >= 16, quiet move), both perspectives with
/// mirroring: the distribution of active features per perspective (sets the
/// trainer's maximum and the accumulator's overflow bound) and how often each
/// feature fires (features that never fire get no gradient).
fn cmd_threatstats(variant: &str, inputs: &[String]) -> Result<(), String> {
    let var: usize = variant.parse().ok().filter(|&v| v < threats::VARIANTS).ok_or("variant must be 0, 1 or 2")?;
    let mut hits = vec![0u64; threats::features(var)];
    let mut per = vec![0u64; 257];
    let mut n = 0u64;
    for input in inputs {
        let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
        viri::for_each_position(&data, |pos, m, score, _| {
            let ply = 2 * (pos.fullmove as i32 - 1) + (pos.stm != position::WHITE) as i32;
            if score == viri::NO_SCORE || pos.checkers != 0 || ply < 16 || position::is_noisy(m) {
                return;
            }
            for p in [position::WHITE, position::BLACK] {
                let a = threats::active(var, pos, p, pawnpairs::mirror_xor(pos, p));
                per[a.len().min(256)] += 1;
                for f in a {
                    hits[f] += 1;
                }
            }
            n += 1;
        })?;
    }
    let views = 2 * n;
    let mut cum = 0u64;
    let (mut p50, mut p99, mut p9999, mut max) = (0, 0, 0, 0);
    for (k, &c) in per.iter().enumerate() {
        if c > 0 {
            max = k;
        }
        cum += c;
        if p50 == 0 && cum * 2 >= views {
            p50 = k;
        }
        if p99 == 0 && cum * 100 >= views * 99 {
            p99 = k;
        }
        if p9999 == 0 && cum * 10000 >= views * 9999 {
            p9999 = k;
        }
    }
    let mean = per.iter().enumerate().map(|(k, &c)| k as f64 * c as f64).sum::<f64>() / views.max(1) as f64;
    println!("variant {} features {}: {} positions, {} perspective views", var, hits.len(), n, views);
    println!("active per view: mean {:.1}, median {}, p99 {}, p99.99 {}, max {}", mean, p50, p99, p9999, max);
    let never = hits.iter().filter(|&&h| h == 0).count();
    let rare = hits.iter().filter(|&&h| h > 0 && h < 100).count();
    println!(
        "features never active: {} ({:.1}%), active in fewer than 100 views: {} ({:.1}%)",
        never,
        100.0 * never as f64 / hits.len() as f64,
        rare,
        100.0 * rare as f64 / hits.len() as f64
    );
    Ok(())
}

/// Score labels against material: counts of (piece lead, pawn lead, stored
/// score in 50 cp bins clamped to +-5000, result), side to move, over the
/// positions training uses (scored, not in check, ply >= 16, quiet move).
/// Piece lead counts minor pieces 3, rooks 5, queens 9. Shows whether labels
/// keep rising with the lead in decided positions, and how far they reach.
fn cmd_matstats(out: &str, inputs: &[String]) -> Result<(), String> {
    use position::{BISHOP, KNIGHT, PAWN, QUEEN, ROOK};
    use std::collections::BTreeMap;
    const STEP: i32 = 50;
    const SCORE_MAX: i32 = 5000;
    let mut counts: BTreeMap<(i32, i32, i32, u8), u64> = BTreeMap::new();
    let mut used = 0u64;
    for input in inputs {
        let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
        viri::for_each_position(&data, |pos, m, score, wdl| {
            let ply = 2 * (pos.fullmove as i32 - 1) + (pos.stm != position::WHITE) as i32;
            if score == viri::NO_SCORE || pos.checkers != 0 || ply < 16 || position::is_noisy(m) {
                return;
            }
            let white = pos.stm == position::WHITE;
            let s = if white { score as i32 } else { -(score as i32) };
            // viri WDL codes are white-relative: 0 black win, 1 draw, 2 white win.
            let r = if white { wdl } else { 2 - wdl };
            let side = |c: usize| -> (i32, i32) {
                let pieces = [(KNIGHT, 3), (BISHOP, 3), (ROOK, 5), (QUEEN, 9)]
                    .iter()
                    .map(|&(pt, v)| v * (pos.pieces[pt] & pos.colors[c]).count_ones() as i32)
                    .sum();
                (pieces, (pos.pieces[PAWN] & pos.colors[c]).count_ones() as i32)
            };
            let (us, them) = (side(pos.stm), side(pos.stm ^ 1));
            let b = (s.clamp(-SCORE_MAX, SCORE_MAX) as f64 / STEP as f64).round() as i32 * STEP;
            *counts.entry((us.0 - them.0, us.1 - them.1, b, r)).or_default() += 1;
            used += 1;
        })?;
    }
    let mut w = String::from("piece_lead,pawn_lead,score,result,count\n");
    for ((a, p, b, r), n) in counts {
        w.push_str(&format!("{},{},{},{},{}\n", a, p, b, r, n));
    }
    std::fs::write(out, w).map_err(|e| format!("{}: {}", out, e))?;
    println!("{} positions counted -> {}", used, out);
    Ok(())
}

/// Positions with exactly one bishop a side: opposite- vs same-coloured
/// bishops, by whether only bishops and pawns are left, pawn lead, stored
/// score and result (side to move; from ply 16, quiet moves, not in check).
/// For checking whether the nets score opposite-coloured bishops as drawish.
fn cmd_bishopstats(out: &str, inputs: &[String]) -> Result<(), String> {
    use position::{BISHOP, KNIGHT, PAWN, QUEEN, ROOK};
    use std::collections::BTreeMap;
    const STEP: i32 = 25;
    const SCORE_MAX: i32 = 3000;
    const DARK: u64 = 0xAA55_AA55_AA55_AA55;
    let mut counts: BTreeMap<(u8, u8, i32, i32, u8), u64> = BTreeMap::new();
    let mut used = 0u64;
    for input in inputs {
        let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
        viri::for_each_position(&data, |pos, m, score, wdl| {
            let ply = 2 * (pos.fullmove as i32 - 1) + (pos.stm != position::WHITE) as i32;
            if score == viri::NO_SCORE || pos.checkers != 0 || ply < 16 || position::is_noisy(m) {
                return;
            }
            let (wb, bb) = (pos.pieces[BISHOP] & pos.colors[0], pos.pieces[BISHOP] & pos.colors[1]);
            if wb.count_ones() != 1 || bb.count_ones() != 1 {
                return;
            }
            let ocb = ((wb & DARK != 0) != (bb & DARK != 0)) as u8;
            let others = pos.pieces[KNIGHT] | pos.pieces[ROOK] | pos.pieces[QUEEN];
            let pure = (others == 0) as u8;
            let white = pos.stm == position::WHITE;
            let s = if white { score as i32 } else { -(score as i32) };
            let r = if white { wdl } else { 2 - wdl };
            let pawns = |c: usize| (pos.pieces[PAWN] & pos.colors[c]).count_ones() as i32;
            let lead = pawns(pos.stm) - pawns(pos.stm ^ 1);
            let b = (s.clamp(-SCORE_MAX, SCORE_MAX) as f64 / STEP as f64).round() as i32 * STEP;
            *counts.entry((pure, ocb, lead, b, r)).or_default() += 1;
            used += 1;
        })?;
    }
    let mut w = String::from("pure,ocb,pawn_lead,score,result,count\n");
    for ((p, o, l, b, r), n) in counts {
        w.push_str(&format!("{},{},{},{},{},{}\n", p, o, l, b, r, n));
    }
    std::fs::write(out, w).map_err(|e| format!("{}: {}", out, e))?;
    println!("{} positions counted -> {}", used, out);
    Ok(())
}

/// Positions per ply (from the FEN's move number, as the trainer's ply filter
/// counts it) and how many are distinct by Zobrist hash, cumulatively below
/// each ply too: how many duplicates each ply cut would remove.
/// Held-out evaluation positions: every Nth position the trainer would use
/// (scored, not in check, ply >= 16, quiet move), as FEN, the game result from
/// the side to move's view (1, 0.5, 0) and its stored score (side to move), tab-separated.
fn cmd_heldout(every: &str, inputs: &[String]) -> Result<(), String> {
    let every: u64 = every.parse().map_err(|_| format!("bad count {}", every))?;
    let mut n = 0u64;
    let out = std::io::stdout();
    let mut out = std::io::BufWriter::new(out.lock());
    for input in inputs {
        let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
        viri::for_each_position(&data, |pos, m, score, wdl| {
            let ply = 2 * (pos.fullmove as i32 - 1) + (pos.stm != position::WHITE) as i32;
            if score == viri::NO_SCORE || pos.checkers != 0 || ply < 16 || position::is_noisy(m) {
                return;
            }
            n += 1;
            if n % every != 0 {
                return;
            }
            // viri WDL codes are white-relative: 0 black win, 1 draw, 2 white win.
            let white = pos.stm == position::WHITE;
            let r = if white { wdl } else { 2 - wdl };
            let s = if white { score as i32 } else { -(score as i32) };
            let _ = writeln!(out, "{}\t{}\t{}", pos.to_fen(), r as f32 / 2.0, s);
        })?;
    }
    Ok(())
}

fn cmd_plystats(inputs: &[String]) -> Result<(), String> {
    const MAX: usize = 64;
    let mut count = [0u64; MAX + 1];
    let mut seen: Vec<std::collections::HashSet<u64>> = vec![std::collections::HashSet::new(); MAX + 1];
    let mut all = std::collections::HashSet::new();
    let mut total = 0u64;
    for input in inputs {
        let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
        viri::for_each_position(&data, |pos, _, _, _| {
            let ply = (2 * (pos.fullmove as usize).max(1) - 2 + (pos.stm != position::WHITE) as usize).min(MAX);
            count[ply] += 1;
            seen[ply].insert(pos.hash);
            all.insert(pos.hash);
            total += 1;
        })?;
    }
    println!(
        "{} positions, {} distinct ({:.1}% duplicates overall)",
        total,
        all.len(),
        100.0 * (1.0 - all.len() as f64 / total as f64)
    );
    println!("ply\tpositions\tdistinct\tdup%\tcum_positions%");
    let mut cum = 0u64;
    for p in 0..=MAX {
        if count[p] == 0 {
            continue;
        }
        cum += count[p];
        let d = seen[p].len() as u64;
        println!(
            "{}{}\t{}\t{}\t{:.1}\t{:.1}",
            p,
            if p == MAX { "+" } else { "" },
            count[p],
            d,
            100.0 * (1.0 - d as f64 / count[p] as f64),
            100.0 * cum as f64 / total as f64
        );
    }
    Ok(())
}

/// Per side and ply band: how often the king has castled (and keeps a pawn
/// shield), stays on its home square, and how the pawns are spread, as rates
/// per position and side. Standard-chess sets only: the squares assume the
/// normal start (DFRC kings start elsewhere).
fn cmd_structstats(inputs: &[String]) -> Result<(), String> {
    const BANDS: [(usize, usize); 5] = [(0, 15), (16, 27), (28, 47), (48, 79), (80, 9999)];
    const NAMES: [&str; 9] = [
        "king_g/h",
        "king_a/b/c",
        "king_g/h+shield",
        "king_home",
        "pawns",
        "doubled",
        "isolated",
        "centre_pawn",
        "castle_rights",
    ];
    let mut sums = [[0f64; NAMES.len()]; BANDS.len()];
    let mut n = [0u64; BANDS.len()];
    let files: Vec<u64> = (0..8).map(|f| 0x0101_0101_0101_0101u64 << f).collect();
    for input in inputs {
        let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
        viri::for_each_position(&data, |pos, _, _, _| {
            let ply = 2 * (pos.fullmove as usize).max(1) - 2 + (pos.stm != position::WHITE) as usize;
            let Some(b) = BANDS.iter().position(|&(lo, hi)| (lo..=hi).contains(&ply)) else { return };
            n[b] += 2;
            for c in 0..2 {
                // Square in this side's frame (black flipped to look like white).
                let rel = |sq: u32| if c == 0 { sq } else { sq ^ 56 };
                let king = rel((pos.pieces[5] & pos.colors[c]).trailing_zeros());
                let pawns = pos.pieces[0] & pos.colors[c];
                let own = |sq: u32| pawns >> rel(sq) & 1 != 0;
                let v = &mut sums[b];
                let kingside = king == 6 || king == 7;
                v[0] += kingside as u8 as f64;
                v[1] += (king <= 2) as u8 as f64;
                let shield = [13, 14, 15, 21, 22, 23].iter().filter(|&&q| own(q)).count();
                v[2] += (kingside && shield >= 2) as u8 as f64;
                v[3] += (king == 4) as u8 as f64;
                v[4] += pawns.count_ones() as f64;
                for f in 0..8 {
                    let k = (pawns & files[f]).count_ones();
                    if k > 1 {
                        v[5] += (k - 1) as f64;
                    }
                    let side = (if f > 0 { files[f - 1] } else { 0 }) | (if f < 7 { files[f + 1] } else { 0 });
                    if k > 0 && pawns & side == 0 {
                        v[6] += k as f64;
                    }
                }
                v[7] += [27, 28, 35, 36].iter().filter(|&&q| own(q)).count() as f64;
                let rights = if c == 0 { pos.castling & 3 } else { pos.castling >> 2 & 3 };
                v[8] += (rights != 0) as u8 as f64;
            }
        })?;
    }
    print!("band\tsides");
    for name in NAMES {
        print!("\t{}", name);
    }
    println!();
    for (b, &(lo, hi)) in BANDS.iter().enumerate() {
        if n[b] == 0 {
            continue;
        }
        print!("{}-{}\t{}", lo, if hi > 999 { "".to_string() } else { hi.to_string() }, n[b]);
        for v in sums[b] {
            print!("\t{:.3}", v / n[b] as f64);
        }
        println!();
    }
    Ok(())
}

fn cmd_wdlstats(out: &str, inputs: &[String]) -> Result<(), String> {
    const EVAL_MAX: i32 = 2000;
    const STEP: i32 = 5;
    let nb = (2 * EVAL_MAX / STEP + 1) as usize;
    // counts[material][eval bucket][stm result: 0 loss, 1 draw, 2 win]
    let mut counts = vec![vec![[0u64; 3]; nb]; 79];
    let mut used = 0u64;
    for input in inputs {
        let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
        viri::for_each_position(&data, |pos, _, score, wdl| {
            if score == viri::NO_SCORE || pos.checkers != 0 {
                return;
            }
            let ply = 2 * (pos.fullmove as i32 - 1) + (pos.stm != position::WHITE) as i32;
            if ply < 16 {
                return;
            }
            let white = pos.stm == position::WHITE;
            let eval = if white { score as i32 } else { -(score as i32) };
            if eval.abs() > EVAL_MAX {
                return;
            }
            // viri WDL codes are white-relative: 0 black win, 1 draw, 2 white win.
            let r = if white { wdl } else { 2 - wdl } as usize;
            let m = material(pos).min(78) as usize;
            let b = ((eval + EVAL_MAX + STEP / 2).div_euclid(STEP)).clamp(0, nb as i32 - 1) as usize;
            counts[m][b][r] += 1;
            used += 1;
        })?;
    }
    let mut w = String::from("material,eval,loss,draw,win\n");
    for (m, row) in counts.iter().enumerate() {
        for (b, c) in row.iter().enumerate() {
            if c.iter().sum::<u64>() > 0 {
                w.push_str(&format!("{},{},{},{},{}\n", m, b as i32 * STEP - EVAL_MAX, c[0], c[1], c[2]));
            }
        }
    }
    std::fs::write(out, w).map_err(|e| format!("{}: {}", out, e))?;
    println!("{} positions counted -> {}", used, out);
    Ok(())
}

/// Fits P(win) = 1 / (1 + exp((a - x) / b)), P(loss) = 1 / (1 + exp((a + x) / b)),
/// with a and b cubic in m = clamp(material, MAT_MIN, MAT_MAX) / MAT_NORM (the
/// form viriformat's WDL filter evaluates, coefficients highest power first),
/// by maximum likelihood over the counted results. x is the eval in cp.
fn cmd_wdlfit(stats: &str) -> Result<(), String> {
    const MAT_MIN: f64 = 17.0;
    const MAT_MAX: f64 = 78.0;
    const MAT_NORM: f64 = 58.0;
    let text = std::fs::read_to_string(stats).map_err(|e| format!("{}: {}", stats, e))?;
    // (m, x, [loss, draw, win])
    let mut rows: Vec<(f64, f64, [f64; 3])> = Vec::new();
    for line in text.lines().skip(1) {
        let v: Vec<f64> = line.split(',').map(|t| t.parse().unwrap_or(0.0)).collect();
        if v.len() == 5 {
            rows.push((v[0].clamp(MAT_MIN, MAT_MAX) / MAT_NORM, v[1], [v[2], v[3], v[4]]));
        }
    }
    let total: f64 = rows.iter().map(|r| r.2.iter().sum::<f64>()).sum();
    let poly = |c: &[f64], m: f64| ((c[0] * m + c[1]) * m + c[2]) * m + c[3];
    let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
    // Mean negative log-likelihood and its gradient over the 8 coefficients.
    let nll = |p: &[f64; 8]| -> (f64, [f64; 8]) {
        let (mut f, mut g) = (0.0, [0.0; 8]);
        for &(m, x, n) in &rows {
            let (a, b) = (poly(&p[0..4], m), poly(&p[4..8], m).max(1.0));
            let (zw, zl) = ((x - a) / b, (-x - a) / b);
            let (w, l) = (sig(zw), sig(zl));
            let d = (1.0 - w - l).max(1e-12);
            let (w, l) = (w.max(1e-12), l.max(1e-12));
            f -= n[2] * w.ln() + n[1] * d.ln() + n[0] * l.ln();
            // d/da and d/db of the log-likelihood of each outcome.
            let (dw, dl) = (w * (1.0 - w), l * (1.0 - l)); // d sigma / dz
            let (dzw_da, dzw_db, dzl_da, dzl_db) = (-1.0 / b, -zw / b, -1.0 / b, -zl / b);
            let dw_da = dw * dzw_da;
            let dw_db = dw * dzw_db;
            let dl_da = dl * dzl_da;
            let dl_db = dl * dzl_db;
            let ga = n[2] * dw_da / w + n[0] * dl_da / l - n[1] * (dw_da + dl_da) / d;
            let gb = n[2] * dw_db / w + n[0] * dl_db / l - n[1] * (dw_db + dl_db) / d;
            let pw = [m * m * m, m * m, m, 1.0];
            for k in 0..4 {
                g[k] -= ga * pw[k];
                g[4 + k] -= gb * pw[k];
            }
        }
        (f / total, g.map(|v| v / total))
    };
    // Adam from a flat start (a = 150, b = 100 cp).
    let mut p = [0.0, 0.0, 0.0, 150.0, 0.0, 0.0, 0.0, 100.0];
    let (mut mo, mut ve) = ([0.0; 8], [0.0; 8]);
    let (lr, b1, b2) = (0.5, 0.9, 0.999);
    let mut last = f64::MAX;
    for it in 1..=20000 {
        let (f, g) = nll(&p);
        for k in 0..8 {
            mo[k] = b1 * mo[k] + (1.0 - b1) * g[k];
            ve[k] = b2 * ve[k] + (1.0 - b2) * g[k] * g[k];
            let mh = mo[k] / (1.0 - b1.powi(it));
            let vh = ve[k] / (1.0 - b2.powi(it));
            p[k] -= lr * mh / (vh.sqrt() + 1e-12);
        }
        if it % 1000 == 0 {
            if (last - f).abs() < 1e-9 {
                break;
            }
            last = f;
        }
    }
    let (f, _) = nll(&p);
    println!("positions {}, mean NLL {:.5} (log 3 = {:.5} for a uniform guess)", total as u64, f, 3f64.ln());
    println!("material {}..{}, normalised by {}", MAT_MIN, MAT_MAX, MAT_NORM);
    println!("a = [{:.6}, {:.6}, {:.6}, {:.6}]", p[0], p[1], p[2], p[3]);
    println!("b = [{:.6}, {:.6}, {:.6}, {:.6}]", p[4], p[5], p[6], p[7]);
    for mat in [20.0, 30.0, 40.0, 50.0, 58.0, 70.0, 78.0] {
        let m = f64::clamp(mat, MAT_MIN, MAT_MAX) / MAT_NORM;
        println!("  material {:>2}: a {:6.1} cp (50% win), b {:5.1}", mat, poly(&p[0..4], m), poly(&p[4..8], m));
    }
    Ok(())
}

/// 1/3/3/5/9 material over both sides.
fn material(pos: &position::Position) -> u32 {
    use position::{BISHOP, KNIGHT, PAWN, QUEEN, ROOK};
    [(PAWN, 1), (KNIGHT, 3), (BISHOP, 3), (ROOK, 5), (QUEEN, 9)]
        .iter()
        .map(|&(pt, v)| v * pos.pieces[pt].count_ones())
        .sum()
}

fn cmd_tbstats(tb_path: &str, inputs: &[String]) -> Result<(), String> {
    let largest = tb::init(tb_path);
    if largest == 0 {
        return Err(format!("no tablebases found in {}", tb_path));
    }
    // m[game result][tablebase result], both white-relative (viri WDL codes).
    let mut m = [[0u64; 3]; 3];
    let (mut games, mut positions, mut reach_pos, mut flip_pos, mut failed) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut running_clock = 0u64;
    for input in inputs {
        let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
        viri::for_each_game(&data, |moves, wdl| {
            games += 1;
            positions += moves.len() as u64;
            // The WDL tables answer for a fresh fifty-move count, so use the
            // first table position with its clock at zero (entered by a
            // capture or pawn move); one with a running clock may already be
            // drawn by the fifty-move rule, and resetting it asks a different
            // question.
            let in_tb = |p: &position::Position| p.occ().count_ones() <= largest && p.castling == 0;
            if !moves.iter().any(|(p, _)| in_tb(p)) {
                return;
            }
            let Some(k) = moves.iter().position(|(p, _)| in_tb(p) && p.halfmove == 0) else {
                running_clock += 1;
                return;
            };
            let p = moves[k].0;
            let Some(r) = tb::probe_wdl(&p) else {
                failed += 1;
                return;
            };
            let stm_white = p.stm == position::WHITE;
            let t = match (r, stm_white) {
                (tb::Wdl::Draw, _) => viri::WDL_DRAW,
                (tb::Wdl::Win, true) | (tb::Wdl::Loss, false) => viri::WDL_WHITE_WIN,
                _ => viri::WDL_BLACK_WIN,
            };
            m[wdl as usize][t as usize] += 1;
            reach_pos += moves.len() as u64;
            if t != wdl {
                flip_pos += moves.len() as u64;
            }
        })
        .map_err(|e| format!("{}: {}", input, e))?;
    }
    let reach: u64 = m.iter().flatten().sum();
    let flips: u64 =
        (0..3).flat_map(|g| (0..3).map(move |t| (g, t))).filter(|(g, t)| g != t).map(|(g, t)| m[g][t]).sum();
    println!("{}-man tables; {} games, {} positions", largest, games, positions);
    println!(
        "games reaching the tables: {} ({:.1}%), probe failures {}; positions in them {} ({:.1}%)",
        reach,
        100.0 * reach as f64 / games.max(1) as f64,
        failed,
        reach_pos,
        100.0 * reach_pos as f64 / positions.max(1) as f64
    );
    println!(
        "games entering the tables with a running fifty-move clock and no later reset (not counted): {}",
        running_clock
    );
    println!(
        "result differs from the tables: {} games ({:.1}% of those reaching them), {} positions ({:.2}% of all)",
        flips,
        100.0 * flips as f64 / reach.max(1) as f64,
        flip_pos,
        100.0 * flip_pos as f64 / positions.max(1) as f64
    );
    let name = ["black win", "draw", "white win"];
    println!("game result -> tablebase result (rows: game):");
    for g in 0..3 {
        println!("  {:>9}: black win {:>8}  draw {:>8}  white win {:>8}", name[g], m[g][0], m[g][1], m[g][2]);
    }
    Ok(())
}

/// Score label of a won tablebase position (cp, for the winner; losses get its
/// negation, draws 0).
#[derive(Clone, Copy)]
enum TbLabel {
    /// The same score for every win.
    Flat(i32),
    /// Graded by progress: `hi` at DTZ 0 falling linearly to `lo` at DTZ 100
    /// (plies), plus `per_pawn` per pawn of material lead (P 1, N 3, B 3,
    /// R 5, Q 9) so that winning material never lowers the label although the
    /// DTZ count restarts; never below `lo / 2`.
    Dtz { lo: i32, hi: i32, per_pawn: i32 },
    /// The search score when it already shows the win at least this clearly,
    /// else this value.
    Keep(i32),
}

impl TbLabel {
    /// `sign`: +1 white wins, -1 black wins (0: draw, label 0). `search`:
    /// the white-relative search score.
    fn win_score(self, p: &position::Position, sign: i32, dtz: u32, search: i32) -> i32 {
        if sign == 0 {
            return 0;
        }
        match self {
            TbLabel::Flat(cp) => cp,
            TbLabel::Dtz { lo, hi, per_pawn } => {
                let values = [1, 3, 3, 5, 9];
                let lead: i32 = (0..5)
                    .map(|pc| {
                        let n = |c: usize| (p.pieces[pc] & p.colors[c]).count_ones() as i32;
                        values[pc] * (n(position::WHITE) - n(position::BLACK))
                    })
                    .sum();
                let base = hi - (hi - lo) * dtz.min(100) as i32 / 100;
                (base + per_pawn * sign * lead).max(lo / 2)
            }
            TbLabel::Keep(cp) => (sign * search).max(cp),
        }
    }
}

fn cmd_tbrelabel(tb_path: &str, out_dir: &str, args: &[String]) -> Result<(), String> {
    let largest = tb::init(tb_path);
    if largest == 0 {
        return Err(format!("no tablebases found in {}", tb_path));
    }
    let (mut fix_result, mut label, mut inputs) = (false, None, Vec::new());
    let mut it = args.iter();
    let cp = |it: &mut std::slice::Iter<String>, flag: &str| -> Result<i32, String> {
        let v: i32 = it.next().and_then(|v| v.parse().ok()).ok_or(format!("{} needs cp values", flag))?;
        if !(0..=19000).contains(&v) {
            return Err(format!("{} value {} out of range 0..19000", flag, v));
        }
        Ok(v)
    };
    while let Some(a) = it.next() {
        match a.as_str() {
            "--fix-result" => fix_result = true,
            "--score" => label = Some(TbLabel::Flat(cp(&mut it, a)?)),
            "--score-dtz" => {
                label = Some(TbLabel::Dtz { lo: cp(&mut it, a)?, hi: cp(&mut it, a)?, per_pawn: cp(&mut it, a)? })
            }
            "--score-keep" => label = Some(TbLabel::Keep(cp(&mut it, a)?)),
            _ => inputs.push(a.clone()),
        }
    }
    if !fix_result && label.is_none() {
        return Err("tbrelabel needs --fix-result and/or a score mode".into());
    }
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {}", out_dir, e))?;
    let in_tb = |p: &position::Position| p.occ().count_ones() <= largest && p.castling == 0;
    // White-relative viri WDL code of a table result for the side to move.
    let white_wdl = |r: tb::Wdl, stm_white: bool| match (r, stm_white) {
        (tb::Wdl::Draw, _) => viri::WDL_DRAW,
        (tb::Wdl::Win, true) | (tb::Wdl::Loss, false) => viri::WDL_WHITE_WIN,
        _ => viri::WDL_BLACK_WIN,
    };
    let (mut games, mut changed, mut relabelled, mut failed) = (0u64, 0u64, 0u64, 0u64);
    for input in &inputs {
        let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
        let out = viri::rewrite_games(&data, |game, wdl, scores| {
            games += 1;
            if fix_result {
                if let Some((p, _)) = game.iter().find(|(p, _)| in_tb(p) && p.halfmove == 0) {
                    match tb::probe_wdl(p) {
                        Some(r) => {
                            let t = white_wdl(r, p.stm == position::WHITE);
                            if t != *wdl {
                                *wdl = t;
                                changed += 1;
                            }
                        }
                        None => failed += 1,
                    }
                }
            }
            if let Some(label) = label {
                for (k, (p, s)) in game.iter().enumerate() {
                    if *s == viri::NO_SCORE || !in_tb(p) {
                        continue;
                    }
                    // Probed with a fresh fifty-move count; with a running clock
                    // this can call a cursed win a win (rare).
                    let mut q = *p;
                    q.halfmove = 0;
                    let Some((r, dtz)) = (match label {
                        TbLabel::Dtz { .. } => tb::probe_dtz(&q),
                        _ => tb::probe_wdl(&q).map(|r| (r, 0)),
                    }) else {
                        failed += 1;
                        continue;
                    };
                    let sign = match white_wdl(r, p.stm == position::WHITE) {
                        viri::WDL_WHITE_WIN => 1,
                        viri::WDL_BLACK_WIN => -1,
                        _ => 0,
                    };
                    scores[k] = (sign * label.win_score(p, sign, dtz, *s as i32)) as i16;
                    relabelled += 1;
                }
            }
        })
        .map_err(|e| format!("{}: {}", input, e))?;
        let name = std::path::Path::new(input).file_name().ok_or("bad input name")?;
        let path = std::path::Path::new(out_dir).join(name);
        std::fs::write(&path, out).map_err(|e| format!("{}: {}", path.display(), e))?;
    }
    println!(
        "{}-man tables; {} games, results changed {}, scores relabelled {}, probe failures {}",
        largest, games, changed, relabelled, failed
    );
    Ok(())
}

fn cmd_net(source: &str, input: &str, out_dir: &str, opts: &[String]) -> Result<(), String> {
    let mut arch = netfmt::Arch {
        hidden: 0,
        king_buckets: [0; 64],
        mirror: false,
        activation: netfmt::ACT_SCRELU,
        output_buckets: 8,
        qa: 255,
        qb: 64,
        scale: 400,
        description: String::new(),
        l1: 0,
        l1_shared: false,
        l1_dual: false,
        l1_skip: false,
        l1_lin: 0,
        pawn_pairs: false,
        pawn_pair_window: 0,
        perm: Vec::new(),
        l1_shift: 0,
        l2: 0,
        threats: None,
    };
    let mut qb_set = false;
    let mut i = 0;
    while i < opts.len() {
        let flag = opts[i].as_str();
        if flag == "--mirror" {
            arch.mirror = true;
            i += 1;
            continue;
        }
        if flag == "--l1-shared" {
            arch.l1_shared = true;
            i += 1;
            continue;
        }
        if flag == "--l1-skip" {
            arch.l1_skip = true;
            i += 1;
            continue;
        }
        if flag == "--l1-dual" {
            arch.l1_dual = true;
            i += 1;
            continue;
        }
        if flag == "--pawn-pairs" {
            arch.pawn_pairs = true;
            i += 1;
            continue;
        }
        if flag == "--pawn-pair-window" {
            let w: usize = opts
                .get(i + 1)
                .and_then(|v| v.parse().ok())
                .filter(|&w| w < 3)
                .ok_or("--pawn-pair-window needs 0, 1 or 2")?;
            arch.pawn_pairs = true;
            arch.pawn_pair_window = w;
            i += 2;
            continue;
        }
        let val = opts.get(i + 1).ok_or_else(|| format!("{} needs a value", flag))?;
        let num = || val.parse::<i64>().map_err(|_| format!("{}: bad number {}", flag, val));
        match flag {
            "--l1-lin" => arch.l1_lin = num()? as usize,
            "--threats" => {
                let v = num()? as usize;
                if v >= threats::VARIANTS {
                    return Err("--threats needs 0, 1 or 2".into());
                }
                arch.threats = Some(v);
            }
            "--hidden" => arch.hidden = num()? as usize,
            "--l1" => arch.l1 = num()? as usize,
            "--l1-shift" => arch.l1_shift = num()? as u32,
            "--l2" => arch.l2 = num()? as usize,
            "--output-buckets" => arch.output_buckets = num()? as usize,
            "--qa" => arch.qa = num()? as i32,
            "--qb" => {
                arch.qb = num()? as i32;
                qb_set = true;
            }
            "--scale" => arch.scale = num()? as i32,
            "--description" => arch.description = val.clone(),
            "--permute" => {
                let text = std::fs::read_to_string(val).map_err(|e| format!("{}: {}", val, e))?;
                arch.perm = text
                    .split_whitespace()
                    .map(|x| x.parse::<usize>())
                    .collect::<Result<_, _>>()
                    .map_err(|_| format!("bad --permute file {}", val))?;
            }
            "--activation" => {
                arch.activation = match val.as_str() {
                    "screlu" => netfmt::ACT_SCRELU,
                    "crelu" => netfmt::ACT_CRELU,
                    "pairwise" => netfmt::ACT_PAIRWISE,
                    _ => return Err(format!("unknown activation {}", val)),
                }
            }
            "--king-buckets" => {
                let v: Vec<u8> = val
                    .split(',')
                    .map(|x| x.trim().parse::<u8>())
                    .collect::<Result<_, _>>()
                    .map_err(|_| format!("bad --king-buckets {}", val))?;
                // A 32-entry table covers files a-d (mirrored boards), as in
                // Bullet's ChessBucketsMirrored; expand it to all 64 squares.
                arch.king_buckets = match v.len() {
                    64 => v.try_into().unwrap(),
                    32 => std::array::from_fn(|sq| v[(sq / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][sq % 8]]),
                    n => return Err(format!("--king-buckets needs 32 or 64 entries, got {}", n)),
                };
            }
            _ => return Err(format!("unknown option {}\n{}", flag, USAGE)),
        }
        i += 2;
    }
    if arch.hidden == 0 {
        return Err("--hidden is required".into());
    }
    // The trainer quantises hidden-layer weights at 128, not the FT-only
    // default of 64; a forgotten --qb would give a self-consistent-looking
    // net whose hidden pre-activations are twice too large.
    if arch.l1 > 0 && !qb_set {
        arch.qb = 128;
    }
    let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
    let net = netfmt::convert(&arch, source, &data)?;
    let name = format!("net-{}.nnue", sha256_prefix(&net));
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {}", out_dir, e))?;
    let out = Path::new(out_dir).join(&name);
    std::fs::write(&out, &net).map_err(|e| format!("{}: {}", out.display(), e))?;
    println!("{}", out.display());
    print!("{}", netfmt::describe(&net)?);
    Ok(())
}

fn cmd_net_info(inputs: &[String]) -> Result<(), String> {
    for input in inputs {
        let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
        println!("{}:", input);
        print!("{}", netfmt::describe(&data).map_err(|e| format!("{}: {}", input, e))?);
    }
    Ok(())
}

/// First 8 hex digits (upper case) of the SHA-256 of `data`, the net naming
/// convention (same as `sha256sum | cut -c1-8 | tr a-f A-F`).
fn sha256_prefix(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] =
        [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for t in 0..16 {
            w[t] = u32::from_be_bytes(block[4 * t..4 * t + 4].try_into().unwrap());
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16].wrapping_add(s0).wrapping_add(w[t - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for t in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[t]).wrapping_add(w[t]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    format!("{:08X}", h[0])
}

/// Sampling categories for `fens`: four game phases by piece count, plus
/// positions that stress particular parts of the search.
const FEN_CATEGORIES: [(&str, usize); 7] = [
    ("opening (26-32 pieces)", 8),
    ("middlegame (16-25)", 12),
    ("endgame (8-15)", 10),
    ("late endgame (3-7)", 6),
    ("tactical (5+ legal captures)", 6),
    ("closed (3+ blocked pawn pairs)", 5),
    ("pawn endgame", 3),
];

/// Picks the category for a position, most specific first.
fn fen_category(pos: &crate::position::Position) -> usize {
    use crate::position::*;
    let pawns = pos.pieces[PAWN];
    let kings = pos.pieces[KING];
    if pos.occ() == pawns | kings && pawns.count_ones() >= 2 {
        return 6;
    }
    let (wp, bp) = (pawns & pos.colors[WHITE], pawns & pos.colors[BLACK]);
    if ((wp << 8) & bp).count_ones() >= 3 {
        return 5;
    }
    let mut list = MoveList::new();
    pos.gen_moves(&mut list, true);
    let captures = (0..list.len())
        .filter(|&k| {
            is_capture(list[k]) && {
                let mut c = *pos;
                c.make_move(list[k])
            }
        })
        .count();
    if captures >= 5 {
        return 4;
    }
    match pos.occ().count_ones() {
        26.. => 0,
        16..=25 => 1,
        8..=15 => 2,
        _ => 3,
    }
}

/// Samples bench-style positions: past the first 16 plies, not in check, with a
/// real score and not decided (|score| <= 400), in the quotas of FEN_CATEGORIES
/// (scaled to `count`). Deterministic for a given seed (reservoir sampling).
fn cmd_fens(count: &str, seed: &str, inputs: &[String]) -> Result<(), String> {
    let count: usize = count.parse().map_err(|_| format!("bad count {}", count))?;
    let mut rng: u64 = seed.parse::<u64>().map_err(|_| format!("bad seed {}", seed))? | 1;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let total: usize = FEN_CATEGORIES.iter().map(|c| c.1).sum();
    let quota: Vec<usize> = FEN_CATEGORIES.iter().map(|c| (c.1 * count).div_ceil(total)).collect();
    let mut picks: Vec<Vec<String>> = vec![Vec::new(); FEN_CATEGORIES.len()];
    let mut seen = vec![0u64; FEN_CATEGORIES.len()];
    for input in inputs {
        let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
        viri::for_each_position(&data, |pos, _m, score, _wdl| {
            let ply = 2 * (pos.fullmove as u32 - 1) + pos.stm as u32;
            if ply < 16 || pos.checkers != 0 || score == viri::NO_SCORE || score.unsigned_abs() > 400 {
                return;
            }
            let c = fen_category(pos);
            seen[c] += 1;
            if picks[c].len() < quota[c] {
                picks[c].push(pos.to_fen());
            } else {
                let j = (next() % seen[c]) as usize;
                if j < quota[c] {
                    picks[c][j] = pos.to_fen();
                }
            }
        })
        .map_err(|e| format!("{}: {}", input, e))?;
    }
    for (c, list) in picks.iter().enumerate() {
        eprintln!("{}: {} candidates, {} picked", FEN_CATEGORIES[c].0, seen[c], list.len());
        println!("// {}", FEN_CATEGORIES[c].0);
        for fen in list {
            println!("{}", fen);
        }
    }
    Ok(())
}

/// `pgnmoves <in.pgn> ...`: one tab-separated line per move - game number, ply,
/// mover's name, FEN before the move, the move in UCI, the move's comment - for
/// analysing time use (tm_difficulty.py).
fn cmd_pgnmoves(inputs: &[String]) -> Result<(), String> {
    let out = std::io::stdout();
    let mut w = std::io::BufWriter::new(out.lock());
    let mut gi = 0usize;
    for path in inputs {
        let f = File::open(path).map_err(|e| format!("{}: {}", path, e))?;
        for g in pgn::Games::new(std::io::BufReader::new(f)) {
            let g = g.map_err(|e| e.to_string())?;
            gi += 1;
            let names = [g.tag("White").unwrap_or("?").to_string(), g.tag("Black").unwrap_or("?").to_string()];
            let Ok(moves) = pgn::moves_with_comments(&g) else { continue };
            for (ply, (pos, m, c)) in moves.iter().enumerate() {
                let _ =
                    writeln!(w, "{}\t{}\t{}\t{}\t{}\t{}", gi, ply, names[pos.stm], pos.to_fen(), pos.move_uci(*m), c);
            }
        }
    }
    Ok(())
}
