// Training-data tools for Incipit: converts OpenBench datagen PGNs and the
// engine's old datagen `.bin` files to viriformat, and summarises viriformat
// files.
//
// The board code is the engine's own, included directly so this tool needs
// neither the engine's net nor any dependency.
#[allow(dead_code)]
#[path = "../../engine/src/attacks.rs"]
mod attacks;
#[allow(dead_code)]
#[path = "../../engine/src/position.rs"]
mod position;

mod oldbin;
mod pgn;
#[cfg(test)]
mod tests;
mod viri;

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const USAGE: &str = "\
usage:
  datatools pgn <out.vf> <in.pgn | -> ...     OpenBench/fastchess PGNs -> viriformat
                                              (- reads stdin, e.g. bzcat *.pgn.bz2 | ...)
  datatools bin <out_dir> <in.bin> ...        old datagen .bin -> viriformat, one
                                              <out_dir>/<name>.vf per input, in parallel
  datatools stats <file.vf> ...               count games and positions";

fn main() {
    attacks::init();
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("pgn") if args.len() >= 4 => cmd_pgn(&args[2], &args[3..]),
        Some("bin") if args.len() >= 4 => cmd_bin(&args[2], &args[3..]),
        Some("stats") if args.len() >= 3 => cmd_stats(&args[2..]),
        _ => Err(USAGE.to_string()),
    };
    if let Err(e) = result {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}

fn read_input(path: &str) -> Result<String, String> {
    let mut text = String::new();
    if path == "-" {
        std::io::stdin().read_to_string(&mut text).map_err(|e| format!("stdin: {}", e))?;
    } else {
        File::open(path).and_then(|mut f| f.read_to_string(&mut text)).map_err(|e| format!("{}: {}", path, e))?;
    }
    Ok(text)
}

fn cmd_pgn(out: &str, inputs: &[String]) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(out).map_err(|e| format!("{}: {}", out, e))?);
    let (mut games, mut moves, mut skipped) = (0u64, 0u64, 0u64);
    for input in inputs {
        let text = read_input(input)?;
        for g in pgn::split_games(&text) {
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
                            input, out.display(), st.records, st.games, st.moves
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
        t.records, t.games, t.moves, scored, 100.0 * scored as f64 / t.records.max(1) as f64, t.unscored
    );
    let bridges: Vec<String> = (1..=oldbin::MAX_GAP).map(|d| format!("{}: {}", d, t.gaps[d])).collect();
    println!("bridge lengths (plies): {}", bridges.join(", "));
    println!("chain restarts within a game: {}; records without a move: {}", t.restarts, t.dropped);
    let errors = errors.into_inner().unwrap();
    if errors.is_empty() { Ok(()) } else { Err(errors.join("\n")) }
}

fn cmd_stats(inputs: &[String]) -> Result<(), String> {
    for input in inputs {
        let data = std::fs::read(input).map_err(|e| format!("{}: {}", input, e))?;
        let (games, moves, unscored) = viri::count(&data).map_err(|e| format!("{}: {}", input, e))?;
        println!("{}: {} games, {} positions ({} unscored)", input, games, moves, unscored);
    }
    Ok(())
}
