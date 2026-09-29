mod attacks;
mod datagen;
mod eval;
mod nnue;
mod params;
mod position;
mod search;
mod tt;
use position::*;
use search::*;
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

const NAME: &str = "Incipit 0.1";
const AUTHOR: &str = "Adam Twiss";

const BENCH_FENS: [&str; 8] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 8",
    "2r3k1/5pp1/p2p3p/1p1Pp3/1P2P3/P1R2P2/6PP/6K1 w - - 0 30",
    "r1b2rk1/2q1bppp/p2p1n2/np2p3/3PP3/2P2N1P/PPB2PP1/RNBQR1K1 w - - 0 12",
    "8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/1Q3PPP/6K1 w - - 0 1",
];

fn perft_suite(path: &str, maxd: usize) {
    let text = std::fs::read_to_string(path).unwrap();
    let t = Instant::now();
    let mut fails = 0;
    let mut total = 0u64;
    for line in text.lines() {
        let parts: Vec<&str> = line.split(';').collect();
        let pos = Position::from_fen(parts[0].trim()).unwrap();
        for p in &parts[1..] {
            let p = p.trim();
            let d: usize = p[1..p.find(' ').unwrap()].parse().unwrap();
            let exp: u64 = p[p.find(' ').unwrap() + 1..].trim().parse().unwrap();
            if d > maxd {
                continue;
            }
            let n = perft(&pos, d as u32);
            total += n;
            if n != exp {
                fails += 1;
                println!("FAIL {} d{} got {} exp {}", parts[0], d, n, exp);
            }
        }
    }
    println!("done fails={} nodes={} time={:?}", fails, total, t.elapsed());
}

fn bench(depth: i32) {
    let mut s = Searcher::new(16, Arc::new(AtomicBool::new(false)));
    s.silent = true;
    let t = Instant::now();
    let mut nodes = 0;
    for f in BENCH_FENS.iter() {
        let pos = Position::from_fen(f).unwrap();
        s.clear();
        s.hash_hist.clear();
        let lim = Limits { soft_ms: None, hard_ms: None, depth, nodes: None };
        let (m, sc) = s.search(&pos, &lim);
        println!("{} -> {} {}", f, move_str(m), sc);
        nodes += s.nodes;
    }
    let el = t.elapsed().as_secs_f64();
    println!("{} nodes {} nps", nodes, (nodes as f64 / el) as u64);
}

struct Uci {
    pos: Position,
    hist: Vec<u64>,
    searcher: Searcher,
    overhead: i64,
}

impl Uci {
    fn set_position(&mut self, toks: &[&str]) {
        // Index of the first token after "startpos" / "fen".
        let mut i = 2;
        let mut pos;
        if toks.get(1) == Some(&"startpos") {
            pos = Position::from_fen(START_FEN).unwrap();
        } else if toks.get(1) == Some(&"fen") {
            let mut fen = String::new();
            while i < toks.len() && toks[i] != "moves" {
                fen.push_str(toks[i]);
                fen.push(' ');
                i += 1;
            }
            match Position::from_fen(&fen) {
                Some(p) => pos = p,
                None => return,
            }
        } else {
            return;
        }
        self.hist.clear();
        if toks.get(i) == Some(&"moves") {
            for mstr in &toks[i + 1..] {
                match pos.parse_move(mstr) {
                    Some(m) => {
                        self.hist.push(pos.hash);
                        pos.make_move(m);
                    }
                    None => break,
                }
            }
        }
        self.pos = pos;
    }

    fn go(&mut self, toks: &[&str]) {
        let mut wtime = None;
        let mut btime = None;
        let mut winc = 0i64;
        let mut binc = 0i64;
        let mut mtg = 0i64;
        let mut movetime = None;
        let mut depth = MAX_PLY as i32;
        let mut nodes = None;
        let mut i = 1;
        while i < toks.len() {
            let v = toks.get(i + 1).and_then(|s| s.parse::<i64>().ok());
            match toks[i] {
                "wtime" => wtime = v,
                "btime" => btime = v,
                "winc" => winc = v.unwrap_or(0),
                "binc" => binc = v.unwrap_or(0),
                "movestogo" => mtg = v.unwrap_or(0),
                "movetime" => movetime = v,
                "depth" => depth = v.unwrap_or(depth as i64).clamp(1, MAX_PLY as i64 - 4) as i32,
                "nodes" => nodes = v.map(|x| x.max(1) as u64),
                _ => {
                    i += 1;
                    continue;
                }
            }
            i += 2;
        }
        let (time, inc) = if self.pos.stm == WHITE { (wtime, winc) } else { (btime, binc) };
        let mut lim = Limits { soft_ms: None, hard_ms: None, depth, nodes };
        if let Some(mt) = movetime {
            lim.hard_ms = Some((mt - self.overhead).max(1) as u64);
        } else if let Some(t) = time {
            let left = (t - self.overhead).max(1);
            let soft = if mtg > 0 {
                left / (mtg + 1).min(40) + inc * 3 / 4
            } else {
                left / params::tp(params::P::TmSoftDiv) as i64 + inc * params::tp(params::P::TmIncPct) as i64 / 100
            };
            let hard = (left * 2 / 5).min(soft * params::tp(params::P::TmHardMul) as i64).max(1);
            let soft = soft.min(hard);
            lim.soft_ms = Some(soft as u64);
            lim.hard_ms = Some(hard as u64);
        }
        self.searcher.hash_hist.clear();
        self.searcher.hash_hist.extend_from_slice(&self.hist);
        let (m, _) = self.searcher.search(&self.pos, &lim);
        println!("bestmove {}", move_str(m));
    }
}

fn main() {
    attacks::init();
    nnue::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 {
        match args[1].as_str() {
            "perftsuite" => {
                perft_suite(&args[2], args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5));
                return;
            }
            "datagen" => {
                // datagen <threads> <prefix> <nodes> <seconds>
                datagen::run(args[2].parse().unwrap(), &args[3], args[4].parse().unwrap(), args[5].parse().unwrap());
                return;
            }
            "nnuecheck" => {
                let mut seed = 12345u64;
                let mut bad = 0;
                for _g in 0..200 {
                    let mut pos = Position::from_fen(START_FEN).unwrap();
                    let mut acc = nnue::Acc { v: [[0; nnue::H]; 2] };
                    acc.refresh(&pos);
                    for _ in 0..200 {
                        let mut list = MoveList::new();
                        pos.gen_moves(&mut list, false);
                        let mut legal = vec![];
                        for i in 0..list.len { let mut c = pos; if c.make_move(list.moves[i]) { legal.push(list.moves[i]); } }
                        if legal.is_empty() { break; }
                        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                        let m = legal[(seed % legal.len() as u64) as usize];
                        let mut child = pos; child.make_move(m);
                        let mut a2 = nnue::Acc { v: [[0; nnue::H]; 2] };
                        a2.update_from(&acc, &pos, &child, m);
                        let mut a3 = nnue::Acc { v: [[0; nnue::H]; 2] };
                        a3.refresh(&child);
                        if a2.v != a3.v { bad += 1; println!("mismatch {} {}", pos.to_fen(), move_str(m)); }
                        pos = child; acc = a3;
                    }
                }
                println!("bad {}", bad);
                println!("startpos eval {}", nnue::evaluate(&{ let mut a = nnue::Acc { v: [[0; nnue::H]; 2] }; a.refresh(&Position::from_fen(START_FEN).unwrap()); a }, &Position::from_fen(START_FEN).unwrap()));
                return;
            }
            "bench" => {
                bench(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12));
                return;
            }
            _ => {}
        }
    }

    let stop = Arc::new(AtomicBool::new(false));
    let searching = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<String>();
    {
        let stop = stop.clone();
        let searching = searching.clone();
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                line.clear();
                match stdin.lock().read_line(&mut line) {
                    Ok(0) | Err(_) => {
                        stop.store(true, Ordering::SeqCst);
                        let _ = tx.send("quit".to_string());
                        break;
                    }
                    _ => {}
                }
                let cmd = line.trim().to_string();
                let first = cmd.split_whitespace().next().unwrap_or("");
                match first {
                    "stop" => stop.store(true, Ordering::SeqCst),
                    "quit" => {
                        stop.store(true, Ordering::SeqCst);
                        let _ = tx.send(cmd);
                        break;
                    }
                    "isready" if searching.load(Ordering::SeqCst) => println!("readyok"),
                    "go" => {
                        stop.store(false, Ordering::SeqCst);
                        searching.store(true, Ordering::SeqCst);
                        let _ = tx.send(cmd);
                    }
                    "ponderhit" => {}
                    _ => {
                        let _ = tx.send(cmd);
                    }
                }
            }
        });
    }

    let mut uci = Uci {
        pos: Position::from_fen(START_FEN).unwrap(),
        hist: Vec::new(),
        searcher: Searcher::new(16, stop.clone()),
        overhead: 20,
    };
    let mut hash_mb = 16usize;
    while let Ok(cmd) = rx.recv() {
        let toks: Vec<&str> = cmd.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }
        match toks[0] {
            "uci" => {
                println!("id name {}", NAME);
                println!("id author {}", AUTHOR);
                println!("option name Hash type spin default 16 min 1 max 65536");
                println!("option name Threads type spin default 1 min 1 max 1");
                println!("option name MoveOverhead type spin default 20 min 0 max 5000");
                if std::env::var("OPUS_TUNE").is_ok() {
                    params::print_options();
                }
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "ucinewgame" => uci.searcher.clear(),
            "setoption" => {
                // setoption name <id> value <x>
                let lower: Vec<String> = toks.iter().map(|s| s.to_lowercase()).collect();
                let ni = lower.iter().position(|s| s == "name");
                let vi = lower.iter().position(|s| s == "value");
                if let (Some(ni), Some(vi)) = (ni, vi) {
                    let name = lower[ni + 1..vi].join(" ");
                    let val = toks.get(vi + 1).copied().unwrap_or("");
                    match name.as_str() {
                        "hash" => {
                            if let Ok(mb) = val.parse::<usize>() {
                                let mb = mb.clamp(1, 65536);
                                if mb != hash_mb {
                                    hash_mb = mb;
                                    uci.searcher.tt = tt::TT::new(mb);
                                }
                            }
                        }
                        "moveoverhead" => {
                            if let Ok(v) = val.parse::<i64>() {
                                uci.overhead = v.clamp(0, 5000);
                            }
                        }
                        other => {
                            if let Ok(v) = val.parse::<i32>() {
                                if params::set(other, v) {
                                    uci.searcher.init_lmr();
                                }
                            }
                        }
                    }
                }
            }
            "position" => uci.set_position(&toks),
            "go" => {
                uci.go(&toks);
                searching.store(false, Ordering::SeqCst);
            }
            "d" => println!("{}", uci.pos.to_fen()),
            "quit" => break,
            _ => {}
        }
    }
}
