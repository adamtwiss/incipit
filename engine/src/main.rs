mod attacks;
mod datagen;
mod eval;
mod nnue;
mod params;
mod position;
mod search;
mod tb;
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

// Bench positions: 50 positions sampled from Incipit's own self-play (clean
// generation c4, OB datagen #3780) with `datatools fens 50 20260930`, spread
// over game phases and search-stressing kinds.
const BENCH_FENS: [&str; 50] = [
    // opening (26-32 pieces)
    "r2qk2r/1b2npbp/ppnpp1p1/2p5/2P5/3P1NP1/PPN1PPBP/1RBQ1RK1 b kq - 3 11",
    "rnbqk2r/p1p2p2/4p2b/NP1n3p/1P1P2p1/1Q2PP2/3B2PP/3RKBNR b Kkq - 5 13",
    "r3k2r/ppqn1bpp/2p2n2/3pb3/3P2P1/7P/PPP3K1/RNB1QB1R w kq - 1 16",
    "r2qkbnr/pp2pppp/2b1n3/2p5/2P5/4P1PP/P2P1PBR/RNBQK1N1 b Qkq - 2 10",
    "r1bqkb1r/p4npp/n1p2p2/3p4/4p3/1PP2NP1/P1NPPPBP/R1BQK2R w KQkq - 0 10",
    "r2q1bnr/3k1p1p/8/p2pp1p1/Np6/5P1P/PPPP3P/R1BQK2R w KQ - 0 12",
    "2kr3r/1bp1n1q1/pp1p1p1p/3P1pp1/2P2P2/1PN1P3/P3BP1P/R2QK1R1 w - - 0 18",
    "r2nkbr1/pppbqp2/3p1p2/3P4/1P2P2p/2P1B1p1/P2NBPPP/2RQ1RK1 b q - 1 15",
    // middlegame (16-25)
    "r4rk1/3Bq3/4p2b/p1pp1p2/Pp1P2pP/4P3/P1PQN3/2R1K2R b - - 0 21",
    "2r2k1r/pp1q1ppp/2n1pn2/8/Q7/2N2N2/P2PPP1P/1R3RK1 b - - 1 18",
    "4rr2/1bp3k1/1p1p2p1/pP2np1p/P3P3/2NP4/2P3PP/3BRRK1 b - - 1 29",
    "6r1/1pp1k2p/3bbp1P/4p3/3pP3/P2B4/1P1B1P2/2K3R1 w - - 6 27",
    "6kb/2rp3p/p3p1p1/n1N5/4NP2/4P1P1/3P2KP/1R6 w - - 0 38",
    "2kr3r/p1p2p2/1p4pp/3b4/P4P1q/4Q1R1/1P4PP/1B3RK1 w - - 0 22",
    "6r1/1pk5/4b1p1/1BPpPp1p/3K1P1P/5P2/8/6R1 w - - 2 48",
    "1rbr4/pp4k1/2p2p2/4b1p1/N7/P4BPp/1PP1R2P/4R2K b - - 9 27",
    "3r3r/p1pqnkp1/b4p2/3P4/PbN1B3/4B1Pp/1PQ2P1P/1R4KR b - - 8 21",
    "4r1k1/pBB2pp1/8/2b4p/6b1/2PK4/PP2r1PP/R6R w - - 3 20",
    "rb4R1/8/1pk2p2/1Np4n/2P4P/p4P2/P3NK2/8 w - - 0 32",
    "r3kr2/1ppbqp1n/p3P3/7p/5P2/PP3Q1P/1B2N1P1/2KR3R b q - 0 25",
    // endgame (8-15)
    "8/1k6/1P4p1/3K1bPp/5P1P/8/8/6B1 b - - 48 98",
    "8/2p2r2/p1pp3P/8/Pk1P4/4KR2/1B6/1b6 b - - 1 44",
    "8/5p1p/3k4/p6R/P4K2/8/6r1/8 b - - 1 46",
    "8/6k1/R7/p2r1BKp/8/6p1/8/8 w - - 0 54",
    "8/3k4/2Np4/2n1b3/2P5/8/2K3B1/8 w - - 27 79",
    "2R5/5b2/5k2/r1p1p3/1p2P3/1P1P4/1KP3N1/8 w - - 0 45",
    "5k2/3R4/4pp2/4p1p1/6P1/4PK1P/3P4/1r6 w - - 0 42",
    "4k3/p7/5p2/4p1rP/4P2K/1Pr3P1/P3R1R1/8 b - - 2 29",
    "8/1R6/6pk/5p1p/2r4P/6PK/8/8 w - - 18 85",
    "8/1kr1rp2/1p3R2/p7/P7/1P6/1KP2R2/8 w - - 8 45",
    // late endgame (3-7)
    "8/8/3B4/3b4/p7/P3K1pk/8/8 w - - 82 103",
    "8/Br6/8/8/4k3/6K1/8/8 w - - 54 113",
    "8/8/4p3/4k3/1R2P3/4P3/5K2/3r4 w - - 12 132",
    "8/8/8/3k4/3p3R/2r5/3K4/8 w - - 14 65",
    "5k2/7R/r7/6PK/8/8/8/8 b - - 10 106",
    "8/6k1/1R6/4K1p1/8/8/5r2/8 w - - 2 65",
    // tactical (5+ legal captures)
    "r3k2r/3pbppp/2b1pn2/1N1n4/4PP2/P1N5/2BB2PP/R3K2R b KQk - 2 19",
    "1k1r4/1bp5/pp3q1p/P4p2/1QPrnN1P/1P1NRpP1/5P1K/5R2 b - - 2 31",
    "2b2r1k/rp1nq1pp/p1p5/3p1p2/1PPP1Nn1/P2BP3/3N1P1P/R2QK1R1 w Q - 2 16",
    "r6r/1kp1qpb1/1pn3p1/1p1pp3/1P1P4/P1PNPPB1/1K3QPP/2R2R2 b - - 0 26",
    "r3q2r/1pp3p1/p1np2k1/P3PRBp/1P5P/2P5/3N2P1/R2Q2K1 b - - 0 20",
    "r4rk1/p5b1/4p1q1/R2p4/3P1nQ1/4N2P/1P1B1PP1/5R1K w - - 1 28",
    // closed (3+ blocked pawn pairs)
    "4b1k1/p1r1r1p1/1p3p1p/P2p1P1P/3PpP2/2P1P3/R3K3/3N3R b - - 0 36",
    "5n2/4kpb1/4p1p1/2Pp2P1/1BrP1PBp/3K3P/4N3/3R4 b - - 0 58",
    "r6r/1bqpb1k1/4p1p1/pp2PpNp/PPpP1PnP/2P2B2/R2BQ1P1/4R1K1 w - - 0 23",
    "1k6/6R1/p2p2p1/P2Pp1P1/1pP1r2P/1P6/6K1/8 b - - 0 53",
    "r2q1rk1/1p2b1p1/4bn1p/2p2p2/PpP1pP2/1P2P1P1/1BQ1BN1P/R4RK1 b - - 1 19",
    // pawn endgame
    "8/p2k4/P7/2pPPppp/2P4P/6K1/8/8 b - - 0 40",
    "8/8/8/8/7p/5k1p/8/7K w - - 6 65",
    "8/8/p3k3/1p2p1P1/4K3/P3P3/8/8 b - - 0 63",
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

fn bench(depth: i32, eval_cache_kb: usize) {
    let mut s = Searcher::new(16, Arc::new(AtomicBool::new(false)));
    s.set_eval_cache_kb(eval_cache_kb);
    s.silent = true;
    let t = Instant::now();
    let mut nodes = 0;
    for f in BENCH_FENS.iter() {
        let pos = Position::from_fen(f).unwrap();
        s.clear();
        s.hash_hist.clear();
        let lim = Limits { soft_ms: None, hard_ms: None, depth, nodes: None };
        let (m, sc) = s.search(&pos, &lim);
        println!("{} -> {} {}", f, pos.move_uci(m), sc);
        nodes += s.nodes;
    }
    let el = t.elapsed().as_secs_f64();
    print!("\n=== Search stats (all bench positions, depth {}) ===\n{}", depth, s.stats.report(nodes));
    // Keep this line last: OpenBench reads "<N> nodes <M> nps".
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
        println!("bestmove {}", self.pos.move_uci(m));
    }
}

fn main() {
    attacks::init();
    nnue::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 {
        // OpenBench runs `./incipit "genfens N seed S book B EXTRA" quit`.
        let toks: Vec<&str> = args[1].split_whitespace().collect();
        if toks.first() == Some(&"genfens") {
            datagen::genfens(&toks);
            return;
        }
        match args[1].as_str() {
            "tune-spec" => {
                params::print_spec();
                return;
            }
            "perftsuite" => {
                perft_suite(&args[2], args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5));
                return;
            }
            "datagen" => {
                // datagen <threads> <prefix> <nodes> <seconds>
                datagen::run(args[2].parse().unwrap(), &args[3], args[4].parse().unwrap(), args[5].parse().unwrap());
                return;
            }
            "l1perm" if args.len() >= 5 => {
                if let Err(e) = nnue::l1perm(&args[2], &args[3], &args[4]) {
                    eprintln!("error: {}", e);
                }
                return;
            }
            "l1stats" if args.len() >= 4 => {
                if let Err(e) = nnue::l1stats(&args[2], &args[3]) {
                    eprintln!("error: {}", e);
                }
                return;
            }
            "l1check" => {
                println!("l1check mismatches {}", nnue::l1check(200));
                return;
            }
            "nnuecheck" => {
                let mut seed = 12345u64;
                let mut bad = 0;
                let mut cache = nnue::RefreshCache::new();
                for _g in 0..200 {
                    let mut pos = Position::from_fen(START_FEN).unwrap();
                    let mut acc = nnue::Acc::new();
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
                        let mut a2 = nnue::Acc::new();
                        a2.update_from(&acc, &pos, &child, m, &mut cache);
                        let mut a3 = nnue::Acc::new();
                        a3.refresh(&child);
                        if a2.v != a3.v { bad += 1; println!("mismatch {} {}", pos.to_fen(), move_str(m)); }
                        pos = child; acc = a3;
                    }
                }
                println!("bad {}", bad);
                println!("startpos eval {}", nnue::evaluate(&{ let mut a = nnue::Acc::new(); a.refresh(&Position::from_fen(START_FEN).unwrap()); a }, &Position::from_fen(START_FEN).unwrap()));
                return;
            }
            "bench" => {
                // `bench [depth] [Param=value ...]`, e.g. `bench 13 UseProbcut=0` for ablations
                // (also EvalCacheKB=N).
                let mut eval_cache_kb = search::EVAL_CACHE_KB;
                for a in args.iter().skip(3) {
                    if let Some((k, v)) = a.split_once('=') {
                        if k.eq_ignore_ascii_case("EvalCacheKB") {
                            eval_cache_kb = v.parse().unwrap_or(eval_cache_kb);
                            continue;
                        }
                        if !v.parse().is_ok_and(|v| params::set(k, v)) {
                            eprintln!("unknown parameter {a}");
                            std::process::exit(1);
                        }
                    }
                }
                bench(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(13), eval_cache_kb);
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
                println!("option name EvalCacheKB type spin default {} min 16 max 1048576", search::EVAL_CACHE_KB);
                println!("option name Threads type spin default 1 min 1 max 1");
                println!("option name MoveOverhead type spin default 20 min 0 max 5000");
                println!("option name UCI_Chess960 type check default false");
                println!("option name UCI_ShowWDL type check default false");
                println!("option name TmLog type check default false");
                println!("option name SyzygyPath type string default <empty>");
                if cfg!(feature = "tune") {
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
                        "evalcachekb" => {
                            if let Ok(kb) = val.parse::<usize>() {
                                uci.searcher.set_eval_cache_kb(kb.clamp(16, 1 << 20));
                            }
                        }
                        "hash" => {
                            if let Ok(mb) = val.parse::<usize>() {
                                let mb = mb.clamp(1, 65536);
                                if mb != hash_mb {
                                    hash_mb = mb;
                                    uci.searcher.tt = tt::TT::new(mb);
                                }
                            }
                        }
                        "syzygypath" => {
                            // The path keeps its case and may contain spaces.
                            let path = toks[vi + 1..].join(" ");
                            let n = tb::init(&path);
                            println!("info string syzygy: {}-man tables loaded", n);
                        }
                        "uci_showwdl" => {
                            search::SHOW_WDL.store(val.eq_ignore_ascii_case("true"), Ordering::Relaxed);
                        }
                        "tmlog" => search::TM_LOG.store(val.eq_ignore_ascii_case("true"), Ordering::Relaxed),
                        "uci_chess960" => {
                            CHESS960.store(val.eq_ignore_ascii_case("true"), Ordering::Relaxed);
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
            // Legal moves of the current position, in UCI notation.
            "legal" => {
                let mut list = MoveList::new();
                uci.pos.gen_moves(&mut list, false);
                let mut v: Vec<String> = (0..list.len)
                    .filter(|&i| uci.pos.clone().make_move(list.moves[i]))
                    .map(|i| uci.pos.move_uci(list.moves[i]))
                    .collect();
                v.sort();
                println!("legal {}", v.join(" "));
            }
            // Static NNUE eval of the current position (side to move's view, cp).
            "eval" => {
                let mut acc = nnue::Acc::new();
                acc.refresh(&uci.pos);
                println!("eval {}", nnue::evaluate(&acc, &uci.pos));
            }
            "genfens" => datagen::genfens(&toks),
            // Tablebase result of the current position: tbprobe [path]
            "tbprobe" => {
                if toks.len() > 1 {
                    tb::init(&toks[1..].join(" "));
                }
                let mut z = uci.pos;
                z.halfmove = 0;
                println!(
                    "tbprobe largest {} wdl {:?} root {:?}",
                    tb::largest(),
                    tb::probe_wdl(&z),
                    tb::probe_root(&uci.pos).map(|(w, m)| (w, uci.pos.move_uci(m)))
                );
            }
            "tune-spec" => params::print_spec(),
            "quit" => break,
            _ => {}
        }
    }
}
