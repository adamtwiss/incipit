// Self-play data generation for NNUE training.
use crate::position::*;
use crate::search::*;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// 32-byte packed position: occupancy, piece nibbles, white-relative score, result (0=black win,1=draw,2=white win), stm.
pub fn pack(pos: &Position, score_white: i16, result: u8) -> [u8; 32] {
    let mut out = [0u8; 32];
    let occ = pos.occ();
    out[0..8].copy_from_slice(&occ.to_le_bytes());
    let mut i = 0;
    for sq in crate::attacks::Bits(occ) {
        let pc = pos.board[sq];
        out[8 + i / 2] |= pc << ((i & 1) * 4);
        i += 1;
    }
    out[24..26].copy_from_slice(&score_white.to_le_bytes());
    out[26] = result;
    out[27] = pos.stm as u8;
    out
}

fn legal_moves(pos: &Position) -> Vec<Move> {
    let mut list = MoveList::new();
    pos.gen_moves(&mut list, false);
    let mut v = Vec::new();
    for i in 0..list.len {
        let mut c = *pos;
        if c.make_move(list.moves[i]) {
            v.push(list.moves[i]);
        }
    }
    v
}

fn insufficient(pos: &Position) -> bool {
    let n = pos.occ().count_ones();
    if n == 2 {
        return true;
    }
    n == 3 && (pos.pieces[KNIGHT] | pos.pieces[BISHOP]) != 0
}

pub fn run(threads: usize, prefix: &str, nodes: u64, seconds: u64) {
    let total_pos = Arc::new(AtomicU64::new(0));
    let total_games = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let mut hs = Vec::new();
    for t in 0..threads {
        let prefix = prefix.to_string();
        let total_pos = total_pos.clone();
        let total_games = total_games.clone();
        hs.push(std::thread::spawn(move || {
            let seed = (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64)
                ^ ((t as u64 + 1) * 0x9E3779B97F4A7C15);
            let mut rng = Rng(seed | 1);
            let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{}_{}.bin", prefix, t)).unwrap());
            let mut s = Searcher::new(8, Arc::new(AtomicBool::new(false)));
            s.silent = true;
            let lim = Limits { soft_ms: None, hard_ms: None, depth: 64, nodes: Some(nodes) };
            while start.elapsed().as_secs() < seconds {
                // random opening
                let mut pos = Position::from_fen(START_FEN).unwrap();
                let mut hist: Vec<u64> = Vec::new();
                let nrand = 8 + (rng.next() % 2) as usize;
                let mut ok = true;
                for _ in 0..nrand {
                    let mv = legal_moves(&pos);
                    if mv.is_empty() {
                        ok = false;
                        break;
                    }
                    let m = mv[(rng.next() % mv.len() as u64) as usize];
                    hist.push(pos.hash);
                    pos.make_move(m);
                }
                if !ok || legal_moves(&pos).is_empty() {
                    continue;
                }
                s.clear();
                s.hash_hist = hist.clone();
                let (_, sc) = s.search(&pos, &lim);
                if sc.abs() > 500 {
                    continue;
                }
                let mut recs: Vec<(Position, i16)> = Vec::new();
                let result: u8;
                let mut win_cnt = 0;
                let mut draw_cnt = 0;
                let mut ply = 0;
                loop {
                    let mv = legal_moves(&pos);
                    if mv.is_empty() {
                        result = if pos.checkers != 0 { if pos.stm == WHITE { 0 } else { 2 } } else { 1 };
                        break;
                    }
                    if pos.halfmove >= 100 || insufficient(&pos) {
                        result = 1;
                        break;
                    }
                    // threefold (approx: one prior occurrence within reversible window counts twice)
                    let mut reps = 0;
                    let n = hist.len();
                    let mut k = 2;
                    while k <= (pos.halfmove as usize).min(n) {
                        if hist[n - k] == pos.hash {
                            reps += 1;
                        }
                        k += 2;
                    }
                    if reps >= 2 {
                        result = 1;
                        break;
                    }
                    s.hash_hist = hist.clone();
                    let (m, sc) = s.search(&pos, &lim);
                    if m == NO_MOVE {
                        result = 1;
                        break;
                    }
                    let ws = if pos.stm == WHITE { sc } else { -sc };
                    if sc.abs() >= 2000 {
                        win_cnt += 1;
                    } else {
                        win_cnt = 0;
                    }
                    if win_cnt >= 4 {
                        result = if ws > 0 { 2 } else { 0 };
                        break;
                    }
                    if ply >= 80 && sc.abs() <= 8 {
                        draw_cnt += 1;
                    } else {
                        draw_cnt = 0;
                    }
                    if draw_cnt >= 10 || ply >= 400 {
                        result = 1;
                        break;
                    }
                    if pos.checkers == 0 && !is_noisy(m) && sc.abs() < MATE_BOUND {
                        recs.push((pos, ws.clamp(-32000, 32000) as i16));
                    }
                    hist.push(pos.hash);
                    pos.make_move(m);
                    ply += 1;
                }
                for (p, ws) in recs.iter() {
                    f.write_all(&pack(p, *ws, result)).unwrap();
                }
                total_pos.fetch_add(recs.len() as u64, Ordering::Relaxed);
                let g = total_games.fetch_add(1, Ordering::Relaxed) + 1;
                if t == 0 && g % 200 < 1 {
                    let _ = f.flush();
                }
                if t == 0 && rng.next() % 50 == 0 {
                    let el = start.elapsed().as_secs_f64();
                    let p = total_pos.load(Ordering::Relaxed);
                    println!(
                        "games {} positions {} ({:.0} pos/s) elapsed {:.0}s",
                        total_games.load(Ordering::Relaxed),
                        p,
                        p as f64 / el,
                        el
                    );
                }
            }
            f.flush().unwrap();
        }));
    }
    for h in hs {
        h.join().unwrap();
    }
    println!("done games {} positions {}", total_games.load(Ordering::Relaxed), total_pos.load(Ordering::Relaxed));
}

/// OpenBench datagen openings:
///   genfens <N> seed <S> book <None|path.epd> [moves <K>] [nodes <K>] [maxeval <cp>]
/// Prints N lines `info string genfens <FEN>`, a deterministic function of the arguments.
/// Each opening is a book line (or the start position) plus `moves` or `moves + 1` random
/// plies (default 8 without a book, 2 with one). Positions in check, with no legal moves,
/// repeated within the run, or scoring beyond `maxeval` in a `nodes`-limited search are
/// rejected. Unrecognised arguments are ignored.
pub fn genfens(toks: &[&str]) {
    let n: usize = toks.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut seed = 0u64;
    let mut book_path: Option<&str> = None;
    let mut moves: Option<usize> = None;
    let mut nodes = 5000u64;
    let mut maxeval = 500i32;
    // dfrc: start each opening from a random double-Fischer-random position
    // (each side gets its own Chess960 back rank) instead of the book or the
    // start position. OB plays Chess960 when the book's name says FRC/960, so
    // pair this with such a book name.
    let mut dfrc = false;
    let mut i = 2;
    while i < toks.len() {
        let v = toks.get(i + 1).copied().unwrap_or("");
        let used = match toks[i] {
            "seed" => v.parse().map(|x| seed = x).is_ok(),
            "book" => {
                book_path = if v == "None" { None } else { Some(v) };
                !v.is_empty()
            }
            "moves" => v.parse().map(|x| moves = Some(x)).is_ok(),
            "nodes" => v.parse().map(|x: u64| nodes = x.max(1)).is_ok(),
            "maxeval" => v.parse().map(|x| maxeval = x).is_ok(),
            "dfrc" => {
                dfrc = true;
                i += 1;
                continue;
            }
            _ => false,
        };
        i += if used { 2 } else { 1 };
    }

    let book: Vec<Position> = match book_path {
        _ if dfrc => Vec::new(),
        None => vec![Position::from_fen(START_FEN).unwrap()],
        Some(path) => {
            let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
                eprintln!("genfens: cannot read book {}: {}", path, e);
                std::process::exit(1);
            });
            // EPD: first four fields are the position; any operations after them are ignored.
            let book: Vec<Position> = text
                .lines()
                .filter_map(|l| {
                    let f: Vec<&str> = l.split_whitespace().take(4).collect();
                    if f.len() == 4 { Position::from_fen(&f.join(" ")) } else { None }
                })
                .collect();
            if book.is_empty() {
                eprintln!("genfens: no positions in book {}", path);
                std::process::exit(1);
            }
            book
        }
    };
    let base = moves.unwrap_or(if book_path.is_some() && !dfrc { 2 } else { 8 });

    // splitmix64 so that nearby seeds (1, 2, 3...) give unrelated streams, and never zero.
    let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    let mut rng = Rng((z ^ (z >> 31)) | 1);

    let mut s = Searcher::new(8, Arc::new(AtomicBool::new(false)));
    s.silent = true;
    let lim = Limits { soft_ms: None, hard_ms: None, depth: 64, nodes: Some(nodes) };
    let mut seen = std::collections::HashSet::new();
    let stdout = std::io::stdout();
    let mut done = 0;
    while done < n {
        let mut pos = if dfrc { dfrc_start(&mut rng) } else { book[(rng.next() % book.len() as u64) as usize] };
        let nrand = base + (rng.next() % 2) as usize;
        let mut ok = true;
        for _ in 0..nrand {
            let mv = legal_moves(&pos);
            if mv.is_empty() {
                ok = false;
                break;
            }
            pos.make_move(mv[(rng.next() % mv.len() as u64) as usize]);
        }
        if !ok || pos.checkers != 0 || legal_moves(&pos).is_empty() || !seen.insert(pos.hash) {
            continue;
        }
        s.clear();
        s.hash_hist.clear();
        let (_, sc) = s.search(&pos, &lim);
        if sc.abs() > maxeval {
            continue;
        }
        let mut out = stdout.lock();
        let _ = writeln!(out, "info string genfens {}", pos.to_fen());
        let _ = out.flush();
        done += 1;
    }
}

/// A random Chess960 back rank: bishops on opposite-coloured squares and the
/// king between the two rooks (the Chess960 rules), as piece letters a..h.
fn chess960_rank(rng: &mut Rng) -> [u8; 8] {
    loop {
        let mut r = *b"RNBQKBNR";
        for i in (1..8).rev() {
            r.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        let b: Vec<usize> = (0..8).filter(|&i| r[i] == b'B').collect();
        let rk: Vec<usize> = (0..8).filter(|&i| r[i] == b'R').collect();
        let k = r.iter().position(|&c| c == b'K').unwrap();
        if (b[0] + b[1]) % 2 == 1 && rk[0] < k && k < rk[1] {
            return r;
        }
    }
}

/// Double Fischer random start: independent Chess960 back ranks for each side,
/// all four castling rights.
fn dfrc_start(rng: &mut Rng) -> Position {
    let w = chess960_rank(rng);
    let b = chess960_rank(rng);
    let rooks = |r: &[u8; 8]| -> String { (0..8).filter(|&i| r[i] == b'R').map(|i| (b'a' + i as u8) as char).collect() };
    let fen = format!(
        "{}/pppppppp/8/8/8/8/PPPPPPPP/{} w {}{} - 0 1",
        String::from_utf8(b.iter().map(|c| c.to_ascii_lowercase()).collect()).unwrap(),
        String::from_utf8(w.to_vec()).unwrap(),
        rooks(&w).to_uppercase(),
        rooks(&b)
    );
    Position::from_fen(&fen).unwrap()
}
