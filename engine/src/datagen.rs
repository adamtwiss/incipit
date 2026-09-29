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
