// Rebuilds games from the engine's old `datagen` output (32-byte records:
// occupancy, piece nibbles, white-relative score, result, side to move).
//
// Records were written in game order, one file per datagen thread, but only
// for positions that were not in check and whose best move was quiet, so the
// moves between consecutive records are missing. They are recovered with a
// short search: the move from a recorded position must be quiet, and every
// skipped position in between must be in check or have played a noisy move.
// Moves from skipped positions are written with NO_SCORE. A chain ends (and a
// new one starts) when the result changes, the piece count goes up, or no path
// of up to MAX_GAP plies is found. (Long capture sequences skip several plies;
// 7 keeps ~98.8% of records at ~20 s per 1.5M-record file, where 3 kept 96%.) The last record of each chain has no known
// move, so it is dropped.
//
// The records don't store castling rights, en passant, or the move counters:
// castling is inferred from kings and rooks on their home squares, en passant
// is left unset, the halfmove clock starts at 0, and the fullmove number is
// estimated (datagen games started after 8-9 random plies, i.e. move 5).
use crate::position::*;
use crate::viri::{self, Game};

pub const MAX_GAP: usize = 7;
const GAME_START_FULLMOVE: u16 = 5;

pub struct Record {
    pub board: [u8; 64],
    pub stm: usize,
    pub score: i16,
    pub result: u8,
    pub pieces: u32,
}

pub fn decode(r: &[u8]) -> Record {
    let occ = u64::from_le_bytes(r[0..8].try_into().unwrap());
    let mut board = [NONE_PC; 64];
    let mut bits = occ;
    let mut i = 0;
    while bits != 0 {
        let sq = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        board[sq] = (r[8 + i / 2] >> ((i & 1) * 4)) & 15;
        i += 1;
    }
    Record {
        board,
        stm: r[27] as usize,
        score: i16::from_le_bytes([r[24], r[25]]),
        result: r[26],
        pieces: occ.count_ones(),
    }
}

/// Builds a Position for a record, inferring castling rights from home squares.
fn position_of(rec: &Record, fullmove: u16) -> Option<Position> {
    let mut fen = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            let pc = rec.board[rank * 8 + file];
            if pc == NONE_PC {
                empty += 1;
                continue;
            }
            if empty > 0 {
                fen.push((b'0' + empty) as char);
                empty = 0;
            }
            let ch = b"pnbrqk"[pc_type(pc)] as char;
            fen.push(if pc_color(pc) == WHITE { ch.to_ascii_uppercase() } else { ch });
        }
        if empty > 0 {
            fen.push((b'0' + empty) as char);
        }
        if rank > 0 {
            fen.push('/');
        }
    }
    let b = &rec.board;
    let (wk, wr, bk, br) = (make_pc(KING, WHITE), make_pc(ROOK, WHITE), make_pc(KING, BLACK), make_pc(ROOK, BLACK));
    let mut castling = String::new();
    if b[4] == wk && b[7] == wr {
        castling.push('K');
    }
    if b[4] == wk && b[0] == wr {
        castling.push('Q');
    }
    if b[60] == bk && b[63] == br {
        castling.push('k');
    }
    if b[60] == bk && b[56] == br {
        castling.push('q');
    }
    if castling.is_empty() {
        castling.push('-');
    }
    let stm = if rec.stm == WHITE { 'w' } else { 'b' };
    Position::from_fen(&format!("{} {} {} - 0 {}", fen, stm, castling, fullmove))
}

fn matches(pos: &Position, target: &Record) -> bool {
    pos.stm == target.stm && pos.board == target.board
}

/// Depth-first search for a path of exactly `depth` plies from `pos` to `target`.
/// `first` marks the recorded position, whose move must be quiet.
fn search(pos: &Position, target: &Record, depth: usize, first: bool, path: &mut Vec<Move>) -> bool {
    let mut list = MoveList::new();
    // A skipped position not in check played a noisy move.
    pos.gen_moves(&mut list, !first && pos.checkers == 0);
    for i in 0..list.len {
        let m = list.moves[i];
        if first && is_noisy(m) {
            continue;
        }
        // The first mover's origin square can't be refilled by the later
        // (noisy or evasion) moves, so it must differ in the target.
        if first && target.board[mfrom(m)] == pos.board[mfrom(m)] {
            continue;
        }
        let mut child = *pos;
        if !child.make_move(m) {
            continue;
        }
        if child.occ().count_ones() < target.pieces {
            continue;
        }
        path.push(m);
        if depth == 1 {
            if matches(&child, target) {
                return true;
            }
        } else if search(&child, target, depth - 1, false, path) {
            return true;
        }
        path.pop();
    }
    false
}

fn find_path(pos: &Position, target: &Record) -> Option<Vec<Move>> {
    let mut path = Vec::with_capacity(MAX_GAP);
    (1..=MAX_GAP).find(|&d| search(pos, target, d, true, &mut path)).map(|_| path)
}

#[derive(Default)]
pub struct Stats {
    pub records: u64,
    pub games: u64,
    pub moves: u64,
    pub unscored: u64,
    pub dropped: u64,
    pub gaps: [u64; MAX_GAP + 1],
    pub restarts: u64,
}

/// A game being rebuilt: `cur` is the last recorded position, whose score
/// is held in `pending` until the move played from it is found.
struct Chain {
    game: Game,
    cur: Position,
    result: u8,
    pieces: u32,
    pending: i16,
}

fn result_wdl(result: u8) -> u8 {
    match result {
        0 => viri::WDL_BLACK_WIN,
        1 => viri::WDL_DRAW,
        _ => viri::WDL_WHITE_WIN,
    }
}

/// Rebuilds games from a buffer of records, calling `emit` for each game.
pub fn rebuild(data: &[u8], stats: &mut Stats, mut emit: impl FnMut(&Game)) {
    let n = data.len() / 32;
    stats.records += n as u64;
    let mut chain: Option<Chain> = None;
    let mut fullmove = GAME_START_FULLMOVE;
    let mut finish = |c: Chain, stats: &mut Stats| {
        stats.dropped += 1; // the chain's last record has no known move
        if !c.game.moves.is_empty() {
            stats.games += 1;
            stats.moves += c.game.moves.len() as u64;
            stats.unscored += c.game.moves.iter().filter(|m| m.1 == viri::NO_SCORE).count() as u64;
            emit(&c.game);
        }
    };
    for k in 0..n {
        let rec = decode(&data[k * 32..k * 32 + 32]);
        if let Some(c) = chain.as_mut() {
            if rec.result == c.result && rec.pieces <= c.pieces {
                if let Some(path) = find_path(&c.cur, &rec) {
                    stats.gaps[path.len()] += 1;
                    for (j, &m) in path.iter().enumerate() {
                        c.game.moves.push((m, if j == 0 { c.pending } else { viri::NO_SCORE }));
                        c.cur.make_move(m);
                    }
                    c.pending = rec.score;
                    c.pieces = rec.pieces;
                    continue;
                }
                stats.restarts += 1;
                fullmove = c.cur.fullmove.saturating_add(1);
            } else {
                fullmove = GAME_START_FULLMOVE;
            }
            finish(chain.take().unwrap(), stats);
        }
        match position_of(&rec, fullmove) {
            Some(start) => {
                let game = Game { start, wdl: result_wdl(rec.result), moves: Vec::new() };
                chain = Some(Chain { game, cur: start, result: rec.result, pieces: rec.pieces, pending: rec.score });
            }
            None => stats.dropped += 1,
        }
    }
    if let Some(c) = chain.take() {
        finish(c, stats);
    }
}
