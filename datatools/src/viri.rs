// viriformat writer (and a minimal reader for `stats`).
//
// A file is a sequence of games. A game is a 32-byte packed start position
// (marlinformat), then (move: u16, score: i16) pairs, then four zero bytes.
// All integers are little-endian; squares are A1=0 .. H8=63. Scores are
// white-relative centipawns for the position the move is played from.
use crate::position::*;
use std::io::{self, Write};

/// Score for positions that must not be trained on (a mate, or a position the
/// data source didn't label). Trainers' |eval| filters drop it.
pub const NO_SCORE: i16 = i16::MAX;

pub const WDL_BLACK_WIN: u8 = 0;
pub const WDL_DRAW: u8 = 1;
pub const WDL_WHITE_WIN: u8 = 2;

const UNMOVED_ROOK: u8 = 6;

/// Packs `pos` as a 32-byte marlinformat board.
pub fn pack_board(pos: &Position, score: i16, wdl: u8) -> [u8; 32] {
    let mut out = [0u8; 32];
    let occ = pos.occ();
    out[0..8].copy_from_slice(&occ.to_le_bytes());
    // Rooks that still carry castling rights (any file, for Chess960).
    let mut i = 0;
    let mut bits = occ;
    while bits != 0 {
        let sq = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let pc = pos.board[sq];
        let colour = pc_color(pc) as u8;
        let mut code = pc_type(pc) as u8;
        if code == ROOK as u8 && (0..4).any(|r| pos.castling & (1 << r) != 0 && pos.rook_sq[r] as usize == sq) {
            code = UNMOVED_ROOK;
        }
        out[8 + i / 2] |= (code | colour << 3) << ((i & 1) * 4);
        i += 1;
    }
    out[24] = (pos.stm as u8) << 7 | pos.ep;
    out[25] = pos.halfmove.min(255) as u8;
    out[26..28].copy_from_slice(&pos.fullmove.to_le_bytes());
    out[28..30].copy_from_slice(&score.to_le_bytes());
    out[30] = wdl;
    out
}

/// Encodes an Incipit move: 6-bit from, 6-bit to, 2-bit promotion piece
/// (N=0, B=1, R=2, Q=3), 2-bit type (0 normal, 1 en passant, 2 castle,
/// 3 promotion). Castling is written king-takes-rook (e1->h1, e1->a1), the
/// rook's square coming from `pos` (any position of the same game).
pub fn encode_move(pos: &Position, m: Move) -> u16 {
    let from = mfrom(m) as u16;
    let mut to = mto(m) as u16;
    let (promo, kind) = match mflag(m) {
        f @ (F_KCASTLE | F_QCASTLE) => {
            let us = if from < 8 { WHITE } else { BLACK };
            to = pos.rook_sq[castle_right(us, f == F_QCASTLE)] as u16;
            (0, 2)
        }
        F_EP => (0, 1),
        _ if is_promo(m) => (promo_pt(m) as u16 - 1, 3),
        _ => (0, 0),
    };
    from | to << 6 | promo << 12 | kind << 14
}

/// One game: a start position, its result, and (move, white-relative score)
/// for each position from which a move was played.
pub struct Game {
    pub start: Position,
    pub wdl: u8,
    pub moves: Vec<(Move, i16)>,
}

impl Game {
    pub fn write(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&pack_board(&self.start, 0, self.wdl))?;
        for &(m, score) in &self.moves {
            w.write_all(&encode_move(&self.start, m).to_le_bytes())?;
            w.write_all(&score.to_le_bytes())?;
        }
        w.write_all(&[0; 4])
    }
}

/// Counts (games, moves) in a viriformat byte stream, checking that every
/// game is terminated. Scores equal to NO_SCORE are counted separately.
pub fn count(data: &[u8]) -> Result<(u64, u64, u64), String> {
    let (mut games, mut moves, mut unscored) = (0, 0, 0);
    let mut i = 0;
    while i < data.len() {
        if i + 32 > data.len() {
            return Err(format!("truncated board at byte {}", i));
        }
        i += 32;
        loop {
            if i + 4 > data.len() {
                return Err(format!("unterminated game at byte {}", i));
            }
            if data[i..i + 4] == [0; 4] {
                i += 4;
                break;
            }
            if i16::from_le_bytes([data[i + 2], data[i + 3]]) == NO_SCORE {
                unscored += 1;
            }
            moves += 1;
            i += 4;
        }
        games += 1;
    }
    Ok((games, moves, unscored))
}

/// Rebuilds a Position from a packed board (the inverse of pack_board).
/// Returns None if the board isn't valid.
pub fn unpack_board(b: &[u8]) -> Option<(Position, i16, u8)> {
    let occ = u64::from_le_bytes(b[0..8].try_into().unwrap());
    // 32 pieces fill the packed board; a WDL above 2 isn't a result.
    if occ.count_ones() > 32 || b[30] > 2 {
        return None;
    }
    let mut sqs = [None; 64];
    let mut castling = String::new();
    let (mut bits, mut i) = (occ, 0);
    while bits != 0 {
        let sq = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let code = (b[8 + i / 2] >> ((i & 1) * 4)) & 15;
        let colour = (code >> 3) as usize;
        let mut pt = (code & 7) as usize;
        if pt == UNMOVED_ROOK as usize {
            pt = ROOK;
            // A castling rook sits on its own back rank; name it by file
            // (Shredder-FEN), which covers standard and Chess960 positions.
            if sq / 8 != if colour == WHITE { 0 } else { 7 } {
                return None;
            }
            let f = (b'a' + (sq % 8) as u8) as char;
            castling.push(if colour == WHITE { f.to_ascii_uppercase() } else { f });
        }
        if pt > KING {
            return None;
        }
        sqs[sq] = Some((pt, colour));
        i += 1;
    }
    let mut fen = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            match sqs[rank * 8 + file] {
                None => empty += 1,
                Some((pt, c)) => {
                    if empty > 0 {
                        fen.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    let ch = b"pnbrqk"[pt] as char;
                    fen.push(if c == WHITE { ch.to_ascii_uppercase() } else { ch });
                }
            }
        }
        if empty > 0 {
            fen.push((b'0' + empty) as char);
        }
        if rank > 0 {
            fen.push('/');
        }
    }
    let castling: String = if castling.is_empty() { "-".into() } else { castling };
    let ep = b[24] & 127;
    let ep = if ep >= 64 { "-".to_string() } else { format!("{}{}", (b'a' + ep % 8) as char, (b'1' + ep / 8) as char) };
    let stm = if b[24] >> 7 == 0 { 'w' } else { 'b' };
    let fullmove = u16::from_le_bytes([b[26], b[27]]).max(1);
    let pos = Position::from_fen(&format!("{} {} {} {} {} {}", fen, stm, castling, ep, b[25], fullmove))?;
    Some((pos, i16::from_le_bytes([b[28], b[29]]), b[30]))
}

/// Visits every game in a viriformat buffer as its (position, move) list and
/// result, replaying each game with the engine's move generator.
pub fn for_each_game(data: &[u8], mut f: impl FnMut(&[(Position, Move)], u8)) -> Result<(), String> {
    let mut i = 0;
    let mut moves = Vec::new();
    while i < data.len() {
        if i + 32 > data.len() {
            return Err(format!("truncated board at byte {}", i));
        }
        let (mut pos, _, wdl) = unpack_board(&data[i..i + 32]).ok_or_else(|| format!("bad board at byte {}", i))?;
        i += 32;
        moves.clear();
        loop {
            if i + 4 > data.len() {
                return Err(format!("unterminated game at byte {}", i));
            }
            let raw = u16::from_le_bytes([data[i], data[i + 1]]);
            let score = i16::from_le_bytes([data[i + 2], data[i + 3]]);
            i += 4;
            if raw == 0 && score == 0 {
                break;
            }
            let mut list = MoveList::new();
            pos.gen_moves(&mut list, false);
            let m = (0..list.len())
                .map(|k| list[k])
                .find(|&m| {
                    encode_move(&pos, m) == raw && {
                        let mut c = pos;
                        c.make_move(m)
                    }
                })
                .ok_or_else(|| format!("illegal move {:04x} in {}", raw, pos.to_fen()))?;
            moves.push((pos, m));
            pos.make_move(m);
        }
        f(&moves, wdl);
    }
    Ok(())
}

/// Visits every (position, move, score) in a viriformat buffer, replaying each
/// game with the engine's move generator. Stops at the first undecodable game.
pub fn for_each_position(data: &[u8], mut f: impl FnMut(&Position, Move, i16, u8)) -> Result<(), String> {
    let mut i = 0;
    while i < data.len() {
        if i + 32 > data.len() {
            return Err(format!("truncated board at byte {}", i));
        }
        let (mut pos, _, wdl) = unpack_board(&data[i..i + 32]).ok_or_else(|| format!("bad board at byte {}", i))?;
        i += 32;
        loop {
            if i + 4 > data.len() {
                return Err(format!("unterminated game at byte {}", i));
            }
            let raw = u16::from_le_bytes([data[i], data[i + 1]]);
            let score = i16::from_le_bytes([data[i + 2], data[i + 3]]);
            i += 4;
            if raw == 0 && score == 0 {
                break;
            }
            let mut list = MoveList::new();
            pos.gen_moves(&mut list, false);
            let m = (0..list.len())
                .map(|k| list[k])
                .find(|&m| {
                    encode_move(&pos, m) == raw && {
                        let mut c = pos;
                        c.make_move(m)
                    }
                })
                .ok_or_else(|| format!("illegal move {:04x} in {}", raw, pos.to_fen()))?;
            f(&pos, m, score, wdl);
            pos.make_move(m);
        }
    }
    Ok(())
}
