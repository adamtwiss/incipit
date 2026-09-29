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
    // Rooks that still carry castling rights: K=h1, Q=a1, k=h8, q=a8.
    let castle_rooks = [(1u8, 7usize), (2, 0), (4, 63), (8, 56)];
    let mut i = 0;
    let mut bits = occ;
    while bits != 0 {
        let sq = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let pc = pos.board[sq];
        let colour = pc_color(pc) as u8;
        let mut code = pc_type(pc) as u8;
        if code == ROOK as u8 && castle_rooks.iter().any(|&(right, rsq)| pos.castling & right != 0 && rsq == sq) {
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
/// 3 promotion). Castling is written king-takes-rook (e1->h1, e1->a1).
pub fn encode_move(m: Move) -> u16 {
    let from = mfrom(m) as u16;
    let mut to = mto(m) as u16;
    let (promo, kind) = match mflag(m) {
        F_KCASTLE => {
            to = from | 7;
            (0, 2)
        }
        F_QCASTLE => {
            to = from & !7;
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
            w.write_all(&encode_move(m).to_le_bytes())?;
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
