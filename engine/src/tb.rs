//! Syzygy tablebases through the vendored Fathom library (third_party/fathom,
//! MIT), used only through its public API (tbprobe.h).

use crate::position::*;
use std::ffi::CString;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

extern "C" {
    fn tb_init(path: *const std::ffi::c_char) -> bool;
    fn tb_probe_wdl_impl(
        white: u64, black: u64, kings: u64, queens: u64, rooks: u64, bishops: u64, knights: u64, pawns: u64,
        ep: u32, turn: bool,
    ) -> u32;
    fn tb_probe_root_impl(
        white: u64, black: u64, kings: u64, queens: u64, rooks: u64, bishops: u64, knights: u64, pawns: u64,
        rule50: u32, ep: u32, turn: bool, results: *mut u32,
    ) -> u32;
    static TB_LARGEST: u32;
}

const RESULT_FAILED: u32 = 0xFFFF_FFFF;

/// Largest piece count the loaded tables cover (0: none loaded).
static LARGEST: AtomicU32 = AtomicU32::new(0);

/// Cache of recent WDL probe results, indexed by the position's hash: each
/// entry is the hash with its low two bits replaced by the result (1 loss,
/// 2 draw, 3 win; 0 empty). 2^16 entries, 512 KB.
const CACHE_SIZE: usize = 1 << 16;
static CACHE: [AtomicU64; CACHE_SIZE] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const EMPTY: AtomicU64 = AtomicU64::new(0);
    [EMPTY; CACHE_SIZE]
};

/// Win-draw-loss for the side to move. Cursed wins and blessed losses (won or
/// lost, but drawn under the fifty-move rule) count as draws.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Wdl {
    Loss,
    Draw,
    Win,
}

fn wdl_of(v: u32) -> Wdl {
    match v {
        0 => Wdl::Loss,
        4 => Wdl::Win,
        _ => Wdl::Draw,
    }
}

/// Loads the tables under `path` (directories separated by ':' or ';' on
/// Windows; empty or "<empty>" unloads). Returns the largest piece count found.
/// Not thread-safe: call only while no search runs.
pub fn init(path: &str) -> u32 {
    let path = if path == "<empty>" { "" } else { path };
    let c = CString::new(path).unwrap_or_default();
    let n = unsafe {
        if tb_init(c.as_ptr()) {
            TB_LARGEST
        } else {
            0
        }
    };
    LARGEST.store(n, Ordering::Relaxed);
    for e in CACHE.iter() {
        e.store(0, Ordering::Relaxed);
    }
    n
}

#[inline(always)]
pub fn largest() -> u32 {
    LARGEST.load(Ordering::Relaxed)
}

fn ep_of(pos: &Position) -> u32 {
    if pos.ep == NO_SQ { 0 } else { pos.ep as u32 }
}

/// WDL probe, for search. Fathom's WDL tables are only valid right after a
/// capture or pawn move (halfmove 0) and without castling rights.
#[inline]
pub fn probe_wdl(pos: &Position) -> Option<Wdl> {
    if pos.occ().count_ones() > largest() || pos.castling != 0 || pos.halfmove != 0 {
        return None;
    }
    let slot = &CACHE[pos.hash as usize & (CACHE_SIZE - 1)];
    let e = slot.load(Ordering::Relaxed);
    if e & !3 == pos.hash & !3 && e & 3 != 0 {
        return Some([Wdl::Loss, Wdl::Draw, Wdl::Win][(e & 3) as usize - 1]);
    }
    let p = &pos.pieces;
    let v = unsafe {
        tb_probe_wdl_impl(
            pos.colors[WHITE], pos.colors[BLACK], p[KING], p[QUEEN], p[ROOK], p[BISHOP], p[KNIGHT], p[PAWN],
            ep_of(pos), pos.stm == WHITE,
        )
    };
    if v == RESULT_FAILED {
        return None;
    }
    let w = wdl_of(v);
    let code = match w {
        Wdl::Loss => 1,
        Wdl::Draw => 2,
        Wdl::Win => 3,
    };
    slot.store((pos.hash & !3) | code, Ordering::Relaxed);
    Some(w)
}

/// Root probe (DTZ): the WDL of the position and a move that keeps it,
/// taking the fifty-move counter into account. Not thread-safe.
pub fn probe_root(pos: &Position) -> Option<(Wdl, Move)> {
    if pos.occ().count_ones() > largest() || pos.castling != 0 {
        return None;
    }
    let p = &pos.pieces;
    let v = unsafe {
        tb_probe_root_impl(
            pos.colors[WHITE], pos.colors[BLACK], p[KING], p[QUEEN], p[ROOK], p[BISHOP], p[KNIGHT], p[PAWN],
            pos.halfmove as u32, ep_of(pos), pos.stm == WHITE, std::ptr::null_mut(),
        )
    };
    if v == RESULT_FAILED {
        return None;
    }
    // TB_RESULT layout (tbprobe.h): wdl bits 0-3, to 4-9, from 10-15, promotes 16-18.
    let (to, from, promo) = ((v >> 4) & 63, (v >> 10) & 63, (v >> 16) & 7);
    if from == to {
        return None; // checkmate or stalemate: no move
    }
    let uci = format!(
        "{}{}{}",
        sq_str(from as usize),
        sq_str(to as usize),
        ["", "q", "r", "b", "n"][promo as usize]
    );
    let m = pos.parse_move(&uci)?;
    Some((wdl_of(v & 15), m))
}
