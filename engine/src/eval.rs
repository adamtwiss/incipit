// Bootstrap hand-crafted evaluation: material + mobility only.
// Used only to generate the first generation of self-play training data (build with `--features hce`).
// Values are simple round numbers chosen from first principles; no piece-square tables.
use crate::attacks::{bishop_attacks, knight_attacks, rook_attacks, Bits};
use crate::position::*;

// pawn, knight, bishop, rook, queen, king
const MATERIAL: [i32; 6] = [100, 300, 300, 500, 900, 0];
// centipawns per square a piece attacks that is not occupied by its own side
const MOBILITY: [i32; 6] = [0, 4, 4, 2, 1, 0];
const TEMPO: i32 = 10;

fn side_score(pos: &Position, c: usize) -> i32 {
    let occ = pos.occ();
    let own = pos.colors[c];
    let mut s = 0;
    for pt in PAWN..=QUEEN {
        let bb = pos.pcs(c, pt);
        s += MATERIAL[pt] * bb.count_ones() as i32;
        if MOBILITY[pt] == 0 {
            continue;
        }
        for sq in Bits(bb) {
            let att = match pt {
                KNIGHT => knight_attacks(sq),
                BISHOP => bishop_attacks(sq, occ),
                ROOK => rook_attacks(sq, occ),
                _ => bishop_attacks(sq, occ) | rook_attacks(sq, occ),
            };
            s += MOBILITY[pt] * (att & !own).count_ones() as i32;
        }
    }
    s
}

/// Static evaluation from the side to move's point of view, in centipawns.
pub fn evaluate(pos: &Position) -> i32 {
    let us = pos.stm;
    side_score(pos, us) - side_score(pos, us ^ 1) + TEMPO
}
