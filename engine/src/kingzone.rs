// King-zone input features (owner idea, 2026-11-15; the trainer computes the
// same features in incipit-research train/src/kingzone.rs, written
// separately, and tests/kingzone_fixture.txt pins the indices for both).
//
// The threat inputs see attacks on occupied squares only; these see the empty
// squares next to each king. Per perspective, in its frame (ranks flipped for
// black, files mirrored when that side's king is on files e-h, as the piece
// and threat inputs): for each king (0 ours, 1 theirs), each of the 8 squares
// next to it (offset (dy + 1) * 3 + (dx + 1) in the frame, the king's own
// square left out) and each piece type of the king's enemy (pawn to king),
// one feature: the square is empty and attacked by an enemy piece of that
// type (sliders stop at the first piece, pins ignored, pawns by their
// captures). Index = (king * 8 + offset) * 6 + type: 96 features, held as a
// bit mask per perspective.
use crate::attacks::king_attacks;
use crate::position::*;

pub const FEATURES: usize = 96;

#[inline(always)]
fn offset(k: usize, t: usize) -> usize {
    let i = ((t / 8) as i32 - (k / 8) as i32 + 1) * 3 + ((t % 8) as i32 - (k % 8) as i32 + 1);
    (i - (i > 4) as i32) as usize
}

/// The active features of both perspectives (bit f = feature f); `fx[p]` is
/// perspective p's frame: its mirror xor, with 56 for black.
#[inline]
pub fn masks(pos: &Position, fx: [usize; 2]) -> [u128; 2] {
    let occ = pos.occ();
    let mut m = [0u128; 2];
    for kc in 0..2 {
        let ks = pos.king_sq(kc);
        let enemy = pos.colors[kc ^ 1];
        let mut zone = king_attacks(ks) & !occ;
        while zone != 0 {
            let t = zone.trailing_zeros() as usize;
            zone &= zone - 1;
            let att = pos.attackers_to(t, occ) & enemy;
            if att == 0 {
                continue;
            }
            let mut types = 0u128;
            for pt in 0..6 {
                types |= ((att & pos.pieces[pt] != 0) as u128) << pt;
            }
            for p in 0..2 {
                m[p] |= types << (((kc ^ p) * 8 + offset(ks ^ fx[p], t ^ fx[p])) * 6);
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frames as the accumulators use them: black's flips ranks, and each
    /// side mirrors files when its king is on files e-h.
    fn frames(pos: &Position) -> [usize; 2] {
        let x = |c: usize| if pos.king_sq(c) % 8 > 3 { 7 } else { 0 };
        [x(WHITE), x(BLACK) ^ 56]
    }

    fn bits(m: u128) -> Vec<usize> {
        (0..FEATURES).filter(|&f| m >> f & 1 != 0).collect()
    }

    /// Prints fixture lines for FENS (cargo test print_fixture -- --ignored --nocapture).
    #[test]
    #[ignore]
    fn print_fixture() {
        crate::attacks::init();
        for fen in FENS {
            let pos = Position::from_fen(fen).unwrap();
            let m = masks(&pos, frames(&pos));
            let s = |v: Vec<usize>| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
            println!("{} | {} | {}", fen, s(bits(m[WHITE])), s(bits(m[BLACK])));
        }
    }

    const FENS: [&str; 10] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "6k1/8/8/8/8/8/r7/6K1 w - - 0 1",
        "6k1/R7/8/8/8/8/8/6K1 b - - 0 1",
        "r3k2r/ppq2ppp/2n5/3Np3/2B5/8/PPP2PPP/R3K2R w KQkq - 0 1",
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2QKB1R b KQ - 0 8",
        "8/1k6/1P5P/5B2/K1p5/p6r/8/3R4 w - - 3 62",
        "2kr3r/ppp2ppp/2n5/8/1b1q4/2N2N2/PPP2PPP/R2QK2R w KQ - 0 12",
        "8/8/4k3/8/2K5/8/3Q4/8 b - - 0 1",
        "4r1k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
        "r5k1/1b3p1p/pp4pB/2p1q3/4P3/2PB4/P4PPP/R4RK1 b - - 1 20",
    ];

    #[test]
    fn fixture_matches() {
        crate::attacks::init();
        let text = include_str!("../tests/kingzone_fixture.txt");
        let mut n = 0;
        for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let cols: Vec<&str> = line.split('|').map(str::trim).collect();
            let pos = Position::from_fen(cols[0]).unwrap();
            let m = masks(&pos, frames(&pos));
            let parse = |s: &str| -> Vec<usize> { s.split_whitespace().map(|x| x.parse().unwrap()).collect() };
            assert_eq!(bits(m[WHITE]), parse(cols[1]), "white, {}", cols[0]);
            assert_eq!(bits(m[BLACK]), parse(cols[2]), "black, {}", cols[0]);
            n += 1;
        }
        assert!(n >= 8);
    }

    #[test]
    fn rook_on_the_back_rank() {
        crate::attacks::init();
        let pos = Position::from_fen("6k1/8/8/8/8/8/r7/6K1 w - - 0 1").unwrap();
        let m = masks(&pos, frames(&pos));
        assert_eq!(bits(m[WHITE]), vec![5 * 6 + 3, 6 * 6 + 3, 7 * 6 + 3]);
        assert!(bits(m[BLACK]).iter().all(|f| f / 48 == 1));
    }
}
