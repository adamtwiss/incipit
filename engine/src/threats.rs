// Threat input features (design: incipit-research docs/threat-features.md).
//
// One binary input per (attacker, attacker square, target square, victim):
// the attacker's attack set in the current position (sliders stop at the
// first piece, pins ignored, pawns by their diagonal captures only) contains
// the target square, which holds the victim. The victim may be either
// colour (attack or defence) but never a king. Seen from a perspective as for
// the piece inputs: squares in that perspective's frame (ranks flipped for
// black, files mirrored when the net mirrors and that side's king is on
// files e-h), colours as ours (0) / theirs (1).
//
// Left out, so they get no index:
// - pawns on ranks 1 and 8, as attacker or victim;
// - a king attacking an enemy piece that would attack the king back from
//   where it stands (the position is illegal or in check, and training
//   filters checks);
// - the lower-value half of a mutual pair: when the victim also attacks the
//   attacker's square (rook and rook on a line, queen and bishop on a
//   diagonal, opposing pawns, ...), the two features always fire together
//   (a slider's path is the same both ways), so one feature carries both.
//   The attacker of higher value keeps it; of equal value, the attacker on
//   the lower frame square. Values: pawn 1, knight 3, bishop 3, rook 5,
//   queen 9;
// - by variant, one-way attacks of a lower-value piece on a higher-value
//   enemy piece:
//     0 (B): none, 56,890 features;
//     1 (D): knight attacking queen, 56,218 features;
//     2 (E): every lower-value attacker on a higher-value enemy, 54,090.
//
// Index = offset[attacker][square][victim] + the number of valid target
// squares below the target, where attacker = colour * 6 + type (12) and
// victim = colour * 5 + type (10, no kings). The trainer computes the same
// features; tests/threats_fixture.txt pins the exact indices for both sides.
use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks};
use crate::position::*;
use std::sync::OnceLock;

/// Number of variants.
pub const VARIANTS: usize = 3;
/// Features in each variant.
pub const VARIANT_FEATURES: [usize; VARIANTS] = [56_890, 56_218, 54_090];

const VALUE: [u32; 6] = [1, 3, 3, 5, 9, 100];
const PAWN_SQUARES: u64 = 0x00ff_ffff_ffff_ff00;

/// Index tables for one variant: per (attacker, square, victim) the first
/// index and the bitboard of valid target squares.
struct Table {
    offset: Box<[[[u32; 10]; 64]; 12]>,
    valid: Box<[[[u64; 10]; 64]; 12]>,
    // The same, flat: [(attacker * 64 + square) * 10 + victim].
    offset_flat: Vec<u32>,
    valid_flat: Vec<u64>,
    slots: Vec<Slot>,
    count: usize,
}

/// One slot of the flat index table: the valid target squares of an
/// (attacker, square, victim) in the frame, and the index of the first.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct Slot {
    pub valid: u64,
    pub offset: u32,
}

/// The flat index table of variant `var`, by `attacker_slot + victim_slot`.
pub fn slots(var: usize) -> &'static [Slot] {
    &table(var).slots
}

/// The first slot of attacker `pc` on board square `s` for perspective
/// `persp`, with `fx` = its board flip ^ mirror xor.
#[inline(always)]
pub fn attacker_slot(pc: u8, s: usize, persp: usize, fx: usize) -> usize {
    (((pc_color(pc) ^ persp) * 6 + pc_type(pc)) * 64 + (s ^ fx)) * 10
}

/// The victim's offset within an attacker's slots.
#[inline(always)]
pub fn victim_slot(vc: u8, persp: usize) -> usize {
    (pc_color(vc) ^ persp) * 5 + pc_type(vc)
}

/// Empty-board attacks of a piece of relative colour `c` (0 = ours, moving
/// up the frame) and type `pt` from frame square `s`.
fn empty_attacks(c: usize, pt: usize, s: usize) -> u64 {
    match pt {
        PAWN => pawn_attacks(c, s),
        KNIGHT => knight_attacks(s),
        BISHOP => bishop_attacks(s, 0),
        ROOK => rook_attacks(s, 0),
        QUEEN => bishop_attacks(s, 0) | rook_attacks(s, 0),
        _ => king_attacks(s),
    }
}

/// Whether (attacker colour ca, type a, square s, target t, victim colour cv,
/// type v) is a feature before deduplication, in variant `var`.
fn allowed(var: usize, ca: usize, a: usize, s: usize, t: usize, cv: usize, v: usize) -> bool {
    if v == KING || (a == PAWN && PAWN_SQUARES >> s & 1 == 0) || (v == PAWN && PAWN_SQUARES >> t & 1 == 0) {
        return false;
    }
    if empty_attacks(ca, a, s) >> t & 1 == 0 {
        return false;
    }
    let enemy = ca != cv;
    if a == KING && enemy && empty_attacks(cv, v, t) >> s & 1 != 0 {
        return false;
    }
    if enemy && a != KING && VALUE[a] < VALUE[v] {
        let one_way = empty_attacks(cv, v, t) >> s & 1 == 0;
        match var {
            1 if one_way && a == KNIGHT && v == QUEEN => return false,
            2 if one_way => return false,
            _ => {}
        }
    }
    true
}

/// Whether an allowed feature is the kept half of its pair (or has none).
fn kept(var: usize, ca: usize, a: usize, s: usize, t: usize, cv: usize, v: usize) -> bool {
    if !allowed(var, cv, v, t, s, ca, a) {
        return true; // no reverse feature: nothing to merge with
    }
    debug_assert!(empty_attacks(cv, v, t) >> s & 1 != 0);
    VALUE[a] > VALUE[v] || (VALUE[a] == VALUE[v] && s < t)
}

fn table(var: usize) -> &'static Table {
    static T: [OnceLock<Table>; VARIANTS] = [const { OnceLock::new() }; VARIANTS];
    T[var].get_or_init(|| {
        let mut offset = Box::new([[[0u32; 10]; 64]; 12]);
        let mut valid = Box::new([[[0u64; 10]; 64]; 12]);
        let mut n = 0u32;
        for ap in 0..12 {
            let (ca, a) = (ap / 6, ap % 6);
            for s in 0..64 {
                for vp in 0..10 {
                    let (cv, v) = (vp / 5, vp % 5);
                    let mut m = 0u64;
                    for t in 0..64 {
                        if allowed(var, ca, a, s, t, cv, v) && kept(var, ca, a, s, t, cv, v) {
                            m |= 1 << t;
                        }
                    }
                    offset[ap][s][vp] = n;
                    valid[ap][s][vp] = m;
                    n += m.count_ones();
                }
            }
        }
        let offset_flat: Vec<u32> = offset.iter().flatten().flatten().copied().collect();
        let valid_flat: Vec<u64> = valid.iter().flatten().flatten().copied().collect();
        let slots = valid_flat.iter().zip(&offset_flat).map(|(&valid, &offset)| Slot { valid, offset }).collect();
        Table { offset, valid, offset_flat, valid_flat, slots, count: n as usize }
    })
}

/// Number of features in variant `var` (from the table; a test checks it
/// against VARIANT_FEATURES).
pub fn features(var: usize) -> usize {
    table(var).count
}

/// Feature index in variant `var` for frame squares and relative colours, or
/// None if that combination has no feature.
#[inline]
pub fn feature(var: usize, ca: usize, a: usize, s: usize, t: usize, cv: usize, v: usize) -> Option<usize> {
    if v == KING {
        return None;
    }
    let tb = table(var);
    let (ap, vp) = (ca * 6 + a, cv * 5 + v);
    let m = tb.valid[ap][s][vp];
    (m >> t & 1 != 0).then(|| tb.offset[ap][s][vp] as usize + (m & ((1u64 << t) - 1)).count_ones() as usize)
}

/// The squares piece `pc` on `sq` attacks with occupancy `occ`.
#[inline]
fn attacks(pc: u8, sq: usize, occ: u64) -> u64 {
    match pc_type(pc) {
        PAWN => pawn_attacks(pc_color(pc), sq),
        KNIGHT => knight_attacks(sq),
        BISHOP => bishop_attacks(sq, occ),
        ROOK => rook_attacks(sq, occ),
        QUEEN => bishop_attacks(sq, occ) | rook_attacks(sq, occ),
        _ => king_attacks(sq),
    }
}

/// The active threat features of a position for perspective `persp` (file
/// mirror `xor`, as for pawn pairs), sorted.
pub fn active(var: usize, pos: &Position, persp: usize, xor: usize) -> Vec<usize> {
    let flip = if persp == WHITE { 0 } else { 56 };
    let occ = pos.occ();
    let mut out = Vec::new();
    let mut bb = occ;
    while bb != 0 {
        let s = bb.trailing_zeros() as usize;
        bb &= bb - 1;
        let pc = pos.board[s];
        let mut targets = attacks(pc, s, occ) & occ;
        while targets != 0 {
            let t = targets.trailing_zeros() as usize;
            targets &= targets - 1;
            let vc = pos.board[t];
            let f = feature(
                var,
                pc_color(pc) ^ persp,
                pc_type(pc),
                s ^ flip ^ xor,
                t ^ flip ^ xor,
                pc_color(vc) ^ persp,
                pc_type(vc),
            );
            out.extend(f);
        }
    }
    out.sort_unstable();
    out
}

/// Most active threats a position can have: each piece attacks at most 8
/// occupied squares (games reach ~65).
pub const MAX_ACTIVE: usize = 256;

/// The active threat features for perspective `persp` (file mirror `xor`),
/// into `out` (unsorted); returns the count. No allocation: used by the
/// accumulator updates.
#[inline]
pub fn active_into(var: usize, pos: &Position, persp: usize, xor: usize, out: &mut [u32; MAX_ACTIVE]) -> usize {
    let flip = if persp == WHITE { 0 } else { 56 };
    let tb = table(var);
    let occ = pos.occ();
    let mut n = 0;
    let mut bb = occ;
    while bb != 0 {
        let s = bb.trailing_zeros() as usize;
        bb &= bb - 1;
        let pc = pos.board[s];
        let ap = (pc_color(pc) ^ persp) * 6 + pc_type(pc);
        let fs = s ^ flip ^ xor;
        let mut targets = attacks(pc, s, occ) & occ;
        while targets != 0 {
            let t = targets.trailing_zeros() as usize;
            targets &= targets - 1;
            let vc = pos.board[t];
            if pc_type(vc) == KING {
                continue;
            }
            let i = (ap * 64 + fs) * 10 + (pc_color(vc) ^ persp) * 5 + pc_type(vc);
            let ft = t ^ flip ^ xor;
            let m = tb.valid_flat[i];
            if m >> ft & 1 != 0 {
                out[n] = tb.offset_flat[i] + (m & ((1u64 << ft) - 1)).count_ones();
                n += 1;
            }
        }
    }
    n
}

/// A threat in board terms (attacker piece, its square, target square,
/// victim piece), packed into one integer so lists sort and compare cheaply.
#[inline(always)]
fn pack(pc: u8, s: usize, t: usize, vc: u8) -> u32 {
    (pc as u32) << 20 | (s as u32) << 14 | (t as u32) << 8 | vc as u32
}

/// Feature index of a packed threat for perspective `persp` (mirror `xor`),
/// or None when that threat has no feature in this frame (excluded, or the
/// merged half of a mutual pair).
#[inline(always)]
pub fn index_of(var: usize, k: u32, persp: usize, xor: usize) -> Option<u32> {
    let flip = if persp == WHITE { 0 } else { 56 };
    let (pc, s, t, vc) = ((k >> 20) as u8, (k >> 14 & 63) as usize, (k >> 8 & 63) as usize, (k & 255) as u8);
    let tb = table(var);
    let i = (((pc_color(pc) ^ persp) * 6 + pc_type(pc)) * 64 + (s ^ flip ^ xor)) * 10
        + (pc_color(vc) ^ persp) * 5
        + pc_type(vc);
    let ft = t ^ flip ^ xor;
    let m = tb.valid_flat[i];
    (m >> ft & 1 != 0).then(|| tb.offset_flat[i] + (m & ((1u64 << ft) - 1)).count_ones())
}

/// The packed threats of the pieces on `squares` in `pos`, into `out`,
/// sorted; returns the count. King victims are skipped (never features).
#[inline]
fn threats_of(pos: &Position, squares: u64, out: &mut [u32; MAX_ACTIVE]) -> usize {
    let occ = pos.occ();
    let mut n = 0;
    let mut bb = squares & occ;
    while bb != 0 {
        let s = bb.trailing_zeros() as usize;
        bb &= bb - 1;
        let pc = pos.board[s];
        let mut targets = attacks(pc, s, occ) & occ & !pos.pieces[KING];
        while targets != 0 {
            let t = targets.trailing_zeros() as usize;
            targets &= targets - 1;
            out[n] = pack(pc, s, t, pos.board[t]);
            n += 1;
        }
    }
    out[..n].sort_unstable();
    n
}

/// The threats that differ between `pos` and `child` (any two positions),
/// by attacker square: `f(square, removed targets, added targets)`; the
/// attacker and victims of a removed threat are `pos`'s, of an added one
/// `child`'s.
///
/// A threat changes only if its attacker's piece or attack set changes, or
/// its victim changes. With C the squares whose contents changed, that
/// means the attacker stands on C (before or after) or attacks a square of
/// C (before or after: sliders whose rays pass through C, and any piece
/// whose victim is on C). For a square of C, the attacker changed: it loses
/// all of its old targets To and gains all of the new Tn. Elsewhere the
/// piece is the same in both positions: it loses To \ Tn, gains Tn \ To,
/// and both for targets on C (the victim changed). A pawn, knight or king
/// off C keeps its attack set, so only its targets on C change.
///
/// The candidate attackers are found once for both positions: pieces off C
/// are the same in both, so the union of the two piece sets serves, and
/// sliders are looked up through the common occupancy (it differs from
/// either position only on C, and a slider seen through a square of C
/// attacks that square in one of the positions, so it is a candidate
/// anyway). Extra candidates are harmless: their removed and added sets
/// come out empty.
#[inline(always)]
pub fn delta_masks(pos: &Position, child: &Position, mut f: impl FnMut(usize, u64, u64)) {
    let mut c = (pos.colors[0] ^ child.colors[0]) | (pos.colors[1] ^ child.colors[1]);
    for pt in 0..6 {
        c |= pos.pieces[pt] ^ child.pieces[pt];
    }
    let (occ_a, occ_b) = (pos.occ(), child.occ());
    // Targets: occupied squares other than kings (never victims).
    let (vic_a, vic_b) = (occ_a & !pos.pieces[KING], occ_b & !child.pieces[KING]);
    let occ_i = occ_a & occ_b;
    let wp = pos.pcs(WHITE, PAWN) | child.pcs(WHITE, PAWN);
    let bp = pos.pcs(BLACK, PAWN) | child.pcs(BLACK, PAWN);
    let kn = pos.pieces[KNIGHT] | child.pieces[KNIGHT];
    let kg = pos.pieces[KING] | child.pieces[KING];
    let diag = pos.pieces[BISHOP] | pos.pieces[QUEEN] | child.pieces[BISHOP] | child.pieces[QUEEN];
    let orth = pos.pieces[ROOK] | pos.pieces[QUEEN] | child.pieces[ROOK] | child.pieces[QUEEN];
    let mut s = c;
    let mut sq = c;
    while sq != 0 {
        let x = sq.trailing_zeros() as usize;
        sq &= sq - 1;
        s |= (pawn_attacks(WHITE, x) & bp)
            | (pawn_attacks(BLACK, x) & wp)
            | (knight_attacks(x) & kn)
            | (king_attacks(x) & kg)
            | (bishop_attacks(x, occ_i) & diag)
            | (rook_attacks(x, occ_i) & orth);
    }
    // Gathered first and handed to `f` from one place, so `f` stays inline.
    // At most one entry per occupied square (64), each written before it is read.
    let mut buf = [std::mem::MaybeUninit::<(u8, u64, u64)>::uninit(); 64];
    let mut n = 0;
    // Changed squares: everything the old piece attacked goes, everything
    // the new one attacks comes.
    let mut sc = c & (occ_a | occ_b);
    while sc != 0 {
        let x = sc.trailing_zeros() as usize;
        sc &= sc - 1;
        let bit = 1u64 << x;
        let to = if occ_a & bit != 0 { attacks(pos.board[x], x, occ_a) & vic_a } else { 0 };
        let tn = if occ_b & bit != 0 { attacks(child.board[x], x, occ_b) & vic_b } else { 0 };
        buf[n] = std::mem::MaybeUninit::new((x as u8, to, tn));
        n += 1;
    }
    // Unchanged pieces. Sliders: their attack sets may differ.
    let sliders = (diag | orth) & occ_i & !c;
    let mut ss = s & sliders;
    while ss != 0 {
        let x = ss.trailing_zeros() as usize;
        ss &= ss - 1;
        let pc = pos.board[x];
        let to = attacks(pc, x, occ_a) & vic_a;
        let tn = attacks(pc, x, occ_b) & vic_b;
        buf[n] = std::mem::MaybeUninit::new((x as u8, to & (!tn | c), tn & (!to | c)));
        n += 1;
    }
    // Pawns, knights and kings keep their attack sets: only victims on C.
    let mut sn = s & occ_i & !c & !sliders;
    while sn != 0 {
        let x = sn.trailing_zeros() as usize;
        sn &= sn - 1;
        let a = attacks(pos.board[x], x, 0) & c;
        buf[n] = std::mem::MaybeUninit::new((x as u8, a & vic_a, a & vic_b));
        n += 1;
    }
    for e in &buf[..n] {
        // Safe: the first n entries were written above.
        let (x, rem, add) = unsafe { e.assume_init() };
        f(x as usize, rem, add);
    }
}

/// The original form of `delta_masks`, kept as the reference for tests.
#[cfg(test)]
pub fn delta_masks_ref(pos: &Position, child: &Position, mut f: impl FnMut(usize, u64, u64)) {
    let mut c = (pos.colors[0] ^ child.colors[0]) | (pos.colors[1] ^ child.colors[1]);
    for pt in 0..6 {
        c |= pos.pieces[pt] ^ child.pieces[pt];
    }
    let (occ_a, occ_b) = (pos.occ(), child.occ());
    let (vic_a, vic_b) = (occ_a & !pos.pieces[KING], occ_b & !child.pieces[KING]);
    let mut s = c;
    let mut sq = c;
    while sq != 0 {
        let x = sq.trailing_zeros() as usize;
        sq &= sq - 1;
        s |= pos.attackers_to(x, occ_a) | child.attackers_to(x, occ_b);
    }
    s &= occ_a | occ_b;
    while s != 0 {
        let x = s.trailing_zeros() as usize;
        s &= s - 1;
        let bit = 1u64 << x;
        let to = if occ_a & bit != 0 { attacks(pos.board[x], x, occ_a) & vic_a } else { 0 };
        let tn = if occ_b & bit != 0 { attacks(child.board[x], x, occ_b) & vic_b } else { 0 };
        let moved = 0u64.wrapping_sub((c >> x) & 1);
        f(x, to & (!tn | c | moved), tn & (!to | c | moved));
    }
}

/// `delta_masks` as packed threats: `f(packed threat, add)`.
#[inline]
pub fn delta(pos: &Position, child: &Position, mut f: impl FnMut(u32, bool)) {
    delta_masks(pos, child, |x, mut rem, mut add| {
        while rem != 0 {
            let t = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            f(pack(pos.board[x], x, t, pos.board[t]), false);
        }
        while add != 0 {
            let t = add.trailing_zeros() as usize;
            add &= add - 1;
            f(pack(child.board[x], x, t, child.board[t]), true);
        }
    });
}

/// One fixture line: `fen | white-perspective features | black-perspective
/// features`, for a mirrored net. The trainer's test reads the same lines.
pub fn fixture_line(var: usize, pos: &Position) -> String {
    let side = |p| {
        let xor = crate::pawnpairs::mirror_xor(pos, p);
        active(var, pos, p, xor).iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" ")
    };
    format!("{} | {} | {}", pos.to_fen(), side(WHITE), side(BLACK))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(fen: &str) -> Position {
        crate::attacks::init();
        Position::from_fen(fen).unwrap()
    }

    /// Counts match the design note (scripts/threat_count_dedupe.py), and
    /// indices are a bijection onto 0..count.
    #[test]
    fn counts_and_bijection() {
        crate::attacks::init();
        for var in 0..VARIANTS {
            assert_eq!(features(var), VARIANT_FEATURES[var], "variant {var}");
            let mut seen = vec![false; features(var)];
            for ap in 0..12 {
                for s in 0..64 {
                    for t in 0..64 {
                        for vp in 0..10 {
                            if let Some(f) = feature(var, ap / 6, ap % 6, s, t, vp / 5, vp % 5) {
                                assert!(!seen[f], "index {f} used twice");
                                seen[f] = true;
                            }
                        }
                    }
                }
            }
            assert!(seen.iter().all(|&x| x));
        }
    }

    /// Mutual pairs are merged and the exclusions hold, in frame squares
    /// (a1 = 0, ours moving up).
    #[test]
    fn exclusions_and_merges() {
        crate::attacks::init();
        let (a1, a2, b3, c3, d4, e4, d5, e5, f6) = (0, 8, 17, 18, 27, 28, 35, 36, 45);
        // Rook a1 and enemy rook a2: one feature, the lower square's.
        assert!(feature(0, 0, ROOK, a1, a2, 1, ROOK).is_some());
        assert!(feature(0, 1, ROOK, a2, a1, 0, ROOK).is_none());
        // Our pawn d4 attacks their bishop e5: merged into the bishop's attack.
        assert!(feature(0, 0, PAWN, d4, e5, 1, BISHOP).is_none());
        assert!(feature(0, 1, BISHOP, e5, d4, 0, PAWN).is_some());
        // Our pawn d4 attacks their rook e5: one-way, kept in B and D, gone in E.
        assert!(feature(0, 0, PAWN, d4, e5, 1, ROOK).is_some());
        assert!(feature(1, 0, PAWN, d4, e5, 1, ROOK).is_some());
        assert!(feature(2, 0, PAWN, d4, e5, 1, ROOK).is_none());
        // Our knight d4 attacks their queen e6 (44): one-way, gone in D.
        assert!(feature(0, 0, KNIGHT, d4, 44, 1, QUEEN).is_some());
        assert!(feature(1, 0, KNIGHT, d4, 44, 1, QUEEN).is_none());
        // Pawns are never on rank 1, kings never victims.
        assert!(feature(0, 1, KNIGHT, b3, a1, 0, PAWN).is_none());
        assert!(feature(0, 0, QUEEN, d4, d5, 1, KING).is_none());
        // Our king d4 attacks their knight e5: fine; their queen e5: in check.
        assert!(feature(0, 0, KING, d4, e5, 1, KNIGHT).is_some());
        assert!(feature(0, 0, KING, d4, e5, 1, QUEEN).is_none());
        // Their pawn e5 would check our king d4 (it attacks d4): excluded;
        // their pawn d5 in front of the king doesn't: kept.
        assert!(feature(0, 0, KING, d4, e5, 1, PAWN).is_none());
        assert!(feature(0, 0, KING, d4, d5, 1, PAWN).is_some());
        // A king defending its own queen keeps the feature (no king victims, so
        // no reverse feature to merge with).
        assert!(feature(0, 0, KING, d4, e4, 0, QUEEN).is_some());
        // Our bishop c3 attacks their knight f6: one-way, kept.
        assert!(feature(0, 0, BISHOP, c3, f6, 1, KNIGHT).is_some());
    }

    /// The start position: knights defend d2/e2 pawns etc. Both sides see the
    /// same features.
    #[test]
    fn start_position_symmetry() {
        let p = pos(START_FEN);
        let a = active(0, &p, WHITE, crate::pawnpairs::mirror_xor(&p, WHITE));
        let b = active(0, &p, BLACK, crate::pawnpairs::mirror_xor(&p, BLACK));
        assert!(!a.is_empty());
        assert_eq!(a, b);
    }

    /// Colour-swapped, rank-flipped positions look the same from the other side.
    #[test]
    fn colour_flip_swaps_perspectives() {
        let a = pos("r3k2r/1bq1nppp/p3p3/1p1nP3/3P4/P1N2N2/1P2BPPP/R2Q1RK1 w kq - 0 1");
        let b = pos("r2q1rk1/1p2bppp/p1n2n2/3p4/1P1Np3/P3P3/1BQ1NPPP/R3K2R b KQ - 0 1");
        for var in 0..VARIANTS {
            assert_eq!(active(var, &a, WHITE, 0), active(var, &b, BLACK, 0));
            assert_eq!(active(var, &a, BLACK, 0), active(var, &b, WHITE, 0));
        }
    }

    /// Mirroring the board left-right with mirror xor 7 gives the same features.
    #[test]
    fn mirror_invariance() {
        let a = pos("r3k2r/1bq1nppp/p3p3/1p1nP3/3P4/P1N2N2/1P2BPPP/R2Q1RK1 w kq - 0 1");
        let m = pos("r2k3r/pppn1qb1/3p3p/3Pn1p1/4P3/2N2N1P/PPPB2P1/1KR1Q2R w - - 0 1");
        for var in 0..VARIANTS {
            assert_eq!(active(var, &a, WHITE, 0), active(var, &m, WHITE, 7));
            assert_eq!(active(var, &a, BLACK, 0), active(var, &m, BLACK, 7));
        }
    }

    /// The allocation-free version gives the same features as `active`.
    #[test]
    fn active_into_matches_active() {
        crate::attacks::init();
        let text = include_str!("../tests/threats_fixture.txt");
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let p = Position::from_fen(line.split('|').next().unwrap().trim()).unwrap();
            for var in 0..VARIANTS {
                for persp in [WHITE, BLACK] {
                    for xor in [0, 7] {
                        let mut out = [0u32; MAX_ACTIVE];
                        let n = active_into(var, &p, persp, xor, &mut out);
                        let mut got: Vec<usize> = out[..n].iter().map(|&f| f as usize).collect();
                        got.sort_unstable();
                        assert_eq!(got, active(var, &p, persp, xor));
                    }
                }
            }
        }
    }

    /// `delta` gives exactly the difference of the full feature lists, for
    /// both perspectives and mirror states, over random games (captures,
    /// en passant, promotions, castling incl. Chess960).
    #[test]
    fn delta_matches_full_lists() {
        use crate::position::MoveList;
        crate::attacks::init();
        let starts = [
            START_FEN,
            "r3k2r/1pp2pp1/p2p3p/4p3/4P3/P2P3P/1PP2PP1/R3K2R w KQkq - 0 1",
            "8/2P1P3/8/8/8/8/3p1p2/k6K w - - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
            "r1b1k1nr/1pq2ppp/p1nbp3/8/3NP3/2N1B3/PPP1BPPP/R2QK2R w KQkq - 4 9",
        ];
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut checked = 0;
        for g in 0..80 {
            let mut pos = Position::from_fen(starts[g % starts.len()]).unwrap();
            for _ in 0..120 {
                let mut list = MoveList::new();
                pos.gen_moves(&mut list, false);
                let legal: Vec<_> = (0..list.len()).map(|i| list[i]).filter(|&m| pos.clone().make_move(m)).collect();
                if legal.is_empty() {
                    break;
                }
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let m = legal[(seed % legal.len() as u64) as usize];
                let mut child = pos;
                child.make_move(m);
                for var in 0..VARIANTS {
                    for persp in [WHITE, BLACK] {
                        for xor in [0, 7] {
                            let mut got = active(var, &pos, persp, xor);
                            delta(&pos, &child, |k, add| {
                                if let Some(f) = index_of(var, k, persp, xor) {
                                    let f = f as usize;
                                    if add {
                                        got.push(f);
                                    } else {
                                        let i = got.iter().position(|&x| x == f).expect("subtracted feature is active");
                                        got.swap_remove(i);
                                    }
                                }
                            });
                            got.sort_unstable();
                            assert_eq!(
                                got,
                                active(var, &child, persp, xor),
                                "after {} in {}",
                                pos.move_uci(m),
                                pos.to_fen()
                            );
                        }
                    }
                }
                checked += 1;
                pos = child;
            }
        }
        assert!(checked > 5000, "only {checked}");
    }

    /// `delta_masks` gives the same per-square removed and added targets as
    /// the reference form, for positions one move apart and several moves
    /// apart (captures, en passant, promotions, castling incl. Chess960).
    #[test]
    fn delta_masks_matches_reference() {
        use crate::position::MoveList;
        crate::attacks::init();
        let starts = [
            START_FEN,
            "r3k2r/1pp2pp1/p2p3p/4p3/4P3/P2P3P/1PP2PP1/R3K2R w KQkq - 0 1",
            "8/2P1P3/8/8/8/8/3p1p2/k6K w - - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
            "r1b1k1nr/1pq2ppp/p1nbp3/8/3NP3/2N1B3/PPP1BPPP/R2QK2R w KQkq - 4 9",
        ];
        let collect = |a: &Position, b: &Position, reference: bool| {
            let mut v = Vec::new();
            let mut push = |x: usize, rem: u64, add: u64| {
                if rem | add != 0 {
                    v.push((x, rem, add));
                }
            };
            if reference {
                delta_masks_ref(a, b, &mut push);
            } else {
                delta_masks(a, b, &mut push);
            }
            v.sort_unstable();
            v
        };
        let mut seed = 0x1F12_3BB5_9E37_79B9u64;
        let mut checked = 0;
        for g in 0..120 {
            let mut hist = vec![Position::from_fen(starts[g % starts.len()]).unwrap()];
            for _ in 0..120 {
                let pos = *hist.last().unwrap();
                let mut list = MoveList::new();
                pos.gen_moves(&mut list, false);
                let legal: Vec<_> = (0..list.len()).map(|i| list[i]).filter(|&m| pos.clone().make_move(m)).collect();
                if legal.is_empty() {
                    break;
                }
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let mut child = pos;
                child.make_move(legal[(seed % legal.len() as u64) as usize]);
                hist.push(child);
                // One move, and from up to four moves back.
                for back in 1..=4.min(hist.len() - 1) {
                    let a = hist[hist.len() - 1 - back];
                    assert_eq!(
                        collect(&a, &child, false),
                        collect(&a, &child, true),
                        "{} -> {}",
                        a.to_fen(),
                        child.to_fen()
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 20000, "only {checked}");
    }

    /// The checked-in fixture matches (the trainer's tests check the same file).
    #[test]
    fn fixture_matches() {
        crate::attacks::init();
        let text = include_str!("../tests/threats_fixture.txt");
        let mut n = 0;
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let p = Position::from_fen(line.split('|').next().unwrap().trim()).unwrap();
            assert_eq!(fixture_line(0, &p), line);
            n += 1;
        }
        assert!(n >= 50);
    }
}
