// Pawn-pair input features (design: incipit-research docs/pawn-features.md).
//
// One binary input per unordered pair of pawn squares on the same or adjacent
// files, for each combination of the two pawns' colours, seen from a
// perspective: squares in that perspective's frame (ranks flipped for black,
// files mirrored when the net mirrors and that side's king is on files e-h),
// colours as ours (0) / theirs (1). Pawns stand on ranks 2-7, so there are
// 8 x C(6,2) same-file and 7 x 6 x 6 adjacent-file pairs: 372 pairs, 4 colour
// combinations each, 1488 features. The feature transformer is linear, so
// these give it the conjunction "pawn here and pawn there" directly.
//
// The trainer computes the same features; tests/pawnpairs_fixture.txt pins the
// exact indices for both sides (see `fixture_line`).
use crate::position::*;
use std::sync::OnceLock;

pub const PAIRS: usize = 372;
pub const FEATURES: usize = PAIRS * 4;
const NONE: u16 = u16::MAX;

/// Pair index of squares (a, b), a < b, or NONE. Pairs are numbered in order
/// of (a, b) over pawn squares (ranks 2-7) on the same or adjacent files.
fn pair_table() -> &'static [[u16; 64]; 64] {
    static T: OnceLock<Box<[[u16; 64]; 64]>> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Box::new([[NONE; 64]; 64]);
        let mut id = 0u16;
        for a in 8..56usize {
            for b in a + 1..56 {
                if (a % 8).abs_diff(b % 8) <= 1 {
                    t[a][b] = id;
                    id += 1;
                }
            }
        }
        assert_eq!(id as usize, PAIRS);
        t
    })
}

/// Feature index for pawns at frame squares `a` and `b` (any order) with
/// relative colours `ca` and `cb` (0 = ours), or None if they don't pair.
#[inline]
pub fn feature(a: usize, ca: usize, b: usize, cb: usize) -> Option<usize> {
    let ((lo, clo), (hi, chi)) = if a < b { ((a, ca), (b, cb)) } else { ((b, cb), (a, ca)) };
    let id = pair_table()[lo][hi];
    (id != NONE).then(|| id as usize * 4 + clo * 2 + chi)
}

/// The pawns of `pos` in perspective `persp`'s frame: (square, colour with
/// 0 = ours). `xor` is the file mirror (7 or 0) that perspective's king sets.
pub fn frame_pawns(pos: &Position, persp: usize, xor: usize) -> Vec<(usize, usize)> {
    let flip = if persp == WHITE { 0 } else { 56 };
    let mut v = Vec::with_capacity(16);
    for c in [WHITE, BLACK] {
        let mut bb = pos.pcs(c, PAWN);
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            v.push((sq ^ flip ^ xor, c ^ persp));
        }
    }
    v
}

/// The active pair features of a position for perspective `persp`, sorted.
pub fn active(pos: &Position, persp: usize, xor: usize) -> Vec<usize> {
    let p = frame_pawns(pos, persp, xor);
    let mut out = Vec::new();
    for i in 0..p.len() {
        for j in i + 1..p.len() {
            if let Some(f) = feature(p[i].0, p[i].1, p[j].0, p[j].1) {
                out.push(f);
            }
        }
    }
    out.sort_unstable();
    out
}

/// Most pairs a position can have: every pair of 16 pawns.
pub const MAX_ACTIVE: usize = 120;

/// Active pair features for perspective `persp` (file mirror `xor`) from the
/// white and black pawn bitboards, into `out` (unsorted); returns the count.
/// No allocation: used by the accumulator updates.
#[inline]
pub fn active_bb(wp: u64, bp: u64, persp: usize, xor: usize, out: &mut [u16; MAX_ACTIVE]) -> usize {
    let flip = if persp == WHITE { 0 } else { 56 };
    let mut sq = [0usize; 16];
    let mut col = [0usize; 16];
    let mut k = 0;
    for (c, mut bb) in [(WHITE, wp), (BLACK, bp)] {
        while bb != 0 && k < 16 {
            sq[k] = bb.trailing_zeros() as usize ^ flip ^ xor;
            col[k] = c ^ persp;
            bb &= bb - 1;
            k += 1;
        }
    }
    let t = pair_table();
    let mut n = 0;
    for i in 0..k {
        for j in i + 1..k {
            let (a, b) = (sq[i], sq[j]);
            let (lo, clo, hi, chi) = if a < b { (a, col[i], b, col[j]) } else { (b, col[j], a, col[i]) };
            let id = t[lo][hi];
            if id != NONE {
                out[n] = id * 4 + (clo * 2 + chi) as u16;
                n += 1;
            }
        }
    }
    n
}

/// The pair features to add (in `new`, not `old`) and subtract (in `old`, not
/// `new`), each passed to `f(feature, add)`. Sorts both lists in place. The
/// reference `delta` is tested against.
#[cfg(test)]
pub fn diff(old: &mut [u16], new: &mut [u16], mut f: impl FnMut(usize, bool)) {
    old.sort_unstable();
    new.sort_unstable();
    let (mut i, mut j) = (0, 0);
    while i < old.len() || j < new.len() {
        if j == new.len() || (i < old.len() && old[i] < new[j]) {
            f(old[i] as usize, false);
            i += 1;
        } else if i == old.len() || new[j] < old[i] {
            f(new[j] as usize, true);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
}

/// For each square, the squares it pairs with: pawn squares (ranks 2-7) on
/// the same or an adjacent file, itself excluded. Flips and mirrors keep
/// files' adjacency, so this holds in every perspective's frame.
fn partner_masks() -> &'static [u64; 64] {
    static M: OnceLock<[u64; 64]> = OnceLock::new();
    M.get_or_init(|| {
        let mut m = [0u64; 64];
        for (a, mask) in m.iter_mut().enumerate() {
            for b in 8..56usize {
                if b != a && (a % 8).abs_diff(b % 8) <= 1 {
                    *mask |= 1 << b;
                }
            }
        }
        m
    })
}

/// The pair features that change between pawn structures `old` and `new`
/// (white, black bitboards) for perspective `persp` with file mirror `xor`,
/// each passed to `f(feature, add)`. Only pairs with a pawn that left or
/// arrived change, so this touches a few pairs instead of rebuilding both
/// lists: a removed pawn's pairs in `old` are subtracted (a pair of two
/// removed pawns once), an added pawn's pairs in `new` are added.
#[inline]
pub fn delta(old: (u64, u64), new: (u64, u64), persp: usize, xor: usize, mut f: impl FnMut(usize, bool)) {
    let flip = if persp == WHITE { 0 } else { 56 };
    let masks = partner_masks();
    let t = pair_table();
    let col = |sq: usize, w: u64| if w >> sq & 1 != 0 { WHITE ^ persp } else { BLACK ^ persp };
    let mut feat = |a: usize, ca: usize, b: usize, cb: usize, add: bool| {
        let (fa, fb) = (a ^ flip ^ xor, b ^ flip ^ xor);
        let (lo, clo, hi, chi) = if fa < fb { (fa, ca, fb, cb) } else { (fb, cb, fa, ca) };
        let id = t[lo][hi];
        debug_assert!(id != NONE);
        f(id as usize * 4 + clo * 2 + chi, add);
    };
    for (pawns, gone_w, gone_b, add) in
        [(old, old.0 & !new.0, old.1 & !new.1, false), (new, new.0 & !old.0, new.1 & !old.1, true)]
    {
        let all = pawns.0 | pawns.1;
        let mut done = 0u64;
        let mut changed = gone_w | gone_b;
        while changed != 0 {
            let a = changed.trailing_zeros() as usize;
            changed &= changed - 1;
            let ca = col(a, pawns.0);
            let mut partners = all & masks[a] & !done;
            while partners != 0 {
                let b = partners.trailing_zeros() as usize;
                partners &= partners - 1;
                feat(a, ca, b, col(b, pawns.0), add);
            }
            done |= 1 << a;
        }
    }
}

/// The file mirror for perspective `persp` with a mirrored net: 7 when that
/// side's king (in its own frame) is on files e-h.
pub fn mirror_xor(pos: &Position, persp: usize) -> usize {
    let k = pos.king_sq(persp) ^ if persp == WHITE { 0 } else { 56 };
    if k & 7 >= 4 {
        7
    } else {
        0
    }
}

/// One fixture line: `fen | white-perspective features | black-perspective
/// features`, for a mirrored net. The trainer's test reads the same lines.
pub fn fixture_line(pos: &Position) -> String {
    let side = |p| active(pos, p, mirror_xor(pos, p)).iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" ");
    format!("{} | {} | {}", pos.to_fen(), side(WHITE), side(BLACK))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(fen: &str) -> Position {
        crate::attacks::init();
        Position::from_fen(fen).unwrap()
    }

    #[test]
    fn counts_and_index_range() {
        let t = pair_table();
        let n = (0..64).flat_map(|a| (0..64).map(move |b| (a, b))).filter(|&(a, b)| t[a][b] != NONE).count();
        assert_eq!(n, PAIRS);
        assert_eq!(feature(8, 0, 9, 1), Some(1)); // a2-b2, the first pair: ours-theirs
        assert_eq!(feature(9, 1, 8, 0), Some(1)); // order of arguments doesn't matter
        assert_eq!(feature(8, 0, 10, 0), None); // a2-c2: two files apart
        assert_eq!(feature(0, 0, 8, 0), None); // rank 1 is not a pawn square
        let last = feature(54, 1, 55, 1).unwrap(); // g7-h7 theirs-theirs
        assert_eq!(last, FEATURES - 1);
    }

    /// The start position, from either side: 7 ours-ours pairs on rank 2, 7
    /// theirs-theirs on rank 7, and ours-theirs across: 8 on the same file
    /// (a2-a7 ...) and 2 x 7 on adjacent files.
    #[test]
    fn start_position_symmetry() {
        let p = pos(START_FEN);
        let w = active(&p, WHITE, mirror_xor(&p, WHITE));
        let b = active(&p, BLACK, mirror_xor(&p, BLACK));
        assert_eq!(w.len(), 7 + 7 + 8 + 14);
        assert_eq!(w, b, "the start position looks the same from both sides");
    }

    /// Colour-swapped, rank-flipped positions look the same from the other side.
    #[test]
    fn colour_flip_swaps_perspectives() {
        let a = pos("4k3/1pp5/3p4/8/2P5/1P2P3/8/4K3 w - - 0 1");
        let b = pos("4k3/8/1p2p3/2p5/8/3P4/1PP5/4K3 b - - 0 1");
        assert_eq!(active(&a, WHITE, 0), active(&b, BLACK, 0));
        assert_eq!(active(&a, BLACK, 0), active(&b, WHITE, 0));
    }

    /// Mirroring the board left-right with mirror xor 7 gives the same features.
    #[test]
    fn mirror_invariance() {
        let a = pos("4k3/1pp5/3p4/8/2P5/1P2P3/8/3K4 w - - 0 1");
        let m = pos("3k4/5pp1/4p3/8/5P2/3P2P1/8/4K3 w - - 0 1");
        assert_eq!(active(&a, WHITE, 0), active(&m, WHITE, 7));
    }

    /// The allocation-free version agrees with `active`, and `diff` gives
    /// exactly the symmetric difference.
    #[test]
    fn active_bb_and_diff() {
        crate::attacks::init();
        for line in include_str!("../tests/pawnpairs_fixture.txt").lines().filter(|l| !l.starts_with('#')) {
            let p = Position::from_fen(line.split('|').next().unwrap().trim()).unwrap();
            for persp in [WHITE, BLACK] {
                for xor in [0, 7] {
                    let mut out = [0u16; MAX_ACTIVE];
                    let n = active_bb(p.pcs(WHITE, PAWN), p.pcs(BLACK, PAWN), persp, xor, &mut out);
                    let mut v: Vec<usize> = out[..n].iter().map(|&x| x as usize).collect();
                    v.sort_unstable();
                    assert_eq!(v, active(&p, persp, xor));
                }
            }
        }
        let (mut a, mut b) = ([5u16, 1, 9, 3], [3u16, 7, 9, 2]);
        let (mut adds, mut subs) = (vec![], vec![]);
        diff(&mut a, &mut b, |f, add| if add { adds.push(f) } else { subs.push(f) });
        assert_eq!((adds, subs), (vec![2, 7], vec![1, 5]));
    }

    /// `delta` gives exactly the symmetric difference of the full lists, for
    /// every legal move from the fixture positions (pawn moves, captures, en
    /// passant, promotions, and moves that touch no pawn).
    #[test]
    fn delta_matches_full_diff() {
        crate::attacks::init();
        let mut n = 0;
        for line in include_str!("../tests/pawnpairs_fixture.txt").lines().filter(|l| !l.starts_with('#')) {
            let p = Position::from_fen(line.split('|').next().unwrap().trim()).unwrap();
            let mut list = MoveList::new();
            p.gen_moves(&mut list, false);
            for i in 0..list.len() {
                let mut c = p;
                if !c.make_move(list[i]) {
                    continue;
                }
                let (old, new) = ((p.pcs(WHITE, PAWN), p.pcs(BLACK, PAWN)), (c.pcs(WHITE, PAWN), c.pcs(BLACK, PAWN)));
                for persp in [WHITE, BLACK] {
                    for xor in [0, 7] {
                        let (mut a, mut b) = ([0u16; MAX_ACTIVE], [0u16; MAX_ACTIVE]);
                        let ka = active_bb(old.0, old.1, persp, xor, &mut a);
                        let kb = active_bb(new.0, new.1, persp, xor, &mut b);
                        let (mut want, mut got) = (Vec::new(), Vec::new());
                        diff(&mut a[..ka], &mut b[..kb], |f, add| want.push((f, add)));
                        delta(old, new, persp, xor, |f, add| got.push((f, add)));
                        want.sort_unstable();
                        got.sort_unstable();
                        assert_eq!(got, want, "{} {}", p.to_fen(), p.move_uci(list[i]));
                        n += 1;
                    }
                }
            }
        }
        assert!(n > 5000, "{n}");
    }

    /// The checked-in fixture matches this implementation (the trainer's test
    /// checks the same file against its own implementation).
    #[test]
    fn fixture_matches() {
        crate::attacks::init();
        let text = include_str!("../tests/pawnpairs_fixture.txt");
        let mut n = 0;
        for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let fen = line.split('|').next().unwrap().trim();
            let p = Position::from_fen(fen).unwrap();
            assert_eq!(fixture_line(&p), line, "fixture mismatch");
            n += 1;
        }
        assert!(n >= 50, "fixture too small: {n}");
    }
}
