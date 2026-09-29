// Board representation, make move, move generation, SEE.
use crate::attacks::*;

pub type Move = u16;
pub const NO_MOVE: Move = 0;

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;
pub const PAWN: usize = 0;
pub const KNIGHT: usize = 1;
pub const BISHOP: usize = 2;
pub const ROOK: usize = 3;
pub const QUEEN: usize = 4;
pub const KING: usize = 5;
pub const NONE_PC: u8 = 12;
pub const NO_SQ: u8 = 64;

// move flags
pub const F_QUIET: u16 = 0;
pub const F_DOUBLE: u16 = 1;
pub const F_KCASTLE: u16 = 2;
pub const F_QCASTLE: u16 = 3;
pub const F_CAPTURE: u16 = 4;
pub const F_EP: u16 = 5;
pub const F_PROMO: u16 = 8; // + (pt-1)
pub const F_PROMO_CAP: u16 = 12;

#[inline(always)]
pub fn mk(from: usize, to: usize, flag: u16) -> Move {
    (from as u16) | ((to as u16) << 6) | (flag << 12)
}
#[inline(always)]
pub fn mfrom(m: Move) -> usize {
    (m & 63) as usize
}
#[inline(always)]
pub fn mto(m: Move) -> usize {
    ((m >> 6) & 63) as usize
}
#[inline(always)]
pub fn mflag(m: Move) -> u16 {
    m >> 12
}
#[inline(always)]
pub fn is_capture(m: Move) -> bool {
    mflag(m) & 4 != 0
}
#[inline(always)]
pub fn is_promo(m: Move) -> bool {
    mflag(m) & 8 != 0
}
#[inline(always)]
pub fn promo_pt(m: Move) -> usize {
    ((mflag(m) & 3) + 1) as usize
}
#[inline(always)]
pub fn is_noisy(m: Move) -> bool {
    mflag(m) & 12 != 0
}

// piece code = pt*2 + color
#[inline(always)]
pub fn pc_type(pc: u8) -> usize {
    (pc >> 1) as usize
}
#[inline(always)]
pub fn pc_color(pc: u8) -> usize {
    (pc & 1) as usize
}
#[inline(always)]
pub fn make_pc(pt: usize, c: usize) -> u8 {
    (pt * 2 + c) as u8
}

pub const SEE_VAL: [i32; 7] = [100, 320, 330, 500, 950, 20000, 0];

static CASTLE_MASK: [u8; 64] = {
    let mut m = [15u8; 64];
    m[0] = 15 & !2;
    m[7] = 15 & !1;
    m[4] = 15 & !3;
    m[56] = 15 & !8;
    m[63] = 15 & !4;
    m[60] = 15 & !12;
    m
};

pub fn move_str(m: Move) -> String {
    if m == NO_MOVE {
        return "0000".to_string();
    }
    let sq = |s: usize| format!("{}{}", (b'a' + (s % 8) as u8) as char, (b'1' + (s / 8) as u8) as char);
    let mut s = format!("{}{}", sq(mfrom(m)), sq(mto(m)));
    if is_promo(m) {
        s.push(['n', 'b', 'r', 'q'][promo_pt(m) - 1]);
    }
    s
}

pub struct MoveList {
    pub moves: [Move; 256],
    pub len: usize,
}
impl MoveList {
    #[inline(always)]
    pub fn new() -> Self {
        #[allow(invalid_value)]
        unsafe { MoveList { moves: std::mem::MaybeUninit::uninit().assume_init(), len: 0 } }
    }
    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        unsafe {
            *self.moves.get_unchecked_mut(self.len) = m;
        }
        self.len += 1;
    }
}

#[derive(Clone, Copy)]
pub struct Position {
    pub pieces: [u64; 6],
    pub colors: [u64; 2],
    pub board: [u8; 64],
    pub stm: usize,
    pub ep: u8,
    pub castling: u8,
    pub halfmove: u16,
    pub fullmove: u16,
    pub hash: u64,
    pub pawn_key: u64,
    pub np_key: [u64; 2],
    pub checkers: u64,
}

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

impl Position {
    pub fn empty() -> Position {
        Position {
            pieces: [0; 6],
            colors: [0; 2],
            board: [NONE_PC; 64],
            stm: WHITE,
            ep: NO_SQ,
            castling: 0,
            halfmove: 0,
            fullmove: 1,
            hash: 0,
            pawn_key: 0,
            np_key: [0; 2],
            checkers: 0,
        }
    }

    pub fn from_fen(fen: &str) -> Option<Position> {
        let mut p = Position::empty();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return None;
        }
        let mut rank = 7i32;
        let mut file = 0i32;
        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => file += ch as i32 - '0' as i32,
                _ => {
                    let c = if ch.is_ascii_uppercase() { WHITE } else { BLACK };
                    let pt = match ch.to_ascii_lowercase() {
                        'p' => PAWN,
                        'n' => KNIGHT,
                        'b' => BISHOP,
                        'r' => ROOK,
                        'q' => QUEEN,
                        'k' => KING,
                        _ => return None,
                    };
                    if !(0..8).contains(&rank) || !(0..8).contains(&file) {
                        return None;
                    }
                    p.put(make_pc(pt, c), (rank * 8 + file) as usize);
                    file += 1;
                }
            }
        }
        p.stm = if parts[1] == "b" { BLACK } else { WHITE };
        for ch in parts[2].chars() {
            match ch {
                'K' => p.castling |= 1,
                'Q' => p.castling |= 2,
                'k' => p.castling |= 4,
                'q' => p.castling |= 8,
                _ => {}
            }
        }
        // sanity: rights require king and rook on home squares
        let ok = |p: &Position, sq: usize, pc: u8| p.board[sq] == pc;
        if !(ok(&p, 4, make_pc(KING, WHITE)) && ok(&p, 7, make_pc(ROOK, WHITE))) {
            p.castling &= !1;
        }
        if !(ok(&p, 4, make_pc(KING, WHITE)) && ok(&p, 0, make_pc(ROOK, WHITE))) {
            p.castling &= !2;
        }
        if !(ok(&p, 60, make_pc(KING, BLACK)) && ok(&p, 63, make_pc(ROOK, BLACK))) {
            p.castling &= !4;
        }
        if !(ok(&p, 60, make_pc(KING, BLACK)) && ok(&p, 56, make_pc(ROOK, BLACK))) {
            p.castling &= !8;
        }
        if parts[3] != "-" {
            let b = parts[3].as_bytes();
            if b.len() >= 2 {
                let sq = (b[1] - b'1') as usize * 8 + (b[0] - b'a') as usize;
                if sq < 64 && pawn_attacks(p.stm ^ 1, sq) & p.pieces[PAWN] & p.colors[p.stm] != 0 {
                    p.ep = sq as u8;
                }
            }
        }
        p.halfmove = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        p.fullmove = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
        if p.pieces[KING].count_ones() != 2 || (p.pieces[KING] & p.colors[WHITE]).count_ones() != 1 {
            return None;
        }
        p.hash ^= zob_castle(p.castling);
        if p.ep != NO_SQ {
            p.hash ^= zob_ep(p.ep as usize % 8);
        }
        if p.stm == BLACK {
            p.hash ^= zob_stm();
        }
        p.checkers = p.attackers_to(p.king_sq(p.stm), p.occ()) & p.colors[p.stm ^ 1];
        Some(p)
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::new();
        for r in (0..8).rev() {
            let mut empty = 0;
            for f in 0..8 {
                let pc = self.board[r * 8 + f];
                if pc == NONE_PC {
                    empty += 1;
                } else {
                    if empty > 0 {
                        s.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    let ch = ['p', 'n', 'b', 'r', 'q', 'k'][pc_type(pc)];
                    s.push(if pc_color(pc) == WHITE { ch.to_ascii_uppercase() } else { ch });
                }
            }
            if empty > 0 {
                s.push((b'0' + empty) as char);
            }
            if r > 0 {
                s.push('/');
            }
        }
        s.push_str(if self.stm == WHITE { " w " } else { " b " });
        if self.castling == 0 {
            s.push('-');
        } else {
            for (i, ch) in ['K', 'Q', 'k', 'q'].iter().enumerate() {
                if self.castling & (1 << i) != 0 {
                    s.push(*ch);
                }
            }
        }
        if self.ep == NO_SQ {
            s.push_str(" -");
        } else {
            let e = self.ep as usize;
            s.push_str(&format!(" {}{}", (b'a' + (e % 8) as u8) as char, (b'1' + (e / 8) as u8) as char));
        }
        s.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        s
    }

    #[inline(always)]
    pub fn occ(&self) -> u64 {
        self.colors[0] | self.colors[1]
    }
    #[inline(always)]
    pub fn pcs(&self, c: usize, pt: usize) -> u64 {
        self.pieces[pt] & self.colors[c]
    }
    #[inline(always)]
    pub fn king_sq(&self, c: usize) -> usize {
        lsb(self.pcs(c, KING))
    }

    #[inline(always)]
    fn put(&mut self, pc: u8, sq: usize) {
        let b = 1u64 << sq;
        self.pieces[pc_type(pc)] |= b;
        self.colors[pc_color(pc)] |= b;
        self.board[sq] = pc;
        self.hash ^= zob_piece(pc as usize, sq);
        if pc_type(pc) == PAWN {
            self.pawn_key ^= zob_piece(pc as usize, sq);
        } else {
            self.np_key[pc_color(pc)] ^= zob_piece(pc as usize, sq);
        }
    }
    #[inline(always)]
    fn remove(&mut self, sq: usize) {
        let pc = self.board[sq];
        let b = 1u64 << sq;
        self.pieces[pc_type(pc)] ^= b;
        self.colors[pc_color(pc)] ^= b;
        self.board[sq] = NONE_PC;
        self.hash ^= zob_piece(pc as usize, sq);
        if pc_type(pc) == PAWN {
            self.pawn_key ^= zob_piece(pc as usize, sq);
        } else {
            self.np_key[pc_color(pc)] ^= zob_piece(pc as usize, sq);
        }
    }

    #[inline(always)]
    pub fn attackers_to(&self, sq: usize, occ: u64) -> u64 {
        (pawn_attacks(WHITE, sq) & self.pcs(BLACK, PAWN))
            | (pawn_attacks(BLACK, sq) & self.pcs(WHITE, PAWN))
            | (knight_attacks(sq) & self.pieces[KNIGHT])
            | (king_attacks(sq) & self.pieces[KING])
            | (bishop_attacks(sq, occ) & (self.pieces[BISHOP] | self.pieces[QUEEN]))
            | (rook_attacks(sq, occ) & (self.pieces[ROOK] | self.pieces[QUEEN]))
    }

    #[inline(always)]
    pub fn attacked(&self, sq: usize, by: usize) -> bool {
        let occ = self.occ();
        (pawn_attacks(by ^ 1, sq) & self.pcs(by, PAWN)) != 0
            || (knight_attacks(sq) & self.pcs(by, KNIGHT)) != 0
            || (king_attacks(sq) & self.pcs(by, KING)) != 0
            || (bishop_attacks(sq, occ) & (self.pieces[BISHOP] | self.pieces[QUEEN]) & self.colors[by]) != 0
            || (rook_attacks(sq, occ) & (self.pieces[ROOK] | self.pieces[QUEEN]) & self.colors[by]) != 0
    }

    /// Make a pseudo-legal move. Returns false if it leaves own king in check (position is then invalid).
    #[inline]
    pub fn make_move(&mut self, m: Move) -> bool {
        let us = self.stm;
        let them = us ^ 1;
        let from = mfrom(m);
        let to = mto(m);
        let flag = mflag(m);
        let pc = self.board[from];
        self.hash ^= zob_castle(self.castling);
        if self.ep != NO_SQ {
            self.hash ^= zob_ep(self.ep as usize & 7);
            self.ep = NO_SQ;
        }
        self.halfmove += 1;
        if flag == F_EP {
            self.remove(to ^ 8);
        } else if self.board[to] != NONE_PC {
            self.remove(to);
            self.halfmove = 0;
        }
        self.remove(from);
        if flag & 8 != 0 {
            self.put(make_pc(promo_pt(m), us), to);
        } else {
            self.put(pc, to);
        }
        if pc_type(pc) == PAWN {
            self.halfmove = 0;
            if flag == F_DOUBLE {
                let e = (from + to) / 2;
                if pawn_attacks(us, e) & self.pcs(them, PAWN) != 0 {
                    self.ep = e as u8;
                    self.hash ^= zob_ep(e & 7);
                }
            }
        } else if flag == F_KCASTLE {
            self.remove(to + 1);
            self.put(make_pc(ROOK, us), to - 1);
        } else if flag == F_QCASTLE {
            self.remove(to - 2);
            self.put(make_pc(ROOK, us), to + 1);
        }
        self.castling &= CASTLE_MASK[from] & CASTLE_MASK[to];
        self.hash ^= zob_castle(self.castling);
        let occ = self.occ();
        if self.attackers_to(self.king_sq(us), occ) & self.colors[them] != 0 {
            return false;
        }
        self.stm = them;
        self.hash ^= zob_stm();
        if us == BLACK {
            self.fullmove += 1;
        }
        self.checkers = self.attackers_to(self.king_sq(them), occ) & self.colors[us];
        true
    }

    pub fn make_null(&mut self) {
        if self.ep != NO_SQ {
            self.hash ^= zob_ep(self.ep as usize & 7);
            self.ep = NO_SQ;
        }
        self.stm ^= 1;
        self.hash ^= zob_stm();
        self.halfmove += 1;
        self.checkers = 0;
    }

    pub fn gen_moves(&self, list: &mut MoveList, noisy_only: bool) {
        let us = self.stm;
        let them = us ^ 1;
        let occ = self.occ();
        let enemy = self.colors[them];
        let empty = !occ;
        let pawns = self.pcs(us, PAWN);
        let (up, rank3, rank8): (i32, u64, u64) = if us == WHITE { (8, RANK_3, RANK_8) } else { (-8, RANK_6, RANK_1) };
        #[inline(always)]
        fn sh(b: u64, d: i32) -> u64 {
            if d > 0 {
                b << d
            } else {
                b >> (-d)
            }
        }
        let push1 = sh(pawns, up) & empty;
        for to in Bits(push1 & rank8) {
            let from = (to as i32 - up) as usize;
            list.push(mk(from, to, F_PROMO + 3));
            if !noisy_only {
                list.push(mk(from, to, F_PROMO));
                list.push(mk(from, to, F_PROMO + 1));
                list.push(mk(from, to, F_PROMO + 2));
            }
        }
        if !noisy_only {
            let push2 = sh(push1 & rank3, up) & empty;
            for to in Bits(push1 & !rank8) {
                list.push(mk((to as i32 - up) as usize, to, F_QUIET));
            }
            for to in Bits(push2) {
                list.push(mk((to as i32 - 2 * up) as usize, to, F_DOUBLE));
            }
        }
        for (mask, d) in [(!FILE_A, up - 1), (!FILE_H, up + 1)] {
            let caps = sh(pawns & mask, d) & enemy;
            for to in Bits(caps & rank8) {
                let from = (to as i32 - d) as usize;
                list.push(mk(from, to, F_PROMO_CAP + 3));
                list.push(mk(from, to, F_PROMO_CAP));
                list.push(mk(from, to, F_PROMO_CAP + 1));
                list.push(mk(from, to, F_PROMO_CAP + 2));
            }
            for to in Bits(caps & !rank8) {
                list.push(mk((to as i32 - d) as usize, to, F_CAPTURE));
            }
        }
        if self.ep != NO_SQ {
            let e = self.ep as usize;
            for from in Bits(pawn_attacks(them, e) & pawns) {
                list.push(mk(from, e, F_EP));
            }
        }
        let targets = if noisy_only { enemy } else { enemy | empty };
        let mut add = |from: usize, att: u64| {
            for to in Bits(att & targets) {
                list.push(mk(from, to, if enemy & (1u64 << to) != 0 { F_CAPTURE } else { F_QUIET }));
            }
        };
        for from in Bits(self.pcs(us, KNIGHT)) {
            add(from, knight_attacks(from));
        }
        for from in Bits((self.pieces[BISHOP] | self.pieces[QUEEN]) & self.colors[us]) {
            add(from, bishop_attacks(from, occ));
        }
        for from in Bits((self.pieces[ROOK] | self.pieces[QUEEN]) & self.colors[us]) {
            add(from, rook_attacks(from, occ));
        }
        let ksq = self.king_sq(us);
        add(ksq, king_attacks(ksq));
        if !noisy_only && self.checkers == 0 {
            let (kr, qr, base) = if us == WHITE { (1u8, 2u8, 0usize) } else { (4u8, 8u8, 56usize) };
            if self.castling & kr != 0
                && occ & (3u64 << (base + 5)) == 0
                && !self.attacked(base + 5, them)
                && !self.attacked(base + 6, them)
            {
                list.push(mk(base + 4, base + 6, F_KCASTLE));
            }
            if self.castling & qr != 0
                && occ & (7u64 << (base + 1)) == 0
                && !self.attacked(base + 3, them)
                && !self.attacked(base + 2, them)
            {
                list.push(mk(base + 4, base + 2, F_QCASTLE));
            }
        }
    }

    pub fn has_non_pawns(&self, c: usize) -> bool {
        (self.colors[c] & !(self.pieces[PAWN] | self.pieces[KING])) != 0
    }

    #[inline]
    pub fn captured_type(&self, m: Move) -> usize {
        if mflag(m) == F_EP {
            PAWN
        } else {
            let pc = self.board[mto(m)];
            if pc == NONE_PC {
                6
            } else {
                pc_type(pc)
            }
        }
    }

    /// Cheapest piece of `side` among `candidates`: returns its square bit and piece type.
    /// Among several pieces of the same type the one on the lowest square is chosen.
    #[inline(always)]
    fn least_valuable_attacker(&self, candidates: u64, side: usize) -> Option<(u64, usize)> {
        let own = candidates & self.colors[side];
        if own == 0 {
            return None;
        }
        for pt in PAWN..=KING {
            let group = own & self.pieces[pt];
            if group != 0 {
                return Some((group & group.wrapping_neg(), pt));
            }
        }
        None
    }

    /// Static exchange evaluation as a yes/no question: does playing `m` and then
    /// letting both sides trade on the destination square (always with their cheapest
    /// piece) leave the mover with at least `threshold` centipawns of material gain?
    ///
    /// The exchange is played out one capture at a time while a running `balance`
    /// (material won so far, seen from the side that played `m`) is updated. After every
    /// capture the other side may simply stop trading. It will stop exactly when stopping
    /// already puts the balance on its side of the threshold, so each capture either
    /// settles the answer immediately or forces the opponent to recapture. When the side
    /// that must recapture has nothing left to capture with, the side that captured last
    /// has won the argument. Removing a capturer from the occupancy can uncover a slider
    /// standing behind it; those x-ray attackers are added to the pool as they appear.
    /// A king may only join in when the other side has no attackers left on the square.
    /// Castling, en passant and promotions are not examined and count as a zero gain.
    pub fn see_ge(&self, m: Move, threshold: i32) -> bool {
        let flag = mflag(m);
        if is_promo(m) || flag == F_EP || flag == F_KCASTLE || flag == F_QCASTLE {
            return threshold <= 0;
        }
        let us = self.stm;
        let origin = mfrom(m);
        let target = mto(m);

        // Best case: nobody recaptures. If even that falls short, the answer is no.
        let mut balance = SEE_VAL[self.captured_type(m)];
        if balance < threshold {
            return false;
        }
        // Worst case: the moved piece is lost for nothing more. If that still clears
        // the bar, the answer is yes.
        let mut victim = pc_type(self.board[origin]);
        if balance - SEE_VAL[victim] >= threshold {
            return true;
        }

        let diagonal = self.pieces[BISHOP] | self.pieces[QUEEN];
        let straight = self.pieces[ROOK] | self.pieces[QUEEN];
        let mut occupied = self.occ() & !(1u64 << origin) & !(1u64 << target);
        let mut pool = self.attackers_to(target, occupied);
        let mut last_capturer = us;

        loop {
            let recapturer = last_capturer ^ 1;
            pool &= occupied;
            let Some((bit, pt)) = self.least_valuable_attacker(pool, recapturer) else {
                return last_capturer == us;
            };
            if pt == KING && pool & self.colors[last_capturer] != 0 {
                // Taking with the king would walk into a recapture: not allowed.
                return last_capturer == us;
            }

            // The recapture happens; the opponent of `recapturer` now decides whether
            // to go on, and stops at once if the balance already suits it.
            if recapturer == us {
                balance += SEE_VAL[victim];
                if balance < threshold {
                    return false;
                }
            } else {
                balance -= SEE_VAL[victim];
                if balance >= threshold {
                    return true;
                }
            }
            victim = pt;
            last_capturer = recapturer;
            occupied ^= bit;
            match pt {
                PAWN | BISHOP => pool |= bishop_attacks(target, occupied) & diagonal,
                ROOK => pool |= rook_attacks(target, occupied) & straight,
                QUEEN => {
                    pool |= (bishop_attacks(target, occupied) & diagonal)
                        | (rook_attacks(target, occupied) & straight)
                }
                _ => {}
            }
        }
    }

    pub fn is_pseudo_legal(&self, m: Move) -> bool {
        if m == NO_MOVE {
            return false;
        }
        let from = mfrom(m);
        let to = mto(m);
        let flag = mflag(m);
        let pc = self.board[from];
        let us = self.stm;
        if pc == NONE_PC || pc_color(pc) != us || from == to {
            return false;
        }
        if flag == F_KCASTLE || flag == F_QCASTLE || flag == F_EP || flag == 6 || flag == 7 {
            let mut list = MoveList::new();
            self.gen_moves(&mut list, false);
            return list.moves[..list.len].contains(&m);
        }
        let target = self.board[to];
        if target != NONE_PC && (pc_color(target) == us || pc_type(target) == KING) {
            return false;
        }
        let cap = flag & 4 != 0;
        if cap != (target != NONE_PC) {
            return false;
        }
        let pt = pc_type(pc);
        if pt == PAWN {
            let last = (to >> 3) == if us == WHITE { 7 } else { 0 };
            if last != (flag & 8 != 0) {
                return false;
            }
            if cap {
                return pawn_attacks(us, from) & (1u64 << to) != 0;
            }
            let fwd = if us == WHITE { from + 8 } else { from.wrapping_sub(8) };
            if flag == F_DOUBLE {
                let start = (from >> 3) == if us == WHITE { 1 } else { 6 };
                let to2 = if us == WHITE { from + 16 } else { from.wrapping_sub(16) };
                return start && to == to2 && self.board[fwd] == NONE_PC;
            }
            return to == fwd;
        }
        if flag != F_QUIET && flag != F_CAPTURE {
            return false;
        }
        let occ = self.occ();
        let att = match pt {
            KNIGHT => knight_attacks(from),
            BISHOP => bishop_attacks(from, occ),
            ROOK => rook_attacks(from, occ),
            QUEEN => bishop_attacks(from, occ) | rook_attacks(from, occ),
            _ => king_attacks(from),
        };
        att & (1u64 << to) != 0
    }

    pub fn parse_move(&self, s: &str) -> Option<Move> {
        let mut list = MoveList::new();
        self.gen_moves(&mut list, false);
        for i in 0..list.len {
            let m = list.moves[i];
            if move_str(m) == s {
                let mut c = *self;
                if c.make_move(m) {
                    return Some(m);
                }
            }
        }
        None
    }
}

pub fn perft(pos: &Position, depth: u32) -> u64 {
    let mut list = MoveList::new();
    pos.gen_moves(&mut list, false);
    let mut n = 0;
    for i in 0..list.len {
        let mut c = *pos;
        if c.make_move(list.moves[i]) {
            n += if depth <= 1 { 1 } else { perft(&c, depth - 1) };
        }
    }
    n
}
