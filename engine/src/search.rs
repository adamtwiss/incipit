// Search: iterative deepening PVS with the usual pruning/reduction heuristics.
use crate::nnue::{self, Acc};
use crate::params::{tp, P};
use crate::position::*;
use crate::tt::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub const INF: i32 = 32000;
pub const MATE: i32 = 31000;
pub const MATE_BOUND: i32 = MATE - 512;
pub const MAX_PLY: usize = 128;
/// Tablebase wins score TB_WIN - ply: above every eval (evals stay below
/// MATE_BOUND) and below every mate (MATE - ply >= MATE - MAX_PLY), so code
/// that treats |score| >= MATE_BOUND as decisive handles them like mates.
pub const TB_WIN: i32 = MATE - 2 * MAX_PLY as i32;
const CORR_SIZE: usize = 16384;
const USE_SCORE_TM: bool = false;
const CORR_GRAIN: i32 = 256;

#[derive(Clone, Copy, Default)]
struct Frame {
    static_eval: i32,
    cont_idx: usize, // piece*64+to of the move made at this ply
    excluded: Move,
    mv: Move,
}

/// Search statistics, cumulative until reset (bench sums them over its
/// positions). Counting doesn't change the search, so bench node counts are
/// unaffected.
#[derive(Clone, Copy, Default)]
pub struct Stats {
    pub qs_nodes: u64,
    pub tt_probes: u64,
    pub tt_hits: u64,
    pub tt_cutoffs: u64,
    pub asp_fail_low: u64,
    pub asp_fail_high: u64,
    pub rfp: u64,
    pub razor_tries: u64,
    pub razor_cuts: u64,
    pub nmp_tries: u64,
    pub nmp_cuts: u64,
    pub probcut_cuts: u64,
    pub iir: u64,
    pub lmp: u64,
    pub futility: u64,
    pub hist_prunes: u64,
    pub see_quiet: u64,
    pub see_noisy: u64,
    pub lmr_searches: u64,
    pub lmr_researches: u64,
    pub se_tries: u64,
    pub se_single: u64,
    pub se_double: u64,
    pub se_negative: u64,
    pub multicut: u64,
    pub check_ext: u64,
    pub beta_cuts: u64,
    pub cut_pos_sum: u64,
    pub cut_pos_sq_sum: u64,
    pub first_move_cuts: u64,
    /// Sum of ln(nodes(d) / nodes(d-1)) over completed iterations with d >= 5.
    pub ebf_log_sum: f64,
    pub ebf_count: u64,
}

impl Stats {
    /// Multi-line summary; `nodes` is the total node count over the same searches.
    pub fn report(&self, nodes: u64) -> String {
        let pct = |a: u64, b: u64| if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 };
        let kn = |a: u64| if nodes == 0 { 0.0 } else { 1000.0 * a as f64 / nodes as f64 };
        let mut o = String::new();
        let mut line = |label: &str, text: String| o.push_str(&format!("{:<16}{}\n", label, text));
        line("Total nodes:", format!("{}  (QS {:.1}%)", nodes, pct(self.qs_nodes, nodes)));
        let ebf = if self.ebf_count > 0 { (self.ebf_log_sum / self.ebf_count as f64).exp() } else { 0.0 };
        line("EBF (d >= 5):", format!("{:.2}  (geometric mean over {} iterations)", ebf, self.ebf_count));
        line(
            "Move ordering:",
            format!(
                "first-move cut {:.1}%, avg cutoff position {:.2}, avg position\u{b2} {:.1}  ({} beta cutoffs)",
                pct(self.first_move_cuts, self.beta_cuts),
                self.cut_pos_sum as f64 / self.beta_cuts.max(1) as f64,
                self.cut_pos_sq_sum as f64 / self.beta_cuts.max(1) as f64,
                self.beta_cuts
            ),
        );
        line(
            "TT:",
            format!(
                "probes {} ({:.0}/Kn), hits {:.1}%, cutoffs {} ({:.1}/Kn)",
                self.tt_probes,
                kn(self.tt_probes),
                pct(self.tt_hits, self.tt_probes),
                self.tt_cutoffs,
                kn(self.tt_cutoffs)
            ),
        );
        line("Aspiration:", format!("fail-low {}, fail-high {}", self.asp_fail_low, self.asp_fail_high));
        line("RFP:", format!("{} cutoffs ({:.1}/Kn)", self.rfp, kn(self.rfp)));
        line("Razoring:", format!("{} tries, {} cutoffs ({:.0}%)", self.razor_tries, self.razor_cuts, pct(self.razor_cuts, self.razor_tries)));
        line("Null move:", format!("{} tries, {} cutoffs ({:.0}%)", self.nmp_tries, self.nmp_cuts, pct(self.nmp_cuts, self.nmp_tries)));
        line("ProbCut:", format!("{} cutoffs ({:.1}/Kn)", self.probcut_cuts, kn(self.probcut_cuts)));
        line("IIR:", format!("{} reductions", self.iir));
        line("LMP:", format!("{} triggers (remaining quiets skipped)", self.lmp));
        line("Futility:", format!("{} triggers (remaining quiets skipped)", self.futility));
        line("History prune:", format!("{} moves ({:.1}/Kn)", self.hist_prunes, kn(self.hist_prunes)));
        line("SEE prune:", format!("{} quiet, {} noisy ({:.1}/Kn)", self.see_quiet, self.see_noisy, kn(self.see_quiet + self.see_noisy)));
        line(
            "LMR:",
            format!(
                "{} reduced searches ({:.1}/Kn), {} re-searched ({:.1}%)",
                self.lmr_searches,
                kn(self.lmr_searches),
                self.lmr_researches,
                pct(self.lmr_researches, self.lmr_searches)
            ),
        );
        line(
            "Singular:",
            format!(
                "{} tries: +1 {}, +2 {}, -1 {}, multi-cut {}",
                self.se_tries, self.se_single, self.se_double, self.se_negative, self.multicut
            ),
        );
        line("Check ext:", format!("{}", self.check_ext));
        o
    }
}

pub struct Limits {
    pub soft_ms: Option<u64>,
    pub hard_ms: Option<u64>,
    pub depth: i32,
    pub nodes: Option<u64>,
}

pub struct Searcher {
    pub tt: TT,
    pub nodes: u64,
    tb_hits: u64,
    pub stop_flag: Arc<AtomicBool>,
    stopped: bool,
    start: Instant,
    hard_ms: Option<u64>,
    node_limit: Option<u64>,
    hist: Box<[[[i16; 64]; 64]; 2]>,
    cont: Box<[[i16; 768]; 768]>,
    capt: Box<[[[i16; 7]; 64]; 12]>,
    killers: [[Move; 2]; MAX_PLY + 4],
    counter: Box<[[Move; 64]; 12]>,
    stack: [Frame; MAX_PLY + 8],
    pv: Box<[[Move; MAX_PLY + 2]; MAX_PLY + 2]>,
    pv_len: [usize; MAX_PLY + 2],
    pub hash_hist: Vec<u64>,
    seldepth: usize,
    root_depth: i32,
    root_best: Move,
    lmr: Box<[[i32; 64]; 64]>,
    pub silent: bool,
    root_node_counts: Box<[[u64; 64]; 64]>,
    acc: Vec<Acc>,
    // Root of the current search, for printing moves (Chess960 castling needs
    // the castling rooks' squares, which are fixed for the game).
    root_pos: Position,
    refresh_cache: nnue::RefreshCache,
    corr: Box<[[i32; CORR_SIZE]; 2]>,
    corr_np: Box<[[[i32; CORR_SIZE]; 2]; 2]>,
    pub stats: Stats,
}

#[inline(always)]
fn pick(moves: &mut [Move; 256], scores: &mut [i32; 256], i: usize, n: usize) {
    unsafe {
        let mut bi = i;
        let mut bs = *scores.get_unchecked(i);
        for j in i + 1..n {
            let s = *scores.get_unchecked(j);
            if s > bs {
                bs = s;
                bi = j;
            }
        }
        if bi != i {
            let sp = scores.as_mut_ptr();
            let mp = moves.as_mut_ptr();
            std::ptr::swap(sp.add(i), sp.add(bi));
            std::ptr::swap(mp.add(i), mp.add(bi));
        }
    }
}

#[inline(always)]
fn upd(h: &mut i16, bonus: i32) {
    let v = *h as i32;
    *h = (v + bonus - v * bonus.abs() / 16384) as i16;
}

fn score_str(s: i32) -> String {
    if s >= MATE - MAX_PLY as i32 {
        format!("mate {}", (MATE - s + 1) / 2)
    } else if s <= -(MATE - MAX_PLY as i32) {
        format!("mate -{}", (MATE + s) / 2)
    } else if s.abs() >= MATE_BOUND {
        // Tablebase win or loss, shown as a large centipawn score.
        format!("cp {}", s.signum() * (20000 - (TB_WIN - s.abs())))
    } else {
        format!("cp {}", s)
    }
}

impl Searcher {
    pub fn new(hash_mb: usize, stop_flag: Arc<AtomicBool>) -> Searcher {
        let mut lmr = Box::new([[0i32; 64]; 64]);
        for d in 1..64 {
            for m in 1..64 {
                lmr[d][m] = (tp(P::LmrBaseX100) as f64 / 100.0 + (d as f64).ln() * (m as f64).ln() / (tp(P::LmrDivX100) as f64 / 100.0)) as i32;
            }
        }
        Searcher {
            tt: TT::new(hash_mb),
            nodes: 0,
            stop_flag,
            stopped: false,
            start: Instant::now(),
            hard_ms: None,
            node_limit: None,
            hist: Box::new([[[0; 64]; 64]; 2]),
            cont: vec![[0i16; 768]; 768].into_boxed_slice().try_into().unwrap(),
            capt: Box::new([[[0; 7]; 64]; 12]),
            killers: [[0; 2]; MAX_PLY + 4],
            counter: Box::new([[0; 64]; 12]),
            stack: [Frame::default(); MAX_PLY + 8],
            pv: Box::new([[0; MAX_PLY + 2]; MAX_PLY + 2]),
            pv_len: [0; MAX_PLY + 2],
            hash_hist: Vec::with_capacity(1024),
            seldepth: 0,
            tb_hits: 0,
            root_depth: 0,
            root_best: 0,
            lmr,
            silent: false,
            root_node_counts: Box::new([[0; 64]; 64]),
            corr_np: vec![[[0i32; CORR_SIZE]; 2]; 2].into_boxed_slice().try_into().unwrap(),
            corr: vec![[0i32; CORR_SIZE]; 2].into_boxed_slice().try_into().unwrap(),
            stats: Stats::default(),
            acc: vec![Acc::new(); MAX_PLY + 8],
            root_pos: Position::empty(),
            refresh_cache: nnue::RefreshCache::new(),
        }
    }

    pub fn init_lmr(&mut self) {
        for d in 1..64 {
            for m in 1..64 {
                self.lmr[d][m] = (tp(P::LmrBaseX100) as f64 / 100.0 + (d as f64).ln() * (m as f64).ln() / (tp(P::LmrDivX100) as f64 / 100.0)) as i32;
            }
        }
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        *self.hist = [[[0; 64]; 64]; 2];
        for r in self.cont.iter_mut() {
            *r = [0; 768];
        }
        *self.capt = [[[0; 7]; 64]; 12];
        *self.counter = [[0; 64]; 12];
        for c in self.corr.iter_mut() {
            c.iter_mut().for_each(|x| *x = 0);
        }
        for a in self.corr_np.iter_mut() {
            for c in a.iter_mut() {
                c.iter_mut().for_each(|x| *x = 0);
            }
        }
    }

    #[inline(always)]
    fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    #[inline(always)]
    fn check_time(&mut self) {
        if self.root_depth <= 1 {
            return;
        }
        if self.stop_flag.load(Ordering::Relaxed) {
            self.stopped = true;
            return;
        }
        if let Some(h) = self.hard_ms {
            if self.elapsed_ms() >= h {
                self.stopped = true;
            }
        }
        if let Some(n) = self.node_limit {
            if self.nodes >= n {
                self.stopped = true;
            }
        }
    }

    #[inline(always)]
    fn push_acc(&mut self, ply: usize, pos: &Position, child: &Position, m: Move) {
        if cfg!(feature = "hce") {
            return; // bootstrap build: no NNUE, skip accumulator updates
        }
        let (a, b) = self.acc.split_at_mut(ply + 1);
        b[0].update_from(&a[ply], pos, child, m, &mut self.refresh_cache);
    }

    /// The network's eval (side to move), without the fifty-move damping: the
    /// value stored in the TT, which is shared by any halfmove count.
    #[inline(always)]
    fn evaluate(&self, pos: &Position, ply: usize) -> i32 {
        let e = if cfg!(feature = "hce") {
            crate::eval::evaluate(pos)
        } else {
            nnue::evaluate(&self.acc[ply], pos)
        };
        e.clamp(-MATE_BOUND + 1, MATE_BOUND - 1)
    }

    /// Pulls an eval towards a draw as the fifty-move counter rises.
    #[inline(always)]
    fn damp(pos: &Position, v: i32) -> i32 {
        v * (200 - pos.halfmove as i32) / 200
    }

    #[inline(always)]
    fn corrected(&self, pos: &Position, raw: i32) -> i32 {
        let m = CORR_SIZE - 1;
        let c = (2 * self.corr[pos.stm][(pos.pawn_key as usize) & m]
            + self.corr_np[pos.stm][0][(pos.np_key[0] as usize) & m]
            + self.corr_np[pos.stm][1][(pos.np_key[1] as usize) & m])
            / (2 * CORR_GRAIN);
        Self::damp(pos, raw + c).clamp(-MATE_BOUND + 1, MATE_BOUND - 1)
    }

    fn update_corr(&mut self, pos: &Position, depth: i32, diff: i32) {
        let w = (depth + 1).min(16);
        let target = diff.clamp(-400, 400) * CORR_GRAIN;
        let m = CORR_SIZE - 1;
        let f = |e: &mut i32| *e = ((*e * (256 - w) + target * w) / 256).clamp(-CORR_GRAIN * 64, CORR_GRAIN * 64);
        f(&mut self.corr[pos.stm][(pos.pawn_key as usize) & m]);
        f(&mut self.corr_np[pos.stm][0][(pos.np_key[0] as usize) & m]);
        f(&mut self.corr_np[pos.stm][1][(pos.np_key[1] as usize) & m]);
    }

    fn is_repetition(&self, pos: &Position) -> bool {
        let n = self.hash_hist.len();
        let lim = (pos.halfmove as usize).min(n);
        let mut k = 4;
        while k <= lim {
            if self.hash_hist[n - k] == pos.hash {
                return true;
            }
            k += 2;
        }
        false
    }

    /// Returns (best move, score).
    pub fn search(&mut self, root: &Position, lim: &Limits) -> (Move, i32) {
        self.start = Instant::now();
        self.nodes = 0;
        self.tb_hits = 0;
        self.stopped = false;
        self.hard_ms = lim.hard_ms;
        self.node_limit = lim.nodes;
        self.root_best = 0;
        self.seldepth = 0;
        self.acc[0].refresh(root);
        self.root_pos = *root;
        for r in self.root_node_counts.iter_mut() {
            *r = [0; 64];
        }
        // fallback move
        let mut list = MoveList::new();
        root.gen_moves(&mut list, false);
        let mut fallback = 0;
        for i in 0..list.len {
            let mut c = *root;
            if c.make_move(list.moves[i]) {
                fallback = list.moves[i];
                break;
            }
        }
        if fallback == 0 {
            if !self.silent {
                println!("info depth 0 score {} nodes 0 time 0", if root.checkers != 0 { "mate 0" } else { "cp 0" });
            }
            return (0, 0);
        }
        // Root in the tablebases: play the move that keeps the result, with the
        // fifty-move counter taken into account (DTZ).
        if crate::tb::largest() > 0 {
            if let Some((w, m)) = crate::tb::probe_root(root) {
                let s = match w {
                    crate::tb::Wdl::Win => TB_WIN - 1,
                    crate::tb::Wdl::Loss => -TB_WIN + 1,
                    crate::tb::Wdl::Draw => 0,
                };
                if !self.silent {
                    println!("info depth 1 score {} nodes 0 tbhits 1 time 0 pv {}", score_str(s), root.move_uci(m));
                }
                return (m, s);
            }
        }
        let mut best = fallback;
        let mut score = 0;
        let mut prev_best = 0;
        let mut stability = 0;
        // first move of the last PV printed, so bestmove always matches the reported PV
        let mut reported: Move = 0;
        let max_depth = lim.depth.min(MAX_PLY as i32 - 4);
        let mut prev_iter_nodes = 0u64;
        for d in 1..=max_depth {
            let nodes_at_start = self.nodes;
            self.root_depth = d;
            let mut delta = tp(P::AspDelta);
            let (mut a, mut b) = if d >= 4 { (score - delta, score + delta) } else { (-INF, INF) };
            let mut fh_move: Move = 0;
            let mut fh_score = 0;
            let mut s;
            loop {
                self.seldepth = 0;
                s = self.negamax(root, a, b, d, 0, false);
                if self.stopped {
                    break;
                }
                if s <= a {
                    self.stats.asp_fail_low += 1;
                    b = (a + b) / 2;
                    a = (s - delta).max(-INF);
                } else if s >= b {
                    self.stats.asp_fail_high += 1;
                    b = (s + delta).min(INF);
                    if self.root_best != 0 {
                        best = self.root_best;
                        fh_move = best;
                        fh_score = s;
                    }
                } else {
                    break;
                }
                delta += delta / 2;
                if delta > 1000 {
                    a = -INF;
                    b = INF;
                }
            }
            if self.stopped {
                if self.root_best != 0 {
                    best = self.root_best;
                }
                if !self.silent && best != reported {
                    let sc = if best == fh_move {
                        format!("{} lowerbound", score_str(fh_score))
                    } else {
                        score_str(score)
                    };
                    let el = self.elapsed_ms();
                    println!(
                        "info depth {} score {} nodes {} nps {} time {} pv {}",
                        d,
                        sc,
                        self.nodes,
                        self.nodes * 1000 / el.max(1),
                        el,
                        root.move_uci(best)
                    );
                }
                break;
            }
            let iter_nodes = self.nodes - nodes_at_start;
            if d >= 5 && prev_iter_nodes > 0 {
                self.stats.ebf_log_sum += (iter_nodes as f64 / prev_iter_nodes as f64).ln();
                self.stats.ebf_count += 1;
            }
            prev_iter_nodes = iter_nodes;
            let prev_score = if d > 1 { score } else { s };
            score = s;
            best = self.pv[0][0];
            if best == 0 {
                best = fallback;
            }
            if !self.silent {
                self.print_info(d, score);
                reported = if self.pv_len[0] > 0 { self.pv[0][0] } else { best };
            }
            if best == prev_best {
                stability = (stability + 1).min(10);
            } else {
                stability = 0;
            }
            prev_best = best;
            if let Some(soft) = lim.soft_ms {
                let total = self.nodes.max(1) as f64;
                let frac = self.root_node_counts[mfrom(best)][mto(best)] as f64 / total;
                let node_scale = (1.5 - frac) * 1.35;
                let stab_scale = [2.2, 1.6, 1.3, 1.1, 1.0, 0.95, 0.9, 0.85, 0.8, 0.78, 0.75][stability];
                let drop = (prev_score - score).clamp(-50, 150) as f64;
                let score_scale = if d >= 6 { 1.0 + drop / 200.0 } else { 1.0 };
                let score_scale = if USE_SCORE_TM { score_scale } else { 1.0 };
                let target = soft as f64 * node_scale * stab_scale * score_scale;
                if self.elapsed_ms() as f64 >= target {
                    break;
                }
            }
            if let Some(n) = lim.nodes {
                if self.nodes >= n {
                    break;
                }
            }
        }
        (best, score)
    }

    fn print_info(&self, d: i32, score: i32) {
        let el = self.elapsed_ms();
        let nps = self.nodes * 1000 / el.max(1);
        let mut pv = String::new();
        for i in 0..self.pv_len[0] {
            pv.push(' ');
            pv.push_str(&self.root_pos.move_uci(self.pv[0][i]));
        }
        println!(
            "info depth {} seldepth {} score {} nodes {} nps {} hashfull {} tbhits {} time {} pv{}",
            d,
            self.seldepth,
            score_str(score),
            self.nodes,
            nps,
            self.tt.hashfull(),
            self.tb_hits,
            el,
            pv
        );
    }

    fn negamax(&mut self, pos: &Position, mut alpha: i32, mut beta: i32, mut depth: i32, ply: usize, cut_node: bool) -> i32 {
        let pv_node = beta - alpha > 1;
        let root = ply == 0;
        self.pv_len[ply] = 0;
        if depth <= 0 {
            return self.qsearch(pos, alpha, beta, ply);
        }
        self.nodes += 1;
        if self.nodes & 1023 == 0 {
            self.check_time();
        }
        if self.stopped {
            return 0;
        }
        if ply > self.seldepth {
            self.seldepth = ply;
        }
        let in_check = pos.checkers != 0;
        if !root {
            if self.is_repetition(pos) || pos.insufficient_material() {
                return 0;
            }
            // Fifty-move rule, unless the side to move is checkmated.
            if pos.halfmove >= 100 {
                return if in_check && !pos.has_legal_move() { -MATE + ply as i32 } else { 0 };
            }
            if ply >= MAX_PLY - 2 {
                return if in_check { 0 } else { Self::damp(pos, self.evaluate(pos, ply)) };
            }
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
        }
        let excluded = self.stack[ply].excluded;
        let tte = if excluded == 0 {
            self.stats.tt_probes += 1;
            self.tt.probe(pos.hash)
        } else {
            None
        };
        let mut tt_move = 0;
        let mut tt_score = 0;
        let mut tt_depth = -1;
        let mut tt_bound = BOUND_NONE;
        if let Some(e) = tte {
            self.stats.tt_hits += 1;
            tt_move = e.mv;
            tt_score = e.score as i32;
            if tt_score >= MATE_BOUND {
                tt_score -= ply as i32;
            } else if tt_score <= -MATE_BOUND {
                tt_score += ply as i32;
            }
            tt_depth = e.depth as i32;
            tt_bound = e.bound;
            if !pv_node
                && tt_depth >= depth
                && pos.halfmove < 90
                && (tt_bound == BOUND_EXACT
                    || (tt_bound == BOUND_LOWER && tt_score >= beta)
                    || (tt_bound == BOUND_UPPER && tt_score <= alpha))
            {
                if tt_score >= beta && tt_move != 0 && !is_noisy(tt_move) {
                    // reward quiet tt move causing cutoff
                    let bonus = (tp(P::HistMul) * depth - tp(P::HistOff)).min(tp(P::HistMax));
                    let pc = pos.board[mfrom(tt_move)] as usize;
                    upd(&mut self.hist[pos.stm][mfrom(tt_move)][mto(tt_move)], bonus);
                    let _ = pc;
                }
                self.stats.tt_cutoffs += 1;
                return tt_score;
            }
        }

        // Tablebase WDL probe: valid right after a capture or pawn move
        // (halfmove 0) without castling rights.
        if !root
            && excluded == 0
            && pos.halfmove == 0
            && pos.castling == 0
            && pos.occ().count_ones() <= crate::tb::largest()
        {
            if let Some(w) = crate::tb::probe_wdl(pos) {
                self.tb_hits += 1;
                let (s, bound) = match w {
                    crate::tb::Wdl::Win => (TB_WIN - ply as i32, BOUND_LOWER),
                    crate::tb::Wdl::Loss => (-TB_WIN + ply as i32, BOUND_UPPER),
                    crate::tb::Wdl::Draw => (0, BOUND_EXACT),
                };
                if bound == BOUND_EXACT || (bound == BOUND_LOWER && s >= beta) || (bound == BOUND_UPPER && s <= alpha) {
                    let ss = if s >= MATE_BOUND { s + ply as i32 } else if s <= -MATE_BOUND { s - ply as i32 } else { s };
                    let ev = if in_check { -INF } else { self.evaluate(pos, ply) };
                    self.tt.store(pos.hash, 0, ss, ev, (depth + 6).min(MAX_PLY as i32 - 1), bound);
                    return s;
                }
            }
        }

        let static_eval;
        let mut eval;
        let mut raw_eval = -INF;
        if in_check {
            static_eval = -INF;
            eval = -INF;
        } else if excluded != 0 {
            static_eval = self.stack[ply].static_eval;
            eval = static_eval;
        } else {
            raw_eval = if let Some(e) = tte { e.eval as i32 } else { self.evaluate(pos, ply) };
            static_eval = self.corrected(pos, raw_eval);
            eval = static_eval;
            if tte.is_some()
                && tt_score.abs() < MATE_BOUND
                && (tt_bound == BOUND_EXACT
                    || (tt_bound == BOUND_LOWER && tt_score > eval)
                    || (tt_bound == BOUND_UPPER && tt_score < eval))
            {
                eval = tt_score;
            }
        }
        self.stack[ply].static_eval = static_eval;
        let improving = !in_check && ply >= 2 && self.stack[ply - 2].static_eval != -INF && static_eval > self.stack[ply - 2].static_eval;
        self.killers[ply + 1] = [0, 0];

        if !pv_node && !in_check && excluded == 0 {
            // reverse futility pruning
            if depth <= tp(P::RfpDepth) && eval.abs() < MATE_BOUND && eval - tp(P::RfpMargin) * (depth - improving as i32) >= beta {
                self.stats.rfp += 1;
                return (eval + beta) / 2;
            }
            // razoring
            // (on the corrected static eval, not the TT-adjusted one)
            if depth <= 3 && static_eval + tp(P::RazorBase) + tp(P::RazorMul) * depth <= alpha {
                self.stats.razor_tries += 1;
                let v = self.qsearch(pos, alpha, alpha + 1, ply);
                if v <= alpha {
                    self.stats.razor_cuts += 1;
                    return v;
                }
            }
            // null move pruning
            if depth >= 3
                && eval >= beta
                && static_eval >= beta - tp(P::NmpDepthMul) * depth + tp(P::NmpMarginBase)
                && ply >= 1
                && self.stack[ply - 1].mv != 0
                && pos.has_non_pawns(pos.stm)
            {
                self.stats.nmp_tries += 1;
                let r = tp(P::NmpBase) + depth / 3 + ((eval - beta) / tp(P::NmpEvalDiv)).min(3);
                let mut child = *pos;
                child.make_null();
                let (a, b) = self.acc.split_at_mut(ply + 1);
                b[0].copy_from(&a[ply]);
                self.stack[ply].mv = 0;
                self.stack[ply].cont_idx = 0;
                self.hash_hist.push(pos.hash);
                let v = -self.negamax(&child, -beta, -beta + 1, depth - r, ply + 1, !cut_node);
                self.hash_hist.pop();
                if self.stopped {
                    return 0;
                }
                if v >= beta {
                    self.stats.nmp_cuts += 1;
                    return if v >= MATE_BOUND { beta } else { v };
                }
            }
            // probcut
            let pc_beta = beta + tp(P::ProbcutMargin);
            if depth >= 5
                && beta.abs() < MATE_BOUND
                && !(tte.is_some() && tt_depth >= depth - 3 && tt_score < pc_beta)
            {
                let mut list = MoveList::new();
                pos.gen_moves(&mut list, true);
                for i in 0..list.len {
                    let m = list.moves[i];
                    if !pos.see_ge(m, pc_beta - static_eval) {
                        continue;
                    }
                    let mut child = *pos;
                    if !child.make_move(m) {
                        continue;
                    }
                    self.push_acc(ply, pos, &child, m);
                    let pc = pos.board[mfrom(m)] as usize;
                    self.stack[ply].mv = m;
                    self.stack[ply].cont_idx = pc * 64 + mto(m);
                    self.hash_hist.push(pos.hash);
                    let mut v = -self.qsearch(&child, -pc_beta, -pc_beta + 1, ply + 1);
                    if v >= pc_beta {
                        v = -self.negamax(&child, -pc_beta, -pc_beta + 1, depth - 4, ply + 1, !cut_node);
                    }
                    self.hash_hist.pop();
                    if self.stopped {
                        return 0;
                    }
                    if v >= pc_beta {
                        self.tt.store(pos.hash, m, v, raw_eval, depth - 3, BOUND_LOWER);
                        self.stats.probcut_cuts += 1;
                        return v;
                    }
                }
            }
        }
        // internal iterative reduction
        if depth >= 4 && tt_move == 0 && (pv_node || cut_node) {
            self.stats.iir += 1;
            depth -= 1;
        }

        let mut list = MoveList::new();
        #[allow(invalid_value)]
        let mut scores: [i32; 256] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        let us = pos.stm;
        let prev1 = if ply >= 1 { self.stack[ply - 1].cont_idx } else { 0 };
        let prev2 = if ply >= 2 { self.stack[ply - 2].cont_idx } else { 0 };
        let prev4 = if ply >= 4 { self.stack[ply - 4].cont_idx } else { 0 };
        let counter_move = if ply >= 1 && self.stack[ply - 1].mv != 0 {
            let pm = self.stack[ply - 1].mv;
            self.counter[pos.board[mto(pm)] as usize][mto(pm)]
        } else {
            0
        };
        let killers = self.killers[ply];
        let mut generated = false;
        if tt_move != 0 && pos.is_pseudo_legal(tt_move) {
            list.push(tt_move);
            scores[0] = 1 << 30;
        } else {
            generated = true;
            pos.gen_moves(&mut list, false);
            self.score_moves(pos, &list.moves[..list.len], &mut scores[..list.len], 0, killers, counter_move, prev1, prev2, prev4);
        }
        let mut best_score = -INF;
        let mut best_move = 0;
        let mut legal = 0;
        let mut quiets: [Move; 64] = [0; 64];
        let mut nq = 0;
        let mut capts: [Move; 32] = [0; 32];
        let mut nc = 0;
        let mut skip_quiets = false;
        let orig_alpha = alpha;
        let tt_capture = tt_move != 0 && is_noisy(tt_move);
        let lmp_limit = (tp(P::LmpBase) + depth * depth) / (2 - improving as i32);

        let mut n = list.len;
        let mut i = 0;
        let mut compacted = false;
        loop {
            if i >= n {
                if generated {
                    break;
                }
                generated = true;
                let mut tmp = MoveList::new();
                pos.gen_moves(&mut tmp, false);
                for k in 0..tmp.len {
                    if tmp.moves[k] != tt_move {
                        list.push(tmp.moves[k]);
                    }
                }
                n = list.len;
                self.score_moves(pos, &list.moves[i..n], &mut scores[i..n], tt_move, killers, counter_move, prev1, prev2, prev4);
                compacted = false;
                if i >= n {
                    break;
                }
            }
            if skip_quiets && !compacted {
                compacted = true;
                let mut k = i;
                for j in i..n {
                    if is_noisy(list.moves[j]) {
                        list.moves[k] = list.moves[j];
                        scores[k] = scores[j];
                        k += 1;
                    }
                }
                n = k;
                if i >= n {
                    break;
                }
            }
            pick(&mut list.moves, &mut scores, i, n);
            let m = list.moves[i];
            let mscore = scores[i];
            i += 1;
            if m == excluded {
                continue;
            }
            let quiet = !is_noisy(m);
            if quiet && skip_quiets {
                continue;
            }
            let hist_score = if quiet && mscore < (1 << 27) { mscore } else { 0 };
            if !root && best_score > -MATE_BOUND && pos.has_non_pawns(us) {
                if quiet {
                    if legal >= lmp_limit && !in_check {
                        self.stats.lmp += 1;
                        skip_quiets = true;
                        continue;
                    }
                    let lmr_d = (depth - self.lmr[depth.min(63) as usize][legal.min(63) as usize]).max(0);
                    if !in_check && lmr_d <= 8 && static_eval + tp(P::FutBase) + tp(P::FutMul) * lmr_d <= alpha {
                        self.stats.futility += 1;
                        skip_quiets = true;
                        continue;
                    }
                    if lmr_d <= 4 && hist_score < -tp(P::HistPrune) * depth {
                        self.stats.hist_prunes += 1;
                        continue;
                    }
                    if !pos.see_ge(m, -tp(P::SeeQuiet) * lmr_d * lmr_d) {
                        self.stats.see_quiet += 1;
                        continue;
                    }
                } else if depth <= 6 && !pos.see_ge(m, -tp(P::SeeNoisy) * depth) {
                    self.stats.see_noisy += 1;
                    continue;
                }
            }

            let mut child = *pos;
            if !child.make_move(m) {
                continue;
            }
            self.tt.prefetch(child.hash);
            legal += 1;

            // extensions
            let mut ext = 0;
            if !root
                && depth >= 7
                && m == tt_move
                && excluded == 0
                && tt_depth >= depth - 3
                && tt_bound != BOUND_UPPER
                && tt_score.abs() < MATE_BOUND
                && ply < 2 * self.root_depth as usize
            {
                self.stats.se_tries += 1;
                let sbeta = tt_score - depth * tp(P::SeMul) / 16;
                self.stack[ply].excluded = m;
                let v = self.negamax(pos, sbeta - 1, sbeta, (depth - 1) / 2, ply, cut_node);
                self.stack[ply].excluded = 0;
                if self.stopped {
                    return 0;
                }
                if v < sbeta {
                    ext = 1;
                    if !pv_node && v < sbeta - tp(P::SeDouble) {
                        ext = 2;
                        self.stats.se_double += 1;
                    } else {
                        self.stats.se_single += 1;
                    }
                } else if sbeta >= beta {
                    self.stats.multicut += 1;
                    return sbeta;
                } else if tt_score >= beta {
                    ext = -1;
                    self.stats.se_negative += 1;
                }
            } else if child.checkers != 0 {
                self.stats.check_ext += 1;
                ext = 1;
            }

            self.push_acc(ply, pos, &child, m);
            let pc = pos.board[mfrom(m)] as usize;
            self.stack[ply].mv = m;
            self.stack[ply].cont_idx = pc * 64 + mto(m);
            self.hash_hist.push(pos.hash);
            let nodes_before = self.nodes;
            let new_depth = depth - 1 + ext;
            let mut score;
            if legal == 1 {
                score = -self.negamax(&child, -beta, -alpha, new_depth, ply + 1, !pv_node && !cut_node);
            } else {
                let mut r = 0;
                if depth >= 3 && legal > 1 + root as i32 && (quiet || mscore < 0) {
                    r = self.lmr[depth.min(63) as usize][legal.min(63) as usize];
                    if !pv_node {
                        r += 1;
                    }
                    if cut_node {
                        r += 1;
                    }
                    if !improving {
                        r += 1;
                    }
                    if child.checkers != 0 {
                        r -= 1;
                    }
                    if mscore >= (1 << 27) {
                        r -= 1;
                    }
                    if tt_capture {
                        r += 1;
                    }
                    if quiet {
                        r -= hist_score / tp(P::LmrHistDiv);
                    } else {
                        let ch = self.capt[pc][mto(m)][pos.captured_type(m)] as i32;
                        r -= ch / tp(P::CapLmrDiv);
                    }
                    r = r.clamp(0, (new_depth - 1).max(0));
                }
                if r > 0 {
                    self.stats.lmr_searches += 1;
                }
                score = -self.negamax(&child, -alpha - 1, -alpha, new_depth - r, ply + 1, true);
                if score > alpha && r > 0 {
                    self.stats.lmr_researches += 1;
                    score = -self.negamax(&child, -alpha - 1, -alpha, new_depth, ply + 1, !cut_node);
                }
                if pv_node && score > alpha && score < beta {
                    score = -self.negamax(&child, -beta, -alpha, new_depth, ply + 1, false);
                }
            }
            self.hash_hist.pop();
            if self.stopped {
                return 0;
            }
            if root {
                self.root_node_counts[mfrom(m)][mto(m)] += self.nodes - nodes_before;
            }
            if score > best_score {
                best_score = score;
                if score > alpha {
                    best_move = m;
                    alpha = score;
                    // update pv
                    self.pv[ply][0] = m;
                    let cl = self.pv_len[ply + 1];
                    for k in 0..cl {
                        self.pv[ply][k + 1] = self.pv[ply + 1][k];
                    }
                    self.pv_len[ply] = cl + 1;
                    if root {
                        self.root_best = m;
                    }
                    if alpha >= beta {
                        let k = legal as u64;
                        self.stats.beta_cuts += 1;
                        self.stats.cut_pos_sum += k;
                        self.stats.cut_pos_sq_sum += k * k;
                        self.stats.first_move_cuts += (k == 1) as u64;
                        break;
                    }
                }
            }
            if m != best_move {
                if quiet {
                    if nq < 64 {
                        quiets[nq] = m;
                        nq += 1;
                    }
                } else if nc < 32 {
                    capts[nc] = m;
                    nc += 1;
                }
            }
        }

        if legal == 0 {
            return if excluded != 0 {
                alpha
            } else if in_check {
                -MATE + ply as i32
            } else {
                0
            };
        }

        if best_score >= beta {
            let bdepth = depth + (best_score > beta + 80) as i32;
            let bonus = (tp(P::HistMul) * bdepth - tp(P::HistOff)).min(tp(P::HistMax));
            let m = best_move;
            if !is_noisy(m) {
                if self.killers[ply][0] != m {
                    self.killers[ply][1] = self.killers[ply][0];
                    self.killers[ply][0] = m;
                }
                if ply >= 1 && self.stack[ply - 1].mv != 0 {
                    let pm = self.stack[ply - 1].mv;
                    self.counter[pos.board[mto(pm)] as usize][mto(pm)] = m;
                }
                self.update_quiet(pos, m, bonus, prev1, prev2, prev4);
                for k in 0..nq {
                    self.update_quiet(pos, quiets[k], -bonus, prev1, prev2, prev4);
                }
            } else {
                let pc = pos.board[mfrom(m)] as usize;
                let ct = pos.captured_type(m);
                upd(&mut self.capt[pc][mto(m)][ct], bonus);
            }
            for k in 0..nc {
                let cm = capts[k];
                let pc = pos.board[mfrom(cm)] as usize;
                let ct = pos.captured_type(cm);
                upd(&mut self.capt[pc][mto(cm)][ct], -bonus);
            }
        }

        if excluded == 0 {
            let bound = if best_score >= beta {
                BOUND_LOWER
            } else if alpha > orig_alpha {
                BOUND_EXACT
            } else {
                BOUND_UPPER
            };
            let mut ss = best_score;
            if ss >= MATE_BOUND {
                ss += ply as i32;
            } else if ss <= -MATE_BOUND {
                ss -= ply as i32;
            }
            self.tt.store(pos.hash, best_move, ss, raw_eval, depth, bound);
            if !in_check
                && (best_move == 0 || !is_noisy(best_move))
                && !(bound == BOUND_LOWER && best_score <= static_eval)
                && !(bound == BOUND_UPPER && best_score >= static_eval)
            {
                self.update_corr(pos, depth, best_score - static_eval);
            }
        }
        best_score
    }

    #[inline]
    fn update_quiet(&mut self, pos: &Position, m: Move, bonus: i32, prev1: usize, prev2: usize, prev4: usize) {
        let pc = pos.board[mfrom(m)] as usize;
        let ci = pc * 64 + mto(m);
        upd(&mut self.hist[pos.stm][mfrom(m)][mto(m)], bonus);
        if prev1 != 0 {
            upd(&mut self.cont[prev1][ci], bonus);
        }
        if prev2 != 0 {
            upd(&mut self.cont[prev2][ci], bonus);
        }
        if prev4 != 0 {
            upd(&mut self.cont[prev4][ci], bonus);
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn score_moves(&self, pos: &Position, moves: &[Move], scores: &mut [i32], tt_move: Move, killers: [Move; 2], counter_move: Move, prev1: usize, prev2: usize, prev4: usize) {
        let us = pos.stm;
        for i in 0..moves.len() {
            let m = moves[i];
            scores[i] = if m == tt_move {
                1 << 30
            } else if is_noisy(m) {
                let ct = pos.captured_type(m);
                let pc = pos.board[mfrom(m)] as usize;
                let base = SEE_VAL[ct.min(5)] * 16 * (ct < 6) as i32
                    + if is_promo(m) && promo_pt(m) == QUEEN { 8000 } else { 0 }
                    + self.capt[pc][mto(m)][ct] as i32 / 16;
                if pos.see_ge(m, -50) {
                    (1 << 28) + base
                } else {
                    -(1 << 28) + base
                }
            } else if m == killers[0] {
                (1 << 27) + 2
            } else if m == killers[1] {
                (1 << 27) + 1
            } else if m == counter_move {
                1 << 27
            } else {
                let pc = pos.board[mfrom(m)] as usize;
                let ci = pc * 64 + mto(m);
                self.hist[us][mfrom(m)][mto(m)] as i32 + self.cont[prev1][ci] as i32 + self.cont[prev2][ci] as i32 + self.cont[prev4][ci] as i32 / 2
            };
        }
    }

    fn qsearch(&mut self, pos: &Position, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        self.stats.qs_nodes += 1;
        if self.nodes & 1023 == 0 {
            self.check_time();
        }
        if self.stopped {
            return 0;
        }
        if ply > self.seldepth {
            self.seldepth = ply;
        }
        let in_check = pos.checkers != 0;
        if ply >= MAX_PLY - 2 {
            return if in_check { 0 } else { Self::damp(pos, self.evaluate(pos, ply)) };
        }
        if pos.insufficient_material() {
            return 0;
        }
        if pos.halfmove >= 100 {
            return if in_check && !pos.has_legal_move() { -MATE + ply as i32 } else { 0 };
        }
        let pv_node = beta - alpha > 1;
        self.stats.tt_probes += 1;
        let tte = self.tt.probe(pos.hash);
        let mut tt_move = 0;
        if let Some(e) = tte {
            self.stats.tt_hits += 1;
            let mut s = e.score as i32;
            if s >= MATE_BOUND {
                s -= ply as i32;
            } else if s <= -MATE_BOUND {
                s += ply as i32;
            }
            tt_move = e.mv;
            if !pv_node
                && pos.halfmove < 90
                && (e.bound == BOUND_EXACT || (e.bound == BOUND_LOWER && s >= beta) || (e.bound == BOUND_UPPER && s <= alpha))
            {
                self.stats.tt_cutoffs += 1;
                return s;
            }
        }
        let mut best;
        let static_eval;
        let mut raw_eval = -INF;
        if in_check {
            best = -INF;
            static_eval = -INF;
        } else {
            raw_eval = if let Some(e) = tte { e.eval as i32 } else { self.evaluate(pos, ply) };
            static_eval = self.corrected(pos, raw_eval);
            best = static_eval;
            if let Some(e) = tte {
                let s = e.score as i32;
                if s.abs() < MATE_BOUND
                    && (e.bound == BOUND_EXACT
                        || (e.bound == BOUND_LOWER && s > best)
                        || (e.bound == BOUND_UPPER && s < best))
                {
                    best = s;
                }
            }
            if best >= beta {
                return best;
            }
            if best > alpha {
                alpha = best;
            }
        }
        let mut list = MoveList::new();
        pos.gen_moves(&mut list, !in_check);
        #[allow(invalid_value)]
        let mut scores: [i32; 256] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..list.len {
            let m = list.moves[i];
            scores[i] = if m == tt_move {
                1 << 30
            } else if is_noisy(m) {
                let ct = pos.captured_type(m);
                let pc = pos.board[mfrom(m)] as usize;
                (1 << 20) + SEE_VAL[ct.min(5)] * 16 * (ct < 6) as i32 - SEE_VAL[pc_type(pc as u8)]
                    + if is_promo(m) { 8000 } else { 0 }
            } else {
                self.hist[pos.stm][mfrom(m)][mto(m)] as i32
            };
        }
        let mut best_move = 0;
        let mut legal = 0;
        let orig_alpha = alpha;
        for i in 0..list.len {
            pick(&mut list.moves, &mut scores, i, list.len);
            let m = list.moves[i];
            if !in_check {
                if !pos.see_ge(m, 0) {
                    continue;
                }
                let ct = pos.captured_type(m);
                if !is_promo(m) && static_eval + tp(P::QsFut) + SEE_VAL[ct.min(5)] <= alpha {
                    best = best.max(static_eval + tp(P::QsFut) + SEE_VAL[ct.min(5)]);
                    continue;
                }
            } else if legal > 0 && best > -MATE_BOUND && !is_noisy(m) && legal >= 3 {
                // limit quiet evasions
                continue;
            }
            let mut child = *pos;
            if !child.make_move(m) {
                continue;
            }
            legal += 1;
            self.push_acc(ply, pos, &child, m);
            let score = -self.qsearch(&child, -beta, -alpha, ply + 1);
            if self.stopped {
                return 0;
            }
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    best_move = m;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        if in_check && legal == 0 {
            // need to verify no legal moves at all (we may have skipped some)
            let mut any = false;
            for i in 0..list.len {
                let mut c = *pos;
                if c.make_move(list.moves[i]) {
                    any = true;
                    break;
                }
            }
            if !any {
                return -MATE + ply as i32;
            }
        }
        let bound = if best >= beta {
            BOUND_LOWER
        } else if alpha > orig_alpha {
            BOUND_EXACT
        } else {
            BOUND_UPPER
        };
        let mut ss = best;
        if ss >= MATE_BOUND {
            ss += ply as i32;
        } else if ss <= -MATE_BOUND {
            ss -= ply as i32;
        }
        self.tt.store(pos.hash, best_move, ss, raw_eval, 0, bound);
        best
    }
}
