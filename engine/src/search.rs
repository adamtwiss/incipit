// Search: iterative deepening PVS with the usual pruning/reduction heuristics.
use crate::nnue::{self, Acc};
use crate::params::{on, tp, P};
use crate::position::*;
use crate::tt::*;
use std::sync::atomic::{AtomicBool, AtomicI16, AtomicI32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub const INF: i32 = 32000;

/// Entries in an eval cache of `kb` kilobytes (8 bytes each, a power of two).
fn eval_cache_entries(kb: usize) -> usize {
    let n = (kb.max(1) << 10) / 8;
    1 << (usize::BITS - 1 - n.leading_zeros())
}
pub const MATE: i32 = 31000;
pub const MATE_BOUND: i32 = MATE - 512;
pub const MAX_PLY: usize = 128;
/// Default eval cache size in KB (UCI EvalCacheKB). Sized for the per-core
/// L2: under concurrent load (16 engines on sn1) 2 MB tables evict the FT
/// weights from the shared L3 and cost ~9% nps, while 256 KB is neutral for
/// the plain net and +5% for the hidden net.
pub const EVAL_CACHE_KB: usize = 256;
/// Tablebase wins score TB_WIN - ply: above every eval (evals stay below
/// MATE_BOUND) and below every mate (MATE - ply >= MATE - MAX_PLY), so code
/// that treats |score| >= MATE_BOUND as decisive handles them like mates.
pub const TB_WIN: i32 = MATE - 2 * MAX_PLY as i32;
const CORR_SIZE: usize = 16384;
pub const CORR_GRAIN: i32 = 256;

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
    pub evals: u64,
    pub eval_hits: u64,
    pub tt_probes: u64,
    pub tt_hits: u64,
    /// Lazy SMP: iterations where the vote changed the main thread's move.
    pub smp_vote_changes: u64,
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
        line(
            "Eval cache:",
            format!(
                "{} hits of {} lookups ({:.1}%)",
                self.eval_hits,
                self.eval_hits + self.evals,
                pct(self.eval_hits, self.eval_hits + self.evals)
            ),
        );
        line("Aspiration:", format!("fail-low {}, fail-high {}", self.asp_fail_low, self.asp_fail_high));
        line("RFP:", format!("{} cutoffs ({:.1}/Kn)", self.rfp, kn(self.rfp)));
        line(
            "Razoring:",
            format!(
                "{} tries, {} cutoffs ({:.0}%)",
                self.razor_tries,
                self.razor_cuts,
                pct(self.razor_cuts, self.razor_tries)
            ),
        );
        line(
            "Null move:",
            format!("{} tries, {} cutoffs ({:.0}%)", self.nmp_tries, self.nmp_cuts, pct(self.nmp_cuts, self.nmp_tries)),
        );
        line("ProbCut:", format!("{} cutoffs ({:.1}/Kn)", self.probcut_cuts, kn(self.probcut_cuts)));
        line("IIR:", format!("{} reductions", self.iir));
        line("LMP:", format!("{} triggers (remaining quiets skipped)", self.lmp));
        line("Futility:", format!("{} triggers (remaining quiets skipped)", self.futility));
        line("History prune:", format!("{} moves ({:.1}/Kn)", self.hist_prunes, kn(self.hist_prunes)));
        line(
            "SEE prune:",
            format!(
                "{} quiet, {} noisy ({:.1}/Kn)",
                self.see_quiet,
                self.see_noisy,
                kn(self.see_quiet + self.see_noisy)
            ),
        );
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
        o
    }
}

/// Milliseconds since the process's first call, for timestamps shared between threads.
pub fn now_ms() -> u64 {
    static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Starts its contents on a cache line: the per-ply arrays inline in Searcher
/// stay put relative to cache lines when other fields are added (field
/// layout changes moved them and cost 0.5-2% nps).
#[repr(C, align(64))]
struct Align64<T>(T);

impl<T> std::ops::Deref for Align64<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Align64<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

pub struct Limits {
    pub soft_ms: Option<u64>,
    pub hard_ms: Option<u64>,
    pub depth: i32,
    pub nodes: Option<u64>,
}

pub struct PonderTm {
    /// Set by the UCI thread on ponderhit: now_ms() + 1 (0 = no hit yet).
    pub hit: Arc<AtomicU64>,
    /// This search is a ponder search (go ponder).
    pub pondering: bool,
    start_ms: u64,
    soft_ms: f64,
    /// Latest soft target (ms), for stopping mid-depth after a ponder hit.
    target_ms: f64,
    /// The current depth failed low at the root and hasn't resolved yet.
    root_fail_low: bool,
    /// Best move and reply of the last completed depth (the root PV is reset
    /// when a new depth starts, so an aborted depth has none).
    last_pv2: (Move, Move),
    /// Last printed info line: depth, seldepth, score, PV (repeated before
    /// bestmove after a ponder hit, so GUIs record this move's search).
    pub last_info: (i32, usize, i32, String),
    /// Running average of our own clock spend / base soft limit this game
    /// (budget feedback; 1.0 at the start of a game).
    pub spend_avg: f64,
    /// The GUI has pondered this game: only then does the budget feedback
    /// apply (instant ponder hits are what leave the budget unspent).
    pub ponder_seen: bool,
}

impl PonderTm {
    /// Soft-limit multiplier from the budget feedback (1 = no change).
    pub fn feed_scale(&self) -> f64 {
        if !on(P::UseTmFeed) || !self.ponder_seen {
            return 1.0;
        }
        (1.0 / self.spend_avg.max(0.01)).clamp(1.0, tp(P::TmFeedMax) as f64 / 100.0)
    }

    /// Records one move: our own clock time and the unscaled soft limit.
    pub fn record_spend(&mut self, own_ms: u64, base_soft_ms: u64) {
        if base_soft_ms > 0 {
            let r = (own_ms as f64 / base_soft_ms as f64).min(5.0);
            self.spend_avg = 0.9 * self.spend_avg + 0.1 * r;
        }
    }
}

/// Field order is fixed (repr(C)): the fields touched at every node come first,
/// at small offsets (short instruction encodings), and fields added later go at
/// the end, so they can't shift the hot ones (layout changes cost 0.5-2% nps).
#[repr(C)]
pub struct Searcher {
    // Every node.
    pub nodes: u64,
    stopped: bool,
    pub silent: bool,
    /// UCI searchmoves: the root moves to consider (empty = all).
    pub search_moves: Vec<Move>,
    /// 0 for the main thread, 1.. for Lazy SMP helpers (see smp.rs).
    pub thread_id: usize,
    /// Helpers' node counts (helper i at i - 1), for the main thread's totals.
    pub helper_slots: Arc<[crate::smp::HelperSlot]>,
    root_best: Move,
    root_depth: i32,
    seldepth: usize,
    /// Raw static evals by position: hash bits 16..64 check the entry, bits
    /// 0..16 hold the eval (i16). Direct-mapped, indexed by the low hash bits.
    eval_cache: Box<[u64]>,
    eval_mask: usize,
    /// Shared with helper threads while they search; only the owner holds
    /// it between searches (see `clear`).
    pub tt: Arc<TT>,
    hist: Box<[[[i16; 64]; 64]; 2]>,
    /// Continuation and correction history: one set shared by all Lazy SMP
    /// threads while they search (see SharedHist).
    pub sh: Arc<SharedHist>,
    capt: Box<[[[i16; 7]; 64]; 12]>,
    pv: Box<[[Move; MAX_PLY + 2]; MAX_PLY + 2]>,
    lmr: Box<[[i32; 64]; 64]>,
    root_node_counts: Box<[[u64; 64]; 64]>,
    acc: Vec<Acc>,
    eval_scratch: Box<nnue::Scratch>,
    pub hash_hist: Vec<u64>,
    tb_hits: u64,
    // Per ply.
    killers: Align64<[[Move; 2]; MAX_PLY + 4]>,
    stack: Align64<[Frame; MAX_PLY + 8]>,
    pv_len: Align64<[usize; MAX_PLY + 2]>,
    pub stats: Stats,
    // Per search / time checks.
    pub stop_flag: Arc<AtomicBool>,
    start: Instant,
    hard_ms: Option<u64>,
    node_limit: Option<u64>,
    // Root of the current search, for printing moves (Chess960 castling needs
    // the castling rooks' squares, which are fixed for the game).
    root_pos: Position,
    refresh_cache: nnue::RefreshCache,
    /// Pondering and time-decision state.
    pub pt: Box<PonderTm>,
}

#[inline(always)]
fn pick(moves: &mut [Move], scores: &mut [i32], i: usize, n: usize) {
    // Slicing to n once lets the loop run without bounds checks.
    let (moves, scores) = (&mut moves[..n], &mut scores[..n]);
    let mut bi = i;
    let mut bs = scores[i];
    for j in i + 1..n {
        let s = scores[j];
        if s > bs {
            bs = s;
            bi = j;
        }
    }
    scores.swap(i, bi);
    moves.swap(i, bi);
}

#[inline(always)]
fn upd(h: &mut i16, bonus: i32) {
    let v = *h as i32;
    *h = (v + bonus - v * bonus.abs() / 16384) as i16;
}

/// `upd` on a shared entry: load then store (see `update_corr`).
#[inline(always)]
fn upd_shared(h: &AtomicI16, bonus: i32) {
    let v = h.load(Ordering::Relaxed) as i32;
    h.store((v + bonus - v * bonus.abs() / 16384) as i16, Ordering::Relaxed);
}

/// History tables shared by all threads of a search. Entries are atomics
/// used only with Relaxed loads and stores (plain moves on x86 and ARM):
/// each entry is a heuristic score, read and written alone, so no ordering
/// is needed and a lost concurrent update is harmless.
pub struct SharedHist {
    cont: Box<[[AtomicI16; 768]; 768]>,
    corr: Box<[[AtomicI32; CORR_SIZE]; 2]>,
    corr_np: Box<[[[AtomicI32; CORR_SIZE]; 2]; 2]>,
}

impl SharedHist {
    fn new() -> SharedHist {
        // Built on the heap (the correction tables are 384 KB): from_fn on
        // a Box<[_]> slice, then converted to the fixed-size box.
        fn zeroed<T>(n: usize, f: impl Fn() -> T) -> Box<[T]> {
            (0..n).map(|_| f()).collect()
        }
        let corr = zeroed(2, || std::array::from_fn(|_| AtomicI32::new(0)));
        let corr_np = zeroed(2, || std::array::from_fn(|_| std::array::from_fn(|_| AtomicI32::new(0))));
        SharedHist {
            cont: zeroed(768, || std::array::from_fn(|_| AtomicI16::new(0))).try_into().ok().unwrap(),
            corr: corr.try_into().ok().unwrap(),
            corr_np: corr_np.try_into().ok().unwrap(),
        }
    }
    /// Largest |entry| of the continuation and correction tables (tests).
    #[cfg(test)]
    pub fn max_abs(&self) -> (i32, i32) {
        let c = self.cont.iter().flatten().map(|x| (x.load(Ordering::Relaxed) as i32).abs()).max().unwrap();
        let k = self
            .corr
            .iter()
            .flatten()
            .chain(self.corr_np.iter().flatten().flatten())
            .map(|x| x.load(Ordering::Relaxed).abs())
            .max()
            .unwrap();
        (c, k)
    }
    /// Zeroes every entry. `&mut self` means no other thread is using the
    /// tables, so they can be zeroed as plain memory (zero atomics).
    pub fn clear(&mut self) {
        fn zero<T>(b: &mut T) {
            unsafe { std::ptr::write_bytes(b as *mut T, 0, 1) }
        }
        zero(&mut *self.cont);
        zero(&mut *self.corr);
        zero(&mut *self.corr_np);
    }
}

/// UCI_ShowWDL: append win/draw/loss permille to info lines.
pub static SHOW_WDL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// TmLog: one "info string pgncomment tm ..." line per search with the time
/// decision (budget, final target, why it stopped), for time-use analysis.
pub static TM_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Incipit's WDL model (datatools wdlfit on 150M positions of c19-c22,
/// 2026-10-05): P(win) = 1/(1+exp((a-x)/b)), P(loss) = 1/(1+exp((a+x)/b)),
/// a and b cubic in m = clamp(material, 17, 78) / 58 (1/3/3/5/9 over both
/// sides), x the score in our cp units.
const WDL_A: [f64; 4] = [-34.395460, 179.411649, -340.422392, 376.065278];
const WDL_B: [f64; 4] = [161.627807, -447.755780, 467.867011, -24.333415];

/// " wdl W D L" (permille, side to move, summing to 1000) when UCI_ShowWDL is
/// on, else "".
fn wdl_str(s: i32, pos: &Position) -> String {
    if !SHOW_WDL.load(std::sync::atomic::Ordering::Relaxed) {
        return String::new();
    }
    let (w, l) = if s.abs() >= MATE_BOUND {
        if s > 0 {
            (1000, 0)
        } else {
            (0, 1000)
        }
    } else {
        let mat: u32 = [(PAWN, 1), (KNIGHT, 3), (BISHOP, 3), (ROOK, 5), (QUEEN, 9)]
            .iter()
            .map(|&(pt, v)| v * pos.pieces[pt].count_ones())
            .sum();
        let m = (mat as f64).clamp(17.0, 78.0) / 58.0;
        let poly = |c: &[f64; 4]| ((c[0] * m + c[1]) * m + c[2]) * m + c[3];
        let (a, b) = (poly(&WDL_A), poly(&WDL_B));
        let x = s as f64;
        let w = 1000.0 / (1.0 + ((a - x) / b).exp());
        let l = 1000.0 / (1.0 + ((a + x) / b).exp());
        (w.round() as i32, l.round() as i32)
    };
    format!(" wdl {} {} {}", w, (1000 - w - l).max(0), l)
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
                lmr[d][m] = (tp(P::LmrBaseX100) as f64 / 100.0
                    + (d as f64).ln() * (m as f64).ln() / (tp(P::LmrDivX100) as f64 / 100.0))
                    as i32;
            }
        }
        Searcher {
            tt: Arc::new(TT::new(hash_mb)),
            eval_cache: vec![0u64; eval_cache_entries(EVAL_CACHE_KB)].into_boxed_slice(),
            eval_mask: eval_cache_entries(EVAL_CACHE_KB) - 1,
            nodes: 0,
            stop_flag,
            pt: Box::new(PonderTm {
                hit: Arc::new(AtomicU64::new(0)),
                pondering: false,
                start_ms: 0,
                soft_ms: 0.0,
                target_ms: f64::INFINITY,
                root_fail_low: false,
                last_pv2: (0, 0),
                last_info: (0, 0, 0, String::new()),
                spend_avg: 1.0,
                ponder_seen: false,
            }),
            stopped: false,
            start: Instant::now(),
            hard_ms: None,
            node_limit: None,
            hist: Box::new([[[0; 64]; 64]; 2]),
            sh: Arc::new(SharedHist::new()),
            capt: Box::new([[[0; 7]; 64]; 12]),
            killers: Align64([[0; 2]; MAX_PLY + 4]),
            stack: Align64([Frame::default(); MAX_PLY + 8]),
            pv: Box::new([[0; MAX_PLY + 2]; MAX_PLY + 2]),
            pv_len: Align64([0; MAX_PLY + 2]),
            hash_hist: Vec::with_capacity(1024),
            seldepth: 0,
            tb_hits: 0,
            root_depth: 0,
            root_best: 0,
            search_moves: Vec::new(),
            lmr,
            silent: false,
            thread_id: 0,
            helper_slots: Arc::new([]),
            root_node_counts: Box::new([[0; 64]; 64]),
            stats: Stats::default(),
            acc: vec![Acc::new(); MAX_PLY + 8],
            eval_scratch: nnue::Scratch::new(),
            root_pos: Position::empty(),
            refresh_cache: nnue::RefreshCache::new(),
        }
    }

    pub fn init_lmr(&mut self) {
        for d in 1..64 {
            for m in 1..64 {
                self.lmr[d][m] = (tp(P::LmrBaseX100) as f64 / 100.0
                    + (d as f64).ln() * (m as f64).ln() / (tp(P::LmrDivX100) as f64 / 100.0))
                    as i32;
            }
        }
    }

    /// Starts loading the eval-cache slot for `hash` (the table is too large
    /// for the CPU caches at bigger sizes).
    #[inline(always)]
    fn prefetch_eval(&self, hash: u64) {
        let p = unsafe { self.eval_cache.as_ptr().add(hash as usize & self.eval_mask) } as *const i8;
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(p, std::arch::x86_64::_MM_HINT_T0)
        };
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags, readonly))
        };
    }

    /// Resizes (and clears) the eval cache to the largest power-of-two
    /// number of entries that fits in `kb` kilobytes.
    /// The old cache is freed first, so the peak is the new size, not both;
    /// if the memory isn't there it falls back to smaller sizes (Err with
    /// the size it got) instead of aborting.
    pub fn set_eval_cache_kb(&mut self, kb: usize) -> Result<(), usize> {
        self.eval_cache = vec![0u64; 1].into_boxed_slice();
        self.eval_mask = 0;
        let mut n = eval_cache_entries(kb);
        loop {
            let mut v: Vec<u64> = Vec::new();
            if v.try_reserve_exact(n).is_ok() {
                v.resize(n, 0);
                self.eval_cache = v.into_boxed_slice();
                self.eval_mask = n - 1;
                return if n == eval_cache_entries(kb) { Ok(()) } else { Err(n * 8 / 1024) };
            }
            if n == 1 {
                return Err(0);
            }
            n /= 2;
        }
    }

    pub fn clear(&mut self) {
        self.pt.spend_avg = 1.0;
        self.pt.ponder_seen = false;
        // Helpers drop their handles when a search ends, so between searches
        // this is the only one.
        Arc::get_mut(&mut self.tt).expect("TT cleared during a search").clear();
        self.eval_cache.fill(0);
        *self.hist = [[[0; 64]; 64]; 2];
        Arc::get_mut(&mut self.sh).expect("histories cleared during a search").clear();
        *self.capt = [[[0; 7]; 64]; 12];
        self.killers = Align64([[0; 2]; MAX_PLY + 4]);
    }

    #[inline(always)]
    fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// After a ponder hit: (ms spent pondering, ms since the hit); None while
    /// still pondering.
    fn ponder_clock(&self) -> Option<(f64, u64)> {
        let hit = self.pt.hit.load(Ordering::Relaxed);
        if hit == 0 {
            return None;
        }
        let hit = hit - 1;
        Some((hit.saturating_sub(self.pt.start_ms) as f64, now_ms().saturating_sub(hit)))
    }

    /// Ponder hit: is the time we wanted for this move (`target`) used up? Move
    /// at once if we pondered most of it (and give the opponent our time
    /// back); else think on until pondering + thinking reaches it. A short
    /// ponder (an instant reply from the opponent) never gets an instant
    /// answer, so two pondering engines can't trade instant moves.
    fn ponder_done(&self, pondered: f64, since: u64, target: f64) -> bool {
        (since as f64) >= self.pt.soft_ms * tp(P::PonderMinPct) as f64 / 100.0
            && (pondered >= target * tp(P::PonderHitPct) as f64 / 100.0
                || pondered * tp(P::PonderCredit) as f64 / 100.0 + since as f64 >= target)
    }

    /// Soft-limit decision time: None means don't stop (still pondering).
    fn tm_elapsed(&self, target: f64) -> Option<f64> {
        if !self.pt.pondering {
            return Some(self.elapsed_ms() as f64);
        }
        let (pondered, since) = self.ponder_clock()?;
        self.ponder_done(pondered, since, target).then_some(f64::INFINITY)
    }

    /// Lazy SMP, after each completed iteration `d` with score `s` and the
    /// PV in pv[0]: a helper publishes its result; the main thread votes
    /// over its own and the helpers' latest results, weighting each thread
    /// by (score - lowest + VoteBase) * depth, and adopts the winning
    /// move with the score and PV of the deepest (then best-scoring) thread
    /// that chose it. Only threads at depth d - 1 or deeper vote. With no
    /// helpers this does nothing.
    fn smp_iteration(&mut self, d: i32, s: &mut i32) {
        if self.helper_slots.is_empty() {
            return;
        }
        let len = self.pv_len[0];
        if self.thread_id > 0 {
            let mut r = self.helper_slots[self.thread_id - 1].result.lock().unwrap();
            r.depth = d + (self.thread_id & 1) as i32;
            r.score = *s;
            r.len = len;
            r.pv[..len].copy_from_slice(&self.pv[0][..len]);
            return;
        }
        if len == 0 {
            return;
        }
        // Candidates: (depth, score, PV); the main thread's first.
        let mut cands = vec![crate::smp::IterResult { depth: d, score: *s, pv: [0; MAX_PLY + 2], len }];
        cands[0].pv[..len].copy_from_slice(&self.pv[0][..len]);
        for slot in self.helper_slots.iter() {
            let r = *slot.result.lock().unwrap();
            if r.depth >= d - 1 && r.len > 0 {
                cands.push(r);
            }
        }
        let low = cands.iter().map(|c| c.score).min().unwrap();
        let weight = |c: &crate::smp::IterResult| (c.score - low + tp(P::VoteBase)) as i64 * c.depth as i64;
        let votes = |m: Move| cands.iter().filter(|c| c.pv[0] == m).map(weight).sum::<i64>();
        // Strictly more votes to override the main thread's move.
        let mut win = cands[0].pv[0];
        let mut win_votes = votes(win);
        for c in &cands[1..] {
            let v = votes(c.pv[0]);
            if v > win_votes {
                (win, win_votes) = (c.pv[0], v);
            }
        }
        if win == cands[0].pv[0] {
            return;
        }
        let c = cands.iter().filter(|c| c.pv[0] == win).max_by_key(|c| (c.depth, c.score)).unwrap();
        self.stats.smp_vote_changes += 1;
        *s = c.score;
        self.pv_len[0] = c.len;
        self.pv[0][..c.len].copy_from_slice(&c.pv[..c.len]);
    }

    /// Nodes searched by all threads (helpers' counts lag by up to 1024 nodes each).
    pub fn total_nodes(&self) -> u64 {
        self.nodes + self.helper_slots.iter().map(|c| c.nodes.load(Ordering::Relaxed)).sum::<u64>()
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
            if self.pt.pondering {
                // Our clock runs only from the ponder hit.
                if let Some((pondered, since)) = self.ponder_clock() {
                    // A mid-depth stop keeps only finished work (aborted subtrees store
                    // nothing), but not while the previous best move has just failed low.
                    if since >= h || (!self.pt.root_fail_low && self.ponder_done(pondered, since, self.pt.target_ms)) {
                        self.stopped = true;
                    }
                }
            } else if self.elapsed_ms() >= h {
                self.stopped = true;
            }
        }
        if self.thread_id > 0 {
            self.helper_slots[self.thread_id - 1].nodes.store(self.nodes, Ordering::Relaxed);
        }
        if let Some(n) = self.node_limit {
            if self.total_nodes() >= n {
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
    fn evaluate(&mut self, pos: &Position, ply: usize) -> i32 {
        // Positions repeat within a search (transpositions, re-searches,
        // nodes that returned before storing to the TT): about 30% of
        // evaluations on bench.
        let slot = unsafe { self.eval_cache.get_unchecked_mut(pos.hash as usize & self.eval_mask) };
        if *slot & !0xffff == pos.hash & !0xffff && *slot != 0 {
            self.stats.eval_hits += 1;
            return *slot as u16 as i16 as i32;
        }
        let e = if cfg!(feature = "hce") {
            crate::eval::evaluate(pos)
        } else {
            nnue::evaluate(&self.acc[ply], pos, &mut self.eval_scratch)
        };
        let e = e.clamp(-MATE_BOUND + 1, MATE_BOUND - 1);
        *slot = (pos.hash & !0xffff) | (e as i16 as u16 as u64);
        self.stats.evals += 1;
        e
    }

    /// Pulls an eval towards a draw as the fifty-move counter rises.
    #[inline(always)]
    fn damp(pos: &Position, v: i32) -> i32 {
        v * (200 - pos.halfmove as i32) / 200
    }

    #[inline(always)]
    fn corrected(&self, pos: &Position, raw: i32) -> i32 {
        let m = CORR_SIZE - 1;
        if !on(P::UseCorrHist) {
            return Self::damp(pos, raw).clamp(-MATE_BOUND + 1, MATE_BOUND - 1);
        }
        let (corr, corr_np) = (&self.sh.corr[pos.stm], &self.sh.corr_np[pos.stm]);
        let c = (tp(P::CorrPawnWeight) * corr[(pos.pawn_key as usize) & m].load(Ordering::Relaxed)
            + tp(P::CorrNonPawnWeight)
                * (corr_np[0][(pos.np_key[0] as usize) & m].load(Ordering::Relaxed)
                    + corr_np[1][(pos.np_key[1] as usize) & m].load(Ordering::Relaxed)))
            / (128 * CORR_GRAIN);
        Self::damp(pos, raw + c).clamp(-MATE_BOUND + 1, MATE_BOUND - 1)
    }

    fn update_corr(&mut self, pos: &Position, depth: i32, diff: i32) {
        let w = (depth + 1).min(tp(P::CorrUpdateCap));
        let target = diff.clamp(-tp(P::CorrDiffClamp), tp(P::CorrDiffClamp)) * CORR_GRAIN;
        let m = CORR_SIZE - 1;
        let lim = CORR_GRAIN * tp(P::CorrLimit);
        // Load then store, not an atomic read-modify-write: when two threads
        // update an entry at once one update is lost, which costs less.
        let f = |e: &AtomicI32| {
            let v = e.load(Ordering::Relaxed);
            e.store(((v * (256 - w) + target * w) / 256).clamp(-lim, lim), Ordering::Relaxed)
        };
        let (corr, corr_np) = (&self.sh.corr[pos.stm], &self.sh.corr_np[pos.stm]);
        f(&corr[(pos.pawn_key as usize) & m]);
        f(&corr_np[0][(pos.np_key[0] as usize) & m]);
        f(&corr_np[1][(pos.np_key[1] as usize) & m]);
    }

    /// Whether playing `m` at the root reaches a position already in the game
    /// history (a repetition the tablebases don't know about).
    fn repeats_after(&self, root: &Position, m: Move) -> bool {
        let mut c = *root;
        c.make_move(m);
        let n = self.hash_hist.len();
        let lim = (c.halfmove as usize).min(n + 1);
        // c's ancestors: root (1 ply back), then hash_hist from the end.
        (2..=lim).step_by(2).any(|k| self.hash_hist.get(n + 1 - k) == Some(&c.hash))
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

    /// The move to ponder on after `best`: the reply in the last PV, if any.
    pub fn ponder_move(&self, best: Move) -> Move {
        if self.pt.last_pv2.0 == best {
            self.pt.last_pv2.1
        } else {
            0
        }
    }

    /// Returns (best move, score).
    pub fn search(&mut self, root: &Position, lim: &Limits) -> (Move, i32) {
        self.start = Instant::now();
        self.pt.start_ms = now_ms();
        self.pt.target_ms = lim.soft_ms.map_or(f64::INFINITY, |s| s as f64);
        self.pt.soft_ms = lim.soft_ms.unwrap_or(0) as f64;
        // One generation per search: with helpers, Pool::start advances it
        // before any thread starts.
        if self.helper_slots.is_empty() {
            self.tt.new_search();
        }
        self.nodes = 0;
        self.tb_hits = 0;
        self.stopped = false;
        self.hard_ms = lim.hard_ms;
        self.node_limit = lim.nodes;
        self.root_best = 0;
        self.pt.last_pv2 = (0, 0);
        self.seldepth = 0;
        self.acc[0].refresh(root);
        self.pt.last_info.0 = 0;
        self.root_pos = *root;
        for r in self.root_node_counts.iter_mut() {
            *r = [0; 64];
        }
        // fallback move
        let mut list = MoveList::new();
        root.gen_moves(&mut list, false);
        let mut fallback = 0;
        for i in 0..list.len() {
            let mut c = *root;
            if (self.search_moves.is_empty() || self.search_moves.contains(&list[i])) && c.make_move(list[i]) {
                fallback = list[i];
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
        if crate::tb::largest() > 0 && self.search_moves.is_empty() {
            if let Some((w, m)) = crate::tb::probe_root(root).filter(|&(_, m)| !self.repeats_after(root, m)) {
                let s = match w {
                    crate::tb::Wdl::Win => TB_WIN - 1,
                    crate::tb::Wdl::Loss => -TB_WIN + 1,
                    crate::tb::Wdl::Draw => 0,
                };
                if !self.silent {
                    // A tablebase result is certain: report it as such, not
                    // through the eval-based WDL model (a draw is 0 1000 0).
                    let wdl = if SHOW_WDL.load(Ordering::Relaxed) {
                        match w {
                            crate::tb::Wdl::Win => " wdl 1000 0 0",
                            crate::tb::Wdl::Loss => " wdl 0 0 1000",
                            crate::tb::Wdl::Draw => " wdl 0 1000 0",
                        }
                    } else {
                        ""
                    };
                    println!(
                        "info depth 1 score {}{} nodes 0 tbhits 1 time 0 pv {}",
                        score_str(s),
                        wdl,
                        root.move_uci(m)
                    );
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
        // TmLog: final soft target, its factors, the last completed depth and why we stopped.
        let mut tm_stop = "depth";
        let (mut tm_target, mut tm_frac, mut tm_done) = (0.0f64, 0.0f64, 0);
        for d in 1..=max_depth {
            let nodes_at_start = self.nodes;
            self.root_depth = d;
            let mut delta = tp(P::AspDelta);
            let (mut a, mut b) = if d >= 4 && on(P::UseAsp) { (score - delta, score + delta) } else { (-INF, INF) };
            let mut fh_move: Move = 0;
            let mut fh_score = 0;
            self.pt.root_fail_low = false;
            let mut fail_lows = 0;
            let mut s;
            loop {
                self.seldepth = 0;
                // Lazy SMP diversity: odd helpers search one ply deeper than
                // the iteration, so threads spread over two depths and the
                // deeper ones fill the TT ahead of the main thread.
                let ds = d + (self.thread_id & 1) as i32;
                s = self.negamax(root, a, b, ds, 0, false);
                if self.stopped {
                    break;
                }
                if s <= a {
                    self.stats.asp_fail_low += 1;
                    self.pt.root_fail_low = true;
                    fail_lows += 1;
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
                tm_stop = if self.stop_flag.load(Ordering::Relaxed) {
                    "stop"
                } else if lim.nodes.is_some_and(|n| self.total_nodes() >= n) {
                    "nodes"
                } else if self.pt.pondering {
                    "ponder"
                } else {
                    "hard"
                };
                if self.root_best != 0 {
                    best = self.root_best;
                }
                // Return the score that goes with the move: a stopped
                // aspiration fail-high's move carries its lower bound (as the
                // info line below reports), not the last completed depth's
                // score.
                if best == fh_move && fh_move != 0 {
                    score = fh_score;
                }
                if !self.silent && best != reported {
                    let sc = if best == fh_move {
                        format!("{} lowerbound{}", score_str(fh_score), wdl_str(fh_score, root))
                    } else {
                        format!("{}{}", score_str(score), wdl_str(score, root))
                    };
                    let el = self.elapsed_ms();
                    println!(
                        "info depth {} score {} nodes {} nps {} time {} pv {}",
                        d,
                        sc,
                        self.total_nodes(),
                        self.total_nodes() * 1000 / el.max(1),
                        el,
                        root.move_uci(best)
                    );
                }
                break;
            }
            let iter_nodes = self.nodes - nodes_at_start;
            let last_ebf = if prev_iter_nodes > 0 { iter_nodes as f64 / prev_iter_nodes as f64 } else { 0.0 };
            if d >= 5 && prev_iter_nodes > 0 {
                self.stats.ebf_log_sum += (iter_nodes as f64 / prev_iter_nodes as f64).ln();
                self.stats.ebf_count += 1;
            }
            prev_iter_nodes = iter_nodes;
            self.smp_iteration(d, &mut s);
            if self.pv_len[0] >= 2 {
                self.pt.last_pv2 = (self.pv[0][0], self.pv[0][1]);
            }
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
                let score_scale = if on(P::UseTmScore) { score_scale } else { 1.0 };
                let fl_scale = if on(P::UseTmFailLow) {
                    1.0 + fail_lows.min(3) as f64 * tp(P::TmFailLow) as f64 / 100.0
                } else {
                    1.0
                };
                let (node_scale, stab_scale) = if on(P::UseTm) { (node_scale, stab_scale) } else { (1.0, 1.0) };
                let ext = if on(P::UseTmExtMax) { score_scale.max(fl_scale) } else { score_scale * fl_scale };
                let target = soft as f64 * node_scale * stab_scale * ext;
                self.pt.target_ms = target;
                (tm_target, tm_frac, tm_done) = (target, frac, d);
                let el = self.elapsed_ms() as f64;
                if self.tm_elapsed(target).is_some_and(|el| el >= target) {
                    tm_stop = "soft";
                    break;
                }
                // The next depth costs about this one times the branching factor;
                // elapsed so far approximates this depth plus all earlier ones.
                if on(P::UseTmFinish) && d >= 6 && last_ebf > 0.0 {
                    if let Some(h) = self.hard_ms {
                        let ebf = last_ebf.clamp(1.2, 4.0);
                        let iter_ms = el * iter_nodes as f64 / self.nodes.max(1) as f64;
                        if el + iter_ms * ebf > h as f64 * tp(P::TmFinishPct) as f64 / 100.0 {
                            tm_stop = "finish";
                            break;
                        }
                    }
                }
            }
            if let Some(n) = lim.nodes {
                if self.total_nodes() >= n {
                    tm_stop = "nodes";
                    break;
                }
            }
        }
        // After a ponder hit the last info line went out while pondering; GUIs
        // attribute this move only to output after the hit, so repeat it.
        if self.pt.pondering && self.pt.hit.load(Ordering::Relaxed) != 0 && !self.silent && self.pt.last_info.0 > 0 {
            self.print_last_info();
        }
        if TM_LOG.load(Ordering::Relaxed) && !self.silent {
            println!(
                "info string pgncomment tm el={} soft={} hard={} tgt={:.0} stab={} frac={:.2} d={} stop={}",
                self.elapsed_ms(),
                lim.soft_ms.unwrap_or(0),
                lim.hard_ms.unwrap_or(0),
                tm_target,
                stability,
                tm_frac,
                tm_done,
                tm_stop
            );
        }
        (best, score)
    }

    fn print_info(&mut self, d: i32, score: i32) {
        let mut pv = String::new();
        for i in 0..self.pv_len[0] {
            pv.push(' ');
            pv.push_str(&self.root_pos.move_uci(self.pv[0][i]));
        }
        self.pt.last_info = (d, self.seldepth, score, pv);
        self.print_last_info();
    }

    /// Prints the last completed depth's info line (current nodes and time).
    pub fn print_last_info(&self) {
        let (d, seldepth, score, ref pv) = self.pt.last_info;
        let el = self.elapsed_ms();
        let nodes = self.total_nodes();
        let nps = nodes * 1000 / el.max(1);
        println!(
            "info depth {} seldepth {} score {}{} nodes {} nps {} hashfull {} tbhits {} time {} pv{}",
            d,
            seldepth,
            score_str(score),
            wdl_str(score, &self.root_pos),
            nodes,
            nps,
            self.tt.hashfull(),
            self.tb_hits,
            el,
            pv
        );
    }

    fn negamax(
        &mut self,
        pos: &Position,
        mut alpha: i32,
        mut beta: i32,
        mut depth: i32,
        ply: usize,
        cut_node: bool,
    ) -> i32 {
        let pv_node = beta - alpha > 1;
        let root = ply == 0;
        self.pv_len[ply] = 0;
        if depth <= 0 {
            // qsearch doesn't check repetitions, so a move into a repeated
            // position at the last ply would otherwise not be scored a draw.
            if !root && self.is_repetition(pos) {
                return 0;
            }
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
                && on(P::UseTtCut)
                && tt_depth >= depth
                // Near the fifty-move draw a stored score may be stale (the
                // counter isn't in the key); refuse only scores well away
                // from a draw, so fortress endgames can still cut.
                && (pos.halfmove < 90 || tt_score.abs() <= tp(P::HmGuard))
                && (tt_bound == BOUND_EXACT
                    || (tt_bound == BOUND_LOWER && tt_score >= beta)
                    || (tt_bound == BOUND_UPPER && tt_score <= alpha))
            {
                if tt_score >= beta && tt_move != 0 && !is_noisy(tt_move) {
                    // reward quiet tt move causing cutoff
                    let bonus = (tp(P::HistMul) * depth - tp(P::HistOff)).min(tp(P::HistMax));
                    upd(&mut self.hist[pos.stm][mfrom(tt_move)][mto(tt_move)], bonus);
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
                    let ss = if s >= MATE_BOUND {
                        s + ply as i32
                    } else if s <= -MATE_BOUND {
                        s - ply as i32
                    } else {
                        s
                    };
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
        // Compare with our previous move's eval; if we were in check then, with the one before.
        let improving = !in_check
            && if ply >= 2 && self.stack[ply - 2].static_eval != -INF {
                static_eval > self.stack[ply - 2].static_eval
            } else if ply >= 4 && self.stack[ply - 4].static_eval != -INF {
                static_eval > self.stack[ply - 4].static_eval
            } else {
                false
            };
        self.killers[ply + 1] = [0, 0];

        if !pv_node && !in_check && excluded == 0 {
            // reverse futility pruning
            if on(P::UseRfp)
                && depth <= tp(P::RfpDepth)
                && eval.abs() < MATE_BOUND
                && eval - tp(P::RfpMargin) * (depth - improving as i32) >= beta
            {
                self.stats.rfp += 1;
                return (eval + beta) / 2;
            }
            // razoring
            if on(P::UseRazor) && eval + tp(P::RazorBase) + tp(P::RazorMul) * depth * depth <= alpha {
                self.stats.razor_tries += 1;
                let v = self.qsearch(pos, alpha, alpha + 1, ply);
                if v <= alpha {
                    self.stats.razor_cuts += 1;
                    return v;
                }
            }
            // null move pruning
            if on(P::UseNmp)
                && depth >= tp(P::NmpDepth)
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
                self.tt.prefetch(child.hash);
                self.prefetch_eval(child.hash);
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
            if on(P::UseProbcut)
                && depth >= tp(P::ProbcutDepth)
                && beta.abs() < MATE_BOUND
                && !(tte.is_some() && tt_depth >= depth - 3 && tt_score < pc_beta)
            {
                let mut list = MoveList::new();
                pos.gen_moves(&mut list, true);
                for i in 0..list.len() {
                    let m = list[i];
                    if !pos.see_ge(m, pc_beta - static_eval) {
                        continue;
                    }
                    let mut child = *pos;
                    if !child.make_move(m) {
                        continue;
                    }
                    self.tt.prefetch(child.hash);
                    self.prefetch_eval(child.hash);
                    self.push_acc(ply, pos, &child, m);
                    let pc = pos.board[mfrom(m)] as usize;
                    self.stack[ply].mv = m;
                    self.stack[ply].cont_idx = pc * 64 + mto(m);
                    self.hash_hist.push(pos.hash);
                    let mut v = -self.qsearch(&child, -pc_beta, -pc_beta + 1, ply + 1);
                    if v >= pc_beta {
                        v = -self.negamax(
                            &child,
                            -pc_beta,
                            -pc_beta + 1,
                            depth - tp(P::ProbcutRed),
                            ply + 1,
                            !cut_node,
                        );
                    }
                    self.hash_hist.pop();
                    if self.stopped {
                        return 0;
                    }
                    if v >= pc_beta {
                        let ss = if v >= MATE_BOUND { v + ply as i32 } else { v };
                        self.tt.store(pos.hash, m, ss, raw_eval, depth - 3, BOUND_LOWER);
                        self.stats.probcut_cuts += 1;
                        return v;
                    }
                }
            }
        }
        // internal iterative reduction
        if on(P::UseIir) && depth >= tp(P::IirDepth) && tt_move == 0 && (pv_node || cut_node) {
            self.stats.iir += 1;
            depth -= 1;
        }

        let mut list = MoveList::new();
        let mut scores = Stack256::<i32>::new();
        let us = pos.stm;
        let prev1 = if ply >= 1 { self.stack[ply - 1].cont_idx } else { 0 };
        let prev2 = if ply >= 2 { self.stack[ply - 2].cont_idx } else { 0 };
        let prev4 = if ply >= 4 { self.stack[ply - 4].cont_idx } else { 0 };
        let killers = if on(P::UseKillers) { self.killers[ply] } else { [0, 0] };
        let mut generated = false;
        if tt_move != 0 && pos.is_pseudo_legal(tt_move) {
            list.push(tt_move);
            scores.push(1 << 30);
        } else {
            generated = true;
            pos.gen_moves(&mut list, false);
            self.score_moves(pos, list.as_slice(), &mut scores, 0, killers, prev1, prev2, prev4);
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

        let mut n = list.len();
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
                for k in 0..tmp.len() {
                    if tmp[k] != tt_move {
                        list.push(tmp[k]);
                    }
                }
                n = list.len();
                // Scored so far: the TT move, so the new scores line up with list[i..].
                debug_assert_eq!(scores.len(), i);
                self.score_moves(pos, &list.as_slice()[i..n], &mut scores, tt_move, killers, prev1, prev2, prev4);
                compacted = false;
                if i >= n {
                    break;
                }
            }
            if skip_quiets && !compacted {
                compacted = true;
                let mut k = i;
                for j in i..n {
                    if is_noisy(list[j]) {
                        list[k] = list[j];
                        scores[k] = scores[j];
                        k += 1;
                    }
                }
                n = k;
                if i >= n {
                    break;
                }
            }
            pick(list.as_mut_slice(), scores.as_mut_slice(), i, n);
            let m = list[i];
            let mscore = scores[i];
            i += 1;
            if m == excluded || (root && !self.search_moves.is_empty() && !self.search_moves.contains(&m)) {
                continue;
            }
            let quiet = !is_noisy(m);
            if quiet && skip_quiets {
                continue;
            }
            let hist_score = if quiet && mscore < (1 << 27) { mscore } else { 0 };
            if !root && best_score > -MATE_BOUND && pos.has_non_pawns(us) {
                if quiet {
                    if on(P::UseLmp) && legal >= lmp_limit && !in_check {
                        self.stats.lmp += 1;
                        skip_quiets = true;
                        continue;
                    }
                    let lmr_d = (depth - self.lmr[depth.min(63) as usize][legal.min(63) as usize]).max(0);
                    if on(P::UseFut)
                        && !in_check
                        && lmr_d <= tp(P::FutDepth)
                        && static_eval + tp(P::FutBase) + tp(P::FutMul) * lmr_d <= alpha
                    {
                        self.stats.futility += 1;
                        skip_quiets = true;
                        continue;
                    }
                    if on(P::UseHistPrune)
                        && !in_check
                        && lmr_d <= tp(P::HistPruneDepth)
                        && hist_score < -tp(P::HistPrune) * depth
                    {
                        self.stats.hist_prunes += 1;
                        continue;
                    }
                    if on(P::UseSeeQuiet) && !in_check && !pos.see_ge(m, -tp(P::SeeQuiet) * lmr_d * lmr_d) {
                        self.stats.see_quiet += 1;
                        continue;
                    }
                } else if on(P::UseSeeNoisy)
                    && depth <= tp(P::SeeNoisyDepth)
                    && !pos.see_ge(m, -tp(P::SeeNoisy) * depth)
                {
                    self.stats.see_noisy += 1;
                    continue;
                }
            }

            let mut child = *pos;
            if !child.make_move(m) {
                continue;
            }
            self.tt.prefetch(child.hash);
            self.prefetch_eval(child.hash);
            legal += 1;

            // extensions
            let mut ext = 0;
            if on(P::UseSe)
                && !root
                && depth >= tp(P::SeDepth)
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
                    if on(P::UseSeDoubleExt) && !pv_node && v < sbeta - tp(P::SeDouble) {
                        ext = 2;
                        self.stats.se_double += 1;
                    } else {
                        self.stats.se_single += 1;
                    }
                } else if on(P::UseMulticut) && sbeta >= beta {
                    self.stats.multicut += 1;
                    return sbeta;
                } else if on(P::UseSeNegExt) && tt_score >= beta {
                    ext = -1;
                    self.stats.se_negative += 1;
                }
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
                if on(P::UseLmr) && depth >= 3 && legal > 1 + root as i32 && (quiet || mscore < 0) {
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
            upd_shared(&self.sh.cont[prev1][ci], bonus);
        }
        if prev2 != 0 {
            upd_shared(&self.sh.cont[prev2][ci], bonus);
        }
        if prev4 != 0 {
            upd_shared(&self.sh.cont[prev4][ci], bonus);
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn score_moves(
        &self,
        pos: &Position,
        moves: &[Move],
        scores: &mut Stack256<i32>,
        tt_move: Move,
        killers: [Move; 2],
        prev1: usize,
        prev2: usize,
        prev4: usize,
    ) {
        let us = pos.stm;
        for &m in moves {
            let score = if m == tt_move {
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
            } else {
                let pc = pos.board[mfrom(m)] as usize;
                let ci = pc * 64 + mto(m);
                self.hist[us][mfrom(m)][mto(m)] as i32
                    + self.sh.cont[prev1][ci].load(Ordering::Relaxed) as i32
                    + self.sh.cont[prev2][ci].load(Ordering::Relaxed) as i32
                    + self.sh.cont[prev4][ci].load(Ordering::Relaxed) as i32 / 2
            };
            scores.push(score);
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
                && (pos.halfmove < 90 || s.abs() <= tp(P::HmGuard))
                && (e.bound == BOUND_EXACT
                    || (e.bound == BOUND_LOWER && s >= beta)
                    || (e.bound == BOUND_UPPER && s <= alpha))
            {
                self.stats.tt_cutoffs += 1;
                return s;
            }
        }
        let mut best;
        let mut raw_eval = -INF;
        if in_check {
            best = -INF;
        } else {
            raw_eval = if let Some(e) = tte { e.eval as i32 } else { self.evaluate(pos, ply) };
            best = self.corrected(pos, raw_eval);
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
        let mut scores = Stack256::<i32>::new();
        for &m in list.as_slice() {
            let score = if m == tt_move {
                1 << 30
            } else if is_noisy(m) {
                let ct = pos.captured_type(m);
                let pc = pos.board[mfrom(m)] as usize;
                (1 << 20) + SEE_VAL[ct.min(5)] * 16 * (ct < 6) as i32 - SEE_VAL[pc_type(pc as u8)]
                    + if is_promo(m) { 8000 } else { 0 }
            } else {
                self.hist[pos.stm][mfrom(m)][mto(m)] as i32
            };
            scores.push(score);
        }
        let mut best_move = 0;
        let mut legal = 0;
        let orig_alpha = alpha;
        let n = list.len();
        for i in 0..n {
            pick(list.as_mut_slice(), scores.as_mut_slice(), i, n);
            let m = list[i];
            if !in_check {
                if on(P::UseQsSee) && !pos.see_ge(m, 0) {
                    continue;
                }
            } else if on(P::UseQsEvasionLimit) && legal > 0 && best > -MATE_BOUND && !is_noisy(m) && legal >= 3 {
                // limit quiet evasions
                continue;
            }
            let mut child = *pos;
            if !child.make_move(m) {
                continue;
            }
            self.tt.prefetch(child.hash);
            self.prefetch_eval(child.hash);
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
            for i in 0..list.len() {
                let mut c = *pos;
                if c.make_move(list[i]) {
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
