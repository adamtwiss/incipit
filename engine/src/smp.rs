// Lazy SMP: helper threads search the same root as the main thread at the
// same time, sharing its transposition table, so each finds the others'
// results there. Helpers never print, keep time or choose the move: the
// main thread searches as with one thread, then stops the helpers and waits
// for all of them before it answers.
use crate::position::{Move, Position};
use crate::search::{Limits, Searcher, SharedHist, MAX_PLY};
use crate::tt::TT;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// A helper's latest completed iteration, for the main thread's vote.
#[derive(Clone, Copy)]
pub struct IterResult {
    /// Depth searched (0: none yet this search).
    pub depth: i32,
    pub score: i32,
    pub pv: [Move; MAX_PLY + 2],
    pub len: usize,
}

/// What one helper publishes, on its own cache lines (128 bytes: Apple's
/// line size, two of x86's) so helpers publishing don't contend: its node
/// count every 1024 nodes, and each completed iteration's result.
#[repr(align(128))]
pub struct HelperSlot {
    pub nodes: AtomicU64,
    pub result: Mutex<IterResult>,
}

impl Default for HelperSlot {
    fn default() -> HelperSlot {
        let r = IterResult { depth: 0, score: 0, pv: [0; MAX_PLY + 2], len: 0 };
        HelperSlot { nodes: AtomicU64::new(0), result: Mutex::new(r) }
    }
}

impl HelperSlot {
    fn reset(&self) {
        self.nodes.store(0, Ordering::Relaxed);
        // Clear the whole result: a stale PV from the previous search is a
        // different position's move, and must not vote (even at weight 0 its
        // score would set the vote floor).
        let mut r = self.result.lock().unwrap();
        (r.depth, r.score, r.len) = (0, 0, 0);
    }
}

type Task = Box<dyn FnOnce(&mut Searcher) + Send>;

enum Msg {
    Search { pos: Position, hist: Vec<u64>, moves: Vec<Move>, tt: Arc<TT>, sh: Arc<SharedHist> },
    Run(Task),
}

pub struct Pool {
    helpers: Vec<(Sender<Msg>, JoinHandle<()>)>,
    done: Receiver<()>,
    stop: Arc<AtomicBool>,
    nodes: Arc<[HelperSlot]>,
    /// Helpers that took the last message, so `wait` knows how many `done`
    /// signals to expect once a helper has died.
    pending: AtomicUsize,
}

/// Sends `done` when dropped: at the end of a helper's task, or while a
/// panic unwinds it, so `wait` never hangs on a dead helper.
struct Done<'a>(&'a Sender<()>);
impl Drop for Done<'_> {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

impl Pool {
    /// `n` helper threads (0: single-threaded search), each with its own
    /// Searcher (histories, stack, eval caches); only the TT is shared.
    pub fn new(n: usize) -> Pool {
        let stop = Arc::new(AtomicBool::new(false));
        let nodes: Arc<[HelperSlot]> = (0..n).map(|_| HelperSlot::default()).collect();
        let (done_tx, done) = channel();
        let helpers = (0..n)
            .map(|i| {
                let (tx, rx) = channel::<Msg>();
                let (stop, nodes, done_tx) = (stop.clone(), nodes.clone(), done_tx.clone());
                let handle = std::thread::Builder::new()
                    .name(format!("helper{}", i + 1))
                    // The main thread's default stack size, which search fits in.
                    .stack_size(8 << 20)
                    .spawn(move || {
                        // Its own TT (1 MB) and SharedHist are unused: during a
                        // search they're swapped for the main thread's. Both
                        // come zeroed from the allocator and are never
                        // touched, so they cost address space, not memory.
                        let mut s = Searcher::new(1, stop);
                        s.silent = true;
                        s.thread_id = i + 1;
                        s.helper_slots = nodes;
                        while let Ok(msg) = rx.recv() {
                            let done = Done(&done_tx);
                            match msg {
                                Msg::Search { pos, hist, moves, tt, sh } => {
                                    let own = std::mem::replace(&mut s.tt, tt);
                                    let own_sh = std::mem::replace(&mut s.sh, sh);
                                    s.hash_hist = hist;
                                    s.search_moves = moves;
                                    let lim =
                                        Limits { soft_ms: None, hard_ms: None, depth: MAX_PLY as i32, nodes: None };
                                    s.search(&pos, &lim);
                                    s.helper_slots[i].nodes.store(s.nodes, Ordering::Relaxed);
                                    // Drop the shared TT before reporting done, so the
                                    // main thread holds it alone between searches.
                                    s.tt = own;
                                    s.sh = own_sh;
                                }
                                Msg::Run(f) => f(&mut s),
                            }
                            drop(done);
                        }
                    })
                    .expect("spawn helper thread");
                (tx, handle)
            })
            .collect();
        Pool { helpers, done, stop, nodes, pending: AtomicUsize::new(0) }
    }

    pub fn len(&self) -> usize {
        self.helpers.len()
    }

    /// The node slots to give the main thread's Searcher.
    pub fn slots(&self) -> Arc<[HelperSlot]> {
        self.nodes.clone()
    }

    /// Starts every helper searching `pos` (game history `hist`) with the
    /// main Searcher's TT, shared histories and root moves (UCI
    /// searchmoves). They run until `finish`.
    pub fn start(&self, pos: &Position, hist: &[u64], main: &Searcher) {
        let (tt, sh) = (&main.tt, &main.sh);
        // Helpers are idle here (after `finish`), so these stores happen
        // before their searches start: the channel send orders them.
        self.stop.store(false, Ordering::Relaxed);
        if self.helpers.is_empty() {
            return;
        }
        // The generation advances once per search, here rather than in
        // Searcher::search (which does it only without helpers), so every
        // thread stores under the new one from its first node.
        tt.new_search();
        for c in self.nodes.iter() {
            c.reset();
        }
        self.send(|| Msg::Search {
            pos: *pos,
            hist: hist.to_vec(),
            moves: main.search_moves.clone(),
            tt: tt.clone(),
            sh: sh.clone(),
        });
    }

    /// Sends a message to every helper still alive and counts those that
    /// took it, for `wait`. A helper that panicked is gone for good: its
    /// channel is closed, so it is skipped from then on.
    fn send(&self, msg: impl Fn() -> Msg) {
        let mut n = 0;
        for (tx, _) in &self.helpers {
            if tx.send(msg()).is_ok() {
                n += 1;
            }
        }
        self.pending.store(n, Ordering::Relaxed);
    }

    /// Stops the helpers and waits until all have finished searching.
    pub fn finish(&self) {
        self.stop.store(true, Ordering::Relaxed);
        self.wait();
    }

    /// Runs `f` on every helper's Searcher (e.g. clear, option changes) and
    /// waits for all. Only call while the helpers are idle.
    pub fn each(&self, f: impl Fn(&mut Searcher) + Clone + Send + 'static) {
        self.send(|| Msg::Run(Box::new(f.clone())));
        self.wait();
    }

    /// Waits for every helper that took the last message to signal done
    /// (a panicking helper signals it while unwinding: see `Done`).
    fn wait(&self) {
        for _ in 0..self.pending.swap(0, Ordering::Relaxed) {
            if self.done.recv().is_err() {
                break; // every helper has exited
            }
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // Closing the channels ends each helper's loop.
        for (tx, handle) in self.helpers.drain(..) {
            drop(tx);
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{tp, P};
    use crate::search::CORR_GRAIN;
    use std::sync::atomic::AtomicBool;

    /// A helper that panics must not hang the pool: `each` and `finish`
    /// return, and the other helpers keep searching.
    #[test]
    fn dead_helper_does_not_hang() {
        crate::attacks::init();
        crate::nnue::init();
        let pool = Pool::new(2);
        let mut main = Searcher::new(8, Arc::new(AtomicBool::new(false)));
        main.silent = true;
        main.helper_slots = pool.slots();
        pool.each(|s| {
            if s.thread_id == 1 {
                panic!("test: helper 1 dies");
            }
        });
        let pos = Position::from_fen(crate::position::START_FEN).unwrap();
        let lim = Limits { soft_ms: None, hard_ms: None, depth: 8, nodes: None };
        for _ in 0..2 {
            pool.start(&pos, &[], &main);
            let (m, _) = main.search(&pos, &lim);
            pool.finish();
            assert!(m != 0);
            assert!(pool.nodes[1].nodes.load(Ordering::Relaxed) > 0, "helper 2 searched");
        }
        pool.each(|_| {});
    }

    /// Four threads search together, sharing the TT and histories, over
    /// several positions: no panic, every search returns a legal move, and
    /// the shared tables stay within the bounds their updates keep (lost
    /// concurrent updates must not break them).
    #[test]
    fn helpers_share_tables() {
        crate::attacks::init();
        crate::nnue::init();
        let pool = Pool::new(3);
        let mut main = Searcher::new(8, Arc::new(AtomicBool::new(false)));
        main.silent = true;
        main.helper_slots = pool.slots();
        let fens = [
            crate::position::START_FEN,
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];
        for _ in 0..3 {
            for f in fens {
                let pos = Position::from_fen(f).unwrap();
                pool.start(&pos, &[], &main);
                let lim = Limits { soft_ms: None, hard_ms: None, depth: 9, nodes: None };
                let (m, _) = main.search(&pos, &lim);
                pool.finish();
                let mut c = pos;
                assert!(m != 0 && pos.is_pseudo_legal(m) && c.make_move(m), "{f}");
                assert!(main.total_nodes() > main.nodes, "helpers searched");
            }
        }
        let (c, k) = main.sh.max_abs();
        assert!(c <= 16384, "continuation history out of range: {c}");
        assert!(k <= CORR_GRAIN * tp(P::CorrLimit), "correction history out of range: {k}");
        eprintln!("vote changed the move in {} iterations", main.stats.smp_vote_changes);
        // Between searches the main thread holds the tables alone again.
        main.clear();
    }
}
