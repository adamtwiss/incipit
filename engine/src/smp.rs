// Lazy SMP: helper threads search the same root as the main thread at the
// same time, sharing its transposition table, so each finds the others'
// results there. Helpers never print, keep time or choose the move: the
// main thread searches as with one thread, then stops the helpers and waits
// for all of them before it answers.
use crate::position::{Move, Position};
use crate::search::{Limits, Searcher, SharedHist, MAX_PLY};
use crate::tt::TT;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
        self.result.lock().unwrap().depth = 0;
    }
}

type Task = Box<dyn FnOnce(&mut Searcher) + Send>;

enum Msg {
    Search { pos: Position, hist: Vec<u64>, tt: Arc<TT>, sh: Arc<SharedHist> },
    Run(Task),
}

pub struct Pool {
    helpers: Vec<(Sender<Msg>, JoinHandle<()>)>,
    done: Receiver<()>,
    stop: Arc<AtomicBool>,
    nodes: Arc<[HelperSlot]>,
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
                        // search they're swapped for the main thread's.
                        let mut s = Searcher::new(1, stop);
                        s.silent = true;
                        s.thread_id = i + 1;
                        s.helper_slots = nodes;
                        while let Ok(msg) = rx.recv() {
                            match msg {
                                Msg::Search { pos, hist, tt, sh } => {
                                    let own = std::mem::replace(&mut s.tt, tt);
                                    let own_sh = std::mem::replace(&mut s.sh, sh);
                                    s.hash_hist = hist;
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
                            if done_tx.send(()).is_err() {
                                break;
                            }
                        }
                    })
                    .expect("spawn helper thread");
                (tx, handle)
            })
            .collect();
        Pool { helpers, done, stop, nodes }
    }

    pub fn len(&self) -> usize {
        self.helpers.len()
    }

    /// The node slots to give the main thread's Searcher.
    pub fn slots(&self) -> Arc<[HelperSlot]> {
        self.nodes.clone()
    }

    /// Starts every helper searching `pos` (game history `hist`) with the
    /// main Searcher's TT and shared histories. They run until `finish`.
    pub fn start(&self, pos: &Position, hist: &[u64], main: &Searcher) {
        let (tt, sh) = (&main.tt, &main.sh);
        // Helpers are idle here (after `finish`), so these stores happen
        // before their searches start: the channel send orders them.
        self.stop.store(false, Ordering::Relaxed);
        if self.helpers.is_empty() {
            return;
        }
        // The main Searcher shares `nodes` and so leaves this to us (see
        // Searcher::search).
        tt.new_search();
        for c in self.nodes.iter() {
            c.reset();
        }
        for (tx, _) in &self.helpers {
            let _ = tx.send(Msg::Search { pos: *pos, hist: hist.to_vec(), tt: tt.clone(), sh: sh.clone() });
        }
    }

    /// Stops the helpers and waits until all have finished searching.
    pub fn finish(&self) {
        self.stop.store(true, Ordering::Relaxed);
        self.wait();
    }

    /// Runs `f` on every helper's Searcher (e.g. clear, option changes) and
    /// waits for all. Only call while the helpers are idle.
    pub fn each(&self, f: impl Fn(&mut Searcher) + Clone + Send + 'static) {
        for (tx, _) in &self.helpers {
            let _ = tx.send(Msg::Run(Box::new(f.clone())));
        }
        self.wait();
    }

    fn wait(&self) {
        for _ in 0..self.helpers.len() {
            // Err: every helper has exited (one panicked); nothing to wait for.
            if self.done.recv().is_err() {
                break;
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
