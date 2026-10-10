// Lazy SMP: helper threads search the same root as the main thread at the
// same time, sharing its transposition table, so each finds the others'
// results there. Helpers never print, keep time or choose the move: the
// main thread searches as with one thread, then stops the helpers and waits
// for all of them before it answers.
use crate::position::{Move, Position};
use crate::search::{Limits, Searcher, MAX_PLY};
use crate::tt::TT;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

/// One helper's node count, on its own cache line (128 bytes: Apple's line
/// size, two of x86's) so helpers publishing don't contend.
#[repr(align(128))]
#[derive(Default)]
pub struct NodeSlot(pub AtomicU64);

type Task = Box<dyn FnOnce(&mut Searcher) + Send>;

enum Msg {
    Search { pos: Position, hist: Vec<u64>, moves: Vec<Move>, tt: Arc<TT> },
    Run(Task),
}

pub struct Pool {
    helpers: Vec<(Sender<Msg>, JoinHandle<()>)>,
    done: Receiver<()>,
    stop: Arc<AtomicBool>,
    nodes: Arc<[NodeSlot]>,
}

impl Pool {
    /// `n` helper threads (0: single-threaded search), each with its own
    /// Searcher (histories, stack, eval caches); only the TT is shared.
    pub fn new(n: usize) -> Pool {
        let stop = Arc::new(AtomicBool::new(false));
        let nodes: Arc<[NodeSlot]> = (0..n).map(|_| NodeSlot::default()).collect();
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
                        // Its own TT is 1 MB and unused: during a search it's
                        // swapped for the shared one.
                        let mut s = Searcher::new(1, stop);
                        s.silent = true;
                        s.thread_id = i + 1;
                        s.pool_nodes = nodes;
                        while let Ok(msg) = rx.recv() {
                            match msg {
                                Msg::Search { pos, hist, moves, tt } => {
                                    let own = std::mem::replace(&mut s.tt, tt);
                                    s.hash_hist = hist;
                                    s.search_moves = moves;
                                    let lim =
                                        Limits { soft_ms: None, hard_ms: None, depth: MAX_PLY as i32, nodes: None };
                                    s.search(&pos, &lim);
                                    s.pool_nodes[i].0.store(s.nodes, Ordering::Relaxed);
                                    // Drop the shared TT before reporting done, so the
                                    // main thread holds it alone between searches.
                                    s.tt = own;
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
    pub fn nodes(&self) -> Arc<[NodeSlot]> {
        self.nodes.clone()
    }

    /// Starts every helper searching `pos` (game history `hist`) with the
    /// main Searcher's TT and root moves (UCI searchmoves). They run until
    /// `finish`.
    pub fn start(&self, pos: &Position, hist: &[u64], main: &Searcher) {
        let tt = &main.tt;
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
            c.0.store(0, Ordering::Relaxed);
        }
        for (tx, _) in &self.helpers {
            let _ = tx.send(Msg::Search {
                pos: *pos,
                hist: hist.to_vec(),
                moves: main.search_moves.clone(),
                tt: tt.clone(),
            });
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
