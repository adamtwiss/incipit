#!/usr/bin/env python3
"""UCI stress test for multi-threaded search.

Drives the engine with random go / stop / ponderhit / isready / ucinewgame /
setoption sequences at random intervals and checks: no crash, no hang
(every isready answered), and exactly one bestmove per go, never one while
pondering or in go infinite before stop/ponderhit.

    python3 tests/uci_stress.py target/release/incipit [rounds] [seed]
"""
import queue
import random
import subprocess
import sys
import threading
import time

ENGINE = sys.argv[1]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else int(time.time())
rng = random.Random(SEED)

POSITIONS = [
    "position startpos",
    "position startpos moves e2e4 e7e5 g1f3 b8c6 f1b5",
    "position fen r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "position fen 8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
    "position fen 6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",  # mate in 1
    "position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # stalemate: no legal move
    "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
]

p = subprocess.Popen([ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
lines = queue.Queue()


def reader():
    for line in p.stdout:
        lines.put(line.rstrip("\n"))
    lines.put(None)


threading.Thread(target=reader, daemon=True).start()
gos = bestmoves = 0
log = []


def send(cmd):
    log.append("> " + cmd)
    p.stdin.write(cmd + "\n")
    p.stdin.flush()


def fail(msg):
    print(f"FAIL (seed {SEED}): {msg}")
    print("\n".join(log[-40:]))
    p.kill()
    sys.exit(1)


def drain(until=None, timeout=10.0):
    """Reads output: everything available now, or with `until`, up to and
    including a line starting with it (failing after `timeout` seconds)."""
    global bestmoves
    deadline = time.time() + timeout
    while True:
        try:
            if until is None:
                line = lines.get_nowait()
            else:
                line = lines.get(timeout=max(deadline - time.time(), 0.001))
        except queue.Empty:
            if until is None:
                return
            fail(f"timeout waiting for {until}")
        if line is None:
            fail(f"engine exited (code {p.wait()})")
        log.append("< " + line[:100])
        if line.startswith("bestmove"):
            bestmoves += 1
            if bestmoves > gos:
                fail("bestmove without a go")
        if until and line.startswith(until):
            return


def sync():
    send("isready")
    drain("readyok")


send("uci")
drain("uciok")
for r in range(ROUNDS):
    if rng.random() < 0.15:
        send(f"setoption name Threads value {rng.choice([1, 2, 3, 4, 8])}")
    if rng.random() < 0.05:
        send(f"setoption name Hash value {rng.choice([1, 8, 16, 64])}")
    if rng.random() < 0.1:
        send("ucinewgame")
    send(rng.choice(POSITIONS))
    kind = rng.choice(["movetime", "depth", "nodes", "infinite", "ponder", "clock"])
    go = {
        "movetime": f"go movetime {rng.randint(1, 60)}",
        "depth": f"go depth {rng.randint(1, 9)}",
        "nodes": f"go nodes {rng.randint(1, 50000)}",
        "infinite": "go infinite",
        "ponder": f"go ponder wtime {rng.randint(50, 3000)} btime {rng.randint(50, 3000)} winc 10 binc 10",
        "clock": f"go wtime {rng.randint(20, 2000)} btime {rng.randint(20, 2000)}",
    }[kind]
    want = gos + 1
    send(go)
    gos += 1
    time.sleep(rng.random() * 0.05)
    if rng.random() < 0.5:
        send("isready")  # answered at once, even mid-search
        drain("readyok")
    if kind in ("infinite", "ponder"):
        time.sleep(rng.random() * 0.05)
        drain()
        if bestmoves >= want:
            fail(f"bestmove during {kind}")
        send("ponderhit" if kind == "ponder" and rng.random() < 0.5 else "stop")
    elif rng.random() < 0.3:
        send("stop")
    if rng.random() < 0.3:
        # Pipelined: the next go may arrive before this bestmove.
        continue
    while bestmoves < want:
        drain("bestmove", timeout=15)
sync()
while bestmoves < gos:
    drain("bestmove", timeout=15)
send("quit")
try:
    code = p.wait(timeout=10)
except subprocess.TimeoutExpired:
    fail("no exit after quit")
while (line := lines.get()) is not None:
    if line.startswith("bestmove"):
        bestmoves += 1
if bestmoves != gos:
    fail(f"{gos} go, {bestmoves} bestmove")
print(f"ok: {ROUNDS} rounds, {gos} go, {bestmoves} bestmove, seed {SEED}, exit {code}")
