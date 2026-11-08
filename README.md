# Incipit

**Incipit** is an experimental chess engine by Adam Twiss, author of [Coda](https://github.com/adamtwiss/coda). Coda (and its predecessor GoChess) was one of the first chess engines developed entirely using AI coding agents, and now plays at a high level (#5 on the CCRL 40/15 single-CPU list, as of September 2026).

Like Coda, Incipit is developed with AI and agent-driven coding throughout. The difference is where its knowledge comes from. Parts of Coda's development leaned heavily on studying other engines and trying similar techniques in it. Incipit takes no knowledge directly from other engines:

* **No other engine's code:** Agents working on Incipit may not read any other engine's source code, commit history or documentation.
* **No other engine's data:** No networks, tuned parameters or training data from other engines. NNUE training data comes entirely from Incipit's own self-play, over multiple generations (bootstrapped from a minimal material and mobility eval).
* **General knowledge is fine:** Well-established techniques (null-move pruning, reverse futility pruning, NNUE and so on) are used as described in the public literature, such as the Chess Programming Wiki and academic papers.

These rules are all codified and enforced in `CLAUDE.md`.

Language models doing the work were almost certainly trained on much of that same literature, and potentially on public engine source code too. These rules can't undo what a model learned during its training; they govern what the agents are given and allowed to consult while working. Language models have advanced a great deal since Coda started, and Incipit is a test of how far this approach can go.

**The name:** The Coda name was an acronym, but in music, a *coda* is also the closing passage of a piece, and an *incipit* is its opening. The name reflects that Incipit starts afresh, from nothing.

**Status:** Incipit is an experiment that is still developing quickly. Version 0.6.0 is its second public release, for testers and rating lists (0.5.0 was the first). Expect regular releases as its search improves and its networks grow with more self-play data.

**License:** Incipit is open-source and licensed under GPLv3 (or later).

**NNUE Training Data:** Incipit's training data is all fully self-generated, [freely-available](https://huggingface.co/datasets/adamtwiss/incipit-data) and openly licensed under a Creative Commons license (with attribution).

## Download

Prebuilt binaries are attached to each [release](https://github.com/adamtwiss/incipit/releases), with the network embedded:

| File | Platform |
|---|---|
| `incipit-vX.Y.Z-linux-x86-64` | Linux, x86-64 |
| `incipit-vX.Y.Z-windows-x86-64.exe` | Windows, x86-64 |
| `incipit-vX.Y.Z-macos-arm64` | macOS, Apple Silicon |
| `incipit-vX.Y.Z-macos-x86-64` | macOS, Intel |
| `incipit-vX.Y.Z-linux-aarch64` | Linux, 64-bit ARM |
| `incipit-vX.Y.Z-linux-aarch64-musl` | Linux, 64-bit ARM, statically linked (e.g. Android/Termux) |

The x86-64 builds need AVX2 and BMI2 (Intel Haswell or AMD Zen 1 and later). Faster code paths (AVX-512, VNNI, PEXT) are chosen at start-up when the CPU has them; `incipit bench` prints what was chosen.

## Features

* **Evaluation:** an NNUE network trained only on Incipit's own self-play games (about 33 billion positions so far, published as the [incipit-data](https://huggingface.co/datasets/adamtwiss/incipit-data) dataset). Its inputs are piece-square features in 16 king buckets (mirrored), plus pawn-pair and threat features (which pieces attack which); then a 1024-wide feature transformer with pairwise activation, hidden layers of 16 and 16, and 6 output buckets by material.
* **Search:** alpha-beta (principal variation search) with the usual selectivity (null-move pruning, late-move reductions, singular extensions and more), a transposition table, and Lazy SMP for multiple threads.
* **Endgame tablebases:** Syzygy, via Fathom.
* **Chess960** (and double Fischer random).
* **Pondering**, and win/draw/loss output (`UCI_ShowWDL`).
* **Normalised scores:** a reported score of +100 means about a 50% chance of winning, whatever the material. Mate and tablebase scores are reported as usual.

## UCI options

| Option | Default | |
|---|---|---|
| `Hash` | 64 | Transposition table size in MB (1-65536). |
| `Threads` | 1 | Search threads (1-256). |
| `MoveOverhead` | 20 | Milliseconds reserved per move for communication lag (0-5000). `Move Overhead` is accepted too. |
| `Ponder` | false | Allow pondering. |
| `MultiPV` | 1 | Number of best lines to search and report (1-256), for analysis. |
| `Clear Hash` | (button) | Empty the transposition table and history tables. |
| `UCI_Chess960` | false | Chess960 castling notation (king takes rook). |
| `UCI_ShowWDL` | false | Add win/draw/loss estimates (per mille) to `info` lines. |
| `NormalizeScore` | true | Report scores on the 100 = 50% winning-chance scale; false gives the network's raw scale. |
| `SyzygyPath` | (empty) | Directories of Syzygy tablebase files, separated by `:` (`;` on Windows). |
| `MoveReserve` | 0 | In sudden-death games, keep time back for this many further moves (useful for online play with network lag). |
| `TmLog` | false | Developer option: print a time-management summary after each search. |

## Command line

* `incipit bench [depth]`: searches a fixed set of positions and prints the node count and speed.
* `incipit epd <file> [nodes N | movetime MS | depth D] [hash MB] [threads T] [perft D] [verbose]`: runs an EPD test suite. It understands `bm`, `am`, `dm` (mate in N) and `Dn` (perft counts). It reports the positions solved and the depth at which each answer was found.

The other commands (`datagen`, `genfens`, `l1perm`, `l1stats`, `l1check`, `nnuecheck`, `tune-spec`) are development tools.

## Building

You need Rust (1.92 or later; 1.98 or later on ARM for the dot-product code) and a C compiler for Fathom.

```
cd engine
make          # downloads the network named in net.txt, then builds ./incipit for this machine
```

`make net` only downloads the network. To build by hand: `EVALFILE=/path/to/net.nnue cargo build --release`.

## Credits

Incipit is built, tested and trained with these tools:

* [OpenBench](https://github.com/AndyGrant/OpenBench), created by Andrew Grant: testing (SPRTs, SPSA tuning and self-play data generation) runs on a self-hosted, modified OpenBench instance.
* [fastchess](https://github.com/Disservin/fastchess): running engine matches for testing.
* [viriformat](https://github.com/cosmobobak/viriformat) by Cosmo Bobak: the training data format.
* [Bullet](https://github.com/jw1912/bullet) by Jamie Whiting: NNUE training.
* [Fathom](https://github.com/jdart1/Fathom) (Ronald de Man, basil00, Jon Dart): Syzygy endgame tablebase probing, vendored in `third_party/fathom`.

Third-party components and their licences are listed in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
