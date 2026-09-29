# Incipit datatools

Converts Incipit's training data to [viriformat](https://github.com/cosmobobak/viriformat), the format used for NNUE training, and trained networks to Incipit's network format ([docs/net-format.md](../docs/net-format.md)).

```
cargo build --release
datatools pgn <out.vf> <in.pgn | -> ...     # OpenBench datagen PGNs
datatools bin <out_dir> <in.bin> ...        # the engine's old datagen .bin files
datatools stats <file.vf> ...               # count games and positions
datatools net <bullet|raw> <in> <out_dir> --hidden N [options]   # network -> Incipit format
datatools net-info <file.nnue> ...          # show a network's header
```

`datatools net` reads Bullet's `quantised.bin` (`bullet`) or the engine's old headerless nets (`raw`) and writes `<out_dir>/net-XXXXXXXX.nnue`, named by its SHA-256 as usual. Options: `--king-buckets` (32 or 64 comma-separated entries), `--mirror`, `--output-buckets N` (default 8), `--activation screlu|crelu`, `--qa`/`--qb`/`--scale` (default 255/64/400), and `--description` to record the training run.

OpenBench PGN archives hold bzip2-compressed shards, so pipe them in:
`tar xf 1234.pgn.tar && bzcat *.pgn.bz2 | datatools pgn 1234.vf -`

* **PGNs**: each game becomes one viriformat game. Scores are converted to white-relative centipawns; mate scores and unscored moves are written as 32767 so trainers' eval filters drop them.
* **Old `.bin` files** only stored positions that weren't in check and whose best move was quiet, without the moves between them. Games are rebuilt by searching for the missing moves (up to 7 plies); positions in between get the 32767 "don't train" score. About 98.8% of records are kept; the rest are the last records of chains that couldn't be continued.

The board code is the engine's own (`engine/src/attacks.rs` and `position.rs`, included directly), so the tool has no dependencies. The `viriformat` crate is used only in tests, to check our output against the reference reader:

```
cargo test --release
DATATOOLS_CHECK=file.vf cargo test --release -- --ignored                   # decode a real file
DATATOOLS_BIN=x.bin DATATOOLS_VF=x.vf cargo test --release -- --ignored     # check a .bin conversion
```
