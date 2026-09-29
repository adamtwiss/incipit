# Incipit

**Incipit** is an experimental chess engine by Adam Twiss, author of [Coda](https://github.com/adamtwiss/coda). Coda (and its predecessor GoChess) was one of the first chess engines developed entirely using AI coding agents, and now plays at a high level (#4 on the CCRL 40/15 single-CPU list, as of September 2026).

Like Coda, Incipit is developed with AI and agent-driven coding throughout. The difference is where its knowledge comes from. Parts of Coda's development leaned heavily on studying other engines and trying similar techniques in it. Incipit takes no knowledge directly from other engines:

* **No other engine's code:** Agents working on Incipit may not read any other engine's source code, commit history or documentation.
* **No other engine's data:** No networks, tuned parameters or training data from other engines. NNUE training data comes entirely from Incipit's own self-play, over multiple generations (bootstrapped from a minimal material and mobility eval).
* **General knowledge is fine:** Well-established techniques (null-move pruning, reverse futility pruning, NNUE and so on) are used as described in the public literature, such as the Chess Programming Wiki and academic papers.

These rules are all codified and enforced in `CLAUDE.md`.

Language models doing the work were almost certainly trained on much of that same literature, and potentially on public engine source code too. These rules can't undo what a model learned during its training; they govern what the agents are given and allowed to consult while working. Language models have advanced a great deal since Coda started, and Incipit is a test of how far this approach can go.

**The name:** in music, a *coda* is the closing passage of a piece, and an *incipit* is its opening. The name reflects that Incipit starts afresh, from nothing.

**Status:** Incipit is an experiment (it may fail, it may become a full-blown engine). Development is being done in the open and this repo is public; however, it isn't yet intended for external testing. Don't expect much in the way of strength until there's a sufficient volume of quality training data to support larger models.

**License:** Incipit is open-source and licensed under GPLv3 (or later).

## Credits

Incipit is built, tested and trained with these tools:

* [OpenBench](https://github.com/AndyGrant/OpenBench), created by Andrew Grant: testing (SPRTs, SPSA tuning and self-play data generation) runs on a self-hosted, modified OpenBench instance.
* [fastchess](https://github.com/Disservin/fastchess): running engine matches for testing.
* [viriformat](https://github.com/cosmobobak/viriformat) by Cosmo Bobak: the training data format.
* [Bullet](https://github.com/jw1912/bullet) by Jamie Whiting: NNUE training.
