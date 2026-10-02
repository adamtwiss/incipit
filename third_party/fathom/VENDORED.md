# Fathom (vendored)

Syzygy tablebase probing library, MIT licence (see LICENSE).

- Upstream: https://github.com/jdart1/Fathom
- Commit: c9c6fef0dddc05d2e242c183acf5833149ab676d (master, fetched 2026-10-03)
- Files: src/tbprobe.c, tbchess.c (included by tbprobe.c), tbprobe.h,
  tbconfig.h, stdendian.h, plus LICENSE and README.md. Unmodified.

Incipit uses it only through the public API in tbprobe.h; the engine's
build.rs compiles tbprobe.c with the system C compiler. Approved by the owner
(incipit-research docs/approved-tools.md): read the README and tbprobe.h only.
