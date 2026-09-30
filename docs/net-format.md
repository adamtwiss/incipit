# Incipit network format

An Incipit network file (`net-XXXXXXXX.nnue`, where `XXXXXXXX` is the first
8 hex digits of the file's SHA-256, upper case) is a self-describing header
followed by the quantised weights. The engine reads the architecture from the
header, so one engine build can load any network whose features it supports,
and rejects anything else with a clear message.

All integers are little-endian.

## Header

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | Magic: the ASCII bytes `INCPTNET` |
| 8 | 2 | Format version (currently 1) |
| 10 | 2 | Reserved, 0 |
| 12 | 4 | Header size in bytes: the weights start at this offset |
| 16 | … | Fields, then zero padding up to the header size |

Each field is:

| Size | Field |
|---|---|
| 2 | Tag |
| 2 | Reserved, 0 |
| 4 | Payload length in bytes |
| n | Payload |

A tag of 0 ends the field list. If bit 15 of a tag is set, the field is
optional: a reader that doesn't know it skips it. A reader that meets an
unknown field without bit 15 must reject the file. New features are added as
new tags, so the format doesn't run out of bits.

### Fields (version 1)

**`0x0001` INPUTS** (required). The input feature sets, in the order their
weights appear in each king bucket's block.

| Size | Field |
|---|---|
| 2 | Number of feature sets |
| 8 each | Kind (u16), reserved (u16), number of features (u32) |

Kinds: `1` = piece-square (768 features: `colour·384 + piece·64 + square`,
from the perspective's point of view; see [Inputs](#inputs)). Threat and pawn
features will be new kinds.

**`0x0002` KING_BUCKETS** (required).

| Size | Field |
|---|---|
| 1 | Number of king buckets |
| 1 | Horizontal mirroring: 1 if features are mirrored when the king is on files e–h |
| 2 | Reserved, 0 |
| 64 | Bucket for each king square, a1=0 … h8=63, from the perspective's point of view |

**`0x0003` FEATURE_TRANSFORMER** (required).

| Size | Field |
|---|---|
| 4 | Hidden size per perspective |
| 1 | Activation |
| 1 | Weight type |
| 1 | Bias type |
| 1 | Reserved, 0 |

**`0x0004` OUTPUT_BUCKETS** (required).

| Size | Field |
|---|---|
| 1 | Scheme: `1` = material count, bucket = min((pieces − 2) / ceil(32 / count), count − 1) |
| 1 | Number of buckets |
| 2 | Reserved, 0 |

**`0x0005` LAYER** (required, one per layer after the feature transformer, in order).

| Size | Field |
|---|---|
| 4 | Inputs |
| 4 | Outputs |
| 1 | Activation applied to the outputs (`0` for the final layer) |
| 1 | Weight type |
| 1 | Bias type |
| 1 | Flags: bit 0 = a separate set of weights and biases per output bucket |

**`0x0006` QUANTISATION** (required).

| Size | Field |
|---|---|
| 4 | QA (i32): the feature transformer's quantisation |
| 4 | QB (i32): the output layer's weight quantisation |
| 4 | Eval scale (i32): the network output is multiplied by this to give centipawns |

**`0x8001` DESCRIPTION** (optional). UTF-8 text: the training run, data and
settings that produced the network.

Codes used above:

* **Activations**: `0` none, `1` ReLU, `2` CReLU (clamp to [0, QA]), `3` SCReLU
  (clamp to [0, QA], then square), `4` pairwise CReLU (feature transformer only:
  clamp to [0, QA], then multiply the first half of each perspective's outputs
  elementwise by the second half, giving hidden / 2 values per perspective).
* **Types**: `1` i8, `2` i16, `3` i32, `4` f32.

## Weights

Directly after the header, with no padding between blocks:

1. Feature transformer weights: `[king bucket][input feature][hidden]`.
2. Feature transformer biases: `[hidden]`.
3. For each LAYER: weights `[output bucket][output][input]` (or `[output][input]`
   if not bucketed), then biases `[output bucket][output]` (or `[output]`).

The file must end exactly after the last block.

## Inputs

Each side has its own accumulator ("perspective"). For perspective `p`, a piece
of colour `c` and type `t` (pawn 0 … king 5) on square `s` gives feature

    bucket(k) · F + (c == p ? 0 : 384) + t · 64 + (s' ^ m)

where `F` is the number of features per bucket, `s'` is `s` for white and
`s ^ 56` for black (so each side sees its own pieces from rank 1), `k` is the
perspective's own king square seen the same way, and `m` is 7 if mirroring is on
and `k` is on files e–h, otherwise 0. This matches Bullet's `Chess768`,
`ChessBuckets` and `ChessBucketsMirrored` inputs.

## Evaluation

With one output layer (the only shape the engine supports so far), the side to
move's accumulator comes first in the output layer's input:

* SCReLU: `sum = Σ clamp(a, 0, QA)² · w` over both perspectives, then
  `eval = (sum / QA + bias) · scale / (QA · QB)`.
* CReLU: `sum = Σ clamp(a, 0, QA) · w`, then
  `eval = (sum + bias) · scale / (QA · QB)`.
* Pairwise CReLU: with `c = clamp(a, 0, QA)` and `h` the hidden size,
  `sum = Σ_{j < h/2} c[j] · c[j + h/2] · w` over both perspectives, then as for
  SCReLU: `eval = (sum / QA + bias) · scale / (QA · QB)`. The output layer then
  has `h` inputs (h/2 per perspective) instead of `2h`.

The output bias is at scale `QA · QB`.
