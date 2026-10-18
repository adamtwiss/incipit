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
from the perspective's point of view; see [Inputs](#inputs)); `2` = pawn pairs
(1488 features, see [Pawn pairs](#pawn-pairs)). Threat features will be a new
kind. Pawn pairs are shared by every king bucket: their rows are not repeated
per bucket but come once, after all the king buckets' rows (hidden-layer nets
only).

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
| 1 | Flags: bit 0 = a separate set of weights and biases per output bucket; bit 1 (first hidden layer only) = linear skip, bit 2 (first hidden layer only) = pre-activation readout, see below |

Linear skip (flag bit 1 on the first hidden layer): the layer's last neuron
has no activation. Its pre-activation (weights times inputs plus bias, in
real terms) is added to the network output, before the eval scale, and the
next layer takes only the other outputs - 1 neurons as its inputs. Needs a
second hidden layer.

Pre-activation readout (flag bit 2 on the first hidden layer): an f32 block
`[output bucket][neuron]` ends the file. Each neuron's pre-activation (in
real terms) times its weight for the position's bucket is added to the
network output, before the eval scale; the neurons still feed the next layer
as usual.

**`0x0006` QUANTISATION** (required).

| Size | Field |
|---|---|
| 4 | QA (i32): the feature transformer's quantisation |
| 4 | QB (i32): the output layer's weight quantisation |
| 4 | Eval scale (i32): the network output is multiplied by this to give centipawns |

**`0x0007` LAYER_QUANT** (required when there is a hidden layer). One 8-byte
entry per LAYER, in order:

| Size | Field |
|---|---|
| 4 | Input shift (u32): for an integer layer fed by the SCReLU feature transformer, its inputs are `clamp(a, 0, QA)² >> shift`, rounded to nearest, as u8 (must fit 0..127) |
| 4 | Weight scale (i32): integer weights are real weights times this; `0` for float layers |

**`0x8001` DESCRIPTION** (optional). UTF-8 text: the training run, data and
settings that produced the network.

Codes used above:

* **Activations**: `0` none, `1` ReLU, `2` CReLU (clamp to [0, QA]), `3` SCReLU
  (clamp to [0, QA], then square), `4` pairwise (feature transformer only:
  output `i` of a perspective is `clamp(a[i]) · clamp(a[i + H/2])`, so each
  perspective feeds H/2 values to the next layer, and H must be a multiple
  of 128).
* **Types**: `1` i8, `2` i16, `3` i32, `4` f32.

## Weights

Directly after the header, with no padding between blocks:

1. Feature transformer weights: `[king bucket][input feature][hidden]`, then,
   with pawn pairs, `[pair feature][hidden]`.
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

## Pawn pairs

One feature per unordered pair of pawn squares on ranks 2–7 that are on the
same file or adjacent files, per colour combination, from the perspective's
point of view: squares as for piece-square inputs (`s ^ 56` for black, then
`^ m` with the mirror), colours as ours (0) / theirs (1). The 372 square pairs
are numbered in order of `(a, b)`, `a < b`, over those squares; the feature is
`pair · 4 + colour(a) · 2 + colour(b)`. The engine's `src/pawnpairs.rs` is the
reference, and `engine/tests/pawnpairs_fixture.txt` lists the exact features
of 71 positions for both perspectives.

## Evaluation

With one output layer, the side to move's accumulator comes first in the output
layer's input:

* SCReLU: `sum = Σ clamp(a, 0, QA)² · w` over both perspectives, then
  `eval = (sum / QA + bias) · scale / (QA · QB)`.
* CReLU: `sum = Σ clamp(a, 0, QA) · w`, then
  `eval = (sum + bias) · scale / (QA · QB)`.

The output bias is at scale `QA · QB`.

With one hidden layer (two LAYERs: `2H -> N`, SCReLU, i8 weights, f32 biases;
then `N -> 1`, f32; both per output bucket; the engine supports N = 8 and 16,
and `H -> N` with the pairwise activation):

* `x = (clamp(a, 0, QA)² + 2^(shift−1)) >> shift` as u8 (round to nearest), side to move's accumulator first;
* `z[n] = Σ x · w1[bucket][n]` (integer), then
  `h[n] = clamp(z[n] / (S · W) + b1[bucket][n], 0, 1)²` with `S = QA² / 2^shift`
  (one input unit) and `W` the weight scale;
* `eval = (Σ h[n] · w2[bucket][n] + b2[bucket]) · scale`.

Weights for these layers are stored `[bucket][output][input]` as in the
general rule: i8 for the hidden layer, then its f32 biases, the final layer's
f32 weights `[bucket][N]` and its f32 biases `[bucket]`.

With a second hidden layer (three LAYERs): the middle one is `M -> L2`, SCReLU,
f32 weights and biases, per output bucket, where `M` is `N`, or `2N` when it
takes both the SCReLU and the CReLU of the first layer's outputs (the "dual"
head: the trainer's `l2relu`; the converter's `--l1-dual`), or `N - 1` with the
linear skip above; the final layer is then `L2 -> 1`. Its blocks follow the
first hidden layer's biases: f32 weights `[bucket][L2][M]`, f32 biases
`[bucket][L2]`, then the final layer's `[bucket][L2]` weights and `[bucket]`
biases, then the readout block if any. LAYER_QUANT carries an all-zero entry
for each f32 layer.
