# `draft_head`: MXFP4 lm_head GEMV + TP argmax, one launch

Design: 256 CTAs quantize the block's `x` into LDS (`quant_x`, the ASM FP4
GEMM's A operand), then each walks this rank's vocab shard in 16-row groups
`bid, bid + 256, ...` via a dynamic `while` loop (`run_head`, `AHEAD = 2`
groups' `fp4_gemv_loads` kept in flight, as `run_ug`'s union tasks): a CTA
whose first group is already past the shard just never enters the loop, so
padding rows (>= `valid`) and a short last group both fall out of a plain
bounds check, not a `-inf` mask. Each group's 16 logits fold into a running
(max, global row) a token via a comparison join (`_better`: strictly greater
wins, ties go to the lower index). A 4/2/1-lane xor-shuffle butterfly
(`stage_cta`) then reduces the 8 row-pair threads of a token to one (max,
local row) a CTA, published to a scratch mailbox. CTA 0, wave t = token t:
the 64 lanes each poll 4 of the 256 CTAs' winners (one batched poll a lane),
a 64-lane xor-shuffle butterfly (32 .. 1) folds those to this rank's (max,
local row); lane 0 adds `vocab_start`, exchanges with every TP rank's peer
region (double-buffered by epoch parity, as K2's `peer`), reduces again, and
stores `out[t]` as int64 (two dwords, the inverse of `load_ptr64`).

Every mailbox access goes through `c["put_words"]` / `c["poll"]` (bound
methods stored once, not called on the `Mailbox` object directly), and every
loop whose body is a kernel op (not just building a Python list) uses
`range_constexpr`: the AST rewriter can turn a bare `for` / method call
inside a dynamic `if` into a dynamic loop over a Python list, which breaks.

## LDS / VGPR / time

LDS: `red` is `WAVES * 64 * 4` f32 = 8 KiB; `xl` (quantized `x`) is up to
`S * (2560 + 8) / 2` i32 ~= 40 KiB at S = 8; ~48 KiB total, under a CU's
160 KiB. VGPRs: `best_val` / `best_idx` (plus `g`, `ops`) are `run_head`'s
while-loop-carried state, atop whatever `fp4_gemv` holds for
`in_proj_qkvz`'s K = 2560 (same helpers) times `AHEAD` buffers in flight;
register pressure at `AHEAD = 2` is unverified without a real compile (try
`AHEAD = 3` if occupancy allows). Weight + scales read:
`V_R * (H / 2 + H / 32)` = 124160 * 1360 ~= 169 MB a rank, bandwidth-bound
(no reuse across CTAs); MI355X HBM floors this around 20-25 us, under the
40 us target with headroom for CTA 0's poll and the two-rank exchange.

## Open questions

- `stage_token_rank`'s 64-lane poll (4 specs a lane, one round trip) replaces
  the old single-thread 256-entry sequential poll; still unverified whether
  CTA 0's total (poll + two butterflies + rank exchange) fits comfortably
  under the GEMV's ~20-25 us on real hardware.
- `124160 / 16 = 7760 = 30 * 256 + 80` row groups: CTAs 80 .. 255 run one
  fewer loop iteration than CTAs 0 .. 79 in the realistic shard too (not
  only the small-shard test) -- both tests exercise the uneven-last-group
  path, but neither asserts the *iteration count* differs, only the result.
