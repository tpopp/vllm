# `draft_head`: MXFP4 lm_head GEMV + TP argmax, one launch

Design: 256 CTAs quantize the block's `x` into LDS (`quant_x`, the ASM FP4
GEMM's A operand), then walk this rank's vocab shard in 16-row groups
`bid, bid + 256, ...` (software-pipelined like K1's `in_proj_qkvz`: the next
group's `fp4_gemv_loads` issued before the current group's `fp4_gemv`),
folding each group's 16 logits into a per-thread running (max, global row)
via a comparison join (`_better`: strictly greater wins, ties go to the
lower index), so the state survives the unrolled round loop without a
traced `if`. Padding rows and non-live rounds (the shard needn't be a
multiple of `256 * 16`) are masked to `-inf` before the fold, not skipped by
branching. A wave-0 lane butterfly (xor 4, 2, 1) reduces the 8 row-pair
threads of a token to one (max, local row) a CTA; every CTA publishes that
to a scratch mailbox. CTA 0, one thread a token, polls its own mailbox,
reduces, adds `vocab_start`, exchanges with every TP rank's peer region
(double-buffered by epoch parity, as K2's `peer`), reduces again, and stores
`out[t]` as int64 (two dwords, the inverse of `load_ptr64`).

## LDS / VGPR / time

LDS: `red` is `WAVES * 64 * 4` f32 = 8 KiB; `xl` (quantized `x`) is up to
`S * (2560 + 8) / 2` i32 ~= 40 KiB at S = 8; ~48 KiB total, under a CU's
160 KiB. VGPRs: `best_val` / `best_idx` live across the round loop, atop
whatever `fp4_gemv` holds for `in_proj_qkvz`'s K = 2560 (same helpers); the
real shard unrolls `ceil(124160 / 16 / 256) = 31` rounds (K1 unrolls only
2), a much larger body -- register pressure is unverified without a real
compile. Weight + scales read: `V_R * (H / 2 + H / 32)` = 124160 * 1360
~= 169 MB a rank, bandwidth-bound (no reuse across CTAs); MI355X HBM floors
this around 20-25 us, under the 40 us target with headroom for CTA 0's poll
and the two-rank exchange.

## Open questions

- CTA 0's `256 * S`-entry cross-CTA poll (one batched `Mailbox.poll`, auto
  split into ceil(/12) rounds): unverified whether it's fast enough at S = 8,
  or whether a tree reduction is needed to hit 40 us.
- No test exercises the "last round only some CTAs live" path at the
  realistic 124160-row size (only the small-shard test hits it).
