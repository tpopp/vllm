# `front.py`: the MTP draft front as one launch
Replaces `pre_fc_norm_embedding`+`fc_embedding`+`pre_fc_norm_hidden`+
`fc_hidden`+the residual add+the separate `k1.hc_front` launch with one
256-CTA kernel. `w_fe`/`w_fh` are every rank's full, gathered `[H, H]` bf16
weights, so F1/F2 cross no rank -- unlike K1/K2, no all-reduce here.

## Phases -> CTAs
- **F1 norms** (CTAs 0..19, `hc.NRED` 128-col tasks): `emb`'s and `prev`'s
  sums of squares, one value a token (`prev`'s over all 4 streams together --
  a single norm, not grouped by stream) -> MSQE/MSQH; `load_rrms_front`
  (every CTA < 160) reduces them to rrms_en/rrms_hn first.
- **F2 fc** (CTAs 0..159, 16-col groups = `hc.UP_TASKS`'s own grouping):
  `w_fe[g]`/`w_fh[g]` loaded first; normalized `en` -> LDS, GEMV -> `fe`; per
  stream, normalized `hn_k` -> LDS (one area, reused 5x), SAME `w_fh`
  registers -> `fh_k`; `bf16(fe+fh_k)` -> global `state` (device scope) +
  `sflag[g]`. CTAs >= 160 skip F1/F2; they prefetch HC-down weights
  (`hc.down_loads`) instead, same as `k1`'s front_only path.
- **HC front**: `k1`'s `front_only` path verbatim -- `hc.stage_msq`,
  `load_rrms`, `stage_down`, `stage_up`, untouched -- gated by a new
  `wait_state` (all 160 `sflag`s) before any of them reads `state`.
  `hc.stage_msq` re-reads the just-written `state` for its own sums of
  squares, as a later layer's front would; F2 hands off none itself, so
  `hc.py` needed no new mailbox region or `load_rrms` variant.

## LDS / VGPR (S = 8, worst case)
`XLds` (one `[S,H]` bf16 row, reused by F2 and F3-down) ~40 KB; `Smem`
(red/rs/rs2/fe/ll) ~15 KB. Well under 160 KB. Registers: 2x16 weight rows
x 2560 K live a CTA (`gemv.loads`, NT) -- K2's router GEMV footprint,
doubled (one set each for `w_fe` and `w_fh`).

## Unsure / worth a second look before the GPU run
- `fe0`/`fe1` (fc_embedding's output, reused across the 4-stream loop) go
  through LDS across barriers rather than staying plain Python locals
  across two `if tid < ...:` blocks -- matches the only proven cross-barrier
  pattern here (`load_rrms`->`rs`->`xn_to_lds`), never trace-confirmed.
- `wait_state`'s unconditional 160-entry poll (vs. each task's own flags)
  is the simple choice allowed; a tighter set costs less if hot.
- `stage_front_msq`'s one `wave_sum` after summing 4 streams in registers
  is the same reduction as 4 separate ones, different rounding -- within
  tolerance.
