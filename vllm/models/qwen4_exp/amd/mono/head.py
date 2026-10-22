# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``draft_head``: the Qwen3.8-Flash-Next MTP draft's lm_head, one persistent
launch a draft step, ``BLOCKS`` x ``THREADS``, S <= 8 rows:

    gemv     (every CTA, row groups 16 g .. of this rank's vocab shard, g =
             bid, bid + 256, ... < ceil(rows / 16), ``key.ahead`` groups'
             weights kept in flight, as ``run_ug``'s union tasks): an
             online-MXFP4 GEMV (``quant_x`` first) of the shard's rows
             against the block's x, bf16-rounded, folded into a running
             (max, global row) a token (padding rows -- >= ``valid`` --
             never win; a non-live group never runs at all, the dynamic
             loop just exits). ``key.one_barrier``: the running MFMA tile
             is reduced through a double-buffered LDS scratch (alternating
             by iteration parity), one ``gpu.barrier()`` a group instead of
             two -- the next iteration's store (the other half) can only
             happen once every thread has passed this iteration's barrier,
             and every thread's own read of the current half precedes its
             own next store in program order, so that barrier already
             implies the current half's reads are done. ``key.reduce``: with
             ``one_barrier``, every group's fold is only ever consumed from
             wave 0's lanes (RP * tokens <= 64), so ``'w0'`` has only wave 0
             do the row_sum + fold (the other 7 waves skip straight to the
             next group's GEMV) instead of every wave redoing it (``'all'``,
             the default).
    cta      (every CTA): the running best reduced to one a token (a
             4/2/1-lane xor-shuffle window: the 8 row-pair threads of a
             token) -> every CTA's (value, local row) to a scratch mailbox
    rank     (CTA 0, a token a wave): the 64 lanes each poll 4 of the 256
             CTAs' winners (one batched poll a lane), a 64-lane butterfly to
             this rank's (value, local row); + ``vocab_start``; lane 0 puts
             to every rank's peer region, polls back in rank order, reduces
             to the step's global argmax -> ``out``

Every rank runs the same reduce in the same order (CTA order, then rank
order), so every rank's ``out`` agrees. Ties break to the lowest global index
(``torch.argmax``'s); a NaN logit never wins, checked by its bit pattern (an
ordered compare is not relied on) -- this diverges from ``torch.argmax``,
whose own NaN behavior is undefined / backend-dependent.

The shard's weight / scales hold the ASM FP4 GEMM's layout (``shuffle_weight
(layout=(16, 16))`` / its e8m0 swizzle, what ``Mxfp4OnlineLinearMethod`` +
``AiterMxfp4LinearKernel(use_asm_gemm=True)`` produce on a ``ParallelLMHead``;
the mono kernels already read this layout for ``in_proj_qkvz`` / ``out_proj``
-- ``fp4_gemv_loads`` / ``fp4_gemv``), so ``rows`` (this rank's shard, padding
included) must be a multiple of 32: the scale swizzle's block.

``scratch`` need not be zeroed between launches: the mailbox tag is the
step's epoch, so a stale pair from an earlier launch never matches the
current poll, and every (CTA, token) slot is overwritten by this launch
before anything polls it.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64

from vllm.models.kimi_k3.amd.mono.common.ops import (
    bf16_round,
    ld_i32,
    load_ptr64,
    row_sum,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from vllm.models.kimi_k3.amd.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
)
from vllm.models.kimi_k3.amd.mono.common.ranks import peer_bases
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, preg, shift, sreg
from vllm.models.kimi_k3.amd.mono.stages import gemv
from vllm.models.qwen3_5.amd.mono import layout as QL
from vllm.models.qwen3_5.amd.mono.moe import (
    fp4_gemv,
    fp4_gemv_loads,
    fp4_mfmas,
    fp4_tile,
    mfma_f4,
    pow2,
    quant_x,
    quant_x_fp4,
    scale_word,
    step_tag,
    xq_row,
    xq_scales,
)
from vllm.models.qwen4_exp.amd.mono import layout as L
from vllm.models.qwen4_exp.amd.mono.layout import need
from vllm.models.qwen4_exp.amd.mono.sources import digest

H = L.HIDDEN  # 2560
ROWS = L.ROWS  # 16: a weight row group
RP = ROWS // 2  # 8: the fold's row-pair threads a token
XROW = H + gemv.LDS_PAD
NEG_INF = -(2.0**127)  # a logit that never wins
RED_BUF = WAVES * 64 * 4  # words a (one-barrier) reduction buffer

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class HeadBuild:
    tokens: int
    tp: int = 2
    rows: int = 124160  # this rank's vocab shard, padding included: a multiple of 32
    ahead: int = 3  # row groups' weights kept in flight (swept 2..5, see NOTES)
    one_barrier: bool = True  # double-buffered reduction: 1 barrier a group
    reduce: str = "w0"  # "w0": only wave 0 does the row_sum+fold a group, the
    # rest skip straight to the next group's GEMV (RP * tokens <= 64 = one
    # wave's lanes, so the other 7 waves' fold was always discarded, see
    # WAVESCOPE_FINDINGS H1); confirmed -9..-13 us vs "all" (every wave
    # redoes it) at every S in 1,2,4,8, graph-replayed, 2 runs (see NOTES).
    # "w0" needs one_barrier=True.
    # ---- WAVESCOPE_FINDINGS H2 (the GEMV body): any of these three forces
    # the one_barrier=True, reduce="w0" loop (`_run_head_1barrier_w0_rot` /
    # `_run_head_1barrier_w0_unroll`); each is independently selectable, so
    # A/B/combined builds share a cache/name per the (fast_addr, unroll,
    # mfma) tuple. Graph-replayed, 3 runs (2 alternating the cases' order),
    # S in 1,2,4,8 (see NOTES.md 2026-10-22): `mfma` alone is ~5.1-5.2 us
    # better at S=1/2 and ~2.7-2.9 us better at S=4/8 than all-False; `unroll`
    # alone *regresses* at every S (worse than `fast_addr` alone always,
    # worse than all-False at S=4/8) -- refuted as a standalone change, kept
    # only as a build option. Combined with `mfma`, though, `unroll` helps:
    # fast_addr+unroll+mfma beat mfma-alone by 0.5-1.9 us at every S, every
    # run. All three default True (this combination): it beats all-False by
    # 5.8-5.9 us at S=1/2 and 3.9-4.3 us at S=4/8, passes the existing
    # tolerance tests unmodified (`test_draft_head_mfma_matches_vllm_ops_
    # full_vocab`) and 100% argmax-agrees with all-False over 2640 random
    # trials across both shard scales (`../mono-draft/mfma_agreement.py`).
    # All-False still traces the exact, unmodified `_run_head_1barrier_w0`
    # (reachable by passing all three explicitly False) -- zero risk to that
    # path, now a secondary one.
    fast_addr: bool = True  # unsigned shift/mask address math (every index
    # here is non-negative: a row-group index, a lane-derived column/K-step)
    # instead of Python `//` `%` on signed Int32 (`arith.floordivsi`/`remsi`,
    # ~7 instructions each), the per-lane terms hoisted out of the per-group
    # call. Bit-identical.
    unroll: bool = True  # `ahead`-wide modulo variable expansion: `ahead`
    # groups a while-iteration at fixed operand slots (refreshed in place, as
    # `moe.py`'s `stage_down`/`down_loads` already do for their own DB-wide
    # batches), instead of `ahead` rotating-list iterations (`ops = ops[1:] +
    # [ops_n]`, real `v_mov_b64`s per WAVESCOPE_FINDINGS, not free phi-node
    # renaming). A CTA's groups past its last whole `ahead`-chunk are
    # over-processed and masked the same way any not-yet-reached prefetch
    # already is. Bit-identical. Regresses alone (see above); default True
    # only because it helps once `mfma` is also on.
    mfma: bool = True  # the GEMV applies gfx950's native scaled FP4 MFMA
    # (`v_mfma_scale_f32_16x16x128_f8f6f4`, as K2's routed-expert GEMV
    # already does) to x's fp4 codes + e8m0 scale in LDS (`quant_x_fp4`,
    # instead of `quant_x`'s MXFP4-exact bf16) and the already-loaded fp4
    # weight tiles, instead of cvt-to-bf16 + bf16 MFMA: 3/4 fewer x LDS reads
    # and no `v_cvt_scalef32_pk_bf16_fp4` / bf16 MFMA. A different
    # accumulation order -- NOT bit-identical, see `test_draft_head_mfma_*`.


ABI = KernelAbi(
    (
        "x",
        "w",
        "ws",
        "valid",
        "vocab_start",
        "out",
        "scratch",
        "peers",
        "rank",
        "epoch",
        "slot",
    )
)


# ---------------------------------------------------------------- layout
def peer_pairs(s, tp):
    """One parity's peer regions, pairs: name -> count."""
    return {"head": tp * s * 2}  # 2 pairs a (rank, token): value, global row


def peer_layout(s, tp):
    return pair_layout(peer_pairs(s, tp).items())


def peer_bytes(tp):
    lay = peer_layout(L.MAX_TOKENS, tp)
    return 2 * max(o + n for o, n in lay.values())


def scratch_layout(key: HeadBuild) -> dict:
    s = key.tokens
    pairs = {"cta": BLOCKS * s * 2}  # 2 pairs a (CTA, token): value, local row
    return pair_layout(pairs.items())


def scratch_bytes(key: HeadBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


# ---------------------------------------------------------------- helpers
def f32_of(w):
    return w.bitcast(fx.Float32)


def _safe(v):
    """``v``, or the sentinel if it is NaN (checked by its bit pattern: a NaN
    is an all-ones exponent with a nonzero mantissa, so its magnitude bits
    exceed +inf's 0x7F800000): a NaN logit never wins."""
    bits = v.bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)
    return (bits <= fx.Int32(0x7F800000)).select(v, fx.Float32(NEG_INF))


def _better(val, idx, best_val, best_idx):
    """(val, idx) over (best_val, best_idx): the strictly greater value wins;
    equal values go to the lower index (``torch.argmax``'s tie-break). A
    commutative, associative, idempotent join: safe to fold in any order."""
    win = (val > best_val) | ((val == best_val) & (idx < best_idx))
    return win.select(val, best_val), win.select(idx, best_idx)


def _group_ops(c, gi):
    """Row group ``gi``'s online-MXFP4 operands for this wave."""
    a = c["args"]
    return fp4_gemv_loads(c["lane"], c["wave"], a["w"], a["ws"], gi, H)


# ---- WAVESCOPE_FINDINGS H2: fast_addr / unroll / mfma (see HeadBuild) ----
_N_STEPS = H // L.FP4_STEP  # 20: K steps of an online-MXFP4 GEMV row group
_PER_WAVE = (_N_STEPS + WAVES - 1) // WAVES  # 3: a wave's steps (head's K
# never pairs two waves a step, see ``moe._paired``)
_PAD_STEP = _PER_WAVE * WAVES != _N_STEPS  # some waves' last step is past K
_SCALE_COLS = QL.scale_cols(H)  # a host constant: the shuffled scale's cols


def _scale_index_fast(row, col):
    """``QL.scale_index(row, col, _SCALE_COLS)``, pure unsigned shift/mask:
    ``row`` (a row-group index) and ``col`` (a lane/K-step-derived column)
    are always non-negative here, so the power-of-two floor-div/mod it uses
    (``row // 32``, ``(row // 16) % 2``, ... ``arith.floordivsi``/``remsi``,
    signed, ~7 instructions each) can become a plain shift/and."""
    cols = _SCALE_COLS
    r32, a, b = row >> 5, (row >> 4) & 1, row & 15
    c8, d, f = col >> 3, (col >> 2) & 1, col & 3
    return ((((r32 * (cols // 8) + c8) * 4 + f) * 16 + b) * 2 + d) * 2 + a


def _scale_code_shift_fast(st, rg):
    """``QL.scale_code_shift``, pure unsigned mask (``st``, ``rg``: a K-step
    and row-group index, always non-negative)."""
    return (st & 1) * 16 + (rg & 1) * 8


def _group_ops_opt(c, gi, fast_addr):
    """``_group_ops``, optionally (``fast_addr``) ``fp4_gemv_loads``'s
    address math as unsigned shift/mask with the per-lane terms (``lane %
    ROWS``, ``lane // 16``) hoisted to ``c["lane_row"]`` / ``c["lane_hi4"]``
    -- only ``gi`` varies call to call, so these are otherwise recomputed
    every one of the ``ahead + (loop iterations)`` calls for nothing. A pure
    address change: the same dwords, same order, bit-identical to
    ``_group_ops`` either way."""
    a = c["args"]
    if not fast_addr:
        return fp4_gemv_loads(c["lane"], c["wave"], a["w"], a["ws"], gi, H)
    lane, wave = c["lane"], c["wave"]
    r_w, r_s = rsrc(a["w"]), rsrc(a["ws"])
    row = gi * ROWS + c["lane_row"]
    sts = [fx.min(wave + WAVES * i, _N_STEPS - 1) for i in range(_PER_WAVE)]
    tiles = [fp4_tile(r_w, QL.fp4_tile_dword(0, gi, H, st, lane)) for st in sts]
    words = [
        scale_word(r_s, _scale_index_fast(row, 4 * st + c["lane_hi4"])) for st in sts
    ]
    return tiles + words


def _apply_group(lane, wave, xl, s, rg, ops, fast_addr, mfma):
    """This wave's fp32 partial accumulator for row group ``rg``'s online-
    MXFP4 GEMV (``ops``: ``_group_ops_opt(.., fast_addr)``) against the
    ``s`` token rows in LDS. ``mfma``: gfx950's native scaled FP4 MFMA
    against x's fp4 codes + e8m0 scale (``quant_x_fp4``'s layout) and the
    already-loaded fp4 weight tiles -- no cvt, a different accumulation
    order than the ``fp4_mfmas`` cvt-to-bf16 tree (NOT bit-identical, see
    ``HeadBuild.mfma``). ``fast_addr``: unsigned shift/mask address math
    (every index here is non-negative) instead of Python ``//`` ``%`` on
    signed Int32 -- a pure address change either way (bit-identical)."""
    if fast_addr:
        t, hi4 = fx.min(lane & 15, s - 1), lane >> 4
    else:
        t, hi4 = fx.min(lane % 16, s - 1), lane // 16
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    if mfma:
        for i in range_constexpr(_PER_WAVE):
            raw = wave + WAVES * i
            st = fx.min(raw, _N_STEPS - 1)
            shift = (
                _scale_code_shift_fast(st, rg)
                if fast_addr
                else QL.scale_code_shift(st, rg)
            )
            code = (ops[_PER_WAVE + i] >> shift) & 0xFF
            tile = ops[i]
            if const_expr(_PAD_STEP):
                live = raw < _N_STEPS
                tile = fx.Vector.from_elements(
                    [live.select(tile[q], fx.Int32(0)) for q in range(4)],
                    fx.Int32,
                )
            xq = (t * xq_row(H) + st * (L.FP4_STEP // 2)) // 4 + 4 * hi4
            xb = fx.Vector(
                fx.ptr_load(xl + xq, result_type=fx.Vector.make_type(4, fx.Int32))
            )
            sb = fx.ptr_load(xl + (xq_scales(s, H) + t * (H // 32) + 4 * st + hi4))
            acc = mfma_f4(tile, xb, acc, code, sb)
        return acc
    for i in range_constexpr(_PER_WAVE):
        raw = wave + WAVES * i
        st = fx.min(raw, _N_STEPS - 1)
        shift = (
            _scale_code_shift_fast(st, rg) if fast_addr else QL.scale_code_shift(st, rg)
        )
        code = (ops[_PER_WAVE + i] >> shift) & 0xFF
        scale = pow2(code)
        if const_expr(_PAD_STEP):
            scale = (raw < _N_STEPS).select(scale, fx.Float32(0.0))
        xoff = t * XROW + st * L.FP4_STEP + 32 * hi4
        acc = fp4_mfmas(ops[i], scale, xl, xoff, acc)
    return acc


def _store_i64(ptr, i, v):
    """``ptr`` (an int64 array) element ``i`` := a non-negative Int32 ``v``
    (low dword ``v``, high dword 0): ``load_ptr64``'s word layout, inverted."""
    bo.buffer_store(
        fx.Vector.from_elements([v, fx.Int32(0)], fx.Int32), rsrc(ptr), i * 2
    )


# ---------------------------------------------------------------- stages
def run_head(c, groups, ahead, one_barrier, reduce, fast_addr, unroll, mfma):
    """Dispatches to the one- or two-``gpu.barrier()``-a-group loop (a
    build-time choice, ``one_barrier``) and, within the one-barrier loop, to
    the all-waves or wave-0-only reduce (``reduce``): all walk this rank's
    vocab shard, 16-row groups ``bid, bid + 256, ...`` (a dynamic loop: a CTA
    whose first group is already past ``groups`` never runs its body, as a
    non-live round used to be masked), ``ahead`` groups' online-MXFP4
    operands kept in flight (``run_ug``'s union-task pipelining), each
    group's bf16-rounded logits folded into a running (max, global row) a
    token. Returns (best_val, best_idx); the per-row math is identical
    across every (one_barrier, reduce) combination, and across (fast_addr,
    unroll) at fixed mfma (both are pure performance changes). Any of
    (fast_addr, unroll, mfma) forces the one_barrier=True, reduce="w0" loop
    (``HeadBuild`` validates this in ``build()``): the other three combos
    (two-barrier, all-wave reduce) are legacy / dead-end paths this file
    doesn't carry the H2 changes into."""
    if fast_addr or unroll or mfma:
        if unroll:
            return _run_head_1barrier_w0_unroll(c, groups, ahead, fast_addr, mfma)
        return _run_head_1barrier_w0_rot(c, groups, ahead, fast_addr, mfma)
    if one_barrier:
        if reduce == "all":
            return _run_head_1barrier(c, groups, ahead)
        return _run_head_1barrier_w0(c, groups, ahead)
    return _run_head_2barrier(c, groups, ahead)


@traced
def _run_head_2barrier(c, groups, ahead):
    lane, wave, tid, s, red = c["lane"], c["wave"], c["tid"], c["S"], c["red"]
    xl, valid, bid = c["xl"], c["valid"], c["bid"]
    last = groups - 1
    g = bid
    ops = [_group_ops(c, fx.min(g + BLOCKS * j, last)) for j in range(ahead)]
    best_val = fx.Float32(NEG_INF)
    best_idx = fx.Int32(0)
    mine = tid < RP * s
    r = 2 * (tid % RP)
    t = fx.min(tid // RP, s - 1)
    while g < groups:
        ops_n = _group_ops(c, fx.min(g + BLOCKS * ahead, last))
        acc = fp4_gemv(lane, wave, xl, XROW, H, s, g, ops[0])
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        v0 = _safe(bf16_round(row_sum(red, r, t)))
        v1 = _safe(bf16_round(row_sum(red, r + 1, t)))
        idx0 = g * ROWS + r
        idx1 = idx0 + 1
        cv0 = (mine & (idx0 < valid)).select(v0, fx.Float32(NEG_INF))
        cv1 = (mine & (idx1 < valid)).select(v1, fx.Float32(NEG_INF))
        best_val, best_idx = _better(cv0, idx0, best_val, best_idx)
        best_val, best_idx = _better(cv1, idx1, best_val, best_idx)
        gpu.barrier()
        g = g + BLOCKS
        ops = ops[1:] + [ops_n]
    return best_val, best_idx


@traced
def _run_head_1barrier(c, groups, ahead):
    """As ``_run_head_2barrier``, but ``red`` is double-buffered by iteration
    parity (``par``): a group's store lands in the other half from the
    previous group's, so the single barrier right after this group's store
    both publishes it and (for free) proves the previous group's reads of
    its own half are done -- each thread's own read-then-next-store is
    already in program order, and no thread passes the barrier until every
    thread has reached it, i.e. has issued its own store, which it can only
    do after its own prior read of that same half."""
    lane, wave, tid, s, red = c["lane"], c["wave"], c["tid"], c["S"], c["red"]
    xl, valid, bid = c["xl"], c["valid"], c["bid"]
    last = groups - 1
    g = bid
    ops = [_group_ops(c, fx.min(g + BLOCKS * j, last)) for j in range(ahead)]
    best_val = fx.Float32(NEG_INF)
    best_idx = fx.Int32(0)
    mine = tid < RP * s
    r = 2 * (tid % RP)
    t = fx.min(tid // RP, s - 1)
    par = fx.Int32(0)
    while g < groups:
        ops_n = _group_ops(c, fx.min(g + BLOCKS * ahead, last))
        acc = fp4_gemv(lane, wave, xl, XROW, H, s, g, ops[0])
        buf = red + par * RED_BUF
        fx.ptr_store(acc, buf + (wave * 64 + lane) * 4)
        gpu.barrier()
        v0 = _safe(bf16_round(row_sum(buf, r, t)))
        v1 = _safe(bf16_round(row_sum(buf, r + 1, t)))
        idx0 = g * ROWS + r
        idx1 = idx0 + 1
        cv0 = (mine & (idx0 < valid)).select(v0, fx.Float32(NEG_INF))
        cv1 = (mine & (idx1 < valid)).select(v1, fx.Float32(NEG_INF))
        best_val, best_idx = _better(cv0, idx0, best_val, best_idx)
        best_val, best_idx = _better(cv1, idx1, best_val, best_idx)
        g = g + BLOCKS
        ops = ops[1:] + [ops_n]
        par = fx.Int32(1) - par
    return best_val, best_idx


@traced
def _run_head_1barrier_w0(c, groups, ahead):
    """As ``_run_head_1barrier``, but only wave 0 runs the row_sum + fold for
    a group; the other 7 waves' store the same as before, then skip straight
    to the next group's GEMV. ``mine`` (and so every thread whose fold result
    ever reaches ``stage_cta``) is confined to tid < RP * S <= 64, i.e. wave
    0's lanes, so waves 1..7 redoing the identical 8-wave row_sum in
    ``_run_head_1barrier`` only adds LDS read traffic nobody reads back --
    WAVESCOPE_FINDINGS.md H1 found that traffic service-limited (queued
    behind itself across all 8 waves), not latency-limited, so cutting it to
    1/8 should cut wave 0's own read latency too. The double buffering's
    safety argument is unchanged: a thread that never reads ``buf`` has
    nothing to order against its own next store, and wave 0 (the only
    reader) still reads-before-its-own-next-store in program order, with
    every thread (incl. wave 0) passing the barrier only after its own
    store -- the same invariant ``_run_head_1barrier`` relies on, now
    trivially satisfied by the non-reading waves. Per-row math for wave 0 is
    bit-identical to ``_run_head_1barrier``."""
    lane, wave, tid, s, red = c["lane"], c["wave"], c["tid"], c["S"], c["red"]
    xl, valid, bid = c["xl"], c["valid"], c["bid"]
    last = groups - 1
    g = bid
    ops = [_group_ops(c, fx.min(g + BLOCKS * j, last)) for j in range(ahead)]
    best_val = fx.Float32(NEG_INF)
    best_idx = fx.Int32(0)
    mine = tid < RP * s
    r = 2 * (tid % RP)
    t = fx.min(tid // RP, s - 1)
    par = fx.Int32(0)
    while g < groups:
        ops_n = _group_ops(c, fx.min(g + BLOCKS * ahead, last))
        acc = fp4_gemv(lane, wave, xl, XROW, H, s, g, ops[0])
        buf = red + par * RED_BUF
        fx.ptr_store(acc, buf + (wave * 64 + lane) * 4)
        gpu.barrier()
        if wave == 0:
            v0 = _safe(bf16_round(row_sum(buf, r, t)))
            v1 = _safe(bf16_round(row_sum(buf, r + 1, t)))
            idx0 = g * ROWS + r
            idx1 = idx0 + 1
            cv0 = (mine & (idx0 < valid)).select(v0, fx.Float32(NEG_INF))
            cv1 = (mine & (idx1 < valid)).select(v1, fx.Float32(NEG_INF))
            best_val, best_idx = _better(cv0, idx0, best_val, best_idx)
            best_val, best_idx = _better(cv1, idx1, best_val, best_idx)
        g = g + BLOCKS
        ops = ops[1:] + [ops_n]
        par = fx.Int32(1) - par
    return best_val, best_idx


@traced
def _run_head_1barrier_w0_rot(c, groups, ahead, fast_addr, mfma):
    """As ``_run_head_1barrier_w0``, with ``fast_addr`` and/or ``mfma``
    (``HeadBuild``): the same rotating-``ops`` loop structure, the per-group
    load (``_group_ops_opt``) and apply (``_apply_group``) parametrized."""
    lane, wave, tid, s, red = c["lane"], c["wave"], c["tid"], c["S"], c["red"]
    xl, valid, bid = c["xl"], c["valid"], c["bid"]
    last = groups - 1
    g = bid
    ops = [
        _group_ops_opt(c, fx.min(g + BLOCKS * j, last), fast_addr) for j in range(ahead)
    ]
    best_val = fx.Float32(NEG_INF)
    best_idx = fx.Int32(0)
    mine = tid < RP * s
    if const_expr(fast_addr):
        r, t = 2 * (tid & (RP - 1)), fx.min(tid >> 3, s - 1)  # RP == 8
    else:
        r, t = 2 * (tid % RP), fx.min(tid // RP, s - 1)
    par = fx.Int32(0)
    while g < groups:
        ops_n = _group_ops_opt(c, fx.min(g + BLOCKS * ahead, last), fast_addr)
        acc = _apply_group(lane, wave, xl, s, g, ops[0], fast_addr, mfma)
        buf = red + par * RED_BUF
        fx.ptr_store(acc, buf + (wave * 64 + lane) * 4)
        gpu.barrier()
        if wave == 0:
            v0 = _safe(bf16_round(row_sum(buf, r, t)))
            v1 = _safe(bf16_round(row_sum(buf, r + 1, t)))
            idx0 = g * ROWS + r
            idx1 = idx0 + 1
            cv0 = (mine & (idx0 < valid)).select(v0, fx.Float32(NEG_INF))
            cv1 = (mine & (idx1 < valid)).select(v1, fx.Float32(NEG_INF))
            best_val, best_idx = _better(cv0, idx0, best_val, best_idx)
            best_val, best_idx = _better(cv1, idx1, best_val, best_idx)
        g = g + BLOCKS
        ops = ops[1:] + [ops_n]
        par = fx.Int32(1) - par
    return best_val, best_idx


@traced
def _run_head_1barrier_w0_unroll(c, groups, ahead, fast_addr, mfma):
    """As ``_run_head_1barrier_w0_rot``, but with ``ahead``-wide modulo
    variable expansion (``HeadBuild.unroll``, WAVESCOPE_FINDINGS.md H2 #3):
    one while-iteration processes ``ahead`` groups at fixed operand slots --
    slot j always means "this iteration's j-th group", refreshed in place --
    instead of ``ahead`` rotating-list iterations (``ops = ops[1:] +
    [ops_n]``, which the compiler turns into real ``v_mov_b64``s, not free
    phi-node renaming: WAVESCOPE_FINDINGS.md counted 24/group).
    ``moe.py``'s ``stage_down``/``down_loads`` already use this shape for
    their own DB-wide batches, over-processing their last (possibly partial)
    batch the same way: a CTA's groups past its last whole ``ahead``-chunk
    (``groups - bid`` need not be a multiple of ``BLOCKS * ahead``) are
    over-processed and masked exactly as any not-yet-reached prefetch
    already is elsewhere in this file -- the load is a real (clamped-to-
    last-group, so in-bounds) group's weights, safe to read, and the fold is
    masked out by ``idx < valid`` (an over-run index is always >= rows >=
    valid). Bit-identical to ``_run_head_1barrier_w0`` (same per-row math,
    same oracle); not necessarily the same number of barriers -- a CTA whose
    groups aren't a multiple of ``ahead`` issues up to ``ahead - 1`` extra,
    all-masked barrier/reduce rounds."""
    lane, wave, tid, s, red = c["lane"], c["wave"], c["tid"], c["S"], c["red"]
    xl, valid, bid = c["xl"], c["valid"], c["bid"]
    last = groups - 1
    g = bid
    ops = [
        _group_ops_opt(c, fx.min(g + BLOCKS * j, last), fast_addr) for j in range(ahead)
    ]
    best_val = fx.Float32(NEG_INF)
    best_idx = fx.Int32(0)
    mine = tid < RP * s
    if const_expr(fast_addr):
        r, t = 2 * (tid & (RP - 1)), fx.min(tid >> 3, s - 1)  # RP == 8
    else:
        r, t = 2 * (tid % RP), fx.min(tid // RP, s - 1)
    par = fx.Int32(0)
    while g < groups:
        new_ops = [None] * ahead
        for j in range_constexpr(ahead):
            gi = fx.min(g + BLOCKS * j, last)
            acc = _apply_group(lane, wave, xl, s, gi, ops[j], fast_addr, mfma)
            buf = red + par * RED_BUF
            fx.ptr_store(acc, buf + (wave * 64 + lane) * 4)
            gpu.barrier()
            if wave == 0:
                v0 = _safe(bf16_round(row_sum(buf, r, t)))
                v1 = _safe(bf16_round(row_sum(buf, r + 1, t)))
                idx0 = (g + BLOCKS * j) * ROWS + r
                idx1 = idx0 + 1
                cv0 = (mine & (idx0 < valid)).select(v0, fx.Float32(NEG_INF))
                cv1 = (mine & (idx1 < valid)).select(v1, fx.Float32(NEG_INF))
                best_val, best_idx = _better(cv0, idx0, best_val, best_idx)
                best_val, best_idx = _better(cv1, idx1, best_val, best_idx)
            new_ops[j] = _group_ops_opt(
                c, fx.min(g + BLOCKS * (ahead + j), last), fast_addr
            )
            par = fx.Int32(1) - par
        ops = new_ops
        g = g + BLOCKS * ahead
    return best_val, best_idx


@traced
def stage_cta(c):
    """Every CTA: the running best (``c["best"]``) xor-shuffled (4, 2, 1: the
    8 row-pair threads of a token share that window) to one (max, local row);
    one thread a token (wave 0, lane 0, 8, 16, ..) publishes it to the
    scratch mailbox."""
    best_val, best_idx = c["best"]
    for off in (4, 2, 1):
        ov, oi = xshfl(best_val, off), xshfl(best_idx, off)
        best_val, best_idx = _better(ov, oi, best_val, best_idx)
    lane, wave, s, bid = c["lane"], c["wave"], c["S"], c["bid"]
    tt = lane // RP
    if (wave == 0) & ((lane % RP) == 0) & (tt < s):
        slot_i = (bid * s + tt) * 2
        c["put_words"](c["cta"], slot_i, [best_val.bitcast(fx.Int32), best_idx])


@traced
def stage_token_rank(c, t):
    """Token t (wave t of CTA 0): the 64 lanes each poll 4 of the 256 CTAs'
    winners (lane l: CTAs l, l + 64, l + 128, l + 192, one batched poll), a
    64-lane xor-shuffle butterfly (32, 16, 8, 4, 2, 1) to this rank's (max,
    local row); + ``vocab_start``. Lane 0 puts the result to every TP rank's
    peer region, polls its own back in rank order, reduces again, and stores
    ``out[t]`` (int64, two dwords)."""
    lane, s = c["lane"], c["S"]
    specs = [(c["cta"], ((lane + 64 * j) * s + t) * 2, 2) for j in range(4)]
    got = c["poll"](specs)
    rv = fx.Float32(NEG_INF)
    ri = fx.Int32(0)
    for j in range_constexpr(4):
        wv, wi = got[j]
        rv, ri = _better(f32_of(wv), wi, rv, ri)
    for off in (32, 16, 8, 4, 2, 1):
        ov, oi = xshfl(rv, off), xshfl(ri, off)
        rv, ri = _better(ov, oi, rv, ri)
    ri = ri + c["vocab_start"]
    if lane == 0:
        tp = c["TP"]
        for p in range_constexpr(tp):
            c["put_words"](
                c["peer_head"][p], (c["rank"] * s + t) * 2, [rv.bitcast(fx.Int32), ri]
            )
        got2 = c["poll"](
            [
                (c["peer_head_own"], src * s * 2 + t * 2, 2)
                for src in range_constexpr(tp)
            ]
        )
        fv = fx.Float32(NEG_INF)
        fi = fx.Int32(0)
        for j in range_constexpr(tp):
            wv, wi = got2[j]
            fv, fi = _better(f32_of(wv), wi, fv, fi)
        _store_i64(c["args"]["out"], t, fi)


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: HeadBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    tp = key.tp
    ahead = key.ahead
    one_barrier = key.one_barrier
    reduce = key.reduce
    fast_addr = key.fast_addr
    unroll = key.unroll
    mfma = key.mfma
    need(1 <= s <= L.MAX_TOKENS, f"{s} rows, the kernel takes 1..{L.MAX_TOKENS}")
    need(
        key.rows % 32 == 0,
        "rows: the ASM FP4 GEMM's scale shuffle needs a 32-row shard",
    )
    need(ahead >= 1, "ahead: at least this group's own operands must be in flight")
    need(reduce in ("all", "w0"), "reduce: 'all' or 'w0'")
    need(one_barrier or reduce == "all", "reduce='w0' needs one_barrier=True")
    need(
        not (fast_addr or unroll or mfma) or (one_barrier and reduce == "w0"),
        "fast_addr/unroll/mfma: only implemented for one_barrier=True, reduce='w0'",
    )
    L.Layout(tp)  # validates tp against L.TPS
    groups = key.rows // ROWS
    scr = scratch_layout(key)
    play = peer_layout(L.MAX_TOKENS, tp)
    half = peer_bytes(tp) // 2
    xwords = s * (H + gemv.LDS_PAD) // 2
    # one_barrier's double-buffered reduction needs 2 x RED_BUF; the plain
    # 2-barrier loop only ever touches the first half, so sharing one (larger)
    # layout keeps Smem's shape independent of key.one_barrier.
    red_words = (2 if one_barrier else 1) * RED_BUF

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, red_words, 16]
        xl: fx.Array[fx.Int32, xwords, 16]

    keyed = key_tuple(key, digest())
    name = (
        f"flashnext_mono_head_s{s}_t{tp}_a{ahead}"
        f"_b{1 if one_barrier else 2}_r{reduce}"
        f"_fa{int(fast_addr)}_ur{int(unroll)}_mf{int(mfma)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def head(
        x: Int64,
        w: Int64,
        ws: Int64,
        valid: Int32,
        vocab_start: Int32,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        red = lds.red.ptr
        xl = lds.xl.ptr
        mb = Mailbox(step_tag(epoch, slot) - 1)
        # every CTA's peer addresses (as K2's), though only CTA 0 uses them
        par = uniform(ld_i32(epoch, 0)) % 2
        bases = peer_bases(peers, tp)
        own = load_ptr64(peers, rank)
        hoff = play["head"][0]
        peer_head = [
            shift(preg(pb, hoff, "head"), fx.Int64(par) * half) for pb in bases
        ]
        peer_head_own = shift(preg(own, hoff, "head"), fx.Int64(par) * half)
        c = {
            "S": s,
            "TP": tp,
            "tid": tid,
            "bid": bid,
            "lane": lane,
            "wave": wave,
            "xl": xl,
            "red": red,
            "rank": rank,
            "valid": valid,
            "vocab_start": vocab_start,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "cta": sreg(scratch, scr["cta"][0], "cta"),
            "peer_head": peer_head,
            "peer_head_own": peer_head_own,
            "args": {"w": w, "ws": ws, "out": out},
            # hoisted for HeadBuild.fast_addr (unused, harmless otherwise):
            # only the row group varies call to call in the GEMV loop, not
            # these per-lane terms (WAVESCOPE_FINDINGS H2 #1)
            "lane_row": lane & (ROWS - 1),
            "lane_hi4": lane >> 4,
        }

        # ---- every CTA: x into LDS, MXFP4-exact (as the ASM GEMM reads it)
        gemv.rows_to_lds(tid, x, H, s, xl, XROW)
        if const_expr(mfma):
            quant_x_fp4(c, H)
        else:
            quant_x(c, H)

        # ---- the shard's row groups -> one (max, local row) a token a CTA
        c["best"] = run_head(
            c, groups, ahead, one_barrier, reduce, fast_addr, unroll, mfma
        )
        stage_cta(c)

        # ---- CTA 0, a token a wave: every CTA's winner, then every rank's
        if (bid == 0) & (wave < s):
            stage_token_rank(c, wave)

    @flyc.jit
    def launch(
        x: Int64,
        w: Int64,
        ws: Int64,
        valid: Int32,
        vocab_start: Int32,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        head(
            x, w, ws, valid, vocab_start, out, scratch, peers, rank, epoch, slot
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(head, launch)
    _BUILDS[key] = launch
    return launch


def draft_head(
    key: HeadBuild,
    *,
    x,
    w,
    ws,
    valid: int,
    vocab_start: int,
    out,
    scratch,
    peers,
    rank: int,
    epoch,
    slot: int,
):
    """Host launcher: one persistent launch of the draft head's MXFP4 lm_head
    GEMV + greedy argmax over the full vocab and the TP ranks. ``x`` [S, H]
    bf16: the final mixer's output. ``w`` / ``ws``: this rank's lm_head
    shard, ``key.rows`` rows, in the ASM FP4 GEMM's layout
    (``Mxfp4OnlineLinearMethod`` + ``AiterMxfp4LinearKernel(use_asm_gemm=
    True)`` on a vocab-parallel ``ParallelLMHead``). ``valid``: rows of this
    shard that are real vocab (<= ``key.rows``; the rest are padding and
    never win). ``vocab_start``: this rank's first row's global vocab id
    (``rank * key.rows``). ``out`` [S] int64: every rank writes the same
    global argmax token ids. ``scratch``: device mailboxes (>=
    ``scratch_bytes(key)``; never zeroed between launches -- the mailbox tag
    carries the step's epoch). ``peers``: the int64 table of every rank's
    peer buffer (>= ``peer_bytes(key.tp)`` each, as ``PeerBuffer`` gives K2).
    ``epoch`` a device int32 bumped once a step, ``slot`` this launch's tag
    slot."""
    s = key.tokens
    bf = torch.bfloat16
    fp4s = (torch.uint8, torch.float4_e2m1fn_x2)
    need(x.shape == (s, H) and x.dtype == bf, f"x: bf16 [{s}, {H}]")
    need(
        w.shape == (key.rows, H // 2) and w.dtype in fp4s,
        f"w: fp4x2 [{key.rows}, {H // 2}]",
    )
    need(
        ws.numel() == key.rows * H // 32 and ws.element_size() == 1,
        "w scales: e8m0, one a 32-block, shuffled",
    )
    need(0 <= valid <= key.rows, "valid: rows of this shard that are real vocab")
    need(out.shape == (s,) and out.dtype == torch.int64, f"out: int64 [{s}]")
    need(all(t.is_contiguous() for t in (x, w, ws, out)), "a non-contiguous input")
    f = build(key)
    f(
        *ABI.pack(
            {
                "x": x.data_ptr(),
                "w": w.data_ptr(),
                "ws": ws.data_ptr(),
                "valid": valid,
                "vocab_start": vocab_start,
                "out": out.data_ptr(),
                "scratch": scratch.data_ptr(),
                "peers": peers.data_ptr(),
                "rank": rank,
                "epoch": epoch.data_ptr(),
                "slot": slot,
            }
        ),
        stream=torch.cuda.current_stream(),
    )
