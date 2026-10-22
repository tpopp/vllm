# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``draft_head``: the Qwen3.8-Flash-Next MTP draft's lm_head, one persistent
launch a draft step, ``BLOCKS`` x ``THREADS``, S <= ``layout.MAX_TOKENS`` rows:

    gemv     (every CTA, row groups 16 g .. of this rank's vocab shard, g =
             bid, bid + 256, ... < ceil(rows / 16), ``AHEAD`` groups' weights
             kept in flight): an online-MXFP4 GEMV (``quant_x_fp4`` first,
             gfx950's native scaled FP4 MFMA) of the shard's rows against the
             block's x, bf16-rounded, folded into a running (max, global row)
             a token (padding rows -- >= ``valid`` -- never win; a non-live
             group never runs at all, the dynamic loop just exits). The
             running MFMA tile is reduced through a double-buffered LDS
             scratch (alternating by iteration parity, one ``gpu.barrier()``
             a group), and only wave 0 does the row_sum + fold for a group
             (the fold only ever reaches RP * tokens <= 64 lanes, i.e. wave
             0's). ``AHEAD``-wide modulo variable expansion: ``AHEAD`` groups
             a while-iteration at fixed operand slots (refreshed in place),
             instead of a rotating-list iteration. Address math is unsigned
             shift/mask (every index here is non-negative), the per-lane
             terms hoisted out of the per-group call.
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
the specialized decode's GDN / o_proj / MoE blocks already read this layout),
so ``rows`` (this rank's shard, padding included) must be a multiple of 32:
the scale swizzle's block. Unlike those blocks, this kernel does not also
support the Triton FP4 GEMM's row-major layout (a drafter-only optimization,
no eligibility path reaches it with the other layout today).

``scratch`` need not be zeroed between launches: the mailbox tag is the
step's epoch, so a stale pair from an earlier launch never matches the
current poll, and every (CTA, token) slot is overwritten by this launch
before anything polls it.

Adapted from the Qwen3.8-Flash-Next MTP mono draft's head kernel
(``VLLM_ROCM_MONO_DRAFT_HEAD``, pre-split), collapsed to its single shipped
variant (the other build knobs it carried for A/B measurement are gone), and
onto Mikko Tukiainen's specialized-decode split (the tagged mailbox's health
/ alarm words, ``mono.stages.moe``'s MXFP4 helpers)."""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64

from vllm.models.kimi_k3.amd.specialized_decode.mono.common.debug import region_ids
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import (
    bf16_round,
    ld_i32,
    load_ptr64,
    row_sum,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ranks import peer_bases
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.sync import (
    Mailbox,
    preg,
    shift,
    sreg,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import gemv
from vllm.models.qwen4_exp.amd.specialized_decode.mono import layout as L
from vllm.models.qwen4_exp.amd.specialized_decode.mono import mx_layout as QL
from vllm.models.qwen4_exp.amd.specialized_decode.mono.layout import need
from vllm.models.qwen4_exp.amd.specialized_decode.mono.sources import digest
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages.moe import (
    fp4_tile,
    mfma_f4,
    pow2,
    quant_x_fp4,
    scale_word,
    step_tag,
    xq_row,
    xq_scales,
)

H = L.HIDDEN  # 2560
ROWS = L.ROWS  # 16: a weight row group
RP = ROWS // 2  # 8: the fold's row-pair threads a token
XROW = H + gemv.LDS_PAD
NEG_INF = -(2.0**127)  # a logit that never wins
RED_BUF = WAVES * 64 * 4  # words of a (double-buffered) reduction buffer
AHEAD = 3  # row groups' weights kept in flight
_N_STEPS = H // QL.FP4_STEP  # 20: K steps of an online-MXFP4 GEMV row group
_PER_WAVE = (_N_STEPS + WAVES - 1) // WAVES  # 3: a wave's steps
_PAD_STEP = _PER_WAVE * WAVES != _N_STEPS  # some waves' last step is past K
_SCALE_COLS = QL.scale_cols(H)  # a host constant: the shuffled scale's cols

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class HeadBuild:
    tokens: int
    tp: int = 2
    rows: int = 124160  # this rank's vocab shard, padding included: a multiple of 32


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
        "health",
        "alarm",
        "layer",
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


def _scale_index_fast(row, col):
    """``QL.scale_index(row, col, _SCALE_COLS)``, pure unsigned shift/mask:
    ``row`` (a row-group index) and ``col`` (a lane/K-step-derived column)
    are always non-negative here, so the power-of-two floor-div/mod it uses
    (signed ``arith.floordivsi`` / ``remsi``, ~7 instructions each) becomes a
    plain shift/and."""
    cols = _SCALE_COLS
    r32, a, b = row >> 5, (row >> 4) & 1, row & 15
    c8, d, f = col >> 3, (col >> 2) & 1, col & 3
    return ((((r32 * (cols // 8) + c8) * 4 + f) * 16 + b) * 2 + d) * 2 + a


def _scale_code_shift_fast(st, rg):
    """``QL.scale_code_shift``, pure unsigned mask (``st``, ``rg``: a K-step
    and row-group index, always non-negative)."""
    return (st & 1) * 16 + (rg & 1) * 8


def _group_ops(c, gi):
    """Row group ``gi``'s online-MXFP4 operands for this wave: unsigned
    shift/mask address math, the per-lane terms hoisted to ``c["lane_row"]``
    / ``c["lane_hi4"]`` -- only ``gi`` varies call to call."""
    a = c["args"]
    lane, wave = c["lane"], c["wave"]
    r_w, r_s = rsrc(a["w"]), rsrc(a["ws"])
    row = gi * ROWS + c["lane_row"]
    sts = [fx.min(wave + WAVES * i, _N_STEPS - 1) for i in range(_PER_WAVE)]
    tiles = [fp4_tile(r_w, QL.fp4_tile_dword(0, gi, H, st, lane)) for st in sts]
    words = [
        scale_word(r_s, _scale_index_fast(row, 4 * st + c["lane_hi4"])) for st in sts
    ]
    return tiles + words


def _apply_group(lane, wave, xl, s, rg, ops):
    """This wave's fp32 partial accumulator for row group ``rg``'s online-
    MXFP4 GEMV (``ops``: ``_group_ops``) against the ``s`` token rows in LDS:
    gfx950's native scaled FP4 MFMA against x's fp4 codes + e8m0 scale in LDS
    (``quant_x_fp4``'s layout) and the already-loaded fp4 weight tiles."""
    t, hi4 = fx.min(lane & 15, s - 1), lane >> 4
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(_PER_WAVE):
        raw = wave + WAVES * i
        st = fx.min(raw, _N_STEPS - 1)
        shift_bits = _scale_code_shift_fast(st, rg)
        code = (ops[_PER_WAVE + i] >> shift_bits) & 0xFF
        tile = ops[i]
        if const_expr(_PAD_STEP):
            live = raw < _N_STEPS
            tile = fx.Vector.from_elements(
                [live.select(tile[q], fx.Int32(0)) for q in range(4)],
                fx.Int32,
            )
        xq = (t * xq_row(H) + st * (QL.FP4_STEP // 2)) // 4 + 4 * hi4
        xb = fx.Vector(
            fx.ptr_load(xl + xq, result_type=fx.Vector.make_type(4, fx.Int32))
        )
        sb = fx.ptr_load(xl + (xq_scales(s, H) + t * (H // 32) + 4 * st + hi4))
        acc = mfma_f4(tile, xb, acc, code, sb)
    return acc


# ---------------------------------------------------------------- stages
@traced
def run_head(c, groups):
    """This rank's vocab shard, 16-row groups ``bid, bid + 256, ...`` (a
    dynamic loop: a CTA whose first group is already past ``groups`` never
    runs its body), ``AHEAD`` groups' online-MXFP4 operands kept in flight,
    each group's bf16-rounded logits folded into a running (max, global row)
    a token. ``AHEAD``-wide modulo variable expansion: one while-iteration
    processes ``AHEAD`` groups at fixed operand slots, refreshed in place.
    Returns (best_val, best_idx)."""
    lane, wave, tid, s, red = c["lane"], c["wave"], c["tid"], c["S"], c["red"]
    xl, valid, bid = c["xl"], c["valid"], c["bid"]
    last = groups - 1
    g = bid
    ops = [_group_ops(c, fx.min(g + BLOCKS * j, last)) for j in range(AHEAD)]
    best_val = fx.Float32(NEG_INF)
    best_idx = fx.Int32(0)
    mine = tid < RP * s
    r, t = 2 * (tid & (RP - 1)), fx.min(tid >> 3, s - 1)  # RP == 8
    par = fx.Int32(0)
    while g < groups:
        new_ops = [None] * AHEAD
        for j in range_constexpr(AHEAD):
            gi = fx.min(g + BLOCKS * j, last)
            acc = _apply_group(lane, wave, xl, s, gi, ops[j])
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
            new_ops[j] = _group_ops(c, fx.min(g + BLOCKS * (AHEAD + j), last))
            par = fx.Int32(1) - par
        ops = new_ops
        g = g + BLOCKS * AHEAD
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
    64-lane xor-shuffle butterfly to this rank's (max, local row); +
    ``vocab_start``. Lane 0 puts the result to every TP rank's peer region,
    polls its own back in rank order, reduces again, and stores ``out[t]``
    (int64, two dwords)."""
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


def _store_i64(ptr, i, v):
    """``ptr`` (an int64 array) element ``i`` := a non-negative Int32 ``v``
    (low dword ``v``, high dword 0): ``load_ptr64``'s word layout, inverted."""
    bo.buffer_store(
        fx.Vector.from_elements([v, fx.Int32(0)], fx.Int32), rsrc(ptr), i * 2
    )


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: HeadBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    tp = key.tp
    need(1 <= s <= L.MAX_TOKENS, f"{s} rows, the kernel takes 1..{L.MAX_TOKENS}")
    need(
        key.rows % 32 == 0,
        "rows: the ASM FP4 GEMM's scale shuffle needs a 32-row shard",
    )
    L.Layout(tp)  # validates tp against L.TPS
    groups = key.rows // ROWS
    scr = scratch_layout(key)
    play = peer_layout(L.MAX_TOKENS, tp)
    half = peer_bytes(tp) // 2
    xwords = s * (H + gemv.LDS_PAD) // 2

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, 2 * RED_BUF, 16]
        xl: fx.Array[fx.Int32, xwords, 16]

    keyed = key_tuple(key, digest())
    name = f"flashnext_draft_head_s{s}_t{tp}"

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
        health: Int64,
        alarm: Int64,
        layer: Int32,
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
        mb = Mailbox(
            step_tag(epoch, layer) - 1, health, alarm, region_ids((*scr, "head"))
        )
        # every CTA's peer addresses (as the o_proj / MoE blocks'), though
        # only CTA 0 uses them
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
            # hoisted: only the row group varies call to call in the GEMV
            # loop, not these per-lane terms
            "lane_row": lane & (ROWS - 1),
            "lane_hi4": lane >> 4,
        }

        # ---- every CTA: x into LDS, MXFP4 codes + e8m0 scales (as the
        # scaled MFMA reads them)
        gemv.rows_to_lds(tid, x, H, s, xl, XROW)
        quant_x_fp4(c, H)

        # ---- the shard's row groups -> one (max, local row) a token a CTA
        c["best"] = run_head(c, groups)
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
        health: Int64,
        alarm: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        head(
            x,
            w,
            ws,
            valid,
            vocab_start,
            out,
            scratch,
            peers,
            rank,
            epoch,
            health,
            alarm,
            layer,
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
    health,
    alarm,
    layer: int,
):
    """Host launcher: one persistent launch of the draft head's MXFP4 lm_head
    GEMV + greedy argmax over the full vocab and the TP ranks. ``x`` [S, H]
    bf16: the draft step's final mixer output. ``w`` / ``ws``: this rank's
    lm_head shard, ``key.rows`` rows, in the ASM FP4 GEMM's layout
    (``Mxfp4OnlineLinearMethod`` + ``AiterMxfp4LinearKernel(use_asm_gemm=
    True)`` on a vocab-parallel ``ParallelLMHead``). ``valid``: rows of this
    shard that are real vocab (<= ``key.rows``; the rest are padding and
    never win). ``vocab_start``: this rank's first row's global vocab id
    (``rank * key.rows``). ``out`` [S] int64: every rank writes the same
    global argmax token ids. ``scratch``: device mailboxes (>=
    ``scratch_bytes(key)``; never zeroed between launches -- the mailbox tag
    carries the step's epoch). ``peers``: the int64 table of every rank's
    peer buffer (>= ``peer_bytes(key.tp)`` each). ``epoch`` a device int32
    bumped once a step, ``layer`` this launch's tag slot."""
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
                "health": health.data_ptr(),
                "alarm": alarm,
                "layer": layer,
            }
        ),
        stream=torch.cuda.current_stream(),
    )
