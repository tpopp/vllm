# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``draft_head``: the Qwen3.8-Flash-Next MTP draft's lm_head, one persistent
launch a draft step, ``BLOCKS`` x ``THREADS``, S <= 8 rows:

    gemv     (every CTA, row groups 16 g .. of this rank's vocab shard, g =
             bid, bid + 256, ... < ceil(rows / 16), ``AHEAD`` groups' weights
             kept in flight, as ``run_ug``'s union tasks): an online-MXFP4
             GEMV (``quant_x`` first) of the shard's rows against the block's
             x, bf16-rounded, folded into a running (max, global row) a token
             (padding rows -- >= ``valid`` -- never win; a non-live group
             never runs at all, the dynamic loop just exits)
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
    CM_NT,
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
from vllm.models.qwen3_5.amd.mono.moe import fp4_gemv, fp4_gemv_loads, quant_x, step_tag
from vllm.models.qwen4_exp.amd.mono import layout as L
from vllm.models.qwen4_exp.amd.mono import stamps
from vllm.models.qwen4_exp.amd.mono.layout import need
from vllm.models.qwen4_exp.amd.mono.sources import digest

H = L.HIDDEN  # 2560
ROWS = L.ROWS  # 16: a weight row group
RP = ROWS // 2  # 8: the fold's row-pair threads a token
XROW = H + gemv.LDS_PAD
NEG_INF = -(2.0**127)  # a logit that never wins
AHEAD = 2  # row groups' weights kept in flight (raise to 3 if VGPRs allow)

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class HeadBuild:
    tokens: int
    tp: int = 2
    rows: int = 124160  # this rank's vocab shard, padding included: a multiple of 32
    ahead: int = AHEAD  # row groups' weights kept in flight (Stage 0c E0.b sweep)
    cm: int = CM_NT  # weight-tile load cache policy (E0.b/c: cm=0 is device-scope)
    loads_only: bool = False  # E0.d: XOR-sink the GEMV, barriers off (loop floor)
    trace: bool = False  # emit in-kernel timestamps to a `dbg` buffer (stamps.py)


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
        "dbg",  # Stage 0a: stamps.py target, 256 * stamps.NSLOT int32; 0 when off
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
    """Row group ``gi``'s online-MXFP4 operands for this wave. ``c["cm"]``:
    the weight-tile load's cache policy (default ``CM_NT``, as before this
    knob existed; Stage 0c E0.b/E0.c sweep it to test MALL residency)."""
    a = c["args"]
    return fp4_gemv_loads(c["lane"], c["wave"], a["w"], a["ws"], gi, H, cm=c.get("cm", CM_NT))


def _xor_fold(acc, op):
    """``op`` (an ``fp4_gemv_loads`` operand: a dwordx4 ``fx.Vector`` tile, or
    a plain ``fx.Int32`` scale word) XOR-folded into ``acc``: E0.d's
    ``loads_only`` sink, enough to keep the loads from being dead-code-
    eliminated without paying for the real GEMV / row reduce / barriers."""
    if isinstance(op, fx.Vector):
        return acc ^ op[0] ^ op[3]
    return acc ^ op


def _store_i64(ptr, i, v):
    """``ptr`` (an int64 array) element ``i`` := a non-negative Int32 ``v``
    (low dword ``v``, high dword 0): ``load_ptr64``'s word layout, inverted."""
    bo.buffer_store(
        fx.Vector.from_elements([v, fx.Int32(0)], fx.Int32), rsrc(ptr), i * 2
    )


# ---------------------------------------------------------------- stages
@traced
def run_head(c, groups):
    """This rank's vocab shard, 16-row groups ``bid, bid + 256, ...`` (a
    dynamic loop: a CTA whose first group is already past ``groups`` never
    runs its body, as a non-live round used to be masked): ``c["ahead"]``
    groups' online-MXFP4 operands kept in flight (``run_ug``'s union-task
    pipelining; ``AHEAD`` default, Stage 0c E0.b sweeps it), each group's
    bf16-rounded logits folded into a running (max, global row) a token.

    ``c["loads_only"]`` (E0.d): skip the GEMV, row reduce, and the loop's two
    barriers, XOR-folding the loaded tiles into a sink instead -- isolates
    the load pipeline's time from the compute/sync floor, at the cost of a
    meaningless ``(best_val, best_idx)`` (never read back by a caller that
    sets this).

    Returns (best_val, best_idx)."""
    lane, wave, tid, s, red = c["lane"], c["wave"], c["tid"], c["S"], c["red"]
    xl, valid, bid = c["xl"], c["valid"], c["bid"]
    # ``.get`` with the pre-knob defaults: ``back.py``'s fused kernel reuses
    # this function (and ``_group_ops``, above) through its own ``c``, which
    # never sets these keys -- this keeps that caller's behavior unchanged
    # rather than a KeyError.
    ahead, loads_only = c.get("ahead", AHEAD), c.get("loads_only", False)
    last = groups - 1
    g = bid
    ops = [_group_ops(c, fx.min(g + BLOCKS * j, last)) for j in range(ahead)]
    best_val = fx.Float32(NEG_INF)
    best_idx = fx.Int32(0)
    sink = fx.Int32(0)
    mine = tid < RP * s
    r = 2 * (tid % RP)
    t = fx.min(tid // RP, s - 1)
    while g < groups:
        ops_n = _group_ops(c, fx.min(g + BLOCKS * ahead, last))
        if const_expr(loads_only):
            for i in range_constexpr(len(ops[0])):
                sink = _xor_fold(sink, ops[0][i])
        else:
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
    if const_expr(loads_only):
        # keep `sink` live (never dead-code-eliminated) without perturbing
        # the real result: this sentinel value is never hit in practice (a
        # NaN-pattern logit would already have lost any real comparison).
        if (tid == 0) & (sink == fx.Int32(0x7EADBEEF)):
            best_val = fx.Float32(NEG_INF)
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


@traced
def head_body(c, groups):
    """``quant_x`` / ``run_head`` / ``stage_cta`` / ``stage_token_rank``:
    every CTA's row groups -> a (max, local row), then the TP-wide argmax ->
    ``c["args"]["out"]``. Reusable in a fused kernel (``back.py``) whose x
    rows are already MXFP4-exact in ``c["xl"]`` (there loaded by an earlier
    phase's ``hc.bin_to_lds``, not this module's own ``gemv.rows_to_lds``)."""
    quant_x(c, H)
    stamps.stamp(c, 8)  # Stage 0a: after rows_to_lds (kernel entry) + quant_x
    c["best"] = run_head(c, groups)
    stamps.stamp(c, 9)  # Stage 0a: after run_head
    stage_cta(c)
    if (c["bid"] == 0) & (c["wave"] < c["S"]):
        stage_token_rank(c, c["wave"])


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
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        xl: fx.Array[fx.Int32, xwords, 16]

    keyed = key_tuple(key, digest())
    name = f"flashnext_mono_head_s{s}_t{tp}"

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
        dbg: Int64,
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
            "ahead": key.ahead,
            "cm": key.cm,
            "loads_only": key.loads_only,
            "dbg": dbg if key.trace else None,
            "args": {"w": w, "ws": ws, "out": out},
        }

        stamps.stamp(c, 0)  # Stage 0a: kernel entry
        # ---- every CTA: x into LDS, MXFP4-exact (as the ASM GEMM reads it)
        gemv.rows_to_lds(tid, x, H, s, xl, XROW)
        head_body(c, groups)
        stamps.stamp(c, 10)  # Stage 0a: kernel end

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
        dbg: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        head(
            x, w, ws, valid, vocab_start, out, scratch, peers, rank, epoch, slot, dbg
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
    dbg=None,
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
    slot. ``dbg``: Stage 0a's stamp buffer (``stamps.dbg_buffer``), required
    iff ``key.trace``; ignored (and may be omitted) otherwise."""
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
    need(key.trace == (dbg is not None), "dbg: required iff key.trace")
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
                "dbg": 0 if dbg is None else dbg.data_ptr(),
            }
        ),
        stream=torch.cuda.current_stream(),
    )
