# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``draft_head``: the Qwen3.8-Flash-Next MTP draft's lm_head, one persistent
launch a draft step, ``BLOCKS`` x ``THREADS``, S <= 8 rows:

    gemv     (every CTA, row groups 16 g .. of this rank's vocab shard, g =
             bid, bid + 256, ... < ceil(rows / 16); the next group's weights
             in flight while the current one's MFMAs run, as K1's in_proj):
             an online-MXFP4 GEMV (``quant_x`` first) of the shard's rows
             against the block's x, bf16-rounded, folded into a running
             (max, global row) a token (padding rows -- >= ``valid`` -- and a
             non-live last group never win)
    cta      (every CTA): the running best reduced to one a token (a
             wave-0 lane butterfly: the 8 row-pair threads of a token share a
             4/8-lane xor-shuffle window) -> every CTA's (value, local row)
             to a scratch mailbox
    rank     (CTA 0, a token a thread): CTA 0's own mailbox polled whole,
             reduced to this rank's (value, global row = local row +
             ``vocab_start``); put to every rank's peer region, polled back
             in rank order, reduced to the step's global argmax -> ``out``

Every rank runs the same reduce in the same order (CTA order, then rank
order), so every rank's ``out`` agrees. Ties break to the lowest global index
(``torch.argmax``'s); a NaN logit never wins.

The shard's weight / scales hold the ASM FP4 GEMM's layout (``shuffle_weight
(layout=(16, 16))`` / its e8m0 swizzle, what ``Mxfp4OnlineLinearMethod`` +
``AiterMxfp4LinearKernel(use_asm_gemm=True)`` produce on a ``ParallelLMHead``;
the mono kernels already read this layout for ``in_proj_qkvz`` / ``out_proj``
-- ``fp4_gemv_loads`` / ``fp4_gemv``), so ``rows`` (this rank's shard, padding
included) must be a multiple of 32: the scale swizzle's block.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64

from vllm.models.kimi_k3.amd.mono.common.ops import (
    bf16_round,
    ld_i32,
    load_ptr64,
    row_sum,
    rsrc,
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
from vllm.models.qwen4_exp.amd.mono.layout import need
from vllm.models.qwen4_exp.amd.mono.sources import digest

H = L.HIDDEN  # 2560
ROWS = L.ROWS  # 16: a weight row group
RP = ROWS // 2  # 8: the fold's row-pair threads a token
XROW = H + gemv.LDS_PAD
NEG_INF = -(2.0**127)  # a logit that never wins (moe.py's NaN sentinel)

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
    """``v``, or the sentinel if it is NaN: a NaN logit never wins."""
    return (v == v).select(v, fx.Float32(NEG_INF))


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
    rounds = -(-groups // BLOCKS)
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
        c = {
            "S": s,
            "tid": tid,
            "lane": lane,
            "wave": wave,
            "xl": xl,
            "args": {"w": w, "ws": ws},
        }

        # ---- every CTA: x into LDS, MXFP4-exact (as the ASM GEMM reads it)
        gemv.rows_to_lds(tid, x, H, s, xl, XROW)
        quant_x(c, H)

        # ---- the shard's row groups, the next group's weights in flight
        g0 = fx.min(bid, groups - 1)
        ops = _group_ops(c, g0)
        best_val = fx.Float32(NEG_INF)
        best_idx = fx.Int32(0)
        for i in range_constexpr(rounds):
            g = bid + BLOCKS * i
            live = g < groups
            gi = fx.min(g, groups - 1)
            if i + 1 < rounds:
                gi_next = fx.min(bid + BLOCKS * (i + 1), groups - 1)
                ops_next = _group_ops(c, gi_next)
            acc = fp4_gemv(lane, wave, xl, XROW, H, s, gi, ops)
            fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
            gpu.barrier()
            mine = tid < RP * s
            r = 2 * (tid % RP)
            t = fx.min(tid // RP, s - 1)
            v0 = _safe(bf16_round(row_sum(red, r, t)))
            v1 = _safe(bf16_round(row_sum(red, r + 1, t)))
            idx0 = gi * ROWS + r
            idx1 = idx0 + 1
            ok0 = mine & live & (idx0 < valid)
            ok1 = mine & live & (idx1 < valid)
            cv0 = ok0.select(v0, fx.Float32(NEG_INF))
            cv1 = ok1.select(v1, fx.Float32(NEG_INF))
            best_val, best_idx = _better(cv0, idx0, best_val, best_idx)
            best_val, best_idx = _better(cv1, idx1, best_val, best_idx)
            gpu.barrier()
            if i + 1 < rounds:
                ops = ops_next

        # ---- one (max, local row) a token: the 8 row-pair threads' window
        for off in (4, 2, 1):
            ov = xshfl(best_val, off)
            oi = xshfl(best_idx, off)
            best_val, best_idx = _better(ov, oi, best_val, best_idx)

        cta = sreg(scratch, scr["cta"][0], "cta")
        tt = lane // RP
        if (wave == 0) & ((lane % RP) == 0) & (tt < s):
            slot_i = (bid * s + tt) * 2
            mb.put_words(cta, slot_i, [best_val.bitcast(fx.Int32), best_idx])

        # ---- CTA 0, a token a thread: every CTA's winner, then every rank's
        if (bid == 0) & (tid < s):
            t = tid
            specs = [(cta, (b * s + t) * 2, 2) for b in range(BLOCKS)]
            got = mb.poll(specs)
            rv = fx.Float32(NEG_INF)
            ri = fx.Int32(0)
            for wv, wi in got:
                rv, ri = _better(f32_of(wv), wi, rv, ri)
            ri = ri + vocab_start
            par = uniform(ld_i32(epoch, 0)) % 2
            bases = peer_bases(peers, tp)
            own = load_ptr64(peers, rank)
            hoff = play["head"][0]
            peer_head = [
                shift(preg(pb, hoff, "head"), fx.Int64(par) * half) for pb in bases
            ]
            peer_head_own = shift(preg(own, hoff, "head"), fx.Int64(par) * half)
            for p in range(tp):
                mb.put_words(
                    peer_head[p], (rank * s + t) * 2, [rv.bitcast(fx.Int32), ri]
                )
            got2 = mb.poll(
                [(peer_head_own, src * s * 2 + t * 2, 2) for src in range(tp)]
            )
            fv = fx.Float32(NEG_INF)
            fi = fx.Int32(0)
            for wv, wi in got2:
                fv, fi = _better(f32_of(wv), wi, fv, fi)
            _store_i64(out, t, fi)

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
    ``scratch_bytes(key)``). ``peers``: the int64 table of every rank's peer
    buffer (>= ``peer_bytes(key.tp)`` each, as ``PeerBuffer`` gives K2).
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
