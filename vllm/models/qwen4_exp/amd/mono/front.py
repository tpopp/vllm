# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``draft_front``: the MTP draft step's front as one persistent launch, 256
CTAs, replacing ~10 stock kernels (``pre_fc_norm_embedding``, ``fc_embedding``,
``pre_fc_norm_hidden``, ``fc_hidden``, the residual add, and ``k1.hc_front``'s
HC mix) with a single launch:

    en  = GemmaRMSNorm_H(emb; w_ne, ne_eps)              # [S, H] bf16
    fe  = bf16(en @ W_fe^T)                               # fc_embedding
    hn  = GemmaRMSNorm_4H(prev; w_nh, nh_eps)            # ONE norm over 4 H
    fh_k= bf16(hn[:, kH:(k+1)H] @ W_fh^T)   k = 0 .. 3    # fc_hidden, shared
    state[:, kH:(k+1)H] = bf16(fe + fh_k)
    x, inj = the attention HC front of state              # k1.hc_front's math

Every rank runs this on the SAME full [H, H] bf16 ``w_fe`` / ``w_fh`` (the
stock ``ColumnParallelLinear(gather_output=True)`` weights gathered once at
load, replicated), so F1 / F2 cross no rank -- unlike K1 / K2's in_proj / MoE,
there is no all-reduce here.

    F1 norms  (``NRED`` = 20 column tasks, CTAs 0 .. 19): ``emb``'s and
              ``prev``'s sums of squares (one value a token each, ``prev``'s
              over all 4 streams together) -> MSQE / MSQH, then every F2 CTA
              polls and reduces them to rrms_en / rrms_hn (``load_rrms_front``)
    F2 fc     (160 row groups of 16 output columns, CTAs 0 .. 159, the SAME
              grouping as ``hc.UP_TASKS``): ``w_fe`` / ``w_fh`` rows loaded
              first; normalized en -> LDS, GEMV -> fe; per stream k:
              normalized hn_k -> LDS (same LDS reused), GEMV (the SAME w_fh
              registers) -> fh_k; bf16(fe + fh_k) -> global ``state``
              (device-scope) + a per-group flag (``sflag``)
    HC front  exactly ``k1``'s ``front_only`` path (``hc.stage_msq`` /
              ``load_rrms`` / ``stage_down`` / ``stage_up``, unmodified),
              waiting on every ``sflag`` (``wait_state``) before any of them
              reads ``state`` -- the HC norm's own sums of squares are read
              back from the just-written ``state``, not recomputed in F2

See ``README_front.md`` for the design rationale and LDS budget.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.kimi_k3.amd.mono.common.ops import (
    CM_DEV,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    ld_i32,
    hw_rsq,
    row_sum,
    rsrc,
    traced,
    uniform,
    wave_sum,
)
from vllm.models.kimi_k3.amd.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
)
from vllm.models.kimi_k3.amd.mono.common.sync import Mailbox, sreg
from vllm.models.kimi_k3.amd.mono.stages import gemv
from vllm.models.qwen3_5.amd.mono.moe import step_tag
from vllm.models.qwen4_exp.amd.mono import hc
from vllm.models.qwen4_exp.amd.mono import layout as L
from vllm.models.qwen4_exp.amd.mono import stamps
from vllm.models.qwen4_exp.amd.mono.layout import need
from vllm.models.qwen4_exp.amd.mono.sources import digest

H, HH, HC, ROWS = L.HIDDEN, L.HH, L.HC, L.ROWS
D0 = BLOCKS - hc.DOWN_TASKS  # 88: the hc down CTAs, to the last
XROW = H + gemv.LDS_PAD  # an en / hn_k row in LDS

assert hc.UP_TASKS == H // ROWS, "F2's 160 row groups double as F3's up tasks"

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class FrontBuild:
    tokens: int
    tp: int = 2
    eps: float = 1e-6  # the attention HC norm (k1 / k2's)
    ne_eps: float = 1e-6  # pre_fc_norm_embedding
    nh_eps: float = 1e-6  # pre_fc_norm_hidden (one norm over all 4 H)
    trace: bool = False  # emit in-kernel timestamps to a `dbg` buffer (stamps.py)
    pre_fh: bool = False  # E6.1: fh := prev[.., kH + 16g ..] straight (no fc_hidden)
    emb_ids: bool = False  # E6b.2: `emb` is a replicated vocab table, gather by `ids`


ABI = KernelAbi(
    (
        "emb",
        "prev",
        "w_ne",
        "w_nh",
        "w_fe",
        "w_fh",
        "hc_w",
        "w_down",
        "w_up",
        "state_out",
        "bin_out",
        "inj_out",
        "scratch",
        "epoch",
        "slot",
        "dbg",  # Stage 0a: stamps.py target, 256 * stamps.NSLOT int32; 0 when off
        "ids",  # E6b.2 (emb_ids): int64 [S], back's picked token ids; 0 when off
    )
)


def f32_of(w):
    return w.bitcast(fx.Float32)


def scratch_layout(key: FrontBuild) -> dict:
    s = key.tokens
    pairs = {
        "msqe": hc.NRED * s,  # emb's sums of squares, one a (task, token)
        "msqh": hc.NRED * s,  # prev's, over all 4 streams together
        "sflag": hc.UP_TASKS,  # F2's "state written" flags
    } | hc.regions(s)
    return pair_layout(pairs.items())


def scratch_bytes(key: FrontBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


# ---------------------------------------------------------------- F1: norms
@traced
def stage_front_msq(c, j):
    """Columns 128 j .. (``hc.COLS`` wide): ``emb``'s sums of squares (one
    value a token) -> MSQE(j, t); ``prev``'s over all 4 streams together (the
    single ``pre_fc_norm_hidden`` norm, not one a stream) -> MSQH(j, t). Wave
    t token t.

    ``c["pre_fh"]`` (E6.1): ``fc_hidden`` never runs, so ``prev``'s sums of
    squares (MSQH) are never consumed -- skip computing and publishing them
    (``load_rrms_front`` skips polling MSQH to match, or this would hang
    waiting for a put that never comes)."""
    s, tid, lane = c["S"], c["tid"], c["lane"]
    a = c["args"]
    mine = tid < s * 64
    t = fx.min(tid // 64, s - 1)
    col = j * hc.COLS + 2 * (tid % 64)
    if const_expr(c["emb_ids"]):
        row = uniform(fx.ptr_load(c["ids_lds"] + t))
    else:
        row = t
    we = fx.Int32(
        bo.buffer_load(rsrc(a["emb"]), (row * H + col) // 2, vec_width=1, dtype=T.i32)
    )
    e0, e1 = bf_lo(we), bf_hi(we)
    en_sq = wave_sum(e0 * e0 + e1 * e1)
    if const_expr(not c["pre_fh"]):
        hn_sq = fx.Float32(0.0)
        for k in range_constexpr(HC):
            wp = fx.Int32(
                bo.buffer_load(
                    rsrc(a["prev"]), (t * HH + k * H + col) // 2, vec_width=1, dtype=T.i32
                )
            )
            p0, p1 = bf_lo(wp), bf_hi(wp)
            hn_sq = hn_sq + p0 * p0 + p1 * p1
        hn_sq = wave_sum(hn_sq)
    if mine & (lane == 0):
        c["put"](c["msqe"], j * s + t, en_sq)
        if const_expr(not c["pre_fh"]):
            c["put"](c["msqh"], j * s + t, hn_sq)


@traced
def load_rrms_front(c):
    """Every F1 task's sums of squares -> rrms_en, rrms_hn (t) = rsqrt(mean +
    eps) -> LDS ``rs2`` [t, 2]. Wave t token t, lane j task j's.

    ``c["pre_fh"]`` (E6.1): rrms_hn is never used (``fc_hidden`` never
    runs), and MSQH was never published (``stage_front_msq``) -- poll only
    MSQE, and store a dummy 0 for ``rs2[t, 1]``."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    if wave < s:
        j = fx.min(lane, hc.NRED - 1)
        at = j * s + wave
        live = lane < hc.NRED
        if const_expr(c["pre_fh"]):
            got = c["poll"]([(c["msqe"], at, 1)])
            en_v = live.select(f32_of(got[0][0]), fx.Float32(0.0))
            r_hn = fx.Float32(0.0)
        else:
            got = c["poll"]([(c["msqe"], at, 1), (c["msqh"], at, 1)])
            en_v = live.select(f32_of(got[0][0]), fx.Float32(0.0))
            hn_v = live.select(f32_of(got[1][0]), fx.Float32(0.0))
            r_hn = hw_rsq(
                wave_sum(hn_v) * fx.Float32(1.0 / HH) + fx.Float32(c["nh_eps"])
            )
        r_en = hw_rsq(wave_sum(en_v) * fx.Float32(1.0 / H) + fx.Float32(c["ne_eps"]))
        if lane == 0:
            fx.ptr_store(r_en, c["rs2"] + wave * 2)
            fx.ptr_store(r_hn, c["rs2"] + wave * 2 + 1)
    gpu.barrier()


@traced
def _norm_to_lds(c, src, ldk, off, w, w_off, r_idx):
    """``H`` columns of token rows (bf16, row stride ``ldk``, starting at
    column ``off``) through the Gemma norm (rrms ``rs2[t, r_idx]``, weight
    ``w`` from column ``w_off``) -> LDS ``xl`` (``XROW`` a row)."""
    s, tid = c["S"], c["tid"]
    words = s * H // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (H // 8)
        kk = e % (H // 8) * 8
        sv = fx.Vector(
            bo.buffer_load(
                rsrc(src), (t * ldk + off + kk) // 2, vec_width=4, dtype=T.i32
            )
        )
        wv = fx.Vector(
            bo.buffer_load(rsrc(w), (w_off + kk) // 2, vec_width=4, dtype=T.i32)
        )
        r = fx.ptr_load(c["rs2"] + t * 2 + r_idx)
        ys = [
            bf16_pair(
                hc.gemma(bf_lo(sv[d]), r, bf_lo(wv[d])),
                hc.gemma(bf_hi(sv[d]), r, bf_hi(wv[d])),
            ).bitcast(fx.Int32)
            for d in range(4)
        ]
        if tid + THREADS * i < words:
            fx.ptr_store(
                fx.Vector.from_elements(ys, fx.Int32), c["xl"] + (t * XROW + kk) // 2
            )
    gpu.barrier()


@traced
def _en_to_lds_ids(c):
    """``en_to_lds``'s math (E6b.2, ``emb_ids``): token t's embedding row is
    ``emb_table[ids[t]]``, not ``emb[t]`` -- the row is read back from
    ``c["ids_lds"]`` (loaded once at kernel entry) instead of being ``t``
    itself. Otherwise identical to ``_norm_to_lds(c, emb, H, 0, w_ne, 0,
    0)``: same Gemma norm, LDS layout, and tid -> (t, kk) traversal."""
    s, tid = c["S"], c["tid"]
    a = c["args"]
    w = a["w_ne"]
    words = s * H // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (H // 8)
        kk = e % (H // 8) * 8
        row = fx.ptr_load(c["ids_lds"] + t)
        sv = fx.Vector(
            bo.buffer_load(rsrc(a["emb"]), (row * H + kk) // 2, vec_width=4, dtype=T.i32)
        )
        wv = fx.Vector(bo.buffer_load(rsrc(w), kk // 2, vec_width=4, dtype=T.i32))
        r = fx.ptr_load(c["rs2"] + t * 2)
        ys = [
            bf16_pair(
                hc.gemma(bf_lo(sv[d]), r, bf_lo(wv[d])),
                hc.gemma(bf_hi(sv[d]), r, bf_hi(wv[d])),
            ).bitcast(fx.Int32)
            for d in range(4)
        ]
        if tid + THREADS * i < words:
            fx.ptr_store(
                fx.Vector.from_elements(ys, fx.Int32), c["xl"] + (t * XROW + kk) // 2
            )
    gpu.barrier()


def en_to_lds(c):
    a = c["args"]
    if c["emb_ids"]:
        _en_to_lds_ids(c)
    else:
        _norm_to_lds(c, a["emb"], H, 0, a["w_ne"], 0, 0)


def hn_to_lds(c, k):
    a = c["args"]
    _norm_to_lds(c, a["prev"], HH, k * H, a["w_nh"], k * H, 1)


# ---------------------------------------------------------------- F2: fc
@traced
def stage_fc(c, g, fe_ops, fh_ops):
    """Row group g (16 output columns): fe = bf16(en @ w_fe[g]^T); for each
    stream k, fh_k = bf16(hn_k @ w_fh[g]^T) (the SAME ``fh_ops`` registers,
    ``w_fh`` is one weight shared across streams) -> state[:, kH + 16g ..] =
    bf16(fe + fh_k) -> global ``state`` + ``sflag[g]``. The HC norm's sums of
    squares are NOT computed here: they're read back from ``state`` by the
    unmodified ``hc.stage_msq``, gated by ``sflag`` (``wait_state``).

    ``c["pre_fh"]`` (E6.1): skip the real ``fc_hidden`` (the 4 ``hn_to_lds`` /
    GEMV passes, and ``fh_ops`` -- never loaded by the caller either) and read
    fh straight out of ``prev`` at ``[t, kH + 16g + 2rp ..]`` instead (two
    bf16 a (row-pair, stream)): a stand-in for ``fc_hidden(norm(prev))``,
    exactly the memory layout a precomputed ``fh`` would need to match for a
    correctness check (random bytes are fine for the perf-only probe)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    en_to_lds(c)
    acc = gemv.mfmas(lane, wave, c["xl"], XROW, H, s, fe_ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        tt = tid // (ROWS // 2)
        fe0 = bf16_round(row_sum(red, 2 * rp, tt))
        fe1 = bf16_round(row_sum(red, 2 * rp + 1, tt))
        fx.ptr_store(fe0, c["fe"] + tt * ROWS + 2 * rp)
        fx.ptr_store(fe1, c["fe"] + tt * ROWS + 2 * rp + 1)
    gpu.barrier()
    stamps.stamp(c, 2)  # Stage 0a: after the fe pass
    if const_expr(c["pre_fh"]):
        if tid < (ROWS // 2) * s:
            rp = tid % (ROWS // 2)
            tt = tid // (ROWS // 2)
            fe0 = fx.ptr_load(c["fe"] + tt * ROWS + 2 * rp)
            fe1 = fx.ptr_load(c["fe"] + tt * ROWS + 2 * rp + 1)
            col = g * ROWS + 2 * rp
            for k in range_constexpr(HC):
                wp = fx.Int32(
                    bo.buffer_load(
                        rsrc(a["prev"]), (tt * HH + k * H + col) // 2,
                        vec_width=1, dtype=T.i32,
                    )
                )  # fmt: skip
                fh0, fh1 = bf_lo(wp), bf_hi(wp)
                v0 = bf16_round(fe0 + fh0)
                v1 = bf16_round(fe1 + fh1)
                bo.buffer_store(
                    bf16_pair(v0, v1).bitcast(fx.Int32),
                    rsrc(c["hc"]["state"]),
                    (tt * HH + k * H + g * ROWS + 2 * rp) // 2,
                    cache_modifier=CM_DEV,
                )
        gpu.barrier()
        stamps.stamp(c, 3)  # Stage 0a: after the (single, combined) hn pass
    else:
        for k in range_constexpr(HC):
            hn_to_lds(c, k)
            acc2 = gemv.mfmas(lane, wave, c["xl"], XROW, H, s, fh_ops)
            fx.ptr_store(acc2, red + (wave * 64 + lane) * 4)
            gpu.barrier()
            if tid < (ROWS // 2) * s:
                rp = tid % (ROWS // 2)
                tt = tid // (ROWS // 2)
                fe0 = fx.ptr_load(c["fe"] + tt * ROWS + 2 * rp)
                fe1 = fx.ptr_load(c["fe"] + tt * ROWS + 2 * rp + 1)
                fh0 = bf16_round(row_sum(red, 2 * rp, tt))
                fh1 = bf16_round(row_sum(red, 2 * rp + 1, tt))
                v0 = bf16_round(fe0 + fh0)
                v1 = bf16_round(fe1 + fh1)
                bo.buffer_store(
                    bf16_pair(v0, v1).bitcast(fx.Int32),
                    rsrc(c["hc"]["state"]),
                    (tt * HH + k * H + g * ROWS + 2 * rp) // 2,
                    cache_modifier=CM_DEV,
                )
            gpu.barrier()
            stamps.stamp(c, 3 + k)  # Stage 0a: after hn pass k (3, 4, 5, 6)
    rocdl.s_waitcnt(vmcnt=0)  # drain this CTA's state stores
    gpu.barrier()
    if tid == 0:
        c["put"](c["sflag"], g, fx.Int32(1))
    stamps.stamp(c, 7)  # Stage 0a: after the sflag put
    gpu.barrier()


# ---------------------------------------------------------------- F3
@traced
def wait_state(c):
    """Every F2 group's ``sflag``: ``state``'s stores are visible after."""
    if c["tid"] < hc.UP_TASKS:
        c["poll"]([(c["sflag"], c["tid"], 1)])
    gpu.barrier()
    stamps.stamp(c, 8)  # Stage 0a: after wait_state


def _put_inj(c, t, r, vs):
    bo.buffer_store(
        bf16_pair(vs[0], vs[1]).bitcast(fx.Int32),
        rsrc(c["args"]["inj_out"]),
        (t * HC + r) // 2,
    )


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: FrontBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    need(1 <= s <= L.MAX_TOKENS, f"{s} rows, the kernels take 1..{L.MAX_TOKENS}")
    need(key.tp in L.TPS, f"tensor parallel size {key.tp}, built for {L.TPS}")
    scr = scratch_layout(key)
    xwords = s * (max(H, hc.DK) + gemv.LDS_PAD) // 2

    @fx.struct
    class XLds:
        xl: fx.Array[fx.Int32, xwords, 16]

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        rs: fx.Array[fx.Float32, L.MAX_TOKENS * HC, 16]
        rs2: fx.Array[fx.Float32, L.MAX_TOKENS * 2, 16]
        fe: fx.Array[fx.Float32, L.MAX_TOKENS * ROWS, 16]
        ll: fx.Array[fx.Int32, s * hc.LROW // 2, 16]
        ids: fx.Array[fx.Int32, L.MAX_TOKENS, 16]  # E6b.2 (emb_ids) only

    keyed = key_tuple(key, digest())
    name = f"flashnext_mono_front_s{s}_t{key.tp}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def front(
        emb: Int64,
        prev: Int64,
        w_ne: Int64,
        w_nh: Int64,
        w_fe: Int64,
        w_fh: Int64,
        hc_w: Int64,
        w_down: Int64,
        w_up: Int64,
        state_out: Int64,
        bin_out: Int64,
        inj_out: Int64,
        scratch: Int64,
        epoch: Int64,
        slot: Int32,
        dbg: Int64,
        ids: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        xst = alloc.allocate(XLds).peek()
        mb = Mailbox(step_tag(epoch, slot) - 1)
        c = {
            "S": s,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "eps": key.eps,
            "ne_eps": key.ne_eps,
            "nh_eps": key.nh_eps,
            "pre_fh": key.pre_fh,
            "emb_ids": key.emb_ids,
            "dbg": dbg if key.trace else None,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "rs": lds.rs.ptr,
            "rs2": lds.rs2.ptr,
            "fe": lds.fe.ptr,
            "ll": lds.ll.ptr,
            "ids_lds": lds.ids.ptr,
            "xl": xst.xl.ptr,
            "msqe": sreg(scratch, scr["msqe"][0], "msqe"),
            "msqh": sreg(scratch, scr["msqh"][0], "msqh"),
            "sflag": sreg(scratch, scr["sflag"][0], "sflag"),
            "args": {
                "emb": emb,
                "prev": prev,
                "w_ne": w_ne,
                "w_nh": w_nh,
                "w_fe": w_fe,
                "w_fh": w_fh,
                "inj_out": inj_out,
            },
        }
        c["hc"] = {
            "state": state_out,
            "state_cm": CM_DEV,
            "w_norm": hc_w,
            "w_down": w_down,
            "w_up": w_up,
            "msq": sreg(scratch, scr["msq"][0], "msq"),
            "msq_in": None,
            "dpart": sreg(scratch, scr["dpart"][0], "dpart"),
            "lora": sreg(scratch, scr["lora"][0], "lora"),
            "binf": None,
            "bin": bin_out,
            "inj_out": lambda t, r, vs: _put_inj(c, t, r, vs),
        }
        stamps.stamp(c, 0)  # Stage 0a: kernel entry
        # ---- E6b.2 (emb_ids): every CTA's own LDS copy of `ids` (int64, low
        # dword only), read once so later per-element lookups are LDS reads
        if const_expr(key.emb_ids):
            if tid < s:
                fx.ptr_store(ld_i32(ids, 2 * tid), c["ids_lds"] + tid)
            gpu.barrier()
        # ---- F1 (norms) + F2 (fc projections), CTAs 0 .. 159 double as F1's
        # producers (0 .. 19) and the HC front's up tasks; w_fe / w_fh loaded
        # first (in flight while F1's producers run and load_rrms_front polls)
        if bid < hc.UP_TASKS:
            fe_ops = gemv.loads(c["lane"], c["wave"], w_fe, bid * ROWS, H)
            if const_expr(not key.pre_fh):
                fh_ops = gemv.loads(c["lane"], c["wave"], w_fh, bid * ROWS, H)
            else:
                fh_ops = None
            if bid < hc.NRED:
                stage_front_msq(c, bid)
            load_rrms_front(c)
            stamps.stamp(c, 1)  # Stage 0a: after load_rrms_front
            stage_fc(c, bid, fe_ops, fh_ops)
        # ---- F3: k1's front_only HC front, waiting on every F2 flag first;
        # the HC norm's own sums of squares (``hc.stage_msq``, unmodified)
        # read ``state`` back the same way a later layer's front would
        if bid < D0:
            uops = hc.up_loads(c, bid)
            wait_state(c)
            if bid < hc.NRED:
                hc.stage_msq(c, bid)
            hc.load_rrms(c)
            hc.stage_up(c, bid, uops)
        if (bid >= D0) & (bid < hc.UP_TASKS):
            dops = hc.down_loads(c, bid - D0)
            uops = hc.up_loads(c, bid)
            wait_state(c)
            hc.load_rrms(c)
            hc.stage_down(c, bid - D0, dops)
            hc.stage_up(c, bid, uops)
        if bid >= hc.UP_TASKS:
            dops = hc.down_loads(c, bid - D0)
            wait_state(c)
            hc.load_rrms(c)
            hc.stage_down(c, bid - D0, dops)
        stamps.stamp(c, 10)  # Stage 0a: kernel end

    @flyc.jit
    def launch(
        emb: Int64,
        prev: Int64,
        w_ne: Int64,
        w_nh: Int64,
        w_fe: Int64,
        w_fh: Int64,
        hc_w: Int64,
        w_down: Int64,
        w_up: Int64,
        state_out: Int64,
        bin_out: Int64,
        inj_out: Int64,
        scratch: Int64,
        epoch: Int64,
        slot: Int32,
        dbg: Int64,
        ids: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        front(
            emb,
            prev,
            w_ne,
            w_nh,
            w_fe,
            w_fh,
            hc_w,
            w_down,
            w_up,
            state_out,
            bin_out,
            inj_out,
            scratch,
            epoch,
            slot,
            dbg,
            ids,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(front, launch)
    _BUILDS[key] = launch
    return launch


def _shape_checks(
    s, emb, prev, w_ne, w_nh, w_fe, w_fh, hc_w, w_down, w_up, state_out, bin_out,
    inj_out, emb_ids=False,
):  # fmt: skip
    bf = torch.bfloat16
    if emb_ids:
        need(
            emb.dim() == 2 and emb.shape[1] == H and emb.dtype == bf,
            "emb (emb_ids=True): bf16 [V, H], a replicated vocab table",
        )
    else:
        need(emb.shape == (s, H) and emb.dtype == bf, "emb: bf16 [S, H]")
    need(prev.shape == (s, HH) and prev.dtype == bf, "prev: bf16 [S, 4 H]")
    need(w_ne.shape == (H,) and w_ne.dtype == bf, "pre_fc_norm_embedding weight: [H]")
    need(w_nh.shape == (HH,) and w_nh.dtype == bf, "pre_fc_norm_hidden weight: [4 H]")
    need(w_fe.shape == (H, H) and w_fe.dtype == bf, "fc_embedding: bf16 [H, H]")
    need(w_fh.shape == (H, H) and w_fh.dtype == bf, "fc_hidden: bf16 [H, H]")
    need(hc_w.shape == (HH,) and hc_w.dtype == bf, "attention HC norm weight: [4 H]")
    need(
        w_down.shape == (L.DOWN_ROWS, HH) and w_down.dtype == bf, "HC down: [336, 4 H]"
    )
    need(w_up.shape == (HH, L.LORA) and w_up.dtype == bf, "HC up: [4 H, 320]")
    need(state_out.shape == (s, HH) and state_out.dtype == bf, "state_out: [S, 4 H]")
    need(bin_out.shape == (s, H) and bin_out.dtype == bf, "bin_out: bf16 [S, H]")
    need(inj_out.shape == (s, HC) and inj_out.dtype == bf, "inj_out: bf16 [S, 4]")
    tensors = [
        emb, prev, w_ne, w_nh, w_fe, w_fh, hc_w, w_down, w_up, state_out, bin_out,
        inj_out,
    ]  # fmt: skip
    need(all(t.is_contiguous() for t in tensors), "a non-contiguous input")


def draft_front(
    key: FrontBuild,
    *,
    emb,
    prev,
    w_ne,
    w_nh,
    w_fe,
    w_fh,
    hc_w,
    w_down,
    w_up,
    state_out,
    bin_out,
    inj_out,
    scratch,
    epoch,
    slot,
    dbg=None,
    ids=None,
):
    """Host launcher of the MTP draft step's front: the ``fc_embedding`` /
    ``fc_hidden`` projections (every rank's full, gathered weights) folded
    into the attention HC's norm and mix, one launch. ``emb``: the step's
    token embedding rows (``pre_fc_norm_embedding``'s input); ``prev``: the
    multi-stream state the previous draft step (or the backbone) left behind
    (``pre_fc_norm_hidden``'s input, one norm over all 4 H); ``state_out``:
    this step's materialized multi-stream state (K2's next input); ``bin_out``
    / ``inj_out``: the QSA layer's attention input and injection logits, as
    ``k1.hc_front`` produces them. ``epoch`` a device int32 bumped once a
    step, ``slot`` this launch's tag slot. ``dbg``: Stage 0a's stamp buffer
    (``stamps.dbg_buffer``), required iff ``key.trace``. ``ids``: E6b.2's
    per-token vocab ids (int64 [S]), required iff ``key.emb_ids`` -- then
    ``emb`` is the full replicated vocab table ``[V, H]``, not ``[S, H]``."""
    s = key.tokens
    _shape_checks(
        s, emb, prev, w_ne, w_nh, w_fe, w_fh, hc_w, w_down, w_up, state_out, bin_out,
        inj_out, emb_ids=key.emb_ids,
    )  # fmt: skip
    need(key.trace == (dbg is not None), "dbg: required iff key.trace")
    need(key.emb_ids == (ids is not None), "ids: required iff key.emb_ids")
    if ids is not None:
        need(
            ids.shape == (s,) and ids.dtype == torch.int64 and ids.is_contiguous(),
            "ids: int64 [S]",
        )
    vals = {
        "emb": emb.data_ptr(),
        "prev": prev.data_ptr(),
        "w_ne": w_ne.data_ptr(),
        "w_nh": w_nh.data_ptr(),
        "w_fe": w_fe.data_ptr(),
        "w_fh": w_fh.data_ptr(),
        "hc_w": hc_w.data_ptr(),
        "w_down": w_down.data_ptr(),
        "w_up": w_up.data_ptr(),
        "state_out": state_out.data_ptr(),
        "bin_out": bin_out.data_ptr(),
        "inj_out": inj_out.data_ptr(),
        "scratch": scratch.data_ptr(),
        "epoch": epoch.data_ptr(),
        "slot": slot,
        "dbg": 0 if dbg is None else dbg.data_ptr(),
        "ids": 0 if ids is None else ids.data_ptr(),
    }
    build(key)(*ABI.pack(vals), stream=torch.cuda.current_stream())
