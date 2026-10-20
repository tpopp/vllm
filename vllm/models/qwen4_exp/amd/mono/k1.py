# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 ``gdn_pre``: the front of one Qwen3.8-Flash-Next layer of a decode step,
S <= 8 rows, one token a request: the attention HC's norm and mix of the
(materialized) multi-stream state, then, on a GDN layer, the linear attention
up to the gated core output (K2's out_proj input).

One launch a layer and rank, ``BLOCKS`` x ``THREADS``:

    msq      (``self_msq``, CTAs 0 .. 19): the state's sums of squares (the
             model's first layer and the one after the PLE; past them K2 hands
             them over as ``msq_in``)
    down     (``hc.DOWN_TASKS``, CTAs 88 ..): xn of the state, [down; inject]
             -> LORA and the injection logits -> ``inj_out`` (bf16)
    up       (``hc.UP_TASKS``, CTAs 0 .. 159): the gate-mix -> block input BIN
             (``front_only``: ``bin_out``, the QSA layers' attention input; the
             launch ends there)
    in_proj  (every CTA: in_proj_qkvz row groups bid, bid + 256 through MXFP4,
             their weights in flight from the start; in_proj_ba as 3 row
             groups x 2 K halves on the unquantized rows, CTAs 160 ..)
    gdn      (S x NV value heads x 8 chunks of 16 state rows, task n on CTA
             (n + goff) % 256): as the Qwen3.8 mono's K1 (``qwen3_5.amd.mono.
             gdn``), at this model's head counts

Every CTA runs its GDN tasks phase by phase (conv, gates, recurrence, norm) and
a phase polls only work of an earlier phase, so the task placement needs no
round structure. A row whose state index is <= 0 writes no state and a zero
core row, as stock.

``proj_only`` (P1b, ``VLLM_ROCM_MONO_K1_PROJ``): a GDN verify step's K1,
in_proj only -- the HC front, in_proj_qkvz and in_proj_ba as the decode build
above runs them, their bf16 rows going straight to ``qkvz_out`` / ``ba_out``
instead of the GDN tasks (not run here: a verify step still needs the stock
AITER conv + gated delta rule for the per-draft-token state rollback K1's
decode recurrence does not do, ``gather/gdn_verify.md`` §2.4).

``qsa_proj`` (P2, ``VLLM_ROCM_MONO_K1_QSA_PROJ``): a QSA layer's K1, the HC
front plus the QSA attention's two input GEMVs -- ``attn.qkv_proj`` (``lay.QKV``
bf16 rows) and the indexer's ``index_qk_proj`` (``L.IDXQK`` bf16 rows,
replicated) against the just-produced block input, straight to ``qkv_out`` /
``idx_out`` -- so the verify step's two skinny BF16 GEMMs after this launch
(``mono_decode._qsa_core``'s ``attn.qkv_proj`` / ``indexer.index_qk_proj``) are
no longer needed; ``qsa_prepare``, top-k and sparse attention stay separate
launches on ``qkv_out`` / ``idx_out``. Both weight streams are BF16 and far
bigger than in_proj_qkvz's MXFP4 one (35.6 MiB combined), so, unlike
``qkvz_loads``, only round 0 of ``qkv_proj`` (one row group a CTA) and
``index_qk_proj``'s (CTAs 0 .. ``idx_tasks``) are issued at kernel entry; the
second ``qkv_proj`` round (CTAs needing two row groups, ``lay.QKV`` having
more than ``BLOCKS`` of them) is issued only after round 0's load has been
consumed, to keep the two BF16 streams' VGPRs from overlapping.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64

from vllm.models.kimi_k3.amd.mono.common.ops import (
    CM_DEV,
    bf16_pair,
    bf16_round,
    row_sum,
    rsrc,
    traced,
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
from vllm.models.qwen3_5.amd.mono import gdn as G
from vllm.models.qwen3_5.amd.mono.gdn import NVCH, VCH, gdn_tasks
from vllm.models.qwen3_5.amd.mono.moe import (
    fp4_gemv,
    fp4_gemv_loads,
    quant_x,
    sigmoid,
    step_tag,
)
from vllm.models.qwen4_exp.amd.mono import hc
from vllm.models.qwen4_exp.amd.mono import layout as L
from vllm.models.qwen4_exp.amd.mono.layout import need
from vllm.models.qwen4_exp.amd.mono.sources import digest

H, HH, HC, ROWS, HD = L.HIDDEN, L.HH, L.HC, L.ROWS, L.HD
D0 = BLOCKS - hc.DOWN_TASKS  # 88: the down CTAs, to the last
SL = L.CONV_W - 1  # conv state entries (no spec decode)
BA_PARTS = 2
BA_K = H // BA_PARTS
BA0 = hc.UP_TASKS  # 160: in_proj_ba's CTAs (down only)
XROW = H + gemv.LDS_PAD

assert hc.UP_TASKS <= BLOCKS and D0 >= 0 and hc.NRED <= hc.UP_TASKS

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K1Build:
    tokens: int
    tp: int = 2
    self_msq: bool = False  # the state's sums of squares computed here
    front_only: bool = False  # the HC front alone (a QSA layer)
    proj_only: bool = False  # the HC front + in_proj_qkvz/ba, no GDN tasks
    qsa_proj: bool = False  # the HC front + qkv_proj/index_qk_proj (a QSA layer)
    eps: float = 1e-6  # the HC norm
    norm_eps: float = 1e-6  # the GDN gated norm


ABI = KernelAbi(
    (
        "state",
        "msq_in",
        "hc_w",
        "w_down",
        "w_up",
        "inj_out",
        "bin_out",
        "w_qkvz",
        "w_qkvzs",
        "w_ba",
        "conv_w",
        "conv_st",
        "cs_seq",
        "cs_dim",
        "cs_tok",
        "a_log",
        "dt_bias",
        "norm_w",
        "rstate",
        "rs_seq",
        "st_idx",
        "core",
        "scratch",
        "epoch",
        "layer",
        "qkvz_out",
        "ba_out",
        "w_qkv",
        "w_idx",
        "qkv_out",
        "idx_out",
    )
)


def ba_tasks(lay):
    return lay.BA // ROWS * BA_PARTS


def gdn_tasks_of(lay):
    """GDN tasks a token."""
    return lay.NV * NVCH


def scratch_layout(key: K1Build) -> dict:
    s = key.tokens
    lay = L.Layout(key.tp)
    pairs = hc.regions(s)
    gdn_layer = not key.front_only and not key.qsa_proj
    if gdn_layer:
        pairs = pairs | {"bapart": ba_tasks(lay) * s * ROWS}
        if not key.proj_only:
            # proj_only stores in_proj_qkvz straight to ``qkvz_out``: no GDN
            # tasks poll a mailbox copy of it, so there is none.
            pairs = pairs | {
                "proj": s * lay.QKVZ,
                "qk": s * lay.NK * 2 * HD,
                "onrm": s * lay.NV * NVCH,
            }
    out = pair_layout(pairs.items())
    if not key.front_only:
        # proj_only, qsa_proj and the full decode build all read the block
        # input back in-launch (``hc.bin_to_lds``), so all three need the
        # scratch copy -- a QSA layer has none of the GDN regions above.
        end = (max(o + n for o, n in out.values()) + 15) // 16 * 16
        out["bin"] = (end, hc.bin_bytes(s))
    return out


def scratch_bytes(key: K1Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def f32_of(w):
    return w.bitcast(fx.Float32)


def gdn_offset(key: K1Build) -> int:
    """The first GDN CTA: the high (down only) CTAs while a single round fits."""
    n = key.tokens * gdn_tasks_of(L.Layout(key.tp))
    return BLOCKS - n if n < BLOCKS else 0


# ---------------------------------------------------------------- in_proj
def qkvz_loads(c, rg):
    a = c["args"]
    return fp4_gemv_loads(c["lane"], c["wave"], a["w_qkvz"], a["w_qkvzs"], rg, H)


@traced
def stage_qkvz(c, rg, ops):
    """in_proj_qkvz rows 16 rg .. on the MXFP4-exact x rows -> bf16 PROJ (or,
    ``K1Build.proj_only``, straight to ``qkvz_out``: no GDN task of this
    launch polls the row there, so a plain store serves it)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    acc = fp4_gemv(lane, wave, c["xl"], XROW, H, s, rg, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS * s:
        r = tid % ROWS
        t = tid // ROWS
        v = bf16_round(row_sum(red, r, t))
        at = t * c["lay"].QKVZ + rg * ROWS + r
        if const_expr(c["qkvz_out"] is not None):
            bo.buffer_store(v.to(fx.BFloat16), rsrc(c["qkvz_out"]), at)
        else:
            c["put"](c["proj"], at, v)
    gpu.barrier()


# ---------------------------------------------------------------- gdn
def _ba_of(c, t, h):
    """Head h's b and a of token t: the sums of in_proj_ba's K halves (bf16)."""
    return G.ba_sums(c, t, h, BA_PARTS)


@traced
def store_ba_out(c):
    """``K1Build.proj_only``: one task a (token, value head), S NV of them --
    always <= BLOCKS (NV <= 24, S <= 8), so one CTA each, no rounds. Finalizes
    ``ba_tasks``'s K-split partials the way ``_ba_of`` already does for the
    decode build's gates stage (a poll of ``bapart``, same hand-off), storing
    to ``ba_out`` [2, S, NV] (b plane, then a plane: the contiguous b / a the
    stock conv + gated delta rule call takes, so no unpack launch is needed)
    instead of reading them in place."""
    lay, s, bid = c["lay"], c["S"], c["bid"]
    if bid < s * lay.NV:
        t, h = bid // lay.NV, bid % lay.NV
        b, a = c["ba_of"](c, t, h)
        bo.buffer_store(b.to(fx.BFloat16), rsrc(c["ba_out"]), t * lay.NV + h)
        bo.buffer_store(a.to(fx.BFloat16), rsrc(c["ba_out"]), (s + t) * lay.NV + h)


# ---------------------------------------------------------------- qsa_proj
@traced
def _store_qsa_row(c, rg, valid, ops, out, nrows):
    """Row group ``rg`` (16 of ``out``'s ``nrows`` output columns, clamped
    into range by the caller) against the block input already in ``c["xl"]``
    (``hc.bin_to_lds``): bf16(x @ w[rg]^T), the same GEMV ``stage_qkvz`` runs
    for in_proj_qkvz (bf16, not FP4) -- stored only while ``valid`` (a CTA
    past the real task count still runs the GEMV, just discards it, rather
    than branch around it) -> ``out`` [S, nrows] bf16, device scope."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    acc = gemv.mfmas(lane, wave, c["xl"], XROW, H, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        tt = tid // (ROWS // 2)
        v0 = bf16_round(row_sum(red, 2 * rp, tt))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, tt))
        if valid:
            bo.buffer_store(
                bf16_pair(v0, v1).bitcast(fx.Int32),
                rsrc(out),
                (tt * nrows + rg * ROWS + 2 * rp) // 2,
                cache_modifier=CM_DEV,
            )
    gpu.barrier()


@traced
def stage_qsa_proj(
    c, qkv_rg0, qkv_ops0, idx_rg, idx_ops, idx_tasks, qkv_rounds, qkv_tasks
):
    """``K1Build.qsa_proj``: round 0's ``qkv_proj`` row group (every CTA,
    always real: ``qkv_tasks >= BLOCKS``) and ``index_qk_proj``'s (CTAs <
    ``idx_tasks``, else a harmlessly discarded clamped GEMV), both loaded at
    kernel entry -> ``qkv_out`` / ``idx_out``. If ``qkv_proj`` has more row
    groups than CTAs (``qkv_rounds > 1``), round 1's load is issued only now
    -- after round 0's is consumed -- so the two concurrent BF16 weight loads
    never overlap round 0's, within the VGPR budget round 0 + ``index_qk_proj``
    already uses."""
    lay, bid = c["lay"], c["bid"]
    _store_qsa_row(c, qkv_rg0, True, qkv_ops0, c["qkv_out"], lay.QKV)
    _store_qsa_row(c, idx_rg, bid < idx_tasks, idx_ops, c["idx_out"], L.IDXQK)
    if const_expr(qkv_rounds > 1):
        rg1_raw = bid + BLOCKS
        rg1 = fx.min(rg1_raw, qkv_tasks - 1)
        ops1 = gemv.loads(c["lane"], c["wave"], c["args"]["w_qkv"], rg1 * ROWS, H)
        _store_qsa_row(c, rg1, rg1_raw < qkv_tasks, ops1, c["qkv_out"], lay.QKV)


def qkv_tasks_of(lay):
    return lay.QKV // ROWS


def idx_tasks_of():
    return L.IDXQK // ROWS


# ---------------------------------------------------------------- kernel
def _put_inj(c, t, r, vs):
    """The injection rows r, r + 1 of token t -> ``inj_out`` [S, HC] bf16."""
    bo.buffer_store(
        bf16_pair(vs[0], vs[1]).bitcast(fx.Int32),
        rsrc(c["args"]["inj_out"]),
        (t * HC + r) // 2,
    )


_BUILDS: dict = {}


def build(key: K1Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    need(1 <= s <= L.MAX_TOKENS, f"{s} rows, the kernels take 1..{L.MAX_TOKENS}")
    need(
        sum((key.front_only, key.proj_only, key.qsa_proj)) <= 1,
        "front_only, proj_only and qsa_proj are mutually exclusive",
    )
    lay = L.Layout(key.tp)
    scr = scratch_layout(key)
    gt = gdn_tasks_of(lay)
    nr = -(-s * gt // BLOCKS)
    nba = ba_tasks(lay)
    need(BA0 + nba <= BLOCKS, "in_proj_ba's CTAs")
    need(lay.QKVZ // ROWS == 2 * BLOCKS, "in_proj_qkvz: two row groups a CTA")
    need(H % (BA_PARTS * gemv.KCH) == 0, "in_proj_ba's K halves")
    need(lay.QKV % ROWS == 0, "qkv_proj rows, a whole number of 16-row tasks")
    need(L.IDXQK % ROWS == 0, "index_qk_proj rows, a whole number of 16-row tasks")
    qkv_tasks = qkv_tasks_of(lay)
    idx_tasks = idx_tasks_of()
    qkv_rounds = -(-qkv_tasks // BLOCKS)
    need(qkv_tasks >= BLOCKS, "qkv_proj: round 0 covers every CTA")
    need(idx_tasks <= BLOCKS, "index_qk_proj: one round covers it")
    xwords = s * (max(H, hc.DK) + gemv.LDS_PAD) // 2

    @fx.struct
    class XLds:
        xl: fx.Array[fx.Int32, xwords, 16]

    @fx.struct
    class GdnLds:
        qkl: fx.Array[fx.Float32, nr * 2 * HD, 16]
        vloc: fx.Array[fx.Float32, nr * VCH, 16]
        ol: fx.Array[fx.Float32, nr * VCH, 16]
        scal: fx.Array[fx.Float32, nr * 4, 16]

    @fx.union
    class StageLds:
        x: XLds
        gdn: GdnLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        rs: fx.Array[fx.Float32, L.MAX_TOKENS * HC, 16]
        ll: fx.Array[fx.Int32, s * hc.LROW // 2, 16]

    keyed = key_tuple(key, digest())
    name = (
        f"flashnext_mono_k1_s{s}_t{key.tp}_m{int(key.self_msq)}_f{int(key.front_only)}"
        f"_p{int(key.proj_only)}_q{int(key.qsa_proj)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k1(
        state: Int64,
        msq_in: Int64,
        hc_w: Int64,
        w_down: Int64,
        w_up: Int64,
        inj_out: Int64,
        bin_out: Int64,
        w_qkvz: Int64,
        w_qkvzs: Int64,
        w_ba: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        norm_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        core: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        qkvz_out: Int64,
        ba_out: Int64,
        w_qkv: Int64,
        w_idx: Int64,
        qkv_out: Int64,
        idx_out: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        st_lds = alloc.allocate(StageLds)
        xl, gl = st_lds.x.peek(), st_lds.gdn.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1)
        c = {
            "S": s,
            "lay": lay,
            "gdn": lay.gdn,
            "ba_of": _ba_of,
            "z_gate": sigmoid,
            "goff": gdn_offset(key),
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "eps": key.eps,
            "norm_eps": key.norm_eps,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "rs": lds.rs.ptr,
            "ll": lds.ll.ptr,
            "xl": xl.xl.ptr,
            "qkl": gl.qkl.ptr,
            "vloc": gl.vloc.ptr,
            "ol": gl.ol.ptr,
            "scal": gl.scal.ptr,
            "args": {
                "inj_out": inj_out,
                "w_qkvz": w_qkvz,
                "w_qkvzs": w_qkvzs,
                "w_ba": w_ba,
                "conv_w": conv_w,
                "conv_st": conv_st,
                "cs_seq": cs_seq,
                "cs_dim": cs_dim,
                "cs_tok": cs_tok,
                "a_log": a_log,
                "dt_bias": dt_bias,
                "norm_w": norm_w,
                "rstate": rstate,
                "rs_seq": rs_seq,
                "st_idx": st_idx,
                "core": core,
                "w_qkv": w_qkv,
                "w_idx": w_idx,
            },
        }
        gdn_layer = not key.front_only and not key.qsa_proj
        if const_expr(gdn_layer):
            regions = ("bapart",) if key.proj_only else ("proj", "qk", "onrm", "bapart")
            for region in regions:
                c[region] = sreg(scratch, scr[region][0], region)
            c["qkvz_out"] = qkvz_out if key.proj_only else None
            c["ba_out"] = ba_out if key.proj_only else None
        c["qkv_out"] = qkv_out if key.qsa_proj else None
        c["idx_out"] = idx_out if key.qsa_proj else None
        c["hc"] = {
            "state": state,
            "state_cm": 0,
            "w_norm": hc_w,
            "w_down": w_down,
            "w_up": w_up,
            "msq": sreg(scratch, scr["msq"][0], "msq"),
            "msq_in": None if key.self_msq else msq_in,
            "dpart": sreg(scratch, scr["dpart"][0], "dpart"),
            "lora": sreg(scratch, scr["lora"][0], "lora"),
            "binf": None if key.front_only else sreg(scratch, scr["binf"][0], "binf"),
            "bin": bin_out if key.front_only else scratch + fx.Int64(scr["bin"][0]),
            "bin_out": bin_out if (key.proj_only or key.qsa_proj) else None,
            "inj_out": lambda t, r, vs: _put_inj(c, t, r, vs),
        }
        # ---- the HC front: a CTA's whole sequence in one branch, its weights
        # (in_proj_qkvz's, or qsa_proj's round-0 qkv_proj / full index_qk_proj
        # ones) in flight first
        if const_expr(gdn_layer):
            qops = [qkvz_loads(c, bid + BLOCKS * j) for j in range(2)]
        if const_expr(key.qsa_proj):
            qkv_ops0 = gemv.loads(c["lane"], c["wave"], w_qkv, bid * ROWS, H)
            idx_rg = fx.min(bid, idx_tasks - 1)
            idx_ops = gemv.loads(c["lane"], c["wave"], w_idx, idx_rg * ROWS, H)
        if const_expr(key.self_msq):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if bid < hc.NRED:
                hc.stage_msq(c, bid)
        if bid < D0:
            uops = hc.up_loads(c, bid)
            hc.load_rrms(c)
            hc.stage_up(c, bid, uops)
        if (bid >= D0) & (bid < hc.UP_TASKS):
            dops = hc.down_loads(c, bid - D0)
            uops = hc.up_loads(c, bid)
            hc.load_rrms(c)
            hc.stage_down(c, bid - D0, dops)
            hc.stage_up(c, bid, uops)
        if const_expr(key.front_only):
            if bid >= hc.UP_TASKS:
                dops = hc.down_loads(c, bid - D0)
                hc.load_rrms(c)
                hc.stage_down(c, bid - D0, dops)
        elif const_expr(key.qsa_proj):
            if bid >= hc.UP_TASKS:
                dops = hc.down_loads(c, bid - D0)
                hc.load_rrms(c)
                hc.stage_down(c, bid - D0, dops)
            # ---- qkv_proj / index_qk_proj against the block input, round 0's
            # (and index_qk_proj's) loads already in flight at kernel entry;
            # round 1's qkv_proj load (if any) is issued only now, inside
            # ``stage_qsa_proj``, after round 0's is consumed
            hc.bin_to_lds(c, XROW)
            stage_qsa_proj(
                c, bid, qkv_ops0, idx_rg, idx_ops, idx_tasks, qkv_rounds, qkv_tasks
            )
        else:
            is_ba = (bid >= BA0) & (bid < BA0 + nba)
            if is_ba:
                dops = hc.down_loads(c, bid - D0)
                bops = G.ba_loads(c, bid - BA0, BA_PARTS, BA_K)
                hc.load_rrms(c)
                hc.stage_down(c, bid - D0, dops)
                hc.bin_to_lds(c, XROW)
                G.stage_ba(c, bid - BA0, bops, BA_PARTS, BA_K)
            if bid >= BA0 + nba:
                dops = hc.down_loads(c, bid - D0)
                hc.load_rrms(c)
                hc.stage_down(c, bid - D0, dops)
            # ---- in_proj on the block input, then the GDN tasks (proj_only:
            # just the projections, finalizing in_proj_ba's K-split partials
            # to ``ba_out`` in place of reading them)
            if ~is_ba:
                hc.bin_to_lds(c, XROW)
            quant_x(c, H)
            for j in range_constexpr(2):
                stage_qkvz(c, bid + BLOCKS * j, qops[j])
            if const_expr(key.proj_only):
                store_ba_out(c)
            else:
                gdn_tasks(c, nr)

    @flyc.jit
    def launch(
        state: Int64,
        msq_in: Int64,
        hc_w: Int64,
        w_down: Int64,
        w_up: Int64,
        inj_out: Int64,
        bin_out: Int64,
        w_qkvz: Int64,
        w_qkvzs: Int64,
        w_ba: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        norm_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        core: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        qkvz_out: Int64,
        ba_out: Int64,
        w_qkv: Int64,
        w_idx: Int64,
        qkv_out: Int64,
        idx_out: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k1(
            state,
            msq_in,
            hc_w,
            w_down,
            w_up,
            inj_out,
            bin_out,
            w_qkvz,
            w_qkvzs,
            w_ba,
            conv_w,
            conv_st,
            cs_seq,
            cs_dim,
            cs_tok,
            a_log,
            dt_bias,
            norm_w,
            rstate,
            rs_seq,
            st_idx,
            core,
            scratch,
            epoch,
            layer,
            qkvz_out,
            ba_out,
            w_qkv,
            w_idx,
            qkv_out,
            idx_out,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(k1, launch)
    _BUILDS[key] = launch
    return launch


def _hc_checks(key, s, state, msq_in, hc_w, w_down, w_up, inj_out):
    bf = torch.bfloat16
    need(state.shape == (s, HH) and state.dtype == bf, "state: bf16 [S, 4 H]")
    need(
        (msq_in is None) == key.self_msq
        and (
            msq_in is None
            or (msq_in.numel() == hc.NRED * s * HC and msq_in.dtype == torch.float32)
        ),
        "msq_in: fp32 [20, S, 4], absent with self_msq only",
    )
    need(hc_w.shape == (HH,) and hc_w.dtype == bf, "attention HC norm weight: [4 H]")
    need(
        w_down.shape == (L.DOWN_ROWS, HH) and w_down.dtype == bf, "HC down: [336, 4 H]"
    )
    need(w_up.shape == (HH, L.LORA) and w_up.dtype == bf, "HC up: [4 H, 320]")
    need(inj_out.shape == (s, HC) and inj_out.dtype == bf, "inj_out: bf16 [S, 4]")
    tensors = [state, hc_w, w_down, w_up, inj_out]
    tensors += [msq_in] if msq_in is not None else []
    need(all(t.is_contiguous() for t in tensors), "a non-contiguous HC input")


def hc_front(
    key: K1Build,
    *,
    state,
    msq_in,
    hc_w,
    w_down,
    w_up,
    inj_out,
    bin_out,
    scratch,
    epoch,
    layer,
):
    """Host launcher of the HC front alone (``key.front_only``, a QSA layer):
    the attention HC's norm and mix of ``state`` [S, 4 H] -> ``bin_out`` [S, H]
    bf16 (the attention's input) and ``inj_out`` [S, 4] bf16. ``msq_in``:
    ``state``'s sums of squares as K2 wrote them, None with ``key.self_msq``."""
    s = key.tokens
    need(key.front_only, "hc_front takes a front_only build")
    _hc_checks(key, s, state, msq_in, hc_w, w_down, w_up, inj_out)
    need(
        bin_out.shape == (s, H)
        and bin_out.dtype == torch.bfloat16
        and bin_out.is_contiguous(),
        "bin_out: bf16 [S, H]",
    )
    vals = dict.fromkeys(ABI.names, 0)
    vals |= {
        "state": state.data_ptr(),
        "msq_in": 0 if msq_in is None else msq_in.data_ptr(),
        "hc_w": hc_w.data_ptr(),
        "w_down": w_down.data_ptr(),
        "w_up": w_up.data_ptr(),
        "inj_out": inj_out.data_ptr(),
        "bin_out": bin_out.data_ptr(),
        "scratch": scratch.data_ptr(),
        "epoch": epoch.data_ptr(),
        "layer": layer,
    }
    build(key)(*ABI.pack(vals), stream=torch.cuda.current_stream())


def gdn_pre(
    key: K1Build,
    *,
    state,
    msq_in,
    hc_w,
    w_down,
    w_up,
    inj_out,
    w_qkvz,
    w_qkvzs,
    w_ba,
    conv_w,
    conv_state,
    a_log,
    dt_bias,
    norm_w,
    rstate,
    st_idx,
    core,
    scratch,
    epoch,
    layer,
):
    """Host launcher of a GDN layer's K1; every weight as vLLM holds it after
    loading (in_proj_qkvz online MXFP4: the ASM FP4 GEMM's shuffled weight and
    e8m0 scales). ``conv_state`` [slots, CONV, 3] or a transposed view of
    [slots, 3, CONV] (strides read); ``rstate`` [slots, NV, 128, 128] fp32, a
    slot dense, slots strided; ``st_idx`` [>= S] int32, <= 0 a pad row;
    ``epoch`` a device int32 bumped once a step, ``layer`` this launch's tag
    slot."""
    s = key.tokens
    lay = L.Layout(key.tp)
    bf = torch.bfloat16
    need(not key.front_only, "gdn_pre takes a GDN build")
    _hc_checks(key, s, state, msq_in, hc_w, w_down, w_up, inj_out)
    need(
        core.shape == (s, lay.CORE) and core.dtype == bf, f"core: bf16 [S, {lay.CORE}]"
    )
    need(
        w_qkvz.shape == (lay.QKVZ, H // 2)
        and w_qkvz.dtype in (torch.uint8, torch.float4_e2m1fn_x2),
        f"in_proj_qkvz: fp4x2 [{lay.QKVZ}, {H // 2}]",
    )
    need(
        w_qkvzs.numel() == lay.QKVZ * H // 32 and w_qkvzs.element_size() == 1,
        "in_proj_qkvz scales: e8m0, one a 32-block",
    )
    need(
        w_ba.shape == (lay.BA, H) and w_ba.dtype == bf,
        f"in_proj_ba: bf16 [{lay.BA}, {H}]",
    )
    need(conv_w.numel() == lay.CONV * L.CONV_W and conv_w.dtype == bf, "conv1d weight")
    need(
        conv_state.dtype == bf and conv_state.shape[1:] == (lay.CONV, SL),
        f"conv state: bf16 [slots, {lay.CONV}, 3]",
    )
    need(
        rstate.dtype == torch.float32 and rstate.shape[1:] == (lay.NV, HD, HD),
        f"recurrent state: fp32 [slots, {lay.NV}, 128, 128]",
    )
    need(rstate[0].is_contiguous(), "a recurrent state slot must be dense")
    need(a_log.dtype == torch.float32 and dt_bias.dtype == bf, "A_log / dt_bias dtype")
    need(norm_w.dtype == bf and norm_w.numel() == HD, "gated norm weight")
    need(st_idx.dtype == torch.int32 and st_idx.numel() >= s, "state indices")
    dense = (w_qkvz, w_qkvzs, w_ba, conv_w, core)
    need(all(t.is_contiguous() for t in dense), "a non-contiguous GDN input")
    vals = {
        "state": state.data_ptr(),
        "msq_in": 0 if msq_in is None else msq_in.data_ptr(),
        "hc_w": hc_w.data_ptr(),
        "w_down": w_down.data_ptr(),
        "w_up": w_up.data_ptr(),
        "inj_out": inj_out.data_ptr(),
        "bin_out": 0,
        "w_qkvz": w_qkvz.data_ptr(),
        "w_qkvzs": w_qkvzs.data_ptr(),
        "w_ba": w_ba.data_ptr(),
        "conv_w": conv_w.data_ptr(),
        "conv_st": conv_state.data_ptr(),
        "cs_seq": conv_state.stride(0),
        "cs_dim": conv_state.stride(1),
        "cs_tok": conv_state.stride(2),
        "a_log": a_log.data_ptr(),
        "dt_bias": dt_bias.data_ptr(),
        "norm_w": norm_w.data_ptr(),
        "rstate": rstate.data_ptr(),
        "rs_seq": rstate.stride(0),
        "st_idx": st_idx.data_ptr(),
        "core": core.data_ptr(),
        "scratch": scratch.data_ptr(),
        "epoch": epoch.data_ptr(),
        "layer": layer,
        "qkvz_out": 0,
        "ba_out": 0,
        "w_qkv": 0,
        "w_idx": 0,
        "qkv_out": 0,
        "idx_out": 0,
    }
    build(key)(*ABI.pack(vals), stream=torch.cuda.current_stream())


def proj_pre(
    key: K1Build,
    *,
    state,
    msq_in,
    hc_w,
    w_down,
    w_up,
    inj_out,
    bin_out,
    w_qkvz,
    w_qkvzs,
    w_ba,
    qkvz_out,
    ba_out,
    scratch,
    epoch,
    layer,
):
    """Host launcher of a GDN layer's K1 ``proj_only`` build (P1b,
    ``VLLM_ROCM_MONO_K1_PROJ``): ``hc_front``'s ``bin_out`` [S, H] / ``inj_out``
    [S, 4], plus in_proj_qkvz's bf16 output -> ``qkvz_out`` [S, QKVZ] and
    in_proj_ba's -> ``ba_out`` [2, S, NV] (b, a planes), the same values/rounding
    ``attn.in_proj_qkvz(x)`` / ``attn.in_proj_ba(x)`` produce. No GDN task
    runs here: the conv update, gated delta rule and gated norm are the stock
    AITER verify core's, called on these two tensors directly
    (``B._gdn_core_raw(qkvz=..., ba=...)``)."""
    s = key.tokens
    lay = L.Layout(key.tp)
    bf = torch.bfloat16
    need(key.proj_only, "proj_pre takes a proj_only build")
    _hc_checks(key, s, state, msq_in, hc_w, w_down, w_up, inj_out)
    need(
        bin_out.shape == (s, H)
        and bin_out.dtype == bf
        and bin_out.is_contiguous(),
        "bin_out: bf16 [S, H]",
    )
    need(
        w_qkvz.shape == (lay.QKVZ, H // 2)
        and w_qkvz.dtype in (torch.uint8, torch.float4_e2m1fn_x2),
        f"in_proj_qkvz: fp4x2 [{lay.QKVZ}, {H // 2}]",
    )
    need(
        w_qkvzs.numel() == lay.QKVZ * H // 32 and w_qkvzs.element_size() == 1,
        "in_proj_qkvz scales: e8m0, one a 32-block",
    )
    need(
        w_ba.shape == (lay.BA, H) and w_ba.dtype == bf,
        f"in_proj_ba: bf16 [{lay.BA}, {H}]",
    )
    need(
        qkvz_out.shape == (s, lay.QKVZ) and qkvz_out.dtype == bf,
        f"qkvz_out: bf16 [S, {lay.QKVZ}]",
    )
    need(
        ba_out.shape == (2, s, lay.NV) and ba_out.dtype == bf,
        f"ba_out: bf16 [2, S, {lay.NV}] (b, a planes)",
    )
    dense = (w_qkvz, w_qkvzs, w_ba, qkvz_out, ba_out)
    need(all(t.is_contiguous() for t in dense), "a non-contiguous GDN input")
    vals = dict.fromkeys(ABI.names, 0)
    vals |= {
        "state": state.data_ptr(),
        "msq_in": 0 if msq_in is None else msq_in.data_ptr(),
        "hc_w": hc_w.data_ptr(),
        "w_down": w_down.data_ptr(),
        "w_up": w_up.data_ptr(),
        "inj_out": inj_out.data_ptr(),
        "bin_out": bin_out.data_ptr(),
        "w_qkvz": w_qkvz.data_ptr(),
        "w_qkvzs": w_qkvzs.data_ptr(),
        "w_ba": w_ba.data_ptr(),
        "qkvz_out": qkvz_out.data_ptr(),
        "ba_out": ba_out.data_ptr(),
        "scratch": scratch.data_ptr(),
        "epoch": epoch.data_ptr(),
        "layer": layer,
    }
    build(key)(*ABI.pack(vals), stream=torch.cuda.current_stream())


def qsa_proj_pre(
    key: K1Build,
    *,
    state,
    msq_in,
    hc_w,
    w_down,
    w_up,
    inj_out,
    bin_out,
    w_qkv,
    w_idx,
    qkv_out,
    idx_out,
    scratch,
    epoch,
    layer,
):
    """Host launcher of a QSA layer's K1 ``qsa_proj`` build (P2,
    ``VLLM_ROCM_MONO_K1_QSA_PROJ``): ``hc_front``'s ``bin_out`` [S, H] /
    ``inj_out`` [S, 4], plus ``attn.qkv_proj``'s bf16 output -> ``qkv_out``
    [S, lay.QKV] and the indexer's ``index_qk_proj``'s -> ``idx_out``
    [S, L.IDXQK], the same values/rounding ``F.linear`` on the HC front's
    ``x`` produces. ``qsa_prepare``, top-k and sparse attention stay a
    separate launch on these two tensors, via
    ``mono_decode._qsa_core``'s ``qkv`` / ``idx_qk`` kwargs."""
    s = key.tokens
    lay = L.Layout(key.tp)
    bf = torch.bfloat16
    need(key.qsa_proj, "qsa_proj_pre takes a qsa_proj build")
    _hc_checks(key, s, state, msq_in, hc_w, w_down, w_up, inj_out)
    need(
        bin_out.shape == (s, H)
        and bin_out.dtype == bf
        and bin_out.is_contiguous(),
        "bin_out: bf16 [S, H]",
    )
    need(
        w_qkv.shape == (lay.QKV, H) and w_qkv.dtype == bf,
        f"qkv_proj: bf16 [{lay.QKV}, {H}]",
    )
    need(
        w_idx.shape == (L.IDXQK, H) and w_idx.dtype == bf,
        f"index_qk_proj: bf16 [{L.IDXQK}, {H}]",
    )
    need(
        qkv_out.shape == (s, lay.QKV) and qkv_out.dtype == bf,
        f"qkv_out: bf16 [S, {lay.QKV}]",
    )
    need(
        idx_out.shape == (s, L.IDXQK) and idx_out.dtype == bf,
        f"idx_out: bf16 [S, {L.IDXQK}]",
    )
    dense = (w_qkv, w_idx, qkv_out, idx_out)
    need(all(t.is_contiguous() for t in dense), "a non-contiguous QSA input")
    vals = dict.fromkeys(ABI.names, 0)
    vals |= {
        "state": state.data_ptr(),
        "msq_in": 0 if msq_in is None else msq_in.data_ptr(),
        "hc_w": hc_w.data_ptr(),
        "w_down": w_down.data_ptr(),
        "w_up": w_up.data_ptr(),
        "inj_out": inj_out.data_ptr(),
        "bin_out": bin_out.data_ptr(),
        "w_qkv": w_qkv.data_ptr(),
        "w_idx": w_idx.data_ptr(),
        "qkv_out": qkv_out.data_ptr(),
        "idx_out": idx_out.data_ptr(),
        "scratch": scratch.data_ptr(),
        "epoch": epoch.data_ptr(),
        "layer": layer,
    }
    build(key)(*ABI.pack(vals), stream=torch.cuda.current_stream())
