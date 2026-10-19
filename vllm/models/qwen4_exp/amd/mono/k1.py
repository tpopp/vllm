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
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64

from vllm.models.kimi_k3.amd.mono.common.ops import (
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
from vllm.models.qwen4_exp.amd.mono import stamps
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
    eps: float = 1e-6  # the HC norm
    norm_eps: float = 1e-6  # the GDN gated norm
    trace: bool = False  # emit in-kernel timestamps to a `dbg` buffer (stamps.py)


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
        "dbg",  # Stage 0a: stamps.py target, 256 * NSLOT int32; 0 when off
    )
)

# ----------------------------------------------------------- in-kernel trace
#
# Slot -> label of this kernel's utilization timeline (``K1Build(trace=True)``,
# a ``dbg`` buffer of ``stamps.dbg_buffer(device, nslot=NSLOT)``). Same
# conventions as K2's (see ``k2.py``): a "_LO" / "_HI" pair brackets a mailbox
# wait (``stamps.stamp_point``, no barrier); a lone slot marks a phase
# boundary (``stamps.stamp``, full-CTA barrier -- every site below is inside
# a ``bid``-gated branch, so it is safe). The four GDN phases run once a
# round (``gdn_tasks``'s ``nr`` rounds on CTAs past a single round's worth of
# tasks); a multi-round CTA's slot holds only its LAST round's timing.
NSLOT = 32
ENTRY = 0
MSQ_SELF_DONE = 2  # stage_msq done (self_msq path, bid < hc.NRED)
MSQ_LO, MSQ_HI = 4, 5  # hc.load_rrms, via "msq" (non-self_msq path)
HC_DOWN_LOADS = 6  # down_loads issued
HC_DOWN_DONE = 7  # hc.stage_down done
LORA_LO, LORA_HI = 8, 9  # hc.lora_to_lds, via "lora"
HC_UP_DONE = 10  # hc.stage_up done
BIN_LO, BIN_HI = 12, 13  # hc.bin_to_lds, via "bin" (non-front_only only)
QKVZ_LOADS = 14  # in_proj_qkvz weights issued (non-front_only only)
QKVZ_DONE = 15  # both stage_qkvz(j) row groups done
BA_DONE = 16  # G.stage_ba done (in_proj_ba CTAs)
QUANTX_DONE = 18  # quant_x(c, H) done, before gdn_tasks
GDN_CONV_LO, GDN_CONV_HI = 20, 21  # gdn.gdn_conv, via "gdn_conv"
GDN_GATES_LO, GDN_GATES_HI = 22, 23  # gdn.gdn_gates, via "gdn_gates"
GDN_RECUR_DONE = 24  # gdn.gdn_recur done, via "gdn_recur" (lo unused, None)
GDN_NORM_LO, GDN_NORM_HI = 26, 27  # gdn.gdn_norm, via "gdn_norm"
EXIT = 31

SLOTS = {
    "entry": ENTRY,
    "msq_self_done": MSQ_SELF_DONE,
    "msq_wait_lo": MSQ_LO,
    "msq_wait_hi": MSQ_HI,
    "hc_down_loads": HC_DOWN_LOADS,
    "hc_down_done": HC_DOWN_DONE,
    "lora_wait_lo": LORA_LO,
    "lora_wait_hi": LORA_HI,
    "hc_up_done": HC_UP_DONE,
    "bin_wait_lo": BIN_LO,
    "bin_wait_hi": BIN_HI,
    "qkvz_loads": QKVZ_LOADS,
    "qkvz_done": QKVZ_DONE,
    "ba_done": BA_DONE,
    "quantx_done": QUANTX_DONE,
    "gdn_conv_wait_lo": GDN_CONV_LO,
    "gdn_conv_wait_hi": GDN_CONV_HI,
    "gdn_gates_wait_lo": GDN_GATES_LO,
    "gdn_gates_wait_hi": GDN_GATES_HI,
    "gdn_recur_done": GDN_RECUR_DONE,
    "gdn_norm_wait_lo": GDN_NORM_LO,
    "gdn_norm_wait_hi": GDN_NORM_HI,
    "exit": EXIT,
}


def _wslot(key: K1Build) -> dict:
    """``c["wslot"]``: every named wait site -> its (before, after) slot pair,
    built once a launch (only when ``key.trace``)."""
    return {
        "msq": (MSQ_LO, MSQ_HI),
        "lora": (LORA_LO, LORA_HI),
        "bin": (BIN_LO, BIN_HI),
        "gdn_conv": (GDN_CONV_LO, GDN_CONV_HI),
        "gdn_gates": (GDN_GATES_LO, GDN_GATES_HI),
        "gdn_recur": (None, GDN_RECUR_DONE),
        "gdn_norm": (GDN_NORM_LO, GDN_NORM_HI),
    }


def ba_tasks(lay):
    return lay.BA // ROWS * BA_PARTS


def gdn_tasks_of(lay):
    """GDN tasks a token."""
    return lay.NV * NVCH


def scratch_layout(key: K1Build) -> dict:
    s = key.tokens
    lay = L.Layout(key.tp)
    pairs = hc.regions(s)
    if not key.front_only:
        pairs = pairs | {
            "proj": s * lay.QKVZ,
            "qk": s * lay.NK * 2 * HD,
            "onrm": s * lay.NV * NVCH,
            "bapart": ba_tasks(lay) * s * ROWS,
        }
    out = pair_layout(pairs.items())
    if not key.front_only:
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
    """in_proj_qkvz rows 16 rg .. on the MXFP4-exact x rows -> bf16 PROJ."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    acc = fp4_gemv(lane, wave, c["xl"], XROW, H, s, rg, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS * s:
        r = tid % ROWS
        t = tid // ROWS
        v = bf16_round(row_sum(red, r, t))
        c["put"](c["proj"], t * c["lay"].QKVZ + rg * ROWS + r, v)
    gpu.barrier()


# ---------------------------------------------------------------- gdn
def _ba_of(c, t, h):
    """Head h's b and a of token t: the sums of in_proj_ba's K halves (bf16)."""
    return G.ba_sums(c, t, h, BA_PARTS)


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
    lay = L.Layout(key.tp)
    scr = scratch_layout(key)
    gt = gdn_tasks_of(lay)
    nr = -(-s * gt // BLOCKS)
    nba = ba_tasks(lay)
    need(BA0 + nba <= BLOCKS, "in_proj_ba's CTAs")
    need(lay.QKVZ // ROWS == 2 * BLOCKS, "in_proj_qkvz: two row groups a CTA")
    need(H % (BA_PARTS * gemv.KCH) == 0, "in_proj_ba's K halves")
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
        dbg: Int64,
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
            "dbg": dbg if key.trace else None,
            "nslot": NSLOT,
            "wslot": _wslot(key) if key.trace else {},
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
            },
        }
        if const_expr(not key.front_only):
            for region in ("proj", "qk", "onrm", "bapart"):
                c[region] = sreg(scratch, scr[region][0], region)
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
            "inj_out": lambda t, r, vs: _put_inj(c, t, r, vs),
        }
        stamps.stamp(c, ENTRY)
        # ---- the HC front: a CTA's whole sequence in one branch, its weights
        # (and its in_proj_qkvz ones) in flight first
        if const_expr(not key.front_only):
            qops = [qkvz_loads(c, bid + BLOCKS * j) for j in range(2)]
            stamps.stamp(c, QKVZ_LOADS)
        if const_expr(key.self_msq):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if bid < hc.NRED:
                hc.stage_msq(c, bid)
                stamps.stamp(c, MSQ_SELF_DONE)
        if bid < D0:
            uops = hc.up_loads(c, bid)
            hc.load_rrms(c)
            hc.stage_up(c, bid, uops)
            stamps.stamp(c, HC_UP_DONE)
        if (bid >= D0) & (bid < hc.UP_TASKS):
            dops = hc.down_loads(c, bid - D0)
            uops = hc.up_loads(c, bid)
            hc.load_rrms(c)
            stamps.stamp(c, HC_DOWN_LOADS)
            hc.stage_down(c, bid - D0, dops)
            stamps.stamp(c, HC_DOWN_DONE)
            hc.stage_up(c, bid, uops)
            stamps.stamp(c, HC_UP_DONE)
        if const_expr(key.front_only):
            if bid >= hc.UP_TASKS:
                dops = hc.down_loads(c, bid - D0)
                hc.load_rrms(c)
                stamps.stamp(c, HC_DOWN_LOADS)
                hc.stage_down(c, bid - D0, dops)
                stamps.stamp(c, HC_DOWN_DONE)
        else:
            is_ba = (bid >= BA0) & (bid < BA0 + nba)
            if is_ba:
                dops = hc.down_loads(c, bid - D0)
                bops = G.ba_loads(c, bid - BA0, BA_PARTS, BA_K)
                hc.load_rrms(c)
                stamps.stamp(c, HC_DOWN_LOADS)
                hc.stage_down(c, bid - D0, dops)
                stamps.stamp(c, HC_DOWN_DONE)
                hc.bin_to_lds(c, XROW)
                G.stage_ba(c, bid - BA0, bops, BA_PARTS, BA_K)
                stamps.stamp(c, BA_DONE)
            if bid >= BA0 + nba:
                dops = hc.down_loads(c, bid - D0)
                hc.load_rrms(c)
                stamps.stamp(c, HC_DOWN_LOADS)
                hc.stage_down(c, bid - D0, dops)
                stamps.stamp(c, HC_DOWN_DONE)
            # ---- in_proj on the block input, then the GDN tasks
            if ~is_ba:
                hc.bin_to_lds(c, XROW)
            quant_x(c, H)
            stamps.stamp(c, QUANTX_DONE)
            for j in range_constexpr(2):
                stage_qkvz(c, bid + BLOCKS * j, qops[j])
            stamps.stamp(c, QKVZ_DONE)
            gdn_tasks(c, nr)
        stamps.stamp(c, EXIT)

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
        dbg: Int64,
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
            dbg,
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
    dbg=None,
):
    """Host launcher of the HC front alone (``key.front_only``, a QSA layer):
    the attention HC's norm and mix of ``state`` [S, 4 H] -> ``bin_out`` [S, H]
    bf16 (the attention's input) and ``inj_out`` [S, 4] bf16. ``msq_in``:
    ``state``'s sums of squares as K2 wrote them, None with ``key.self_msq``.
    ``dbg``: Stage 0a's stamp buffer (``stamps.dbg_buffer(device, nslot=
    k1.NSLOT)``), required iff ``key.trace``; ignored (and may be omitted)
    otherwise."""
    s = key.tokens
    need(key.front_only, "hc_front takes a front_only build")
    _hc_checks(key, s, state, msq_in, hc_w, w_down, w_up, inj_out)
    need(
        bin_out.shape == (s, H)
        and bin_out.dtype == torch.bfloat16
        and bin_out.is_contiguous(),
        "bin_out: bf16 [S, H]",
    )
    need(key.trace == (dbg is not None), "dbg: required iff key.trace")
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
        "dbg": 0 if dbg is None else dbg.data_ptr(),
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
    dbg=None,
):
    """Host launcher of a GDN layer's K1; every weight as vLLM holds it after
    loading (in_proj_qkvz online MXFP4: the ASM FP4 GEMM's shuffled weight and
    e8m0 scales). ``conv_state`` [slots, CONV, 3] or a transposed view of
    [slots, 3, CONV] (strides read); ``rstate`` [slots, NV, 128, 128] fp32, a
    slot dense, slots strided; ``st_idx`` [>= S] int32, <= 0 a pad row;
    ``epoch`` a device int32 bumped once a step, ``layer`` this launch's tag
    slot. ``dbg``: Stage 0a's stamp buffer (``stamps.dbg_buffer(device,
    nslot=k1.NSLOT)``), required iff ``key.trace``; ignored (and may be
    omitted) otherwise."""
    s = key.tokens
    lay = L.Layout(key.tp)
    bf = torch.bfloat16
    need(not key.front_only, "gdn_pre takes a GDN build")
    need(key.trace == (dbg is not None), "dbg: required iff key.trace")
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
        "dbg": 0 if dbg is None else dbg.data_ptr(),
    }
    build(key)(*ABI.pack(vals), stream=torch.cuda.current_stream())
