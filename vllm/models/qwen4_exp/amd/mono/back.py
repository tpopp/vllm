# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``draft_back``: K2 (``k2.layer_post``) + the final mixer's HC front
(``k1.hc_front``, ``front_only``) + the draft head (``head.draft_head``)
fused into ONE persistent launch, ``BLOCKS`` x ``THREADS``, S <= 8 rows.

The three phases run on the SAME CTA grid, back to back, in program order;
every CTA passes through all three (a phase's own CTA-range ``if``s gate
whether it does anything), and the cross-phase hand-offs are the same tagged
mailbox / device-scope-store-then-flag primitives (``k2.py``, ``hc.py``,
``head.py``) already use between launches -- there is no additional
synchronization primitive here, only one shared Mailbox tag instead of three:

    K2      (``k2.k2_body``, unchanged): o_proj + the attention combine + the
             mlp HC mix + router / sgate, the routed experts, the mlp combine.
             Its own ``stage_final`` still materializes the next state into
             ``out`` (device-scope bf16), but instead of (or besides) a plain
             store into an external ``msq_out``, it publishes the sums of
             squares through the MIXER's own ``msq`` mailbox region
             (``c["msq_final_put"]``, set before ``k2_body`` runs).
    mixer   (``k1.front_body``, unchanged): the final mixer's HC front, its
             own ``c["hc"]`` (distinct ``msq`` / ``dpart`` / ``lora`` /
             ``binf`` scratch regions, so K2's own mlp-mix bookkeeping and the
             mixer's never alias): ``state`` is K2's ``out`` (CM_DEV),
             ``msq_in=None`` so ``hc.load_rrms`` polls the mailbox K2 just fed;
             ``bin`` is ``h_out`` directly (the mixer's own output tensor, no
             internal scratch copy needed) -- its own injection output is
             discarded (unused downstream, as the unfused 3-launch wiring
             already treats it).
    head    (``head.head_body``, unchanged): once the mixer's ``stage_up``
             (``binf`` set) has posted every flag, ``hc.bin_to_lds`` loads
             ``h_out`` into LDS (replacing ``head.py``'s own
             ``gemv.rows_to_lds`` from an external ``x``), then the shard's
             MXFP4 GEMV + TP-wide argmax -> ``ids``.

See ``k2.py`` / ``k1.py`` / ``head.py`` for each phase's own numerics; this
module only re-threads their already-reusable bodies through one shared
context and one combined scratch / peer layout.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import Int32, Int64

from vllm.models.kimi_k3.amd.mono.common.ops import (
    CM_DEV,
    ld_i32,
    load_ptr64,
    rsrc,
    uniform,
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
from vllm.models.qwen3_5.amd.mono import moe
from vllm.models.qwen3_5.amd.mono.moe import PS_MIN, step_tag
from vllm.models.qwen4_exp.amd.mono import hc, head, k1, k2
from vllm.models.qwen4_exp.amd.mono import layout as L
from vllm.models.qwen4_exp.amd.mono.layout import need
from vllm.models.qwen4_exp.amd.mono.sources import digest

H, HH, HC, ROWS = L.HIDDEN, L.HH, L.HC, L.ROWS
E, TOPK, KT = L.E, L.TOPK, L.TOPK + 1
XROW = k2.XROW  # H + gemv.LDS_PAD: the o_proj / BIN / x rows' shared LDS pitch

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class BackBuild:
    tokens: int
    tp: int = 2
    eps: float = 1e-6  # K2's own mlp HC norm
    mixer_eps: float = 1e-6  # the final mixer's HC norm
    oq: bool = False  # out_proj MXFP4 (ASM FP4 GEMM layout)
    og: bool = False  # o_proj's input is core x sigmoid(gate)
    rows: int = 124160  # the draft head's per-rank vocab shard, padding included
    want_msq_out: bool = False  # also plain-store msq_out (debug / parity tests)

    @property
    def u1(self) -> bool:
        """One row: U is the token's picks in top-k order (no union pass)."""
        return self.tokens == 1

    @property
    def ps(self) -> bool:
        """The fused shared expert's ug tasks run before the route."""
        return self.tokens >= PS_MIN


ABI = KernelAbi(
    (
        "core",
        "gate",
        "state",
        "inj_a",
        "w_o",
        "w_os",
        "hc_w",
        "w_down",
        "w_up",
        "w_gate",
        "w_sg",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "mix_hc_w",
        "mix_w_down",
        "mix_w_up",
        "head_w",
        "head_ws",
        "valid",
        "vocab_start",
        "out",
        "h_out",
        "ids",
        "msq_out",
        "scratch",
        "peers",
        "rank",
        "epoch",
        "slot",
    )
)


# ---------------------------------------------------------------- layout
def peer_pairs(s, tp):
    """One parity's peer regions, pairs: name -> count (K2's AR / FINAL
    combine regions and the head's TP-argmax region, at distinct offsets)."""
    return k2.peer_pairs(s, tp) | head.peer_pairs(s, tp)


def peer_layout(s, tp):
    return pair_layout(peer_pairs(s, tp).items())


def peer_bytes(tp):
    lay = peer_layout(L.MAX_TOKENS, tp)
    return 2 * max(o + n for o, n in lay.values())


def scratch_layout(key: BackBuild) -> dict:
    """K2's own regions (``hc.regions`` + its router / MoE ones) at their
    usual names; the mixer's own copy of ``hc.regions`` under ``mixer_*``
    names (distinct scratch, so neither phase's bookkeeping aliases the
    other's); the head's ``cta`` region."""
    s = key.tokens
    lay = L.Layout(key.tp)
    pairs = hc.regions(s) | {
        "inj": s * HC,
        "rlog": s * E // 2,
        "sgate": s,
        "route": s * 2 * TOPK,
        "inter": s * KT * lay.RI,
    }
    pairs |= {f"mixer_{name}": n for name, n in hc.regions(s).items()}
    pairs |= {"cta": BLOCKS * s * 2}
    out = pair_layout(pairs.items())
    end = max(o + n for o, n in out.values())
    for name, n in (("state1", s * HH * 2), ("bin", hc.bin_bytes(s))):
        end = (end + 15) // 16 * 16
        out[name] = (end, n)
        end += n
    return out


def scratch_bytes(key: BackBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def _msq_final_put(c, key, mixer_msq, jj, t, k, v):
    """K2's own ``stage_final`` sums-of-squares sink: publish through the
    mixer's own ``msq`` mailbox region, and (``key.want_msq_out``, a build-
    time flag) also a plain store into ``msq_out``, for debug / parity
    tests -- production never reads it, since the mixer gets them through
    the mailbox instead."""
    c["put"](mixer_msq, (jj * c["S"] + t) * HC + k, v)
    if key.want_msq_out:
        bo.buffer_store(v, rsrc(c["args"]["msq_out"]), (jj * c["S"] + t) * HC + k)


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: BackBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    tp = key.tp
    need(1 <= s <= L.MAX_TOKENS, f"{s} rows, the kernels take 1..{L.MAX_TOKENS}")
    need(
        key.rows % 32 == 0,
        "rows: the ASM FP4 GEMM's scale shuffle needs a 32-row shard",
    )
    lay = L.Layout(tp)
    need(k2.PS0 + lay.RI // moe.UG_COLS <= BLOCKS, "the shared expert's early ug CTAs")
    need(lay.RI % 32 == 0 and lay.RIP % L.FP4_STEP == 0, "the routed intermediate")
    scr = scratch_layout(key)
    groups = key.rows // ROWS
    # every width at the widest's offsets, as k2 / head do: a step's parity
    # half never overlaps the other half of a step of another width
    play = peer_layout(L.MAX_TOKENS, tp)
    half = peer_bytes(tp) // 2
    # the o_proj / BIN / x rows' shared LDS union, as k2.build's own xl: wide
    # enough for K2's o_proj (CORE), its down-projection's xn (DK) and the
    # mixer's / head's H-wide rows, whichever is widest
    xwords = s * (max(lay.CORE, H, hc.DK) + gemv.LDS_PAD) // 2

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * KT, 16]
        flag: fx.Array[fx.Int32, E, 16]
        expert: fx.Array[fx.Int32, moe.MAX_U, 16]
        kof: fx.Array[fx.Int32, moe.MAX_U * 8, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    @fx.struct
    class XLds:
        xl: fx.Array[fx.Int32, xwords, 16]

    @fx.struct
    class DownLds:
        il: fx.Array[fx.Int32, s * KT * (moe.irow(lay.moe) // 4 + lay.RIP // 32), 16]
        contrib: fx.Array[fx.Float32, s * KT * ROWS, 16]

    @fx.union
    class StageLds:
        x: XLds
        down: DownLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        rs: fx.Array[fx.Float32, L.MAX_TOKENS * HC, 16]
        ll: fx.Array[fx.Int32, s * hc.LROW // 2, 16]

    keyed = key_tuple(key, digest())
    name = f"flashnext_mono_back_s{s}_t{tp}_o{int(key.oq)}_a{int(key.og)}_r{key.rows}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def back(
        core: Int64,
        gate: Int64,
        state: Int64,
        inj_a: Int64,
        w_o: Int64,
        w_os: Int64,
        hc_w: Int64,
        w_down: Int64,
        w_up: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        mix_hc_w: Int64,
        mix_w_down: Int64,
        mix_w_up: Int64,
        head_w: Int64,
        head_ws: Int64,
        valid: Int32,
        vocab_start: Int32,
        out: Int64,
        h_out: Int64,
        ids: Int64,
        msq_out: Int64,
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
        rl = alloc.allocate(RouteLds).peek()
        st_lds = alloc.allocate(StageLds)
        xl, dl = st_lds.x.peek(), st_lds.down.peek()
        # one Mailbox tag for the whole launch: every phase's regions are
        # distinct scratch / peer addresses, so they need no tag of their own
        mb = Mailbox(step_tag(epoch, slot) - 1)
        par = uniform(ld_i32(epoch, 0)) % 2
        bases = peer_bases(peers, tp)
        own = load_ptr64(peers, rank)
        peer = {}
        for region in ("ar", "final"):
            off = play[region][0]
            peer[region] = [
                shift(preg(b, off, region), fx.Int64(par) * half) for b in bases
            ]
            peer[region + "_own"] = shift(preg(own, off, region), fx.Int64(par) * half)
        hoff = play["head"][0]
        peer_head = [shift(preg(b, hoff, "head"), fx.Int64(par) * half) for b in bases]
        peer_head_own = shift(preg(own, hoff, "head"), fx.Int64(par) * half)

        c = {
            "S": s,
            "TP": tp,
            "lay": lay,
            "moe": lay.moe,
            "route_cta": k2.RT0,
            "tid": tid,
            "bid": bid,
            "lane": lane,
            "wave": wave,
            "rank": rank,
            "eps": key.eps,
            "fse": True,
            "oq": key.oq,
            "og": key.og,
            "u1": key.u1,
            "ps": key.ps,
            "KT": KT,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "rs": lds.rs.ptr,
            "ll": lds.ll.ptr,
            "peer": peer,
            "xl": xl.xl.ptr,
            "il": dl.il.ptr,
            "contrib": dl.contrib.ptr,
            "rt": {
                "route": rl.route.ptr,
                "flag": rl.flag.ptr,
                "expert": rl.expert.ptr,
                "kof": rl.kof.ptr,
                "scan": rl.scan.ptr,
            },
            "state1": scratch + fx.Int64(scr["state1"][0]),
            "args": {
                "core": core,
                "gate": gate,
                "state": state,
                "inj_a": inj_a,
                "w_o": w_o,
                "w_os": w_os,
                "w_gate": w_gate,
                "w_sg": w_sg,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "out": out,
                "msq_out": msq_out,
            },
        }
        for region in ("inj", "rlog", "sgate", "route", "inter"):
            c[region] = sreg(scratch, scr[region][0], region)
        rlog = c["rlog"]
        c["rlog_at"] = lambda t, lane, i: (rlog, (t * E + lane + 64 * i) // 2, 1)
        c["rlog_last"] = (1, lambda i: (rlog, (s * E - 2) // 2, 1))
        mixer_msq = sreg(scratch, scr["mixer_msq"][0], "mixer_msq")
        c["hc"] = {
            "state": c["state1"],
            "state_cm": CM_DEV,
            "w_norm": hc_w,
            "w_down": w_down,
            "w_up": w_up,
            "msq": sreg(scratch, scr["msq"][0], "msq"),
            "msq_in": None,
            "dpart": sreg(scratch, scr["dpart"][0], "dpart"),
            "lora": sreg(scratch, scr["lora"][0], "lora"),
            "binf": sreg(scratch, scr["binf"][0], "binf"),
            "bin": scratch + fx.Int64(scr["bin"][0]),
            "inj_out": lambda t, r, vs: k2._put_inj(c, t, r, vs),
        }
        # the mixer phase reads ``out`` on other XCDs: device-scope stores
        c["out_cm"] = CM_DEV
        c["msq_final_put"] = lambda jj, t, k, v: _msq_final_put(
            c, key, mixer_msq, jj, t, k, v
        )

        # ---- phase 1: K2's own body (o_proj, the attention combine, the mlp
        # HC mix, router / sgate, the routed experts, the mlp combine);
        # stage_final materializes ``out`` and feeds the mixer's own msq
        k2.k2_body(c)

        # ---- phase 2: the final mixer's HC front. ``out`` is K2's just-
        # materialized state (CM_DEV); msq_in=None polls the mailbox K2 just
        # fed (c["msq_final_put"], above); its own injection output is
        # unused, as the unfused 3-launch wiring already discards it
        c["eps"] = key.mixer_eps
        c["self_msq"] = False
        c["hc"] = {
            "state": out,
            "state_cm": CM_DEV,
            "w_norm": mix_hc_w,
            "w_down": mix_w_down,
            "w_up": mix_w_up,
            "msq": mixer_msq,
            "msq_in": None,
            "dpart": sreg(scratch, scr["mixer_dpart"][0], "mixer_dpart"),
            "lora": sreg(scratch, scr["mixer_lora"][0], "mixer_lora"),
            "binf": sreg(scratch, scr["mixer_binf"][0], "mixer_binf"),
            "bin": h_out,
            "inj_out": lambda t, r, vs: None,
        }
        k1.front_body(c)

        # ---- phase 3: the draft head. the mixer's up tasks (``binf`` set)
        # posted BIN (``h_out``) and every flag; load it into LDS the same
        # way a later layer's front would, then run the shard GEMV + argmax
        hc.bin_to_lds(c, XROW)
        c["valid"] = valid
        c["vocab_start"] = vocab_start
        c["cta"] = sreg(scratch, scr["cta"][0], "cta")
        c["peer_head"] = peer_head
        c["peer_head_own"] = peer_head_own
        c["args"] = {"w": head_w, "ws": head_ws, "out": ids}
        head.head_body(c, groups)

    @flyc.jit
    def launch(
        core: Int64,
        gate: Int64,
        state: Int64,
        inj_a: Int64,
        w_o: Int64,
        w_os: Int64,
        hc_w: Int64,
        w_down: Int64,
        w_up: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        mix_hc_w: Int64,
        mix_w_down: Int64,
        mix_w_up: Int64,
        head_w: Int64,
        head_ws: Int64,
        valid: Int32,
        vocab_start: Int32,
        out: Int64,
        h_out: Int64,
        ids: Int64,
        msq_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        back(
            core,
            gate,
            state,
            inj_a,
            w_o,
            w_os,
            hc_w,
            w_down,
            w_up,
            w_gate,
            w_sg,
            w13,
            w13s,
            w2,
            w2s,
            mix_hc_w,
            mix_w_down,
            mix_w_up,
            head_w,
            head_ws,
            valid,
            vocab_start,
            out,
            h_out,
            ids,
            msq_out,
            scratch,
            peers,
            rank,
            epoch,
            slot,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(back, launch)
    _BUILDS[key] = launch
    return launch


# ---------------------------------------------------------------- host launcher
def draft_back(
    key: BackBuild,
    *,
    core,
    state,
    inj_a,
    w_o,
    hc_w,
    w_down,
    w_up,
    w_gate,
    w_sg,
    w13,
    w13s,
    w2,
    w2s,
    mix_hc_w,
    mix_w_down,
    mix_w_up,
    head_w,
    head_ws,
    valid: int,
    vocab_start: int,
    out,
    h_out,
    ids,
    scratch,
    peers,
    rank: int,
    epoch,
    slot: int,
    w_os=None,
    gate=None,
    msq_out=None,
):
    """Host launcher: K2's ``layer_post`` + the final mixer's ``hc_front`` +
    the draft head's ``draft_head``, one launch. ``core`` / ``state`` /
    ``inj_a`` / ``w_o`` / ``w_os`` / ``gate`` / ``hc_w`` / ``w_down`` /
    ``w_up`` / ``w_gate`` / ``w_sg`` / ``w13`` / ``w13s`` / ``w2`` / ``w2s``:
    as ``k2.layer_post`` takes them (K2's own mlp HC and the routed experts).
    ``mix_hc_w`` / ``mix_w_down`` (already zero-padded to ``L.DOWN_ROWS``,
    as ``mono_draft.py``'s ``_mixer_w_down``) / ``mix_w_up``: the final
    mixer's HC weights. ``head_w`` / ``head_ws``: this rank's lm_head shard
    in the ASM FP4 GEMM layout, ``key.rows`` rows; ``valid`` / ``vocab_start``:
    as ``head.draft_head``. ``out`` [S, 4H] bf16: the materialized next
    state (K2's ``out``); ``h_out`` [S, H] bf16: the mixer's block input (a
    copy of what the unfused wiring calls ``h``); ``ids`` [S] int64: the
    greedy draft token ids, same on every rank. ``msq_out`` [20, S, 4] fp32,
    optional: K2's own sums of squares, written only when ``key.
    want_msq_out`` (debug / parity tests -- production never needs it, since
    the mixer reads them through the in-kernel mailbox instead). ``scratch``:
    device mailboxes (>= ``scratch_bytes(key)``). ``peers``: the int64 table
    of every rank's peer buffer (>= ``peer_bytes(key.tp)`` each). ``epoch`` a
    device int32 bumped once a step, ``slot`` this launch's tag slot."""
    s = key.tokens
    lay = L.Layout(key.tp)
    bf = torch.bfloat16
    fp4s = (torch.uint8, torch.float4_e2m1fn_x2)
    ne = E + 1
    k = lay.CORE
    need(core.shape == (s, k), f"core: [{s}, {k}]")
    need(state.shape == (s, HH) and out.shape == (s, HH), "state / out: [S, 4 H]")
    need(inj_a.shape == (s, HC) and inj_a.dtype == bf, "inj_a: bf16 [S, 4]")
    need(
        (gate is not None) == key.og and (gate is None or gate.shape == core.shape),
        "gate: core's shape, with og only",
    )
    if key.oq:
        need(
            w_o.shape == (H, k // 2) and w_o.dtype in fp4s,
            f"o_proj: fp4x2 [{H}, {k // 2}]",
        )
        need(
            w_os is not None
            and w_os.numel() == H * k // 32
            and w_os.element_size() == 1,
            "o_proj scales: e8m0, one a 32-block",
        )
    else:
        need(w_o.shape == (H, k) and w_o.dtype == bf, f"o_proj: bf16 [{H}, {k}]")
    need(hc_w.shape == (HH,), "K2's mlp HC norm weight: [4 H]")
    need(w_down.shape == (L.DOWN_ROWS, HH), "K2's mlp HC down: [336, 4 H]")
    need(w_up.shape == (HH, L.LORA), "K2's mlp HC up: [4 H, 320]")
    need(w_gate.shape == (E, H) and w_sg.numel() == H, "router shape")
    need(
        w13.shape == (ne, 2 * lay.RIP, H // 2) and w13.dtype in fp4s,
        f"w13: fp4x2 [{ne}, {2 * lay.RIP}, {H // 2}]",
    )
    need(
        w2.shape == (ne, H, lay.RIP // 2) and w2.dtype in fp4s,
        f"w2: fp4x2 [{ne}, {H}, {lay.RIP // 2}]",
    )
    need(
        w13s.numel() == ne * 2 * lay.RIP * QL.scale_cols(H)
        and w13s.element_size() == 1,
        "w13 scales: e8m0, one a 32-block, shuffled",
    )
    need(
        w2s.numel() == ne * H * QL.scale_cols(lay.RIP) and w2s.element_size() == 1,
        "w2 scales: e8m0, one a 32-block, shuffled (K blocks padded to 8)",
    )
    need(
        mix_hc_w.shape == (HH,) and mix_hc_w.dtype == bf,
        "mixer HC norm weight: [4 H]",
    )
    need(
        mix_w_down.shape == (L.DOWN_ROWS, HH) and mix_w_down.dtype == bf,
        "mixer HC down (zero-padded to DOWN_ROWS): [336, 4 H]",
    )
    need(
        mix_w_up.shape == (HH, L.LORA) and mix_w_up.dtype == bf,
        "mixer HC up: [4 H, 320]",
    )
    need(
        head_w.shape == (key.rows, H // 2) and head_w.dtype in fp4s,
        f"head_w: fp4x2 [{key.rows}, {H // 2}]",
    )
    need(
        head_ws.numel() == key.rows * H // 32 and head_ws.element_size() == 1,
        "head_ws: e8m0, one a 32-block, shuffled",
    )
    need(0 <= valid <= key.rows, "valid: rows of this shard that are real vocab")
    need(h_out.shape == (s, H) and h_out.dtype == bf, f"h_out: bf16 [{s}, {H}]")
    need(ids.shape == (s,) and ids.dtype == torch.int64, f"ids: int64 [{s}]")
    need(
        (msq_out is not None) == key.want_msq_out,
        "msq_out: given iff key.want_msq_out",
    )
    if msq_out is not None:
        need(
            msq_out.numel() == hc.NRED * s * HC and msq_out.dtype == torch.float32,
            "msq_out: fp32 [20, S, 4]",
        )
    used = [w_o, w_os, w_gate, w_sg, w13, w13s, w2, w2s, gate, msq_out]
    tensors = [
        core, state, inj_a, out, hc_w, w_down, w_up, mix_hc_w, mix_w_down,
        mix_w_up, head_w, head_ws, h_out, ids,
    ]  # fmt: skip
    tensors += [t for t in used if t is not None]
    need(all(t.is_contiguous() for t in tensors), "a non-contiguous input")
    dense = [
        core, state, out, hc_w, w_down, w_up, w_gate, w_sg,
        mix_hc_w, mix_w_down, mix_w_up,
    ]  # fmt: skip
    dense += [gate] if gate is not None else []
    need(all(t.dtype == bf for t in dense), "activations and dense weights: bf16")

    def ptr(t):
        return 0 if t is None else t.data_ptr()

    f = build(key)
    f(
        *ABI.pack(
            {
                "core": core.data_ptr(),
                "gate": ptr(gate),
                "state": state.data_ptr(),
                "inj_a": inj_a.data_ptr(),
                "w_o": w_o.data_ptr(),
                "w_os": ptr(w_os) if key.oq else 0,
                "hc_w": hc_w.data_ptr(),
                "w_down": w_down.data_ptr(),
                "w_up": w_up.data_ptr(),
                "w_gate": w_gate.data_ptr(),
                "w_sg": w_sg.data_ptr(),
                "w13": w13.data_ptr(),
                "w13s": w13s.data_ptr(),
                "w2": w2.data_ptr(),
                "w2s": w2s.data_ptr(),
                "mix_hc_w": mix_hc_w.data_ptr(),
                "mix_w_down": mix_w_down.data_ptr(),
                "mix_w_up": mix_w_up.data_ptr(),
                "head_w": head_w.data_ptr(),
                "head_ws": head_ws.data_ptr(),
                "valid": valid,
                "vocab_start": vocab_start,
                "out": out.data_ptr(),
                "h_out": h_out.data_ptr(),
                "ids": ids.data_ptr(),
                "msq_out": ptr(msq_out),
                "scratch": scratch.data_ptr(),
                "peers": peers.data_ptr(),
                "rank": rank,
                "epoch": epoch.data_ptr(),
                "slot": slot,
            }
        ),
        stream=torch.cuda.current_stream(),
    )
