# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the Qwen3.8-Flash-Next mono decode kernels on MI355X against the
ops vLLM runs for the same decode step, random weights in the checkpoint's
formats at TP2 shapes.

- The HC front (one GPU): ``GatedResidual.mix`` (grouped Gemma norm, the
  down + inject GEMM, hc_silu, the up GEMM, the gate-mix) on a materialized
  state, from K2's sums of squares or its own.
- K1 (one GPU): the front -> in_proj_qkvz (online MXFP4) / in_proj_ba ->
  AITER's conv update and fused gated delta rule -> RMSNormGated, the states
  it leaves behind, pad rows.
- K2 (two GPUs): out_proj / o_proj + all-reduce + the attention HC combine,
  the MLP HC's mix, the sparse MoE block (softmax top-10, the MXFP4 experts
  with the intermediate padded to 128, the fused shared expert) + all-reduce +
  the MLP HC combine, and the sums of squares it hands the next front.
- Graph replay (two GPUs): a GDN and a QSA layer at mixed widths in one HIP
  graph, replayed bit-identically.
"""

import pytest
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.models.qwen3_5.test_amd_mono import (
    COS_MIN,
    REPLAY_WIDTHS,
    REPLAYS,
    STOCK_COS_MIN,
    _all_reduce_fp32,
    _check_route,
    _cos,
    _gdn_core_ref,
    _moe_emulation,
    _moe_reference,
    _mxfp4_dequant,
    _mxfp4_linear,
    _mxfp4_linear_stock,
    _mxfp4_qdq,
    _on_gfx950,
    _pairs,
    _randn,
    _rel_l2,
    _route_and_shared,
    _u8,
)
from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)

bf = torch.bfloat16
TP = 2

pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="the Flash-Next mono kernels need gfx950 (MI355X)"
)


# ---------------------------------------------------------------- HC
def _hc_weights(g):
    from vllm.models.qwen4_exp.amd.mono.layout import DOWN_ROWS, HH, LORA

    return {
        "hc_w": _randn(g, HH, scale=0.1),
        "w_down": _randn(g, DOWN_ROWS, HH, scale=0.01),
        "w_up": _randn(g, HH, LORA, scale=0.05),
    }


def _msq(state):
    """K2's ``msq_out`` of a state: [20, S, 4] fp32 sums of squares by 128
    columns of each stream."""
    from vllm.models.qwen4_exp.amd.mono import hc
    from vllm.models.qwen4_exp.amd.mono.layout import HC

    s = state.size(0)
    sq = state.float().view(s, HC, hc.NRED, hc.COLS).pow(2).sum(-1)
    return sq.permute(2, 0, 1).contiguous()


def _mix_ref(state, hw):
    """``GatedResidual.mix`` on ``state`` -> (block input, injection, xn)."""
    from vllm.models.qwen4_exp.amd.mono.layout import HC, LORA
    from vllm.models.qwen4_exp.amd.ops.hc import (
        grouped_gemma_rmsnorm,
        hc_gate_mix,
        hc_silu,
    )

    xn = grouped_gemma_rmsnorm(state, hw["hc_w"], 1e-6, HC)
    d = F.linear(xn, hw["w_down"])
    lora = hc_silu(d[:, :LORA].contiguous(), HC)
    gate = F.linear(lora, hw["w_up"])
    return hc_gate_mix(xn, gate, HC), d[:, LORA : LORA + HC].contiguous(), xn


@pytest.mark.parametrize("s", [1, 3, 8])
@pytest.mark.parametrize("self_msq", [False, True])
@torch.inference_mode()
def test_hc_front_matches_vllm_ops(s: int, self_msq: bool) -> None:
    """The QSA layers' HC front (``hc_front``) against ``GatedResidual.mix``:
    the block input and the injection logits."""
    from vllm.models.qwen4_exp.amd.mono import k1
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HH, HIDDEN

    g = torch.Generator(device="cuda").manual_seed(s)
    state = _randn(g, s, HH, scale=2.0)
    hw = _hc_weights(g)
    want_x, want_inj, _ = _mix_ref(state, hw)
    key = k1.K1Build(tokens=s, tp=TP, self_msq=self_msq, front_only=True)
    x = torch.full((s, HIDDEN), float("nan"), dtype=bf, device="cuda")
    inj = torch.full((s, HC), float("nan"), dtype=bf, device="cuda")
    scratch = torch.zeros(k1.scratch_bytes(key), dtype=torch.uint8, device="cuda")
    k1.hc_front(
        key,
        state=state,
        msq_in=None if self_msq else _msq(state),
        **hw,
        inj_out=inj,
        bin_out=x,
        scratch=scratch,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        layer=0,
    )
    torch.accelerator.synchronize()
    assert _cos(inj, want_inj) > COS_MIN
    torch.testing.assert_close(inj, want_inj, atol=2e-2, rtol=2e-2)
    assert _cos(x, want_x) > COS_MIN
    torch.testing.assert_close(x, want_x, atol=2e-2, rtol=2e-2)


def _front_ref(emb, prev, w_ne, w_nh, w_fe, w_fh, ne_eps, nh_eps):
    """``pre_fc_norm_embedding`` -> ``fc_embedding``, ``pre_fc_norm_hidden``
    (one norm over all 4 H, not one a stream) -> ``fc_hidden`` (one weight
    shared across the 4 streams), the embedding added as a residual to every
    stream: the MTP draft's front, stock ``GemmaRMSNorm`` + plain matmuls, as
    ``Qwen4ExpMultiTokenPredictor.forward`` computes it."""
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HIDDEN

    def gemma(x, w, eps):
        # GemmaRMSNorm.forward_native: fp32, x rrms (1 + w), back to bf16
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (xf * (1.0 + w.float())).to(x.dtype)

    s = emb.size(0)
    fe = F.linear(gemma(emb, w_ne, ne_eps), w_fe)
    hn = gemma(prev, w_nh, nh_eps).view(s, HC, HIDDEN)
    fh = F.linear(hn, w_fh)
    return (fe.unsqueeze(1) + fh).flatten(1).contiguous()


@pytest.mark.parametrize("s", [1, 3, 8])
@torch.inference_mode()
def test_draft_front_matches_vllm_ops(s: int) -> None:
    """``draft_front``: the MTP draft step's ``fc_embedding`` / ``fc_hidden``
    projections folded into the attention HC front, one launch, against the
    stock torch ops (``_front_ref``) feeding ``_mix_ref`` -- the same
    reference ``test_hc_front_matches_vllm_ops`` uses for the HC front
    itself. Also times the fused launch against the stock sequence + K1's
    separate ``hc_front`` launch (what ``draft_front`` replaces)."""
    from vllm.models.qwen4_exp.amd.mono import front, k1
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HH, HIDDEN

    g = torch.Generator(device="cuda").manual_seed(s)
    emb = _randn(g, s, HIDDEN, scale=2.0)
    prev = _randn(g, s, HH, scale=2.0)
    w_ne = _randn(g, HIDDEN, scale=0.1)
    w_nh = _randn(g, HH, scale=0.1)
    w_fe = _randn(g, HIDDEN, HIDDEN, scale=0.05)
    w_fh = _randn(g, HIDDEN, HIDDEN, scale=0.05)
    hw = _hc_weights(g)
    # distinct from each other and from _mix_ref's hardcoded hc eps (1e-6):
    # a kernel that swapped ne_eps / nh_eps would still pass with equal values
    ne_eps, nh_eps = 1e-5, 3e-6

    want_state = _front_ref(emb, prev, w_ne, w_nh, w_fe, w_fh, ne_eps, nh_eps)
    want_x, want_inj, _ = _mix_ref(want_state, hw)

    key = front.FrontBuild(tokens=s, tp=TP, ne_eps=ne_eps, nh_eps=nh_eps)
    state = torch.full((s, HH), float("nan"), dtype=bf, device="cuda")
    x = torch.full((s, HIDDEN), float("nan"), dtype=bf, device="cuda")
    inj = torch.full((s, HC), float("nan"), dtype=bf, device="cuda")
    scratch = torch.zeros(front.scratch_bytes(key), dtype=torch.uint8, device="cuda")
    epoch = torch.ones(1, dtype=torch.int32, device="cuda")

    def _fused():
        epoch.add_(1)
        front.draft_front(
            key,
            emb=emb,
            prev=prev,
            w_ne=w_ne,
            w_nh=w_nh,
            w_fe=w_fe,
            w_fh=w_fh,
            **hw,
            state_out=state,
            bin_out=x,
            inj_out=inj,
            scratch=scratch,
            epoch=epoch,
            slot=0,
        )

    _fused()
    torch.accelerator.synchronize()
    assert _cos(state, want_state) > COS_MIN
    torch.testing.assert_close(state, want_state, atol=2e-2, rtol=2e-2)
    assert _cos(inj, want_inj) > COS_MIN
    torch.testing.assert_close(inj, want_inj, atol=2e-2, rtol=2e-2)
    assert _cos(x, want_x) > COS_MIN
    torch.testing.assert_close(x, want_x, atol=2e-2, rtol=2e-2)

    # ---- timing: the fused launch vs the stock sequence + k1's separate
    # HC front launch
    k1key = k1.K1Build(tokens=s, tp=TP, self_msq=True, front_only=True)
    k1scratch = torch.zeros(k1.scratch_bytes(k1key), dtype=torch.uint8, device="cuda")

    def _stock():
        epoch.add_(1)
        state_ref = _front_ref(emb, prev, w_ne, w_nh, w_fe, w_fh, ne_eps, nh_eps)
        k1.hc_front(
            k1key,
            state=state_ref,
            msq_in=None,
            **hw,
            inj_out=inj,
            bin_out=x,
            scratch=k1scratch,
            epoch=epoch,
            layer=0,
        )

    for fn in (_stock, _fused):
        for _ in range(3):
            fn()
    torch.accelerator.synchronize()

    reps = 20
    e0, e1, e2, e3 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
    e0.record()
    for _ in range(reps):
        _stock()
    e1.record()
    e2.record()
    for _ in range(reps):
        _fused()
    e3.record()
    torch.accelerator.synchronize()
    stock_us = e0.elapsed_time(e1) * 1000 / reps
    fused_us = e2.elapsed_time(e3) * 1000 / reps
    print(
        f"\ndraft_front timing (s={s}): stock {stock_us:.1f} us, "
        f"fused {fused_us:.1f} us"
    )


# ---------------------------------------------------------------- K1
def _gdn_inputs(s: int, n_real: int, slots: int = 16):
    from vllm.models.qwen4_exp.amd.mono.layout import HD, HH, HIDDEN, Layout

    lay = Layout(TP)
    g = torch.Generator(device="cuda").manual_seed(0)
    w = {
        "state": _randn(g, s, HH, scale=2.0),
        **_hc_weights(g),
        "w_ba": _randn(g, lay.BA, HIDDEN, scale=0.05),
        "conv_w": _randn(g, lay.CONV, 1, 4, scale=0.4),
        "a_log": _randn(g, lay.NV, scale=0.5, dtype=torch.float32),
        "dt_bias": _randn(g, lay.NV, scale=0.5),
        "norm_w": (1 + _randn(g, HD, scale=0.1, dtype=torch.float32)).to(bf),
    }
    q, sc, w["w_qkvz"], w["w_qkvzs"] = _mxfp4_linear(g, lay.QKVZ, HIDDEN)
    w["w_qkvz_ref"] = _mxfp4_dequant(q, sc).to(bf)
    # vLLM packs a layer's conv and SSM state in one padded page: strided slots
    n = lay.NV * HD * HD
    pad = _randn(g, slots, n + 8192, scale=0.05, dtype=torch.float32)
    w["rstate"] = pad[:, :n].view(slots, lay.NV, HD, HD)
    conv = _randn(g, slots, lay.CONV * 3 + 1024)[:, : lay.CONV * 3]
    w["conv_state"] = conv.view(slots, 3, lay.CONV).transpose(-1, -2)
    # live rows on slots 3 ..; pad rows on the null block 0
    idx = torch.zeros(s, dtype=torch.int32, device="cuda")
    idx[:n_real] = torch.arange(3, 3 + n_real, dtype=torch.int32, device="cuda")
    w["st_idx"] = idx
    return w


def _gdn_reference(w, n: int, qkvz=None, ba=None):
    """The attention HC's mix, then the GDN decode path on ROCm (MXFP4
    in_proj_qkvz, AITER's conv update and fused gated delta rule,
    RMSNormGated) on the first ``n`` rows; the rest are the cudagraph pad.
    ``qkvz`` / ``ba``: in_proj's outputs to start from."""
    from vllm.models.qwen4_exp.amd.mono.layout import Layout

    lay = Layout(TP)
    x, inj, _ = _mix_ref(w["state"], w)
    if qkvz is None:
        qkvz = F.linear(_mxfp4_qdq(x).to(bf), w["w_qkvz_ref"])
        ba = F.linear(x, w["w_ba"])
    conv_state, rstate, core = _gdn_core_ref(w, n, qkvz, ba, lay.NK, lay.NV, "sigmoid")
    return {
        "x": x,
        "inj": inj,
        "qkvz": qkvz,
        "ba": ba,
        "conv_state": conv_state,
        "rstate": rstate,
        "core": core,
    }


@pytest.mark.parametrize(
    "s,n_real,self_msq", [(1, 1, False), (5, 4, False), (8, 8, False), (8, 7, True)]
)
@torch.inference_mode()
def test_k1_matches_vllm_ops(s: int, n_real: int, self_msq: bool) -> None:
    """K1 against the HC mix -> in_proj -> AITER's conv update + fused gated
    delta rule -> RMSNormGated on one decode step, the states it leaves behind,
    and the pad rows (null block: no state written, zero core)."""
    from vllm.models.qwen4_exp.amd.mono import k1
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HIDDEN, ROWS, Layout

    lay = Layout(TP)
    w = _gdn_inputs(s, n_real)
    ref = _gdn_reference(w, n_real)
    stock = _mxfp4_linear_stock(ref["x"], w["w_qkvz"], w["w_qkvzs"])
    assert _cos(stock, ref["qkvz"]) > STOCK_COS_MIN
    key = k1.K1Build(tokens=s, tp=TP, self_msq=self_msq)
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    core = torch.full((s, lay.CORE), float("nan"), dtype=bf, device="cuda")
    inj = torch.full((s, HC), float("nan"), dtype=bf, device="cuda")
    scratch = torch.zeros(k1.scratch_bytes(key), dtype=torch.uint8, device="cuda")
    k1.gdn_pre(
        key,
        state=w["state"],
        msq_in=None if self_msq else _msq(w["state"]),
        hc_w=w["hc_w"],
        w_down=w["w_down"],
        w_up=w["w_up"],
        inj_out=inj,
        w_qkvz=w["w_qkvz"],
        w_qkvzs=w["w_qkvzs"],
        w_ba=w["w_ba"],
        conv_w=w["conv_w"],
        conv_state=conv_state,
        a_log=w["a_log"],
        dt_bias=w["dt_bias"],
        norm_w=w["norm_w"],
        rstate=rstate,
        st_idx=w["st_idx"],
        core=core,
        scratch=scratch,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        layer=0,
    )
    torch.accelerator.synchronize()

    scr = k1.scratch_layout(key)
    off, n = scr["bin"]
    x = scratch[off : off + n].view(bf).view(s, HIDDEN)
    assert _cos(x, ref["x"]) > COS_MIN
    torch.testing.assert_close(inj, ref["inj"], atol=2e-2, rtol=2e-2)
    qkvz = _pairs(scratch, scr["proj"], s * lay.QKVZ).view(s, lay.QKVZ)
    # b and a: the K halves summed in order, as the gates do
    parts = _pairs(scratch, scr["bapart"], k1.ba_tasks(lay) * s * ROWS)
    parts = parts.view(lay.BA // ROWS, k1.BA_PARTS, s, ROWS)
    ba = parts[:, 0]
    for q in range(1, k1.BA_PARTS):
        ba = ba + parts[:, q]
    ba = ba.permute(1, 0, 2).reshape(s, lay.BA).to(bf)
    # in_proj on the kernel's own block input
    want_qkvz = F.linear(_mxfp4_qdq(x).to(bf), w["w_qkvz_ref"])
    assert _cos(qkvz, want_qkvz) > COS_MIN
    assert _cos(ba, F.linear(x, w["w_ba"])) > COS_MIN
    live = w["st_idx"][:n_real].long()

    def errs_vs(r):
        return {
            "conv_state": _rel_l2(conv_state[live], r["conv_state"][live]),
            "rstate": _rel_l2(rstate[live], r["rstate"][live]),
            "core": _rel_l2(core[:n_real], r["core"][:n_real]),
        }

    # from K1's own in_proj output: the conv, gates, recurrence and norm alone
    own = errs_vs(_gdn_reference(w, n_real, qkvz=qkvz.to(bf), ba=ba))
    assert own["conv_state"] == 0 and own["rstate"] < 1e-6, own
    assert own["core"] < 1e-4, own
    # end to end: the HC GEMVs' and in_proj's summation order flips a few bf16
    # roundings, an x an ulp off the reference's can land in another e2m1 bin
    errs = errs_vs(ref)
    assert errs["conv_state"] < 5e-3 and errs["rstate"] < 5e-3, errs
    assert _cos(core[:n_real], ref["core"][:n_real]) > COS_MIN, errs
    assert torch.equal(core[n_real:], torch.zeros_like(core[n_real:]))
    # the null block is left alone
    torch.testing.assert_close(rstate[0], w["rstate"][0], atol=0, rtol=0)
    torch.testing.assert_close(conv_state[0], w["conv_state"][0], atol=0, rtol=0)


# ---------------------------------------------------------------- K2
def _pad_halves(x, ri, rip, fill):
    """[ne, 2 ri, ...] [gate; up] rows -> [ne, 2 rip, ...], each half padded."""
    ne = x.shape[0]
    out = torch.full((ne, 2 * rip, *x.shape[2:]), fill, dtype=x.dtype, device=x.device)
    out[:, :ri] = x[:, :ri]
    out[:, rip : rip + ri] = x[:, ri:]
    return out


def _moe_weights(rank: int, device):
    """One layer's MoE block at TP2 as vLLM holds it after loading the fused
    shared expert as expert 512: bf16 router and shared-expert gate; the
    experts' MXFP4 with the intermediate zero-padded from RI to RIP (the pad
    scales e8m0 1.0), through ``shuffle_weights`` and ``e8m0_shuffle``."""
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.qwen4_exp.amd.mono.layout import HIDDEN, E, Layout

    lay = Layout(TP)
    ri, rip, ne = lay.RI, lay.RIP, E + 1
    gc = torch.Generator(device=device).manual_seed(7)  # replicated weights
    gr = torch.Generator(device=device).manual_seed(100 + rank)  # TP shards
    fp4, e8 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    w13q, w2q = _u8(gr, ne, 2 * ri, HIDDEN // 2), _u8(gr, ne, HIDDEN, ri // 2)
    w13s = _u8(gr, ne, 2 * ri, HIDDEN // 32, lo=118, hi=123)
    w2s = _u8(gr, ne, HIDDEN, ri // 32, lo=118, hi=123)
    raw = {"w13": w13q, "w2": w2q, "w13s": w13s, "w2s": w2s}
    w13p = _pad_halves(w13q, ri, rip, 0)
    w13sp = _pad_halves(w13s, ri, rip, 127)
    w2p = torch.zeros(ne, HIDDEN, rip // 2, dtype=torch.uint8, device=device)
    w2p[..., : ri // 2] = w2q
    w2sp = torch.full((ne, HIDDEN, rip // 32), 127, dtype=torch.uint8, device=device)
    w2sp[..., : ri // 32] = w2s
    w13, w2 = rocm_aiter_ops.shuffle_weights(w13p.view(fp4), w2p.view(fp4))
    return {
        "raw": raw,
        "w_gate": _randn(gc, E, HIDDEN, scale=0.02),
        "w_sg": _randn(gc, 1, HIDDEN, scale=0.02),
        "w13": w13,
        "w2": w2,
        "w13s": e8m0_shuffle(w13sp.view(ne * 2 * rip, -1).view(e8)).view(
            ne, 2 * rip, -1
        ),
        "w2s": e8m0_shuffle(w2sp.view(ne * HIDDEN, -1).view(e8)).view(ne, HIDDEN, -1),
    }


def _moe_exact(x, mw, ids):
    """The routed experts in fp32 on the dequantized weights, x and the
    intermediate unquantized, then the all-reduce."""
    from vllm.models.qwen4_exp.amd.mono.layout import Layout

    ri = Layout(TP).RI
    raw = mw["raw"]
    wts, ids, _ = _route_and_shared(x, mw, ids)
    routed = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = ids[t, k].item()
            gu = _mxfp4_dequant(raw["w13"][e], raw["w13s"][e]) @ x[t].float()
            h = F.silu(gu[:ri]) * gu[ri:]
            routed[t] += wts[t, k] * (_mxfp4_dequant(raw["w2"][e], raw["w2s"][e]) @ h)
    return _all_reduce_fp32(routed.to(bf))


@ray.remote(num_gpus=1, max_calls=1)
def _k2_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.qwen4_exp.amd.mono import k2
    from vllm.models.qwen4_exp.amd.mono.layout import (
        HC,
        HH,
        HIDDEN,
        MAX_TOKENS,
        Layout,
    )
    from vllm.models.qwen4_exp.amd.ops.hc import hc_combine

    lay = Layout(TP)
    tp = get_tp_group()
    peers = PeerBuffer(
        k2.peer_bytes(MAX_TOKENS, TP), tp.cpu_group, rank, tp_size, device
    )
    peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    mw = _moe_weights(rank, device)
    gc = torch.Generator(device=device).manual_seed(11)
    hw = _hc_weights(gc)
    step = 0
    for flags in K2_MODES:
        gr = torch.Generator(device=device).manual_seed(200 + rank)
        oq = flags.get("oq", False)
        if oq:
            oq_raw, os_raw, w_o, w_os = _mxfp4_linear(gr, HIDDEN, lay.CORE)
            w_o_ref = _mxfp4_dequant(oq_raw, os_raw).to(bf)
        else:
            w_o, w_os = _randn(gr, HIDDEN, lay.CORE, scale=0.02), None
        mode = str(flags)
        # both epoch parities, a layer slot each; 1 row (the picks as the
        # union), 4 and 8 (the fused shared expert's tasks before the route)
        for s in (1, 3, 4, 8, 8):
            step += 1
            epoch.add_(1)
            key = k2.K2Build(tokens=s, tp=TP, **flags)
            core = _randn(gr, s, lay.CORE)
            gate = _randn(gr, s, lay.CORE, scale=2.0) if key.og else None
            state = _randn(gc, s, HH, scale=2.0)
            inj_a = _randn(gc, s, HC)
            scratch = torch.zeros(
                k2.scratch_bytes(key), dtype=torch.uint8, device=device
            )
            out = torch.full((s, HH), float("nan"), dtype=bf, device=device)
            msq = torch.full((20, s, HC), float("nan"), device=device)
            k2.layer_post(
                key,
                core=core,
                gate=gate,
                state=state,
                inj_a=inj_a,
                w_o=w_o,
                w_os=w_os,
                **hw,
                out=out,
                msq_out=msq,
                scratch=scratch,
                peers=peers.addresses,
                rank=rank,
                epoch=epoch,
                layer=step,
                **{k: v for k, v in mw.items() if k != "raw"},
            )
            torch.accelerator.synchronize()

            if gate is not None:
                core = (core.float() * torch.sigmoid(gate.float())).to(bf)
            if oq:
                part = F.linear(_mxfp4_qdq(core).to(bf), w_o_ref)
                part_stock = _mxfp4_linear_stock(core, w_o, w_os)
                assert _cos(part_stock, part) > STOCK_COS_MIN, (mode, s)
            else:
                part = F.linear(core, w_o)
            state1_ref = hc_combine(state, _all_reduce_fp32(part), inj_a, HC)
            scr = k2.scratch_layout(key)
            off, n = scr["state1"]
            state1 = scratch[off : off + n].view(bf).view(s, HH)
            # the kernel sums the ranks in rank order, NCCL in its own order
            assert _rel_l2(state1, state1_ref) < 1e-2, (mode, s)
            # the MLP HC mix on the kernel's own state
            x_ref, inj_ref, _ = _mix_ref(state1.clone(), hw)
            off, n = scr["bin"]
            x = scratch[off : off + n].view(bf).view(s, HIDDEN)
            assert _cos(x, x_ref) > COS_MIN, (mode, s)
            inj_m = _pairs(scratch, scr["inj"], s * HC).view(s, HC).to(bf)
            torch.testing.assert_close(inj_m, inj_ref, atol=2e-2, rtol=2e-2)
            # the MoE on the kernel's own block input, where routing cannot differ
            ids = _check_route(scratch, scr["route"], x.clone(), mw)
            emu = _moe_emulation(x.clone(), mw, ids)
            want = hc_combine(state1.clone(), emu, inj_m, HC)
            c = _cos(out - state1, want - state1)
            assert c > COS_MIN, (mode, s, c)
            # fused_moe rounds x and the intermediate into other e2m1 bins than
            # the emulation (cosine 0.999 between the two): measured against
            # the unquantized MoE, K2 must be no further off than fused_moe
            ref = hc_combine(
                state1.clone(), _moe_reference(x.clone(), mw, ids), inj_m, HC
            )
            exact = hc_combine(
                state1.clone(), _moe_exact(x.clone(), mw, ids), inj_m, HC
            )
            err = _rel_l2(out - state1, exact - state1)
            err_stock = _rel_l2(ref - state1, exact - state1)
            assert err < 1.02 * err_stock, (mode, s, err, err_stock)
            torch.testing.assert_close(msq, _msq(out), rtol=1e-5, atol=1e-3)
            outs = [torch.empty_like(out) for _ in range(tp_size)]
            dist.all_gather(outs, out)
            assert all(torch.equal(outs[0], o) for o in outs), (mode, "ranks disagree")
    peers.close()


# the GDN layers' MXFP4 out_proj; the QSA layers' bf16 o_proj with the gate
K2_MODES: tuple[dict[str, bool], ...] = ({"oq": True}, {"og": True})


@multi_gpu_test(num_gpus=2)
def test_k2_tp2_matches_vllm_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 2, 1, _k2_worker)


# ---------------------------------------------------------------- replay
@ray.remote(num_gpus=1, max_calls=1)
def _replay_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.qwen4_exp.amd.mono import hc, k1, k2
    from vllm.models.qwen4_exp.amd.mono.layout import (
        HC,
        HH,
        HIDDEN,
        MAX_TOKENS,
        Layout,
    )

    lay = Layout(TP)
    tp = get_tp_group()
    peers = PeerBuffer(
        k2.peer_bytes(MAX_TOKENS, TP), tp.cpu_group, rank, tp_size, device
    )
    peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    mw = _moe_weights(rank, device)
    mw.pop("raw")
    w = _gdn_inputs(MAX_TOKENS, MAX_TOKENS)
    gc = torch.Generator(device=device).manual_seed(13)  # replicated weights
    qsa_hw = _hc_weights(gc)
    mlp_hw = [_hc_weights(gc) for _ in range(2)]
    gr = torch.Generator(device=device).manual_seed(300 + rank)  # TP shards
    _, _, w_o, w_os = _mxfp4_linear(gr, HIDDEN, lay.CORE)
    o_qsa = _randn(gr, HIDDEN, lay.CORE, scale=0.02)
    # the QSA layer's attention output and gate: vLLM's, fixed here
    core_q = _randn(gr, MAX_TOKENS, lay.CORE)
    gate_q = _randn(gr, MAX_TOKENS, lay.CORE, scale=2.0)
    conv0, rstate0 = w["conv_state"].clone(), w["rstate"].clone()
    conv, rstate = conv0.clone(), rstate0.clone()
    widths = set(REPLAY_WIDTHS)
    k1_scr = {
        s: torch.zeros(
            max(
                k1.scratch_bytes(k1.K1Build(tokens=s, tp=TP, self_msq=m, front_only=f))
                for m in (False, True)
                for f in (False, True)
            ),
            dtype=torch.uint8,
            device=device,
        )
        for s in widths
    }
    k2_scr = {
        s: torch.zeros(
            k2.scratch_bytes(k2.K2Build(tokens=s, tp=TP)),
            dtype=torch.uint8,
            device=device,
        )
        for s in widths
    }
    outs = [torch.empty(s, HH, dtype=bf, device=device) for s in REPLAY_WIDTHS]
    msqs = [torch.empty(hc.NRED, s, HC, device=device) for s in REPLAY_WIDTHS]
    xs = [torch.empty(s, HIDDEN, dtype=bf, device=device) for s in REPLAY_WIDTHS]
    # the last row of odd widths is a cudagraph pad row
    idxs = {}
    for s in widths:
        idxs[s] = w["st_idx"][:s].clone()
        if s % 2:
            idxs[s][-1] = 0

    def post(key, state, inj, core, out, msq, li, **kw):
        k2.layer_post(
            key,
            core=core,
            state=state,
            inj_a=inj,
            **mlp_hw[li],
            out=out,
            msq_out=msq,
            scratch=k2_scr[key.tokens],
            peers=peers.addresses,
            rank=rank,
            epoch=epoch,
            layer=k2.TAG_SLOTS // 2 + li,
            **mw,
            **kw,
        )

    def steps() -> None:
        conv.copy_(conv0)
        rstate.copy_(rstate0)
        for i, s in enumerate(REPLAY_WIDTHS):
            epoch.add_(1)
            state = w["state"][:s].contiguous()
            # a GDN layer from the step's state, then a QSA layer's front and K2
            inj = torch.empty(s, HC, dtype=bf, device=device)
            core = torch.empty(s, lay.CORE, dtype=bf, device=device)
            k1.gdn_pre(
                k1.K1Build(tokens=s, tp=TP, self_msq=True),
                state=state,
                msq_in=None,
                hc_w=w["hc_w"],
                w_down=w["w_down"],
                w_up=w["w_up"],
                inj_out=inj,
                w_qkvz=w["w_qkvz"],
                w_qkvzs=w["w_qkvzs"],
                w_ba=w["w_ba"],
                conv_w=w["conv_w"],
                conv_state=conv,
                a_log=w["a_log"],
                dt_bias=w["dt_bias"],
                norm_w=w["norm_w"],
                rstate=rstate,
                st_idx=idxs[s],
                core=core,
                scratch=k1_scr[s],
                epoch=epoch,
                layer=0,
            )
            state1 = torch.empty(s, HH, dtype=bf, device=device)
            msq = torch.empty(hc.NRED, s, HC, device=device)
            key = k2.K2Build(tokens=s, tp=TP, oq=True)
            post(key, state, inj, core, state1, msq, 0, w_o=w_o, w_os=w_os)
            inj = torch.empty(s, HC, dtype=bf, device=device)
            k1.hc_front(
                k1.K1Build(tokens=s, tp=TP, front_only=True),
                state=state1,
                msq_in=msq,
                **qsa_hw,
                inj_out=inj,
                bin_out=xs[i],
                scratch=k1_scr[s],
                epoch=epoch,
                layer=1,
            )
            key = k2.K2Build(tokens=s, tp=TP, og=True)
            post(
                key,
                state1,
                inj,
                core_q[:s],
                outs[i],
                msqs[i],
                1,
                w_o=o_qsa,
                gate=gate_q[:s],
            )

    def snapshot() -> list[torch.Tensor]:
        return [t.clone() for t in (*outs, *msqs, *xs, conv, rstate)]

    steps()  # eager: builds every width before the capture
    torch.accelerator.synchronize()
    eager = snapshot()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        steps()
    graph.replay()
    torch.accelerator.synchronize()
    first = snapshot()
    assert all(torch.equal(a, b) for a, b in zip(eager, first)), "graph != eager"
    for n in range(REPLAYS - 1):
        graph.replay()
        torch.accelerator.synchronize()
        bad = [
            i
            for i, (a, b) in enumerate(zip(snapshot(), first))
            if not torch.equal(a, b)
        ]
        assert not bad, f"replay {n + 2}: tensors {bad} differ"
    for o in outs:
        gathered = [torch.empty_like(o) for _ in range(tp_size)]
        dist.all_gather(gathered, o)
        assert all(torch.equal(gathered[0], g) for g in gathered), "ranks disagree"
    peers.close()


@multi_gpu_test(num_gpus=2)
def test_mixed_width_graph_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GDN layer (K1, K2 with the MXFP4 out_proj) and a QSA layer (the HC
    front, K2 with the gated o_proj) at six consecutive decode steps of widths
    1-8 (pad rows included), captured in one HIP graph as a FULL decode graph
    holds them and replayed 300 times: bit-identical to the eager run every
    time, on both ranks. Steps of different widths share the peer buffer and
    the epoch; a scratch race between widths shows up as a mismatch."""
    multi_process_parallel(monkeypatch, 2, 1, _replay_worker)


# ---------------------------------------------------------------- draft head
def _lmhead_weights(rank: int, device, v_r: int):
    """One rank's lm_head shard as vLLM holds it after loading
    (``Mxfp4OnlineLinearMethod`` on the ASM FP4 GEMM, ``AiterMxfp4LinearKernel
    (use_asm_gemm=True)``): random fp4x2 / e8m0, through ``_mxfp4_linear``'s
    shuffle. Returns (raw q, raw scales, w, ws)."""
    from vllm.models.qwen4_exp.amd.mono.layout import HIDDEN

    gr = torch.Generator(device=device).manual_seed(500 + rank)
    return _mxfp4_linear(gr, v_r, HIDDEN)


def _lmhead_logits(x, q, sc):
    """fp32-of-bf16-rounded logits of the MXFP4-exact ``x`` (as ``quant_x``
    quantizes it) against the dequantized shard: what the kernel compares.
    The matmul itself stays fp32 (``_mxfp4_dequant`` / ``_mxfp4_qdq`` are
    already fp32); only the final logit is rounded to bf16."""
    w_ref = _mxfp4_dequant(q, sc)
    xq = _mxfp4_qdq(x)
    return F.linear(xq, w_ref).to(bf).float()


def _lmhead_reference(x, tp: int, v_r: int, valid: int, device):
    """The global argmax over every rank's shard (``_lmhead_weights``, the
    same seeds the worker uses), padding rows (>= ``valid``) excluded:
    ``torch.argmax``'s tie-break (the lowest index)."""
    logits = []
    for r in range(tp):
        q, sc, _, _ = _lmhead_weights(r, device, v_r)
        lg = _lmhead_logits(x, q, sc)
        lg[:, valid:] = float("-inf")
        logits.append(lg)
    return torch.argmax(torch.cat(logits, dim=-1), dim=-1)


def _bf16_ulp(v):
    """One bf16 ulp below ``v`` (already bf16-exact, held as fp32)."""
    vb = v.to(bf)
    nxt = torch.nextafter(vb, torch.full_like(vb, float("-inf")))
    return (vb.float() - nxt.float()).abs()


def _lmhead_check(out, x, tp: int, v_r: int, valid: int, device) -> int:
    """Checks the kernel's ``out`` against the reference, one rank's huge
    dequantized shard at a time (computed, used, freed -- never holding two
    ranks' shards live together). Requires an exact match to the reference
    argmax OR (fp32 summation order can differ between the kernel's MFMA
    reduction tree and torch's, which can flip the argmax between two
    near-tied logits) that the kernel's picked logit is within 1 bf16 ulp of
    the true per-token max. Returns the number of exact matches."""
    s = x.shape[0]
    best_val = torch.full((s,), float("-inf"), device=device)
    picked_val = torch.full((s,), float("nan"), device=device)
    for r in range(tp):
        q, sc, _, _ = _lmhead_weights(r, device, v_r)
        lg = _lmhead_logits(x, q, sc)
        lg[:, valid:] = float("-inf")
        best_val = torch.maximum(best_val, lg.max(dim=-1).values)
        local = (out - r * v_r).clamp(0, v_r - 1)
        mine = (out >= r * v_r) & (out < r * v_r + v_r)
        picked_val = torch.where(
            mine, lg.gather(-1, local.unsqueeze(-1)).squeeze(-1), picked_val
        )
        del q, sc, lg
        torch.cuda.empty_cache()
    exact = picked_val == best_val
    tol = _bf16_ulp(best_val)
    close = (best_val - picked_val).abs() <= tol
    assert torch.all(exact | close), (picked_val, best_val, out)
    return int(exact.sum().item())


def _reshuffle_lmhead(q, sc):
    """``_mxfp4_linear``'s shuffle, standalone: raw (q, sc) -> (w, ws). Lets a
    test plant specific rows in the raw codes, then re-derive the kernel's
    (shuffled) input from them."""
    from aiter.ops.shuffle import shuffle_weight

    w = shuffle_weight(q.clone().view(torch.float4_e2m1fn_x2), layout=(16, 16))
    sm, sn = sc.shape
    ws = sc.view(sm // 32, 2, 16, sn // 8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6)
    return w, ws.contiguous().view(sm, sn)


@ray.remote(num_gpus=1, max_calls=1)
def _head_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.qwen4_exp.amd.mono import head
    from vllm.models.qwen4_exp.amd.mono.layout import HIDDEN, MAX_TOKENS

    tp = get_tp_group()
    peers = PeerBuffer(head.peer_bytes(TP), tp.cpu_group, rank, tp_size, device)
    peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)

    # a padded shard smaller than the real one, a non-16-aligned valid cutoff
    v_r, valid = 16 * 300 + 32, 16 * 300 + 5
    vocab_start = rank * v_r
    q, sc, w, ws = _lmhead_weights(rank, device, v_r)

    # layout sanity: (w, ws) is what AITER's own ASM FP4 GEMM reads, cross-
    # checked against the dequantized reference (as _k2_worker does for
    # in_proj_qkvz / out_proj), not just internally self-consistent
    gl = torch.Generator(device=device).manual_seed(950)
    xl = _randn(gl, 4, HIDDEN, scale=0.3)
    stock = _mxfp4_linear_stock(xl, w, ws)
    ref_l = F.linear(_mxfp4_qdq(xl).to(bf), _mxfp4_dequant(q, sc).to(bf))
    assert _cos(stock, ref_l) > STOCK_COS_MIN

    # production pattern: one scratch buffer (sized for the widest S), one
    # fixed tag slot, mixed widths across launches -- only ``epoch`` (bumped
    # every launch) keeps the mailbox tags apart
    slot = 0
    scratch = torch.zeros(
        head.scratch_bytes(head.HeadBuild(tokens=MAX_TOKENS, tp=TP, rows=v_r)),
        dtype=torch.uint8,
        device=device,
    )
    for s in (1, 3, 8, 3, 1):
        epoch.add_(1)
        key = head.HeadBuild(tokens=s, tp=TP, rows=v_r)
        g = torch.Generator(device=device).manual_seed(900 + s)
        x = _randn(g, s, HIDDEN, scale=0.3)
        out = torch.full((s,), -1, dtype=torch.int64, device=device)
        head.draft_head(
            key,
            x=x,
            w=w,
            ws=ws,
            valid=valid,
            vocab_start=vocab_start,
            out=out,
            scratch=scratch[: head.scratch_bytes(key)],
            peers=peers.addresses,
            rank=rank,
            epoch=epoch,
            slot=slot,
        )
        torch.accelerator.synchronize()
        want = _lmhead_reference(x, TP, v_r, valid, device)
        torch.testing.assert_close(out, want, atol=0, rtol=0)
        outs = [torch.empty_like(out) for _ in range(tp_size)]
        dist.all_gather(outs, out)
        assert all(torch.equal(outs[0], o) for o in outs), (s, "ranks disagree")

    # four independent planted scenarios (each its own token and weight
    # column, so none of them can interact): a tie within one thread (rows
    # 4 / 5, the same row-pair thread's idx0 / idx1), a tie within a CTA
    # across lanes (rows 5 / 13, different row-pair threads of the same
    # group), a clear valid winner next to a bigger padding-row value that
    # must be ignored (row 45 vs the padding row 60), and a cross-rank tie
    # (rank 0's row 20 against rank 1's row 0: rank 0 must win on vocab_start
    # alone, 0 < pv_r)
    epoch.add_(1)
    pv_r, pvalid = 64, 48
    tie_a, tie_b = 4, 5
    tie_c, tie_d = 5, 13
    champ, pad_row = 45, 60
    cross_lo, cross_hi = 20, 0

    def _plant(q, sc, this_rank):
        """Nibble 7 -> e2m1 magnitude 6.0 (``_mxfp4_dequant``'s ``_FP4``
        max); ``col``'s byte is ``col // 2``, its low nibble if ``col`` is
        even, high if odd. Every plant lives in its own column, zero in
        every other token's x row, so the scenarios cannot interact."""

        def set_val(row, col, code):
            byte, hi = col // 2, col % 2 == 1
            if hi:
                q[row, byte] = (q[row, byte] & 0x0F) | 0x70
            else:
                q[row, byte] = (q[row, byte] & 0xF0) | 0x07
            sc[row, col // 32] = code

        # a planted row's block scale applies to its whole 32-column block:
        # zero the block's other codes first, or their random nibbles at the
        # planted scale outgrow the plants
        rows = (tie_a, tie_b, tie_c, tie_d, champ, pad_row, cross_lo)
        for row in rows if this_rank == 0 else (cross_hi,):
            q[row, :16] = 0
        if this_rank == 0:
            set_val(tie_a, 0, 180)
            set_val(tie_b, 0, 180)
            set_val(tie_c, 1, 180)
            set_val(tie_d, 1, 180)
            set_val(champ, 2, 170)
            set_val(pad_row, 2, 220)
            set_val(cross_lo, 3, 180)
        else:
            set_val(cross_hi, 3, 180)

    s = 4
    g = torch.Generator(device=device).manual_seed(901)
    x = _randn(g, s, HIDDEN, scale=0.3)
    for tok, col in enumerate((0, 1, 2, 3)):
        x[tok].zero_()
        x[tok, col] = 1.0
    pq, psc, _, _ = _lmhead_weights(rank, device, pv_r)
    _plant(pq, psc, rank)
    pw, pws = _reshuffle_lmhead(pq, psc)
    key = head.HeadBuild(tokens=s, tp=TP, rows=pv_r)
    out = torch.full((s,), -1, dtype=torch.int64, device=device)
    pscratch = torch.zeros(head.scratch_bytes(key), dtype=torch.uint8, device=device)
    head.draft_head(
        key,
        x=x,
        w=pw,
        ws=pws,
        valid=pvalid,
        vocab_start=rank * pv_r,
        out=out,
        scratch=pscratch,
        peers=peers.addresses,
        rank=rank,
        epoch=epoch,
        slot=slot,
    )
    torch.accelerator.synchronize()
    # the reference, reconstructed identically on every rank (the plant is a
    # pure function of the rank being reconstructed, not of this process)
    logits = []
    for r in range(TP):
        rq, rsc, _, _ = _lmhead_weights(r, device, pv_r)
        _plant(rq, rsc, r)
        lg = _lmhead_logits(x, rq, rsc)
        lg[:, pvalid:] = float("-inf")
        logits.append(lg)
    want = torch.argmax(torch.cat(logits, dim=-1), dim=-1)
    # sanity: the plants' own reasoning
    assert want.tolist() == [tie_a, tie_c, champ, cross_lo], want
    torch.testing.assert_close(out, want, atol=0, rtol=0)
    outs = [torch.empty_like(out) for _ in range(tp_size)]
    dist.all_gather(outs, out)
    assert all(torch.equal(outs[0], o) for o in outs), "ranks disagree (plant)"

    peers.close()


@multi_gpu_test(num_gpus=2)
def test_draft_head_matches_vllm_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """``draft_head``'s MXFP4 lm_head GEMV + greedy argmax across the vocab
    shard and the TP ranks, against a dequantized fp32 reference, at a small
    padded vocab shard (a non-16-aligned valid cutoff) replayed at mixed
    widths from one scratch buffer and tag slot (the production pattern);
    plus a layout sanity check against AITER's own ASM FP4 GEMM, and four
    independent planted ties / a planted padding-row max (must be ignored) /
    a cross-rank tie."""
    multi_process_parallel(monkeypatch, 2, 1, _head_worker)


@ray.remote(num_gpus=1, max_calls=1)
def _head_worker_full_vocab(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.qwen4_exp.amd.mono import head
    from vllm.models.qwen4_exp.amd.mono.layout import HIDDEN

    tp = get_tp_group()
    peers = PeerBuffer(head.peer_bytes(TP), tp.cpu_group, rank, tp_size, device)
    peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)

    # 124160 / 16 = 7760 = 30 * 256 + 80 row groups: CTAs 80 .. 255 run one
    # fewer loop iteration than CTAs 0 .. 79 here too, not only in the small
    # shard above
    v_r = 124160  # Qwen3.8-Flash-Next draft head's real per-rank vocab shard
    _, _, w, ws = _lmhead_weights(rank, device, v_r)
    s = 8
    key = head.HeadBuild(tokens=s, tp=TP, rows=v_r)
    g = torch.Generator(device=device).manual_seed(902)
    x = _randn(g, s, HIDDEN, scale=0.3)
    out = torch.full((s,), -1, dtype=torch.int64, device=device)
    scratch = torch.zeros(head.scratch_bytes(key), dtype=torch.uint8, device=device)
    epoch.add_(1)
    head.draft_head(
        key,
        x=x,
        w=w,
        ws=ws,
        valid=v_r,
        vocab_start=rank * v_r,
        out=out,
        scratch=scratch,
        peers=peers.addresses,
        rank=rank,
        epoch=epoch,
        slot=0,
    )
    torch.accelerator.synchronize()
    n_exact = _lmhead_check(out, x, TP, v_r, v_r, device)
    if rank == 0:
        print(f"draft_head s={s} rows={v_r}: {n_exact}/{s} tokens exact")

    # both ranks must keep launching in lockstep (the cross-rank reduce
    # rendezvous-waits on every rank), so only the timing / print is rank 0's
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n_iters = 100
    torch.accelerator.synchronize()
    if rank == 0:
        start.record()
    for _ in range(n_iters):
        epoch.add_(1)
        head.draft_head(
            key,
            x=x,
            w=w,
            ws=ws,
            valid=v_r,
            vocab_start=rank * v_r,
            out=out,
            scratch=scratch,
            peers=peers.addresses,
            rank=rank,
            epoch=epoch,
            slot=0,
        )
    if rank == 0:
        end.record()
    torch.accelerator.synchronize()
    if rank == 0:
        us = start.elapsed_time(end) * 1000 / n_iters
        print(f"draft_head s={s} rows={v_r}: {us:.1f} us/iter")
    # re-check correctness after the timing loop, not just once up front
    n_exact = _lmhead_check(out, x, TP, v_r, v_r, device)
    if rank == 0:
        print(f"draft_head s={s} rows={v_r}: {n_exact}/{s} tokens exact (post-timing)")
    peers.close()


@pytest.mark.slow_test
@multi_gpu_test(num_gpus=2)
def test_draft_head_matches_vllm_ops_full_vocab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``draft_head`` at the real Qwen3.8-Flash-Next draft head's per-rank
    vocab shard (124160 rows, no padding), S = 8, checked before and after
    timing 100 iterations (CUDA events, per-iteration microseconds printed).
    The reference (``_lmhead_check``) dequantizes one rank's ~169 MB shard at
    a time and accepts the kernel's pick if it is within 1 bf16 ulp of the
    true max, not only an exact match: at 8 x 248320 candidates, fp32
    summation order can differ between the kernel's MFMA tree and torch's
    enough to flip an exact near-tie."""
    multi_process_parallel(monkeypatch, 2, 1, _head_worker_full_vocab)


# ---------------------------------------------------------------- Stage 0a trace
#
# The in-kernel utilization timeline (K1 / K2's ``trace=True`` + ``dbg``
# stamps, ``vllm.models.qwen4_exp.amd.mono.stamps``) must be a pure
# observer: an off build's device code is unchanged (``const_expr``-gated),
# and an on build's extra stores all land in ``dbg``, never in a tensor the
# model reads. These tests check that bitwise -- not just numerically close
# (``_cos`` / ``assert_close``, what the matches_vllm_ops tests above use).
@pytest.mark.parametrize("s", [1, 8])
@torch.inference_mode()
def test_hc_front_trace_bitwise_identical(s: int) -> None:
    """K1's ``hc_front`` (front_only): the same inputs through ``trace=False``
    and ``trace=True`` builds produce bitwise-identical ``bin_out`` /
    ``inj_out``, and tracing visits every wait site this build actually
    reaches (a handful of ``k1.SLOTS`` entries, decoded nonzero)."""
    from vllm.models.qwen4_exp.amd.mono import k1, stamps
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HIDDEN

    g = torch.Generator(device="cuda").manual_seed(900 + s)
    state = _randn(g, s, k1.HH, scale=2.0)
    hw = _hc_weights(g)
    msq_in = _msq(state)

    def run(trace: bool):
        key = k1.K1Build(tokens=s, tp=TP, self_msq=False, front_only=True, trace=trace)
        x = torch.full((s, HIDDEN), float("nan"), dtype=bf, device="cuda")
        inj = torch.full((s, HC), float("nan"), dtype=bf, device="cuda")
        scratch = torch.zeros(k1.scratch_bytes(key), dtype=torch.uint8, device="cuda")
        dbg = stamps.dbg_buffer("cuda", nslot=k1.NSLOT) if trace else None
        k1.hc_front(
            key,
            state=state,
            msq_in=msq_in,
            **hw,
            inj_out=inj,
            bin_out=x,
            scratch=scratch,
            epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
            layer=0,
            dbg=dbg,
        )
        torch.accelerator.synchronize()
        return x, inj, dbg

    x_off, inj_off, _ = run(False)
    x_on, inj_on, dbg = run(True)
    assert torch.equal(x_off, x_on), "trace=True changed bin_out"
    assert torch.equal(inj_off, inj_on), "trace=True changed inj_out"
    us = stamps.decode(dbg, nslot=k1.NSLOT)
    for name in ("entry", "hc_down_loads", "hc_down_done", "lora_wait_hi", "exit"):
        slot = k1.SLOTS[name]
        assert (us[:, slot] != 0).any(), f"slot {name} ({slot}) never stamped"


@pytest.mark.parametrize("s,n_real", [(1, 1), (8, 7)])
@torch.inference_mode()
def test_k1_gdn_pre_trace_bitwise_identical(s: int, n_real: int) -> None:
    """K1's ``gdn_pre`` (a GDN layer): ``trace=False`` vs ``trace=True`` leave
    ``core``, the conv state and the recurrent state bitwise unchanged."""
    from vllm.models.qwen4_exp.amd.mono import k1, stamps
    from vllm.models.qwen4_exp.amd.mono.layout import Layout

    lay = Layout(TP)
    w = _gdn_inputs(s, n_real)

    def run(trace: bool):
        key = k1.K1Build(tokens=s, tp=TP, self_msq=False, trace=trace)
        conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
        core = torch.full((s, lay.CORE), float("nan"), dtype=bf, device="cuda")
        inj = torch.full((s, k1.HC), float("nan"), dtype=bf, device="cuda")
        scratch = torch.zeros(k1.scratch_bytes(key), dtype=torch.uint8, device="cuda")
        dbg = stamps.dbg_buffer("cuda", nslot=k1.NSLOT) if trace else None
        k1.gdn_pre(
            key,
            state=w["state"],
            msq_in=_msq(w["state"]),
            hc_w=w["hc_w"],
            w_down=w["w_down"],
            w_up=w["w_up"],
            inj_out=inj,
            w_qkvz=w["w_qkvz"],
            w_qkvzs=w["w_qkvzs"],
            w_ba=w["w_ba"],
            conv_w=w["conv_w"],
            conv_state=conv_state,
            a_log=w["a_log"],
            dt_bias=w["dt_bias"],
            norm_w=w["norm_w"],
            rstate=rstate,
            st_idx=w["st_idx"],
            core=core,
            scratch=scratch,
            epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
            layer=0,
            dbg=dbg,
        )
        torch.accelerator.synchronize()
        return core, inj, conv_state, rstate, dbg

    core_off, inj_off, conv_off, rst_off, _ = run(False)
    core_on, inj_on, conv_on, rst_on, dbg = run(True)
    assert torch.equal(core_off, core_on), "trace=True changed core"
    assert torch.equal(inj_off, inj_on), "trace=True changed inj_out"
    assert torch.equal(conv_off, conv_on), "trace=True changed the conv state"
    assert torch.equal(rst_off, rst_on), "trace=True changed the recurrent state"
    us = stamps.decode(dbg, nslot=k1.NSLOT)
    for name in ("entry", "qkvz_loads", "qkvz_done", "quantx_done", "exit"):
        slot = k1.SLOTS[name]
        assert (us[:, slot] != 0).any(), f"slot {name} ({slot}) never stamped"


@ray.remote(num_gpus=1, max_calls=1)
def _k2_trace_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.qwen4_exp.amd.mono import k2, stamps
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HH, HIDDEN, MAX_TOKENS, Layout

    lay = Layout(TP)
    tp = get_tp_group()
    peers = PeerBuffer(
        k2.peer_bytes(MAX_TOKENS, TP), tp.cpu_group, rank, tp_size, device
    )
    peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    mw = _moe_weights(rank, device)
    gc = torch.Generator(device=device).manual_seed(500)
    hw = _hc_weights(gc)
    step = 0
    for flags in K2_MODES:  # ({"oq": True}, {"og": True}): GDN then QSA layers
        gr = torch.Generator(device=device).manual_seed(5200 + rank)
        oq = flags.get("oq", False)
        if oq:
            _, _, w_o, w_os = _mxfp4_linear(gr, HIDDEN, lay.CORE)
        else:
            w_o, w_os = _randn(gr, HIDDEN, lay.CORE, scale=0.02), None
        for s in (1, 4, 8):
            gate = _randn(gr, s, lay.CORE, scale=2.0) if flags.get("og") else None
            core = _randn(gr, s, lay.CORE)
            state = _randn(gc, s, HH, scale=2.0)
            inj_a = _randn(gc, s, HC)

            def run(trace: bool, core=core, gate=gate, state=state, inj_a=inj_a):
                nonlocal step
                step += 1
                epoch.add_(1)
                key = k2.K2Build(tokens=s, tp=TP, trace=trace, **flags)
                scratch = torch.zeros(
                    k2.scratch_bytes(key), dtype=torch.uint8, device=device
                )
                out = torch.full((s, HH), float("nan"), dtype=bf, device=device)
                msq = torch.full((20, s, HC), float("nan"), device=device)
                dbg = stamps.dbg_buffer(device, nslot=k2.NSLOT) if trace else None
                k2.layer_post(
                    key,
                    core=core,
                    gate=gate,
                    state=state,
                    inj_a=inj_a,
                    w_o=w_o,
                    w_os=w_os,
                    **hw,
                    out=out,
                    msq_out=msq,
                    scratch=scratch,
                    peers=peers.addresses,
                    rank=rank,
                    epoch=epoch,
                    layer=step,
                    dbg=dbg,
                )
                torch.accelerator.synchronize()
                return out, msq, dbg

            out_off, msq_off, _ = run(False)
            out_on, msq_on, dbg = run(True)
            mode = (str(flags), s)
            assert torch.equal(out_off, out_on), f"trace=True changed out {mode}"
            assert torch.equal(msq_off, msq_on), f"trace=True changed msq_out {mode}"
            if rank == 0:
                us = stamps.decode(dbg, nslot=k2.NSLOT)
                for name in ("entry", "oproj_done", "msq_wait_hi", "exit"):
                    slot = k2.SLOTS[name]
                    msg = f"{mode}: slot {name} never stamped"
                    assert (us[:, slot] != 0).any(), msg
    peers.close()


@multi_gpu_test(num_gpus=2)
def test_k2_trace_bitwise_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """K2's ``layer_post`` (a GDN layer's ``oq=True`` out_proj and a QSA
    layer's ``og=True`` o_proj): ``trace=False`` vs ``trace=True`` leave
    ``out`` and ``msq_out`` bitwise unchanged, every rank agreeing."""
    multi_process_parallel(monkeypatch, 2, 1, _k2_trace_worker)
