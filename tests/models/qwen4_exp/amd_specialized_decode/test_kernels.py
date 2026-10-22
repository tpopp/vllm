# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the Qwen3.8-Flash-Next specialized decode kernels on MI355X
against the ops vLLM runs for the same modules, random weights in the
checkpoint's formats at TP2 shapes, every width the decode runs (1 .. 16).

- The HC front (one GPU): ``GatedResidual.mix`` (grouped Gemma norm, the
  down + inject GEMM, hc_silu, the up GEMM, the gate-mix) on a materialized
  state, from the previous block's sums of squares or its own.
- The GDN block (one GPU): in_proj_qkvz (online MXFP4) / in_proj_ba -> AITER's
  conv update and fused gated delta rule -> RMSNormGated, the states it leaves
  behind, pad rows.
- The o_proj and MoE blocks (two GPUs): out_proj (MXFP4) / o_proj (bf16, the
  output gate), the sparse MoE block (softmax top-10, the MXFP4 experts, the
  fused shared expert), each + the all-reduce + the HC combine into the
  multi-stream state, and the sums of squares they hand the next front.

No wait of any launch here gives up (the health words stay zero), except in
``test_stalled_rank_fails_stop``: a rank whose peer never launches gives up its
waits within the bound, records them and raises its flag on the output rank.
"""

import pytest
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.models.qwen4_exp.amd_specialized_decode.reference import (
    COS_MIN,
    _all_reduce_fp32,
    _check_route,
    _cos,
    _gdn_inputs,
    _gdn_reference,
    _hc_weights,
    _mix_ref,
    _moe_emulation,
    _moe_exact,
    _moe_reference,
    _moe_weights,
    _msq,
    _mxfp4_dequant,
    _mxfp4_linear,
    _mxfp4_qdq,
    _on_gfx950,
    _randn,
    _rel_l2,
)
from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)

bf = torch.bfloat16
TP = 2
WIDTHS = (1, 3, 8, 9, 15, 16)

pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="the Flash-Next specialized decode needs gfx950"
)


def _health(device="cuda") -> dict:
    """One GPU's ``health`` words and an ``alarm`` word (its own, no output rank)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import (
        HEALTH_WORDS,
    )

    health = torch.zeros(HEALTH_WORDS + 1, dtype=torch.int32, device=device)
    return {
        "health": health[:HEALTH_WORDS],
        "alarm": health[HEALTH_WORDS:].data_ptr(),
    }


@pytest.mark.parametrize("s", WIDTHS)
@pytest.mark.parametrize("self_msq", [False, True])
@torch.inference_mode()
def test_hc_front_matches_vllm_ops(s: int, self_msq: bool) -> None:
    """``hc_front`` against ``GatedResidual.mix``: the block input and the
    injection logits."""
    from vllm.models.qwen4_exp.amd.specialized_decode.mono import front
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.layout import (
        HC,
        HH,
        HIDDEN,
    )

    g = torch.Generator(device="cuda").manual_seed(s)
    state = _randn(g, s, HH, scale=2.0)
    hw = _hc_weights(g)
    want_x, want_inj, _ = _mix_ref(state, hw)
    key = front.FrontBuild(tokens=s, tp=TP, self_msq=self_msq)
    x = torch.full((s, HIDDEN), float("nan"), dtype=bf, device="cuda")
    inj = torch.full((s, HC), float("nan"), dtype=bf, device="cuda")
    scratch = torch.zeros(front.scratch_bytes(key), dtype=torch.uint8, device="cuda")
    hl = _health()
    front.hc_front(
        key,
        state=state,
        msq_in=None if self_msq else _msq(state),
        **hw,
        inj_out=inj,
        bin_out=x,
        scratch=scratch,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        **hl,
        layer=0,
    )
    torch.accelerator.synchronize()
    assert not hl["health"].any()
    assert _cos(inj, want_inj) > COS_MIN
    torch.testing.assert_close(inj, want_inj, atol=2e-2, rtol=2e-2)
    assert _cos(x, want_x) > COS_MIN
    torch.testing.assert_close(x, want_x, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("s", (1, 3, 8))
@torch.inference_mode()
def test_draft_front_matches_reference(s: int) -> None:
    """``mono.draft_front`` (the fc_embedding / fc_hidden projections fused
    with the attention HC's mix) against the MTP draft step's stock
    embedding branch (``_draft_front_ref``) followed by ``GatedResidual.mix``
    (``_mix_ref``): the materialized state, the block input and the
    injection logits."""
    from tests.models.qwen4_exp.amd_specialized_decode.reference import (
        _draft_front_ref,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.mono import draft_front as DF
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.layout import HC, HH, HIDDEN

    g = torch.Generator(device="cuda").manual_seed(s + 100)
    emb = _randn(g, s, HIDDEN, scale=1.5)
    prev = _randn(g, s, HH, scale=2.0)
    w_ne = _randn(g, HIDDEN, scale=0.1)
    w_nh = _randn(g, HH, scale=0.1)
    w_fe = _randn(g, HIDDEN, HIDDEN, scale=0.05)
    w_fh = _randn(g, HIDDEN, HIDDEN, scale=0.05)
    hw = _hc_weights(g)

    want_state = _draft_front_ref(emb, prev, w_ne, w_nh, w_fe, w_fh, 1e-6)
    want_x, want_inj, _ = _mix_ref(want_state, hw)

    key = DF.FrontBuild(tokens=s, tp=TP)
    state_out = torch.full((s, HH), float("nan"), dtype=bf, device="cuda")
    x = torch.full((s, HIDDEN), float("nan"), dtype=bf, device="cuda")
    inj = torch.full((s, HC), float("nan"), dtype=bf, device="cuda")
    scratch = torch.zeros(DF.scratch_bytes(key), dtype=torch.uint8, device="cuda")
    hl = _health()
    DF.draft_front(
        key,
        emb=emb.contiguous(),
        prev=prev.contiguous(),
        w_ne=w_ne,
        w_nh=w_nh,
        w_fe=w_fe,
        w_fh=w_fh,
        **hw,
        state_out=state_out,
        bin_out=x,
        inj_out=inj,
        scratch=scratch,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        **hl,
        layer=0,
    )
    torch.accelerator.synchronize()
    assert not hl["health"].any()
    assert _cos(state_out, want_state) > COS_MIN
    torch.testing.assert_close(state_out, want_state, atol=2e-2, rtol=2e-2)
    assert _cos(inj, want_inj) > COS_MIN
    torch.testing.assert_close(inj, want_inj, atol=2e-2, rtol=2e-2)
    assert _cos(x, want_x) > COS_MIN
    torch.testing.assert_close(x, want_x, atol=2e-2, rtol=2e-2)


# (1, 1) twice: a width again after others compiled (the JIT globals check)
@pytest.mark.parametrize(
    "s,n_real", [(1, 1), (5, 4), (8, 8), (1, 1), (9, 9), (16, 15), (16, 16)]
)
@pytest.mark.parametrize("shuffled", [True, False])
@torch.inference_mode()
def test_gdn_block_matches_vllm_ops(s: int, n_real: int, shuffled: bool) -> None:
    """The GDN block against in_proj -> AITER's conv update + fused gated delta
    rule -> RMSNormGated on one decode step, the states it leaves behind, and
    the pad rows (null block: no state written, zero core); in_proj_qkvz on
    either of AITER's FP4 GEMMs (``shuffled``: the ASM one)."""
    from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.layout import Layout

    lay = Layout(TP)
    w = _gdn_inputs(s, n_real, shuffled=shuffled)
    ref = _gdn_reference(w, n_real)
    key = blocks.GdnBuild(tokens=s, tp=TP, shuffled=shuffled)
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    core = torch.full((s, lay.CORE), float("nan"), dtype=bf, device="cuda")
    scratch = torch.zeros(
        blocks.scratch_bytes(blocks.gdn_scratch_layout(key)),
        dtype=torch.uint8,
        device="cuda",
    )
    hl = _health()
    blocks.gdn_block(
        key,
        x=ref["x"].contiguous(),
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
        **hl,
        layer=0,
    )
    torch.accelerator.synchronize()
    assert not hl["health"].any()
    live = w["st_idx"][:n_real].long()
    errs = {
        "conv_state": _rel_l2(conv_state[live], ref["conv_state"][live]),
        "rstate": _rel_l2(rstate[live], ref["rstate"][live]),
        "core": _rel_l2(core[:n_real], ref["core"][:n_real]),
    }
    assert errs["conv_state"] < 5e-3 and errs["rstate"] < 5e-3, errs
    assert _cos(core[:n_real], ref["core"][:n_real]) > COS_MIN, errs
    assert torch.equal(core[n_real:], torch.zeros_like(core[n_real:]))
    torch.testing.assert_close(rstate[0], w["rstate"][0], atol=0, rtol=0)
    torch.testing.assert_close(conv_state[0], w["conv_state"][0], atol=0, rtol=0)


@pytest.mark.parametrize("vch", [16, 32])
@pytest.mark.parametrize(
    "accepted,pad_last",
    [
        ([1], False),
        ([2], False),
        ([4], False),
        ([1, 4], False),
        ([2, 3], False),
        ([3, 1], True),
    ],
)
@torch.inference_mode()
def test_gdn_block_verify_matches_stock_chain(
    accepted: list, pad_last: bool, vch: int
) -> None:
    """P3: the GDN block's ``verify`` build against the stock spec-decode
    chain (``causal_conv1d_update`` + FLA's non-AITER
    ``fused_sigmoid_gating_delta_rule_update`` + ``RMSNormGated``) that a
    pure, uniform verify step otherwise runs (``gdn_verify_core``), across
    ``n_req in {1, 2}``, every accepted-count in ``1..q``, a whole padding
    request (``pad_last``), and both GDN task chunk sizes (``vch``: 16 is the
    8-chunk layout at one request, 32 the 4-chunk layout that halves the task
    count a request so a 2-request batch still fits one task round)."""
    from tests.models.qwen4_exp.amd_specialized_decode.reference import (
        _verify_inputs,
        _verify_reference,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks

    q = 4
    w, lay, s, n_req = _verify_inputs(accepted, pad_last, q=q)
    live_n_req = n_req - 1 if pad_last else n_req
    live_s = live_n_req * q
    ref = _verify_reference(w, lay, live_n_req, q=q)

    key = blocks.GdnBuild(tokens=s, tp=TP, verify=True, q=q, vch=vch)
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    core = torch.full((s, lay.CORE), float("nan"), dtype=bf, device="cuda")
    scratch = torch.zeros(
        blocks.scratch_bytes(blocks.gdn_scratch_layout(key)),
        dtype=torch.uint8,
        device="cuda",
    )
    hl = _health()
    blocks.gdn_block(
        key,
        x=ref["x"].contiguous(),
        w_qkvz=w["w_qkvz"],
        w_qkvzs=w["w_qkvzs"],
        w_ba=w["w_ba"],
        conv_w=w["conv_w"],
        conv_state=conv_state,
        a_log=w["a_log"],
        dt_bias=w["dt_bias"],
        norm_w=w["norm_w"],
        rstate=rstate,
        core=core,
        scratch=scratch,
        epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
        **hl,
        layer=0,
        spec_idx=w["spec_state_indices"],
        num_accepted=w["num_accepted_tokens"],
    )
    torch.accelerator.synchronize()
    assert not hl["health"].any()

    live_slots = []
    for r in range(live_n_req):
        acc = accepted[r]
        live_slots += [w["spec_state_indices"][r, 0].item()]
        live_slots += [w["spec_state_indices"][r, t].item() for t in range(acc)]
    live_slots = sorted(set(live_slots))
    errs = {
        "conv_state": _rel_l2(conv_state[live_slots], ref["conv_state"][live_slots]),
        "rstate": _rel_l2(rstate[live_slots], ref["rstate"][live_slots]),
        "core": _rel_l2(core[:live_s], ref["core"]),
    }
    assert errs["conv_state"] < 5e-3 and errs["rstate"] < 5e-3, errs
    assert _cos(core[:live_s], ref["core"]) > COS_MIN, errs
    if pad_last:
        pr = n_req - 1
        rows = slice(pr * q, (pr + 1) * q)
        assert torch.equal(core[rows], torch.zeros_like(core[rows]))
        # an all-zero padding row-group's slot 0 is never touched
        torch.testing.assert_close(conv_state[0], w["conv_state"][0], atol=0, rtol=0)
        torch.testing.assert_close(rstate[0], w["rstate"][0], atol=0, rtol=0)


def _tp_state(rank: int, tp_size: int, device) -> dict:
    """A rank's launch state as ``Launcher`` builds it: the peer buffer (the
    output rank's flags in its head), the health words, the alarm word."""
    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.specialized_decode.mono.abi import PEER_HEAD
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.peer import PeerBuffer
    from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.layout import MAX_TOKENS

    peers = PeerBuffer(
        PEER_HEAD + blocks.peer_bytes(MAX_TOKENS, TP),
        get_tp_group().cpu_group,
        rank,
        tp_size,
        device,
    )
    peers.bytes.zero_()
    return {
        "buffer": peers,
        "flags": peers.bytes[: 4 * tp_size].view(torch.int32),
        "launch": {
            "peers": peers.addresses + PEER_HEAD,
            "rank": rank,
            "health": _health(device)["health"],
            "alarm": int(peers.addresses[0].item()) + 4 * rank,
        },
    }


@ray.remote(num_gpus=1, max_calls=1)
def _blocks_worker(
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

    from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.layout import (
        HC,
        HIDDEN,
        Layout,
    )

    lay = Layout(TP)
    tps = _tp_state(rank, tp_size, device)
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    mw = _moe_weights(rank, device)
    gr = torch.Generator(device=device).manual_seed(200 + rank)
    gc = torch.Generator(device=device).manual_seed(11)
    oq_raw, os_raw, w_oq, w_os = _mxfp4_linear(gr, HIDDEN, lay.CORE)
    w_oq_ref = _mxfp4_dequant(oq_raw, os_raw).to(bf)
    # the out_proj Triton FP4 GEMM's layout: the same weight, unshuffled
    flat = _mxfp4_linear(gr, HIDDEN, lay.CORE, shuffled=False)
    w_of_ref = _mxfp4_dequant(flat[0], flat[1]).to(bf)
    w_o = _randn(gr, HIDDEN, lay.CORE, scale=0.02)

    def agree(t, what):
        outs = [torch.empty_like(t) for _ in range(tp_size)]
        dist.all_gather(outs, t)
        assert all(torch.equal(outs[0], o) for o in outs), (what, "ranks disagree")

    def launch(block, key, state, inj, **kw):
        s = key.tokens
        out = torch.full_like(state, float("nan"))
        msq = torch.full((blocks.NRED, s, HC), float("nan"), device=device)
        block(key, out=out, state=state, inj=inj, msq_out=msq, **kw)
        torch.accelerator.synchronize()
        return out.unflatten(-1, (HC, HIDDEN)), msq

    def check(block, key, what, layer, **kw):
        """``block`` into a zero state with zero injection logits (every stream
        the all-reduced block output: the multiplier is 1), then into a random
        state against vLLM's HC combine of that output and the sums of
        squares the next front reads; the two launches on mailbox slots
        ``layer`` and ``layer + 2``. -> the block output."""
        s = key.tokens
        zero = torch.zeros(s, HC * HIDDEN, dtype=bf, device=device)
        got, _ = launch(block, key, zero, zero.new_zeros(s, HC), layer=layer, **kw)
        blk = got[:, 0].contiguous()
        assert all(torch.equal(got[:, k], blk) for k in range(HC)), what
        agree(blk, what)
        state = _randn(gc, s, HC * HIDDEN)
        inj = _randn(gc, s, HC, scale=2.0)
        got, msq = launch(block, key, state, inj, layer=layer + 2, **kw)
        m = 2.0 * torch.sigmoid(inj.float() / HC)
        ref = (
            state.float().unflatten(-1, (HC, HIDDEN))
            + blk.float()[:, None] * m[..., None]
        ).to(bf)
        torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2, msg=str(what))
        sq = got.float().unflatten(-1, (blocks.NRED, -1)).square().sum(-1)
        torch.testing.assert_close(msq, sq.permute(2, 0, 1), rtol=1e-4, atol=0)
        agree(got, what)
        return blk

    # each width twice: both epoch parities, one layer's oproj and moe a step
    for s in [w for w in (1, 3, 8, 9, 15, 16) for _ in range(2)]:
        epoch.add_(1)
        # (oq, shuffled): out_proj on the ASM / Triton FP4 GEMM, o_proj bf16
        for slot, (oq, shuffled) in enumerate(
            ((True, True), (True, False), (False, True))
        ):
            core = _randn(gr, s, lay.CORE)
            gate = None if oq else _randn(gr, s, lay.CORE, scale=2.0)
            w, ws, ref_w = (
                (w_oq, w_os, w_oq_ref) if shuffled else (flat[2], flat[3], w_of_ref)
            )
            out = check(
                blocks.oproj_block,
                blocks.OprojBuild(tokens=s, tp=TP, oq=oq, og=not oq, shuffled=shuffled),
                ("oproj", oq, shuffled, s),
                core=core,
                gate=gate,
                w_o=w if oq else w_o,
                w_os=ws if oq else None,
                **tps["launch"],
                epoch=epoch,
                layer=blocks.OPROJ_SLOT + 4 * slot,
            )
            if oq:
                part = F.linear(_mxfp4_qdq(core, shuffled).to(bf), ref_w)
            else:
                assert gate is not None
                gated = (core.float() * torch.sigmoid(gate.float())).to(bf)
                part = F.linear(gated, w_o)
            err = _rel_l2(out, _all_reduce_fp32(part))
            assert err < 1e-2, ("oproj", oq, shuffled, s, err)

        key = blocks.MoeBuild(tokens=s, tp=TP)
        x = _randn(gc, s, HIDDEN)
        scr = blocks.moe_scratch_layout(key)
        scratch = torch.zeros(
            blocks.scratch_bytes(scr), dtype=torch.uint8, device=device
        )
        out = check(
            blocks.moe_block,
            key,
            ("moe", s),
            x=x,
            scratch=scratch,
            **tps["launch"],
            epoch=epoch,
            layer=blocks.MOE_SLOT,
            **{k: v for k, v in mw.items() if k != "raw"},
        )
        ids = _check_route(scratch, scr["route"], x.clone(), mw)
        assert _cos(out, _moe_emulation(x.clone(), mw, ids)) > COS_MIN, ("moe", s)
        exact = _moe_exact(x.clone(), mw, ids)
        err = _rel_l2(out, exact)
        err_stock = _rel_l2(_moe_reference(x.clone(), mw, ids), exact)
        assert err < 1.02 * err_stock, ("moe", s, err, err_stock)
    assert not tps["launch"]["health"].any() and not tps["flags"].any()
    tps["buffer"].close()


@multi_gpu_test(num_gpus=2)
def test_blocks_tp2_match_vllm_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 2, 1, _blocks_worker)


@ray.remote(num_gpus=1, max_calls=1)
def _stall_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    import time

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.specialized_decode.mono.abi import (
        FAILED,
        WAIT_SECONDS,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.mono.common.debug import (
        given_up_waits,
    )
    from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.layout import (
        HC,
        HIDDEN,
        Layout,
    )

    lay = Layout(TP)
    tps = _tp_state(rank, tp_size, device)
    health = tps["launch"]["health"]
    key = blocks.OprojBuild(tokens=1, tp=TP, oq=False, og=True)
    g = torch.Generator(device=device).manual_seed(rank)
    state = _randn(g, 1, HC * HIDDEN)

    def launch_alone() -> float:
        """The o_proj block on this rank only: its peer never sends its partial."""
        t0 = time.monotonic()
        blocks.oproj_block(
            key,
            core=_randn(g, 1, lay.CORE),
            gate=_randn(g, 1, lay.CORE),
            w_o=_randn(g, HIDDEN, lay.CORE, scale=0.02),
            state=state,
            inj=_randn(g, 1, HC),
            out=torch.empty_like(state),
            msq_out=torch.empty(blocks.NRED, 1, HC, device=device),
            **tps["launch"],
            epoch=torch.ones(1, dtype=torch.int32, device=device),
            layer=blocks.OPROJ_SLOT,
        )
        torch.accelerator.synchronize()
        return time.monotonic() - t0

    if rank == 0:
        took = launch_alone()
        assert WAIT_SECONDS * 0.9 < took < WAIT_SECONDS * 6, took
        assert health[FAILED] != 0
        waits = given_up_waits(health.cpu(), blocks.PEER_REGIONS)
        assert waits and all(" ar pair " in w or " final pair " in w for w in waits)
        assert tps["flags"].tolist() == [1, 0], "the output rank's flags"
        # sticky: a failed rank's waits exit at once
        assert launch_alone() < WAIT_SECONDS / 2
    get_tp_group().barrier()
    if rank == 1:
        assert not health.any()
    tps["buffer"].close()


@multi_gpu_test(num_gpus=2)
def test_stalled_rank_fails_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rank whose peer never launches gives up its waits within the bound
    instead of hanging: FAILED, a record naming the peer region, its flag on
    the output rank, and its later launches return at once."""
    multi_process_parallel(monkeypatch, 2, 1, _stall_worker)


@ray.remote(num_gpus=1, max_calls=1)
def _draft_head_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    """``mono.draft_head`` (the MTP draft's fused MXFP4 lm_head GEMV + TP
    greedy argmax) against a dequantized full-vocab reference: each rank
    holds its own vocab shard (as a vocab-parallel ``ParallelLMHead`` would),
    the global argmax must agree with ``torch.argmax`` over every rank's
    shard concatenated, and every rank must agree with each other."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.peer import PeerBuffer
    from vllm.models.qwen4_exp.amd.specialized_decode.mono import draft_head as DH
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.layout import HIDDEN

    rows = 256  # a small shard: a multiple of 32, several row groups
    valid = 251  # some padding rows, never win
    health = _health(device)["health"]
    peers = PeerBuffer(
        DH.peer_bytes(tp_size), get_tp_group().cpu_group, rank, tp_size, device
    )
    peers.bytes.zero_()
    alarm = int(peers.addresses[0].item()) + 4 * rank

    gr = torch.Generator(device=device).manual_seed(300 + rank)
    gc = torch.Generator(device=device).manual_seed(42)  # the replicated x
    q, sc, w, ws = _mxfp4_linear(gr, rows, HIDDEN)
    w_ref = _mxfp4_dequant(q, sc).to(bf)
    epoch = torch.zeros(1, dtype=torch.int32, device=device)

    for s in (1, 2, 4, 8):
        x = _randn(gc, s, HIDDEN, scale=1.2)
        key = DH.HeadBuild(tokens=s, tp=tp_size, rows=rows)
        scratch = torch.zeros(DH.scratch_bytes(key), dtype=torch.uint8, device=device)
        out = torch.full((s,), -1, dtype=torch.int64, device=device)
        epoch.add_(1)
        DH.draft_head(
            key,
            x=x.contiguous(),
            w=w,
            ws=ws,
            valid=valid,
            vocab_start=rank * rows,
            out=out,
            scratch=scratch,
            peers=peers.addresses,
            rank=rank,
            epoch=epoch,
            health=health,
            alarm=alarm,
            layer=0,
        )
        torch.accelerator.synchronize()
        assert not health.any()

        # every rank's dequantized shard, masked to its valid rows
        logits = F.linear(_mxfp4_qdq(x).to(bf), w_ref).to(bf).float()
        logits[:, valid:] = float("-inf")
        all_logits = [torch.empty_like(logits) for _ in range(tp_size)]
        dist.all_gather(all_logits, logits)
        full = torch.cat(all_logits, dim=-1)
        want = full.argmax(dim=-1)
        assert torch.equal(out, want), (s, rank, out.tolist(), want.tolist())

        outs = [torch.empty_like(out) for _ in range(tp_size)]
        dist.all_gather(outs, out)
        assert all(torch.equal(outs[0], o) for o in outs), (s, "ranks disagree")
    peers.close()


@multi_gpu_test(num_gpus=2)
def test_draft_head_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 2, 1, _draft_head_worker)
