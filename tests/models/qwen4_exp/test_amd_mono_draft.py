# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.8-Flash-Next MTP mono draft step (``mono_draft.DraftMonoDecode``):

- a pure-torch check that the padded-mixer trick (the final mixer's down
  projection has no injection rows, but the kernel always wants
  ``DOWN_ROWS``) cannot perturb the mixed block input, CPU-only.
- ``DraftMonoDecode.eligible`` against fake forward-context metadata,
  CPU-only.
- the draft's three launches (the attention HC front, K2, the mixer HC
  front) against the stock torch ops, on MI355X (skipped elsewhere, as
  ``test_amd_mono.py``).
"""

import types

import pytest
import torch
import torch.nn.functional as F

bf = torch.bfloat16
TP = 2
COS_MIN = 0.9999


def _on_gfx950() -> bool:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


# ---------------------------------------------------------------- (a) padding
def _grouped_gemma_rmsnorm_ref(x, w, eps, hc_count):
    """``ops/hc.py``'s grouped Gemma RMSNorm, plain torch: each ``hc_count``
    group of the last dim normalized on its own sum of squares, a per-branch
    (not shared) affine (``hc_per_branch_norm=True``: ``w`` is ``[hc_count *
    group]``)."""
    s, dim = x.shape
    group = dim // hc_count
    xg = x.float().view(s, hc_count, group)
    rrms = torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
    wg = w.float().view(hc_count, group)
    y = xg * rrms
    return (y + y * wg).reshape(s, dim)


def _hc_silu_ref(x, hc_count):
    y = x.float() / hc_count
    return y * torch.sigmoid(y)


def _hc_gate_mix_ref(xn, gate, hc_count):
    s, dim = xn.shape
    group = dim // hc_count
    xg = xn.view(s, hc_count, group)
    gg = gate.view(s, hc_count, group)
    return (torch.sigmoid(gg) * xg).mean(1)


def _mix_cpu(state, hc_w, w_down, w_up, hc_count, lora_rank, eps=1e-6):
    """``GatedResidual.mix`` on ``state`` [S, hc_count*H] -> block input, plain
    torch (no Triton): a CPU-safe stand-in for ``test_amd_mono.py``'s
    ``_mix_ref``, which calls the GPU HC ops directly."""
    xn = _grouped_gemma_rmsnorm_ref(state, hc_w, eps, hc_count)
    d = F.linear(xn, w_down.float())
    lora = _hc_silu_ref(d[:, :lora_rank], hc_count)
    gate = F.linear(lora, w_up.float())
    return _hc_gate_mix_ref(xn, gate, hc_count)


def test_padded_mixer_down_matches_unpadded() -> None:
    """The final mixer's down projection has ``LORA`` rows (``use_combine=
    False``: no injection), but ``k1.hc_front`` always reads ``DOWN_ROWS``
    rows. Zero-padding it to ``DOWN_ROWS`` must not change the mixed block
    input: a down GEMM's output row only depends on that row of the weight,
    so the extra zero rows (which only ever reach the discarded injection
    output, never the gate-mix) cannot perturb it."""
    from vllm.models.qwen4_exp.amd.mono.layout import DOWN_ROWS, HC, HH, LORA

    g = torch.Generator().manual_seed(0)
    for s in (1, 3, 8):
        state = torch.randn(s, HH, generator=g)
        hc_w = torch.randn(HH, generator=g) * 0.1
        down = torch.randn(LORA, HH, generator=g) * 0.01
        up = torch.randn(HH, LORA, generator=g) * 0.05

        bin_unpadded = _mix_cpu(state, hc_w, down, up, HC, LORA)

        padded = torch.zeros(DOWN_ROWS, HH)
        padded[:LORA] = down
        bin_padded = _mix_cpu(state, hc_w, padded, up, HC, LORA)

        torch.testing.assert_close(bin_padded, bin_unpadded, atol=0, rtol=0)


# ---------------------------------------------------------------- (b) eligible
def _fake_qsa_metadata(
    *, num_prefills: int = 0, num_decodes: int = 3, num_decode_tokens: int | None = None
):
    """A real ``QSAForwardMetadata`` (the type ``self_attn.indexer.raw_key_
    cache.prefix`` actually keys in ``attn_metadata``, unlike the plain
    ``FlashAttentionMetadata`` at ``self_attn.layer_name``, which has no
    ``num_prefills``): every field but the three ``eligible`` reads is an
    empty placeholder. ``num_decode_tokens`` defaults to ``num_decodes`` (a
    1-row-a-request decode step); pass it explicitly > ``num_decodes`` for a
    draft-position-0-shaped step (query_len k + 1)."""
    from vllm.models.qwen4_exp.common.qsa_cache import QSAForwardMetadata

    if num_decode_tokens is None:
        num_decode_tokens = num_decodes
    empty = torch.empty(0, dtype=torch.int32)
    return QSAForwardMetadata(
        block_table=empty,
        slot_mapping=empty,
        seq_lens=empty,
        query_start_loc=empty,
        token_to_req=empty,
        logical_positions=empty,
        visible_blocks=empty,
        k_work_metadata=empty,
        num_actual_tokens=num_decode_tokens,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=0,
        max_query_len=1,
        decode_query_len=1,
        max_seq_len=1,
        storage_block_size=1,
        compress_ratio=1,
    )


def _fake_mono():
    """A ``DraftMonoDecode`` with no model / kernels: just enough state for
    ``eligible`` (``ok``, ``_probe``, a no-op ``check_weights``)."""
    from vllm.models.qwen4_exp.amd import mono_draft

    mono = object.__new__(mono_draft.DraftMonoDecode)
    mono.ok = True
    mono._probe = "model.layers.48.self_attn.indexer.raw_key_cache"
    mono.checked = 0
    mono.check_weights = lambda: setattr(mono, "checked", mono.checked + 1)
    return mono


def _patch_context(monkeypatch, *, piecewise: bool, metadata):
    from vllm.models.qwen4_exp.amd import mono_draft

    monkeypatch.setattr(mono_draft, "in_piecewise_cudagraph", lambda: piecewise)
    ctx = types.SimpleNamespace(attn_metadata=metadata)
    monkeypatch.setattr(mono_draft, "get_forward_context", lambda: ctx)


def test_eligible_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    mono = _fake_mono()
    md = {mono._probe: _fake_qsa_metadata()}
    _patch_context(monkeypatch, piecewise=False, metadata=md)
    input_ids = torch.zeros(3, dtype=torch.int64)
    assert mono.eligible(input_ids, None) is True
    assert mono.checked == 1


@pytest.mark.parametrize(
    "setup",
    [
        "not_ok",
        "intermediate_tensors",
        "piecewise",
        "no_input_ids",
        "too_many_rows",
        "no_metadata_dict",
        "missing_probe",
        "has_prefills",
        "draft_position_0",
    ],
)
def test_eligible_rejects(monkeypatch: pytest.MonkeyPatch, setup: str) -> None:
    from vllm.models.qwen4_exp.amd.mono.layout import MAX_TOKENS

    mono = _fake_mono()
    input_ids = torch.zeros(3, dtype=torch.int64)
    intermediate_tensors = None
    piecewise = False
    md: object = {mono._probe: _fake_qsa_metadata()}

    if setup == "not_ok":
        mono.ok = False
    elif setup == "intermediate_tensors":
        intermediate_tensors = object()
    elif setup == "piecewise":
        piecewise = True
    elif setup == "no_input_ids":
        input_ids = None
    elif setup == "too_many_rows":
        input_ids = torch.zeros(MAX_TOKENS + 1, dtype=torch.int64)
    elif setup == "no_metadata_dict":
        md = None
    elif setup == "missing_probe":
        md = {}
    elif setup == "has_prefills":
        md = {mono._probe: _fake_qsa_metadata(num_prefills=1)}
    elif setup == "draft_position_0":
        # with VLLM_ROCM_MONO_DRAFT_POS0 off: supports_spec_as_decode also
        # reports num_prefills == 0 here, but every request's query_len is
        # k + 1 (3 requests, 4 rows each), not 1 row a request
        from vllm import envs

        monkeypatch.setattr(envs, "VLLM_ROCM_MONO_DRAFT_POS0", False)
        md = {mono._probe: _fake_qsa_metadata(num_decodes=3, num_decode_tokens=12)}

    _patch_context(monkeypatch, piecewise=piecewise, metadata=md)
    assert mono.eligible(input_ids, intermediate_tensors) is False
    assert mono.checked == 0


# ---------------------------------------------------------------- (c) GPU
@pytest.mark.skipif(
    not _on_gfx950(), reason="the Flash-Next mono kernels need gfx950 (MI355X)"
)
def test_draft_step_matches_vllm_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """The draft's three launches (``hc_front`` on the attention HC, ``K2``,
    ``hc_front`` on the final mixer with a zero-padded down) against the
    stock torch ops, S in {1, 3, 8}, two ranks. Models ``test_amd_mono.py``'s
    ``_k2_worker``: the attention's (core, gate) output is a fixed random
    tensor (the real QSA attention is out of scope here), so this isolates
    the HC / MoE kernel numerics the draft forward adds around it."""
    import ray

    from tests.utils import multi_process_parallel

    worker = ray.remote(num_gpus=1, max_calls=1)(_draft_step_worker)
    multi_process_parallel(monkeypatch, 2, 1, worker)


def _draft_step_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    """Not a ray remote function itself (ray need not be importable just to
    collect this module): ``test_draft_step_matches_vllm_ops`` wraps it."""
    import torch.distributed as dist

    from tests.models.qwen3_5.test_amd_mono import (
        _moe_emulation,
        _all_reduce_fp32,
        _check_route,
        _cos,
        _pairs,
        _randn,
        _rel_l2,
    )
    from tests.models.qwen4_exp.test_amd_mono import (
        _hc_weights,
        _mix_ref,
        _moe_weights,
        _msq,
    )
    from tests.utils import init_test_distributed_environment
    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.qwen4_exp.amd.mono import k1, k2
    from vllm.models.qwen4_exp.amd.mono.layout import (
        DOWN_ROWS,
        HC,
        HH,
        HIDDEN,
        LORA,
        MAX_TOKENS,
        Layout,
    )
    from vllm.models.qwen4_exp.amd.ops.hc import hc_combine

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

    lay = Layout(TP)
    tp = get_tp_group()
    peers = PeerBuffer(
        k2.peer_bytes(MAX_TOKENS, TP), tp.cpu_group, rank, tp_size, device
    )
    peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)

    gc = torch.Generator(device=device).manual_seed(21)  # replicated weights
    gr = torch.Generator(device=device).manual_seed(400 + rank)  # TP shards
    attn_hw = _hc_weights(gc)
    mlp_hw = _hc_weights(gc)
    mixer_down = _randn(gc, LORA, HH, scale=0.01)
    mixer_padded = torch.zeros(DOWN_ROWS, HH, dtype=bf, device=device)
    mixer_padded[:LORA] = mixer_down
    mixer_hw = {
        "hc_w": _randn(gc, HH, scale=0.1),
        "w_down": mixer_padded,
        "w_up": _randn(gc, HH, LORA, scale=0.05),
    }
    w_o = _randn(gr, HIDDEN, lay.CORE, scale=0.02)  # bf16 o_proj: not asm_mxfp4
    mw = _moe_weights(rank, device)

    for s in (1, 3, 8):
        epoch.add_(1)
        state = _randn(gc, s, HH, scale=2.0)
        core = _randn(gr, s, lay.CORE)
        gate = _randn(gr, s, lay.CORE, scale=2.0)

        inj = torch.full((s, HC), float("nan"), dtype=bf, device=device)
        x = torch.full((s, HIDDEN), float("nan"), dtype=bf, device=device)
        k1_key = k1.K1Build(tokens=s, tp=TP, self_msq=True, front_only=True)
        scratch1 = torch.zeros(
            k1.scratch_bytes(k1_key), dtype=torch.uint8, device=device
        )
        k1.hc_front(
            k1_key,
            state=state,
            msq_in=None,
            **attn_hw,
            inj_out=inj,
            bin_out=x,
            scratch=scratch1,
            epoch=epoch,
            layer=48,
        )

        state2 = torch.empty_like(state)
        msq = torch.full((20, s, HC), float("nan"), device=device)
        k2_key = k2.K2Build(tokens=s, tp=TP, oq=False, og=True)
        scratch2 = torch.zeros(
            k2.scratch_bytes(k2_key), dtype=torch.uint8, device=device
        )
        k2.layer_post(
            k2_key,
            core=core,
            gate=gate,
            state=state,
            inj_a=inj,
            w_o=w_o,
            **mlp_hw,
            w_gate=mw["w_gate"],
            w_sg=mw["w_sg"],
            w13=mw["w13"],
            w13s=mw["w13s"],
            w2=mw["w2"],
            w2s=mw["w2s"],
            out=state2,
            msq_out=msq,
            scratch=scratch2,
            peers=peers.addresses,
            rank=rank,
            epoch=epoch,
            layer=k2.TAG_SLOTS // 2 + 48,
        )

        h = torch.full((s, HIDDEN), float("nan"), dtype=bf, device=device)
        mixer_inj = torch.full((s, HC), float("nan"), dtype=bf, device=device)
        k1_key2 = k1.K1Build(tokens=s, tp=TP, self_msq=False, front_only=True)
        scratch3 = torch.zeros(
            k1.scratch_bytes(k1_key2), dtype=torch.uint8, device=device
        )
        k1.hc_front(
            k1_key2,
            state=state2,
            msq_in=msq,
            **mixer_hw,
            inj_out=mixer_inj,
            bin_out=h,
            scratch=scratch3,
            epoch=epoch,
            layer=49,
        )
        torch.accelerator.synchronize()

        # reference: the attention HC's mix, o_proj + gate + combine, the
        # MLP HC's mix, the MoE on the kernel's own picks (routing cannot
        # differ), the mixer's mix on the unpadded down
        x_ref, inj_ref, _ = _mix_ref(state, attn_hw)
        assert _cos(x, x_ref) > COS_MIN, s
        gated = (core.float() * torch.sigmoid(gate.float())).to(bf)
        attn_out = _all_reduce_fp32(F.linear(gated, w_o))
        state1_ref = hc_combine(state, attn_out, inj_ref, HC)

        # K2 is test_amd_mono.py's (_k2_worker checks it in depth); here only
        # that the draft wiring drives it: the MLP combine of the kernel's own
        # state1 with the MoE emulated on its own picks, as _k2_worker does
        scr2 = k2.scratch_layout(k2_key)
        off, n = scr2["state1"]
        state1 = scratch2[off : off + n].view(bf).view(s, HH)
        assert _rel_l2(state1, state1_ref) < 1e-2, s
        off, n = scr2["bin"]
        x_mlp = scratch2[off : off + n].view(bf).view(s, HIDDEN)
        ids = _check_route(scratch2, scr2["route"], x_mlp.clone(), mw)
        emu = _moe_emulation(x_mlp.clone(), mw, ids)
        inj_m = _pairs(scratch2, scr2["inj"], s * HC).view(s, HC).to(bf)
        want_state2 = hc_combine(state1.clone(), emu, inj_m, HC)
        assert _cos(state2 - state1, want_state2 - state1) > COS_MIN, s
        torch.testing.assert_close(msq, _msq(state2), rtol=1e-5, atol=1e-3)
        want_state2 = state2

        bin_ref = _mix_cpu(
            want_state2.float(),
            mixer_hw["hc_w"].float(),
            mixer_down.float(),
            mixer_hw["w_up"].float(),
            HC,
            LORA,
        )
        assert _cos(h, bin_ref.to(bf)) > COS_MIN, s

        outs = [torch.empty_like(h) for _ in range(tp_size)]
        dist.all_gather(outs, h)
        assert all(torch.equal(outs[0], o) for o in outs), (s, "ranks disagree")
    peers.close()
