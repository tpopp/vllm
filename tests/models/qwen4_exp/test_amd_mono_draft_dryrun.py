# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only "dry run" of the Qwen3.8-Flash-Next MTP mono draft step
(``mono_draft.DraftMonoDecode``): exercises the host-side Python wiring of
``DraftMonoDecode.forward`` / ``_fused_front`` / ``_front`` / ``top_tokens`` /
``prepare_weights`` / ``weights_unsupported`` / ``eligible`` / ``head_eligible``
against the real ``k1.hc_front`` / ``k2.layer_post`` / ``front.draft_front`` /
``head.draft_head`` host launchers -- without a GPU.

The launchers run real ``need(...)`` argument checks (shape / dtype /
contiguity; kept even under ``python -O``, see ``mono/layout.py``) before
building a FlyDSL kernel, so this catches host-side argument-order, shape,
dtype and contiguity mistakes before any scarce GPU time is spent. FlyDSL /
AITER are stubbed (fake modules in ``sys.modules``, installed before the
launcher modules are imported) so the module-level ``import flydsl...`` /
``from aiter.ops.flydsl...`` succeed; each launcher module's own ``build()``
is then replaced with a recorder, so the kernel itself (and its JIT-building
decorators) never runs -- only the real pre-``build()`` host code does.

On top of the ``need()`` checks, every launcher call's packed arguments are
cross-checked by ``data_ptr()`` identity against the exact weight / state
tensor ``forward()`` is supposed to feed it, catching an argument swap
(e.g. the attention HC's weights going to the MLP HC's launch) that two
same-shaped tensors would hide from a shape check alone.

GPU-only: the kernels' own correctness (FlyDSL compilation, the actual HC /
MoE / GDN arithmetic) -- see ``test_amd_mono_draft.py``'s
``test_draft_step_matches_vllm_ops`` (skipped off MI355X).
"""

import importlib.machinery
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

bf = torch.bfloat16
TP = 2
LAYER_IDX = 48  # as the mono kernels' own tests use: a plausible MTP layer index

# ---------------------------------------------------------------- flydsl/aiter
# Every dotted module path ``k1.py`` / ``k2.py`` / ``front.py`` / ``head.py``
# (and the ``kimi_k3`` / ``qwen3_5`` mono common code they import) reach via
# ``import`` or ``from ... import ...``, at module level. None of it is
# actually exercised here: a module's own ``build()`` (the only place a
# FlyDSL-decorated kernel is defined) is replaced with a recorder before any
# launcher is called, so a bare auto-vivifying mock is enough to satisfy the
# imports.
_FAKE_MODULES = (
    "flydsl",
    "flydsl.compiler",
    "flydsl.compiler.ast_rewriter",
    "flydsl.expr",
    "flydsl.expr.typing",
    "flydsl._mlir",
    "flydsl._mlir.ir",
    "flydsl._mlir.dialects",
    "flydsl._mlir.dialects.llvm",
    "aiter",
    "aiter.ops",
    "aiter.ops.flydsl",
    "aiter.ops.flydsl.kernels",
    "aiter.ops.flydsl.kernels.buffer_ops",
    "aiter.ops.flydsl.kernels.dpp_utils",
    "aiter.ops.flydsl.kernels.kernels_common",
    "aiter.ops.flydsl.quick_allreduce_int4_ipc",
)


class _AutoMockModule(types.ModuleType):
    """A module whose every attribute access auto-vivifies a ``MagicMock``:
    enough for ``import flydsl.compiler as flyc`` / ``from flydsl.expr import
    gpu, range_constexpr`` etc. to succeed without a real FlyDSL install."""

    def __getattr__(self, name):
        val = MagicMock()
        setattr(self, name, val)
        return val


def _fake_module(name: str) -> types.ModuleType:
    mod = _AutoMockModule(name)
    # a real ModuleSpec: importlib.util.find_spec (vllm._aiter_ops's
    # is_aiter_found) raises ValueError on a sys.modules entry with a None spec
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    mod.__path__ = []
    return mod


def _install_fake_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _FAKE_MODULES:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, _fake_module(name))


def _patch_build(monkeypatch: pytest.MonkeyPatch, module) -> list:
    """Replace ``module.build`` (what every host launcher calls) with a
    recorder: every call's ABI-packed values, by name (so a call can be
    checked against the exact weight / state tensor it should carry), keyed
    by its build key (tokens / tp / front_only / self_msq / ...)."""
    records: list = []

    def fake_build(key):
        names = module.ABI.names

        def launch(*args, stream=None):
            records.append({"key": key, "args": dict(zip(names, args))})

        return launch

    monkeypatch.setattr(module, "build", fake_build)
    return records


def _u8(*shape: int) -> torch.Tensor:
    return torch.randint(0, 255, shape, dtype=torch.uint8)


def _randn_bf(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(bf)


class _Weight:
    """A plain ``.weight`` holder (and any extra attributes): what
    ``weights_unsupported`` / the launchers read off a vLLM ``Linear`` or
    norm module, without needing a real one."""

    def __init__(self, weight: torch.Tensor, **extra):
        self.weight = weight
        for k, v in extra.items():
            setattr(self, k, v)


class _ColLinear:
    """A ``ColumnParallelLinear(gather_output=True)`` stand-in: ``.weight``
    is the local (TP-sharded) shape ``_fc_error`` checks; calling it emulates
    the internal all-gather by matmul-ing the full weight, so the eager
    ``_front`` path sees the right output width."""

    def __init__(self, full_weight: torch.Tensor, tp: int, quant_method):
        self._full_weight = full_weight
        self.weight = full_weight[: full_weight.shape[0] // tp].clone()
        self.quant_method = quant_method

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self._full_weight)


class _Norm:
    """A ``GemmaRMSNorm`` stand-in: shape-preserving (its own numerics are
    GPU-tested elsewhere; here only the host wiring around it matters)."""

    def __init__(self, dim: int, eps: float):
        self.weight = _randn_bf(dim) * 0.1
        self.variance_epsilon = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeTPGroup:
    world_size = TP
    rank_in_group = 0
    cpu_group = None


class _FakePeerBuffer:
    """``PeerBuffer`` needs a real IPC heap (``aiter``) and a real process
    group; the host launchers only ever read ``.bytes`` / ``.addresses``."""

    def __init__(self, nbytes, group, rank, npes, device):
        del group, rank
        self.bytes = torch.zeros(max(int(nbytes), 1), dtype=torch.uint8, device=device)
        self.addresses = torch.zeros(npes, dtype=torch.int64, device=device)


def _hc_module(layout, down_rows: int, eps: float = 1e-6) -> types.SimpleNamespace:
    """A ``GatedResidual(use_combine=True)`` stand-in (the attention / MLP
    hyper-connections): real production shapes at ``down_rows`` rows."""
    hc = types.SimpleNamespace()
    hc.hc_norm = _Weight(_randn_bf(layout.HH) * 0.1)
    hc.input_mix_weight_down_block_inject = _Weight(
        _randn_bf(down_rows, layout.HH) * 0.01
    )
    hc.input_mix_weight_up = _Weight(_randn_bf(layout.HH, layout.LORA) * 0.05)
    hc.use_combine = True
    hc.pad_size = layout.DOWN_ROWS - layout.LORA - layout.HC
    hc.config = types.SimpleNamespace(rms_norm_eps=eps)
    return hc


@pytest.fixture
def mono_env(monkeypatch: pytest.MonkeyPatch):
    """Everything a ``DraftMonoDecode`` dry run needs: FlyDSL / AITER
    stubbed, the collectives (``get_tp_group`` / ``tensor_model_parallel_
    all_gather`` / ``PeerBuffer``) faked for a single CPU "rank", every
    launcher module's ``build()`` replaced with a recorder, ``_qsa_core``
    faked (the real QSA attention is GPU / Triton-only, out of scope here),
    and a fake draft model object graph at Qwen3.8-Flash-Next's real TP2
    shapes."""
    _install_fake_modules(monkeypatch)

    # torch.cuda / torch.accelerator calls the host launchers and vLLM's ROCm
    # platform probe make on a machine with no GPU at all
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a, **k: types.SimpleNamespace(gcnArchName="gfx950"),
        raising=False,
    )
    monkeypatch.setattr(
        torch.accelerator, "current_device_index", lambda: 0, raising=False
    )

    import vllm.distributed as vdist
    import vllm.platforms.rocm as vrocm
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.kimi_k3.amd.mono.common import peer_memory as pm
    from vllm.models.qwen3_5.amd import mono_base as B
    from vllm.models.qwen3_5.amd.mono import layout as QL
    from vllm.models.qwen4_exp.amd import mono_draft
    from vllm.models.qwen4_exp.amd.model import Qwen4ExpSparseMoeBlock
    from vllm.models.qwen4_exp.amd.mono import back, front, head, k1, k2
    from vllm.models.qwen4_exp.amd.mono import layout as L
    from vllm.models.qwen4_exp.amd.qsa import Qwen4ExpQSAAttention
    from vllm.platforms import current_platform

    lay = L.Layout(TP)
    H, HC, HH = L.HIDDEN, L.HC, L.HH
    LORA, DOWN_ROWS = L.LORA, L.DOWN_ROWS
    E, TOPK, CORE, RIP = L.E, L.TOPK, lay.CORE, lay.RIP
    NE = E + 1

    # ---- collectives / platform probes every rank reaches alike
    monkeypatch.setattr(vdist, "get_tp_group", lambda: _FakeTPGroup())
    monkeypatch.setattr(
        vdist, "get_pp_group", lambda: types.SimpleNamespace(world_size=1)
    )
    # mono_base.py imports get_tp_group / get_pp_group eagerly at module
    # level: patching vllm.distributed alone would miss that bound reference
    monkeypatch.setattr(B, "get_tp_group", lambda: _FakeTPGroup())
    monkeypatch.setattr(B, "get_pp_group", lambda: types.SimpleNamespace(world_size=1))
    monkeypatch.setattr(
        vdist,
        "tensor_model_parallel_all_gather",
        lambda t, dim=0: torch.cat([t, t], dim=dim).contiguous(),
    )
    monkeypatch.setattr(vrocm, "get_cdna_version", lambda: 4)
    monkeypatch.setattr(current_platform, "num_compute_units", lambda *a, **k: 10**6)
    monkeypatch.setattr(pm, "PeerBuffer", _FakePeerBuffer)
    monkeypatch.setattr(mono_draft, "fused_head", lambda: True)

    # ---- the launchers: real need() checks, build() replaced with a recorder
    records = {
        "k1": _patch_build(monkeypatch, k1),
        "k2": _patch_build(monkeypatch, k2),
        "front": _patch_build(monkeypatch, front),
        "head": _patch_build(monkeypatch, head),
        "back": _patch_build(monkeypatch, back),
    }

    # ---- _qsa_core (GPU / Triton QSA attention): a fixed random (core, gate)
    # of the right per-rank shape, recording what it was called with so a
    # test can check the HC front's bin_out reaches it unchanged
    qsa_calls: list = []

    def fake_qsa_core(attn, x, positions):
        del attn
        s = x.size(0)
        assert x.shape == (s, H) and x.dtype == bf and x.is_contiguous()
        assert positions.shape == (s,)
        core = torch.randn(s, CORE).to(bf).contiguous()
        gate = torch.randn(s, CORE).to(bf).contiguous()
        qsa_calls.append(
            {
                "x_ptr": x.data_ptr(),
                "core_ptr": core.data_ptr(),
                "gate_ptr": gate.data_ptr(),
            }
        )
        return core, gate

    monkeypatch.setattr(mono_draft, "_qsa_core", fake_qsa_core)

    # ---- the fake draft model object graph, real production shapes at TP2
    layer = types.SimpleNamespace()
    layer.layer_idx = LAYER_IDX
    layer.layer_type = "qwen_sparse_attention"  # anything but "linear_attention"
    layer.ple = None

    a = object.__new__(Qwen4ExpQSAAttention)
    nn.Module.__init__(a)
    a.num_heads = CORE // 256
    a.head_dim = 256
    a.attn_output_gate = True
    a.o_proj = _Weight(
        _randn_bf(H, CORE) * 0.02, quant_method=UnquantizedLinearMethod()
    )
    a.indexer = types.SimpleNamespace(
        raw_key_cache=types.SimpleNamespace(
            prefix=f"model.layers.{LAYER_IDX}.self_attn.indexer.raw_key_cache"
        )
    )
    layer.self_attn = a

    mlp = object.__new__(Qwen4ExpSparseMoeBlock)
    nn.Module.__init__(mlp)
    mlp.is_fused_shared_expert_enabled = True
    mlp.gate = _Weight(_randn_bf(E, H) * 0.02)
    mlp.shared_expert_gate = _Weight(_randn_bf(1, H) * 0.02)

    ex = types.SimpleNamespace()
    ex.quant_method = types.SimpleNamespace(
        mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4
    )
    ex.w13_weight = _u8(NE, 2 * RIP, H // 2)
    ex.w13_weight.is_shuffled = True
    ex.w2_weight = _u8(NE, H, RIP // 2)
    ex.w13_weight_scale = _u8(NE, 2 * RIP, QL.scale_cols(H))
    ex.w2_weight_scale = _u8(NE, H, QL.scale_cols(RIP))
    mlp.experts = types.SimpleNamespace(routed_experts=ex)
    layer.mlp = mlp

    layer.attn_hyper_connection = _hc_module(L, DOWN_ROWS)
    layer.mlp_hyper_connection = _hc_module(L, DOWN_ROWS)

    model = types.SimpleNamespace()
    model.layers = [layer]
    model.config = types.SimpleNamespace(
        hidden_size=H,
        hc_count=HC,
        hc_lowrank=LORA,
        num_experts=E,
        num_experts_per_tok=TOPK,
        moe_intermediate_size=L.MOE_I,
        shared_expert_intermediate_size=L.MOE_I,
        hidden_act="silu",
        norm_topk_prob=True,
    )

    mixer = types.SimpleNamespace()
    mixer.use_combine = False
    mixer.hc_norm = _Weight(_randn_bf(HH) * 0.1)
    mixer.input_mix_weight_up = _Weight(_randn_bf(HH, LORA) * 0.05)
    mixer.input_mix_weight_down = _Weight(_randn_bf(LORA, HH) * 0.01)
    mixer.config = types.SimpleNamespace(rms_norm_eps=1e-6)
    model.hyper_connection_mixer = mixer

    full_fe = _randn_bf(H, H) * 0.05
    full_fh = _randn_bf(H, H) * 0.05
    model.fc_embedding = _ColLinear(full_fe, TP, UnquantizedLinearMethod())
    model.fc_hidden = _ColLinear(full_fh, TP, UnquantizedLinearMethod())
    # distinct from the HC fronts' hardcoded eps (1e-6): a kernel that
    # swapped ne_eps / nh_eps would still pass with equal values
    model.pre_fc_norm_embedding = _Norm(H, 1e-5)
    model.pre_fc_norm_hidden = _Norm(HH, 3e-6)
    model.embed_tokens = types.SimpleNamespace(weight=torch.zeros(1, device="cpu"))

    embed_calls: list = []

    def embed_input_ids(ids):
        out = _randn_bf(ids.size(0), H)
        embed_calls.append(out.data_ptr())
        return out

    model.embed_input_ids = embed_input_ids

    vllm_config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(
            enable_expert_parallel=False,
            enable_eplb=False,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        lora_config=None,
        speculative_config=types.SimpleNamespace(method="mtp"),
        kv_transfer_config=None,
    )

    lm_head = types.SimpleNamespace(
        weight=_u8(124160, H // 2),
        weight_scale=_u8(124160 * (H // 32)),
        shard_indices=types.SimpleNamespace(
            org_vocab_start_index=0, org_vocab_end_index=124160
        ),
        tp_size=TP,
    )
    # the lm_head is the online-MXFP4 ASM-GEMM layout asm_mxfp4 detects via
    # isinstance on its quant_method; o_proj (and fc_embedding / fc_hidden)
    # stay dense bf16, so only the lm_head should ever report True
    monkeypatch.setattr(B, "asm_mxfp4", lambda lin: lin is lm_head)

    return types.SimpleNamespace(
        mono_draft=mono_draft,
        k1=k1,
        k2=k2,
        front=front,
        head=head,
        back=back,
        L=L,
        lay=lay,
        model=model,
        vllm_config=vllm_config,
        lm_head=lm_head,
        records=records,
        qsa_calls=qsa_calls,
        embed_calls=embed_calls,
    )


def _make_mono(
    env, monkeypatch: pytest.MonkeyPatch, *, fused_front: bool, use_back: bool = False
):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_ROCM_MONO_DRAFT_FRONT", fused_front)
    monkeypatch.setattr(envs, "VLLM_ROCM_MONO_DRAFT_BACK", use_back)
    mono = env.mono_draft.DraftMonoDecode(env.model, env.vllm_config)
    assert mono.ok, "the fake model graph should pass unsupported()"
    mono.prepare_weights()
    mono.check_weights()
    assert mono.ok, "the fake model graph's weights should pass weights_unsupported()"
    assert (mono._fc is not None) == fused_front
    assert (mono._back is not None) == use_back
    return mono


# ---------------------------------------------------------------- construction
@pytest.mark.cpu_test
@pytest.mark.parametrize("fused_front", [False, True])
def test_prepare_and_check_weights(mono_env, monkeypatch, fused_front: bool) -> None:
    """``DraftMonoDecode.__init__`` / ``prepare_weights`` / ``check_weights``
    against the fake model graph's real production shapes: ``ok`` stays
    True, and ``_fc`` (the fused front's gathered weights) is built only
    when ``VLLM_ROCM_MONO_DRAFT_FRONT=1``."""
    _make_mono(mono_env, monkeypatch, fused_front=fused_front)


# ---------------------------------------------------------------- forward
@pytest.mark.cpu_test
@pytest.mark.parametrize("s", [1, 3, 8])
@pytest.mark.parametrize("fused_front", [False, True])
def test_forward_wiring(mono_env, monkeypatch, s: int, fused_front: bool) -> None:
    """``forward``'s three launches (the attention HC front / ``front.
    draft_front``, K2, the mixer HC front) against real ``need()`` checks and
    exact (``data_ptr()``) argument identity: every weight / state tensor the
    launch receives is the one ``forward`` is supposed to feed it, not merely
    a same-shaped stand-in."""
    env = mono_env
    L = env.L
    mono = _make_mono(env, monkeypatch, fused_front=fused_front)

    layer = env.model.layers[0]
    attn_hc = layer.attn_hyper_connection
    mlp_hc = layer.mlp_hyper_connection
    mixer = env.model.hyper_connection_mixer
    o_proj = layer.self_attn.o_proj
    mlp = layer.mlp
    ex = mlp.experts.routed_experts

    input_ids = torch.randint(0, 1000, (s,), dtype=torch.int64)
    positions = torch.arange(s, dtype=torch.int64)
    hidden_states = torch.randn(s, L.HH).to(bf)

    for rec in env.records.values():
        rec.clear()
    env.qsa_calls.clear()
    env.embed_calls.clear()

    epoch_before = int(mono.epoch.item())
    h, state2 = mono.forward(input_ids, positions, hidden_states, None)
    assert int(mono.epoch.item()) - epoch_before == 1, "epoch bumped once a step"

    assert h.shape == (s, L.HIDDEN) and h.dtype == bf and h.is_contiguous()
    assert state2.shape == (s, L.HH) and state2.dtype == bf

    assert len(env.qsa_calls) == 1
    qsa = env.qsa_calls[0]

    assert len(env.records["k2"]) == 1
    k2_call = env.records["k2"][0]
    k2_args = k2_call["args"]
    k2_key = k2_call["key"]
    assert k2_key.tokens == s and k2_key.tp == TP
    assert k2_args["layer"] == env.k2.TAG_SLOTS // 2 + LAYER_IDX

    # the mixer HC front is always the LAST k1 call; a fused front skips the
    # attention HC's own (front.draft_front takes it over instead)
    assert len(env.records["k1"]) == (1 if fused_front else 2)
    mixer_call = env.records["k1"][-1]
    mixer_args = mixer_call["args"]
    mixer_key = mixer_call["key"]
    assert mixer_key.tokens == s and mixer_key.tp == TP
    assert mixer_key.front_only is True and mixer_key.self_msq is False
    assert mixer_args["layer"] == LAYER_IDX + 1

    if fused_front:
        assert len(env.records["front"]) == 1
        front_call = env.records["front"][0]
        front_args = front_call["args"]
        front_key = front_call["key"]
        assert front_key.tokens == s and front_key.tp == TP
        assert front_args["slot"] == LAYER_IDX

        assert len(env.embed_calls) == 1
        assert front_args["emb"] == env.embed_calls[0]
        assert front_args["prev"] == hidden_states.data_ptr()
        assert front_args["w_ne"] == env.model.pre_fc_norm_embedding.weight.data_ptr()
        assert front_args["w_nh"] == env.model.pre_fc_norm_hidden.weight.data_ptr()
        assert front_args["w_fe"] == mono._fc[0].data_ptr()
        assert front_args["w_fh"] == mono._fc[1].data_ptr()
        assert front_args["hc_w"] == attn_hc.hc_norm.weight.data_ptr()
        assert (
            front_args["w_down"]
            == attn_hc.input_mix_weight_down_block_inject.weight.data_ptr()
        )
        assert front_args["w_up"] == attn_hc.input_mix_weight_up.weight.data_ptr()
        assert front_args["bin_out"] == qsa["x_ptr"]
        front_state = front_args["state_out"]
        front_inj = front_args["inj_out"]
    else:
        assert len(env.records["front"]) == 0
        attn_call = env.records["k1"][0]
        attn_args = attn_call["args"]
        attn_key = attn_call["key"]
        assert attn_key.tokens == s and attn_key.tp == TP
        assert attn_key.front_only is True and attn_key.self_msq is True
        assert attn_args["layer"] == LAYER_IDX
        assert attn_args["msq_in"] == 0, "self_msq: no msq_in"
        assert attn_args["hc_w"] == attn_hc.hc_norm.weight.data_ptr()
        assert (
            attn_args["w_down"]
            == attn_hc.input_mix_weight_down_block_inject.weight.data_ptr()
        )
        assert attn_args["w_up"] == attn_hc.input_mix_weight_up.weight.data_ptr()
        assert attn_args["bin_out"] == qsa["x_ptr"]
        front_state = attn_args["state"]
        front_inj = attn_args["inj_out"]

    # the attention HC front's state / injection feed K2 unchanged
    assert k2_args["state"] == front_state
    assert k2_args["inj_a"] == front_inj
    assert k2_args["core"] == qsa["core_ptr"]
    assert k2_args["gate"] == qsa["gate_ptr"]
    assert k2_args["w_o"] == o_proj.weight.data_ptr()
    assert k2_args["w_os"] == 0, "o_proj is dense bf16 here, not online-MXFP4"
    assert k2_args["hc_w"] == mlp_hc.hc_norm.weight.data_ptr()
    assert (
        k2_args["w_down"] == mlp_hc.input_mix_weight_down_block_inject.weight.data_ptr()
    )
    assert k2_args["w_up"] == mlp_hc.input_mix_weight_up.weight.data_ptr()
    assert k2_args["w_gate"] == mlp.gate.weight.data_ptr()
    assert k2_args["w_sg"] == mlp.shared_expert_gate.weight.data_ptr()
    assert k2_args["w13"] == ex.w13_weight.data_ptr()
    assert k2_args["w13s"] == ex.w13_weight_scale.data_ptr()
    assert k2_args["w2"] == ex.w2_weight.data_ptr()
    assert k2_args["w2s"] == ex.w2_weight_scale.data_ptr()
    assert k2_args["out"] == state2.data_ptr()

    # the mixer HC front reads K2's materialized state and its sums of
    # squares, and its own (separate) weights -- never the attention HC's
    assert mixer_args["state"] == k2_args["out"]
    assert mixer_args["msq_in"] == k2_args["msq_out"]
    assert mixer_args["hc_w"] == mixer.hc_norm.weight.data_ptr()
    assert mixer_args["w_down"] == mono._mixer_w_down.data_ptr()
    assert mixer_args["w_up"] == mixer.input_mix_weight_up.weight.data_ptr()
    assert mixer_args["bin_out"] == h.data_ptr()


# ---------------------------------------------------------------- top_tokens
@pytest.mark.cpu_test
@pytest.mark.parametrize("s", [1, 3, 8])
def test_top_tokens_wiring(mono_env, monkeypatch, s: int) -> None:
    """``top_tokens``: one ``head.draft_head`` launch, tag slot 0, the
    hidden-state rows forwarded unchanged, the head's own epoch bumped once
    (never the step epoch ``forward`` bumps)."""
    env = mono_env
    L = env.L
    mono = _make_mono(env, monkeypatch, fused_front=False)

    h = torch.randn(s, L.HIDDEN).to(bf)
    for rec in env.records.values():
        rec.clear()

    head_epoch_before = int(mono._head["epoch"].item())
    step_epoch_before = int(mono.epoch.item())
    out = mono.top_tokens(env.lm_head, h)
    assert int(mono._head["epoch"].item()) - head_epoch_before == 1
    assert int(mono.epoch.item()) == step_epoch_before, "top_tokens: not the step epoch"

    assert out.shape == (s,) and out.dtype == torch.int64

    assert len(env.records["head"]) == 1
    call = env.records["head"][0]
    args, key = call["args"], call["key"]
    assert key.tokens == s and key.tp == TP
    assert key.rows == env.lm_head.weight.size(0)
    assert args["slot"] == 0
    assert args["x"] == h.data_ptr()
    assert args["w"] == env.lm_head.weight.data_ptr()
    assert args["ws"] == env.lm_head.weight_scale.data_ptr()
    assert args["valid"] == 124160
    assert args["vocab_start"] == 0
    assert args["out"] == out.data_ptr()


@pytest.mark.cpu_test
def test_head_eligible_only_for_asm_mxfp4_lm_head(mono_env, monkeypatch) -> None:
    """``head_eligible``: only the online-MXFP4 ASM-GEMM lm_head
    (``asm_mxfp4``) at this rank's TP size is eligible; o_proj is dense bf16
    here and must not be mistaken for it."""
    env = mono_env
    mono = _make_mono(env, monkeypatch, fused_front=False)

    logits_processor = types.SimpleNamespace(soft_cap=None, scale=1.0)
    hidden_states = torch.randn(3, env.L.HIDDEN).to(bf)
    assert mono.head_eligible(env.lm_head, logits_processor, hidden_states) is True

    # not the lm_head asm_mxfp4 was told about
    other = types.SimpleNamespace(tp_size=TP)
    assert mono.head_eligible(other, logits_processor, hidden_states) is False


# ---------------------------------------------------------------- draft_back
@pytest.mark.cpu_test
@pytest.mark.parametrize("s", [1, 3, 8])
def test_forward_uses_back_once_head_eligible_has_run(
    mono_env, monkeypatch, s: int
) -> None:
    """``forward``'s ``_use_back`` decision: ``forward`` never receives
    ``lm_head`` (the MTP wrapper owns it, not the predictor ``forward`` is
    bound to), so it trusts the lm_head-format part of the LAST
    ``head_eligible`` call. Before that has ever run, ``forward`` falls back
    to K2 + the mixer's own ``hc_front`` (the existing wiring, already
    checked by ``test_forward_wiring``); once it has, ``forward`` launches
    ``back.draft_back`` instead (the attention HC front still runs -- only
    the post-QSA-core part is replaced), and ``top_tokens`` returns its ids
    without launching the head again."""
    env = mono_env
    L = env.L
    mono = _make_mono(env, monkeypatch, fused_front=False, use_back=True)

    layer = env.model.layers[0]
    mlp_hc = layer.mlp_hyper_connection
    mixer = env.model.hyper_connection_mixer
    o_proj = layer.self_attn.o_proj
    mlp = layer.mlp
    ex = mlp.experts.routed_experts

    input_ids = torch.randint(0, 1000, (s,), dtype=torch.int64)
    positions = torch.arange(s, dtype=torch.int64)
    hidden_states = torch.randn(s, L.HH).to(bf)

    # ---- before any head_eligible() call: falls back to the unfused wiring
    for rec in env.records.values():
        rec.clear()
    assert mono._use_back() is False
    mono.forward(input_ids, positions, hidden_states, None)
    assert len(env.records["k2"]) == 1
    assert len(env.records["back"]) == 0

    # ---- head_eligible (as get_top_tokens calls it) caches the lm_head check
    logits_processor = types.SimpleNamespace(soft_cap=None, scale=1.0)
    assert mono.head_eligible(env.lm_head, logits_processor, hidden_states) is True
    assert mono._use_back() is True

    # ---- the next step: forward() now fuses via draft_back
    for rec in env.records.values():
        rec.clear()
    h, state2 = mono.forward(input_ids, positions, hidden_states, None)
    assert len(env.records["back"]) == 1
    assert len(env.records["k2"]) == 0
    assert len(env.records["k1"]) == 1, "the attention HC front still runs"

    call = env.records["back"][0]
    args, key = call["args"], call["key"]
    assert key.tokens == s and key.tp == TP
    assert key.rows == env.lm_head.weight.size(0)
    assert args["out"] == state2.data_ptr()
    assert args["h_out"] == h.data_ptr()
    assert args["w_gate"] == mlp.gate.weight.data_ptr()
    assert args["w_sg"] == mlp.shared_expert_gate.weight.data_ptr()
    assert args["w13"] == ex.w13_weight.data_ptr()
    assert args["w2"] == ex.w2_weight.data_ptr()
    assert args["w_o"] == o_proj.weight.data_ptr()
    assert args["hc_w"] == mlp_hc.hc_norm.weight.data_ptr()
    assert (
        args["w_down"] == mlp_hc.input_mix_weight_down_block_inject.weight.data_ptr()
    )
    assert args["w_up"] == mlp_hc.input_mix_weight_up.weight.data_ptr()
    assert args["mix_hc_w"] == mixer.hc_norm.weight.data_ptr()
    assert args["mix_w_down"] == mono._mixer_w_down.data_ptr()
    assert args["mix_w_up"] == mixer.input_mix_weight_up.weight.data_ptr()
    assert args["head_w"] == env.lm_head.weight.data_ptr()
    assert args["head_ws"] == env.lm_head.weight_scale.data_ptr()
    assert args["valid"] == 124160
    assert args["vocab_start"] == 0
    assert args["slot"] == 0

    # ---- top_tokens picks up the pending ids, no further head launch
    assert s in mono._pending_ids
    ids = mono._pending_ids[s]
    assert args["ids"] == ids.data_ptr()
    out = mono.top_tokens(env.lm_head, h)
    assert out is ids
    assert s not in mono._pending_ids, "top_tokens: clears the mark"
    assert len(env.records["head"]) == 0, "top_tokens: no separate head launch"
