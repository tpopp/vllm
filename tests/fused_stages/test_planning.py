# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of fused-stage selection, refusal and step planning with a fake
provider (no GPU, no distributed init)."""

from collections.abc import Sequence
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.config.kernel import FusedStagesConfig
from vllm.forward_context import (
    BatchDescriptor,
    ForwardContext,
    override_forward_context,
)
from vllm.fused_stages import attach_stages, state
from vllm.fused_stages.api import (
    Acceptance,
    Feature,
    LayerBinding,
    Refusal,
    Regime,
    Seam,
    StageClaim,
    StageContext,
    StageProvider,
    StageRuntime,
    StageSpec,
    StepContext,
    TensorSpec,
    TPContext,
    WeightRef,
)
from vllm.fused_stages.layouts import BF16_ROWMAJOR, MXFP4_AITER_A4W4_SEPARATED
from vllm.fused_stages.manager import FusedStageManager
from vllm.fused_stages.testing import candidate_keys, check_claims, deployment_for_tests

H = 8
SEAM_IN = Seam("in", (("x", TensorSpec(("M", H), torch.float32)),))
SEAM_OUT = Seam("out", (("x", TensorSpec(("M", H), torch.float32)),))
LAYER = StageSpec("toy", "layer", 1, SEAM_IN, SEAM_OUT, "layer", False)


class Layer(nn.Module):
    def __init__(self, layer_id: int, layout: str = MXFP4_AITER_A4W4_SEPARATED):
        super().__init__()
        self.layer_id = layer_id
        self.w = nn.Parameter(torch.randn(H, H), requires_grad=False)
        self.layout = layout
        self.stages = attach_stages(
            self, layer_id=layer_id, stages=(LAYER,), binding=self.binding
        )

    @staticmethod
    def binding(m: "Layer") -> LayerBinding:
        return LayerBinding(
            m.layer_id,
            {"w": WeightRef(m.w, m.layout), "norm": WeightRef(m.w, BF16_ROWMAJOR)},
            {"is_first_layer": m.layer_id == 0},
        )

    def forward(self, x):
        out = self.stages.try_run(LAYER, x=x)
        if out is not None:
            return out[0]
        return x @ self.w


class Runtime(StageRuntime):
    def __init__(self):
        self.weights, self.runs, self.begun = {}, 0, 0

    def bind_layers(self, layers: Sequence[LayerBinding]):
        self.weights = {b.layer_id: b.weights["w"].tensor for b in layers}

    def begin_step(self, ctx: StepContext):
        self.begun += 1

    def run(self, layer_id, ctx, inputs, outputs):
        self.runs += 1
        outputs["x"].copy_(inputs["x"] @ self.weights[layer_id])

    def close(self):
        self.weights.clear()


class Provider(StageProvider):
    provider_id = "toy_mono"
    provider_version = "1.0"

    def __init__(self, supports=frozenset({Feature.ASYNC_SCHEDULING}), max_rows=48):
        self.supports, self.max_rows = supports, max_rows
        self.runtime = Runtime()

    def claims(self):
        return [
            StageClaim(
                stage="toy.layer",
                stage_versions=frozenset({1}),
                regimes=Regime.EAGER | Regime.FULL_GRAPH,
                layouts={"w": frozenset({MXFP4_AITER_A4W4_SEPARATED})},
                supports=self.supports,
            )
        ]

    def accept(self, claim, spec, deployment, layers):
        if deployment.tp_size not in (1, 2, 4):
            raise Refusal("needs TP 1, 2 or 4")
        max_rows = self.max_rows
        return Acceptance(
            layer_ids=frozenset(
                b.layer_id for b in layers if not b.consts["is_first_layer"]
            ),
            steps=lambda k: k.uniform_decode and k.num_tokens <= max_rows,
        )

    def create(self, claim, ctx: StageContext):
        return self.runtime


def make_manager(monkeypatch, provider, layers, priority=None, **dep):
    monkeypatch.setattr(
        "vllm.fused_stages.manager.discover_providers",
        lambda: {provider.provider_id: provider},
    )
    monkeypatch.setattr(
        FusedStageManager,
        "_deployment_info",
        lambda self: deployment_for_tests(**dep),
    )
    monkeypatch.setattr(
        FusedStageManager, "_tp_context", lambda self: TPContext(0, 1, None, None)
    )
    monkeypatch.setattr(FusedStageManager, "_all_gather", lambda self, obj: [obj])
    cfg = SimpleNamespace(
        kernel_config=SimpleNamespace(
            fused_stages=FusedStagesConfig(priority=priority or {"toy.*": ["toy_mono"]})
        )
    )
    model = nn.ModuleList(layers)
    return FusedStageManager.maybe_create(cfg, model, torch.device("cpu"), 6)


@pytest.fixture(autouse=True)
def _clean():
    state._PENDING_HANDLES.clear()
    yield
    state._PENDING_HANDLES.clear()
    state.set_active_manager(None)


def decode_desc(reqs: int, q: int = 6) -> BatchDescriptor:
    return BatchDescriptor(num_tokens=reqs * q, num_reqs=reqs, uniform=True)


def test_selection_and_plan(monkeypatch):
    layers = [Layer(i) for i in range(4)]
    m = make_manager(monkeypatch, Provider(), layers)
    assert m is not None
    (sel,) = m.selected
    assert sel.layer_ids == {1, 2, 3}  # layer 0 refused by the provider

    plan = m.plan_step(decode_desc(8), CUDAGraphMode.FULL, None, {})
    assert plan is not None and plan.key.uniform_decode and plan.key.query_len == 6
    assert set(plan.assignments) == {("toy.layer", i) for i in (1, 2, 3)}
    # over the row limit, mixed batches, DBO and missing metadata: reference
    assert m.plan_step(decode_desc(9), CUDAGraphMode.FULL, None, {}) is None
    mixed = BatchDescriptor(num_tokens=40, num_reqs=None, uniform=False)
    assert m.plan_step(mixed, CUDAGraphMode.PIECEWISE, None, {}) is None
    assert m.plan_step(decode_desc(2), CUDAGraphMode.NONE, object(), {}) is None
    assert m.plan_step(decode_desc(2), CUDAGraphMode.NONE, None, None) is None
    # memoized: same object for the same key
    assert m.plan_step(decode_desc(8), CUDAGraphMode.FULL, None, {}) is plan


def test_try_run_matches_reference(monkeypatch):
    layers = [Layer(i) for i in range(3)]
    provider = Provider()
    m = make_manager(monkeypatch, provider, layers)
    x = torch.randn(12, H)
    ref = [layer(x) for layer in layers]  # no forward context: reference

    plan = m.plan_step(decode_desc(2), CUDAGraphMode.NONE, None, {})
    fc = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        fused_step=m.new_step(plan),
    )
    with override_forward_context(fc):
        got = [layer(x) for layer in layers]
    for r, g in zip(ref, got):
        torch.testing.assert_close(r, g)
    assert provider.runtime.runs == 2  # layers 1 and 2
    assert provider.runtime.begun == 1  # once per step per runtime


@pytest.mark.parametrize(
    "dep, reason",
    [
        ({"features": frozenset({Feature.DBO})}, "dbo"),
        ({"features": frozenset({Feature.KV_OFFLOAD})}, "kv_offload"),
        ({"features": frozenset({Feature.LORA})}, "lora"),
        ({"regimes": Regime.EAGER | Regime.COMPILED}, "compile"),
        ({"tp_size": 8}, "TP"),
    ],
)
def test_refusals_fall_back(monkeypatch, dep, reason):
    reasons = _capture_refusals(monkeypatch)
    layers = [Layer(i) for i in range(3)]
    assert make_manager(monkeypatch, Provider(), layers, **dep) is None
    assert any(reason in r for r in reasons), reasons


def test_layout_mismatch_refuses(monkeypatch):
    reasons = _capture_refusals(monkeypatch)
    layers = [Layer(i, layout="untagged@0") for i in range(3)]
    assert make_manager(monkeypatch, Provider(), layers) is None
    assert any("layout untagged@0" in r for r in reasons), reasons


def _capture_refusals(monkeypatch) -> list[str]:
    reasons: list[str] = []
    monkeypatch.setattr(
        FusedStageManager,
        "_log_refusal",
        lambda self, stage, provider, reason: reasons.append(reason),
    )
    return reasons


def test_reference_only_config_creates_nothing(monkeypatch):
    layers = [Layer(i) for i in range(3)]
    assert (
        make_manager(monkeypatch, Provider(), layers, priority={"*": ["reference"]})
        is None
    )


def test_check_claims_kit():
    layers = [Layer(i) for i in range(3)]
    state._PENDING_HANDLES.clear()
    reports = check_claims(
        Provider(),
        {"toy.layer": LAYER},
        deployment_for_tests(),
        {"toy.layer": [Layer.binding(m) for m in layers]},
        candidate_keys(64, query_lens=(1, 6)),
    )
    assert reports[0].accepted_layers == {1, 2} and reports[0].accepted_keys > 0


class DraftLayer(Layer):
    @staticmethod
    def binding(m: "Layer") -> LayerBinding:
        b = Layer.binding(m)
        return LayerBinding(b.layer_id, b.weights, b.consts, draft=m.layer_id >= 3)


def test_draft_layers_need_opt_in(monkeypatch):
    layers = [DraftLayer(i) for i in range(5)]
    m = make_manager(monkeypatch, Provider(), layers)
    assert m is not None
    (sel,) = m.selected
    assert sel.layer_ids == {1, 2}  # 0: provider refuses; 3, 4: draft


def test_eager_step_planned_from_batch_shape(monkeypatch):
    """Eager dispatch yields BatchDescriptor(num_tokens) only; the runner
    passes the batch shape as the planning descriptor instead."""
    layers = [Layer(i) for i in range(3)]
    m = make_manager(monkeypatch, Provider(), layers)
    bare = BatchDescriptor(num_tokens=12)
    assert m.plan_step(bare, CUDAGraphMode.NONE, None, {}) is None
    plan = m.plan_step(decode_desc(2), CUDAGraphMode.NONE, None, {})
    assert plan is not None and plan.key.regime == Regime.EAGER
