# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model Runner V2 integration of fused stages (CPU).

V2 (the default runner, and the only one implementing DSpark) replays FULL
graphs without creating a forward context, so the fused-stage plan taken in
the capture forward context is what every replay runs. These tests check the
V2 descriptor -> planning-key conversion, the capture key list handed to
provider warmup, and that warmup runs once per key across the profiling and
real captures.
"""

from types import SimpleNamespace

from vllm.config import CUDAGraphMode
from vllm.fused_stages.api import Regime
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    fused_capture_descs,
    fused_plan_descriptor,
)

from .test_planning import Layer, Provider, Runtime, _clean, make_manager  # noqa: F401

DSPARK_Q = 6  # 1 + 5 drafts


def full_desc(reqs: int, ubatches: int = 1) -> BatchExecutionDescriptor:
    return BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=reqs * DSPARK_Q,
        num_reqs=reqs,
        uniform_token_count=DSPARK_Q,
        num_ubatches=ubatches,
    )


def test_v2_capture_descriptor_plans(monkeypatch):
    layers = [Layer(i) for i in range(3)]
    m = make_manager(monkeypatch, Provider(), layers)
    key_desc = fused_plan_descriptor(full_desc(8), has_lora=False)
    plan = m.plan_step(key_desc, CUDAGraphMode.FULL, None, {})
    assert plan is not None
    assert plan.key.regime == Regime.FULL_GRAPH and plan.key.uniform_decode
    # The eager warmup forward that precedes each capture plans the same
    # layers (EAGER regime), so JIT happens outside stream capture.
    eager = m.plan_step(key_desc, CUDAGraphMode.NONE, None, {})
    assert eager is not None and set(eager.assignments) == set(plan.assignments)


def test_v2_varlen_decode_graph_is_reference(monkeypatch):
    layers = [Layer(i) for i in range(3)]
    m = make_manager(monkeypatch, Provider(), layers)
    varlen = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=40,
        num_reqs=8,
        uniform_token_count=None,
        max_query_len=DSPARK_Q,
    )
    assert (
        m.plan_step(
            fused_plan_descriptor(varlen, has_lora=False), CUDAGraphMode.FULL, None, {}
        )
        is None
    )


def test_v2_capture_keys_and_idempotent_warmup(monkeypatch):
    class CountingRuntime(Runtime):
        warmed: list = []

        def warmup(self, steps):
            self.warmed.extend(steps)

    provider = Provider()
    provider.runtime = CountingRuntime()
    layers = [Layer(i) for i in range(3)]
    m = make_manager(monkeypatch, provider, layers)

    cg_manager = SimpleNamespace(
        _capture_descs={
            CUDAGraphMode.FULL: [full_desc(8), full_desc(1), full_desc(4, ubatches=2)]
        }
    )
    descs = fused_capture_descs(cg_manager, has_lora=False)
    assert [d.num_tokens for _, ds in descs for d in ds] == [48, 6]  # no DBO desc

    m.warmup(descs)  # profiling capture
    m.warmup(descs)  # real capture: nothing new to warm up
    assert sorted(k.num_tokens for k in provider.runtime.warmed) == [6, 48]
