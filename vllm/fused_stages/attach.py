# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model-facing half: how a model declares replaceable spans and hands
control to a provider for planned steps.

Usage in a decoder layer::

    self.stages = attach_stages(
        self, layer_id=layer_id, stages=(DECODER_LAYER,),
        binding=decoder_layer_binding, attention_layers=(self.attn.prefix,),
    )

    def forward(self, x, positions, residual, ...):
        out = self.stages.try_run(DECODER_LAYER, x=x, residual=residual, ...)
        if out is not None:
            return out
        ...  # the reference code path, unchanged

``try_run`` costs a dict lookup when nothing is planned. The decision was
made once for the step's graph key (see ``FusedStageManager.plan_step``);
the model never inspects attention metadata itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from vllm.fused_stages import state
from vllm.fused_stages.api import REFERENCE, LayerBinding, StageSpec, StepContext
from vllm.fused_stages.views import build_view
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.fused_stages.manager import Assignment, FusedStep

logger = init_logger(__name__)

BindingFn = Callable[[nn.Module], LayerBinding]


class StageHandle:
    """Per-layer handle returned by ``attach_stages``."""

    def __init__(
        self,
        module: nn.Module,
        layer_id: int,
        stages: Sequence[StageSpec],
        binding: BindingFn,
        attention_layers: Sequence[str],
    ) -> None:
        self.module = module
        self.layer_id = layer_id
        self.stages = {spec.id: spec for spec in stages}
        self.binding = binding
        self.attention_layers = tuple(attention_layers)
        self._configured = {spec.id: _configured(spec) for spec in stages}

    def binding_for(self, spec: StageSpec) -> LayerBinding:
        b = self.binding(self.module)
        if not b.attention_layers and spec.subsumes_attention:
            b = replace(b, attention_layers=self.attention_layers)
        return b

    def configured(self, spec: StageSpec) -> bool:
        """True if a non-reference provider is configured for ``spec``.

        Decided from config alone at module construction, identically on
        every rank. Use it for structural choices that must be made before
        weights are loaded, e.g. building ``wo_b`` with ``reduce_results=False``
        for a seam with the ``"tp_partial_sum"`` option. The model must then
        perform the reduction itself whenever ``try_run`` returns None."""
        return self._configured[spec.id]

    def try_run(self, spec: StageSpec, /, **tensors: Any) -> tuple | None:
        """Run ``spec`` on its planned provider; None means: run the
        reference code. Exit tensors are returned in ``spec.exit`` order."""
        if torch.compiler.is_compiling():  # v0: compiled models are reference
            return None
        step = _current_step()
        if step is None:
            return None
        assignment = step.plan.assignments.get((spec.id, self.layer_id))
        if assignment is None:
            return None
        return _run(self, spec, assignment, step, tensors)


def attach_stages(
    module: nn.Module,
    *,
    layer_id: int,
    stages: Sequence[StageSpec],
    binding: BindingFn,
    attention_layers: Sequence[str] = (),
) -> StageHandle:
    """Declare that ``module`` (one layer) has replaceable spans. Call from
    ``__init__``; ``binding`` is called after weights are processed."""
    handle = StageHandle(module, layer_id, stages, binding, attention_layers)
    state._PENDING_HANDLES.append(handle)
    return handle


# ------------------------------------------------------------------ internals


def _configured(spec: StageSpec) -> bool:
    from vllm.config import get_current_vllm_config

    try:
        priority = get_current_vllm_config().kernel_config.fused_stages.priority_for(
            spec.id
        )
    except Exception:  # no config (unit tests building modules directly)
        return False
    return any(p != REFERENCE for p in priority)


def _current_step() -> FusedStep | None:
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    return get_forward_context().fused_step


def _resolve(dims: tuple[int | str, ...], rows: int) -> tuple[int, ...]:
    return tuple(rows if d == "M" else int(d) for d in dims)


def _run(
    handle: StageHandle,
    spec: StageSpec,
    assignment: Assignment,
    step: FusedStep,
    tensors: dict[str, Any],
) -> tuple | None:
    from vllm.forward_context import get_forward_context

    fc = get_forward_context()
    inputs: dict[str, torch.Tensor] = {}
    for name, ts in spec.entry.tensors:
        t = tensors.get(name)
        if t is None:
            if ts.optional:
                continue
            return step.fallback(spec, f"missing seam input {name}")
        inputs[name] = t
    rows = next(iter(inputs.values())).shape[0]

    view = None
    if assignment.step_view is not None:
        view = build_view(
            assignment.step_view, spec.family, handle.module, fc.attn_metadata
        )
        if view is None:
            return step.fallback(spec, "step view unavailable")

    device = next(iter(inputs.values())).device
    outputs = {
        name: torch.empty(_resolve(ts.dims, rows), dtype=ts.dtype, device=device)
        for name, ts in spec.exit.tensors
    }
    stream = torch.cuda.current_stream() if device.type == "cuda" else None
    ctx = StepContext(step.plan.key, view, stream)

    runtime = assignment.runtime
    if id(runtime) not in step.begun:
        runtime.begin_step(ctx)
        step.begun.add(id(runtime))

    if assignment.kv_hooks:
        _kv_wait(handle.attention_layers)
    runtime.run(handle.layer_id, ctx, inputs, outputs)
    if assignment.kv_hooks:
        _kv_save(handle.attention_layers)

    step.ran(spec)
    return tuple(outputs[name] for name in spec.exit.names)


def _kv_wait(layer_names: Sequence[str]) -> None:
    """Mirror of ``maybe_transfer_kv_layer``'s entry half for every attention
    layer the span subsumes (layer-wise KV connectors)."""
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
        is_v1_kv_transfer_group,
    )

    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return
    connector = get_kv_transfer_group()
    if not connector.has_connector_metadata():
        return
    for name in layer_names:
        connector.wait_for_layer_load(name)


def _kv_save(layer_names: Sequence[str]) -> None:
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
        is_v1_kv_transfer_group,
    )
    from vllm.model_executor.layers.attention.attention import get_attention_context

    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return
    connector = get_kv_transfer_group()
    if not connector.has_connector_metadata():
        return
    for name in layer_names:
        attn_metadata, _, kv_cache, _ = get_attention_context(name)
        if attn_metadata is not None:
            connector.save_kv_layer(name, kv_cache, attn_metadata)
