# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned views: provider-facing descriptions of vLLM-internal state.

A view type is a frozen dataclass (subclass of ``api.View``) with a
``TYPE_ID`` such as ``"dsv4.sparse_swa_mla.step@1"``. Its *definition* lives
here so providers can import it without importing model or attention-backend
internals. Its *builder* lives with the owner of the internal state (the
attention backend or model directory) and is registered with
``register_view_builder``. When the internal state changes, the owner updates
the builder in the same PR; the view, and therefore every provider, is
unaffected unless the semantics change, in which case the owner adds ``@2``
(and may keep building ``@1`` for a deprecation window).

View definitions are semantic: they describe what a kernel needs (tables,
lengths, layout constants), not a mirror of whichever metadata dataclass
exists today.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm.fused_stages.api import View

# (type_id, family) -> builder(layer_module, step_metadata | None) -> View
_BUILDERS: dict[tuple[str, str], Callable[..., View | None]] = {}


def register_view_builder(type_id: str, family: str):
    """Decorator registering how a model family builds a view type.

    Builders take ``(layer_module, attn_metadata)``: ``attn_metadata`` is the
    forward context's metadata dict for step views and ``None`` for cache
    views (built at KV-cache bind time). They return ``None`` when the view
    cannot be built for this step (e.g. missing metadata), which makes the
    framework run the reference path for that layer and step.
    """

    def deco(fn: Callable[..., View | None]) -> Callable[..., View | None]:
        key = (type_id, family)
        assert key not in _BUILDERS, f"duplicate view builder for {key}"
        _BUILDERS[key] = fn
        return fn

    return deco


def build_view(type_id: str, family: str, layer: Any, metadata: Any) -> View | None:
    builder = _BUILDERS.get((type_id, family))
    if builder is None:
        raise KeyError(f"no builder registered for view {type_id} ({family})")
    return builder(layer, metadata)


def has_view_builder(type_id: str, family: str) -> bool:
    return (type_id, family) in _BUILDERS
