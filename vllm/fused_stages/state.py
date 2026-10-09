# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-level fused-stage state, kept dependency-free so that
``vllm.forward_context`` can consult it without import cycles."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.fused_stages.attach import StageHandle
    from vllm.fused_stages.manager import FusedStageManager

# Handles created by attach_stages() during model construction, drained by the
# manager once weights are loaded.
_PENDING_HANDLES: list[StageHandle] = []

_ACTIVE_MANAGER: FusedStageManager | None = None


def set_active_manager(manager: FusedStageManager | None) -> None:
    global _ACTIVE_MANAGER
    _ACTIVE_MANAGER = manager


def get_active_manager() -> FusedStageManager | None:
    return _ACTIVE_MANAGER


def clear_pending_handles() -> None:
    _PENDING_HANDLES.clear()
