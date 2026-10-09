# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused stages: replaceable spans of a model's forward with pluggable
providers (mono kernels / megakernels).

* ``vllm.fused_stages.api``: the provider contract (the only module providers
  import).
* ``vllm.fused_stages.attach``: how models declare stages and hand control to
  a provider (``attach_stages`` / ``StageHandle.try_run``).
* ``vllm.fused_stages.manager``: selection, collective acceptance, step
  planning and lifecycle, driven by the model runner.
* ``vllm.fused_stages.views`` / ``vllm.fused_stages.layouts``: versioned
  descriptions of attention state and processed weight layouts.

See docs/design/fused_stages.md.
"""

from vllm.fused_stages.attach import StageHandle, attach_stages

__all__ = ["StageHandle", "attach_stages"]
