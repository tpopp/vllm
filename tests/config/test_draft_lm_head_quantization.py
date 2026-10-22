# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation of `draft_lm_head_quantization="mxfp4"`, which only ROCm's MXFP4
linear kernel supports. "fp8"/"nvfp4" are not platform-gated here: their own
kernel selection rejects an unsupported platform."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.v1.worker.gpu.spec_decode.eagle.config import (
    check_draft_lm_head_quantization,
)


def _spec(quantization: str) -> SimpleNamespace:
    draft = SimpleNamespace(dtype="bfloat16", head_dtype="bfloat16")
    return SimpleNamespace(
        draft_lm_head_quantization=quantization, draft_model_config=draft
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("is_rocm", [True, False])
@pytest.mark.parametrize("quantization", ["fp8", "nvfp4", "mxfp4"])
def test_only_mxfp4_is_rocm_gated(quantization: str, is_rocm: bool):
    with patch("vllm.platforms.current_platform.is_rocm", return_value=is_rocm):
        if quantization == "mxfp4" and not is_rocm:
            with pytest.raises(ValueError, match="only supported on ROCm"):
                check_draft_lm_head_quantization(_spec(quantization))
        else:
            check_draft_lm_head_quantization(_spec(quantization))


@pytest.mark.cpu_test
def test_mxfp4_lm_head_uses_online_mxfp4_linear_method():
    from vllm.model_executor.layers.quantization.online.lm_head import (
        _LM_HEAD_METHODS,
    )
    from vllm.model_executor.layers.quantization.online.mxfp4 import (
        Mxfp4OnlineLinearMethod,
    )

    assert _LM_HEAD_METHODS["mxfp4"] is Mxfp4OnlineLinearMethod
