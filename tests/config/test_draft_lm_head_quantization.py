# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`SpeculativeConfig._verify_args` gating for `draft_lm_head_quantization`.

"mxfp4" additionally needs AITER's ROCm MXFP4 GEMM kernel
(`vllm/model_executor/layers/quantization/online/lm_head.py::_LMHeadMxfp4Method`),
unlike "fp8"/"nvfp4" which are not platform-gated here (their kernel
selection fails on its own, at model-build time, on an unsupported
platform). These tests exercise just the validator method on a
minimally-populated, un-validated instance (`object.__new__` plus the
handful of attributes `_verify_args` actually reads up to this point),
instead of constructing a full `SpeculativeConfig` -- which would otherwise
need a mocked draft/target `ModelConfig` pair uninvolved in what is being
tested here.
"""

from unittest.mock import patch

import pytest

from vllm.config.speculative import SpeculativeConfig


def _bare_speculative_config(**fields) -> SpeculativeConfig:
    """A `SpeculativeConfig` with only the fields `_verify_args` reads before
    (and including) the `draft_lm_head_quantization` block set, bypassing
    pydantic construction/validation entirely."""
    defaults = dict(
        tensor_parallel_size=None,
        num_speculative_tokens=3,
        method="mtp",
        draft_model_config=None,
        draft_lm_head_quantization=None,
    )
    defaults.update(fields)
    config = object.__new__(SpeculativeConfig)
    for key, value in defaults.items():
        object.__setattr__(config, key, value)
    return config


@pytest.mark.cpu_test
def test_mxfp4_rejected_off_rocm():
    config = _bare_speculative_config(draft_lm_head_quantization="mxfp4")
    with (
        patch("vllm.platforms.current_platform.is_rocm", return_value=False),
        pytest.raises(ValueError, match="only supported on ROCm"),
    ):
        config._verify_args()


@pytest.mark.cpu_test
def test_mxfp4_allowed_on_rocm():
    config = _bare_speculative_config(draft_lm_head_quantization="mxfp4")
    with patch("vllm.platforms.current_platform.is_rocm", return_value=True):
        config._verify_args()  # must not raise


@pytest.mark.cpu_test
@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_fp8_and_nvfp4_are_not_rocm_gated(quantization):
    """Only "mxfp4" is platform-gated in `_verify_args`; "fp8"/"nvfp4" must
    pass this validator regardless of `current_platform.is_rocm()` (their own
    kernel-selection code is what rejects an unsupported platform)."""
    config = _bare_speculative_config(draft_lm_head_quantization=quantization)
    with patch("vllm.platforms.current_platform.is_rocm", return_value=False):
        config._verify_args()  # must not raise


@pytest.mark.cpu_test
def test_mxfp4_still_rejected_for_unsupported_method():
    config = _bare_speculative_config(
        draft_lm_head_quantization="mxfp4", method="dspark"
    )
    with (
        patch("vllm.platforms.current_platform.is_rocm", return_value=True),
        pytest.raises(ValueError, match="not supported for method"),
    ):
        config._verify_args()


@pytest.mark.cpu_test
def test_lm_head_methods_registers_mxfp4_method():
    from vllm.model_executor.layers.quantization.online.lm_head import (
        _LM_HEAD_METHODS,
        _LMHeadMxfp4Method,
    )

    assert _LM_HEAD_METHODS["mxfp4"] is _LMHeadMxfp4Method
