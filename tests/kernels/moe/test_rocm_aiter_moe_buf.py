# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for AITER fused MoE fake `moe_buf` pass-through and dispatch policy env."""

import pytest
import torch

from vllm._aiter_ops import _rocm_aiter_fused_moe_fake
from vllm.envs import environment_variables


def _fused_moe_cpu_inputs():
    hidden_states = torch.randn(8, 32, dtype=torch.float32)
    w1 = torch.randn(4, 32, 64)
    w2 = torch.randn(4, 64, 32)
    topk_weight = torch.randn(8, 2, dtype=torch.float32)
    topk_ids = torch.randint(0, 4, (8, 2), dtype=torch.int32)
    return hidden_states, w1, w2, topk_weight, topk_ids


def test_rocm_aiter_fused_moe_fake_with_moe_buf():
    hidden_states, w1, w2, topk_weight, topk_ids = _fused_moe_cpu_inputs()
    moe_buf = torch.randn(8, 32, dtype=torch.bfloat16)

    out = _rocm_aiter_fused_moe_fake(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        moe_buf=moe_buf,
    )

    assert out.shape == moe_buf.shape
    assert out.dtype == moe_buf.dtype
    assert out.device == moe_buf.device

    out_no_buf = _rocm_aiter_fused_moe_fake(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        moe_buf=None,
    )
    expected = torch.empty_like(hidden_states)
    assert out_no_buf.shape == expected.shape
    assert out_no_buf.dtype == expected.dtype
    assert out_no_buf.device == expected.device


def test_rocm_aiter_fused_moe_fake_without_moe_buf():
    hidden_states, w1, w2, topk_weight, topk_ids = _fused_moe_cpu_inputs()

    out_fp16 = _rocm_aiter_fused_moe_fake(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        moe_buf=None,
        output_dtype=torch.float16,
    )
    expected_fp16 = torch.empty_like(hidden_states, dtype=torch.float16)
    assert out_fp16.shape == expected_fp16.shape
    assert out_fp16.dtype == torch.float16
    assert out_fp16.device == hidden_states.device

    out_default = _rocm_aiter_fused_moe_fake(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        moe_buf=None,
        output_dtype=None,
    )
    expected_default = torch.empty_like(hidden_states)
    assert out_default.shape == expected_default.shape
    assert out_default.dtype == expected_default.dtype


def test_dispatch_policy_env_var(monkeypatch: pytest.MonkeyPatch):
    getter = environment_variables["VLLM_ROCM_AITER_MOE_DISPATCH_POLICY"]

    monkeypatch.delenv("VLLM_ROCM_AITER_MOE_DISPATCH_POLICY", raising=False)
    assert getter() == 0

    monkeypatch.setenv("VLLM_ROCM_AITER_MOE_DISPATCH_POLICY", "2")
    assert getter() == 2
