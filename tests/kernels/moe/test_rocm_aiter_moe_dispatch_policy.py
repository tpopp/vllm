# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AITER MoE sorting dispatch policy env var and parameter plumbing."""

import pytest
import torch

from vllm._aiter_ops import _rocm_aiter_fused_moe_fake
from vllm.envs import environment_variables


def _fused_moe_cpu_inputs():
    hidden_states = torch.randn(8, 32, dtype=torch.float32)
    w1 = torch.randn(4, 64, 32)
    w2 = torch.randn(4, 32, 32)
    topk_weight = torch.randn(8, 2, dtype=torch.float32)
    topk_ids = torch.randint(0, 4, (8, 2), dtype=torch.int32)
    return hidden_states, w1, w2, topk_weight, topk_ids


def test_dispatch_policy_env_var_default(monkeypatch: pytest.MonkeyPatch):
    getter = environment_variables["VLLM_ROCM_AITER_MOE_DISPATCH_POLICY"]
    monkeypatch.delenv("VLLM_ROCM_AITER_MOE_DISPATCH_POLICY", raising=False)
    assert getter() == 0


def test_dispatch_policy_env_var_override(monkeypatch: pytest.MonkeyPatch):
    getter = environment_variables["VLLM_ROCM_AITER_MOE_DISPATCH_POLICY"]
    monkeypatch.setenv("VLLM_ROCM_AITER_MOE_DISPATCH_POLICY", "2")
    assert getter() == 2


def test_fused_moe_fake_accepts_dispatch_policy():
    hidden, w1, w2, topk_w, topk_i = _fused_moe_cpu_inputs()

    out_default = _rocm_aiter_fused_moe_fake(
        hidden, w1, w2, topk_w, topk_i, moe_sorting_dispatch_policy=0
    )
    assert out_default.shape == hidden.shape
    assert out_default.dtype == hidden.dtype

    out_policy2 = _rocm_aiter_fused_moe_fake(
        hidden, w1, w2, topk_w, topk_i, moe_sorting_dispatch_policy=2
    )
    assert out_policy2.shape == hidden.shape
    assert out_policy2.dtype == hidden.dtype
