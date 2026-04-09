# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoE copy elimination and fused_topk buffer reuse."""

from unittest.mock import patch

import pytest
import torch

import vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe as rafm
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
)


def _reference_topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference matching vllm_topk_softmax contract: in-place + return."""
    probs = torch.softmax(gating_output.float(), dim=-1)
    vals, indices = torch.topk(probs, k=topk_weights.shape[1], dim=-1)
    if renormalize:
        vals = vals / vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    topk_weights.copy_(vals.to(topk_weights.dtype))
    topk_ids.copy_(indices.to(topk_ids.dtype))
    token_expert_indices.copy_(indices.to(token_expert_indices.dtype))
    return topk_weights, topk_ids


def test_fused_topk_router_buffer_reuse():
    M, topk, extra = 8, 2, 2
    num_experts = 16
    hidden = torch.randn(M, 32)
    gating = torch.randn(M, num_experts)

    total_topk_weights = torch.empty(M, topk + extra, dtype=torch.float32)
    total_topk_ids = torch.empty(M, topk + extra, dtype=torch.int32)
    prev_meta = rafm.aiter_topK_meta_data
    try:
        rafm.aiter_topK_meta_data = (total_topk_weights, total_topk_ids)
        with (
            patch.object(
                rocm_aiter_ops,
                "is_fusion_moe_shared_experts_enabled",
                return_value=True,
            ),
            patch.object(rocm_aiter_ops, "is_fused_moe_enabled", return_value=False),
            patch(
                "vllm.model_executor.layers.fused_moe.router.fused_topk_router.vllm_topk_softmax",
                side_effect=_reference_topk_softmax,
            ),
        ):
            ret_w, ret_i, _ = fused_topk(
                hidden, gating, topk, renormalize=True, scoring_func="softmax"
            )

        assert (
            ret_w.untyped_storage().data_ptr()
            == total_topk_weights.untyped_storage().data_ptr()
        )
        assert (
            ret_i.untyped_storage().data_ptr()
            == total_topk_ids.untyped_storage().data_ptr()
        )
        assert ret_w.shape == (M, topk + extra)
        assert ret_i.shape == (M, topk + extra)
    finally:
        rafm.aiter_topK_meta_data = prev_meta


def test_fused_topk_router_no_reuse_when_disabled():
    M, topk = 8, 2
    num_experts = 16
    hidden = torch.randn(M, 32)
    gating = torch.randn(M, num_experts)

    prev_meta = rafm.aiter_topK_meta_data
    try:
        rafm.aiter_topK_meta_data = None
        with (
            patch.object(
                rocm_aiter_ops,
                "is_fusion_moe_shared_experts_enabled",
                return_value=False,
            ),
            patch.object(rocm_aiter_ops, "is_fused_moe_enabled", return_value=False),
            patch(
                "vllm.model_executor.layers.fused_moe.router.fused_topk_router.vllm_topk_softmax",
                side_effect=_reference_topk_softmax,
            ),
        ):
            tw1, ti1, _ = fused_topk(
                hidden, gating, topk, renormalize=True, scoring_func="softmax"
            )
            tw2, ti2, _ = fused_topk(
                hidden, gating, topk, renormalize=True, scoring_func="softmax"
            )

        assert tw1.data_ptr() != tw2.data_ptr()
        assert ti1.data_ptr() != ti2.data_ptr()
    finally:
        rafm.aiter_topK_meta_data = prev_meta


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
