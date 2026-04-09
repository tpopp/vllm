# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for inject_shared_expert_weights used in AITER FSE support."""

import pytest
import torch

import vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe as rafm


@pytest.fixture(autouse=True)
def _reset_topk_meta():
    prev = rafm.aiter_topK_meta_data
    yield
    rafm.aiter_topK_meta_data = prev


def test_noop_when_num_fused_shared_experts_zero():
    topk_w = torch.randn(8, 2, dtype=torch.float32)
    topk_i = torch.randint(0, 16, (8, 2), dtype=torch.int32)
    ret_w, ret_i = rafm.inject_shared_expert_weights(
        topk_w, topk_i, topk=2, num_fused_shared_experts=0
    )
    assert ret_w.data_ptr() == topk_w.data_ptr()
    assert ret_i.data_ptr() == topk_i.data_ptr()


def test_noop_when_meta_data_is_none():
    rafm.aiter_topK_meta_data = None
    topk_w = torch.randn(8, 2, dtype=torch.float32)
    topk_i = torch.randint(0, 16, (8, 2), dtype=torch.int32)
    ret_w, ret_i = rafm.inject_shared_expert_weights(
        topk_w, topk_i, topk=2, num_fused_shared_experts=1
    )
    assert ret_w.data_ptr() == topk_w.data_ptr()
    assert ret_i.data_ptr() == topk_i.data_ptr()


def test_copies_routed_to_combined_buffer():
    """FusedTopKRouter path: when topk_weights.shape[1] == topk, routed
    values are copied into the pre-allocated combined buffer."""
    M, topk, n_shared = 8, 2, 1
    total_w = torch.zeros(M, topk + n_shared, dtype=torch.float32)
    total_i = torch.zeros(M, topk + n_shared, dtype=torch.int32)
    rafm.aiter_topK_meta_data = (total_w, total_i)

    routed_w = torch.randn(M, topk, dtype=torch.float32)
    routed_i = torch.randint(0, 16, (M, topk), dtype=torch.int32)
    shared_w = torch.full((M, n_shared), 0.5, dtype=torch.float32)

    ret_w, ret_i = rafm.inject_shared_expert_weights(
        routed_w,
        routed_i,
        topk=topk,
        num_fused_shared_experts=n_shared,
        shared_expert_weights=shared_w,
    )

    assert ret_w.untyped_storage().data_ptr() == total_w.untyped_storage().data_ptr()
    assert ret_i.untyped_storage().data_ptr() == total_i.untyped_storage().data_ptr()
    torch.testing.assert_close(ret_w[:, :topk], routed_w)
    torch.testing.assert_close(ret_i[:, :topk], routed_i)
    torch.testing.assert_close(ret_w[:, topk : topk + n_shared], shared_w[:M])


def test_no_copy_for_combined_buffer():
    """GroupedTopKRouter path: when topk_weights.shape[1] > topk, the buffer
    is already combined so only shared weights are injected."""
    M, topk, n_shared = 8, 2, 1
    total_w = torch.randn(M, topk + n_shared, dtype=torch.float32)
    total_i = torch.randint(0, 16, (M, topk + n_shared), dtype=torch.int32)
    rafm.aiter_topK_meta_data = (total_w, total_i)

    routed_w_orig = total_w[:, :topk].clone()
    shared_w = torch.full((M, n_shared), 0.7, dtype=torch.float32)

    ret_w, ret_i = rafm.inject_shared_expert_weights(
        total_w,
        total_i,
        topk=topk,
        num_fused_shared_experts=n_shared,
        shared_expert_weights=shared_w,
    )

    assert ret_w.data_ptr() == total_w.data_ptr()
    torch.testing.assert_close(ret_w[:, :topk], routed_w_orig)
    torch.testing.assert_close(ret_w[:, topk : topk + n_shared], shared_w[:M])


def test_shared_weights_none_skips_injection():
    """When shared_expert_weights is None, shared columns are not touched."""
    M, topk, n_shared = 8, 2, 1
    total_w = torch.zeros(M, topk + n_shared, dtype=torch.float32)
    total_i = torch.zeros(M, topk + n_shared, dtype=torch.int32)
    sentinel = -99.0
    total_w[:, topk:].fill_(sentinel)
    rafm.aiter_topK_meta_data = (total_w, total_i)

    routed_w = torch.randn(M, topk, dtype=torch.float32)
    routed_i = torch.randint(0, 16, (M, topk), dtype=torch.int32)

    ret_w, _ = rafm.inject_shared_expert_weights(
        routed_w,
        routed_i,
        topk=topk,
        num_fused_shared_experts=n_shared,
        shared_expert_weights=None,
    )

    assert (ret_w[:, topk:] == sentinel).all()


def test_token_slicing():
    """Only the first `token` rows of the meta buffer are used."""
    max_tokens, topk, n_shared = 32, 2, 1
    total_w = torch.zeros(max_tokens, topk + n_shared, dtype=torch.float32)
    total_i = torch.zeros(max_tokens, topk + n_shared, dtype=torch.int32)
    rafm.aiter_topK_meta_data = (total_w, total_i)

    actual_tokens = 8
    routed_w = torch.randn(actual_tokens, topk, dtype=torch.float32)
    routed_i = torch.randint(0, 16, (actual_tokens, topk), dtype=torch.int32)
    shared_w = torch.full((actual_tokens, n_shared), 0.3, dtype=torch.float32)

    ret_w, ret_i = rafm.inject_shared_expert_weights(
        routed_w,
        routed_i,
        topk=topk,
        num_fused_shared_experts=n_shared,
        shared_expert_weights=shared_w,
    )

    assert ret_w.shape == (actual_tokens, topk + n_shared)
    assert ret_i.shape == (actual_tokens, topk + n_shared)
    assert (total_w[actual_tokens:] == 0).all()
    assert (total_i[actual_tokens:] == 0).all()
