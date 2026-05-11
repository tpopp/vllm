# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    dispatch_topk_softmax_func,
)


def _inject_shared_expert_weights(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    num_fused_shared_experts: int,
    shared_expert_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge routed topk results with the shared expert buffer and inject
    dynamic per-token shared expert gate values for AITER fusion.

    For routers that already return the combined buffer (e.g. GroupedTopKRouter
    via rocm_aiter_grouped_topk), only the dynamic weight injection is needed.
    For routers that return only routed slots (e.g. FusedTopKRouter), this also
    copies the routed results into the pre-allocated combined buffer.
    """
    if num_fused_shared_experts == 0:
        return topk_weights, topk_ids

    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        aiter_topK_meta_data,
    )

    assert aiter_topK_meta_data is not None, (
        "aiter_topK_meta_data is not initialized but "
        "num_fused_shared_experts > 0. Ensure init_aiter_topK_meta_data "
        "is called before routing."
    )

    total_topk_weights, total_topk_ids = aiter_topK_meta_data
    token = topk_weights.shape[0]

    assert total_topk_weights.shape[0] >= token, (
        f"AITER topK meta data supports {total_topk_weights.shape[0]} "
        f"tokens, but got {token} tokens."
    )

    total_topk_weights_slice = total_topk_weights[:token]
    total_topk_ids_slice = total_topk_ids[:token]

    if topk_weights.shape[1] == topk:
        total_topk_weights_slice[:, :topk] = topk_weights
        total_topk_ids_slice[:, :topk] = topk_ids
        topk_weights = total_topk_weights_slice
        topk_ids = total_topk_ids_slice

    if shared_expert_weights is not None:
        topk_weights[:, topk : topk + num_fused_shared_experts] = shared_expert_weights[
            :token
        ]

    return topk_weights, topk_ids


class AiterSharedRoutedFusedMoERouter(BaseRouter):
    """
    ROCm AITER router for models with fused shared experts (e.g. Qwen3-MoE).

    When the AITER topk_softmax kernel supports sigmoid fusion, the routing
    softmax and shared-expert sigmoid are computed in a single kernel launch.
    Otherwise the shared-expert weights are injected into the pre-allocated
    AITER buffer via a fallback path.

    Only instantiated when rocm_aiter fused-MoE is active and
    num_fused_shared_experts > 0.
    """

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        num_fused_shared_experts: int,
        scoring_func: str = "softmax",
        renormalize: bool = True,
        enable_eplb: bool = False,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        self.num_fused_shared_experts = num_fused_shared_experts

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=None,
            has_e_score_bias=False,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.size(0) == router_logits.size(0), (
            "Number of tokens mismatch"
        )

        from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
            aiter_topK_meta_data,
        )

        M = hidden_states.size(0)
        topk = self.top_k
        num_fse = self.num_fused_shared_experts

        token_expert_indices = torch.empty(
            M, topk, dtype=torch.int32, device=hidden_states.device
        )

        if rocm_aiter_ops.fuse_sigmoid_in_kernel(aiter_topK_meta_data):
            total_topk_weights, total_topk_ids = aiter_topK_meta_data  # type: ignore[misc]
            total_topk_weights_slice = total_topk_weights[:M]
            topk_ids_slice = total_topk_ids[:M, :topk]

            topk_func = dispatch_topk_softmax_func(use_rocm_aiter=True)
            topk_func(
                total_topk_weights_slice,
                topk_ids_slice,
                token_expert_indices,
                router_logits,
                self.renormalize,
                num_fse,
                "sigmoid",
            )
            return total_topk_weights_slice, total_topk_ids[:M]

        routing_logits = router_logits[:, :-num_fse]
        shared_logits = router_logits[:, -num_fse:]

        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(
            M,
            topk,
            dtype=torch.int32 if indices_type is None else indices_type,
            device=hidden_states.device,
        )

        topk_func = dispatch_topk_softmax_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights,
            topk_ids,
            token_expert_indices,
            routing_logits,
            self.renormalize,
        )

        shared_weights = torch.sigmoid(shared_logits)
        topk_weights, topk_ids = _inject_shared_expert_weights(
            topk_weights,
            topk_ids,
            topk=topk,
            num_fused_shared_experts=num_fse,
            shared_expert_weights=shared_weights,
        )

        return topk_weights, topk_ids
