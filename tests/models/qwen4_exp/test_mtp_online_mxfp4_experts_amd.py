# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online MXFP4 for the Qwen4Exp MTP draft routed experts (ROCm, opt-in).

With ``speculative_config.draft_moe_online_quant="mxfp4"`` the draft's own
Quark config gets an online ``targets`` config naming only the draft expert
layers. Those resolve to the online MXFP4 MoE method with the target experts'
weight and activation keys; every other draft layer and every target layer
keeps its checkpoint method. Runs on CPU.
"""

import pytest

import vllm.models.qwen4_exp.amd.model  # noqa: F401  (import order: qsa cycle)
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.utils.config_utils import (
    is_shared_expert_quant_fse_compatible,
)
from vllm.models.qwen4_exp.amd import mtp

DRAFT_LAYER = "mtp.layers.48"
DRAFT_EXPERTS = f"{DRAFT_LAYER}.mlp.experts"
TARGET_EXPERTS = "model.layers.3.mlp.experts"

_MX = {
    "dtype": "fp4",
    "qscheme": "per_group",
    "group_size": 32,
    "is_dynamic": False,
    "scale_format": "e8m0",
    "ch_axis": -1,
    "round_method": "half_even",
    "symmetric": None,
    "observer_cls": "PerBlockMXObserver",
    "is_scale_quant": False,
    "scale_calculation_mode": "even",
}


def _quark_config() -> QuarkConfig:
    """Qwen3.8 Quark MXFP4 W4A4 config with the draft layer excluded."""
    config = QuarkConfig.from_config(
        {
            "quant_method": "quark",
            "global_quant_config": {
                "weight": _MX,
                "input_tensors": {**_MX, "is_dynamic": True},
                "output_tensors": None,
                "bias": None,
            },
            "layer_quant_config": {},
            "layer_type_quant_config": {},
            "exclude": ["lm_head", r"re:mtp\..*"],
            "export": {"kv_cache_group": [], "pack_method": "reorder"},
        }
    )
    config.packed_modules_mapping = mtp.Qwen4ExpMTP.packed_modules_mapping
    return config


def _enable(config):
    mtp._quantize_draft_experts_online(config, [DRAFT_EXPERTS], "mxfp4")
    return config


def test_draft_experts_only():
    config = _enable(_quark_config())
    online = config.online_quantization_config
    assert online is not None

    # Excluded in the checkpoint, so composition takes the online branch
    # (not the unsupported MoE requantization path).
    method = config.get_quant_method_target(DRAFT_EXPERTS, RoutedExperts)[2]
    assert method is UnquantizedFusedMoEMethod
    resolved = online.resolve_quant_method_cls(RoutedExperts, DRAFT_EXPERTS)
    assert resolved is not None
    _, quant_key, _, spec, method_cls = resolved
    assert quant_key == "mxfp4" and method_cls is Mxfp4OnlineMoEMethod

    # Same OCP MX W4A4 scheme as the target's Quark routed experts.
    target_weight, target_act, _ = config.get_quant_method_target(
        TARGET_EXPERTS, RoutedExperts
    )
    assert spec.weight == target_weight
    assert Mxfp4OnlineMoEMethod.activation_quant_key == target_act

    for layer_cls, prefix in [
        (RoutedExperts, TARGET_EXPERTS),
        (LinearBase, f"{DRAFT_LAYER}.mlp.shared_expert.gate_up_proj"),
        (LinearBase, f"{DRAFT_LAYER}.mlp.shared_expert.down_proj"),
        (LinearBase, f"{DRAFT_LAYER}.mlp.gate"),
        (LinearBase, f"{DRAFT_LAYER}.self_attn.qkv_proj"),
        (LinearBase, "mtp.fc_embedding"),
        (LinearBase, "model.layers.3.mlp.shared_expert.down_proj"),
    ]:
        assert online.resolve_quant_method_cls(layer_cls, prefix) is None, prefix


def test_fused_shared_expert_decision_unchanged():
    composed = _enable(_quark_config())
    args = (DRAFT_EXPERTS, f"{DRAFT_LAYER}.mlp.shared_expert")
    before = is_shared_expert_quant_fse_compatible(_quark_config(), *args)
    after = is_shared_expert_quant_fse_compatible(composed, *args)
    assert before == after
    assert before[0] is False


def test_rejects_unquantized_or_already_online_draft():
    with pytest.raises(ValueError, match="no quantization config"):
        mtp._quantize_draft_experts_online(None, [DRAFT_EXPERTS], "mxfp4")

    config = _enable(_quark_config())
    with pytest.raises(ValueError, match="existing online quantization"):
        _enable(config)
