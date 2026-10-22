# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The refusals that need the loaded model (``model_reasons``: the module
variants and every weight's dtype and layout as the kernels read it in place)
and the GDN states vLLM binds (``cache_reasons``). Each lists every reason."""

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.qwen4_exp.amd.specialized_decode.mono import layout as L

FP4 = (torch.uint8, torch.float4_e2m1fn_x2)


def o_proj(layer):
    if layer.layer_type == "linear_attention":
        return layer.linear_attn.out_proj
    return layer.self_attn.o_proj


def mxfp4_shuffled(linear) -> bool | None:
    """An online-MXFP4 linear on AITER's FP4 GEMMs: True with its weight and
    scales shuffled for the ASM FP4 GEMM, False row-major with transposed
    scales for the Triton FP4 GEMM (AITER picks per shape); None otherwise."""
    from vllm.model_executor.kernels.linear.mxfp4.aiter import AiterMxfp4LinearKernel
    from vllm.model_executor.layers.quantization.online.mxfp4 import (
        Mxfp4OnlineLinearMethod,
    )

    qm = linear.quant_method
    if not (
        isinstance(qm, Mxfp4OnlineLinearMethod)
        and isinstance(qm.kernel, AiterMxfp4LinearKernel)
    ):
        return None
    return bool(qm.kernel.use_asm_gemm)


def _fp4_reasons(name: str, linear, rows: int, k: int) -> list[str]:
    from vllm.models.qwen4_exp.amd.specialized_decode.mono.blocks import (
        fp4_scale_shape,
    )

    shuffled = mxfp4_shuffled(linear)
    assert shuffled is not None
    w, ws = linear.weight, linear.weight_scale
    if w.shape != (rows, k // 2) or w.dtype not in FP4:
        return [f"{name} {tuple(w.shape)} {w.dtype}"]
    if (
        ws.shape != fp4_scale_shape(rows, k, shuffled)
        or ws.element_size() != 1
        or not ws.is_contiguous()
    ):
        return [f"{name} scales {tuple(ws.shape)} {ws.dtype}"]
    return []


def gdn_conv_state(attn) -> tuple[torch.Tensor, torch.Tensor]:
    """A GDN layer's (conv state [slots, CONV, entries], recurrent state)."""
    conv_state, rstate = attn.kv_cache
    if not is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    return conv_state, rstate


def _compact(idx: list[int]) -> str:
    """[0, 1, 2, 5] -> '0-2, 5'."""
    runs: list[list[int]] = []
    for i in idx:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def _layer_reasons(layer, lay: L.Layout) -> list[str]:
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.models.qwen4_exp.amd.model import Qwen4ExpSparseMoeBlock
    from vllm.models.qwen4_exp.amd.qsa import Qwen4ExpQSAAttention

    out = []
    bf = torch.bfloat16
    mlp = layer.mlp
    dense: dict[str, torch.Tensor] = {}
    if not isinstance(layer.mlp, Qwen4ExpSparseMoeBlock):
        out.append("a dense MLP")
    elif not mlp.is_fused_shared_expert_enabled:
        out.append("the shared expert not fused")
    else:
        ex = mlp.experts.routed_experts
        backend = getattr(ex.quant_method, "mxfp4_backend", None)
        if backend is not Mxfp4MoeBackend.AITER_MXFP4_MXFP4:
            out.append(f"routed experts on {backend}, not AITER_MXFP4_MXFP4")
        elif not getattr(ex.w13_weight, "is_shuffled", False):
            out.append("routed experts not shuffled")
        elif ex.w13_weight.shape != (L.E + 1, 2 * lay.RIP, L.HIDDEN // 2):
            out.append(f"w13 {tuple(ex.w13_weight.shape)}")
        elif ex.w2_weight.shape != (L.E + 1, L.HIDDEN, lay.RIP // 2):
            out.append(f"w2 {tuple(ex.w2_weight.shape)}")
        dense |= {
            "gate": mlp.gate.weight,
            "shared_expert_gate": mlp.shared_expert_gate.weight,
        }
    for name in ("attn_hyper_connection", "mlp_hyper_connection"):
        hc = getattr(layer, name)
        if not hc.use_combine or hc.pad_size != L.DOWN_ROWS - L.LORA - L.HC:
            out.append(f"{name} variant")
            continue
        dense[f"{name}.hc_norm"] = hc.hc_norm.weight
        dense[f"{name}.down"] = hc.input_mix_weight_down_block_inject.weight
        dense[f"{name}.up"] = hc.input_mix_weight_up.weight
    if layer.layer_type == "linear_attention":
        a = layer.linear_attn
        if not (
            hasattr(a, "in_proj_qkvz")
            and not a.disable_tp_for_ba_proj
            and a.qkvz_layout == "flat"
            and a.activation == "silu"
            and a.conv1d.bias is None
            and a.norm.activation == "sigmoid"
            and a.norm.group_size is None
            and a.norm.norm_before_gate
        ):
            out.append("GDN layer variant")
        elif mxfp4_shuffled(a.in_proj_qkvz) is None:
            out.append("in_proj_qkvz not online MXFP4 on AITER's FP4 GEMMs")
        else:
            out += _fp4_reasons("in_proj_qkvz", a.in_proj_qkvz, lay.QKVZ, L.HIDDEN)
            dense |= {
                "in_proj_ba": a.in_proj_ba.weight,
                "conv1d": a.conv1d.weight,
                "dt_bias": a.dt_bias,
                "norm": a.norm.weight,
            }
            if a.A_log.dtype != torch.float32:
                out.append("A_log not fp32")
    else:
        a = layer.self_attn
        if not isinstance(a, Qwen4ExpQSAAttention):
            out.append("attention without the QSA indexer")
        elif a.num_heads * a.head_dim != lay.CORE:
            out.append(f"attention width {a.num_heads * a.head_dim}")
        elif not a.attn_output_gate:
            out.append("attention without the output gate")
    o = o_proj(layer)
    if mxfp4_shuffled(o) is not None:
        out += _fp4_reasons("o_proj", o, L.HIDDEN, lay.CORE)
    else:
        dense["o_proj"] = o.weight
    out += [
        f"{name} {w.dtype}, neither bf16 nor a quantized layout the kernels read"
        for name, w in dense.items()
        if w.dtype != bf
    ]
    return out


def model_reasons(model, tp: int) -> list[str]:
    """Every reason the kernels cannot serve the loaded ``Qwen4ExpModel``."""
    reasons = []
    if model.start_layer != 0 or model.end_layer != len(model.layers):
        reasons.append("partial layers")
    mixer = model.hyper_connection_mixer
    if mixer is None or mixer.use_combine:
        reasons.append("no final HC mixer")
    if not any(ly.layer_type == "linear_attention" for ly in model.layers):
        reasons.append("no GDN layer")
    lay = L.Layout(tp)
    by_reason: dict[str, list[int]] = {}
    for layer in model.layers:
        for why in _layer_reasons(layer, lay):
            by_reason.setdefault(why, []).append(layer.layer_idx)
    reasons += [f"{why} (layers {_compact(idx)})" for why, idx in by_reason.items()]
    return reasons


def cache_reasons(model, tp: int, q: int = 1) -> list[str]:
    """Every reason the kernels cannot update the GDN states vLLM bound (with
    MTP, ``q`` > 1, the conv state also keeps the drafts' entries)."""
    lay = L.Layout(tp)
    conv = (lay.CONV, L.CONV_W - 1 + q - 1)
    by_reason: dict[str, list[int]] = {}
    for layer in model.layers:
        if layer.layer_type != "linear_attention":
            continue
        conv_state, rstate = gdn_conv_state(layer.linear_attn)
        why = []
        if conv_state.dtype != L.CONV_DTYPE or conv_state.shape[1:] != conv:
            why.append(
                f"conv state {conv_state.dtype} {tuple(conv_state.shape[1:])}, "
                f"needs bf16 {conv}"
            )
        if rstate.dtype != L.STATE_DTYPE or rstate.shape[1:] != (lay.NV, L.HD, L.HD):
            why.append(
                f"recurrent state {rstate.dtype} {tuple(rstate.shape[1:])}, needs "
                f"fp32 ({lay.NV}, {L.HD}, {L.HD})"
            )
        elif not rstate[0].is_contiguous():
            why.append("a recurrent state slot not dense")
        for w in why:
            by_reason.setdefault(w, []).append(layer.layer_idx)
    return [f"{why} (layers {_compact(idx)})" for why, idx in by_reason.items()]
