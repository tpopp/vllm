# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 fused stages: what a provider (mono kernel) may replace.

This file is the whole vLLM-side contract for DSv4.1 providers. It is owned
by the model maintainers and updated in the same PR as any change to the
decoder layer's forward, its module structure or the attention metadata the
views read. Kernel eligibility (CDNA version, TP size, expert count, rows per
step...) is the provider's business, not this file's.

Stages:

* ``deepseek_v41.decoder_layer``: the whole backbone layer (attention seam,
  attention, TP all-reduce, FFN seam, MoE, TP all-reduce), from the previous
  layer's FFN output to this layer's.
* ``deepseek_v41.ffn_after_attention``: everything after attention, from
  ``wo_b``'s un-reduced TP partial sum (seam option ``tp_partial_sum``).
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.fused_stages.api import LayerBinding, Seam, StageSpec, TensorSpec, WeightRef
from vllm.fused_stages.layouts import BF16_ROWMAJOR, F32_ROWMAJOR, UNTAGGED, layout_of
from vllm.fused_stages.views import register_view_builder
from vllm.fused_stages.views.sparse_swa_mla import (
    SparseSWAMLACacheViewV1,
    SparseSWAMLAStepViewV1,
)

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.amd.model import DeepseekV4DecoderLayer

FAMILY = "deepseek_v41"
CACHE_VIEW = SparseSWAMLACacheViewV1.TYPE_ID
STEP_VIEW = SparseSWAMLAStepViewV1.TYPE_ID

ATTENTION_ROLES = (
    "attn.wqkv", "attn.wqkv_scale", "attn.q_norm", "attn.kv_norm",
    "attn.wq_b", "attn.wq_b_scale", "attn.wo_a", "attn.wo_a_scale",
    "attn.wo_b", "attn.wo_b_scale", "attn.sink", "attn.cos_sin",
)  # fmt: skip
FFN_ROLES = (
    "attn_norm", "ffn_norm",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
    "router.weight", "router.bias", "router.bias_vl",
    "experts.w13", "experts.w13_scale", "experts.w2", "experts.w2_scale",
    "shared.gate_up", "shared.gate_up_scale", "shared.down", "shared.down_scale",
)  # fmt: skip


@cache
def stage_specs(hidden: int, hc: int) -> tuple[StageSpec, StageSpec]:
    """The two DSv4.1 stages for a model with ``hidden`` size and ``hc``
    hyper-connection streams."""
    bf16, f32 = torch.bfloat16, torch.float32
    state = (
        ("residual", TensorSpec(("M", hc, hidden), bf16)),
        ("post_mix", TensorSpec(("M", hc, 1), f32)),
        ("res_mix", TensorSpec(("M", hc, hc), f32)),
        ("pre_mix", TensorSpec(("M", hc), f32)),
    )
    x = ("x", TensorSpec(("M", hidden), bf16))
    # Token ids select the MoE routing bias (image sentinels use bias_vl on
    # VL checkpoints) and hash routing; optional because text-only layers
    # without hash routing do not need them.
    ids = ("input_ids", TensorSpec(("M",), torch.int64, optional=True))
    layer_in = Seam(
        "layer_in", (x, *state, ("positions", TensorSpec(("M",), torch.int64)), ids)
    )
    attn_out = Seam("attn_out", (x, *state, ids), options=frozenset({"tp_partial_sum"}))
    layer_out = Seam("layer_out", (x, *state))
    decoder_layer = StageSpec(
        family=FAMILY,
        name="decoder_layer",
        version=1,
        entry=layer_in,
        exit=layer_out,
        scope="layer",
        subsumes_attention=True,
        collectives=("tp_all_reduce", "tp_all_reduce"),
        views=(CACHE_VIEW, STEP_VIEW),
        weight_roles=ATTENTION_ROLES + FFN_ROLES,
    )
    ffn_after_attention = StageSpec(
        family=FAMILY,
        name="ffn_after_attention",
        version=1,
        entry=attn_out,
        exit=layer_out,
        scope="sublayer",
        subsumes_attention=False,
        collectives=("tp_all_reduce", "tp_all_reduce"),
        weight_roles=FFN_ROLES,
    )
    return decoder_layer, ffn_after_attention


def _w(param: torch.Tensor, default: str = UNTAGGED) -> WeightRef:
    """Quantized params carry the tag their quant method attached and stay
    UNTAGGED otherwise (providers must refuse an unknown layout). Plain params
    (norms, hyper-connection mixes, router, biases) pass their known
    row-major default."""
    tag = layout_of(param)
    return WeightRef(param, default if tag == UNTAGGED else tag)


def binding(layer: DeepseekV4DecoderLayer) -> LayerBinding:
    """Role -> processed tensor for one layer. A module refactor (e.g. the
    MoE runner moving ``w13_weight``) only edits this function."""
    a, f = layer.attn, layer.ffn
    e, sh = f.experts.routed_experts, f.shared_experts
    w: dict[str, WeightRef] = {
        "attn_norm": _w(layer.attn_norm.weight, BF16_ROWMAJOR),
        "ffn_norm": _w(layer.ffn_norm.weight, BF16_ROWMAJOR),
        "hc_attn_fn": _w(layer.hc_attn_fn, F32_ROWMAJOR),
        "hc_attn_scale": _w(layer.hc_attn_scale, F32_ROWMAJOR),
        "hc_attn_base": _w(layer.hc_attn_base, F32_ROWMAJOR),
        "hc_ffn_fn": _w(layer.hc_ffn_fn, F32_ROWMAJOR),
        "hc_ffn_scale": _w(layer.hc_ffn_scale, F32_ROWMAJOR),
        "hc_ffn_base": _w(layer.hc_ffn_base, F32_ROWMAJOR),
        "router.weight": _w(f.gate.weight, BF16_ROWMAJOR),
        "router.bias": _w(f.gate.e_score_correction_bias, F32_ROWMAJOR),
        **(
            {"router.bias_vl": _w(f.gate.bias_vl, F32_ROWMAJOR)}
            if getattr(f.gate, "bias_vl", None) is not None
            else {}
        ),
        "experts.w13": _w(e.w13_weight),
        "experts.w13_scale": _w(e.w13_weight_scale),
        "experts.w2": _w(e.w2_weight),
        "experts.w2_scale": _w(e.w2_weight_scale),
    }
    if sh is not None:
        w |= {
            "shared.gate_up": _w(sh.gate_up_proj.weight),
            "shared.gate_up_scale": _w(sh.gate_up_proj.weight_scale),
            "shared.down": _w(sh.down_proj.weight),
            "shared.down_scale": _w(sh.down_proj.weight_scale),
        }
    for role, param in {
        "attn.wqkv": a.fused_wqa_wkv.weight,
        "attn.wqkv_scale": getattr(a.fused_wqa_wkv, "weight_scale", None),
        "attn.wq_b": a.wq_b.weight,
        "attn.wq_b_scale": getattr(a.wq_b, "weight_scale", None),
        "attn.wo_a": a.wo_a.weight,
        "attn.wo_a_scale": getattr(a.wo_a, "weight_scale", None),
        "attn.wo_b": a.wo_b.weight,
        "attn.wo_b_scale": getattr(a.wo_b, "weight_scale", None),
    }.items():
        if param is not None:
            w[role] = _w(param)
    w |= {
        "attn.q_norm": _w(a.q_norm.weight, BF16_ROWMAJOR),
        "attn.kv_norm": _w(a.kv_norm.weight, BF16_ROWMAJOR),
        "attn.sink": _w(a.attn_sink, F32_ROWMAJOR),
        "attn.cos_sin": _w(a.rotary_emb.cos_sin_cache, F32_ROWMAJOR),
    }

    num_hidden_layers, _ = layer.fused_stage_config
    consts = {
        "hidden_size": layer.hidden_size,
        "hc_mult": layer.hc_mult,
        "hc_sinkhorn_iters": layer.hc_sinkhorn_iters,
        "hc_eps": layer.hc_eps,
        "rms_norm_eps": layer.rms_norm_eps,
        "n_routed_experts": f.n_routed_experts,
        "top_k": f.n_activated_experts,
        "scoring_func": f.scoring_func,
        "routed_scaling_factor": f.routed_scaling_factor,
        "renormalize": f.renormalize,
        "swiglu_limit": f.swiglu_limit,
        "has_shared_expert": sh is not None,
        "has_hash_routing": f.gate.tid2eid is not None,
        "has_vision_routing_bias": getattr(f.gate, "bias_vl", None) is not None,
        "image_sentinel_lo": getattr(f, "image_sentinel_lo", 0),
        "compress_ratio": a.compress_ratio,
        "has_compressor": a.compressor is not None,
        "has_indexer": a.indexer is not None,
        "has_engram": layer.engram is not None,
        "kv_cache_dtype": a.kv_cache_dtype,
        "kv_mxfp8": a.kv_mxfp8,
        "uses_sequence_parallel": layer.use_sequence_parallel,
        "fused_seam_norm": layer.fuse_seam_norm,
        "is_first_layer": a.layer_id == 0,
        "is_draft_layer": a.layer_id >= num_hidden_layers,
    }
    return LayerBinding(a.layer_id, w, consts, draft=a.layer_id >= num_hidden_layers)


# --------------------------------------------------------------------- views

RECORD_BYTES = 584  # fp8_ds_mla: 576 data bytes + 8 scale bytes per token


def _records(kv: torch.Tensor, block: int) -> torch.Tensor:
    kv = kv if kv.dtype == torch.uint8 else kv.view(torch.uint8)
    return torch.as_strided(
        kv, (kv.shape[0], block, RECORD_BYTES), (kv.stride(0), RECORD_BYTES, 1)
    )


@register_view_builder(CACHE_VIEW, FAMILY)
def cache_view(layer: DeepseekV4DecoderLayer, _md: None):
    a = layer.attn
    if a.kv_cache_dtype != "fp8_ds_mla" or a.kv_mxfp8:
        return None
    swa = a.swa_cache_layer
    if swa.kv_cache is None or swa.kv_cache.numel() == 0:
        return None
    comp, comp_block = None, 0
    if a.compressed_cache_prefix is not None and a.compress_ratio > 0:
        _, cache_block_size = layer.fused_stage_config
        comp_block = cache_block_size // a.compress_ratio
        comp = _records(a._compressed_kv_cache(), comp_block)
    return SparseSWAMLACacheViewV1(
        swa_records=_records(swa.kv_cache, swa.block_size),
        swa_block_size=swa.block_size,
        comp_records=comp,
        comp_block_size=comp_block,
        record_bytes=RECORD_BYTES,
        kv_cache_dtype=a.kv_cache_dtype,
        compress_ratio=a.compress_ratio,
    )


@register_view_builder(STEP_VIEW, FAMILY)
def step_view(layer: DeepseekV4DecoderLayer, md):
    """Decode metadata of one layer for this step; None if not a decode step
    the view can describe (the framework then runs the reference)."""
    if not isinstance(md, dict):  # profile/dummy runs, DBO microbatch lists
        return None
    a = layer.attn
    swa = md.get(a.swa_cache_layer.prefix)
    if swa is None or swa.decode_swa_indices is None or swa.decode_swa_lens is None:
        return None
    if swa.num_prefills != 0 or swa.token_to_req_indices is None:
        return None
    comp = (
        md.get(a.compressed_cache_prefix)
        if a.compressed_cache_prefix is not None
        else None
    )
    return SparseSWAMLAStepViewV1(
        slot_mapping=swa.slot_mapping,
        swa_indices=swa.decode_swa_indices,
        swa_lens=swa.decode_swa_lens,
        swa_width=swa.decode_swa_width,
        token_to_req=swa.token_to_req_indices,
        topk_indices=a.topk_indices_buffer,
        comp_block_table=None if comp is None else comp.block_table,
        comp_block_size=0 if comp is None else comp.block_size // a.compress_ratio,
    )
