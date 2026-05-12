# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherRMSNorm, MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_fp32, empty_i64

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Custom op: fused QK-norm + RoPE + KV cache update
# ---------------------------------------------------------------------------


def fused_qk_norm_rope_and_unified_kv_cache_update_impl(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> torch.Tensor:
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        attn_layer.impl.do_qk_norm_rope_kvcache_update(
            attn_layer,
            qkv,
            q_out,
            k_out,
            positions,
            q_weight,
            k_weight,
            rms_norm_eps,
            cos_sin_cache,
            is_neox,
            kv_cache,
            layer_slot_mapping,
        )

    return torch.empty(0, device=qkv.device, dtype=qkv.dtype)


def fused_qk_norm_rope_and_unified_kv_cache_update_fake(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> torch.Tensor:
    return torch.empty(0, device=qkv.device, dtype=qkv.dtype)


direct_register_custom_op(
    op_name="fused_qk_norm_rope_and_unified_kv_cache_update",
    op_func=fused_qk_norm_rope_and_unified_kv_cache_update_impl,
    mutates_args=["q_out", "k_out"],
    fake_impl=fused_qk_norm_rope_and_unified_kv_cache_update_fake,
)


# ---------------------------------------------------------------------------
# Custom op: Triton-based fused QKV split + QK-norm + RoPE + KV cache update
# ---------------------------------------------------------------------------


def fused_triton_qk_norm_rope_kvcache_update_impl(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    attn_output_gate: bool,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    layer_name: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cache import (
        fused_qkv_split_qk_norm_rope_cache,
    )

    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)

    T = qkv.shape[0]
    dummy = torch.empty(0, device=qkv.device, dtype=qkv.dtype)

    if layer_slot_mapping is None:
        q = torch.empty(T, num_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
        k = torch.empty(T, num_kv_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
        v = torch.empty(T, num_kv_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
        gate = torch.empty(T, num_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
        return dummy, q, k, v, gate

    rdh = cos_sin_cache.shape[-1] // 2
    cos_full = cos_sin_cache[:, :rdh]
    sin_full = cos_sin_cache[:, rdh:]

    key_cache, value_cache = kv_cache.unbind(0)

    k_scale_f = getattr(attn_layer, "_k_scale_float", 1.0)
    v_scale_f = getattr(attn_layer, "_v_scale_float", 1.0)
    k_scale_t = (
        None
        if k_scale_f == 1.0
        else torch.tensor(k_scale_f, dtype=torch.float32, device=qkv.device)
    )
    v_scale_t = (
        None
        if v_scale_f == 1.0
        else torch.tensor(v_scale_f, dtype=torch.float32, device=qkv.device)
    )

    kv_layout = "HND" if key_cache.shape[1] == num_kv_heads else "NHD"

    q, gate, k, v = fused_qkv_split_qk_norm_rope_cache(
        qkv=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        cos=cos_full,
        sin=sin_full,
        positions=positions,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=layer_slot_mapping,
        qh=num_heads,
        kvh=num_kv_heads,
        head_dim=head_dim,
        is_neox=is_neox,
        attn_output_gate=attn_output_gate,
        k_scale=k_scale_t,
        v_scale=v_scale_t,
        eps=rms_norm_eps,
        kv_cache_layout=kv_layout,
    )
    return dummy, q, k, v, gate


def fused_triton_qk_norm_rope_kvcache_update_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    attn_output_gate: bool,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    layer_name: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T = qkv.shape[0]
    dummy = torch.empty(0, device=qkv.device, dtype=qkv.dtype)
    q = torch.empty(T, num_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
    k = torch.empty(T, num_kv_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
    v = torch.empty(T, num_kv_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
    gate = torch.empty(T, num_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
    return dummy, q, k, v, gate


direct_register_custom_op(
    op_name="fused_triton_qk_norm_rope_kvcache_update",
    op_func=fused_triton_qk_norm_rope_kvcache_update_impl,
    mutates_args=[],
    fake_impl=fused_triton_qk_norm_rope_kvcache_update_fake,
)


# ---------------------------------------------------------------------------
# Pattern: QK-norm + RoPE + unified_kv_cache_update
# ---------------------------------------------------------------------------


class QkNormRopeKvCachePattern:
    """
    Match the unfused sequence:
      q, k, v = split(qkv, ...)
      q = rms_norm(q.view(heads), q_weight).view(flat)
      k = rms_norm(k.view(heads), k_weight).view(flat)
      q, k = rotary_embedding(positions, q, k, cos_sin_cache, is_neox)
      q = q.view(num_heads, head_dim)
      k = k.view(num_kv_heads, head_dim)
      v = v.view(num_kv_heads, head_dim)
      dummy = unified_kv_cache_update(k, v, layer_name)

    Replace with:
      q_out = empty(...)
      k_out = empty(...)
      dummy = fused_qk_norm_rope_and_unified_kv_cache_update(
          q_out, k_out, qkv, positions, q_weight, k_weight,
          eps, cos_sin_cache, is_neox, layer_name)
      v = split(qkv, ...)[2].view(num_kv_heads, head_dim)
    """

    FUSED_OP = torch.ops.vllm.fused_qk_norm_rope_and_unified_kv_cache_update.default

    def __init__(
        self,
        layer: Attention,
        eps: float,
        is_neox: bool,
        rope_flashinfer: bool = False,
        match_rocm_aiter_rope: bool = False,
    ) -> None:
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_size = layer.head_size
        self.head_size_v = layer.head_size_v
        self.eps = eps
        self.is_neox = is_neox
        self.rope_flashinfer = rope_flashinfer

        self.q_size = self.num_heads * self.head_size
        self.k_size = self.num_kv_heads * self.head_size
        self.v_size = self.num_kv_heads * self.head_size_v

        self.rmsnorm_matcher = MatcherRMSNorm(eps)
        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_flashinfer=rope_flashinfer,
            match_rocm_aiter=match_rocm_aiter_rope if match_rocm_aiter_rope else None,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        L = 4096
        qkv = empty_bf16(T, self.q_size + self.k_size + self.v_size)
        positions = empty_i64(T)
        q_weight = empty_bf16(1, self.head_size)
        k_weight = empty_bf16(1, self.head_size)
        if self.rope_flashinfer:
            cos_sin_cache = empty_fp32(L, self.head_size)
        else:
            cos_sin_cache = empty_bf16(L, self.head_size)
        return [qkv, positions, q_weight, k_weight, cos_sin_cache]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

            q_by_head = q.view(-1, self.q_size // self.head_size, self.head_size)
            q_normed = self.rmsnorm_matcher(q_by_head, q_weight)
            q_flat = q_normed.view(-1, self.q_size)

            k_by_head = k.view(-1, self.k_size // self.head_size, self.head_size)
            k_normed = self.rmsnorm_matcher(k_by_head, k_weight)
            k_flat = k_normed.view(-1, self.k_size)

            q_rope, k_rope = self.rope_matcher(positions, q_flat, k_flat, cos_sin_cache)

            q_rope = q_rope.view(-1, self.num_heads, self.head_size)
            k_rope = k_rope.view(-1, self.num_kv_heads, self.head_size)
            v = v.view(-1, self.num_kv_heads, self.head_size_v)
            dummy = torch.ops.vllm.unified_kv_cache_update(k_rope, v, self.layer_name)
            return dummy, q_rope, k_rope, v

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q_out = torch.empty(
                qkv.shape[0],
                self.num_heads,
                self.head_size,
                device=qkv.device,
                dtype=qkv.dtype,
            )
            k_out = torch.empty(
                qkv.shape[0],
                self.num_kv_heads,
                self.head_size,
                device=qkv.device,
                dtype=qkv.dtype,
            )
            _, _, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
            v = v.view(qkv.shape[0], self.num_kv_heads, self.head_size_v)

            results = auto_functionalized(
                self.FUSED_OP,
                q_out=q_out,
                k_out=k_out,
                qkv=qkv,
                positions=positions,
                q_weight=q_weight,
                k_weight=k_weight,
                rms_norm_eps=self.eps,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
                layer_name=self.layer_name,
            )

            # results[0] = dummy, results[1] = q_out, results[2] = k_out
            return results[0], results[1], results[2], v

        def fwd_and_view_to_reshape(*args, **kwargs) -> fx.GraphModule:
            gm = pm.fwd_only(*args, **kwargs)
            view_to_reshape(gm)
            return gm

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            fwd_and_view_to_reshape,
            pm_pass,
        )


# ---------------------------------------------------------------------------
# Pattern: Qwen3Next QK-norm + RoPE + unified_kv_cache_update (with gate)
# ---------------------------------------------------------------------------


class Qwen3NextQkNormRopeKvCachePattern:
    """Extend ``QkNormRopeKvCachePattern`` for the Qwen3Next attention layout.

    When *attn_output_gate* is True the QKV projection emits
    ``[q||gate, k, v]`` where the q portion is doubled.  The pattern
    matches the gate extraction (view -> chunk -> clone) before the
    standard QK-norm + RoPE + KV-cache sequence.  The replacement
    delegates to the Triton-based ``fused_qkv_split_qk_norm_rope_cache``
    kernel which handles the split, norm, RoPE, gate extraction, and KV
    cache update in a single launch.

    Qwen3Next uses GemmaRMSNorm which applies ``(1 + weight)`` before the
    norm.  The pattern includes the ``.float() + 1.0`` nodes so they are
    consumed; the fused op receives the raw weight and applies the
    adjustment internally.
    """

    FUSED_OP = torch.ops.vllm.fused_triton_qk_norm_rope_kvcache_update.default

    def __init__(
        self,
        layer: Attention,
        eps: float,
        is_neox: bool,
        attn_output_gate: bool,
        rope_flashinfer: bool = False,
        match_rocm_aiter_rope: bool = False,
    ) -> None:
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_size = layer.head_size
        self.head_size_v = layer.head_size_v
        self.eps = eps
        self.is_neox = is_neox
        self.rope_flashinfer = rope_flashinfer
        self.attn_output_gate = attn_output_gate

        self.q_size = self.num_heads * self.head_size
        self.k_size = self.num_kv_heads * self.head_size
        self.v_size = self.num_kv_heads * self.head_size_v

        self.rmsnorm_matcher = MatcherRMSNorm(eps)
        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_flashinfer=rope_flashinfer,
            match_rocm_aiter=(match_rocm_aiter_rope if match_rocm_aiter_rope else None),
        )

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        L = 4096
        q_portion = self.q_size * 2 if self.attn_output_gate else self.q_size
        qkv = empty_bf16(T, q_portion + self.k_size + self.v_size)
        positions = empty_i64(T)
        q_weight = empty_bf16(self.head_size)
        k_weight = empty_bf16(self.head_size)
        if self.rope_flashinfer:
            cos_sin_cache = empty_fp32(L, self.head_size)
        else:
            cos_sin_cache = empty_bf16(L, self.head_size)
        return [qkv, positions, q_weight, k_weight, cos_sin_cache]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_size
        head_dim_v = self.head_size_v
        q_size = self.q_size
        k_size = self.k_size
        v_size = self.v_size
        layer_name = self.layer_name
        eps = self.eps
        is_neox = self.is_neox
        attn_output_gate = self.attn_output_gate

        rmsnorm_matcher = self.rmsnorm_matcher
        rope_matcher = self.rope_matcher

        if attn_output_gate:

            def pattern(
                qkv: torch.Tensor,
                positions: torch.Tensor,
                q_weight: torch.Tensor,
                k_weight: torch.Tensor,
                cos_sin_cache: torch.Tensor,
            ) -> tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]:
                q_gate, k, v = qkv.split([q_size * 2, k_size, v_size], dim=-1)

                q_gate_3d = q_gate.view(-1, num_heads, 2 * head_dim)
                q_3d, gate_3d = q_gate_3d.chunk(2, dim=-1)

                q_3d = q_3d.contiguous()
                q_w = q_weight.float() + 1.0
                q_normed = rmsnorm_matcher(q_3d, q_w)
                q_normed_flat = q_normed.view(-1, q_size)

                k_3d = k.view(-1, num_kv_heads, head_dim)
                k_w = k_weight.float() + 1.0
                k_normed = rmsnorm_matcher(k_3d, k_w)
                k_flat = k_normed.view(-1, k_size)

                q_rope, k_rope = rope_matcher(
                    positions, q_normed_flat, k_flat, cos_sin_cache
                )

                q_rope = q_rope.view(-1, num_heads, head_dim)
                k_rope = k_rope.view(-1, num_kv_heads, head_dim)
                v = v.view(-1, num_kv_heads, head_dim_v)
                dummy = torch.ops.vllm.unified_kv_cache_update(k_rope, v, layer_name)
                return dummy, q_rope, k_rope, v, gate_3d

            def replacement(
                qkv: torch.Tensor,
                positions: torch.Tensor,
                q_weight: torch.Tensor,
                k_weight: torch.Tensor,
                cos_sin_cache: torch.Tensor,
            ) -> tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]:
                results = self.FUSED_OP(
                    qkv=qkv,
                    positions=positions,
                    q_weight=q_weight,
                    k_weight=k_weight,
                    rms_norm_eps=eps,
                    cos_sin_cache=cos_sin_cache,
                    is_neox=is_neox,
                    attn_output_gate=True,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    layer_name=layer_name,
                )

                _, _, v = qkv.split([q_size * 2, k_size, v_size], dim=-1)
                v = v.view(qkv.shape[0], num_kv_heads, head_dim_v)

                return results[0], results[1], results[2], v, results[4]
        else:

            def pattern(  # type: ignore[misc]
                qkv: torch.Tensor,
                positions: torch.Tensor,
                q_weight: torch.Tensor,
                k_weight: torch.Tensor,
                cos_sin_cache: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)

                q_by_head = q.view(-1, q_size // head_dim, head_dim)
                q_w = q_weight.float() + 1.0
                q_normed = rmsnorm_matcher(q_by_head, q_w)
                q_flat = q_normed.view(-1, q_size)

                k_by_head = k.view(-1, k_size // head_dim, head_dim)
                k_w = k_weight.float() + 1.0
                k_normed = rmsnorm_matcher(k_by_head, k_w)
                k_flat = k_normed.view(-1, k_size)

                q_rope, k_rope = rope_matcher(positions, q_flat, k_flat, cos_sin_cache)

                q_rope = q_rope.view(-1, num_heads, head_dim)
                k_rope = k_rope.view(-1, num_kv_heads, head_dim)
                v = v.view(-1, num_kv_heads, head_dim_v)
                dummy = torch.ops.vllm.unified_kv_cache_update(k_rope, v, layer_name)
                return dummy, q_rope, k_rope, v

            def replacement(  # type: ignore[misc]
                qkv: torch.Tensor,
                positions: torch.Tensor,
                q_weight: torch.Tensor,
                k_weight: torch.Tensor,
                cos_sin_cache: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                results = self.FUSED_OP(
                    qkv=qkv,
                    positions=positions,
                    q_weight=q_weight,
                    k_weight=k_weight,
                    rms_norm_eps=eps,
                    cos_sin_cache=cos_sin_cache,
                    is_neox=is_neox,
                    attn_output_gate=False,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    layer_name=layer_name,
                )

                _, _, v = qkv.split([q_size, k_size, v_size], dim=-1)
                v = v.view(qkv.shape[0], num_kv_heads, head_dim_v)

                return results[0], results[1], results[2], v

        def fwd_and_view_to_reshape(*args, **kwargs) -> fx.GraphModule:
            gm = pm.fwd_only(*args, **kwargs)
            view_to_reshape(gm)
            return gm

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            fwd_and_view_to_reshape,
            pm_pass,
        )


# ---------------------------------------------------------------------------
# Pass class
# ---------------------------------------------------------------------------


class QkNormRopeKvCacheFusionPass(VllmPatternMatcherPass):
    """
    Fuse QK-norm + RoPE + KV cache update into a single AITER HIP kernel.

    Supersedes both QKNormRoPEFusionPass and RopeKVCacheFusionPass for
    attention layers that support the combined operation, eliminating two
    separate kernel launches and the intermediate memory traffic.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="qk_norm_rope_kvcache_fusion_pass"
        )

        cc = config.compilation_config
        self.max_token_num = cc.pass_config.rope_kvcache_fusion_max_token_num

        dtype = config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.warning_once(
                "QK Norm+RoPE+KVCache fusion not enabled: unsupported dtype %s", dtype
            )
            return

        attn_layers = get_layers_from_vllm_config(config, Attention)

        rope_custom_enabled = cc.is_custom_op_enabled("rotary_embedding")
        rms_custom_enabled = cc.is_custom_op_enabled("rms_norm")
        logger.debug(
            "QkNormRopeKvCacheFusionPass init: "
            "RotaryEmbedding.enabled()=%s, rope_custom_enabled=%s, "
            "RMSNorm custom_op_enabled=%s",
            RotaryEmbedding.enabled(),
            rope_custom_enabled,
            rms_custom_enabled,
        )

        # RMS norm variants are no longer iterated: after the vLLM IR
        # migration (#33825), `MatcherRMSNorm` dispatches via
        # `ir.ops.rms_norm`, which resolves to the same backend (native /
        # vllm_c / aiter / oink / ...) that the model's RMSNorm layer
        # picks.  The pattern graph tracks the target graph automatically.
        aiter_rope_variants = [False]
        if rocm_aiter_ops.is_triton_rotary_embed_enabled():
            aiter_rope_variants.append(True)

        for _, layer in attn_layers.items():
            if not layer.impl.fused_qk_norm_rope_kvcache_supported():
                continue
            layer.impl.set_fused_kv_cache_layout()
            for aiter_rope in aiter_rope_variants:
                for epsilon in [1e-5, 1e-6]:
                    for neox in [True, False]:
                        if RotaryEmbedding.enabled():
                            for rope_flashinfer in [False, True]:
                                try:
                                    QkNormRopeKvCachePattern(
                                        layer=layer,
                                        eps=epsilon,
                                        is_neox=neox,
                                        rope_flashinfer=rope_flashinfer,
                                        match_rocm_aiter_rope=aiter_rope,
                                    ).register(self.patterns)
                                except RuntimeError as e:
                                    if "Duplicate pattern" in str(e):
                                        logger.debug(
                                            "Skipping duplicate pattern: "
                                            "aiter_rope=%s eps=%s neox=%s fi=%s",
                                            aiter_rope,
                                            epsilon,
                                            neox,
                                            rope_flashinfer,
                                        )
                                    else:
                                        raise
                        else:
                            try:
                                QkNormRopeKvCachePattern(
                                    layer=layer,
                                    eps=epsilon,
                                    is_neox=neox,
                                    match_rocm_aiter_rope=aiter_rope,
                                ).register(self.patterns)
                            except RuntimeError as e:
                                if "Duplicate pattern" in str(e):
                                    logger.debug(
                                        "Skipping duplicate pattern: "
                                        "aiter_rope=%s eps=%s neox=%s fi=N/A",
                                        aiter_rope,
                                        epsilon,
                                        neox,
                                    )
                                else:
                                    raise

        # Qwen3Next-specific patterns with attn_output_gate handling.
        hf_config = config.model_config.hf_text_config
        attn_output_gate = getattr(hf_config, "attn_output_gate", None)
        if attn_output_gate is None and type(hf_config).__name__ in (
            "Qwen3NextConfig",
        ):
            attn_output_gate = True
        if attn_output_gate is not None:
            for gate_val in (
                [attn_output_gate]
                if isinstance(attn_output_gate, bool)
                else [True, False]
            ):
                for _, layer in attn_layers.items():
                    if not layer.impl.fused_qk_norm_rope_kvcache_supported():
                        continue
                    for aiter_rope in aiter_rope_variants:
                        for epsilon in [1e-5, 1e-6]:
                            for neox in [True, False]:
                                fi_variants = (
                                    [False, True]
                                    if RotaryEmbedding.enabled()
                                    else [False]
                                )
                                for rope_fi in fi_variants:
                                    try:
                                        Qwen3NextQkNormRopeKvCachePattern(
                                            layer=layer,
                                            eps=epsilon,
                                            is_neox=neox,
                                            attn_output_gate=gate_val,
                                            rope_flashinfer=rope_fi,
                                            match_rocm_aiter_rope=aiter_rope,
                                        ).register(self.patterns)
                                    except RuntimeError as e:
                                        if "Duplicate pattern" not in str(e):
                                            raise
                                        logger.debug(
                                            "Skipping duplicate Qwen3Next "
                                            "pattern: gate=%s "
                                            "rope=%s eps=%s neox=%s fi=%s",
                                            gate_val,
                                            aiter_rope,
                                            epsilon,
                                            neox,
                                            rope_fi,
                                        )

        # Backends that set _use_interleaved_v_cache (e.g. ROCM_ATTN)
        # require a consistent V-cache layout across ALL compile ranges.
        # If max_token_num is too small, unfused ranges would write
        # standard-layout V while the attention kernel reads interleaved,
        # corrupting long-sequence generation.  Force fusion to cover all
        # ranges so both write and read paths agree on the layout.
        max_batched = config.scheduler_config.max_num_batched_tokens
        needs_full_coverage = any(
            getattr(layer.impl, "_use_interleaved_v_cache", False)
            for _, layer in attn_layers.items()
            if layer.impl.fused_qk_norm_rope_kvcache_supported()
        )
        if (
            needs_full_coverage
            and max_batched is not None
            and self.max_token_num < max_batched
        ):
            logger.info(
                "Raising rope_kvcache_fusion_max_token_num from %d to %d "
                "to maintain consistent interleaved V-cache layout across "
                "all compile ranges (required by attention backend).",
                self.max_token_num,
                max_batched,
            )
            self.max_token_num = max_batched

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        _orig_fx_to_pat = pm.fx_to_pattern

        def _relaxed_fx_to_pattern(*a, **kw):
            kw["ignore_types"] = (int, torch.SymInt)
            return _orig_fx_to_pat(*a, **kw)

        pm.fx_to_pattern = _relaxed_fx_to_pattern
        try:
            self.matched_count = self.patterns.apply(graph)
        finally:
            pm.fx_to_pattern = _orig_fx_to_pat

        logger.info(
            "QK-Norm+RoPE+KVCache fusion: replaced %s pattern(s) "
            "with AITER fused_qk_norm_rope_cache_pts_quant_shuffle",
            self.matched_count,
        )

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return compile_range.end <= self.max_token_num

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(
            self, QkNormRopeKvCachePattern, Qwen3NextQkNormRopeKvCachePattern
        )
