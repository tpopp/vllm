# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp weight-free QSA indexer."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope

from ..common.qsa_cache import (
    QSACompressedKeyCache,
    QSAForwardMetadata,
    QSAKeyStateCache,
    canonical_qsa_rope_positions,
)
from .ops.qsa_prepare import qsa_prepare, supports_fused_pre_indexer

if TYPE_CHECKING:
    from .qsa import Qwen4ExpQSAAttention


def apply_qsa_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply the main attention's exact 1D/MRoPE composition to QSA heads."""
    num_tokens, _, head_dim = tensor.shape
    rotary_dim = rotary_emb.rotary_dim
    cache = rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        shape = tensor.shape
        tensor, _ = triton_mrope(
            tensor.reshape(num_tokens, -1),
            tensor.new_empty((num_tokens, head_dim)),
            cos,
            sin,
            rotary_emb.mrope_section,
            head_dim,
            rotary_dim,
            rotary_emb.mrope_interleaved,
            rotary_emb.is_neox_style,
        )
        return tensor.reshape(shape)

    rotated = rotary_emb.apply_rotary_emb(
        tensor[..., :rotary_dim],
        cos,
        sin,
    )
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


def apply_qsa_rmsnorm(
    norm: GemmaRMSNorm,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Use vLLM's portable RMSNorm implementation on ROCm."""
    return cast(torch.Tensor, norm(tensor))


class QSAIndexer(nn.Module):
    """Replicated Q/K projection plus paged, weight-free QSA selection.

    ``prefix`` must be the checkpoint's indexer prefix, normally
    ``model.layers.N.self_attn.indexer``.  Consequently the trainable names are
    ``index_qk_proj``, ``q_layernorm`` and ``k_layernorm`` under that prefix.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        rotary_emb: nn.Module,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if vllm_config.cache_config is None:
            raise ValueError("QSA requires a paged KV cache")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")

        self.layer_id = int(layer_id)
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.rotary_emb = rotary_emb
        self.use_fused_pre_indexer = supports_fused_pre_indexer(
            rotary_emb,
            self.index_head_dim,
            self.index_kv_heads,
            self.compress_ratio,
        )
        self.prefix = prefix
        # MTP step 0 selects the target-aligned rows; later steps reuse them
        # while continuing to update the QSA side cache.
        self.skip_topk = False

        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )

        cache_config = vllm_config.cache_config
        cache_prefix = f"{prefix}." if prefix else ""
        self.raw_key_cache = QSAKeyStateCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            cache_rope_positions=vllm_config.model_config.uses_mrope,
            prefix=f"{cache_prefix}raw_key_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
            vllm_config=vllm_config,
        )
        self.compressed_key_cache = QSACompressedKeyCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            compress_ratio=self.compress_ratio,
            prefix=f"{cache_prefix}compressed_key_cache",
            cache_config=cache_config,
            vllm_config=vllm_config,
        )

    @property
    def output_width(self) -> int:
        return self.token_topk + self.compress_ratio - 1

    def _split_qk(self, qk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``qk`` [*, (index_n_heads + index_kv_heads) index_head_dim] -> the
        unrotated Q / K operands ``project`` itself also splits out."""
        return qk.split(
            (
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ),
            dim=-1,
        )

    def project(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project replicated Q/K and split, leaving both operands unrotated."""
        qk, _ = self.index_qk_proj(hidden_states)
        return self._split_qk(qk)

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project replicated Q/K, normalize+rotate Q, and preserve raw K."""
        q_raw, token_k = self.project(hidden_states)
        q = q_raw.reshape(-1, self.index_n_heads, self.index_head_dim)
        q = apply_qsa_rmsnorm(
            self.q_layernorm,
            q.reshape(-1, self.index_head_dim),
        ).reshape_as(q)
        q = apply_qsa_rope(self.rotary_emb, positions, q)
        return q, token_k.reshape(-1, 1, self.index_head_dim)

    def normalize_compressed_keys(
        self,
        compressed_keys: torch.Tensor,
        first_rope_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize pooled K and apply the first token's exact group position."""
        keys = compressed_keys.reshape(-1, self.index_head_dim)
        keys = apply_qsa_rmsnorm(self.k_layernorm, keys).reshape(
            -1, 1, self.index_head_dim
        )
        if getattr(self.rotary_emb, "mrope_section", None):
            positions = first_rope_positions.transpose(0, 1)
        else:
            positions = first_rope_positions[:, 0]
        return apply_qsa_rope(self.rotary_emb, positions, keys)

    def _metadata(
        self,
    ) -> tuple[QSAForwardMetadata, QSAForwardMetadata] | None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            return None
        raw = cast(QSAForwardMetadata, metadata[self.raw_key_cache.prefix])
        compressed = cast(
            QSAForwardMetadata, metadata[self.compressed_key_cache.prefix]
        )
        if raw.num_actual_tokens != compressed.num_actual_tokens:
            raise RuntimeError("QSA side-cache metadata token counts disagree")
        if not raw.logical_positions.is_cuda and (
            not torch.equal(raw.logical_positions, compressed.logical_positions)
        ):
            raise RuntimeError("QSA side-cache metadata positions disagree")
        return raw, compressed

    def _update_and_compress(
        self,
        token_k: torch.Tensor,
        positions: torch.Tensor,
        raw_metadata: QSAForwardMetadata,
        compressed_metadata: QSAForwardMetadata,
    ) -> None:
        num_tokens = raw_metadata.num_actual_tokens
        raw_key_cache = self.raw_key_cache.key_cache
        rope_position_cache = self.raw_key_cache.rope_position_cache
        from .ops.qsa import qsa_compress_groups_with_ratio, qsa_store_cache_rows

        if rope_position_cache is None:
            position_rows = raw_metadata.logical_positions.view(-1, 1, 1).expand(
                -1, 1, 3
            )
        else:
            position_rows = canonical_qsa_rope_positions(positions)[:num_tokens].to(
                device=raw_key_cache.device
            )
        pooled, first_positions = qsa_compress_groups_with_ratio(
            token_k[:num_tokens],
            position_rows,
            raw_key_cache,
            raw_metadata.block_table,
            raw_metadata.token_to_req,
            raw_metadata.query_start_loc,
            raw_metadata.logical_positions,
            compressed_metadata.slot_mapping,
            self.compress_ratio,
            rope_position_cache,
        )
        normalized = self.normalize_compressed_keys(pooled, first_positions)
        qsa_store_cache_rows(
            self.compressed_key_cache.kv_cache,
            compressed_metadata.slot_mapping,
            normalized,
        )
        qsa_store_cache_rows(
            raw_key_cache,
            raw_metadata.slot_mapping,
            token_k[:num_tokens],
        )
        if rope_position_cache is not None:
            qsa_store_cache_rows(
                rope_position_cache,
                raw_metadata.slot_mapping,
                position_rows,
            )

    def _fused_prepare(
        self,
        projected_q: torch.Tensor,
        raw_keys: torch.Tensor,
        positions: torch.Tensor,
        raw_metadata: QSAForwardMetadata,
        compressed_metadata: QSAForwardMetadata,
        attn: Qwen4ExpQSAAttention,
        qkv: torch.Tensor,
        slot_mapping: torch.Tensor,
        gate_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Prepare the indexer and ``attn``'s main attention in one launch.

        Normalizes and rotates the index Q, compresses K and writes both side
        caches, which replaces ``project_qk``'s norm/RoPE half and the whole of
        ``_update_and_compress``. The same launch applies ``attn``'s QK-norm and
        RoPE, copies its gate and writes its K/V into ``attn.kv_cache``,
        replacing ``_project_qkv_gate`` and ``do_kv_cache_update``. Both halves
        must stay numerically interchangeable with what they replace.
        """
        num_tokens = raw_metadata.num_actual_tokens
        raw_key_cache = self.raw_key_cache
        q = projected_q.new_empty(num_tokens, self.index_n_heads, self.index_head_dim)
        main_outputs = qsa_prepare(
            projected_q,
            raw_keys,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.q_layernorm.weight,
            self.k_layernorm.weight,
            self.q_layernorm.variance_epsilon,
            q,
            raw_key_cache.kv_cache,
            raw_metadata.slot_mapping,
            raw_metadata.block_table,
            raw_metadata.query_start_loc,
            raw_metadata.logical_positions,
            self.compressed_key_cache.kv_cache,
            compressed_metadata.slot_mapping,
            compressed_metadata.k_work_metadata,
            compress_ratio=self.compress_ratio,
            mrope_section=getattr(self.rotary_emb, "mrope_section", None),
            rope_pos_offset=(
                raw_key_cache.rope_position_offset
                if raw_key_cache.cache_rope_positions
                else None
            ),
            main_qkv=qkv[:num_tokens],
            main_q_norm_weight=attn.q_norm.weight,
            main_k_norm_weight=attn.k_norm.weight,
            main_eps=attn.q_norm.variance_epsilon,
            main_kv_cache=attn.kv_cache.transpose(1, 2),
            main_slot_mapping=slot_mapping[:num_tokens],
            main_k_scale=attn._k_scale_float,
            main_v_scale=attn._v_scale_float,
            main_gate_out=None if gate_out is None else gate_out[:num_tokens],
        )
        return q, main_outputs

    def _select_rows(
        self,
        q: torch.Tensor,
        metadata: QSAForwardMetadata,
        out: torch.Tensor | None,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        """Per-row selection for ``rows`` only; other rows of ``out`` keep
        their contents. Rows are scored independently, so each selected row
        matches a full-batch call bit-exactly. Duplicate rows are allowed."""
        from .ops.qsa import qsa_select_paged_tokens

        if out is None:
            out = torch.empty(
                (q.shape[0], self.output_width), dtype=torch.int32, device=q.device
            )
        selected = qsa_select_paged_tokens(
            q.index_select(0, rows),
            self.compressed_key_cache.kv_cache,
            metadata.block_table,
            metadata.token_to_req.index_select(0, rows),
            metadata.logical_positions.index_select(0, rows),
            metadata.seq_lens,
            self.token_topk,
            self.compress_ratio,
        )
        out.index_copy_(0, rows, selected)
        return out

    def _select(
        self,
        q: torch.Tensor,
        metadata: QSAForwardMetadata,
        out: torch.Tensor | None,
        tail_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select every row, or with ``tail_rows`` only the decode rows
        ``[0, num_decode_tokens)`` plus ``tail_rows`` (rows of ``out``
        outside that set are left as they were)."""
        from .ops.qsa import qsa_select_paged_tokens
        from .ops.qsa_uniform_indexer import (
            MAX_UNIFORM_QUERY_LEN,
            qsa_select_paged_tokens_uniform,
        )

        # require_uniform puts requests of exactly decode_query_len rows (or
        # zero-length padding) first: rows [0, num_decode_tokens) are
        # request-major.
        query_len = metadata.decode_query_len
        num_uniform_rows = metadata.num_decode_tokens
        if not (1 < query_len <= MAX_UNIFORM_QUERY_LEN and num_uniform_rows > 0):
            if tail_rows is not None:
                head_rows = torch.arange(
                    metadata.num_decode_tokens,
                    dtype=tail_rows.dtype,
                    device=tail_rows.device,
                )
                return self._select_rows(
                    q, metadata, out, torch.cat((head_rows, tail_rows))
                )
            return qsa_select_paged_tokens(
                q,
                self.compressed_key_cache.kv_cache,
                metadata.block_table,
                metadata.token_to_req,
                metadata.logical_positions,
                metadata.seq_lens,
                self.token_topk,
                self.compress_ratio,
                out,
            )
        if metadata.num_prefills == 0:
            # Pure verify batches may carry padding rows past num_decode_tokens.
            num_uniform_rows = q.shape[0]
        if out is None:
            out = torch.empty(
                (q.shape[0], self.output_width), dtype=torch.int32, device=q.device
            )
        uniform_rows = slice(0, num_uniform_rows)
        qsa_select_paged_tokens_uniform(
            q[uniform_rows],
            self.compressed_key_cache.kv_cache,
            metadata.block_table,
            metadata.token_to_req[uniform_rows],
            metadata.logical_positions[uniform_rows],
            metadata.seq_lens,
            self.token_topk,
            self.compress_ratio,
            query_len,
            out[uniform_rows],
        )
        if tail_rows is not None:
            self._select_rows(q, metadata, out, tail_rows)
        elif num_uniform_rows < q.shape[0]:
            tail = slice(num_uniform_rows, q.shape[0])
            qsa_select_paged_tokens(
                q[tail],
                self.compressed_key_cache.kv_cache,
                metadata.block_table,
                metadata.token_to_req[tail],
                metadata.logical_positions[tail],
                metadata.seq_lens,
                self.token_topk,
                self.compress_ratio,
                out[tail],
            )
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        attn: Qwen4ExpQSAAttention,
        qkv: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        gate_out: torch.Tensor | None = None,
        kept_tail_rows: torch.Tensor | None = None,
        idx_qk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Return fixed-width request-relative token indices padded with ``-1``.

        With ``kept_tail_rows``, the caches are still written for every row
        but only the decode rows and ``kept_tail_rows`` are selected; the
        other rows of ``out`` are left stale (KV-only MTP draft prefill).

        With ``attn.use_fused_qsa_prepare``, the same launch also writes
        ``attn``'s K/V into ``attn.kv_cache`` at ``slot_mapping`` and prepares
        its Q and gate from ``qkv``, returned as the second element (None
        otherwise). The gate is written into ``gate_out`` when given. ``qkv``,
        ``slot_mapping`` and ``gate_out`` are only read in that mode.

        ``idx_qk`` (P2, ``VLLM_ROCM_MONO_K1_QSA_PROJ``): a caller that already
        ran ``index_qk_proj`` (K1's ``qsa_proj`` build) passes its output
        here, the same width ``self.index_qk_proj`` itself produces; this
        skips that GEMM instead of calling ``self.project``. Only read
        alongside ``attn.use_fused_qsa_prepare``; None (the default) is the
        stock path.
        """
        metadata = self._metadata()
        if metadata is None:
            # Preserve step-0 indices when later MTP steps reuse the buffer.
            if self.skip_topk and out is not None:
                return out, None
            result = torch.full(
                (hidden_states.shape[0], self.output_width),
                -1,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            if out is not None:
                out.copy_(result)
                return out, None
            return result, None
        raw_metadata, compressed_metadata = metadata
        num_tokens = raw_metadata.num_actual_tokens
        hidden_states = hidden_states[:num_tokens]
        positions = positions[..., :num_tokens]
        main_outputs: tuple[torch.Tensor, torch.Tensor] | None = None
        if attn.use_fused_qsa_prepare:
            if qkv is None or slot_mapping is None:
                raise ValueError("fused QSA prepare requires qkv and slot_mapping")
            if idx_qk is None:
                projected_q, raw_keys = self.project(hidden_states)
            else:
                projected_q, raw_keys = self._split_qk(idx_qk[:num_tokens])
            q, main_outputs = self._fused_prepare(
                projected_q,
                raw_keys,
                positions,
                raw_metadata,
                compressed_metadata,
                attn,
                qkv,
                slot_mapping,
                gate_out,
            )
        else:
            q, token_k = self.project_qk(hidden_states, positions)
            self._update_and_compress(
                token_k,
                positions,
                raw_metadata,
                compressed_metadata,
            )
        if self.skip_topk:
            if out is None:
                raise RuntimeError("QSA top-k reuse requires an output buffer")
            return out, main_outputs
        return (
            self._select(q, compressed_metadata, out, kept_tail_rows),
            main_outputs,
        )


__all__ = ["QSAIndexer", "apply_qsa_rope"]
