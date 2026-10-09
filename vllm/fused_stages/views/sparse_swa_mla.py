# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Views of DeepSeek V4/V4.1 sparse-SWA MLA attention state.

Builders: ``vllm/models/deepseek_v41/amd/fused_stages.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.fused_stages.api import View


@dataclass(frozen=True)
class SparseSWAMLACacheViewV1(View):
    """One layer's KV caches as raw records. Built at KV-cache bind time.

    Record layout (``fp8_ds_mla``, 584 bytes): 576 data bytes per token, then
    the block's scale words after its data rows; kernels index both from the
    ``[blocks, block_size, record_bytes]`` view.
    """

    TYPE_ID = "dsv4.sparse_swa_mla.cache@1"

    swa_records: torch.Tensor
    """[num_blocks, swa_block_size, record_bytes] uint8 view of the SWA cache."""
    swa_block_size: int
    comp_records: torch.Tensor | None
    """[num_blocks, block_size // compress_ratio, record_bytes] uint8 view of
    the compressed cache, or None for layers without one."""
    comp_block_size: int
    record_bytes: int
    kv_cache_dtype: str
    compress_ratio: int


@dataclass(frozen=True)
class SparseSWAMLAStepViewV1(View):
    """One layer's per-step decode metadata. Device tensors only; every
    tensor is graph-static (same storage on every replay of a captured key).
    """

    TYPE_ID = "dsv4.sparse_swa_mla.step@1"

    slot_mapping: torch.Tensor
    """[M] int64; -1 for padding rows (must not be written)."""
    swa_indices: torch.Tensor
    """[>= M, 1, swa_width] int32 slots of each token's sliding window."""
    swa_lens: torch.Tensor
    """[>= M] int32 valid entries of each window."""
    swa_width: int
    token_to_req: torch.Tensor
    """[>= M] int32 request index of each token."""
    topk_indices: torch.Tensor | None
    """Shared top-k indices buffer written by the index-source layer."""
    comp_block_table: torch.Tensor | None
    """Block table of the compressed cache, or None."""
    comp_block_size: int
    """Compressed records per block as the metadata sees it; providers check
    it once against ``SparseSWAMLACacheViewV1.comp_block_size``."""
