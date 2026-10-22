# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-major QSA scoring for uniform decode/verify batches on ROCm.

``qsa_mqa_paged`` runs one program per query row and capacity-wide column
tile, so a spec-verify batch with ``q_len`` rows per request reads each
request's compressed keys ``q_len * heads`` times and writes ``-inf`` over the
whole ``max_model_len`` score row. Here one program owns all rows of a request:
it loads every key tile once, scores all ``q_len * heads`` queries with one
``tl.dot``, and stops at the request's last visible block. Scores are written
only for visible columns, which is all ``top_k_per_row_decode`` reads.
"""

from __future__ import annotations

import math

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton

from .qsa import (
    _LOGITS_WORKSPACE_BYTES,
    _validate_mqa,
    expand_qsa_block_indices_cuda,
)

logger = init_logger(__name__)

MAX_UNIFORM_QUERY_LEN = 4
_BLOCK_N = 64


@triton.jit
def _qsa_mqa_paged_uniform_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
) -> None:
    NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE
    HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # MFMA needs every tl.dot dimension >= 16; pad the packed query axis.
    QUERY_COLS: tl.constexpr = max(16, triton.next_power_of_2(QUERY_LEN) * HEADS_PADDED)
    QUERY_SLOTS: tl.constexpr = QUERY_COLS // HEADS_PADDED
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))

    group = tl.program_id(0)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    slots = tl.arange(0, QUERY_SLOTS)
    rows = group * QUERY_LEN + slots
    live_rows = (slots < QUERY_LEN) & (rows < num_rows)
    row_requests = tl.load(token_to_req_ptr + rows, mask=live_rows, other=-1)
    valid_rows = live_rows & (row_requests >= 0) & (row_requests < num_requests)
    safe_row_requests = tl.minimum(tl.maximum(row_requests, 0), num_requests - 1)
    query_positions = tl.load(query_positions_ptr + rows, mask=live_rows, other=0)
    sequence_lengths = tl.load(
        sequence_lengths_ptr + safe_row_requests, mask=valid_rows, other=0
    )
    # Same per-row bound as qsa_mqa_paged, including padding rows.
    visible = tl.minimum(
        (query_positions + 1) // COMPRESS_RATIO,
        sequence_lengths // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + rows, visible, mask=live_rows)
    visible = tl.where(valid_rows, visible, 0)
    max_visible = tl.minimum(tl.max(visible, axis=0), NUM_COLUMNS)
    if tile_start * BLOCK_N >= max_visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(max_visible, BLOCK_N))
    # Uniform batches are request-major: every live row of a group shares the
    # first row's request.
    request = tl.max(tl.where(slots == 0, safe_row_requests, 0), axis=0)

    dims = tl.arange(0, BLOCK_D)
    packed = tl.arange(0, QUERY_COLS)
    packed_slot = packed // HEADS_PADDED
    packed_head = packed % HEADS_PADDED
    packed_rows = group * QUERY_LEN + packed_slot
    valid_packed = (
        (packed_slot < QUERY_LEN) & (packed_head < NUM_HEADS) & (packed_rows < num_rows)
    )
    query = tl.load(
        q_ptr
        + packed_rows[None, :] * stride_q_row
        + packed_head[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim,
        mask=valid_packed[None, :] & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )

    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + request * stride_table_req
            + logical_page * stride_table_page,
            mask=live,
            other=-1,
        )
        live &= (physical_page >= 0) & (physical_page < num_pages)
        # physical_page * block stride can overflow int32 for large caches.
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        ).to(query.dtype)
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(valid_packed[None, :], tl.maximum(scores, 0.0), 0.0)
        score = tl.sum(tl.reshape(scores, (BLOCK_N, QUERY_SLOTS, HEADS_PADDED)), axis=2)
        score /= score_divisor
        score = tl.where(live[:, None], score, -float("inf"))
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=valid_rows[None, :]
            & (columns[:, None] < NUM_COLUMNS)
            & (columns[:, None] < visible[None, :]),
        )


def _tiles_per_program(num_groups: int, columns: int) -> int:
    programs = num_groups * triton.cdiv(columns, _BLOCK_N)
    if programs < 16384:
        return 1
    if programs < 32768:
        return 2
    if programs < 131072:
        return 4
    return 8


def qsa_mqa_paged_uniform(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    query_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``qsa_mqa_paged`` for request-major rows, ``query_len`` per request.

    Rows ``[r * query_len, (r + 1) * query_len)`` must belong to one request
    (trailing padding rows with no visible blocks are allowed). Scores match
    ``qsa_mqa_paged`` for columns below each row's visible-block count; the
    remaining columns are left unwritten.
    """
    _validate_mqa(q)
    if not HAS_TRITON:
        raise RuntimeError("paged QSA scoring requires Triton")
    if not 1 <= query_len <= MAX_UNIFORM_QUERY_LEN:
        raise ValueError("uniform QSA scoring supports 1-4 rows per request")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("QSA request mapping must match query rows")
    if query_positions.shape != (q.shape[0],):
        raise ValueError("QSA query positions must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")

    columns = page_table.shape[1] * k_cache.shape[1]
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    if not q.shape[0] or not columns:
        return logits, visible_blocks
    num_groups = triton.cdiv(q.shape[0], query_len)
    tiles_per_program = _tiles_per_program(num_groups, columns)
    grid = (num_groups, triton.cdiv(columns, _BLOCK_N * tiles_per_program))
    _qsa_mqa_paged_uniform_kernel[grid](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        k_cache.shape[0],
        page_table.shape[0],
        float(math.sqrt(q.shape[2])),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        QUERY_LEN=query_len,
        COMPRESS_RATIO=compress_ratio,
        BLOCK_N=_BLOCK_N,
        TILES_PER_PROG=tiles_per_program,
        num_warps=4,
    )
    return logits, visible_blocks


def qsa_select_paged_tokens_uniform(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    query_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """ROCm ``qsa_select_paged_tokens`` for request-major uniform batches."""
    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, output_width):
        raise ValueError("QSA selection output has an invalid shape")
    if not rows:
        return out
    logger.info_once(
        "Using the request-major QSA indexer for uniform spec-verify rows."
    )

    columns = page_table.shape[1] * k_cache.shape[1]
    block_topk = token_topk // compress_ratio
    # Chunks must keep each request's rows together.
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    rows_per_chunk = max(query_len, rows_per_chunk // query_len * query_len)
    chunk_rows = min(rows, rows_per_chunk)
    blocks_buffer = torch.empty(
        (chunk_rows, block_topk), dtype=torch.int32, device=q.device
    )
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        logits, visible_blocks = qsa_mqa_paged_uniform(
            q[row_slice],
            k_cache,
            page_table,
            token_to_req[row_slice],
            query_positions[row_slice],
            sequence_lengths,
            compress_ratio,
            query_len,
        )
        blocks = blocks_buffer[: row_end - row_start]
        # top_k_per_row_decode reads only the first visible_blocks columns.
        ops.top_k_per_row_decode(
            logits,
            1,
            visible_blocks,
            blocks,
            blocks.shape[0],
            logits.stride(0),
            logits.stride(1),
            block_topk,
        )
        expand_qsa_block_indices_cuda(
            blocks,
            query_positions[row_slice],
            sequence_lengths,
            token_to_req[row_slice],
            compress_ratio,
            token_topk,
            out[row_slice],
        )
    return out


__all__ = [
    "MAX_UNIFORM_QUERY_LEN",
    "qsa_mqa_paged_uniform",
    "qsa_select_paged_tokens_uniform",
]
