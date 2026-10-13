# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-major QSA indexer for uniform spec-verify batches (ROCm).

The scoring tests also run on CPU under the Triton interpreter in FP32:
``CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 pytest <this file>``.
"""

import math
import os
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.amd.indexer_qsa import QSAIndexer
from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.amd.ops import qsa_uniform_indexer as uniform_ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if current_platform.is_rocm() else "cpu"
# The interpreter computes BF16 inaccurately, so it checks the FP32 path.
DTYPE = torch.bfloat16 if current_platform.is_rocm() else torch.float32
COMPRESS_RATIO = 4
HEAD_DIM = 128

requires_scoring = pytest.mark.skipif(
    not HAS_TRITON or not (current_platform.is_rocm() or INTERPRET),
    reason="needs ROCm, or the Triton interpreter on CPU",
)
requires_rocm = pytest.mark.skipif(
    not HAS_TRITON or not current_platform.is_rocm(),
    reason="row-kernel comparison and top-k need ROCm",
)


def _batch(
    context_lens: list[int],
    query_lens: list[int],
    page_size: int,
    num_heads: int,
    padding_rows: int = 0,
    seed: int = 0,
):
    """Rows of request r are its last query_lens[r] positions, request-major,
    plus trailing padding rows, as the QSA builder makes."""
    torch.manual_seed(seed)
    num_requests = len(context_lens)
    pages_per_request = [
        math.ceil(length // COMPRESS_RATIO / page_size) for length in context_lens
    ]
    width = max(pages_per_request) + 2
    num_pages = sum(pages_per_request) + 3
    k_cache = torch.randn(num_pages, page_size, 1, HEAD_DIM).to(DTYPE)
    page_table = torch.full((num_requests, width), -1, dtype=torch.int32)
    physical = torch.randperm(num_pages)
    used = 0
    for request, count in enumerate(pages_per_request):
        page_table[request, :count] = physical[used : used + count]
        used += count
    rows = sum(query_lens) + padding_rows
    token_to_req = torch.zeros(rows, dtype=torch.int32)
    positions = torch.full((rows,), -1, dtype=torch.int64)
    start = 0
    for request, (length, query_len) in enumerate(zip(context_lens, query_lens)):
        token_to_req[start : start + query_len] = request
        positions[start : start + query_len] = torch.arange(length - query_len, length)
        start += query_len
    q = torch.randn(rows, num_heads, HEAD_DIM).to(DTYPE)
    return SimpleNamespace(
        q=q.to(DEVICE),
        k_cache=k_cache.to(DEVICE),
        page_table=page_table.to(DEVICE),
        token_to_req=token_to_req.to(DEVICE),
        positions=positions.to(DEVICE),
        seq_lens=torch.tensor(context_lens, dtype=torch.int32, device=DEVICE),
        query_lens=query_lens,
    )


def _reference_mqa_paged(
    q, k_cache, page_table, token_to_req, positions, seq_lens, compress_ratio
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch ``qsa_mqa_paged``: full-width scores and visible block counts."""
    req = token_to_req.long()
    visible = torch.minimum(
        (positions + 1) // compress_ratio,
        seq_lens[req].long() // compress_ratio,
    )
    pages = page_table[req].long()
    keys = k_cache[pages.clamp_min(0), :, 0, :].flatten(1, 2).float()
    scores = torch.einsum("rhd,rnd->rnh", q.float(), keys)
    logits = scores.relu().sum(-1) / math.sqrt(HEAD_DIM)
    page_size = k_cache.shape[1]
    valid_pages = (pages >= 0).repeat_interleave(page_size, dim=1)
    return logits.masked_fill(~valid_pages, -torch.inf), visible.to(torch.int32)


def _reference_scores(batch) -> tuple[torch.Tensor, torch.Tensor]:
    return _reference_mqa_paged(
        batch.q,
        batch.k_cache,
        batch.page_table,
        batch.token_to_req,
        batch.positions,
        batch.seq_lens,
        COMPRESS_RATIO,
    )


def _indexer(kv_cache, token_topk: int = 256):
    return SimpleNamespace(
        compressed_key_cache=SimpleNamespace(kv_cache=kv_cache),
        token_topk=token_topk,
        compress_ratio=COMPRESS_RATIO,
        output_width=token_topk + COMPRESS_RATIO - 1,
    )


def _metadata(batch, num_prefills: int, num_decode_tokens: int, decode_query_len):
    return SimpleNamespace(
        block_table=batch.page_table,
        token_to_req=batch.token_to_req,
        logical_positions=batch.positions,
        seq_lens=batch.seq_lens,
        num_prefills=num_prefills,
        num_decode_tokens=num_decode_tokens,
        decode_query_len=decode_query_len,
    )


def _assert_same_selection(actual, expected, scores) -> None:
    """Equal token sets up to swaps between blocks tied within FP32 rounding."""
    for row in range(actual.shape[0]):
        new = set(actual[row].tolist()) - {-1}
        old = set(expected[row].tolist()) - {-1}
        assert len(new) == len(old)
        if new == old:
            continue
        gained = sorted(scores[row, t // COMPRESS_RATIO].item() for t in new - old)
        lost = sorted(scores[row, t // COMPRESS_RATIO].item() for t in old - new)
        assert gained == pytest.approx(lost, rel=1e-4, abs=1e-4)


def _assert_visible_scores_close(actual, expected, visible) -> None:
    for row, count in enumerate(visible.tolist()):
        torch.testing.assert_close(
            actual[row, :count], expected[row, :count], rtol=1e-4, atol=1e-4
        )


CASES = [
    pytest.param([5, 70, 263, 1031], 4, 4, 4, 0, id="q4_page4"),
    pytest.param([2, 517, 129, 2048], 3, 16, 8, 5, id="q3_page16_padded"),
    pytest.param([1999, 64, 3, 770, 33], 2, 64, 4, 2, id="q2_page64_padded"),
    pytest.param([4100, 4096], 4, 16, 8, 3, id="q4_multi_tile"),
]


@requires_scoring
@pytest.mark.parametrize(
    ("context_lens", "query_len", "page_size", "num_heads", "padding_rows"), CASES
)
def test_uniform_scores_match_reference(
    context_lens, query_len, page_size, num_heads, padding_rows
) -> None:
    batch = _batch(
        context_lens,
        [query_len] * len(context_lens),
        page_size,
        num_heads,
        padding_rows,
    )
    logits, visible = uniform_ops.qsa_mqa_paged_uniform(
        batch.q,
        batch.k_cache,
        batch.page_table,
        batch.token_to_req,
        batch.positions,
        batch.seq_lens,
        COMPRESS_RATIO,
        query_len,
    )
    expected, expected_visible = _reference_scores(batch)
    assert torch.equal(visible, expected_visible)
    _assert_visible_scores_close(logits, expected, visible)


@requires_rocm
@pytest.mark.parametrize(
    ("context_lens", "query_len", "page_size", "num_heads", "padding_rows"), CASES
)
def test_uniform_scores_match_row_kernel(
    context_lens, query_len, page_size, num_heads, padding_rows
) -> None:
    batch = _batch(
        context_lens,
        [query_len] * len(context_lens),
        page_size,
        num_heads,
        padding_rows,
    )
    args = (
        batch.q,
        batch.k_cache,
        batch.page_table,
        batch.token_to_req,
        batch.positions,
        batch.seq_lens,
        COMPRESS_RATIO,
    )
    logits, visible = uniform_ops.qsa_mqa_paged_uniform(*args, query_len)
    row_logits, row_visible = qsa_ops.qsa_mqa_paged(*args)
    assert torch.equal(visible, row_visible)
    _assert_visible_scores_close(logits, row_logits, visible)


@requires_rocm
@pytest.mark.parametrize(
    ("context_lens", "query_len", "page_size", "num_heads", "padding_rows"), CASES
)
def test_uniform_selection_matches_row_selection(
    monkeypatch, context_lens, query_len, page_size, num_heads, padding_rows
) -> None:
    token_topk = 256
    batch = _batch(
        context_lens,
        [query_len] * len(context_lens),
        page_size,
        num_heads,
        padding_rows,
    )
    # Five rows' worth of workspace: chunks must be rounded to query_len.
    columns = batch.page_table.shape[1] * page_size
    monkeypatch.setattr(uniform_ops, "_LOGITS_WORKSPACE_BYTES", 5 * columns * 4)
    args = (
        batch.q,
        batch.k_cache,
        batch.page_table,
        batch.token_to_req,
        batch.positions,
        batch.seq_lens,
        token_topk,
        COMPRESS_RATIO,
    )
    expected = qsa_ops.qsa_select_paged_tokens(*args)
    actual = uniform_ops.qsa_select_paged_tokens_uniform(*args, query_len)

    _assert_same_selection(actual, expected, _reference_scores(batch)[0])


@pytest.mark.parametrize(
    ("num_prefills", "decode_tokens", "query_len", "rows", "calls"),
    [
        (0, 8, 4, 8, [("uniform", 8, 4)]),
        (0, 8, 1, 8, [("row", 8)]),
        (0, 10, 5, 10, [("row", 10)]),
        # Mixed: decode prefix to the uniform scorer, prefill tail per row.
        (3, 12, 4, 60, [("uniform", 12, 4), ("row", 48)]),
        (2, 0, 0, 40, [("row", 40)]),
    ],
)
def test_select_dispatch(
    monkeypatch, num_prefills, decode_tokens, query_len, rows, calls
) -> None:
    recorded = []

    def uniform(q, *args):
        assert args[2].shape[0] == q.shape[0] == args[-1].shape[0]
        recorded.append(("uniform", q.shape[0], args[-2]))
        args[-1].fill_(1)
        return args[-1]

    def row(q, *args):
        assert args[2].shape[0] == q.shape[0]
        recorded.append(("row", q.shape[0]))
        if args[-1] is not None:
            args[-1].fill_(2)
        return args[-1]

    monkeypatch.setattr(uniform_ops, "qsa_select_paged_tokens_uniform", uniform)
    monkeypatch.setattr(qsa_ops, "qsa_select_paged_tokens", row)
    indexer = _indexer("cache")
    batch = SimpleNamespace(
        page_table="table",
        token_to_req=torch.zeros(rows, dtype=torch.int32),
        positions=torch.zeros(rows, dtype=torch.int64),
        seq_lens="seq_lens",
    )
    metadata = _metadata(batch, num_prefills, decode_tokens, query_len)
    out = torch.zeros(rows, indexer.output_width, dtype=torch.int32)
    result = QSAIndexer._select(indexer, torch.zeros(rows, 1, 1), metadata, out)
    assert recorded == calls
    assert result is out
    if calls[0][0] == "uniform":
        uniform_rows = calls[0][1]
        assert torch.all(out[:uniform_rows] == 1)
        assert torch.all(out[uniform_rows:] == 2)


def _mixed_batch(order: str):
    """Spec decodes plus prefill chunks in current-main or #59098 order.

    ``main`` lets a fresh chunk of exactly decode_query_len tokens join the
    leading uniform run; ``extends_first`` (vllm#59098) moves it behind the
    context-carrying extend.
    """
    decodes = ([1031, 70, 263], [4, 4, 4])
    extend = ([600], [35])
    fresh_q = ([4], [4])
    fresh = ([9], [9])
    if order == "main":
        parts = [decodes, fresh_q, extend, fresh]
    else:
        parts = [decodes, extend, fresh_q, fresh]
    context_lens = [c for p in parts for c in p[0]]
    query_lens = [q for p in parts for q in p[1]]
    return _batch(context_lens, query_lens, page_size=16, num_heads=4, seed=3)


def _split(query_lens: list[int], decode_query_len: int) -> tuple[int, int, int]:
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills

    query_start_loc = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(query_lens), 0, out=query_start_loc[1:])
    num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
        SimpleNamespace(
            max_query_len=max(query_lens),
            num_reqs=len(query_lens),
            num_actual_tokens=sum(query_lens),
            query_start_loc_cpu=query_start_loc,
        ),
        decode_threshold=decode_query_len,
        require_uniform=True,
    )
    return num_decodes, num_prefills, num_decode_tokens


def _reference_topk_per_row_decode(logits, next_n, lengths, out, *args) -> None:
    assert next_n == 1
    block_topk = out.shape[1]
    out.fill_(-1)
    for row, length in enumerate(lengths.tolist()):
        if length <= 0:
            continue
        if length <= block_topk:
            out[row, :length] = torch.arange(length, dtype=out.dtype)
        else:
            out[row] = logits[row, :length].topk(block_topk).indices.to(out.dtype)


def _copy_blocks(blocks, *args) -> None:
    out = args[-1]
    out.fill_(-1)
    out[:, : blocks.shape[1]] = blocks


@requires_scoring
@pytest.mark.parametrize("order", ["main", "extends_first"])
def test_mixed_selection_matches_reference(monkeypatch, order) -> None:
    """Decode rows of a mixed batch take the uniform scorer, the rest the row
    scorer, and every row selects the reference top-k over its visible blocks.

    Top-k, expansion and (off ROCm) row scoring are replaced by torch
    references, so this also runs on CPU under the interpreter; the comparison
    against the real kernels is ``test_mixed_selection_matches_row_selection``.
    """
    decode_query_len = 4
    batch = _mixed_batch(order)
    num_decodes, num_prefills, num_decode_tokens = _split(
        batch.query_lens, decode_query_len
    )
    assert num_decode_tokens == (16 if order == "main" else 12)
    assert num_prefills > 0

    monkeypatch.setattr(
        qsa_ops.ops, "top_k_per_row_decode", _reference_topk_per_row_decode
    )
    if not current_platform.is_rocm():
        monkeypatch.setattr(qsa_ops, "qsa_mqa_paged", _reference_mqa_paged)
    monkeypatch.setattr(qsa_ops, "expand_qsa_block_indices_cuda", _copy_blocks)
    monkeypatch.setattr(uniform_ops, "expand_qsa_block_indices_cuda", _copy_blocks)
    scored_rows = {"uniform": 0, "row": 0}
    for module, name, kind in (
        (uniform_ops, "qsa_mqa_paged_uniform", "uniform"),
        (qsa_ops, "qsa_mqa_paged", "row"),
    ):
        original = getattr(module, name)

        def counted(q, *args, _original=original, _kind=kind, **kwargs):
            scored_rows[_kind] += q.shape[0]
            return _original(q, *args, **kwargs)

        monkeypatch.setattr(module, name, counted)

    token_topk = 16
    indexer = _indexer(batch.k_cache, token_topk)
    metadata = _metadata(batch, num_prefills, num_decode_tokens, decode_query_len)
    actual = QSAIndexer._select(indexer, batch.q, metadata, None).cpu()

    rows = batch.q.shape[0]
    assert scored_rows == {
        "uniform": num_decode_tokens,
        "row": rows - num_decode_tokens,
    }
    scores, visible = _reference_scores(batch)
    scores, visible = scores.cpu(), visible.cpu()
    block_topk = token_topk // COMPRESS_RATIO
    for row, count in enumerate(visible.tolist()):
        new = set(actual[row, :block_topk].tolist()) - {-1}
        if count <= block_topk:
            assert new == set(range(count))
            continue
        old = set(scores[row, :count].topk(block_topk).indices.tolist())
        assert len(new) == block_topk
        gained = sorted(scores[row, b].item() for b in new - old)
        lost = sorted(scores[row, b].item() for b in old - new)
        assert gained == pytest.approx(lost, rel=1e-4, abs=1e-4)


@requires_rocm
@pytest.mark.parametrize("order", ["main", "extends_first"])
def test_mixed_selection_matches_row_selection(order) -> None:
    decode_query_len = 4
    token_topk = 16
    batch = _mixed_batch(order)
    _, num_prefills, num_decode_tokens = _split(batch.query_lens, decode_query_len)
    args = (
        batch.q,
        batch.k_cache,
        batch.page_table,
        batch.token_to_req,
        batch.positions,
        batch.seq_lens,
        token_topk,
        COMPRESS_RATIO,
    )
    expected = qsa_ops.qsa_select_paged_tokens(*args).cpu()
    indexer = _indexer(batch.k_cache, token_topk)
    metadata = _metadata(batch, num_prefills, num_decode_tokens, decode_query_len)
    actual = QSAIndexer._select(indexer, batch.q, metadata, None).cpu()
    scores, _ = _reference_scores(batch)
    _assert_same_selection(actual, expected, scores.cpu())
