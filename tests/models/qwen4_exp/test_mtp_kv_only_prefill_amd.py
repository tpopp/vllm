# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-only Qwen4Exp MTP draft prefill (ROCm).

Draft step 0 keeps QSA selection and sparse attention only for the decode
rows and the last ``k + 1`` rows of every prefill request; the kept rows must
match the all-rows path bit-exactly.

The selection test also runs on CPU under the Triton interpreter in FP32:
``CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 pytest <this file>``.
"""

import os
import random
from types import SimpleNamespace

import pytest
import torch

import vllm.models.qwen4_exp.amd.model  # noqa: F401  (import order: qsa cycle)
from tests.models.qwen4_exp.test_qsa_uniform_indexer_amd import (
    _batch,
    _copy_blocks,
    _indexer,
    _metadata,
    _reference_mqa_paged,
    _reference_topk_per_row_decode,
)
from vllm.models.qwen4_exp.amd import qsa as qsa_owner
from vllm.models.qwen4_exp.amd.indexer_qsa import QSAIndexer
from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.amd.ops import qsa_uniform_indexer as uniform_ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
ROCM = current_platform.is_rocm()
DEVICE = "cuda" if ROCM else "cpu"
# The interpreter computes BF16 inaccurately, so it checks the FP32 path.
DTYPE = torch.bfloat16 if ROCM else torch.float32
COMPRESS_RATIO = 4
HEAD_DIM = 128
NUM_SPEC = 3

requires_kernels = pytest.mark.skipif(
    not HAS_TRITON or not (ROCM or INTERPRET),
    reason="needs ROCm, or the Triton interpreter on CPU",
)
requires_rocm = pytest.mark.skipif(
    not HAS_TRITON or not ROCM, reason="sparse attention kernel needs ROCm"
)


def _query_start_loc(query_lens: list[int]) -> torch.Tensor:
    loc = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(query_lens), 0, out=loc[1:])
    return loc


def _sampled_rows(query_lens: list[int], num_rejected: list[int]) -> list[int]:
    """Draft step 0's last_token_indices (prepare_prefill_inputs)."""
    loc = _query_start_loc(query_lens).tolist()
    return [
        loc[r] + q - n - 1 for r, (q, n) in enumerate(zip(query_lens, num_rejected))
    ]


def _kept(query_lens: list[int], num_decodes: int) -> torch.Tensor:
    loc = _query_start_loc(query_lens)
    num_decode_tokens = int(loc[num_decodes])
    tail = qsa_owner.kv_only_prefill_tail_rows(
        loc,
        num_decodes,
        len(query_lens) - num_decodes,
        int(loc[-1]),
        NUM_SPEC + 1,
    )
    return torch.cat((torch.arange(num_decode_tokens), tail))


@pytest.mark.parametrize("seed", range(10))
def test_kept_rows_cover_sampled_rows(seed: int) -> None:
    """Decode rows plus each tail request's last k + 1 rows contain every row
    draft step 0 samples, in either batch order (verify rows may trail)."""
    rng = random.Random(seed)
    num_decodes = rng.randint(0, 6)
    query_lens = [NUM_SPEC + 1] * num_decodes
    num_rejected = [rng.randint(0, NUM_SPEC) for _ in range(num_decodes)]
    for _ in range(rng.randint(1, 5)):
        if rng.random() < 0.3:
            # A spec request behind a prefill (vllm#59098 order) or short chunk.
            query_lens.append(rng.randint(1, NUM_SPEC + 1))
            num_rejected.append(rng.randint(0, query_lens[-1] - 1))
        else:
            query_lens.append(rng.randint(NUM_SPEC + 2, 300))
            num_rejected.append(0)
    kept = _kept(query_lens, num_decodes)
    loc = _query_start_loc(query_lens)
    tail_requests = len(query_lens) - num_decodes
    assert kept.numel() == int(loc[num_decodes]) + tail_requests * (NUM_SPEC + 1)
    assert set(_sampled_rows(query_lens, num_rejected)) <= set(kept.tolist())
    for row in kept[int(loc[num_decodes]) :].tolist():
        request = int(torch.searchsorted(loc, row, right=True)) - 1
        assert num_decodes <= request < len(query_lens)
        assert loc[request + 1] - row <= NUM_SPEC + 1


def _attention_owner(rows_per_request: int, skip_topk: bool = False):
    return SimpleNamespace(
        kv_only_prefill_rows_per_request=rows_per_request,
        indexer=SimpleNamespace(skip_topk=skip_topk),
    )


def _side_metadata(query_lens: list[int], num_decodes: int):
    loc = _query_start_loc(query_lens)
    return SimpleNamespace(
        query_start_loc=loc,
        num_decodes=num_decodes,
        num_prefills=len(query_lens) - num_decodes,
        num_decode_tokens=int(loc[num_decodes]),
        num_prefill_tokens=int(loc[-1] - loc[num_decodes]),
        num_actual_tokens=int(loc[-1]),
        token_to_req=None,
    )


@pytest.mark.parametrize(
    ("rows_per_request", "skip_topk", "query_lens", "num_decodes", "active"),
    [
        (0, False, [4, 4, 100], 2, False),  # target layers
        (4, False, [4, 4, 4], 3, False),  # pure decode: FULL graphs unchanged
        (4, True, [4, 4, 100], 2, False),  # index-sharing reuse mode
        (4, False, [4, 3, 2], 1, False),  # nothing to skip
        (4, False, [4, 4, 100], 2, True),
        (4, False, [100, 7], 0, True),
    ],
)
def test_kv_only_dispatch(
    rows_per_request, skip_topk, query_lens, num_decodes, active
) -> None:
    owner = _attention_owner(rows_per_request, skip_topk)
    metadata = _side_metadata(query_lens, num_decodes)
    rows = qsa_owner.Qwen4ExpQSAAttention._kv_only_prefill_tail_rows(owner, metadata)
    assert (rows is not None) == active
    if active:
        expected = _kept(query_lens, num_decodes)[metadata.num_decode_tokens :]
        assert torch.equal(rows, expected)


@requires_kernels
def test_kept_selection_matches_all_rows(monkeypatch) -> None:
    """Kept rows select exactly what the all-rows path selects; other rows of
    the top-k buffer are left untouched."""
    num_decodes = 3
    query_lens = [NUM_SPEC + 1] * num_decodes + [37, 2, 61]
    context_lens = [1031, 70, 263, 600, 9, 400]
    batch = _batch(context_lens, query_lens, page_size=16, num_heads=4, seed=1)
    if not ROCM:
        monkeypatch.setattr(
            qsa_ops.ops, "top_k_per_row_decode", _reference_topk_per_row_decode
        )
        monkeypatch.setattr(qsa_ops, "qsa_mqa_paged", _reference_mqa_paged)
        monkeypatch.setattr(qsa_ops, "expand_qsa_block_indices_cuda", _copy_blocks)
        monkeypatch.setattr(uniform_ops, "expand_qsa_block_indices_cuda", _copy_blocks)
    indexer = _indexer(batch.k_cache, token_topk=16)
    indexer._select_rows = lambda *args: QSAIndexer._select_rows(indexer, *args)
    loc = _query_start_loc(query_lens)
    metadata = _metadata(
        batch, len(query_lens) - num_decodes, int(loc[num_decodes]), NUM_SPEC + 1
    )
    rows = batch.q.shape[0]
    full = torch.empty(rows, indexer.output_width, dtype=torch.int32, device=DEVICE)
    QSAIndexer._select(indexer, batch.q, metadata, full)

    kept = _kept(query_lens, num_decodes).to(DEVICE)
    tail = kept[metadata.num_decode_tokens :]
    sentinel = -7
    partial = torch.full_like(full, sentinel)
    QSAIndexer._select(indexer, batch.q, metadata, partial, tail)
    assert torch.equal(partial[kept], full[kept])
    skipped = torch.ones(rows, dtype=torch.bool, device=DEVICE)
    skipped[kept] = False
    assert skipped.any()
    assert torch.all(partial[skipped] == sentinel)


def _expanded_indices(positions, seq_len, budget, generator) -> torch.Tensor:
    width = budget + COMPRESS_RATIO - 1
    out = torch.full((len(positions), width), -1, dtype=torch.int32)
    for row, position in enumerate(positions):
        visible = min((position + 1) // COMPRESS_RATIO, seq_len // COMPRESS_RATIO)
        blocks = torch.randperm(visible, generator=generator)[
            : budget // COMPRESS_RATIO
        ]
        tokens = (
            blocks[:, None] * COMPRESS_RATIO + torch.arange(COMPRESS_RATIO)
        ).flatten()
        tail_start = (position + 1) // COMPRESS_RATIO * COMPRESS_RATIO
        tail = torch.arange(tail_start, position + 1)[: COMPRESS_RATIO - 1]
        selected = torch.cat([tokens, tail])
        out[row, : selected.numel()] = selected.to(torch.int32)
    return out


def _attention_batch(query_lens, context_lens, heads, budget, page_size, seed):
    generator = torch.Generator().manual_seed(seed)
    pages = [-(-length // page_size) for length in context_lens]
    width = max(pages) + 1
    num_pages = sum(pages) + 2
    kv = torch.randn(num_pages, page_size, 1, 2 * HEAD_DIM, generator=generator)
    k_cache, v_cache = kv.to(DTYPE).split(HEAD_DIM, dim=-1)
    block_table = torch.full((len(context_lens), width), -1, dtype=torch.int32)
    physical = torch.randperm(num_pages, generator=generator)
    used = 0
    for request, count in enumerate(pages):
        block_table[request, :count] = physical[used : used + count]
        used += count
    token_to_req = []
    indices = []
    for request, (length, query_len) in enumerate(zip(context_lens, query_lens)):
        positions = list(range(length - query_len, length))
        token_to_req += [request] * query_len
        indices.append(_expanded_indices(positions, length, budget, generator))
    q = torch.randn(len(token_to_req), heads, HEAD_DIM, generator=generator)
    return SimpleNamespace(
        q=q.to(DTYPE).to(DEVICE),
        k_cache=k_cache.to(DEVICE),
        v_cache=v_cache.to(DEVICE),
        indices=torch.cat(indices).to(DEVICE),
        block_table=block_table.to(DEVICE),
        token_to_req=torch.tensor(token_to_req, dtype=torch.int32, device=DEVICE),
    )


@requires_rocm
@pytest.mark.parametrize(
    ("query_lens", "context_lens", "heads", "budget"),
    [
        # 40 rows -> (64, 8) profile; the 17 kept rows alone would pick (16, 32).
        ([4, 4, 4, 28], [300, 90, 520, 700], 6, 256),
        # Pure-prefill tail; the 8 kept rows alone would pick (16, 64)/(16, 32).
        ([30, 30], [400, 64], 4, 128),
    ],
)
def test_kept_attention_matches_all_rows(
    query_lens, context_lens, heads, budget
) -> None:
    """Kept rows attend with the full batch's launch profile: bit-exact; the
    skipped rows stay zero (forward_qsa zeroes the output first)."""
    batch = _attention_batch(query_lens, context_lens, heads, budget, 16, seed=2)
    args = (batch.q, batch.k_cache, batch.v_cache, batch.indices, batch.block_table)
    full = torch.zeros_like(batch.q)
    qsa_ops.qsa_sparse_paged_attention(*args, batch.token_to_req, full)

    num_decodes = sum(q == NUM_SPEC + 1 for q in query_lens)
    kept = _kept(query_lens, num_decodes).to(DEVICE)
    partial = torch.zeros_like(batch.q)
    qsa_owner.qsa_sparse_paged_attention_rows(*args, batch.token_to_req, partial, kept)
    assert torch.equal(partial[kept], full[kept])
    skipped = torch.ones(batch.q.shape[0], dtype=torch.bool, device=DEVICE)
    skipped[kept] = False
    assert torch.all(partial[skipped] == 0)


def test_run_qsa_plumbing(monkeypatch) -> None:
    """_run_qsa passes the kept rows to the indexer and to forward_qsa only on
    the KV-only path, and leaves both calls unchanged otherwise."""
    query_lens = [4, 4, 50]
    metadata = _side_metadata(query_lens, 2)
    calls = {}

    class FakeIndexer:
        skip_topk = False
        raw_key_cache = SimpleNamespace(prefix="raw")

        def __call__(self, *args, **kwargs):
            calls["indexer"] = kwargs
            return torch.zeros(metadata.num_actual_tokens, 3), None

    class FakeImpl(qsa_owner.Qwen4ExpQSAFlashAttentionImpl):
        def __init__(self):
            pass

        def do_kv_cache_update(self, *args):
            pass

        def forward_qsa(self, *args, **kwargs):
            calls["attention"] = kwargs

    main = SimpleNamespace(
        num_actual_tokens=metadata.num_actual_tokens, slot_mapping=None
    )
    monkeypatch.setattr(
        qsa_owner,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"main": main, "raw": metadata}),
    )
    for rows_per_request, active in ((0, False), (NUM_SPEC + 1, True)):
        calls.clear()
        owner = SimpleNamespace(
            layer_name="main",
            kv_cache=torch.ones(1),
            indexer=FakeIndexer(),
            topk_indices_buffer=torch.zeros(64, 3),
            impl=FakeImpl(),
            kv_only_prefill_rows_per_request=rows_per_request,
        )
        owner.indexer.output_width = 3
        owner._kv_only_prefill_tail_rows = lambda md, o=owner: (
            qsa_owner.Qwen4ExpQSAAttention._kv_only_prefill_tail_rows(o, md)
        )
        q = torch.zeros(metadata.num_actual_tokens, 1, 1)
        qsa_owner.Qwen4ExpQSAAttention._run_qsa(
            owner, q, None, q, q, q, torch.zeros_like(q)
        )
        kept = calls["attention"]["kept_rows"]
        if not active:
            assert "kept_tail_rows" not in calls["indexer"]
            assert kept is None
            continue
        expected = _kept(query_lens, 2)
        assert torch.equal(kept, expected)
        assert torch.equal(
            calls["indexer"]["kept_tail_rows"], expected[metadata.num_decode_tokens :]
        )
