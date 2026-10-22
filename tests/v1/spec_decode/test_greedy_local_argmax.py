# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The local-argmax greedy verify must match the stock greedy rejection sampler.

Runs on GPU, or on CPU with ``TRITON_INTERPRET=1``. TP is simulated by
splitting the vocab into shards and concatenating the per-shard partials
rank-major, as ``tensor_model_parallel_all_gather(partials, dim=0)`` does.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.greedy_local_argmax import (
    GreedyLocalArgmaxVerifier,
    _find_head_owner,
    greedy_verify_from_partials,
    local_argmax_partials,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif os.environ.get("TRITON_INTERPRET") == "1":
    DEVICE = "cpu"
else:
    pytest.skip("Needs a GPU or TRITON_INTERPRET=1", allow_module_level=True)


def _sharded_partials(logits: torch.Tensor, tp_size: int) -> torch.Tensor:
    """Per-rank partials of a vocab-parallel lm_head, gathered rank-major."""
    vocab = logits.shape[1]
    shard = -(-vocab // tp_size)
    parts = []
    for rank in range(tp_size):
        start = rank * shard
        local = logits[:, start : start + shard]
        n_valid = local.shape[1]
        if n_valid < shard:
            # The last shard is padded; padding must never win.
            pad = torch.full(
                (logits.shape[0], shard - n_valid),
                1e4,
                dtype=logits.dtype,
                device=logits.device,
            )
            local = torch.cat([local, pad], dim=1)
        parts.append(local_argmax_partials(local, n_valid, start))
    return torch.cat(parts, dim=0)


def _make_batch(num_drafts: list[int], vocab: int, dtype, seed: int):
    gen = torch.Generator().manual_seed(seed)
    num_logits = sum(n + 1 for n in num_drafts)
    logits = torch.randn(num_logits, vocab, generator=gen).to(dtype)
    # Ties: duplicate the row max at a later column (possibly another shard).
    for row in range(0, num_logits, 3):
        col = int(torch.randint(vocab, (1,), generator=gen))
        logits[row, col] = logits[row].max()
    logits[1, :] = 0.0  # All-equal row resolves to token 0.

    target_argmax = logits.float().argmax(dim=-1)
    cu = np.concatenate([[0], np.cumsum([n + 1 for n in num_drafts])])
    draft_sampled = torch.zeros(num_logits, dtype=torch.int64)
    for req, n in enumerate(num_drafts):
        start = int(cu[req])
        for i in range(n):
            # Mostly accept, sometimes reject, sometimes a -1 placeholder.
            r = float(torch.rand((), generator=gen))
            tok = int(target_argmax[start + i])
            if r < 0.15:
                tok = (tok + 1) % vocab
            elif r < 0.2:
                tok = -1
            draft_sampled[start + i + 1] = tok
    return logits, draft_sampled, torch.tensor(cu, dtype=torch.int32)


def _stock_greedy(
    logits, draft_sampled, cu_num_logits, num_spec_steps, synthetic_rates=None
):
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits = logits.shape[0]
    counts = (cu_num_logits[1:] - cu_num_logits[:-1]).long()
    expanded_idx_mapping = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int32, device=logits.device),
        counts.to(logits.device),
    )
    starts = cu_num_logits[:-1].long().to(logits.device)
    expanded_local_pos = (
        torch.arange(num_logits, device=logits.device)
        - starts[expanded_idx_mapping.long()]
    ).to(torch.int32)
    return rejection_sample(
        logits,
        None,
        draft_sampled,
        cu_num_logits,
        torch.arange(num_logits, dtype=torch.int32, device=logits.device),
        torch.arange(num_reqs, dtype=torch.int32, device=logits.device),
        expanded_idx_mapping,
        expanded_local_pos,
        torch.zeros(num_reqs, dtype=torch.float32, device=logits.device),
        _seeds(num_reqs, logits.device),
        num_spec_steps,
        synthetic_rates,
    )


def _seeds(num_reqs: int, device) -> torch.Tensor:
    return 1000 + 7919 * torch.arange(num_reqs, dtype=torch.int64, device=device)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("vocab", [3001, 12_000])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_matches_stock_greedy_rejection(tp_size, vocab, dtype):
    """Accepted tokens and counts are bit-identical to rejection_sample."""
    num_spec_steps = 3
    # Includes 1-logit (prefill / no-draft) requests.
    num_drafts = [3, 3, 0, 3, 2, 3, 0, 3, 3, 1]
    logits, draft_sampled, cu = _make_batch(num_drafts, vocab, dtype, seed=vocab)
    logits = logits.to(DEVICE)
    draft_sampled = draft_sampled.to(DEVICE)
    cu = cu.to(DEVICE)

    ref_sampled, ref_num = _stock_greedy(logits, draft_sampled, cu, num_spec_steps)
    partials = _sharded_partials(logits, tp_size)
    sampled, num = greedy_verify_from_partials(
        partials, tp_size, draft_sampled, cu, num_spec_steps
    )

    torch.testing.assert_close(num, ref_num, rtol=0, atol=0)
    for req in range(len(num_drafts)):
        n = int(ref_num[req])
        assert torch.equal(sampled[req, :n], ref_sampled[req, :n]), req
    # The batch exercises full acceptance, rejection and placeholders.
    accepted = (ref_num.cpu() - 1).tolist()
    assert any(a == d for a, d in zip(accepted, num_drafts) if d > 0)
    assert any(a < d for a, d in zip(accepted, num_drafts))


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize(
    "rates", [[0.77, 0.61, 0.45], [1.0, 1.0, 1.0], [0.0, 0.5, 1.0]]
)
def test_matches_stock_synthetic_greedy_rejection(tp_size, rates):
    """Synthetic acceptance: same draws, accepted lengths and tokens as stock."""
    from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

    num_spec_steps, vocab = 3, 5000
    num_drafts = [3] * 24 + [0, 2, 1]
    logits, draft_sampled, cu = _make_batch(num_drafts, vocab, torch.bfloat16, 7)
    logits = logits.to(DEVICE)
    draft_sampled = draft_sampled.to(DEVICE)
    cu = cu.to(DEVICE)
    cond = torch.tensor(
        unconditional_to_conditional_rates(rates), dtype=torch.float32, device=DEVICE
    )
    num_reqs = len(num_drafts)

    ref_sampled, ref_num = _stock_greedy(
        logits, draft_sampled, cu, num_spec_steps, cond
    )
    sampled, num = greedy_verify_from_partials(
        _sharded_partials(logits, tp_size),
        tp_size,
        draft_sampled,
        cu,
        num_spec_steps,
        cond,
        torch.arange(num_reqs, dtype=torch.int32, device=DEVICE),
        _seeds(num_reqs, DEVICE),
        torch.arange(logits.shape[0], dtype=torch.int32, device=DEVICE),
    )

    torch.testing.assert_close(num, ref_num, rtol=0, atol=0)
    for req in range(num_reqs):
        n = int(ref_num[req])
        assert torch.equal(sampled[req, :n], ref_sampled[req, :n]), req
    if rates == [0.77, 0.61, 0.45]:
        accepted = (ref_num.cpu() - 1).tolist()
        assert len(set(accepted[:24])) > 1


def test_ties_pick_smallest_id_across_shards():
    """A max shared by two shards resolves to the lower shard's token."""
    vocab, tp_size = 4096, 2
    logits = torch.zeros(2, vocab, device=DEVICE)
    logits[0, 100] = logits[0, 3000] = 5.0
    logits[1, 2048] = logits[1, 2049] = 7.0
    logits[1, 10] = float("nan")
    draft_sampled = torch.tensor([0, 100], device=DEVICE)
    cu = torch.tensor([0, 2], dtype=torch.int32, device=DEVICE)
    sampled, num = greedy_verify_from_partials(
        _sharded_partials(logits, tp_size), tp_size, draft_sampled, cu, 1
    )
    assert int(num[0]) == 2
    assert sampled[0].tolist() == [100, 2048]


def _verifier(temps, needs_processing, num_logprobs, vocab=1000):
    verifier = object.__new__(GreedyLocalArgmaxVerifier)
    states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.array(temps, dtype=np.float32)),
        max_num_logprobs=lambda idx: int(np.max(np.array(num_logprobs)[idx])),
    )
    verifier.sampler = SimpleNamespace(
        sampling_states=states,
        needs_logits_processing=np.array(needs_processing, dtype=bool),
    )
    verifier.vocab_size = vocab
    return verifier


@pytest.mark.parametrize(
    "temps,needs,logprobs,num_draft_tokens,expected",
    [
        ([0.0, 0.0], [False, False], [NO_LOGPROBS] * 2, 6, True),
        ([0.0, 0.7], [False, False], [NO_LOGPROBS] * 2, 6, False),
        ([0.0, 0.0], [False, True], [NO_LOGPROBS] * 2, 6, False),
        ([0.0, 0.0], [False, False], [NO_LOGPROBS, 5], 6, False),
        ([0.0, 0.0], [False, False], [NO_LOGPROBS] * 2, 0, False),
    ],
)
def test_can_verify_falls_back(temps, needs, logprobs, num_draft_tokens, expected):
    """Non-greedy, logits processing, logprobs and no-draft batches fall back."""
    verifier = _verifier(temps, needs, logprobs)
    batch = SimpleNamespace(
        num_reqs=2,
        num_draft_tokens=num_draft_tokens,
        idx_mapping_np=np.array([0, 1]),
    )
    assert verifier.can_verify(batch, None) is expected
    # A draft vocab narrower than the target changes the stock argmax range.
    small_draft = torch.empty(1, 1, 999)
    assert verifier.can_verify(batch, small_draft) is False


def test_find_head_owner_unwraps_language_model():
    """Multimodal wrappers delegate compute_logits to their language model."""
    lm = nn.Module()
    lm.lm_head = object.__new__(ParallelLMHead)
    lm.logits_processor = object.__new__(LogitsProcessor)
    wrapper = nn.Module()
    wrapper.language_model = lm
    assert _find_head_owner(lm) is lm
    assert _find_head_owner(wrapper) is lm
    assert _find_head_owner(nn.Module()) is None


def _head_owner(tp_size: int, vocab: int = 1000) -> nn.Module:
    owner = nn.Module()
    lm_head = object.__new__(ParallelLMHead)
    lm_head.__dict__.update(
        tp_size=tp_size,
        parallel_group=None,
        org_vocab_size=vocab,
        shard_indices=SimpleNamespace(
            num_added_elements=0,
            num_org_elements=-(-vocab // tp_size),
            org_vocab_start_index=0,
        ),
    )
    logits_processor = object.__new__(LogitsProcessor)
    logits_processor.__dict__.update(org_vocab_size=vocab, logits_as_input=False)
    owner.lm_head = lm_head
    owner.logits_processor = logits_processor
    return owner


def _rejection_sampler() -> RejectionSampler:
    sampler = object.__new__(Sampler)
    sampler.__dict__.update(compute_nans=False, return_sampling_mask=False)
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.__dict__.update(
        sampler=sampler,
        watermark_key=None,
        enable_adaptive_verification=False,
        use_block_verification=False,
    )
    return rejection_sampler


@pytest.mark.parametrize(
    "use_local_argmax_reduction,tp_size,enabled",
    [(False, 2, False), (True, 2, True), (True, 1, False)],
)
def test_maybe_create_gating(use_local_argmax_reduction, tp_size, enabled):
    """Opt-in via use_local_argmax_reduction, and only for vocab-parallel TP."""
    spec_config = SimpleNamespace(use_local_argmax_reduction=use_local_argmax_reduction)
    verifier = GreedyLocalArgmaxVerifier.maybe_create(
        _head_owner(tp_size), _rejection_sampler(), spec_config
    )
    assert (verifier is not None) is enabled
    assert GreedyLocalArgmaxVerifier.maybe_create(_head_owner(2), None, None) is None
