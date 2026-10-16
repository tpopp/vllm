# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side eligibility and gating logic of ``PostForwardCudaGraphs``.

CUDA graph capture/replay itself needs a real CUDA (or ROCm/HIP) device and is
not exercised here; see ``gather/post_forward_graph.md`` for what remains to
be validated on GPU. Everything tested here -- ``create()``'s gating and
``_step_eligible()``'s per-step checks -- is plain host-side Python/NumPy
logic that runs identically on CPU.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR
from vllm.v1.worker.gpu.post_forward_cudagraph import (
    PostForwardCudaGraphs,
    _reading_copies,
    _tensor_key,
)
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler


def _mtp_speculator() -> MTPSpeculator:
    # isinstance() needs no __init__ state.
    return object.__new__(MTPSpeculator)


def _make_runner(**overrides) -> SimpleNamespace:
    hidden_states = torch.zeros(8, 4)
    base = dict(
        speculator=_mtp_speculator(),
        sampler=SimpleNamespace(
            compute_nans=False,
            return_sampling_mask=False,
            needs_logits_processing=np.zeros(4, dtype=bool),
            sampling_states=SimpleNamespace(max_num_logprobs=lambda idx: NO_LOGPROBS),
        ),
        rejection_sampler=object.__new__(RejectionSampler),
        vllm_config=SimpleNamespace(watermark_config=None),
        use_pp=False,
        pcp_manager=None,
        dcp_size=1,
        batch_sharder=None,
        adaptive_verification=None,
        lora_config=None,
        kv_connector=NO_OP_KV_CONNECTOR,
        aux_output_connector=None,
        fast_prefill=None,
        max_num_reqs=4,
        max_num_tokens=16,
        device=torch.device("cpu"),
        num_speculative_steps=2,
        vocab_size=128,
        cudagraph_manager=SimpleNamespace(hidden_states=hidden_states),
        prompt_logprobs_worker=SimpleNamespace(
            uses_prompt_logprobs=np.zeros(4, dtype=bool),
            token_id_scores={},
        ),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_input_batch(**overrides) -> SimpleNamespace:
    # num_reqs=2, num_speculative_steps=2 -> uniform decode: each request
    # schedules num_steps + 1 = 3 tokens, num_draft_tokens = num_reqs*steps = 4.
    base = dict(
        num_reqs=2,
        has_prefill=False,
        has_structured_output_reqs=False,
        num_draft_tokens=4,
        num_scheduled_tokens=np.array([3, 3], dtype=np.int32),
        num_tokens=6,
        idx_mapping_np=np.array([0, 1], dtype=np.int64),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def graphs() -> PostForwardCudaGraphs:
    return PostForwardCudaGraphs(max_num_reqs=4, max_num_tokens=16, device="cpu")


class TestStepEligible:
    def test_uniform_decode_step_is_eligible(self, graphs):
        runner = _make_runner()
        batch = _make_input_batch()
        hidden_states = runner.cudagraph_manager.hidden_states
        assert graphs._step_eligible(runner, batch, hidden_states, None) is True

    def test_non_static_hidden_states_ineligible(self, graphs):
        runner = _make_runner()
        batch = _make_input_batch()
        # A different tensor (e.g. from an eager, non-FULL-graph target step)
        # does not share the target graph's static output buffer address.
        other_hidden_states = torch.zeros(8, 4)
        assert graphs._step_eligible(runner, batch, other_hidden_states, None) is False

    def test_no_target_cudagraph_manager_ineligible(self, graphs):
        runner = _make_runner(cudagraph_manager=None)
        batch = _make_input_batch()
        hidden_states = torch.zeros(8, 4)
        assert graphs._step_eligible(runner, batch, hidden_states, None) is False

    def test_zero_reqs_ineligible(self, graphs):
        runner = _make_runner()
        batch = _make_input_batch(num_reqs=0, num_draft_tokens=0)
        hidden_states = runner.cudagraph_manager.hidden_states
        assert graphs._step_eligible(runner, batch, hidden_states, None) is False

    def test_prefill_ineligible(self, graphs):
        runner = _make_runner()
        batch = _make_input_batch(has_prefill=True)
        hidden_states = runner.cudagraph_manager.hidden_states
        assert graphs._step_eligible(runner, batch, hidden_states, None) is False

    def test_grammar_output_ineligible(self, graphs):
        runner = _make_runner()
        batch = _make_input_batch()
        hidden_states = runner.cudagraph_manager.hidden_states
        assert graphs._step_eligible(runner, batch, hidden_states, object()) is False

    def test_structured_output_reqs_ineligible(self, graphs):
        runner = _make_runner()
        batch = _make_input_batch(has_structured_output_reqs=True)
        hidden_states = runner.cudagraph_manager.hidden_states
        assert graphs._step_eligible(runner, batch, hidden_states, None) is False

    def test_non_uniform_draft_tokens_ineligible(self, graphs):
        runner = _make_runner()
        # num_draft_tokens should be num_reqs * num_steps == 4; 3 is a
        # non-uniform (e.g. partially rejected-and-repadded) shape.
        batch = _make_input_batch(num_draft_tokens=3)
        hidden_states = runner.cudagraph_manager.hidden_states
        assert graphs._step_eligible(runner, batch, hidden_states, None) is False

    def test_non_uniform_scheduled_tokens_ineligible(self, graphs):
        runner = _make_runner()
        batch = _make_input_batch(num_scheduled_tokens=np.array([3, 2], dtype=np.int32))
        hidden_states = runner.cudagraph_manager.hidden_states
        assert graphs._step_eligible(runner, batch, hidden_states, None) is False

    def test_chunk_too_large_ineligible(self, graphs):
        # A huge vocab_size shrinks get_max_chunk_logits() below this batch's
        # num_tokens, so even a tiny batch exceeds one verification chunk.
        runner = _make_runner(vocab_size=1 << 30)
        batch = _make_input_batch()
        hidden_states = runner.cudagraph_manager.hidden_states
        assert graphs._step_eligible(runner, batch, hidden_states, None) is False

    def test_needs_logits_processing_ineligible(self, graphs):
        runner = _make_runner()
        runner.sampler.needs_logits_processing[0] = True
        batch = _make_input_batch()
        hidden_states = runner.cudagraph_manager.hidden_states
        assert graphs._step_eligible(runner, batch, hidden_states, None) is False

    def test_logprobs_requested_ineligible(self, graphs):
        runner = _make_runner()
        runner.sampler.sampling_states.max_num_logprobs = lambda idx: 5
        batch = _make_input_batch()
        hidden_states = runner.cudagraph_manager.hidden_states
        assert graphs._step_eligible(runner, batch, hidden_states, None) is False

    def test_prompt_logprobs_ineligible(self, graphs):
        runner = _make_runner()
        runner.prompt_logprobs_worker.uses_prompt_logprobs[1] = True
        batch = _make_input_batch()
        hidden_states = runner.cudagraph_manager.hidden_states
        assert graphs._step_eligible(runner, batch, hidden_states, None) is False

    def test_token_id_scores_ineligible(self, graphs):
        runner = _make_runner()
        runner.prompt_logprobs_worker.token_id_scores = {"req-0": object()}
        batch = _make_input_batch()
        hidden_states = runner.cudagraph_manager.hidden_states
        assert graphs._step_eligible(runner, batch, hidden_states, None) is False


class TestCreate:
    """On CPU, ``current_platform.is_cuda()`` is always False, so ``create()``
    always returns None here regardless of the other gates -- these tests
    exercise the gates that are checked (and can short-circuit) before that
    point, which is everything this module can verify without a GPU."""

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("VLLM_POST_FORWARD_CUDAGRAPH", raising=False)
        runner = _make_runner()
        assert PostForwardCudaGraphs.create(runner) is None

    def test_requires_mtp_speculator(self, monkeypatch):
        monkeypatch.setenv("VLLM_POST_FORWARD_CUDAGRAPH", "1")
        runner = _make_runner(speculator=object())
        assert PostForwardCudaGraphs.create(runner) is None

    def test_requires_cuda_platform(self, monkeypatch):
        monkeypatch.setenv("VLLM_POST_FORWARD_CUDAGRAPH", "1")
        runner = _make_runner()
        # CPU test process: current_platform.is_cuda() is False, so create()
        # takes the "a non-CUDA platform" unsupported branch and returns None.
        assert PostForwardCudaGraphs.create(runner) is None


class TestTensorKeyAndReadingCopies:
    def test_tensor_key_distinguishes_shape_stride_dtype_and_address(self):
        a = torch.zeros(4, dtype=torch.float32)
        b = torch.zeros(4, dtype=torch.float32)
        assert _tensor_key(a) != _tensor_key(b)  # different storage
        assert _tensor_key(a) == _tensor_key(a)
        c = torch.zeros(4, dtype=torch.int64)
        assert _tensor_key(a[:0]) != _tensor_key(c[:0])  # dtype differs

    def test_reading_copies_swaps_and_restores_gpu_view(self):
        original = torch.tensor([1, 2, 3])
        held = SimpleNamespace(gpu=original)
        copy = torch.tensor([9, 9, 9])
        with _reading_copies((held,), (copy,)):
            assert held.gpu is copy
        assert held.gpu is original
