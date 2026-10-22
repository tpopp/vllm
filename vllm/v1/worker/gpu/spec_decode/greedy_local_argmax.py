# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy spec-decode verification without all-gathering the target logits.

Enabled by ``SpeculativeConfig.use_local_argmax_reduction``. For an all-greedy
verify batch with vocab-parallel target logits (TP > 1) the rejection sampler
only needs the target argmax of every logits row. Each TP rank reduces its
vocab shard to per-chunk ``(max, global id)`` partials, only those partials
are all-gathered (~0.1 MB instead of ``rows * vocab`` logits), and one kernel
per request reduces them and runs the greedy accept/reject loop of
``_rejection_kernel``.

With synthetic acceptance rates (``rejection_sample_method="synthetic"``) the
accept decision uses the same per-position uniform draws as
``_rejection_kernel`` instead of comparing with the target argmax; rejected and
bonus positions still take the target argmax from the partials.

Accepted tokens are bit-identical to the stock greedy path: both take the
first (smallest id) maximum of the fp32-cast logits, and the bonus/recovery
token is the target argmax in both.
"""

import numpy as np
import torch
import torch.nn as nn

from vllm.config import SpeculativeConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    gather_draft_sampled,
)

logger = init_logger(__name__)

_CHUNK = 2048
# Token ids travel as fp32 next to the values.
_MAX_EXACT_INDEX = 1 << 24


@triton.jit
def _chunk_argmax_kernel(
    logits_ptr,
    partial_ptr,
    stride_row,
    n_valid,
    vocab_start,
    num_chunks,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(
        logits_ptr + row * stride_row + offs,
        mask=offs < n_valid,
        other=float("-inf"),
    ).to(tl.float32)
    x = tl.where(x == x, x, float("-inf"))
    best, idx = tl.max(
        x, axis=0, return_indices=True, return_indices_tie_break_left=True
    )
    out = partial_ptr + (row * num_chunks + chunk) * 2
    tl.store(out, best)
    tl.store(out + 1, (vocab_start + chunk * CHUNK + idx).to(tl.float32))


@triton.jit
def _row_argmax(
    partial_ptr,
    row,
    num_rows,
    num_chunks,
    num_candidates,
    BLOCK: tl.constexpr,
):
    # partials are rank-major: [tp_size * num_rows, num_chunks, 2].
    k = tl.arange(0, BLOCK)
    mask = k < num_candidates
    rank = k // num_chunks
    base = ((rank * num_rows + row).to(tl.int64) * num_chunks + k % num_chunks) * 2
    vals = tl.load(partial_ptr + base, mask=mask, other=float("-inf"))
    ids = tl.load(partial_ptr + base + 1, mask=mask, other=3.0e38)
    best = tl.max(vals, axis=0)
    return tl.min(tl.where(vals == best, ids, 3.0e38), axis=0).to(tl.int64)


@triton.jit
def _greedy_verify_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [tp_size * num_logits, num_chunks, 2]
    partial_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    # [num_speculative_steps]
    synthetic_conditional_rates_ptr,
    num_logits,
    num_chunks,
    num_candidates,
    BLOCK: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    if SYNTHETIC_MODE:
        req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
        seed = tl.load(seed_ptr + req_state_idx)

    accepted_length = tl.zeros((), tl.int32)
    verifying = True
    for i in range(num_draft_tokens + 1):
        if verifying:
            target_argmax = _row_argmax(
                partial_ptr,
                start_idx + i,
                num_logits,
                num_chunks,
                num_candidates,
                BLOCK,
            )
            if i < num_draft_tokens:
                draft = tl.load(draft_sampled_ptr + start_idx + i + 1).to(tl.int64)
                if SYNTHETIC_MODE:
                    # Same draw as _rejection_kernel's synthetic greedy branch.
                    pos = tl.load(pos_ptr + start_idx + i)
                    u = tl_rand32(seed, pos, includes_zero=False)
                    accepted = u < tl.load(synthetic_conditional_rates_ptr + i)
                else:
                    accepted = draft == target_argmax
                # -1 placeholders are rejected, as in _rejection_kernel.
                accepted &= draft >= 0
                verifying = accepted
                accepted_length += accepted.to(tl.int32)
                token = tl.where(accepted, draft, target_argmax)
            else:
                # Bonus token.
                token = target_argmax
            tl.store(sampled_ptr + req_idx * sampled_stride + i, token)
    tl.store(num_sampled_ptr + req_idx, accepted_length + 1)


def local_argmax_partials(
    logits: torch.Tensor, n_valid: int, vocab_start: int
) -> torch.Tensor:
    """Per-chunk ``(max, global id)`` of ``[rows, local_vocab]`` shard logits.

    Columns ``>= n_valid`` are vocab padding and never win; NaN never wins.
    """
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    num_rows = logits.shape[0]
    num_chunks = triton.cdiv(logits.shape[-1], _CHUNK)
    partials = torch.empty(
        (num_rows, num_chunks, 2), dtype=torch.float32, device=logits.device
    )
    if num_rows > 0:
        _chunk_argmax_kernel[(num_rows, num_chunks)](
            logits,
            partials,
            logits.stride(0),
            n_valid,
            vocab_start,
            num_chunks,
            CHUNK=_CHUNK,
            num_warps=4,
        )
    return partials


def greedy_verify_from_partials(
    partials: torch.Tensor,
    tp_size: int,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
    synthetic_conditional_rates: torch.Tensor | None = None,
    idx_mapping: torch.Tensor | None = None,
    seed: torch.Tensor | None = None,
    pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy rejection from rank-major ``[tp_size * num_logits, chunks, 2]``.

    Returns ``(sampled, num_sampled)`` like ``rejection_sample`` for
    all-greedy requests (num_sampled includes the recovery/bonus token).
    ``synthetic_conditional_rates`` needs ``idx_mapping``, ``seed`` and ``pos``.
    """
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits = partials.shape[0] // tp_size
    num_chunks = partials.shape[1]
    num_candidates = tp_size * num_chunks
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    synthetic = synthetic_conditional_rates is not None
    if synthetic:
        assert idx_mapping is not None and seed is not None and pos is not None
    if num_reqs > 0:
        _greedy_verify_kernel[(num_reqs,)](
            sampled,
            sampled.stride(0),
            num_sampled,
            partials,
            draft_sampled,
            cu_num_logits,
            idx_mapping,
            seed,
            pos,
            synthetic_conditional_rates,
            num_logits,
            num_chunks,
            num_candidates,
            BLOCK=triton.next_power_of_2(num_candidates),
            SYNTHETIC_MODE=synthetic,
            num_warps=1,
        )
    return sampled, num_sampled


def _find_head_owner(model: nn.Module) -> nn.Module | None:
    """The module whose compute_logits is ``logits_processor(lm_head, h)``.

    Multimodal wrappers delegate compute_logits to their language model.
    """
    module: nn.Module | None = model
    for _ in range(3):
        if module is None:
            return None
        if isinstance(
            getattr(module, "lm_head", None), VocabParallelEmbedding
        ) and isinstance(getattr(module, "logits_processor", None), LogitsProcessor):
            return module
        inner = getattr(module, "language_model", None)
        if inner is None and hasattr(module, "get_language_model"):
            try:
                inner = module.get_language_model()
            except Exception:
                inner = None
        module = inner if inner is not module else None
    return None


class GreedyLocalArgmaxVerifier:
    """Verifies all-greedy spec-decode batches from vocab-parallel argmaxes."""

    def __init__(
        self,
        head_owner: nn.Module,
        rejection_sampler: RejectionSampler,
    ):
        self.rejection_sampler = rejection_sampler
        self.sampler = rejection_sampler.sampler
        self.lm_head: VocabParallelEmbedding = head_owner.lm_head
        self.logits_processor: LogitsProcessor = head_owner.logits_processor
        # Width of the stock compute_logits() output.
        self.vocab_size = self.logits_processor.org_vocab_size
        shard = self.lm_head.shard_indices
        self.n_valid = shard.num_org_elements
        self.vocab_start = shard.org_vocab_start_index
        self.tp_size = self.lm_head.tp_size

    @classmethod
    def maybe_create(
        cls,
        model: nn.Module,
        rejection_sampler: RejectionSampler | None,
        speculative_config: SpeculativeConfig | None,
    ) -> "GreedyLocalArgmaxVerifier | None":
        if (
            speculative_config is None
            or not speculative_config.use_local_argmax_reduction
            or rejection_sampler is None
        ):
            return None
        head_owner = _find_head_owner(model)
        if head_owner is None:
            reason: str | None = "no vocab-parallel lm_head/logits_processor found"
        else:
            reason = cls._unsupported_reason(head_owner, rejection_sampler)
        if reason is not None:
            logger.info_once(
                "Greedy spec-decode local argmax verification disabled: %s.", reason
            )
            return None
        logger.info_once(
            "Using vocab-parallel local argmax for all-greedy spec-decode "
            "verification: target logits are not all-gathered."
        )
        assert head_owner is not None
        return cls(head_owner, rejection_sampler)

    @staticmethod
    def _unsupported_reason(
        head_owner: nn.Module, rejection_sampler: RejectionSampler
    ) -> str | None:
        sampler = rejection_sampler.sampler
        if type(rejection_sampler) is not RejectionSampler:
            return "custom rejection sampler"
        if type(sampler) is not Sampler:
            return "custom sampler"
        if (
            rejection_sampler.watermark_key is not None
            or rejection_sampler.enable_adaptive_verification
            or rejection_sampler.use_block_verification
        ):
            return "watermarking, adaptive or block verification"
        if sampler.compute_nans or sampler.return_sampling_mask:
            return "NaN counting or sampling masks requested"
        lm_head = head_owner.lm_head
        logits_processor = head_owner.logits_processor
        if lm_head.tp_size == 1:
            return "target logits are not vocab-parallel"
        if lm_head.parallel_group is not None:
            return "lm_head is not sharded over the TP group"
        shard = getattr(lm_head, "shard_indices", None)
        if (
            shard is None
            or shard.num_added_elements != 0
            or logits_processor.logits_as_input
        ):
            return "added vocab or logits-as-input"
        org_vocab = logits_processor.org_vocab_size
        if org_vocab != lm_head.org_vocab_size:
            return "lm_head vocab differs from the logits vocab"
        if org_vocab > _MAX_EXACT_INDEX:
            return "vocab too large"
        return None

    def can_verify(
        self, input_batch: InputBatch, draft_logits: torch.Tensor | None
    ) -> bool:
        """Host-side check; identical on every TP rank."""
        if input_batch.num_reqs == 0 or input_batch.num_draft_tokens == 0:
            return False
        if draft_logits is not None and draft_logits.shape[-1] < self.vocab_size:
            return False
        idx = input_batch.idx_mapping_np
        states = self.sampler.sampling_states
        return (
            bool(np.all(states.temperature.np[idx] == 0.0))
            and not bool(np.any(self.sampler.needs_logits_processing[idx]))
            and states.max_num_logprobs(idx) == NO_LOGPROBS
        )

    def __call__(
        self, sample_hidden_states: torch.Tensor, input_batch: InputBatch
    ) -> SamplerOutput:
        # Same as compute_logits() minus the gather and the padding slice.
        local_logits = self.logits_processor(
            self.lm_head, sample_hidden_states, skip_gather=True
        )
        partials = local_argmax_partials(local_logits, self.n_valid, self.vocab_start)
        partials = tensor_model_parallel_all_gather(partials, dim=0)

        draft_sampled, pos = gather_draft_sampled(
            input_batch.input_ids,
            input_batch.positions,
            input_batch.logits_indices,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.sampler.req_states.prefill_len.gpu,
        )
        sampled, num_sampled = greedy_verify_from_partials(
            partials,
            self.tp_size,
            draft_sampled,
            input_batch.cu_num_logits,
            self.rejection_sampler.num_speculative_steps,
            self.rejection_sampler.synthetic_conditional_rates,
            input_batch.idx_mapping,
            self.sampler.sampling_states.seeds.gpu,
            pos,
        )
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
