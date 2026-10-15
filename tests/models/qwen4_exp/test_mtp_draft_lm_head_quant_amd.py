# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4ExpMTP draft lm_head quantization wiring (ROCm MRV2).

``speculative_config.draft_lm_head_quantization`` (upstream #59973,
784d1fff0c) swaps a sharing drafter's ``lm_head`` for a quantized copy inside
``load_eagle_model``, and relies on every logits/top-token path going through
``lm_head.quant_method.apply`` rather than a raw ``F.linear(lm_head.weight)``.

``Qwen4ExpMTP`` does not set ``has_own_lm_head``, so MRV2's ``_should_share``
shares the target head by default (and would install a quantized copy once
``draft_lm_head_quantization`` is set); ``compute_logits`` and
``get_top_tokens`` (added by "Add get_top_tokens to the AMD MTP draft") must
honor whatever ``lm_head`` object that sharing installs for the quantized
copy to actually be used.

These tests run entirely on CPU: they stub ``lm_head.quant_method.apply``
instead of running a real quantized GEMM, so no GPU/AITER kernel is needed.
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.models.qwen4_exp.amd.mtp import Qwen4ExpMTP
from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share


def test_qwen4exp_mtp_does_not_declare_has_own_lm_head():
    """MRV2's ``_should_share`` shares unconditionally unless the drafter
    opts out via ``has_own_lm_head = True``
    (``vllm/model_executor/models/interfaces.py``). ``Qwen4ExpMTP`` must NOT
    opt out, or ``load_eagle_model`` would stop installing (quantized or
    plain) the target lm_head onto it, and ``draft_lm_head_quantization``
    would then hit the "needs a drafter that shares" ``ValueError`` in
    ``eagle/utils.py``.
    """
    assert getattr(Qwen4ExpMTP, "has_own_lm_head", False) is False
    assert _should_share(Qwen4ExpMTP, "has_own_lm_head", object(), object())


class _StubShardIndices:
    num_org_vocab_padding = 0
    org_vocab_start_index = 0


class _StubLMHead:
    """Mimics the parts of ``ParallelLMHead`` that ``LogitsProcessor``
    touches, standing in for a real (possibly quantized) lm_head copy."""

    def __init__(self, apply_return: torch.Tensor):
        self.quant_method = MagicMock()
        self.quant_method.apply.return_value = apply_return
        self.tp_size = 1
        self.shard_indices = _StubShardIndices()


def _bare_draft(default_vllm_config, vocab: int, lm_head: _StubLMHead) -> Qwen4ExpMTP:
    """A ``Qwen4ExpMTP`` instance with only the attributes ``compute_logits``
    / ``get_top_tokens`` touch, skipping the full backbone construction."""
    draft = object.__new__(Qwen4ExpMTP)
    torch.nn.Module.__init__(draft)
    draft.logits_processor = LogitsProcessor(vocab)
    draft.lm_head = lm_head
    return draft


@pytest.mark.parametrize("method_name", ["compute_logits", "get_top_tokens"])
def test_qwen4exp_mtp_logit_paths_use_lm_head_quant_method(
    default_vllm_config, method_name
):
    """Both ``compute_logits`` and ``get_top_tokens`` must route through
    ``self.lm_head.quant_method.apply(self.lm_head, ...)``: that is the hook
    a ``quantized_lm_head_copy`` relies on to actually run the quantized
    GEMM instead of being silently bypassed."""
    vocab = 32
    hidden = torch.randn(2, 8)
    logits = torch.randn(2, vocab)
    lm_head = _StubLMHead(logits)
    draft = _bare_draft(default_vllm_config, vocab, lm_head)

    getattr(draft, method_name)(hidden)

    lm_head.quant_method.apply.assert_called_once()
    called_lm_head = lm_head.quant_method.apply.call_args.args[0]
    assert called_lm_head is lm_head
