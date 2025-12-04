# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._ops import OpOverload

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.models.utils import fast_topk
from vllm.platforms import current_platform

from .inductor_pass import enable_fake_mode
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


def empty_bf16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16, device="cuda")


def empty_fp32(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.float32, device="cuda")


def empty_i32(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.int32, device="cuda")


def empty_i64(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.int64, device="cuda")

class RocmAiterFusionPattern(ABC):
    """
    The base class for RocmAiterPatterns.

	Should not be used directly.
    """


    @staticmethod
    def wrap_trace_fn(trace_fn, *process_fx_fns: Callable[[fx.GraphModule], None]):
        def wrapped(*args, **kwargs):
            gm = trace_fn(*args, **kwargs)
            for process_fx in process_fx_fns:
                process_fx(gm)

            return gm

        return wrapped

    @abstractmethod
    def register(self, pm_pass: PatternMatcherPass):
        raise NotImplementedError


class RocmAiterTopkSigmoidPattern(RocmAiterFusionPattern):
    def __init__(
        self,
    ):
        super().__init__()

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(gating_output: torch.Tensor, topk: int):
			router_scores, router_indices = fast_topk(gating_output, topk=topk, dim=-1)
			router_scores = torch.sigmoid(router_scores.float())
            return (router_scores, router_indices.to(torch.int32))

        def replacement(
            gating_output: torch.Tensor, topk: int
        ):
            fused = auto_functionalized(
                torch.ops.vllm.rocm_aiter_topk_sigmoid,
                gating_output=gating_output,
				topk=topk,
            )

            # result, residual, scale
            return fused

		for dtype in (torch.float32, torch.float16, torch.bfloat16):
			pm.register_replacement(
				pattern,
				replacement,
				(torch.empty(2,2, dtype=dtype, device="cuda"), 2),
				pm.fwd_only,
				pm_pass,
			)
			pm.register_replacement(
				pattern,
				replacement,
				(torch.empty(2,2, dtype=dtype, device="cuda"), 1),
				pm.fwd_only,
				pm_pass,
			)


class RocmAiterFusionPass(VllmPatternMatcherPass):
    """TODO
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_fusion_pass"
        )

        RocmAiterTopkSigmoidPattern().register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph):
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> Any:
        return self.hash_source(
            self,
			RocmAiterTopkSigmoidPattern,
        )
