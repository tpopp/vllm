# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import operator
from typing import Any

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.model_executor.layers.quantization.utils.fp8_utils  # noqa: F401
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    kFp8Dynamic128Sym,
)
from vllm.platforms import current_platform

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .act_quant_fusion import ActivationQuantPattern
from .matcher_utils import (
    MatcherFusedAddRMSNorm,
    MatcherQuantFP8,
    MatcherSiluAndMul,
)
from .rms_quant_fusion import (
    FusedRMSQuantKey,
)

logger = init_logger(__name__)
FP8_DTYPE = current_platform.fp8_dtype()


class AiterRMSNormQuantPattern:
    def __init__(
        self, epsilon: float, key: FusedRMSQuantKey, match_aiter_quant: bool = True
    ):
        self.epsilon = epsilon
        self.quant_dtype = key.quant.dtype
        self.device = torch.device("cuda")

        if key.fused_add:
            self.rmsnorm_matcher = MatcherFusedAddRMSNorm(
                epsilon, match_rocm_aiter=True
            )
        self.quant_matcher = MatcherQuantFP8(
            key.quant,
            match_rocm_aiter=match_aiter_quant,
        )

    def empty(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=torch.bfloat16, device=self.device, **kwargs)

    def empty_f32(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=torch.float32, device=self.device, **kwargs)


class AiterRMSNormDynamicQuantPattern(AiterRMSNormQuantPattern):
    """AITER RMSNorm + Dynamic Quantization pattern."""

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_fused_dynamic_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        match_aiter_quant: bool = True,
        group_shape: GroupShape = GroupShape.PER_TOKEN,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms = torch.ops.vllm_ir.rms_norm(input, weight, self.epsilon)
            result, scale = self.quant_matcher(result_rms)
            return result, scale

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result = self.FUSED_OP(
                x=input,
                weight=weight,
                epsilon=self.epsilon,
                quant_dtype=self.quant_dtype,
            )

            return result[0], result[1]

        pm.register_replacement(
            pattern,
            replacement,
            # input, weight
            [self.empty(5, 16), self.empty(16)],
            pm.fwd_only,
            pm_pass,
        )


class AiterFusedAddRMSNormDynamicQuantPattern(AiterRMSNormQuantPattern):
    """AITER RMSNorm Fused Add + Dynamic Quantization pattern."""

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_fused_add_dynamic_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        match_aiter_quant: bool = True,
        group_shape: GroupShape = GroupShape.PER_TOKEN,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms, residual_out = self.rmsnorm_matcher(input, weight, residual)
            result, scale = self.quant_matcher(result_rms)

            return result, residual_out, scale

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result = self.FUSED_OP(
                x=input,
                residual=residual,
                weight=weight,
                epsilon=self.epsilon,
                quant_dtype=self.quant_dtype,
            )

            return result[0], result[1], result[2]

        pm.register_replacement(
            pattern,
            replacement,
            self.rmsnorm_matcher.inputs(),
            pm.fwd_only,
            pm_pass,
        )


class AiterRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    """
    This pattern fuses aiter rms_norm & group fp8 quant custom
    ops into an aiter rms_norm_group_fp8_quant op.
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        match_aiter_quant: bool = True,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms = torch.ops.vllm_ir.rms_norm(input, weight, self.epsilon)
            result, scale = self.quant_matcher(result_rms)
            return result, scale

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at = self.FUSED_OP(
                x=input,
                weight=weight,
                variance_epsilon=self.epsilon,
                group_size=128,
            )

            return at[0], at[1]

        pm.register_replacement(
            pattern,
            replacement,
            # input, weight
            [self.empty(5, 16), self.empty(16)],
            pm.fwd_only,
            pm_pass,
        )


class AiterFusedAddRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    """
    This pattern fuses aiter rms_norm_with_add & group fp8 quant custom ops
    into a aiter rms_norm_with_add_group_fp8_quant op.
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_group_add_fused_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        match_aiter_quant: bool = True,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms, residual_out = self.rmsnorm_matcher(input, weight, residual)
            result, scale = self.quant_matcher(result_rms)

            return result, residual_out, scale

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            at = self.FUSED_OP(
                x=input,
                residual=residual,
                weight=weight,
                variance_epsilon=self.epsilon,
                group_size=128,
            )

            # result, scale, residual
            return at[0], at[1], at[2]

        pm.register_replacement(
            pattern, replacement, self.rmsnorm_matcher.inputs(), pm.fwd_only, pm_pass
        )


class AiterGemmaRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    """
    This pattern fuses aiter gemma_rms_norm & group fp8 quant custom
    ops into an aiter rmsnorm_input_quant_fp8 op (with optional z gate).
    Matches: convert_element_type(weight, f32) -> add(w, 1.0) -> rms_norm -> quant
    Returns rms_norm result as extra output to handle multi-consumer case.
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_input_quant_fp8_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        match_aiter_quant: bool = True,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            w_fp32 = torch.ops.prims.convert_element_type(weight, torch.float32)
            w_plus_1 = w_fp32 + 1.0
            result_rms = torch.ops.vllm_ir.rms_norm(input, w_plus_1, self.epsilon)
            result, scale = self.quant_matcher(result_rms)
            return result, scale, result_rms

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            w_fp32 = torch.ops.prims.convert_element_type(weight, torch.float32)
            w_plus_1 = w_fp32 + 1.0
            result_rms = torch.ops.vllm_ir.rms_norm(input, w_plus_1, self.epsilon)
            at = self.FUSED_OP(
                x=input,
                weight=w_plus_1.to(input.dtype),
                bias=None,
                z=input,
                eps=self.epsilon,
                norm_before_gate=True,
                activation="silu",
                group_size=128,
            )

            return at[0], at[1], result_rms

        pm.register_replacement(
            pattern,
            replacement,
            [self.empty(4, 2048), self.empty(2048)],
            pm.fwd_only,
            pm_pass,
        )


class AiterFusedAddGemmaRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    """
    This pattern fuses aiter gemma_rms_norm_with_add & group fp8 quant custom
    ops into an aiter rmsnorm_input_quant_fp8 op (with optional z gate).
    Matches: convert_element_type(weight, f32) -> add(w, 1.0)
             -> input+residual -> rms_norm -> quant
    Returns rms_norm and residual as extra outputs for multi-consumer case.
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_input_quant_fp8_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        match_aiter_quant: bool = True,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            w_fp32 = torch.ops.prims.convert_element_type(weight, torch.float32)
            w_plus_1 = w_fp32 + 1.0
            residual_out = input + residual
            result_rms = torch.ops.vllm_ir.rms_norm(
                residual_out, w_plus_1, self.epsilon
            )
            result, scale = self.quant_matcher(result_rms)

            return result, scale, residual_out, result_rms

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            w_fp32 = torch.ops.prims.convert_element_type(weight, torch.float32)
            w_plus_1 = w_fp32 + 1.0
            residual_out = input + residual
            result_rms = torch.ops.vllm_ir.rms_norm(
                residual_out, w_plus_1, self.epsilon
            )
            at = self.FUSED_OP(
                x=residual_out,
                weight=w_plus_1.to(input.dtype),
                bias=None,
                z=residual_out,
                eps=self.epsilon,
                norm_before_gate=True,
                activation="silu",
                group_size=128,
            )

            return at[0], at[1], residual_out, result_rms

        pm.register_replacement(
            pattern,
            replacement,
            [self.empty(4, 2048), self.empty(2048), self.empty(4, 2048)],
            pm.fwd_only,
            pm_pass,
        )


class RMSNormGatedQuantManualFusion:
    """
    Manually matches the decomposed RMSNormGated(x, z, weight, eps,
    norm_before_gate=True) + reshape + rocm_aiter_group_fp8_quant pattern
    and replaces it with rocm_aiter_rmsnorm_input_quant_fp8.

    The pattern in the FX graph is:
      convert_element_type(x, fp32)                [x_fp32]
      pow(x_fp32, 2) -> mean(-1, True) -> add(eps) -> rsqrt
      mul(x_fp32, rsqrt)                           [x_normed]
      convert_element_type(weight, fp32)            [w_fp32]
      mul(x_normed, w_fp32)                         [weighted]
      convert_element_type(z, fp32)                 [z_fp32]
      sigmoid(z_fp32)
      mul(z_fp32, sigmoid)                          [silu_z]
      mul(weighted, silu_z)                         [gated]
      convert_element_type(gated, bf16)             [result_bf16]
      reshape(result_bf16, [num_tokens, hidden])    [reshaped]
      rocm_aiter_group_fp8_quant(reshaped, 128)     [quant, scale]
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_input_quant_fp8_op()
    QUANT_OP = torch.ops.vllm.rocm_aiter_group_fp8_quant.default

    @staticmethod
    def _is_op(node: fx.Node, target: Any) -> bool:
        return node.op == "call_function" and node.target == target

    def apply(self, graph: fx.Graph) -> int:
        count = 0
        nodes_to_remove: list[fx.Node] = []

        for node in graph.nodes:
            if not self._is_op(node, self.QUANT_OP):
                continue

            quant_input = node.args[0]
            if not isinstance(quant_input, fx.Node):
                continue

            # quant_input should be a reshape
            if not self._is_op(quant_input, torch.ops.aten.reshape.default):
                continue
            reshape_node = quant_input
            reshape_input = reshape_node.args[0]
            if not isinstance(reshape_input, fx.Node):
                continue

            # reshape_input should be convert_element_type to bf16
            if not self._is_op(
                reshape_input, torch.ops.prims.convert_element_type.default
            ):
                continue
            if reshape_input.args[1] != torch.bfloat16:
                continue
            bf16_cast = reshape_input

            # bf16_cast input should be mul (gated = weighted * silu_z)
            gated_mul = bf16_cast.args[0]
            if not isinstance(gated_mul, fx.Node):
                continue
            if not self._is_op(gated_mul, torch.ops.aten.mul.Tensor):
                continue

            # gated_mul args: (weighted, silu_z)
            weighted = gated_mul.args[0]
            silu_z = gated_mul.args[1]
            if not isinstance(weighted, fx.Node) or not isinstance(silu_z, fx.Node):
                continue

            # silu_z should be mul(z_fp32, sigmoid(z_fp32))
            if not self._is_op(silu_z, torch.ops.aten.mul.Tensor):
                continue
            z_fp32 = silu_z.args[0]
            sigmoid_node = silu_z.args[1]
            if not isinstance(z_fp32, fx.Node) or not isinstance(sigmoid_node, fx.Node):
                continue
            if not self._is_op(sigmoid_node, torch.ops.aten.sigmoid.default):
                continue
            if sigmoid_node.args[0] is not z_fp32:
                continue

            # z_fp32 should be convert_element_type(z, fp32)
            if not self._is_op(z_fp32, torch.ops.prims.convert_element_type.default):
                continue
            if z_fp32.args[1] != torch.float32:
                continue
            z_view = z_fp32.args[0]  # the view/reshape of z

            # weighted should be mul(x_normed, w_fp32)
            if not self._is_op(weighted, torch.ops.aten.mul.Tensor):
                continue
            x_normed = weighted.args[0]
            w_fp32 = weighted.args[1]
            if not isinstance(x_normed, fx.Node) or not isinstance(w_fp32, fx.Node):
                continue

            # w_fp32 should be convert_element_type(weight, fp32)
            if not self._is_op(w_fp32, torch.ops.prims.convert_element_type.default):
                continue
            if w_fp32.args[1] != torch.float32:
                continue
            weight_node = w_fp32.args[0]  # the original weight

            # x_normed should be mul(x_fp32, rsqrt_node)
            if not self._is_op(x_normed, torch.ops.aten.mul.Tensor):
                continue
            x_fp32 = x_normed.args[0]
            rsqrt_node = x_normed.args[1]
            if not isinstance(x_fp32, fx.Node) or not isinstance(rsqrt_node, fx.Node):
                continue

            # Verify the RMS path: rsqrt(mean(pow(x_fp32, 2), -1, True) + eps)
            if not self._is_op(rsqrt_node, torch.ops.aten.rsqrt.default):
                continue
            add_eps = rsqrt_node.args[0]
            if not isinstance(add_eps, fx.Node):
                continue
            if not self._is_op(add_eps, torch.ops.aten.add.Tensor):
                continue
            mean_node = add_eps.args[0]
            eps_val = add_eps.args[1]
            if not isinstance(mean_node, fx.Node):
                continue
            if not self._is_op(mean_node, torch.ops.aten.mean.dim):
                continue
            pow_node = mean_node.args[0]
            if not isinstance(pow_node, fx.Node):
                continue
            if not self._is_op(pow_node, torch.ops.aten.pow.Tensor_Scalar):
                continue
            if pow_node.args[0] is not x_fp32:
                continue

            # x_fp32 should be convert_element_type(x_view, fp32)
            if not self._is_op(x_fp32, torch.ops.prims.convert_element_type.default):
                continue
            if x_fp32.args[1] != torch.float32:
                continue
            x_view = x_fp32.args[0]  # the view/reshape of x

            # Check that intermediate nodes don't have other users
            # (except x_fp32 which feeds both pow and mul)
            if len(bf16_cast.users) != 1:  # only reshape
                continue
            if len(reshape_node.users) != 1:  # only quant
                continue

            logger.info(
                "RMSNormGatedQuantManualFusion: matched pattern at %s",
                node.name,
            )

            # Build the replacement:
            # 1. Call rmsnorm_input_quant_fp8(x, weight, bias=None, z, eps, ...)
            # 2. Reshape outputs to match expected shapes
            reshape_size = reshape_node.args[1]  # [num_tokens, hidden_size]

            with graph.inserting_before(node):
                fused = graph.call_function(
                    self.FUSED_OP,
                    kwargs={
                        "x": x_view,
                        "weight": weight_node,
                        "bias": None,
                        "z": z_view,
                        "eps": eps_val,
                        "norm_before_gate": True,
                        "activation": "silu",
                        "group_size": 128,
                    },
                )
                quant_out = graph.call_function(operator.getitem, (fused, 0))
                scale_out = graph.call_function(operator.getitem, (fused, 1))
                quant_reshaped = graph.call_function(
                    torch.ops.aten.reshape.default,
                    (quant_out, reshape_size),
                )
                # Scale: from [n, 1] to [num_tokens, num_heads]
                scale_reshaped = graph.call_function(
                    torch.ops.aten.reshape.default,
                    (scale_out, [reshape_size[0], -1]),
                )

            # Replace uses of the quant node's getitem users
            for user in list(node.users):
                if self._is_op(user, operator.getitem):
                    idx = user.args[1]
                    if idx == 0:
                        user.replace_all_uses_with(quant_reshaped)
                    elif idx == 1:
                        user.replace_all_uses_with(scale_reshaped)
                    nodes_to_remove.append(user)

            nodes_to_remove.extend(
                [
                    node,
                    reshape_node,
                    bf16_cast,
                    gated_mul,
                    silu_z,
                    sigmoid_node,
                    weighted,
                    x_normed,
                    rsqrt_node,
                    add_eps,
                    mean_node,
                    pow_node,
                ]
            )
            # Only remove z_fp32, w_fp32, x_fp32 if they have no other users
            for n in [z_fp32, w_fp32, x_fp32]:
                if all(u in nodes_to_remove or u is fused for u in n.users):
                    nodes_to_remove.append(n)

            count += 1

        for n in reversed(nodes_to_remove):
            if len(n.users) == 0:
                graph.erase_node(n)

        return count


class RocmAiterRMSNormQuantFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses aiter rms_norm & vllm/aiter quant custom ops
    into a fused rms_norm_quant op.
    It also supports fused_add_rms_norm.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_rms_norm_quant_fusion_pass"
        )

        # Make sure fused add patterns are before simple rms norm,
        # as the latter is a subset of the former in torch ops
        for epsilon in [1e-5, 1e-6]:
            # Fuse gemma_rms_norm + aiter dynamic group fp8 quant
            AiterGemmaRMSFp8GroupQuantPattern(
                epsilon, FP8_DTYPE, GroupShape(1, 128)
            ).register(self.patterns)

            # Fuse gemma_rms_norm_with_add + aiter dynamic group fp8 quant
            AiterFusedAddGemmaRMSFp8GroupQuantPattern(
                epsilon, FP8_DTYPE, GroupShape(1, 128)
            ).register(self.patterns)

            #  Fuse aiter rms_norm + aiter dynamic group fp8 quant
            AiterRMSFp8GroupQuantPattern(
                epsilon, FP8_DTYPE, GroupShape(1, 128)
            ).register(self.patterns)

            # Fuse aiter fused_add_rms_norm + aiter dynamic group fp8 quant
            AiterFusedAddRMSFp8GroupQuantPattern(
                epsilon, FP8_DTYPE, GroupShape(1, 128)
            ).register(self.patterns)

            for match_aiter_quant in [True, False]:
                # Fuse aiter rms_norm + (aiter / vllm built-in)
                # dynamic per-token fp8 quant
                AiterRMSNormDynamicQuantPattern(
                    epsilon, FP8_DTYPE, match_aiter_quant=match_aiter_quant
                ).register(self.patterns)

                # Fuse aiter fused_add_rms_norm + (aiter / vllm built-in)
                # dynamic per-token fp8 quant
                AiterFusedAddRMSNormDynamicQuantPattern(
                    epsilon, FP8_DTYPE, match_aiter_quant=match_aiter_quant
                ).register(self.patterns)

        self._gated_fusion = RMSNormGatedQuantManualFusion()

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        gated_count = self._gated_fusion.apply(graph)
        if gated_count > 0:
            logger.info(
                "%s RMSNormGatedQuant manual fusion replaced %s patterns",
                self.__class__.__name__,
                gated_count,
            )
        self.matched_count = self.patterns.apply(graph) + gated_count
        logger.debug(
            "%s Replaced %s patterns", self.__class__.__name__, self.matched_count
        )

    def uuid(self) -> str:
        fusion_patterns = [
            AiterRMSNormDynamicQuantPattern,
            AiterFusedAddRMSNormDynamicQuantPattern,
            AiterRMSFp8GroupQuantPattern,
            AiterFusedAddRMSFp8GroupQuantPattern,
            AiterGemmaRMSFp8GroupQuantPattern,
            AiterFusedAddGemmaRMSFp8GroupQuantPattern,
            RMSNormGatedQuantManualFusion,
        ]
        return self.hash_source(self, *fusion_patterns)


class AiterSiluMulFp8GroupQuantPattern(ActivationQuantPattern):
    """
    This pattern fuses aiter silu_and_mul & group fp8 quant custom
    ops into an aiter silu_and_mul_group_fp8_quant op.
    """

    FUSED_SILU_MUL_QUANT_OP = rocm_aiter_ops.get_act_mul_fused_fp8_group_quant_op()

    def __init__(self) -> None:
        self.silu_and_mul_matcher = MatcherSiluAndMul()
        self.quant_matcher = MatcherQuantFP8(
            quant_key=kFp8Dynamic128Sym, match_rocm_aiter=True
        )

    def get_inputs(self) -> list[torch.Tensor]:
        return [
            self.silu_and_mul_matcher.inputs()[0],
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at1 = self.silu_and_mul_matcher(input)
            at2 = self.quant_matcher(at1)
            return at2[0], at2[1]

        def replacement(
            input: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at = self.FUSED_SILU_MUL_QUANT_OP(x=input, group_size=128)
            return at[0], at[1]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class RocmAiterSiluMulFp8GroupQuantFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses a pre-defined set of custom ops into fused ops.
    It uses the torch pattern matcher to find the patterns and replace them.

    Because patterns can only be registered once, the pass is a singleton.
    This will be addressed in a future version of PyTorch:
    https://github.com/pytorch/pytorch/pull/139321#issuecomment-2452354980
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_silu_mul_fp8_group_quant_fusion_pass"
        )

        AiterSiluMulFp8GroupQuantPattern().register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        fusion_patterns = [
            ActivationQuantPattern,
            AiterSiluMulFp8GroupQuantPattern,
        ]
        return VllmInductorPass.hash_source(self, *fusion_patterns)


class AddAiterRMSNormPadPattern:
    """
    This pattern replaces an aiter_rmsnorm_with_add & a pad op
    with a custom triton_add_rmsnorm_pad op from AITER.
    """

    AITER_TRITON_ADD_RMSNORM_PAD_OP = rocm_aiter_ops.get_triton_add_rmsnorm_pad_op()

    def __init__(
        self,
        epsilon: float,
        hidden_size: int,
        x_pad_to_multiple: int,
    ):
        self.epsilon = epsilon
        self.hidden_size = hidden_size
        self.x_pad_to_multiple = x_pad_to_multiple
        self.rmsnorm_matcher = MatcherFusedAddRMSNorm(epsilon, match_rocm_aiter=True)

    def get_inputs(self) -> list[torch.Tensor]:
        input, weight, residual = self.rmsnorm_matcher.inputs()
        router_weight = torch.empty([8, 16], dtype=weight.dtype, device=weight.device)
        router_bias = torch.empty([8], dtype=weight.dtype, device=weight.device)
        return [input, weight, residual, router_weight, router_bias]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            router_weight: torch.Tensor,
            router_bias: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            pad_size = self.x_pad_to_multiple - (
                self.hidden_size % self.x_pad_to_multiple
            )
            result_rms, residual_out = self.rmsnorm_matcher(input, weight, residual)
            router_logits = torch.ops.vllm.rocm_unquantized_gemm(
                result_rms, router_weight, router_bias
            )
            result = torch.nn.functional.pad(
                result_rms, (0, pad_size), mode="constant", value=0.0
            )
            return result, residual_out, router_logits

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            router_weight: torch.Tensor,
            router_bias: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            at = self.AITER_TRITON_ADD_RMSNORM_PAD_OP(
                x=input,
                weight=weight,
                variance_epsilon=self.epsilon,
                residual=residual,
                x_pad_to_multiple=self.x_pad_to_multiple,
            )
            result_padded = at[0]
            router_logits = torch.ops.vllm.rocm_unquantized_gemm(
                result_padded[:, : self.hidden_size], router_weight, router_bias
            )
            residual_out = at[1]
            return result_padded, residual_out, router_logits

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class RocmAiterTritonAddRMSNormPadFusionPass(VllmPatternMatcherPass):
    """
    This pass replaces an AITER CK RMSNorm + residual add and a pad op
    with an triton_add_rmsnorm_pad op from AITER.
    """

    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_triton_add_rmsnorm_pad_fusion_pass"
        )

        # gpt-oss has hidden size 2880
        # padded to a multiple of 128 on gfx942 and 256 on gfx950 respectively
        hidden_size = 2880
        for epsilon in [1e-5, 1e-6]:
            for x_pad_to_multiple in [128, 256]:
                AddAiterRMSNormPadPattern(
                    epsilon, hidden_size, x_pad_to_multiple
                ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, AddAiterRMSNormPadPattern)
