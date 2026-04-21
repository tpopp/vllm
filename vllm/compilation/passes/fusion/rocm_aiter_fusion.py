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
    norm_before_gate=True) + group fp8 quant pattern and replaces it
    with rocm_aiter_rmsnorm_input_quant_fp8.

    Supports two quant representations:
      1. rocm_aiter_group_fp8_quant custom op (when +quant_fp8)
      2. Fully decomposed group quant aten ops (when -quant_fp8)
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_input_quant_fp8_op()

    @staticmethod
    def _is_op(node: fx.Node, target: Any) -> bool:
        return node.op == "call_function" and node.target == target

    def _match_gated_norm(
        self, bf16_cast: fx.Node
    ) -> tuple[fx.Node, fx.Node, fx.Node, Any, list[fx.Node], list[fx.Node]] | None:
        """Walk backwards from bf16_cast to match decomposed RMSNormGated.
        Returns (x_view, z_view, weight_node, eps_val, nodes_to_remove,
        conditionally_removable_nodes) or None if no match.
        """
        gated_mul = bf16_cast.args[0]
        if not isinstance(gated_mul, fx.Node):
            return None
        if not self._is_op(gated_mul, torch.ops.aten.mul.Tensor):
            return None

        weighted = gated_mul.args[0]
        silu_z = gated_mul.args[1]
        if not isinstance(weighted, fx.Node) or not isinstance(silu_z, fx.Node):
            return None

        if not self._is_op(silu_z, torch.ops.aten.mul.Tensor):
            return None
        z_fp32 = silu_z.args[0]
        sigmoid_node = silu_z.args[1]
        if not isinstance(z_fp32, fx.Node) or not isinstance(sigmoid_node, fx.Node):
            return None
        if not self._is_op(sigmoid_node, torch.ops.aten.sigmoid.default):
            return None
        if sigmoid_node.args[0] is not z_fp32:
            return None

        if not self._is_op(z_fp32, torch.ops.prims.convert_element_type.default):
            return None
        if z_fp32.args[1] != torch.float32:
            return None
        z_view = z_fp32.args[0]

        if not self._is_op(weighted, torch.ops.aten.mul.Tensor):
            return None
        x_normed = weighted.args[0]
        w_fp32 = weighted.args[1]
        if not isinstance(x_normed, fx.Node) or not isinstance(w_fp32, fx.Node):
            return None

        if not self._is_op(w_fp32, torch.ops.prims.convert_element_type.default):
            return None
        if w_fp32.args[1] != torch.float32:
            return None
        weight_node = w_fp32.args[0]

        if not self._is_op(x_normed, torch.ops.aten.mul.Tensor):
            return None
        x_fp32 = x_normed.args[0]
        rsqrt_node = x_normed.args[1]
        if not isinstance(x_fp32, fx.Node) or not isinstance(rsqrt_node, fx.Node):
            return None

        if not self._is_op(rsqrt_node, torch.ops.aten.rsqrt.default):
            return None
        add_eps = rsqrt_node.args[0]
        if not isinstance(add_eps, fx.Node):
            return None
        if not self._is_op(add_eps, torch.ops.aten.add.Tensor):
            return None
        mean_node = add_eps.args[0]
        eps_val = add_eps.args[1]
        if not isinstance(mean_node, fx.Node):
            return None
        if not self._is_op(mean_node, torch.ops.aten.mean.dim):
            return None
        pow_node = mean_node.args[0]
        if not isinstance(pow_node, fx.Node):
            return None
        if not self._is_op(pow_node, torch.ops.aten.pow.Tensor_Scalar):
            return None
        if pow_node.args[0] is not x_fp32:
            return None

        if not self._is_op(x_fp32, torch.ops.prims.convert_element_type.default):
            return None
        if x_fp32.args[1] != torch.float32:
            return None
        x_view = x_fp32.args[0]

        nodes = [
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
        conditionally_remove = [z_fp32, w_fp32, x_fp32]
        return (x_view, z_view, weight_node, eps_val, nodes, conditionally_remove)

    def _match_custom_op_quant(
        self, bf16_cast: fx.Node
    ) -> (
        tuple[
            fx.Node,
            list[fx.Node],
            list[fx.Node],
            list,
        ]
        | None
    ):
        """Match bf16_cast → reshape → rocm_aiter_group_fp8_quant.
        Returns (insert_before_node, quant_nodes_to_remove,
                 getitem_nodes_to_remove, reshape_size) or None.
        """
        if len(bf16_cast.users) != 1:
            return None
        reshape_node = next(iter(bf16_cast.users))
        if not self._is_op(reshape_node, torch.ops.aten.reshape.default):
            return None
        if len(reshape_node.users) != 1:
            return None
        quant_node = next(iter(reshape_node.users))
        if not self._is_op(
            quant_node,
            torch.ops.vllm.rocm_aiter_group_fp8_quant.default,
        ):
            return None

        reshape_size = reshape_node.args[1]

        getitem_nodes = []
        for user in list(quant_node.users):
            if self._is_op(user, operator.getitem):
                getitem_nodes.append(user)

        return (
            quant_node,
            [quant_node, reshape_node],
            getitem_nodes,
            reshape_size,
        )

    def _match_decomposed_quant(
        self, bf16_cast: fx.Node
    ) -> (
        tuple[
            fx.Node,
            list[fx.Node],
            dict[int, fx.Node],
            list,
            Any,
        ]
        | None
    ):
        """Match bf16_cast → decomposed group quant chain.
        Pattern: reshape(bf16, [-1, ng, gs]) → abs → max → getitem[0] →
        f32 → div(448) → clamp_min(min_scale) [scales_3d]
        → div(grouped, scales) → clamp(-448, 448) → f8e4m3fn →
        reshape(-1, hidden) [quant_flat]
        → squeeze(scales_3d, -1) [scales_squeezed]

        Returns (insert_before_node, quant_nodes_to_remove,
                 output_map {0: quant_flat, 1: scales_squeezed},
                 reshape_size_for_quant, num_groups) or None.
        """
        if len(bf16_cast.users) != 1:
            return None
        group_reshape = next(iter(bf16_cast.users))
        if not self._is_op(group_reshape, torch.ops.aten.reshape.default):
            return None

        grouped = group_reshape  # [n, ng, gs]

        # grouped should have 2 users: abs (for absmax) and div (x/scales)
        if len(grouped.users) != 2:
            return None

        abs_node = None
        x_div_scales = None
        for u in grouped.users:
            if self._is_op(u, torch.ops.aten.abs.default):
                abs_node = u
            elif self._is_op(u, torch.ops.aten.div.Tensor):
                x_div_scales = u
        if abs_node is None or x_div_scales is None:
            return None

        # abs → max(-1, True) → getitem[0] → f32 → div(448) → clamp_min
        max_node = None
        for u in abs_node.users:
            if self._is_op(u, torch.ops.aten.max.dim):
                max_node = u
        if max_node is None:
            return None

        getitem_max = None
        for u in max_node.users:
            if self._is_op(u, operator.getitem) and u.args[1] == 0:
                getitem_max = u
        if getitem_max is None:
            return None

        if len(getitem_max.users) != 1:
            return None
        absmax_f32 = next(iter(getitem_max.users))
        if not self._is_op(absmax_f32, torch.ops.prims.convert_element_type.default):
            return None
        if absmax_f32.args[1] != torch.float32:
            return None

        if len(absmax_f32.users) != 1:
            return None
        scales_raw_div = next(iter(absmax_f32.users))
        if not self._is_op(scales_raw_div, torch.ops.aten.div.Tensor):
            return None

        if len(scales_raw_div.users) != 1:
            return None
        scales_3d = next(iter(scales_raw_div.users))
        if not self._is_op(scales_3d, torch.ops.aten.clamp_min.default):
            return None

        # scales_3d should have 2 users: x_div_scales and squeeze
        if len(scales_3d.users) != 2:
            return None
        if x_div_scales.args[1] is not scales_3d:
            return None

        squeeze_node = None
        for u in scales_3d.users:
            if self._is_op(u, torch.ops.aten.squeeze.dim):
                squeeze_node = u
        if squeeze_node is None:
            return None

        # x_div_scales → clamp_min(-448) → clamp_max(448) → f8e4m3fn →
        # reshape(-1, hidden)
        if len(x_div_scales.users) != 1:
            return None
        clamp_lo = next(iter(x_div_scales.users))
        if not self._is_op(clamp_lo, torch.ops.aten.clamp_min.default):
            return None

        if len(clamp_lo.users) != 1:
            return None
        clamp_hi = next(iter(clamp_lo.users))
        if not self._is_op(clamp_hi, torch.ops.aten.clamp_max.default):
            return None

        if len(clamp_hi.users) != 1:
            return None
        fp8_cast = next(iter(clamp_hi.users))
        if not self._is_op(fp8_cast, torch.ops.prims.convert_element_type.default):
            return None

        if len(fp8_cast.users) != 1:
            return None
        quant_flat = next(iter(fp8_cast.users))
        if not self._is_op(quant_flat, torch.ops.aten.reshape.default):
            return None

        quant_flat_size = quant_flat.args[1]
        group_reshape_size = group_reshape.args[1]
        num_groups = group_reshape_size[1]

        quant_nodes = [
            group_reshape,
            abs_node,
            max_node,
            getitem_max,
            absmax_f32,
            scales_raw_div,
            scales_3d,
            x_div_scales,
            clamp_lo,
            clamp_hi,
            fp8_cast,
            quant_flat,
            squeeze_node,
        ]

        output_map = {0: quant_flat, 1: squeeze_node}

        return (
            quant_flat,
            quant_nodes,
            output_map,
            quant_flat_size,
            num_groups,
        )

    def apply(self, graph: fx.Graph) -> int:
        count = 0
        nodes_to_remove: list[fx.Node] = []

        for node in graph.nodes:
            if not self._is_op(node, torch.ops.prims.convert_element_type.default):
                continue
            if node.args[1] != torch.bfloat16:
                continue

            bf16_cast = node
            norm_result = self._match_gated_norm(bf16_cast)
            if norm_result is None:
                continue

            (
                x_view,
                z_view,
                weight_node,
                eps_val,
                norm_nodes,
                norm_cond_nodes,
            ) = norm_result

            # Try custom op quant first, then decomposed quant
            custom_quant = self._match_custom_op_quant(bf16_cast)
            decomposed_quant = (
                self._match_decomposed_quant(bf16_cast)
                if custom_quant is None
                else None
            )

            if custom_quant is None and decomposed_quant is None:
                continue

            if custom_quant is not None:
                (
                    insert_before,
                    quant_remove,
                    getitem_nodes,
                    reshape_size,
                ) = custom_quant
                num_groups = None
            else:
                assert decomposed_quant is not None
                (
                    insert_before,
                    quant_remove,
                    output_map,
                    reshape_size,
                    num_groups,
                ) = decomposed_quant

            logger.info(
                "RMSNormGatedQuantManualFusion: matched %s pattern at %s",
                "custom-op" if custom_quant else "decomposed",
                insert_before.name,
            )

            with graph.inserting_before(insert_before):
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
                if num_groups is not None:
                    scale_reshape_size = [-1, num_groups]
                else:
                    scale_reshape_size = [reshape_size[0], -1]
                scale_reshaped = graph.call_function(
                    torch.ops.aten.reshape.default,
                    (scale_out, scale_reshape_size),
                )

            if custom_quant is not None:
                for gi_node in getitem_nodes:
                    idx = gi_node.args[1]
                    if idx == 0:
                        gi_node.replace_all_uses_with(quant_reshaped)
                    elif idx == 1:
                        gi_node.replace_all_uses_with(scale_reshaped)
                    nodes_to_remove.append(gi_node)
            else:
                assert decomposed_quant is not None
                output_map[0].replace_all_uses_with(quant_reshaped)
                output_map[1].replace_all_uses_with(scale_reshaped)

            nodes_to_remove.extend(quant_remove)
            nodes_to_remove.extend(norm_nodes)
            for n in norm_cond_nodes:
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

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
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


class RocmAiterRMSNormGatedQuantFusionPass(VllmInductorPass):
    """
    Standalone pass that fuses decomposed RMSNormGated + quant into
    rocm_aiter_rmsnorm_input_quant_fp8. Runs independently of
    fuse_norm_quant so it works with -rms_norm -quant_fp8 custom_ops.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self._fusion = RMSNormGatedQuantManualFusion()

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self._fusion.apply(graph)
        if self.matched_count > 0:
            logger.info(
                "%s replaced %s patterns",
                self.__class__.__name__,
                self.matched_count,
            )

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, RMSNormGatedQuantManualFusion)
