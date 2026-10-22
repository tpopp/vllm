# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm GDN speculative-verify fast path for the flat ``[q|k|v|z]`` packing.

Pure spec-decode batches (every request verifies ``num_spec + 1`` tokens)
skip the generic path's zero-fill, q/k/v re-packing, a/b ``.contiguous()``
copies and output copy. The conv update stays on vLLM's Triton kernel (the
only one that applies ``num_accepted_tokens`` to the conv window) and runs in
place on the packed projection; the recurrent update runs on AITER's
``fused_rearrange_sigmoid_gated_delta_rule``, which reads q/k/v straight from
the strided packed buffer and writes into ``core_attn_out``.
"""

import functools
import inspect

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.triton_utils import tl, triton


@triton.jit
def _gdn_unpack_z_ba_kernel(
    qkvz_ptr,
    ba_ptr,
    z_ptr,
    b_ptr,
    a_ptr,
    stride_qkvz_t,
    stride_ba_t,
    QKV_SIZE: tl.constexpr,
    Z_SIZE: tl.constexpr,
    HV: tl.constexpr,
    BLOCK_Z: tl.constexpr,
    BLOCK_HV: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    zb = tl.program_id(1)
    offs = zb * BLOCK_Z + tl.arange(0, BLOCK_Z)
    mask = offs < Z_SIZE
    z = tl.load(qkvz_ptr + t * stride_qkvz_t + QKV_SIZE + offs, mask=mask)
    tl.store(z_ptr + t * Z_SIZE + offs, z, mask=mask)
    if zb == 0:
        h = tl.arange(0, BLOCK_HV)
        mask_h = h < HV
        b = tl.load(ba_ptr + t * stride_ba_t + h, mask=mask_h)
        a = tl.load(ba_ptr + t * stride_ba_t + HV + h, mask=mask_h)
        tl.store(b_ptr + t * HV + h, b, mask=mask_h)
        tl.store(a_ptr + t * HV + h, a, mask=mask_h)


def gdn_unpack_z_ba(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    z_out: torch.Tensor,
    qkv_size: int,
    num_v_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy z into ``z_out`` and split ``[b|a]`` into contiguous b, a.

    One launch replacing ``z_out[:] = z`` plus the two ``.contiguous()``
    copies of the strided b/a halves.
    """
    num_tokens = qkvz.shape[0]
    z_size = z_out[0].numel()
    assert qkvz.stride(-1) == 1 and ba.stride(-1) == 1
    assert qkvz.shape[-1] == qkv_size + z_size
    assert ba.shape[-1] == 2 * num_v_heads
    assert z_out.is_contiguous() and z_out.dtype == qkvz.dtype
    assert z_out.shape[0] == num_tokens and ba.shape[0] == num_tokens
    b = torch.empty((num_tokens, num_v_heads), dtype=ba.dtype, device=ba.device)
    a = torch.empty_like(b)
    if num_tokens == 0:
        return b, a
    block_z = min(1024, triton.next_power_of_2(z_size))
    grid = (num_tokens, triton.cdiv(z_size, block_z))
    _gdn_unpack_z_ba_kernel[grid](
        qkvz,
        ba,
        z_out,
        b,
        a,
        qkvz.stride(0),
        ba.stride(0),
        QKV_SIZE=qkv_size,
        Z_SIZE=z_size,
        HV=num_v_heads,
        BLOCK_Z=block_z,
        BLOCK_HV=triton.next_power_of_2(num_v_heads),
        num_warps=4,
    )
    return b, a


@functools.cache
def _aiter_gdr():
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule,
    )

    params = inspect.signature(fused_rearrange_sigmoid_gated_delta_rule).parameters
    supports_spec = {"num_accepted_tokens", "ssm_state_indices"} <= params.keys()
    accepts_draft_window = "draft_window" in params
    return fused_rearrange_sigmoid_gated_delta_rule, supports_spec, accepts_draft_window


def aiter_gdr_supports_spec_decode() -> bool:
    """Whether AITER's recurrence takes ``num_accepted_tokens`` and 2-D
    ``ssm_state_indices``; older builds keep the generic spec path."""
    return _aiter_gdr()[1]


def gdn_spec_decode_packed_aiter(
    *,
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_bias: torch.Tensor | None,
    activation: str | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    spec_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    num_actual_tokens: int,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    draft_window: int | None = None,
) -> None:
    """Conv1d + gated delta rule for a pure spec-decode batch (flat layout).

    Matches the generic ``_forward_core`` spec path: same conv kernel and
    arguments, same recurrence and state-slot semantics. ``qkvz`` is
    (num_tokens, qkvz_dim) and its qkv columns are overwritten with the conv
    output, as in the generic path. ``z_out`` and the contiguous
    ``core_attn_out`` are (num_tokens, num_v_heads, head_v_dim) outputs.
    ``draft_window`` is ``num_spec + 1`` when every spec request carries the
    full draft window, which lets AITER's opt-in FlyDSL kernel take the call.
    """
    qkv_size = 2 * key_dim + value_dim
    num_v_heads = value_dim // head_v_dim

    b, a = gdn_unpack_z_ba(qkvz, ba, z_out, qkv_size, num_v_heads)
    # The generic path zero-fills core_attn_out and then overwrites the first
    # num_actual_tokens rows; only the tail beyond them survives the overwrite.
    if num_actual_tokens < core_attn_out.shape[0]:
        core_attn_out[num_actual_tokens:].zero_()

    gdn_spec_decode_conv_gdr_aiter(
        qkv=qkvz[:num_actual_tokens, :qkv_size],
        b=b,
        a=a,
        core_attn_out=core_attn_out,
        conv_state=conv_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation=activation,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_state=ssm_state,
        spec_query_start_loc=spec_query_start_loc,
        spec_state_indices=spec_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        num_spec_decodes=num_spec_decodes,
        key_dim=key_dim,
        value_dim=value_dim,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        draft_window=draft_window,
    )


def gdn_spec_decode_conv_gdr_aiter(
    *,
    qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_bias: torch.Tensor | None,
    activation: str | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    spec_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    draft_window: int | None = None,
) -> None:
    """Spec-token conv update + AITER recurrence on already-split inputs.

    ``qkv`` is the (strided) qkv column view of the spec tokens and is
    overwritten with the conv output; ``b``/``a`` are contiguous and at least
    as long as ``qkv``; ``core_attn_out`` is a contiguous output whose first
    ``qkv.shape[0]`` rows are written.
    """
    mixed_qkv = causal_conv1d_update(
        qkv,
        conv_state,
        conv_weights,
        conv_bias,
        activation,
        conv_state_indices=spec_state_indices[:, 0][:num_spec_decodes],
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=spec_query_start_loc,
        max_query_len=spec_state_indices.size(-1),
        validate_data=False,
    )

    gdr, _, accepts_draft_window = _aiter_gdr()
    kwargs = {}
    if accepts_draft_window and draft_window is not None:
        kwargs["draft_window"] = draft_window
    gdr(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        qkv=mixed_qkv,
        key_dim=key_dim,
        value_dim=value_dim,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        initial_state=ssm_state,
        inplace_final_state=True,
        cu_seqlens=spec_query_start_loc[: num_spec_decodes + 1],
        ssm_state_indices=spec_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
        core_attn_out=core_attn_out.view(-1),
        **kwargs,
    )
