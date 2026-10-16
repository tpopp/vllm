# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``fused_gdn_verify_core`` must match the stock spec-decode verify chain:
``causal_conv1d_update`` (conv1d, rewound by ``num_accepted_tokens``) ->
``fused_sigmoid_gating_delta_rule_update`` (the in-tree, non-AITER FLA
kernel -- not AITER's, so this test needs no ``aiter`` install, matching
``tests/kernels/mamba/test_gdn_spec_decode_rocm.py``'s own reference) ->
``RMSNormGated.forward_static`` (the gated RMSNorm).

Shapes are Qwen3.8-Flash-Next's TP2 GDN layer: NK=8 key heads, NV=24 value
heads (3 value heads share each key head), head dim 128, conv width 4, MTP
k=3 (a verify step is up to 4 tokens/request).
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only GDN verify fused kernel", allow_module_level=True)

from vllm.model_executor.layers.layernorm import RMSNormGated  # noqa: E402
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.gdn_spec_decode_mono_fused import (  # noqa: E402
    fused_gdn_verify_core,
)
from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update,
)

NUM_SPEC = 3  # Qwen3.8-Flash-Next MTP k
DRAFT_WINDOW = NUM_SPEC + 1  # 4 tokens/request at most
NK, NV, HD = 8, 24, 128  # TP2 per-rank shape (16/2, 48/2, 128)
RATIO = NV // NK  # 3
CONV_W = 4
NORM_EPS = 1e-6
DEVICE = "cuda"
DTYPE = torch.bfloat16


def _reference(
    *,
    qkvz,
    ba,
    conv_state,
    conv_weight,
    A_log,
    dt_bias,
    norm_weight,
    norm_activation,
    ssm_state,
    qsl,
    idx,
    nacc,
    n_req,
):
    key_dim, value_dim = NK * HD, NV * HD
    qkv_size = 2 * key_dim + value_dim
    mixed_qkv, z = qkvz.split([qkv_size, value_dim], dim=-1)
    b, a = ba.chunk(2, dim=-1)

    # causal_conv1d_update's grid size comes from conv_state_indices'
    # length, so a longer (padding-row-including) query_start_loc is read
    # harmlessly only up to that many rows.
    mixed_qkv = causal_conv1d_update(
        mixed_qkv.clone(),
        conv_state,
        conv_weight,
        None,
        "silu",
        conv_state_indices=idx[:, 0][:n_req],
        num_accepted_tokens=nacc[:n_req],
        query_start_loc=qsl,
        max_query_len=idx.size(-1),
        validate_data=False,
    )
    q, k, v = mixed_qkv.split([key_dim, key_dim, value_dim], dim=-1)
    t = mixed_qkv.shape[0]
    # fused_sigmoid_gating_delta_rule_update's grid size comes from
    # len(cu_seqlens) - 1, so this *must* exclude any padding row (unlike the
    # conv call above) or it indexes ssm_state_indices/num_accepted_tokens
    # out of their (sliced) bounds.
    o, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q.contiguous().view(1, t, NK, HD),
        k=k.contiguous().view(1, t, NK, HD),
        v=v.contiguous().view(1, t, NV, HD),
        initial_state=ssm_state,
        inplace_final_state=True,
        cu_seqlens=qsl[: n_req + 1],
        ssm_state_indices=idx[:n_req],
        num_accepted_tokens=nacc[:n_req],
        use_qk_l2norm_in_kernel=True,
    )
    core = o.squeeze(0)  # [t, NV, HD]
    core = RMSNormGated.forward_static(
        core,
        z.view(t, NV, HD),
        norm_weight,
        NORM_EPS,
        core.dtype,
        group_size=None,
        norm_before_gate=True,
        activation=norm_activation,
    )
    return core, conv_state, ssm_state


# (q_lens per request, accepted-tokens-from-prior-cycle per request,
# has a zero-accept / invalid padding request appended)
CASES = {
    "single-full": ([4], [1], False),
    "single-partial-accept": ([4], [3], False),
    "two-uniform": ([4, 4], [1, 4], False),
    "varlen": ([4, 2], [2, 1], False),
    "padded-row": ([4, 4], [1, 2], True),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("norm_activation", ["sigmoid", "silu"])
@torch.inference_mode()
def test_fused_gdn_verify_core_matches_stock_chain(case, norm_activation):
    q_lens, accepted, add_padding_row = CASES[case]
    torch.manual_seed(0)

    n_req = len(q_lens)
    # The padding row has a real (non-empty) token span, like a CUDA-graph-
    # padded or zero-accept "stale spec row" (background: vllm#50021 /
    # vllm#51508) still occupying batch width this step, but its state_
    # indices row is all NULL_BLOCK_ID (0) -- the "no slot" sentinel both the
    # stock kernels and the fused one special-case, independent of whichever
    # column num_accepted_tokens would otherwise select.
    pad_len = DRAFT_WINDOW if add_padding_row else 0
    batch = n_req + (1 if add_padding_row else 0)
    num_tokens = sum(q_lens) + pad_len
    key_dim, value_dim = NK * HD, NV * HD
    conv_dim = 2 * key_dim + value_dim
    num_slots = 16

    qsl = torch.zeros(batch + 1, dtype=torch.int32)
    qsl[1 : n_req + 1] = torch.tensor(q_lens, dtype=torch.int32).cumsum(0)
    if add_padding_row:
        qsl[n_req + 1] = qsl[n_req] + pad_len

    nacc = torch.ones(batch, dtype=torch.int32)
    nacc[:n_req] = torch.tensor(accepted, dtype=torch.int32)

    idx = torch.zeros(batch, DRAFT_WINDOW, dtype=torch.int32)
    slots = torch.randperm(num_slots - 1)[: n_req * DRAFT_WINDOW] + 1
    idx[:n_req] = slots.view(n_req, DRAFT_WINDOW)

    qsl, nacc, idx = (x.to(DEVICE) for x in (qsl, nacc, idx))

    qkvz = torch.randn(num_tokens, conv_dim + value_dim, device=DEVICE, dtype=DTYPE)
    ba = torch.randn(num_tokens, 2 * NV, device=DEVICE, dtype=DTYPE)
    conv_weight = torch.randn(conv_dim, CONV_W, device=DEVICE, dtype=DTYPE) * 0.3
    # Default (SD) layout: stored (slots, state_len, dim); the kernels (both
    # stock and fused) take the transposed (slots, dim, state_len) view.
    conv_state = torch.randn(
        num_slots, CONV_W - 1 + NUM_SPEC, conv_dim, device=DEVICE, dtype=DTYPE
    ).transpose(-1, -2)
    ssm_state = torch.randn(num_slots, NV, HD, HD, device=DEVICE) * 0.05
    A_log = torch.randn(NV, device=DEVICE)
    dt_bias = torch.randn(NV, device=DEVICE, dtype=DTYPE)
    norm_weight = torch.rand(HD, device=DEVICE, dtype=DTYPE) + 0.5

    def run(fn, **extra):
        return fn(
            qkvz=qkvz,
            ba=ba,
            conv_state=conv_state.clone(memory_format=torch.preserve_format),
            conv_weight=conv_weight,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_state=ssm_state.clone(),
            **extra,
        )

    ref_core, ref_conv_state, ref_ssm_state = run(
        _reference,
        norm_weight=norm_weight,
        norm_activation=norm_activation,
        qsl=qsl,
        idx=idx,
        nacc=nacc,
        n_req=n_req,
    )

    core = torch.full(
        (num_tokens, NV, HD), float("nan"), device=DEVICE, dtype=DTYPE
    )
    new_conv_state = conv_state.clone(memory_format=torch.preserve_format)
    new_ssm_state = ssm_state.clone()
    fused_gdn_verify_core(
        qkvz=qkvz,
        ba=ba,
        core_attn_out=core,
        conv_state=new_conv_state,
        conv_weight=conv_weight,
        conv_bias=None,
        A_log=A_log,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS,
        norm_activation=norm_activation,
        ssm_state=new_ssm_state,
        spec_query_start_loc=qsl,
        spec_state_indices=idx,
        num_accepted_tokens=nacc,
        num_k_heads=NK,
        num_v_heads=NV,
        head_k_dim=HD,
        head_v_dim=HD,
    )

    if add_padding_row:
        # fused_sigmoid_gating_delta_rule_update (unlike
        # fused_gdn_decode_post_conv_mtp_triton, which this kernel's post-conv
        # math otherwise mirrors) early-returns without zeroing `o` for an
        # invalid checkpoint slot, leaving its padding-row output
        # uninitialized; the fused kernel always zeros a padding row. Both
        # conventions mean "don't care here" -- force zero on both sides
        # before comparing so neither garbage nor a convention mismatch fails
        # the test, and separately assert the fused kernel's own convention.
        core[-pad_len:] = 0
        ref_core[-pad_len:] = 0
        assert torch.all(core[-pad_len:] == 0)

    torch.testing.assert_close(core, ref_core, rtol=2e-2, atol=2e-2)
    cos = torch.nn.functional.cosine_similarity(
        core.flatten().float(), ref_core.flatten().float(), dim=0
    )
    assert cos > 0.9999
    torch.testing.assert_close(new_conv_state, ref_conv_state, rtol=0, atol=0)
    torch.testing.assert_close(new_ssm_state, ref_ssm_state, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("accepted_value", [1, 2, 3, 4])
@torch.inference_mode()
def test_fused_gdn_verify_core_each_accepted_count(accepted_value):
    """Every num_accepted_tokens value in [1, DRAFT_WINDOW] rewinds the conv
    window and picks the SSM checkpoint column correctly."""
    torch.manual_seed(accepted_value)
    num_tokens = DRAFT_WINDOW
    key_dim, value_dim = NK * HD, NV * HD
    conv_dim = 2 * key_dim + value_dim
    num_slots = 16

    qsl = torch.tensor([0, num_tokens], dtype=torch.int32, device=DEVICE)
    nacc = torch.tensor([accepted_value], dtype=torch.int32, device=DEVICE)
    idx = (
        (torch.arange(1, DRAFT_WINDOW + 1, dtype=torch.int32) * 2)
        .view(1, DRAFT_WINDOW)
        .to(DEVICE)
    )

    qkvz = torch.randn(num_tokens, conv_dim + value_dim, device=DEVICE, dtype=DTYPE)
    ba = torch.randn(num_tokens, 2 * NV, device=DEVICE, dtype=DTYPE)
    conv_weight = torch.randn(conv_dim, CONV_W, device=DEVICE, dtype=DTYPE) * 0.3
    conv_state = torch.randn(
        num_slots, CONV_W - 1 + NUM_SPEC, conv_dim, device=DEVICE, dtype=DTYPE
    ).transpose(-1, -2)
    ssm_state = torch.randn(num_slots, NV, HD, HD, device=DEVICE) * 0.05
    A_log = torch.randn(NV, device=DEVICE)
    dt_bias = torch.randn(NV, device=DEVICE, dtype=DTYPE)
    norm_weight = torch.rand(HD, device=DEVICE, dtype=DTYPE) + 0.5

    ref_core, ref_conv_state, ref_ssm_state = _reference(
        qkvz=qkvz,
        ba=ba,
        conv_state=conv_state.clone(memory_format=torch.preserve_format),
        conv_weight=conv_weight,
        A_log=A_log,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
        norm_activation="sigmoid",
        ssm_state=ssm_state.clone(),
        qsl=qsl,
        idx=idx,
        nacc=nacc,
        n_req=1,
    )

    core = torch.full((num_tokens, NV, HD), float("nan"), device=DEVICE, dtype=DTYPE)
    new_conv_state = conv_state.clone(memory_format=torch.preserve_format)
    new_ssm_state = ssm_state.clone()
    fused_gdn_verify_core(
        qkvz=qkvz,
        ba=ba,
        core_attn_out=core,
        conv_state=new_conv_state,
        conv_weight=conv_weight,
        conv_bias=None,
        A_log=A_log,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS,
        norm_activation="sigmoid",
        ssm_state=new_ssm_state,
        spec_query_start_loc=qsl,
        spec_state_indices=idx,
        num_accepted_tokens=nacc,
        num_k_heads=NK,
        num_v_heads=NV,
        head_k_dim=HD,
        head_v_dim=HD,
    )

    torch.testing.assert_close(core, ref_core, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(new_conv_state, ref_conv_state, rtol=0, atol=0)
    torch.testing.assert_close(new_ssm_state, ref_ssm_state, rtol=1e-2, atol=1e-2)
