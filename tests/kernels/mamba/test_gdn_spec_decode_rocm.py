# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm GDN spec-verify fast path must match the generic spec path.

The reference replays the generic ``_forward_core_rocm`` -> ``_forward_core``
sequence for a pure spec-decode batch with the flat ``[q|k|v|z]`` packing
(zero-fill, split, ``z_out[:] = z``, Triton conv update, q/k/v re-pack, FLA
``fused_sigmoid_gating_delta_rule_update``, output copy). Both paths run on
cloned caches; outputs, ``z_out``, conv state and recurrent state must agree.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only GDN spec-decode path", allow_module_level=True)
pytest.importorskip("aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr")

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.gdn_spec_decode_rocm import (  # noqa: E402
    gdn_spec_decode_packed_aiter,
)
from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update,
)

NUM_SPEC = 3
WIDTH = 4
K = V = 128
DEVICE = "cuda"


def _reference(
    *,
    qkvz,
    ba,
    z_out,
    core_attn_out,
    conv_state,
    conv_weights,
    A_log,
    dt_bias,
    ssm_state,
    qsl,
    idx,
    nacc,
    num_spec_decodes,
    num_actual_tokens,
    hk,
    hv,
):
    key_dim, value_dim = hk * K, hv * V
    qkv_size = 2 * key_dim + value_dim
    core_attn_out.zero_()
    mixed_qkv, z = qkvz.split([qkv_size, value_dim], dim=-1)
    z_out[:] = z.reshape(qkvz.shape[0], hv, V)
    b, a = ba.chunk(2, dim=-1)
    mixed_qkv = mixed_qkv[:num_actual_tokens]
    a, b = a[:num_actual_tokens], b[:num_actual_tokens]
    mixed_qkv = causal_conv1d_update(
        mixed_qkv,
        conv_state,
        conv_weights,
        None,
        "silu",
        conv_state_indices=idx[:, 0][:num_spec_decodes],
        num_accepted_tokens=nacc,
        query_start_loc=qsl,
        max_query_len=idx.size(-1),
        validate_data=False,
    )
    q, k, v = mixed_qkv.split([key_dim, key_dim, value_dim], dim=-1)
    t = mixed_qkv.shape[0]
    o, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q.contiguous().view(1, t, hk, K),
        k=k.contiguous().view(1, t, hk, K),
        v=v.contiguous().view(1, t, hv, V),
        initial_state=ssm_state,
        inplace_final_state=True,
        cu_seqlens=qsl[: num_spec_decodes + 1],
        ssm_state_indices=idx,
        num_accepted_tokens=nacc,
        use_qk_l2norm_in_kernel=True,
    )
    core_attn_out[:num_actual_tokens] = o.squeeze(0)


CASES = {
    # q_lens of real spec requests, padded requests, padded tokens inside
    # num_actual_tokens (FULL graph padding), rows beyond num_actual_tokens.
    "single": ([4], 0, 0, 0),
    "uniform": ([4] * 5, 0, 0, 0),
    "ragged": ([4, 1, 3, 2, 4], 0, 0, 0),
    "graph-padded": ([4, 4, 4], 2, 5, 3),
    "ragged-padded": ([2, 4, 1], 3, 7, 1),
    "large": ([4] * 64, 0, 0, 0),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("heads", [(2, 4), (8, 16)], ids=["h2v4", "h8v16"])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("flydsl", [False, True], ids=["triton", "flydsl"])
@torch.inference_mode()
def test_spec_fast_path_matches_generic_path(
    case, heads, state_dtype, flydsl, monkeypatch
):
    monkeypatch.setenv("AITER_GDR_FLYDSL", "1" if flydsl else "0")
    q_lens, num_pad_seqs, pad_tokens, extra_rows = CASES[case]
    hk, hv = heads
    key_dim, value_dim = hk * K, hv * V
    conv_dim = 2 * key_dim + value_dim
    dtype = torch.bfloat16
    torch.manual_seed(0)

    num_spec_decodes = len(q_lens)
    batch = num_spec_decodes + num_pad_seqs
    num_real_tokens = sum(q_lens)
    num_actual_tokens = num_real_tokens + pad_tokens
    num_rows = num_actual_tokens + extra_rows
    num_slots = batch * (NUM_SPEC + 1) + 8

    qsl = torch.full((batch + 1,), num_real_tokens, dtype=torch.int32)
    qsl[0] = 0
    qsl[1 : num_spec_decodes + 1] = torch.tensor(q_lens).cumsum(0)
    nacc = torch.ones(batch, dtype=torch.int32)
    nacc[:num_spec_decodes] = torch.randint(1, NUM_SPEC + 2, (num_spec_decodes,))
    idx = torch.zeros(batch, NUM_SPEC + 1, dtype=torch.int32)
    slots = torch.randperm(num_slots - 1)[: num_spec_decodes * (NUM_SPEC + 1)] + 1
    idx[:num_spec_decodes] = slots.view(num_spec_decodes, NUM_SPEC + 1)
    qsl, nacc, idx = (x.to(DEVICE) for x in (qsl, nacc, idx))

    qkvz = torch.randn(num_rows, conv_dim + value_dim, device=DEVICE, dtype=dtype)
    ba = torch.randn(num_rows, 2 * hv, device=DEVICE, dtype=dtype)
    conv_weights = torch.randn(conv_dim, WIDTH, device=DEVICE, dtype=dtype) * 0.3
    # Default (SD) layout: stored (slots, state_len, dim), kernels see the
    # transposed (slots, dim, state_len) view.
    conv_state = torch.randn(
        num_slots, WIDTH - 1 + NUM_SPEC, conv_dim, device=DEVICE, dtype=dtype
    ).transpose(-1, -2)
    ssm_state = (torch.randn(num_slots, hv, V, K, device=DEVICE) * 0.05).to(state_dtype)
    A_log = torch.randn(hv, device=DEVICE)
    dt_bias = torch.randn(hv, device=DEVICE, dtype=dtype)

    def run(fn, extra):
        nan = torch.full((num_rows, hv, V), float("nan"), device=DEVICE, dtype=dtype)
        st = dict(
            qkvz=qkvz.clone(),
            ba=ba.clone(),
            z_out=nan.clone(),
            core_attn_out=nan.clone(),
            conv_state=conv_state.clone(memory_format=torch.preserve_format),
            ssm_state=ssm_state.clone(),
        )
        assert st["conv_state"].stride(-2) == 1
        fn(
            **st,
            conv_weights=conv_weights,
            A_log=A_log,
            dt_bias=dt_bias,
            num_spec_decodes=num_spec_decodes,
            num_actual_tokens=num_actual_tokens,
            **extra,
        )
        return st

    ref = run(_reference, dict(qsl=qsl, idx=idx, nacc=nacc, hk=hk, hv=hv))
    uniform = all(n == NUM_SPEC + 1 for n in q_lens) and num_pad_seqs == 0
    new = run(
        gdn_spec_decode_packed_aiter,
        dict(
            conv_bias=None,
            activation="silu",
            spec_query_start_loc=qsl,
            spec_state_indices=idx,
            num_accepted_tokens=nacc,
            key_dim=key_dim,
            value_dim=value_dim,
            head_k_dim=K,
            head_v_dim=V,
            draft_window=NUM_SPEC + 1 if uniform else None,
        ),
    )

    torch.testing.assert_close(new["z_out"], ref["z_out"], rtol=0, atol=0)
    torch.testing.assert_close(new["conv_state"], ref["conv_state"], rtol=0, atol=0)
    tol = 2e-2 if flydsl else 1e-2
    torch.testing.assert_close(
        new["core_attn_out"][:num_real_tokens],
        ref["core_attn_out"][:num_real_tokens],
        rtol=tol,
        atol=tol,
    )
    state_tol = 1e-2 if state_dtype == torch.bfloat16 or flydsl else 1e-4
    torch.testing.assert_close(
        new["ssm_state"], ref["ssm_state"], rtol=state_tol, atol=state_tol
    )
    # Rows past num_actual_tokens are zero-filled, as in the generic path.
    assert torch.all(new["core_attn_out"][num_actual_tokens:] == 0)
