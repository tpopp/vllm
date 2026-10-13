# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm GDN mixed (prefill + spec verify) fast path must match the generic
mixed path.

Both sides run the real ``_forward_core_rocm`` on metadata from the real
``GDNAttentionMetadataBuilder``: the reference with the bool-mask gather
metadata and the generic ``_forward_core`` (index_select / index_copy_), the
new side with sliced spec-prefix metadata and ``_forward_core_spec_mixed_aiter``.
Outputs, ``z_out``, conv state and recurrent state must agree.

CPU check: ``VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1``
with an importable ``aiter`` GDR Triton kernel (FP32 activations).
"""

import importlib.util
import os
import types
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

ON_ROCM = current_platform.is_rocm()
if not (ON_ROCM or os.environ.get("TRITON_INTERPRET") == "1"):
    pytest.skip(
        "ROCm-only GDN mixed spec path (or TRITON_INTERPRET=1 on CPU)",
        allow_module_level=True,
    )
pytest.importorskip("aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr")

from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops import (  # noqa: E402
    gdn_spec_decode_rocm,
)
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

NUM_SPEC = 3
H, HV, K, V = 2, 4, 128, 128
CONV_KERNEL = 4
KEY_DIM, VALUE_DIM = H * K, HV * V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"
DEVICE = torch.device("cuda" if ON_ROCM else "cpu")
ACT_DTYPE = torch.bfloat16 if ON_ROCM else torch.float32

# (seq_lens, query_lens, num_decode_draft_tokens); spec rows lead the batch.
CASES = {
    "mtp-uniform": ([80, 96, 70, 30], [4, 4, 70, 9], [3, 3, -1, -1]),
    "fresh-prefill": ([80, 70, 100], [4, 70, 60], [3, -1, -1]),
    "ragged-1tok-chunk": ([80, 96, 77, 65, 130], [2, 4, 1, 1, 66], [1, 3, -1, -1, -1]),
    "single-spec": ([64, 5], [4, 5], [3, -1]),
}


@pytest.fixture(autouse=True)
def _triton_conv_update_on_cpu(monkeypatch):
    """The CPU platform swaps in C++ conv ops: run Triton's conv update under
    the interpreter, keep the torch reference for the prefill conv (whose Triton
    kernel the interpreter cannot run); both sides use the same functions."""
    if ON_ROCM:
        return
    spec = importlib.util.find_spec(
        "vllm.model_executor.layers.mamba.ops.causal_conv1d"
    )
    assert spec is not None and spec.loader is not None
    triton_conv = importlib.util.module_from_spec(spec)
    with patch.object(current_platform, "is_cpu", return_value=False):
        spec.loader.exec_module(triton_conv)
    for module in (qwen_gdn_linear_attn, gdn_spec_decode_rocm):
        monkeypatch.setattr(
            module, "causal_conv1d_update", triton_conv.causal_conv1d_update
        )


def _chunk_gated_delta_rule_torch(
    *, q, k, v, g, beta, initial_state, cu_seqlens, core_attn_out=None, **_
):
    """Sequential gated delta rule; stands in for FLA's BF16-only chunk kernel
    under the FP32 interpreter (both paths call the same function)."""
    hv = v.shape[2]
    q = q[0].repeat_interleave(hv // q.shape[2], dim=1).float()
    k = k[0].repeat_interleave(hv // k.shape[2], dim=1).float()
    v, g, beta = v[0].float(), g[0].float(), beta[0].float()
    out = torch.empty_like(v)
    final = torch.empty_like(initial_state)
    bounds = cu_seqlens.tolist()
    for n, (bos, eos) in enumerate(zip(bounds[:-1], bounds[1:])):
        state = initial_state[n].float()  # (HV, V, K)
        for t in range(bos, eos):
            state = state * g[t].exp()[:, None, None]
            u = (v[t] - torch.einsum("hvk,hk->hv", state, k[t])) * beta[t][:, None]
            state = state + u[:, :, None] * k[t][:, None, :]
            out[t] = torch.einsum("hvk,hk->hv", state, q[t]) * q.shape[-1] ** -0.5
        final[n] = state.to(final.dtype)
    out = out.to(v.dtype).unsqueeze(0)
    if core_attn_out is not None:
        core_attn_out.reshape(-1)[: out.numel()].copy_(out.reshape(-1))
    return out, final


def _vllm_config():
    cfg = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        hf_config_override={"linear_key_head_dim": K},
    )
    cfg.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    cfg.cache_config.mamba_cache_mode = "none"
    return cfg


def _layer(vllm_config, kv_cache, params, fast_path):
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_packed_recurrent_decode=False,
        enable_rocm_spec_fast_path=fast_path,
        gqa_interleaved_layout=False,
        tp_size=1,
        num_k_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        num_spec=NUM_SPEC,
        activation="silu",
        A_log=params["A_log"],
        dt_bias=params["dt_bias"],
        conv1d=types.SimpleNamespace(
            weight=params["conv_weight"], bias=params["conv_bias"]
        ),
        kv_cache=kv_cache,
        _is_pure_spec_decode=QwenGatedDeltaNetAttention._is_pure_spec_decode,
        _is_spec_prefix_mixed=QwenGatedDeltaNetAttention._is_spec_prefix_mixed,
    )
    if ON_ROCM:
        with set_current_vllm_config(vllm_config):
            layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    else:
        layer.chunk_gated_delta_rule = _chunk_gated_delta_rule_torch
    for name in (
        "prepare_gdn_attention_core_inputs",
        "rearrange_mixed_qkv",
        "_forward_core",
        "_forward_core_rocm",
        "_forward_core_spec_mixed_aiter",
    ):
        method = getattr(QwenGatedDeltaNetAttention, name)
        setattr(layer, name, types.MethodType(method, layer))
    return layer


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_spec_mixed_fast_path_matches_generic_path(case, state_dtype):
    if state_dtype == torch.bfloat16 and not ON_ROCM:
        pytest.skip("Triton interpreter runs FP32 only")
    seq_lens, query_lens, draft_tokens = CASES[case]
    torch.manual_seed(0)
    vllm_config = _vllm_config()
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE, arange_block_indices=True
    )
    # Slot 0 is NULL_BLOCK_ID, which FLA skips and AITER does not.
    common = common.replace(block_table_tensor=common.block_table_tensor + 1)
    num_spec_decodes = sum(d >= 0 for d in draft_tokens)
    num_accepted = torch.ones(batch.batch_size, dtype=torch.int32)
    num_accepted[:num_spec_decodes] = torch.randint(
        1, NUM_SPEC + 2, (num_spec_decodes,)
    )
    num_accepted = num_accepted.to(DEVICE)

    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, H, HV, K, V, CONV_KERNEL, num_spec=NUM_SPEC
    )
    metas = []
    for use_prefix in (False, True):
        with set_current_vllm_config(vllm_config):
            builder = GDNAttentionMetadataBuilder(
                kv_cache_spec=MambaSpec(
                    block_size=BLOCK_SIZE,
                    shapes=(conv_shape, ssm_shape),
                    dtypes=(ACT_DTYPE, state_dtype),
                    num_speculative_blocks=NUM_SPEC,
                ),
                layer_names=[PREFIX],
                vllm_config=vllm_config,
                device=DEVICE,
            )
            builder.use_spec_rows_prefix_metadata = use_prefix
            metas.append(
                builder.build(
                    0,
                    common,
                    num_accepted_tokens=num_accepted,
                    num_decode_draft_tokens_cpu=torch.tensor(
                        draft_tokens, dtype=torch.int32
                    ),
                )
            )
    meta_gather, meta_prefix = metas
    assert meta_prefix.spec_rows_prefix and not meta_gather.spec_rows_prefix
    assert meta_prefix.num_prefills > 0 and meta_prefix.num_decodes == 0

    pool = int(common.block_table_tensor.max()) + 1
    conv_state0 = torch.randn(pool, *conv_shape, device=DEVICE, dtype=ACT_DTYPE)
    ssm_state0 = (torch.randn(pool, *ssm_shape, device=DEVICE) * 0.05).to(state_dtype)
    params = dict(
        A_log=torch.randn(HV, device=DEVICE) * 0.5,
        dt_bias=torch.randn(HV, device=DEVICE, dtype=ACT_DTYPE) * 0.5,
        conv_weight=torch.randn(
            CONV_DIM, 1, CONV_KERNEL, device=DEVICE, dtype=ACT_DTYPE
        )
        * 0.3,
        conv_bias=torch.randn(CONV_DIM, device=DEVICE, dtype=ACT_DTYPE) * 0.1,
    )
    num_tokens = sum(query_lens)
    num_rows = num_tokens + 3  # PIECEWISE padding rows past num_actual_tokens
    qkvz = torch.randn(num_rows, CONV_DIM + VALUE_DIM, device=DEVICE, dtype=ACT_DTYPE)
    ba = torch.randn(num_rows, 2 * HV, device=DEVICE, dtype=ACT_DTYPE)

    def run(meta, fast_path):
        kv_cache = (conv_state0.clone(), ssm_state0.clone())
        layer = _layer(vllm_config, kv_cache, params, fast_path)
        calls = []
        fast = layer._forward_core_spec_mixed_aiter

        def counted_fast(**kw):
            calls.append(1)
            return fast(**kw)

        layer._forward_core_spec_mixed_aiter = counted_fast
        nan = torch.full(
            (num_rows, HV, V), float("nan"), device=DEVICE, dtype=ACT_DTYPE
        )
        z_out, core_attn_out = nan.clone(), nan.clone()
        ctx = types.SimpleNamespace(attn_metadata={PREFIX: meta})
        with patch.object(
            qwen_gdn_linear_attn, "get_forward_context", return_value=ctx
        ):
            layer._forward_core_rocm(
                qkvz=qkvz.clone(),
                ba=ba.clone(),
                z_out=z_out,
                core_attn_out=core_attn_out,
            )
        return (z_out, core_attn_out, *kv_cache), len(calls)

    ref, ref_fast = run(meta_gather, fast_path=True)  # gather metadata: generic
    new, new_fast = run(meta_prefix, fast_path=True)
    old_on_prefix_meta, old_fast = run(meta_prefix, fast_path=False)
    assert (ref_fast, new_fast, old_fast) == (0, 1, 0)

    exact = dict(rtol=0, atol=0)
    if ON_ROCM:
        tol = dict(rtol=1e-2, atol=1e-2)
        state_tol = tol if state_dtype == torch.bfloat16 else dict(rtol=1e-4, atol=1e-4)
    else:
        tol = state_tol = exact  # FP32 interpreter: bit-identical
    for other in (old_on_prefix_meta, new):
        z_out, out, conv_state, ssm_state = other
        torch.testing.assert_close(z_out, ref[0], **exact)
        torch.testing.assert_close(conv_state, ref[2], **exact)
        torch.testing.assert_close(out[:num_tokens], ref[1][:num_tokens], **tol)
        torch.testing.assert_close(ssm_state, ref[3], **state_tol)
        assert torch.all(out[num_tokens:] == 0)
