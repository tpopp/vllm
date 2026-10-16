# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One Triton kernel for a pure spec-decode GDN verify step, flat ``[q|k|v|z]``
/ ``[b|a]`` packing: unpack + ``causal_conv1d_update`` + the sigmoid-gated
delta rule + the gated RMSNorm, in one launch per ``(request, value head)``.

Stock (``gdn_spec_decode_rocm.py``) runs this as unpack (1 launch) + the
Triton conv1d update (1 launch, the only kernel that applies
``num_accepted_tokens``) + AITER's ``fused_rearrange_sigmoid_gated_delta_rule``
(1 launch) + the gated-norm ``torch.compile`` (1 launch): 4 launches reading
and writing ``qkvz``/``core_attn_out`` from global memory between each step.
This kernel never materializes the unpacked or conv-updated q/k/v anywhere:
each program reads its request's packed in_proj outputs once, does the
width-4 causal conv in registers (a sequential token loop, since the conv
window of token ``t`` depends on token ``t-1``), feeds the conv output
straight into the same program's delta-rule recurrence and gated RMSNorm, and
writes only ``core_attn_out`` (plus the conv/recurrent state) to memory.

One program per ``(request, value head)``, matching upstream #61855's port of
``fused_gdn_decode_post_conv_mtp`` (``vllm/third_party/flash_linear_attention/
ops/fused_gdn_decode_post_conv_mtp.py``), whose post-conv math (L2-norm,
sigmoid gate, delta rule, gated RMSNorm, state read/write semantics) this
kernel reuses verbatim; the piece #61855 does not fuse -- the conv1d update,
and reading q/k/v out of the *packed* qkvz buffer rather than a pre-unpacked
``mixed_qkv`` -- is added here.

Key heads are shared by ``HV // H`` value heads (``NV=24``/``NK=8`` -> 3 for
Qwen3.8-Flash-Next TP2). Every program in a key head's group recomputes that
head's q/k conv redundantly (cheap: 128 lanes, 4 taps, a handful of FMAs per
verify token) so that each program can feed its own delta-rule step without a
cross-program handoff; only the group's first program (``i_hv % RATIO == 0``)
persists the shared q/k conv state back to memory, so the write happens once.

Conv state addressing (``conv_state`` already oriented to a logical
``(slots, dim, state_len)`` view, matching ``causal_conv1d_update``'s
``conv_state_indices``/``num_accepted_tokens`` contract bit for bit):
``state_len = CONV_W - 1 + num_spec`` is a fixed, model-wide allocation
(``MambaStateShapeCalculator.gated_delta_net_state_shape``); a request's own
verify width this step, ``S``, can be anywhere up to ``num_spec + 1``.  Given
``offset = num_accepted_tokens[req] - 1`` (the last valid position of last
step's window):

- the ``CONV_W - 1`` taps needed to seed this step's conv are
  ``conv_state[slot, ch, offset : offset + CONV_W - 1]``;
- the window written back after processing this step's ``S`` new tokens is
  exactly the top ``CONV_W - 2`` of those taps (the oldest tap is dropped)
  followed by the ``S`` new (pre-conv) inputs, length
  ``(CONV_W - 2) + S``, which is always <= the physical ``state_len``
  (``CONV_W - 1 + num_spec``) since ``S <= num_spec + 1``.

This holds per request independent of any other request's ``S``, so unlike
the batched stock kernel (which pads every row to a shared ``state_len`` to
share one Triton grid axis), this one-program-per-request kernel needs no
batch-wide uniform-width assumption for the conv or delta-rule math; see
``gdn_verify_eligible`` for the (narrower) conditions this module still
requires before the caller uses it.

A request row is treated as padding -- core zeroed, no conv/state writes --
when its dedicated conv slot (``spec_state_indices[req, 0]``) is <= 0, or its
``num_accepted_tokens`` is out of ``[1, spec_state_indices.size(1)]`` (the
delta rule's own checkpoint-read bound, mirrored from
``fused_gdn_decode_post_conv_mtp_triton``/``fused_sigmoid_gating_delta_rule_update``).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_gdn_verify_core_kernel(
    qkvz_ptr,
    ba_ptr,
    conv_weight_ptr,
    conv_bias_ptr,
    conv_state_ptr,
    A_log_ptr,
    dt_bias_ptr,
    norm_weight_ptr,
    ssm_state_ptr,
    cu_seqlens_ptr,
    state_indices_ptr,
    num_accepted_ptr,
    out_ptr,
    stride_qkvz_row,
    stride_ba_row,
    stride_out_row,
    stride_conv_w_row,
    stride_conv_slot,
    stride_conv_dim,
    stride_conv_tok,
    stride_ssm_slot,
    stride_indices_row,
    num_indices_per_row,
    scale,
    norm_eps,
    H: tl.constexpr,
    HV: tl.constexpr,
    RATIO: tl.constexpr,  # HV // H
    K: tl.constexpr,
    V: tl.constexpr,
    KEY: tl.constexpr,  # H * K
    VAL: tl.constexpr,  # HV * V
    CONV_W: tl.constexpr,
    HAS_CONV_BIAS: tl.constexpr,
    SIGMOID_GATE: tl.constexpr,
):
    i_n, i_hv = tl.program_id(0), tl.program_id(1)
    i_h = i_hv // RATIO
    is_qk_owner = i_hv % RATIO == 0

    bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int64)
    num_tokens = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int64) - bos
    if num_tokens <= 0:
        return

    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    p_out = out_ptr + bos * stride_out_row + i_hv * V + o_v

    conv_slot = tl.load(state_indices_ptr + i_n * stride_indices_row).to(tl.int64)
    accepted = tl.load(num_accepted_ptr + i_n).to(tl.int64)
    bad = (conv_slot <= 0) | (accepted <= 0) | (accepted > num_indices_per_row)
    if bad:
        zero_v = tl.zeros([V], dtype=out_ptr.dtype.element_ty)
        for t in range(num_tokens):
            tl.store(p_out + t * stride_out_row, zero_v)
        return

    ssm_src = tl.load(state_indices_ptr + i_n * stride_indices_row + accepted - 1).to(
        tl.int64
    )
    if ssm_src <= 0:
        zero_v = tl.zeros([V], dtype=out_ptr.dtype.element_ty)
        for t in range(num_tokens):
            tl.store(p_out + t * stride_out_row, zero_v)
        return

    offset = accepted - 1

    def _conv_base(ch_off):
        return (
            conv_state_ptr
            + conv_slot * stride_conv_slot
            + ch_off * stride_conv_dim
        )

    # STEP 1: seed the width-4 FIR's CONV_W - 1 = 3 taps from the rolled-back
    # window (bit-for-bit the `_causal_conv1d_update_kernel` IS_SPEC_DECODING
    # read: `conv_state[offset + j]` for j in [0, CONV_W - 1)).
    q_base = _conv_base(i_h * K + o_k)
    k_base = _conv_base(KEY + i_h * K + o_k)
    v_base = _conv_base(2 * KEY + i_hv * V + o_v)

    q_t0 = tl.load(q_base + (offset + 0) * stride_conv_tok)
    q_t1 = tl.load(q_base + (offset + 1) * stride_conv_tok)
    q_t2 = tl.load(q_base + (offset + 2) * stride_conv_tok)
    k_t0 = tl.load(k_base + (offset + 0) * stride_conv_tok)
    k_t1 = tl.load(k_base + (offset + 1) * stride_conv_tok)
    k_t2 = tl.load(k_base + (offset + 2) * stride_conv_tok)
    v_t0 = tl.load(v_base + (offset + 0) * stride_conv_tok)
    v_t1 = tl.load(v_base + (offset + 1) * stride_conv_tok)
    v_t2 = tl.load(v_base + (offset + 2) * stride_conv_tok)
    # The window this step writes back keeps only the newest CONV_W - 2 = 2
    # of these taps (the oldest, _t0, is dropped) ahead of the new tokens.
    q_hist1, q_hist2 = q_t1, q_t2
    k_hist1, k_hist2 = k_t1, k_t2
    v_hist1, v_hist2 = v_t1, v_t2

    q_wbase = (i_h * K + o_k) * stride_conv_w_row
    k_wbase = (KEY + i_h * K + o_k) * stride_conv_w_row
    v_wbase = (2 * KEY + i_hv * V + o_v) * stride_conv_w_row
    q_w0 = tl.load(conv_weight_ptr + q_wbase + 0)
    q_w1 = tl.load(conv_weight_ptr + q_wbase + 1)
    q_w2 = tl.load(conv_weight_ptr + q_wbase + 2)
    q_w3 = tl.load(conv_weight_ptr + q_wbase + 3)
    k_w0 = tl.load(conv_weight_ptr + k_wbase + 0)
    k_w1 = tl.load(conv_weight_ptr + k_wbase + 1)
    k_w2 = tl.load(conv_weight_ptr + k_wbase + 2)
    k_w3 = tl.load(conv_weight_ptr + k_wbase + 3)
    v_w0 = tl.load(conv_weight_ptr + v_wbase + 0)
    v_w1 = tl.load(conv_weight_ptr + v_wbase + 1)
    v_w2 = tl.load(conv_weight_ptr + v_wbase + 2)
    v_w3 = tl.load(conv_weight_ptr + v_wbase + 3)

    if HAS_CONV_BIAS:
        q_bias = tl.load(conv_bias_ptr + i_h * K + o_k).to(tl.float32)
        k_bias = tl.load(conv_bias_ptr + KEY + i_h * K + o_k).to(tl.float32)
        v_bias = tl.load(conv_bias_ptr + 2 * KEY + i_hv * V + o_v).to(tl.float32)
    else:
        q_bias = tl.zeros([K], dtype=tl.float32)
        k_bias = tl.zeros([K], dtype=tl.float32)
        v_bias = tl.zeros([V], dtype=tl.float32)

    neg_A = -tl.exp(tl.load(A_log_ptr + i_hv).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
    norm_w = tl.load(norm_weight_ptr + o_v).to(tl.float32)

    b_h = tl.zeros([V, K], dtype=tl.float32)
    p_ssm = ssm_state_ptr + ssm_src * stride_ssm_slot + i_hv * V * K
    b_h += tl.load(p_ssm + o_v[:, None] * K + o_k[None, :]).to(tl.float32)

    for t in range(num_tokens):
        row = bos + t
        p_qkvz = qkvz_ptr + row * stride_qkvz_row
        x_q = tl.load(p_qkvz + i_h * K + o_k)
        x_k = tl.load(p_qkvz + KEY + i_h * K + o_k)
        x_v = tl.load(p_qkvz + 2 * KEY + i_hv * V + o_v)

        # Width-4 causal conv: per-tap products stay in the loaded (bf16)
        # dtype before the fp32 accumulate, matching
        # `_causal_conv1d_update_kernel`'s rounding exactly.
        acc_q = q_bias + (q_t0 * q_w0) + (q_t1 * q_w1) + (q_t2 * q_w2) + (x_q * q_w3)
        acc_k = k_bias + (k_t0 * k_w0) + (k_t1 * k_w1) + (k_t2 * k_w2) + (x_k * k_w3)
        acc_v = v_bias + (v_t0 * v_w0) + (v_t1 * v_w1) + (v_t2 * v_w2) + (x_v * v_w3)
        acc_q = acc_q / (1.0 + tl.exp(-acc_q))
        acc_k = acc_k / (1.0 + tl.exp(-acc_k))
        acc_v = acc_v / (1.0 + tl.exp(-acc_v))
        conv_q = acc_q.to(out_ptr.dtype.element_ty)
        conv_k = acc_k.to(out_ptr.dtype.element_ty)
        conv_v = acc_v.to(out_ptr.dtype.element_ty)
        # The FIR's history is the *raw* (pre-conv) input sequence, never the
        # activated output -- shift in x_q/x_k/x_v, matching stock's
        # `col2 = matrix_x` (the freshly-loaded raw sample, not `acc`).
        q_t0, q_t1, q_t2 = q_t1, q_t2, x_q
        k_t0, k_t1, k_t2 = k_t1, k_t2, x_k
        v_t0, v_t1, v_t2 = v_t1, v_t2, x_v

        b_q = conv_q.to(tl.float32)
        b_k = conv_k.to(tl.float32)
        b_v = conv_v.to(tl.float32)

        x = tl.load(ba_ptr + row * stride_ba_row + HV + i_hv).to(tl.float32) + dt_bias
        decay = tl.exp(neg_A * tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x))
        beta = tl.sigmoid(tl.load(ba_ptr + row * stride_ba_row + i_hv).to(tl.float32))

        b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6) * scale
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_h *= decay
        b_v = (b_v - tl.sum(b_h * b_k[None, :], 1)) * beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1).to(out_ptr.dtype.element_ty).to(tl.float32)

        z = tl.load(p_qkvz + 2 * KEY + VAL + i_hv * V + o_v).to(tl.float32)
        gate = tl.sigmoid(z) if SIGMOID_GATE else z * tl.sigmoid(z)
        rstd = tl.rsqrt(tl.sum(b_o * b_o) / V + norm_eps)
        y = b_o * rstd * norm_w * gate
        tl.store(p_out + t * stride_out_row, y.to(out_ptr.dtype.element_ty))

        dst = tl.load(state_indices_ptr + i_n * stride_indices_row + t).to(tl.int64)
        if dst > 0:
            p_dst = ssm_state_ptr + dst * stride_ssm_slot + i_hv * V * K
            tl.store(p_dst + o_v[:, None] * K + o_k[None, :], b_h.to(ssm_state_ptr.dtype.element_ty))

    # Write-back window: the first CONV_W - 2 = 2 positions are the dropped-
    # oldest-tap history (a fixed, model-constant count -- independent of
    # this request's token count), followed by exactly `num_tokens` (a
    # runtime value) new raw (pre-conv) inputs. CONV_W - 2 is a Python int
    # (CONV_W is constexpr), so these two history stores are unrolled; the
    # new-token part is a genuine dynamic-trip-count loop, so it must address
    # memory purely through pointer arithmetic -- not Python-level tuple
    # indexing by the runtime loop variable.
    if is_qk_owner:
        tl.store(q_base + 0 * stride_conv_tok, q_hist1)
        tl.store(q_base + 1 * stride_conv_tok, q_hist2)
        tl.store(k_base + 0 * stride_conv_tok, k_hist1)
        tl.store(k_base + 1 * stride_conv_tok, k_hist2)
    tl.store(v_base + 0 * stride_conv_tok, v_hist1)
    tl.store(v_base + 1 * stride_conv_tok, v_hist2)

    for t in range(num_tokens):
        row = bos + t
        p_qkvz = qkvz_ptr + row * stride_qkvz_row
        pos = (CONV_W - 2) + t
        if is_qk_owner:
            q_val = tl.load(p_qkvz + i_h * K + o_k)
            k_val = tl.load(p_qkvz + KEY + i_h * K + o_k)
            tl.store(q_base + pos * stride_conv_tok, q_val)
            tl.store(k_base + pos * stride_conv_tok, k_val)
        v_val = tl.load(p_qkvz + 2 * KEY + i_hv * V + o_v)
        tl.store(v_base + pos * stride_conv_tok, v_val)


# This kernel's register-level FIR is hand-unrolled for a width-4 conv (three
# taps plus the new sample); a different kernel width needs a rewrite of the
# tap/history registers above, not just a different constexpr value.
_CONV_W = 4
# ``MAX_FUSED_GDN_VERIFY_ROWS``: the mono path's verify steps are capped at
# ``MAX_TOKENS`` rows (<= 8, ``vllm/models/qwen4_exp/amd/mono/layout.py``);
# reuse that as this kernel's own eligibility cap too, since nothing about the
# kernel itself needs a tighter limit.
MAX_FUSED_GDN_VERIFY_ROWS = 8


def fused_gdn_verify_core(
    *,
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    norm_activation: str,
    ssm_state: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    spec_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> None:
    """Fused spec-decode GDN verify core: writes ``core_attn_out`` (the
    out_proj input) directly from the packed in_proj outputs.

    ``qkvz``: ``[n, 2*key_dim + 2*value_dim]`` packed ``[q|k|v|z]`` (the raw
    ``in_proj_qkvz`` output; read-only, never mutated -- unlike the stock
    path, which conv-updates the qkv columns of this same buffer in place).
    ``ba``: ``[n, 2*num_v_heads]`` packed ``[b|a]``.
    ``core_attn_out``: ``[n, num_v_heads, head_v_dim]``, fully overwritten
    (every row gets a value: a real output, or zero for a padding row).
    ``conv_state``: oriented to a logical ``(slots, dim, state_len)`` view,
    same as ``causal_conv1d_update``'s ``conv_state`` argument (the caller
    picks the view, e.g. ``self.kv_cache[0] if is_conv_state_dim_first() else
    self.kv_cache[0].transpose(-1, -2)``).
    ``ssm_state``: ``[slots, num_v_heads, head_v_dim, head_k_dim]``.
    ``spec_query_start_loc``: cu_seqlens, ``[n_req + 1]``.
    ``spec_state_indices``: ``[n_req, num_spec + 1]``; column 0 is this
    request's single conv-state slot, column ``t`` is the SSM-state slot
    written after processing this step's ``t``-th token (``<= 0`` means no
    slot / don't store).
    ``num_accepted_tokens``: ``[n_req]``, how many draft positions of the
    *previous* verify cycle were accepted -- selects both the conv window's
    rewind offset and the SSM checkpoint-read column (``accepted - 1``).
    """
    assert conv_weight.shape[-1] == _CONV_W, (
        f"fused_gdn_verify_core is hand-unrolled for conv width {_CONV_W}, "
        f"got {conv_weight.shape[-1]}"
    )
    assert num_v_heads % num_k_heads == 0
    assert norm_activation in ("sigmoid", "silu", "swish")
    n_req = spec_query_start_loc.numel() - 1
    if n_req == 0:
        return
    ratio = num_v_heads // num_k_heads
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim

    grid = (n_req, num_v_heads)
    _fused_gdn_verify_core_kernel[grid](
        qkvz,
        ba,
        conv_weight,
        conv_bias,
        conv_state,
        A_log,
        dt_bias,
        norm_weight,
        ssm_state,
        spec_query_start_loc,
        spec_state_indices,
        num_accepted_tokens,
        core_attn_out,
        qkvz.stride(0),
        ba.stride(0),
        core_attn_out.stride(0),
        conv_weight.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        ssm_state.stride(0),
        spec_state_indices.stride(0),
        spec_state_indices.shape[1],
        head_k_dim**-0.5,
        norm_eps,
        H=num_k_heads,
        HV=num_v_heads,
        RATIO=ratio,
        K=head_k_dim,
        V=head_v_dim,
        KEY=key_dim,
        VAL=value_dim,
        CONV_W=_CONV_W,
        HAS_CONV_BIAS=conv_bias is not None,
        SIGMOID_GATE=norm_activation == "sigmoid",
        num_warps=4,
    )


def gdn_verify_eligible(attn, attn_metadata) -> bool:
    """Whether ``attn``'s verify step can use ``fused_gdn_verify_core``: a
    pure spec-decode batch (no prefill/plain-decode rows mixed in, matching
    ``QwenGatedDeltaNetAttention._is_pure_spec_decode``), within this
    kernel's row cap, on the flat/no-bias/SiLU-conv/sigmoid-or-silu-gate
    shape the register-level unroll above assumes.
    """
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    if not isinstance(attn_metadata, GDNAttentionMetadata):
        return False
    md = attn_metadata
    return (
        md.spec_sequence_masks is not None
        and md.num_prefills == 0
        and md.num_decodes == 0
        and md.num_spec_decodes > 0
        and md.spec_query_start_loc is not None
        and md.spec_state_indices_tensor is not None
        and md.num_accepted_tokens is not None
        and md.num_actual_tokens <= MAX_FUSED_GDN_VERIFY_ROWS
        and attn.conv1d.bias is None
        and attn.activation == "silu"
        and attn.qkvz_layout == "flat"
        and attn.num_v_heads % attn.num_k_heads == 0
    )


def try_fused_gdn_verify_core(attn, x, qkvz: torch.Tensor, ba: torch.Tensor, core: torch.Tensor) -> bool:
    """Try the fused verify core for this GDN layer's current step.

    On success, ``core`` (``[n, num_v_heads, head_v_dim]``) is fully written
    and this returns True; the caller should skip the stock path entirely
    (no separate z/unpack buffers, no ``_gated_rms_norm`` call needed). On
    False, ``core`` is untouched and the caller must fall back to stock.
    """
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    forward_context = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata
    attn_metadata = None
    if isinstance(attn_metadata_raw, dict):
        attn_metadata = attn_metadata_raw.get(attn.prefix)
    if not gdn_verify_eligible(attn, attn_metadata):
        return False

    conv_state = (
        attn.kv_cache[0]
        if is_conv_state_dim_first()
        else attn.kv_cache[0].transpose(-1, -2)
    )
    fused_gdn_verify_core(
        qkvz=qkvz,
        ba=ba,
        core_attn_out=core,
        conv_state=conv_state,
        conv_weight=attn.conv1d.weight.view(
            attn.conv1d.weight.size(0), attn.conv1d.weight.size(2)
        ),
        conv_bias=attn.conv1d.bias,
        A_log=attn.A_log,
        dt_bias=attn.dt_bias,
        norm_weight=attn.norm.weight,
        norm_eps=attn.norm.eps,
        norm_activation=attn.norm.activation,
        ssm_state=attn.kv_cache[1],
        spec_query_start_loc=attn_metadata.spec_query_start_loc,
        spec_state_indices=attn_metadata.spec_state_indices_tensor,
        num_accepted_tokens=attn_metadata.num_accepted_tokens,
        num_k_heads=attn.num_k_heads // attn.tp_size,
        num_v_heads=attn.num_v_heads // attn.tp_size,
        head_k_dim=attn.head_k_dim,
        head_v_dim=attn.head_v_dim,
    )
    return True
