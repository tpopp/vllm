# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Processed-weight layout tags.

A quantization method or linear/MoE kernel that rewrites a weight in
``process_weights_after_loading`` (shuffles, preshuffles, transposes, reorders
scales, dequantizes...) tags the resulting parameter with the layout it
produced. Fused-stage providers read the tag instead of inferring the layout
from ``is_shuffled`` flags, kernel class names or environment variables.

Tags are ``"<format>.<arrangement>@<version>"``. Rules:

* A change that alters the bytes of a tagged layout MUST bump the version.
  ``tests/fused_stages/test_layout_fingerprints.py`` pins a fingerprint of
  every tag (a hash of a fixed random weight pushed through the real
  processing function), so an unbumped change fails CI.
* A layout without a tag is ``UNTAGGED``; providers must refuse it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

LAYOUT_ATTR = "vllm_layout"
UNTAGGED = "untagged@0"


@dataclass(frozen=True)
class LayoutTag:
    tag: str
    description: str


# Registry of known tags. Add a tag here (with a description precise enough
# to implement a reader from) when a processing function starts producing it.
KNOWN_LAYOUTS: dict[str, LayoutTag] = {}


def _register(tag: str, description: str) -> str:
    KNOWN_LAYOUTS[tag] = LayoutTag(tag, description)
    return tag


# ---- unquantized
BF16_ROWMAJOR = _register("bf16.rowmajor@1", "[N, K] bf16/fp32, row-major")
F32_ROWMAJOR = _register("f32.rowmajor@1", "[...] float32, row-major")

# ---- MXFP8 linears (vllm/model_executor/kernels/linear/mxfp8/rocm_native.py)
MXFP8_ROWMAJOR_E8M0_32x32 = _register(
    "mxfp8.rowmajor.e8m0_32x32@1",
    "[N, K] float8_e4m3fn row-major; scale [N/32, K/32] e8m0, one per 32x32 "
    "block (DeepSeek V4/V4.1 checkpoints)",
)
MXFP8_ROWMAJOR_E8M0_1x32 = _register(
    "mxfp8.rowmajor.e8m0_1x32@1",
    "[N, K] float8_e4m3fn row-major; scale [N, K/32] e8m0, one per row and "
    "32 K elements",
)
MXFP8_DEQUANT_BF16 = _register(
    "mxfp8.dequantized_bf16@1",
    "[N, K] bf16: MXFP8 checkpoint dequantized at load (K not aligned for "
    "the dot-scaled kernel); no scale",
)

# ---- block-FP8 linears preshuffled for AITER (DSv4.1 attention fast path)
FP8_BLOCK128_BPRESHUFFLE16 = _register(
    "fp8.block128.aiter_bpreshuffle_16x16@1",
    "[N, K] float8 shuffled by aiter.shuffle_weight(layout=(16, 16)); scale "
    "[N/128, K/128] fp32 (upcast from e8m0 if needed), not shuffled",
)

# ---- MXFP4 MoE experts (vllm/model_executor/layers/fused_moe/oracle/mxfp4.py)
MXFP4_AITER_A16W4_SITUV2_SEPARATED = _register(
    "mxfp4.aiter_a16w4_situv2.separated@1",
    "w13/w2 fp4x2 shuffled by rocm_aiter_ops.shuffle_weight_a16w4(16, False); "
    "w13 scale shuffle_scale_a16w4(..., False), w2 scale e8m0_shuffle",
)
MXFP4_AITER_A16W4_SITUV2_INTERLEAVED = _register(
    "mxfp4.aiter_a16w4_situv2.interleaved@1",
    "as mxfp4.aiter_a16w4_situv2.separated@1 with gate/up interleaved "
    "(rocm_aiter_ops.is_fused_moe_situv2_gate_up_interleaved())",
)
SCALE_FOLLOWS_WEIGHT = _register(
    "scale.follows_weight@1",
    "a scale tensor whose arrangement is defined by its weight's tag",
)
MXFP4_AITER_A4W4_SEPARATED = _register(
    "mxfp4.aiter_a4w4.separated@1",
    "w13/w2 fp4x2 shuffled by aiter.ops.shuffle.shuffle_weight("
    "is_guinterleave=False); scales aiter shuffle_scale(..., False). "
    "Gate and up halves separated (DeepSeek V4.1 a4w4, ATOM layout)",
)
MXFP4_AITER_A4W4_INTERLEAVED = _register(
    "mxfp4.aiter_a4w4.interleaved@1",
    "as mxfp4.aiter_a4w4.separated@1 but with gate/up interleaved",
)


def tag_layout(tensor: torch.Tensor, tag: str) -> torch.Tensor:
    """Attach a layout tag to a processed parameter (in place) and return it."""
    assert tag in KNOWN_LAYOUTS, f"unknown layout tag {tag}: register it first"
    setattr(tensor, LAYOUT_ATTR, tag)
    return tensor


def layout_of(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return UNTAGGED
    return getattr(tensor, LAYOUT_ATTR, UNTAGGED)


def fingerprint(tensor: torch.Tensor) -> str:
    """Stable digest of a tensor's bytes, for layout fingerprint tests."""
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]
