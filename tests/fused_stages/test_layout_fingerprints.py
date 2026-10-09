# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout tags must pin bytes.

Each case pushes a fixed pseudo-random weight through the *real* processing
function that produces a tag and compares a digest of the result with the
digest recorded for that tag. A processing change that alters the bytes of a
tagged layout fails here until the tag version is bumped (``@1`` -> ``@2``)
and the new digest recorded, which is exactly when fused-stage providers
consuming the old layout need to know.
"""

import pytest
import torch
from torch import nn

from vllm.fused_stages import layouts

FINGERPRINTS = {
    # tag -> (weight digest, scale digest); record with `pytest --record`.
    layouts.MXFP8_ROWMAJOR_E8M0_32x32: ("7cd1e29acf57b5d4", "339aeecc7d2dc804"),
    layouts.MXFP4_AITER_A4W4_SEPARATED: None,
}


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def mxfp8_block32_layer() -> nn.Module:
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        RocmDotScaledMxfp8LinearKernel,
    )

    n, k = 64, 128
    layer = nn.Module()
    w = torch.randn(n, k, generator=_gen(0)).to(torch.float8_e4m3fn)
    blocks = torch.randint(118, 130, (n // 32, k // 32), generator=_gen(1))
    scale = (
        blocks.repeat_interleave(32, dim=0).to(torch.uint8).view(torch.float8_e8m0fnu)
    )
    layer.weight = nn.Parameter(w, requires_grad=False)
    layer.weight_scale = nn.Parameter(scale, requires_grad=False)
    kernel = object.__new__(RocmDotScaledMxfp8LinearKernel)  # stateless processing
    RocmDotScaledMxfp8LinearKernel.process_weights_after_loading(kernel, layer)
    return layer


def test_mxfp8_block32_tag_and_bytes(request):
    layer = mxfp8_block32_layer()
    assert layouts.layout_of(layer.weight) == layouts.MXFP8_ROWMAJOR_E8M0_32x32
    assert layouts.layout_of(layer.weight_scale) == layouts.SCALE_FOLLOWS_WEIGHT
    got = (layouts.fingerprint(layer.weight), layouts.fingerprint(layer.weight_scale))
    expected = FINGERPRINTS[layouts.MXFP8_ROWMAJOR_E8M0_32x32]
    if expected is None:
        pytest.xfail(f"fingerprint not recorded yet: {got}")
    assert got == expected, "layout bytes changed: bump the tag version"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AITER shuffles need a ROCm GPU"
)
def test_mxfp4_a4w4_separated_tag_and_bytes():
    pytest.importorskip("aiter")
    from aiter.ops.shuffle import shuffle_scale, shuffle_weight

    # Same calls as convert_weight_to_mxfp4_moe_kernel_format's a4w4 branch.
    e, n, k = 4, 256, 512
    w = torch.randint(0, 255, (e, n, k // 2), generator=_gen(2), dtype=torch.uint8)
    s = torch.randint(118, 130, (e, n, k // 32), generator=_gen(3), dtype=torch.uint8)
    w = shuffle_weight(
        w.cuda().view(torch.float4_e2m1fn_x2), is_guinterleave=False, gate_up=True
    )
    s = shuffle_scale(s.cuda().view(-1, k // 32), e, False, True)
    got = (layouts.fingerprint(w), layouts.fingerprint(s))
    expected = FINGERPRINTS[layouts.MXFP4_AITER_A4W4_SEPARATED]
    if expected is None:
        pytest.xfail(f"fingerprint not recorded yet: {got}")
    assert got == expected, "layout bytes changed: bump the tag version"


def test_every_tag_registered_and_versioned():
    for tag in layouts.KNOWN_LAYOUTS:
        name, _, version = tag.partition("@")
        assert name and version.isdigit(), tag
