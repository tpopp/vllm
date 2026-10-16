# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the fused GDN verify kernel against the stock spec-decode verify
chain, at Qwen3.8-Flash-Next's TP2 per-rank shape (NK=8, NV=24, head dim 128,
conv width 4, MTP k=3 -> up to 4 verify tokens/request).

Single-GPU: a GDN layer's verify core is a per-rank computation (no TP
all-reduce in this part of the layer), so this needs no torchrun/distributed
setup, unlike ``bench_qwen4exp_mono_draft.py``.

Usage:
    python benchmarks/kernels/bench_gdn_verify_fused.py [--n-req 1,2,4,8]
        [--iters 200]

Per n_req (each request a full 4-token verify window -- the mono decode
path's own n_req in {1, 2} is the <= MAX_TOKENS=8-row window this kernel is
wired for; n_req in {4, 8} (16, 32 rows) are included for reference only,
beyond what the mono path itself ever calls this with), prints eager and
CUDA-graph-replayed median-of-5 per-call microseconds for:
(1) stock's real ROCm fast path (``gdn_spec_decode_packed_aiter``, AITER's
``fused_rearrange_sigmoid_gated_delta_rule`` + the Triton conv1d update +
unpack), skipped if AITER isn't installed;
(2) the same chain built from in-tree-only kernels (``causal_conv1d_update``
+ the non-AITER ``fused_sigmoid_gating_delta_rule_update`` + unpack +
``RMSNormGated``), always available, so this benchmark still runs without
AITER;
(3) ``fused_gdn_verify_core``, one launch.

Needs MI355X (gfx950) and Triton; this file is authored CPU-only and not run
here.
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

bf = torch.bfloat16
DEVICE = "cuda"
WARMUP = 3
REPEATS = 5
GRAPH_LAUNCHES = 20

NUM_SPEC = 3
DRAFT_WINDOW = NUM_SPEC + 1  # 4
NK, NV, HD = 8, 24, 128
CONV_W = 4
NORM_EPS = 1e-6
NORM_ACTIVATION = "sigmoid"  # Flash-Next's GDN output gate


# ---------------------------------------------------------------- timing
def _time_eager(fn, iters: int):
    try:
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
    except Exception as exc:  # robustness: report, let the caller move on
        return None, repr(exc)
    samples = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iters)
    return statistics.median(samples), None


def _time_graph(fn, iters: int):
    try:
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(GRAPH_LAUNCHES):
                fn()
        torch.cuda.synchronize()
    except Exception as exc:
        return None, repr(exc)
    replays = max(1, iters // GRAPH_LAUNCHES)
    samples = []
    try:
        for _ in range(REPEATS):
            torch.cuda.synchronize()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / (replays * GRAPH_LAUNCHES))
    except Exception as exc:
        return None, repr(exc)
    return statistics.median(samples), None


# ---------------------------------------------------------------- fixtures
def _build_inputs(n_req: int, seed: int):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    num_tokens = n_req * DRAFT_WINDOW
    key_dim, value_dim = NK * HD, NV * HD
    conv_dim = 2 * key_dim + value_dim
    num_slots = n_req * DRAFT_WINDOW + 8

    qsl = torch.arange(0, num_tokens + 1, DRAFT_WINDOW, dtype=torch.int32, device=DEVICE)
    nacc = torch.randint(
        1, DRAFT_WINDOW + 1, (n_req,), dtype=torch.int32, device=DEVICE, generator=g
    )
    idx = (
        torch.randperm(num_slots - 1, device=DEVICE, generator=g)[
            : n_req * DRAFT_WINDOW
        ]
        + 1
    ).view(n_req, DRAFT_WINDOW).to(torch.int32)

    qkvz = torch.randn(
        num_tokens, conv_dim + value_dim, device=DEVICE, dtype=bf, generator=g
    )
    ba = torch.randn(num_tokens, 2 * NV, device=DEVICE, dtype=bf, generator=g)
    conv_weight = (
        torch.randn(conv_dim, CONV_W, device=DEVICE, dtype=bf, generator=g) * 0.3
    )
    conv_state = torch.randn(
        num_slots, CONV_W - 1 + NUM_SPEC, conv_dim, device=DEVICE, dtype=bf,
        generator=g,
    ).transpose(-1, -2)
    ssm_state = (
        torch.randn(num_slots, NV, HD, HD, device=DEVICE, generator=g) * 0.05
    )
    A_log = torch.randn(NV, device=DEVICE, generator=g)
    dt_bias = torch.randn(NV, device=DEVICE, dtype=bf, generator=g)
    norm_weight = torch.rand(HD, device=DEVICE, dtype=bf, generator=g) + 0.5
    return dict(
        qsl=qsl, nacc=nacc, idx=idx, qkvz=qkvz, ba=ba, conv_weight=conv_weight,
        conv_state=conv_state, ssm_state=ssm_state, A_log=A_log, dt_bias=dt_bias,
        norm_weight=norm_weight, num_tokens=num_tokens, n_req=n_req,
        key_dim=key_dim, value_dim=value_dim,
    )


# ---------------------------------------------------------------- cases
def _bench_aiter(fx, iters: int):
    from vllm.model_executor.layers.mamba.ops.gdn_spec_decode_rocm import (
        aiter_gdr_supports_spec_decode,
        gdn_spec_decode_packed_aiter,
    )

    if not aiter_gdr_supports_spec_decode():
        msg = "AITER's GDR kernel doesn't support spec decode here"
        return (None, msg), (None, msg)

    qkvz, ba = fx["qkvz"].clone(), fx["ba"].clone()
    z_out = torch.empty(fx["num_tokens"], NV, HD, device=DEVICE, dtype=bf)
    core = torch.empty(fx["num_tokens"], NV, HD, device=DEVICE, dtype=bf)
    conv_state = fx["conv_state"].clone(memory_format=torch.preserve_format)
    ssm_state = fx["ssm_state"].clone()

    def _fn():
        gdn_spec_decode_packed_aiter(
            qkvz=qkvz, ba=ba, z_out=z_out, core_attn_out=core,
            conv_state=conv_state, conv_weights=fx["conv_weight"], conv_bias=None,
            activation="silu", A_log=fx["A_log"], dt_bias=fx["dt_bias"],
            ssm_state=ssm_state, spec_query_start_loc=fx["qsl"],
            spec_state_indices=fx["idx"], num_accepted_tokens=fx["nacc"],
            num_spec_decodes=fx["n_req"], num_actual_tokens=fx["num_tokens"],
            key_dim=fx["key_dim"], value_dim=fx["value_dim"],
            head_k_dim=HD, head_v_dim=HD, draft_window=DRAFT_WINDOW,
        )
        from vllm.models.qwen3_5.amd.mono_base import _gated_rms_norm

        _gated_rms_norm(core, z_out, fx["norm_weight"], NORM_EPS, NORM_ACTIVATION)

    try:
        _fn()
    except Exception as exc:
        return (None, repr(exc)), (None, repr(exc))
    return _time_eager(_fn, iters), _time_graph(_fn, iters)


def _bench_generic(fx, iters: int):
    from vllm.model_executor.layers.layernorm import RMSNormGated
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.model_executor.layers.mamba.ops.gdn_spec_decode_rocm import (
        gdn_unpack_z_ba,
    )
    from vllm.third_party.flash_linear_attention.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )

    qkvz, ba = fx["qkvz"].clone(), fx["ba"].clone()
    qkv_size = fx["key_dim"] * 2 + fx["value_dim"]
    z_out = torch.empty(fx["num_tokens"], NV, HD, device=DEVICE, dtype=bf)
    conv_state = fx["conv_state"].clone(memory_format=torch.preserve_format)
    ssm_state = fx["ssm_state"].clone()
    n_req = fx["n_req"]

    def _fn():
        b, a = gdn_unpack_z_ba(qkvz, ba, z_out, qkv_size, NV)
        mixed_qkv = causal_conv1d_update(
            qkvz[:, :qkv_size].clone(), conv_state, fx["conv_weight"], None, "silu",
            conv_state_indices=fx["idx"][:, 0], num_accepted_tokens=fx["nacc"],
            query_start_loc=fx["qsl"], max_query_len=DRAFT_WINDOW,
            validate_data=False,
        )
        q, k, v = mixed_qkv.split([fx["key_dim"], fx["key_dim"], fx["value_dim"]], -1)
        t = mixed_qkv.shape[0]
        o, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=fx["A_log"], a=a, b=b, dt_bias=fx["dt_bias"],
            q=q.contiguous().view(1, t, NK, HD), k=k.contiguous().view(1, t, NK, HD),
            v=v.contiguous().view(1, t, NV, HD), initial_state=ssm_state,
            inplace_final_state=True, cu_seqlens=fx["qsl"][: n_req + 1],
            ssm_state_indices=fx["idx"], num_accepted_tokens=fx["nacc"],
            use_qk_l2norm_in_kernel=True,
        )
        core = o.squeeze(0)
        RMSNormGated.forward_static(
            core, z_out, fx["norm_weight"], NORM_EPS, core.dtype,
            group_size=None, norm_before_gate=True, activation=NORM_ACTIVATION,
        )

    try:
        _fn()
    except Exception as exc:
        return (None, repr(exc)), (None, repr(exc))
    return _time_eager(_fn, iters), _time_graph(_fn, iters)


def _bench_fused(fx, iters: int, bv=None, nw=None):
    from vllm.model_executor.layers.mamba.ops.gdn_spec_decode_mono_fused import (
        fused_gdn_verify_core,
    )

    qkvz, ba = fx["qkvz"].clone(), fx["ba"].clone()
    core = torch.empty(fx["num_tokens"], NV, HD, device=DEVICE, dtype=bf)
    conv_state = fx["conv_state"].clone(memory_format=torch.preserve_format)
    ssm_state = fx["ssm_state"].clone()

    def _fn():
        fused_gdn_verify_core(
            qkvz=qkvz, ba=ba, core_attn_out=core, conv_state=conv_state,
            conv_weight=fx["conv_weight"], conv_bias=None, A_log=fx["A_log"],
            dt_bias=fx["dt_bias"], norm_weight=fx["norm_weight"], norm_eps=NORM_EPS,
            norm_activation=NORM_ACTIVATION, ssm_state=ssm_state,
            spec_query_start_loc=fx["qsl"], spec_state_indices=fx["idx"],
            num_accepted_tokens=fx["nacc"], num_k_heads=NK, num_v_heads=NV,
            head_k_dim=HD, head_v_dim=HD, bv=bv, num_warps=nw,
        )

    try:
        _fn()
    except Exception as exc:
        return (None, repr(exc)), (None, repr(exc))
    return _time_eager(_fn, iters), _time_graph(_fn, iters)


# ---------------------------------------------------------------- plumbing
def _fmt(result) -> str:
    val, err = result
    if val is not None:
        return f"{val:9.2f} us"
    return f"FAILED ({str(err)[:70]})"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-req", type=str, default="1,2,4,8")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--sweep", type=str, default="32:4", help="BV:num_warps,...")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(0)
    for n_req in [int(v) for v in args.n_req.split(",") if v]:
        print(f"\n=== n_req={n_req} (rows={n_req * DRAFT_WINDOW}) ===")
        fx = _build_inputs(n_req, seed=1000 + n_req)
        cases = (
            ("stock: AITER GDR fast path (gdn_spec_decode_packed_aiter)", _bench_aiter),
            ("stock: in-tree-only chain (causal_conv1d_update + FLA GDR)", _bench_generic),
        ) + tuple(
            (f"fused_gdn_verify_core BV={bv} warps={nw}",
             lambda fx, it, bv=bv, nw=nw: _bench_fused(fx, it, bv, nw))
            for bv, nw in (tuple(map(int, c.split(":"))) for c in args.sweep.split(","))
        )
        for name, fn in cases:
            eager, graph = fn(fx, args.iters)
            print(f"  {name:<58s} eager {_fmt(eager)}  graph {_fmt(graph)}")


if __name__ == "__main__":
    main()
