# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate experiment for the Qwen3.8-Flash-Next GDN verify "bracket" (P1 in
UTILIZATION_DESIGN.md #1): graph-replayed timing of K1's full decode build
(``k1.gdn_pre``, what P1b would widen to a verify variant) against the
``front_only`` build the spec verify path already runs (``k1.hc_front``), at
the production TP=2 shapes, plus the stock op chain that build would replace
(in_proj_qkvz's quant + ASM FP4 GEMM, in_proj_ba's bf16 GEMM, the unpack
Triton kernel, the gated RMSNorm), K1's ``proj_only`` build
(P1b, shipped: the HC front + in_proj_qkvz/ba, no GDN tasks) against
``hc_front`` + the stock quant+GEMM / bf16 GEMM / unpack chain it folds in
(the GDN verify core's conv + gated delta rule are untouched either way,
left to AITER -- §1 of gather/gdn_verify.md), and (P3) K1's ``verify``
build (the GDN verify core -- conv, gated delta rule, gated norm -- also
in-kernel) against ``proj_only`` chained into the stock calls it still
needs (``gdn_spec_decode_conv_gdr_aiter`` + ``_gated_rms_norm``): only
meaningful at S a multiple of ``k1.VERIFY_WINDOW`` (4), i.e. ``--s 4,8``.

Usage:
    torchrun --nproc-per-node 2 benchmarks/kernels/bench_qwen4exp_k1_gate.py \
        [--s 1,2,4,8] [--iters 200]

Decision rule (UTILIZATION_DESIGN.md #1): Delta(gdn_pre - hc_front) at S=4
<= 8 us builds the full P1b verify variant; >= 20 us kills it. Section 5
(P3) has no such threshold yet; report the eager and graph-replayed
deltas at S=4 and S=8 for the first real decision. This file is authored
CPU-only and not run here.
"""

import argparse
import os
import statistics
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

bf = torch.bfloat16
TP = 2
WARMUP = 3
REPEATS = 5
GRAPH_LAUNCHES = 20  # launches captured in one graph, replayed and divided


# ---------------------------------------------------------------- timing
# (identical to benchmarks/kernels/bench_qwen4exp_mono_draft.py's helpers)
def _time_eager(fn, iters: int):
    """Median of ``REPEATS`` repeats of ``iters`` eager launches: (us, None),
    or (None, error) if ``fn`` raised."""
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
    """``GRAPH_LAUNCHES`` launches captured once, replayed until >= ``iters``
    launches are timed, median of ``REPEATS`` repeats, divided back to a
    per-launch microsecond figure: graph replay is how production runs these
    kernels. (None, error) if capture or replay raised."""
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


# ---------------------------------------------------------------- 1. K1 gate
def bench_k1_gate(s, device, rank, world_size, epoch, iters):
    """``k1.gdn_pre`` (full decode K1: in_proj + the GDN tasks, what the
    non-spec decode path and P1b's widened verify variant both run) against
    ``k1.hc_front`` (``front_only``, the HC front alone: what the spec verify
    path runs today before calling vLLM's stock GDN core)."""
    from tests.models.qwen4_exp.test_amd_mono import _gdn_inputs
    from vllm.models.qwen4_exp.amd.mono import k1
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HIDDEN, Layout

    lay = Layout(TP)
    # every row live (n_real = s): the worst case, no cudagraph pad rows
    w = _gdn_inputs(s, s)
    conv_state, rstate = w["conv_state"].clone(), w["rstate"].clone()
    core = torch.empty(s, lay.CORE, dtype=bf, device=device)
    inj = torch.empty(s, HC, dtype=bf, device=device)
    x_bin = torch.empty(s, HIDDEN, dtype=bf, device=device)

    key_gdn = k1.K1Build(tokens=s, tp=TP, self_msq=True)
    scr_gdn = torch.zeros(k1.scratch_bytes(key_gdn), dtype=torch.uint8, device=device)

    def _gdn_pre():
        epoch.add_(1)
        k1.gdn_pre(
            key_gdn,
            state=w["state"],
            msq_in=None,
            hc_w=w["hc_w"],
            w_down=w["w_down"],
            w_up=w["w_up"],
            inj_out=inj,
            w_qkvz=w["w_qkvz"],
            w_qkvzs=w["w_qkvzs"],
            w_ba=w["w_ba"],
            conv_w=w["conv_w"],
            conv_state=conv_state,
            a_log=w["a_log"],
            dt_bias=w["dt_bias"],
            norm_w=w["norm_w"],
            rstate=rstate,
            st_idx=w["st_idx"],
            core=core,
            scratch=scr_gdn,
            epoch=epoch,
            layer=0,
        )

    key_front = k1.K1Build(tokens=s, tp=TP, self_msq=True, front_only=True)
    scr_front = torch.zeros(
        k1.scratch_bytes(key_front), dtype=torch.uint8, device=device
    )

    def _hc_front():
        epoch.add_(1)
        k1.hc_front(
            key_front,
            state=w["state"],
            msq_in=None,
            hc_w=w["hc_w"],
            w_down=w["w_down"],
            w_up=w["w_up"],
            inj_out=inj,
            bin_out=x_bin,
            scratch=scr_front,
            epoch=epoch,
            layer=1,
        )

    cases = (
        ("k1.gdn_pre (full decode K1: in_proj + GDN tasks)", _gdn_pre),
        ("k1.hc_front (front_only: the spec verify path's K1)", _hc_front),
    )
    return [
        (name, _time_eager(fn, iters), _time_graph(fn, iters)) for name, fn in cases
    ]


# -------------------------------------------------------- 2. the stock chain
def bench_stock_pieces(s, device, rank, world_size, epoch, iters):
    """The stock verify-core launches the P1 bracket would remove at this S:
    in_proj_qkvz's fp4 quant + ASM FP4 GEMM, in_proj_ba's bf16 GEMM,
    ``gdn_unpack_z_ba``, and ``_gated_rms_norm`` -- NOT conv1d / the fused
    gated delta rule (``gdn_spec_decode_conv_gdr_aiter``, P1b leaves those
    untouched, §1 of gather/gdn_verify.md)."""
    from tests.models.qwen3_5.test_amd_mono import _mxfp4_linear, _mxfp4_linear_stock
    from tests.models.qwen4_exp.test_amd_mono import _randn
    from vllm.model_executor.layers.mamba.ops.gdn_spec_decode_rocm import (
        gdn_unpack_z_ba,
    )
    from vllm.models.qwen3_5.amd.mono_base import _gated_rms_norm
    from vllm.models.qwen4_exp.amd.mono.layout import HD, HIDDEN, Layout

    lay = Layout(TP)
    g = torch.Generator(device=device).manual_seed(1000 + s)
    x = _randn(g, s, HIDDEN, scale=0.3)
    _, _, w_qkvz, w_qkvzs = _mxfp4_linear(g, lay.QKVZ, HIDDEN)
    w_ba = _randn(g, lay.BA, HIDDEN, scale=0.05)

    def _in_proj_qkvz():
        epoch.add_(1)
        _mxfp4_linear_stock(x, w_qkvz, w_qkvzs)

    def _in_proj_ba():
        epoch.add_(1)
        F.linear(x, w_ba)

    qkvz = _randn(g, s, lay.QKVZ)
    ba = _randn(g, s, lay.BA)
    z_out = torch.empty(s, lay.VAL, dtype=bf, device=device)

    def _unpack():
        epoch.add_(1)
        gdn_unpack_z_ba(qkvz, ba, z_out, lay.CONV, lay.NV)

    core_raw = _randn(g, s, lay.NV, HD)
    z_raw = _randn(g, s, lay.NV, HD, scale=2.0)
    norm_w = (1 + _randn(g, HD, scale=0.1, dtype=torch.float32)).to(bf)

    def _gated_norm():
        epoch.add_(1)
        _gated_rms_norm(core_raw, z_raw, norm_w, 1e-6, "sigmoid")

    cases = (
        ("in_proj_qkvz: per_1x32_f4_quant_hip + gemm_a4w4 (ASM FP4)", _in_proj_qkvz),
        ("in_proj_ba: F.linear (bf16)", _in_proj_ba),
        ("gdn_unpack_z_ba", _unpack),
        ("_gated_rms_norm (RMSNormGated.forward_static)", _gated_norm),
    )
    return [
        (name, _time_eager(fn, iters), _time_graph(fn, iters)) for name, fn in cases
    ]


# ------------------------------------------------------ 4. K1 proj_only (P1b)
def bench_k1_proj_gate(s, device, rank, world_size, epoch, iters):
    """P1b: K1's ``proj_only`` build (the HC front + in_proj_qkvz/ba, no GDN
    tasks) against ``hc_front`` + the stock in_proj chain a verify step runs
    today in its place (fp4 quant + ASM FP4 GEMM, bf16 in_proj_ba,
    ``gdn_unpack_z_ba``) -- section 2's pieces chained into one realistic
    replacement, ``hc_front``'s own x feeding the stock in_proj."""
    from tests.models.qwen3_5.test_amd_mono import _mxfp4_linear_stock
    from tests.models.qwen4_exp.test_amd_mono import _gdn_inputs
    from vllm.model_executor.layers.mamba.ops.gdn_spec_decode_rocm import (
        gdn_unpack_z_ba,
    )
    from vllm.models.qwen4_exp.amd.mono import k1
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HIDDEN, Layout

    lay = Layout(TP)
    w = _gdn_inputs(s, s)
    inj = torch.empty(s, HC, dtype=bf, device=device)
    x_bin = torch.empty(s, HIDDEN, dtype=bf, device=device)
    z_out = torch.empty(s, lay.VAL, dtype=bf, device=device)

    key_proj = k1.K1Build(tokens=s, tp=TP, self_msq=True, proj_only=True)
    scr_proj = torch.zeros(
        k1.scratch_bytes(key_proj), dtype=torch.uint8, device=device
    )
    qkvz_out = torch.empty(s, lay.QKVZ, dtype=bf, device=device)
    ba_out = torch.empty(2, s, lay.NV, dtype=bf, device=device)

    def _k1_proj():
        epoch.add_(1)
        k1.proj_pre(
            key_proj,
            state=w["state"],
            msq_in=None,
            hc_w=w["hc_w"],
            w_down=w["w_down"],
            w_up=w["w_up"],
            inj_out=inj,
            bin_out=x_bin,
            w_qkvz=w["w_qkvz"],
            w_qkvzs=w["w_qkvzs"],
            w_ba=w["w_ba"],
            qkvz_out=qkvz_out,
            ba_out=ba_out,
            scratch=scr_proj,
            epoch=epoch,
            layer=2,
        )

    key_front = k1.K1Build(tokens=s, tp=TP, self_msq=True, front_only=True)
    scr_front = torch.zeros(
        k1.scratch_bytes(key_front), dtype=torch.uint8, device=device
    )

    def _front_plus_stock():
        epoch.add_(1)
        k1.hc_front(
            key_front,
            state=w["state"],
            msq_in=None,
            hc_w=w["hc_w"],
            w_down=w["w_down"],
            w_up=w["w_up"],
            inj_out=inj,
            bin_out=x_bin,
            scratch=scr_front,
            epoch=epoch,
            layer=3,
        )
        stock_qkvz = _mxfp4_linear_stock(x_bin, w["w_qkvz"], w["w_qkvzs"])
        stock_ba = F.linear(x_bin, w["w_ba"])
        gdn_unpack_z_ba(stock_qkvz, stock_ba, z_out, lay.CONV, lay.NV)

    cases = (
        ("k1.proj_pre (P1b: HC front + in_proj_qkvz/ba, no GDN tasks)", _k1_proj),
        ("hc_front + stock quant+gemm + in_proj_ba + unpack", _front_plus_stock),
    )
    return [
        (name, _time_eager(fn, iters), _time_graph(fn, iters)) for name, fn in cases
    ]


# ------------------------------------------------------------- 5. K1 verify
def bench_k1_verify_gate(s, device, rank, world_size, epoch, iters):
    """P3: K1's ``verify`` build (the HC front + in_proj + the GDN verify
    core -- conv, gated delta rule, gated norm -- all in-kernel) against
    P1b's ``proj_pre`` (the HC front + in_proj alone) chained into the stock
    verify-core calls it still needs (``gdn_spec_decode_conv_gdr_aiter`` +
    ``_gated_rms_norm``) -- the pair the task asks to gate P3 by, at this
    deployment's only verify width (``k1.VERIFY_WINDOW`` tokens/request,
    ``n_req = s // k1.VERIFY_WINDOW`` live requests, no padding row-group:
    every request's state read/written, the stock chain's worst case).
    Only meaningful at ``s`` a multiple of ``k1.VERIFY_WINDOW`` (4): the
    production verify widths, S=4 (one request) and S=8 (two)."""
    from tests.models.qwen4_exp.test_amd_mono import _verify_inputs
    from vllm.model_executor.layers.mamba.ops.gdn_spec_decode_rocm import (
        gdn_spec_decode_conv_gdr_aiter,
    )
    from vllm.models.qwen3_5.amd.mono_base import _gated_rms_norm
    from vllm.models.qwen4_exp.amd.mono import k1
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HD, HIDDEN, Layout

    lay = Layout(TP)
    assert s % k1.VERIFY_WINDOW == 0, "P3 needs a multiple of k1.VERIFY_WINDOW rows"
    n_req = s // k1.VERIFY_WINDOW
    w, _, _, _ = _verify_inputs([k1.VERIFY_WINDOW] * n_req, pad_last=False)
    inj = torch.empty(s, HC, dtype=bf, device=device)
    qsl = torch.arange(0, s + 1, k1.VERIFY_WINDOW, dtype=torch.int32, device=device)

    key_verify = k1.K1Build(tokens=s, tp=TP, self_msq=True, verify=True)
    scr_verify = torch.zeros(
        k1.scratch_bytes(key_verify), dtype=torch.uint8, device=device
    )
    v_conv_state, v_rstate = w["conv_state"].clone(), w["rstate"].clone()
    v_core = torch.empty(s, lay.CORE, dtype=bf, device=device)

    def _k1_verify():
        epoch.add_(1)
        k1.verify_pre(
            key_verify,
            state=w["state"],
            msq_in=None,
            hc_w=w["hc_w"],
            w_down=w["w_down"],
            w_up=w["w_up"],
            inj_out=inj,
            w_qkvz=w["w_qkvz"],
            w_qkvzs=w["w_qkvzs"],
            w_ba=w["w_ba"],
            conv_w=w["conv_w"],
            conv_state=v_conv_state,
            a_log=w["a_log"],
            dt_bias=w["dt_bias"],
            norm_w=w["norm_w"],
            rstate=v_rstate,
            spec_state_indices=w["spec_state_indices"],
            num_accepted_tokens=w["num_accepted_tokens"],
            core=v_core,
            scratch=scr_verify,
            epoch=epoch,
            layer=6,
        )

    x_bin = torch.empty(s, HIDDEN, dtype=bf, device=device)
    key_proj = k1.K1Build(tokens=s, tp=TP, self_msq=True, proj_only=True)
    scr_proj = torch.zeros(
        k1.scratch_bytes(key_proj), dtype=torch.uint8, device=device
    )
    qkvz_out = torch.empty(s, lay.QKVZ, dtype=bf, device=device)
    ba_out = torch.empty(2, s, lay.NV, dtype=bf, device=device)
    p_conv_state, p_rstate = w["conv_state"].clone(), w["rstate"].clone()
    p_core = torch.empty(s, lay.NV, HD, dtype=bf, device=device)
    qkv_size = 2 * lay.KEY + lay.VAL

    def _proj_plus_stock():
        epoch.add_(1)
        k1.proj_pre(
            key_proj,
            state=w["state"],
            msq_in=None,
            hc_w=w["hc_w"],
            w_down=w["w_down"],
            w_up=w["w_up"],
            inj_out=inj,
            bin_out=x_bin,
            w_qkvz=w["w_qkvz"],
            w_qkvzs=w["w_qkvzs"],
            w_ba=w["w_ba"],
            qkvz_out=qkvz_out,
            ba_out=ba_out,
            scratch=scr_proj,
            epoch=epoch,
            layer=7,
        )
        gdn_spec_decode_conv_gdr_aiter(
            qkv=qkvz_out[:, :qkv_size],
            b=ba_out[0],
            a=ba_out[1],
            core_attn_out=p_core,
            conv_state=p_conv_state,
            conv_weights=w["conv_w"].view(lay.CONV, -1),
            conv_bias=None,
            activation="silu",
            A_log=w["a_log"],
            dt_bias=w["dt_bias"],
            ssm_state=p_rstate,
            spec_query_start_loc=qsl,
            spec_state_indices=w["spec_state_indices"],
            num_accepted_tokens=w["num_accepted_tokens"],
            num_spec_decodes=n_req,
            key_dim=lay.KEY,
            value_dim=lay.VAL,
            head_k_dim=HD,
            head_v_dim=HD,
            draft_window=k1.VERIFY_WINDOW,
        )
        _gated_rms_norm(
            p_core,
            qkvz_out[:, qkv_size:].view(s, lay.NV, HD),
            w["norm_w"],
            1e-6,
            "sigmoid",
        )

    cases = (
        ("k1.verify_pre (P3: HC front + in_proj + GDN verify core)", _k1_verify),
        ("proj_pre (P1b) + stock conv_gdr + _gated_rms_norm", _proj_plus_stock),
    )
    return [
        (name, _time_eager(fn, iters), _time_graph(fn, iters)) for name, fn in cases
    ]


# ---------------------------------------------------------------- plumbing
def setup_distributed():
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"} <= os.environ.keys():
        raise RuntimeError("launch this benchmark with torchrun --nproc-per-node 2")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != TP:
        raise RuntimeError(f"built for TP={TP}, got WORLD_SIZE={world_size}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment()
        ensure_model_parallel_initialized(world_size, 1)
    return rank, world_size, device


def _fmt(result) -> str:
    val, err = result
    if val is not None:
        return f"{val:9.2f} us"
    return f"FAILED ({err[:70]})"


def _print_rows(title: str, rows) -> None:
    print(f"-- {title} --")
    for name, eager, graph in rows:
        print(f"  {name:<62s} eager {_fmt(eager)}  graph {_fmt(graph)}")
    if title.startswith("1.") and len(rows) == 2:
        (_, _, g_gdn), (_, _, g_front) = rows
        if g_gdn[0] is not None and g_front[0] is not None:
            delta = g_gdn[0] - g_front[0]
            verdict = "build P1b" if delta <= 8 else "kill P1b" if delta >= 20 else "?"
            print(f"  Delta(gdn_pre - hc_front) = {delta:7.2f} us  [{verdict}]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s", type=str, default="1,2,4,8")
    p.add_argument("--iters", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    s_list = [int(v) for v in args.s.split(",") if v]
    rank, world_size, device = setup_distributed()

    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.qwen4_exp.amd.mono import k2
    from vllm.models.qwen4_exp.amd.mono.layout import MAX_TOKENS

    tp = get_tp_group()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    k2_peers = PeerBuffer(
        k2.peer_bytes(MAX_TOKENS, TP), tp.cpu_group, rank, world_size, device
    )
    k2_peers.bytes.zero_()

    sections = (
        ("1. K1 gate: gdn_pre vs hc_front", bench_k1_gate, {}),
        ("2. the stock chain the bracket would remove", bench_stock_pieces, {}),
        (
            "4. K1 proj_only (P1b) vs hc_front + stock in_proj",
            bench_k1_proj_gate,
            {},
        ),
        (
            "5. K1 verify (P3) vs proj_only + stock conv_gdr + gated norm",
            bench_k1_verify_gate,
            {},
        ),
    )
    for s in s_list:
        if rank == 0:
            print(f"\n=== S={s} ===")
        for title, fn, extra in sections:
            try:
                rows = fn(s, device, rank, world_size, epoch, args.iters, **extra)
            except Exception as exc:  # a whole section's kernel may not build
                if rank == 0:
                    print(f"-- {title} -- FAILED: {exc!r}")
                    traceback.print_exc()
                continue
            if rank == 0:
                _print_rows(title, rows)

    k2_peers.close()
    dist.barrier()


if __name__ == "__main__":
    main()
