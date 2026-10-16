# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the Qwen3.8-Flash-Next MTP draft step's mono decode kernels on
MI355X: each fused launch against the stock op sequence it replaces, at the
production TP=2 shapes (H=2560, HC=4, LORA=320, DOWN_ROWS=336, a 124160-row
vocab shard a rank).

Usage:
    torchrun --nproc-per-node 2 benchmarks/kernels/bench_qwen4exp_mono_draft.py \
        [--s 1,2,4,8] [--iters 200]

Per S, rank 0 prints eager and CUDA-graph-replayed median-of-5 per-launch
microseconds for: (1) ``draft_head``'s MXFP4 lm_head GEMV + TP argmax vs a
bf16 and an MXFP4 stock greedy pick; (2) ``draft_front`` vs the stock norms +
column-parallel GEMMs + ``k1.hc_front``; (3) the final mixer's ``hc_front``
(zero-padded down) vs the unfused ``ops/hc.py`` sequence; (4) K2's
``layer_post`` for the draft layer, its own time. Needs MI355X (gfx950),
AITER, and the mono kernels; this file is authored CPU-only and not run here.
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


# ---------------------------------------------------------------- 1. draft_head
def bench_draft_head(s, device, rank, world_size, epoch, iters, peers):
    from tests.models.qwen3_5.test_amd_mono import (
        _mxfp4_dequant,
        _mxfp4_linear_stock,
        _randn,
    )
    from tests.models.qwen4_exp.test_amd_mono import _lmhead_weights
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.models.qwen4_exp.amd.mono import head
    from vllm.models.qwen4_exp.amd.mono.layout import HIDDEN

    v_r = 124160  # the real per-rank vocab shard (no padding)
    vocab_start = rank * v_r
    q, sc, w, ws = _lmhead_weights(rank, device, v_r)
    w_bf16 = _mxfp4_dequant(q, sc).to(bf)
    g = torch.Generator(device=device).manual_seed(900 + s)
    x = _randn(g, s, HIDDEN, scale=0.3)

    def _pick(local_val, global_idx):
        pair = torch.stack([local_val.float(), global_idx.float()], dim=-1)
        gathered = tensor_model_parallel_all_gather(pair, dim=-1)
        gathered = gathered.view(s, world_size, 2)
        best = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        return gathered[:, :, 1].gather(dim=-1, index=best).squeeze(-1).to(torch.int64)

    def _stock_bf16():
        epoch.add_(1)
        logits = F.linear(x, w_bf16)
        lv, li = logits.max(dim=-1)
        _pick(lv, li + vocab_start)

    def _stock_mxfp4():
        epoch.add_(1)
        logits = _mxfp4_linear_stock(x, w, ws)
        lv, li = logits.max(dim=-1)
        _pick(lv, li + vocab_start)

    out = torch.full((s,), -1, dtype=torch.int64, device=device)
    key = head.HeadBuild(tokens=s, tp=TP, rows=v_r)
    scratch = torch.zeros(head.scratch_bytes(key), dtype=torch.uint8, device=device)

    def _fused():
        epoch.add_(1)
        head.draft_head(
            key, x=x, w=w, ws=ws, valid=v_r, vocab_start=vocab_start, out=out,
            scratch=scratch, peers=peers.addresses, rank=rank, epoch=epoch, slot=0,
        )

    cases = (
        ("stock: bf16 F.linear + argmax + all_gather", _stock_bf16),
        ("stock: MXFP4 a4w4 GEMM + argmax + all_gather", _stock_mxfp4),
        ("draft_head (fused)", _fused),
    )
    return [
        (name, _time_eager(fn, iters), _time_graph(fn, iters)) for name, fn in cases
    ]


# --------------------------------------------------------------- 2. draft_front
def bench_draft_front(s, device, rank, world_size, epoch, iters):
    from tests.models.qwen3_5.test_amd_mono import _randn
    from tests.models.qwen4_exp.test_amd_mono import _hc_weights
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.models.qwen4_exp.amd.mono import front, k1
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HH, HIDDEN

    h2 = HIDDEN // 2
    gc = torch.Generator(device=device).manual_seed(21)
    emb = _randn(gc, s, HIDDEN, scale=2.0)
    prev = _randn(gc, s, HH, scale=2.0)
    w_ne = _randn(gc, HIDDEN, scale=0.1)
    w_nh = _randn(gc, HH, scale=0.1)
    w_fe = _randn(gc, HIDDEN, HIDDEN, scale=0.05)
    w_fh = _randn(gc, HIDDEN, HIDDEN, scale=0.05)
    hw = _hc_weights(gc)
    ne_eps, nh_eps = 1e-5, 3e-6
    # this rank's ColumnParallelLinear(gather_output=True) shard: half the
    # output rows of the SAME full weight ``draft_front`` holds replicated
    w_fe_local = w_fe[rank * h2 : (rank + 1) * h2].contiguous()
    w_fh_local = w_fh[rank * h2 : (rank + 1) * h2].contiguous()

    def _gemma(x, w, eps):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (xf * (1.0 + w.float())).to(x.dtype)

    state = torch.empty(s, HH, dtype=bf, device=device)
    inj = torch.empty(s, HC, dtype=bf, device=device)
    x_bin = torch.empty(s, HIDDEN, dtype=bf, device=device)
    k1key = k1.K1Build(tokens=s, tp=TP, self_msq=True, front_only=True)
    k1scratch = torch.zeros(k1.scratch_bytes(k1key), dtype=torch.uint8, device=device)

    def _stock():
        epoch.add_(1)
        en = _gemma(emb, w_ne, ne_eps)
        fe = tensor_model_parallel_all_gather(F.linear(en, w_fe_local), dim=-1)
        hn = _gemma(prev, w_nh, nh_eps).view(s, HC, HIDDEN)
        fh = tensor_model_parallel_all_gather(F.linear(hn, w_fh_local), dim=-1)
        state.copy_((fe.unsqueeze(1) + fh).flatten(1))
        k1.hc_front(
            k1key, state=state, msq_in=None, **hw, inj_out=inj, bin_out=x_bin,
            scratch=k1scratch, epoch=epoch, layer=0,
        )

    fkey = front.FrontBuild(tokens=s, tp=TP, ne_eps=ne_eps, nh_eps=nh_eps)
    fscratch = torch.zeros(front.scratch_bytes(fkey), dtype=torch.uint8, device=device)
    state_out = torch.empty(s, HH, dtype=bf, device=device)

    def _fused():
        epoch.add_(1)
        front.draft_front(
            fkey, emb=emb, prev=prev, w_ne=w_ne, w_nh=w_nh, w_fe=w_fe, w_fh=w_fh,
            **hw, state_out=state_out, bin_out=x_bin, inj_out=inj, scratch=fscratch,
            epoch=epoch, slot=0,
        )

    cases = (
        ("stock: 2 norms + 2 col-parallel GEMMs + all_gather + hc_front", _stock),
        ("draft_front (fused)", _fused),
    )
    return [
        (name, _time_eager(fn, iters), _time_graph(fn, iters)) for name, fn in cases
    ]


# ---------------------------------------------------------------- 3. final mixer
def bench_mixer(s, device, rank, world_size, epoch, iters):
    from tests.models.qwen3_5.test_amd_mono import _randn
    from vllm.models.qwen4_exp.amd.mono import k1
    from vllm.models.qwen4_exp.amd.mono.layout import DOWN_ROWS, HC, HH, HIDDEN, LORA
    from vllm.models.qwen4_exp.amd.ops.hc import (
        grouped_gemma_rmsnorm,
        hc_gate_mix,
        hc_silu,
    )

    gc = torch.Generator(device=device).manual_seed(31)
    state = _randn(gc, s, HH, scale=2.0)
    hc_w = _randn(gc, HH, scale=0.1)
    w_down = _randn(gc, LORA, HH, scale=0.01)  # no injection rows: use_combine=False
    w_up = _randn(gc, HH, LORA, scale=0.05)
    w_down_padded = torch.zeros(DOWN_ROWS, HH, dtype=bf, device=device)
    w_down_padded[:LORA] = w_down

    def _stock():
        epoch.add_(1)
        xn = grouped_gemma_rmsnorm(state, hc_w, 1e-6, HC)
        d = F.linear(xn, w_down)
        lora = hc_silu(d, HC)
        gate = F.linear(lora, w_up)
        hc_gate_mix(xn, gate, HC)

    h = torch.empty(s, HIDDEN, dtype=bf, device=device)
    inj = torch.empty(s, HC, dtype=bf, device=device)  # discarded: no combine here
    key = k1.K1Build(tokens=s, tp=TP, self_msq=True, front_only=True)
    scratch = torch.zeros(k1.scratch_bytes(key), dtype=torch.uint8, device=device)

    def _fused():
        epoch.add_(1)
        k1.hc_front(
            key, state=state, msq_in=None, hc_w=hc_w, w_down=w_down_padded,
            w_up=w_up, inj_out=inj, bin_out=h, scratch=scratch, epoch=epoch, layer=0,
        )

    cases = (
        ("stock: grouped Gemma norm + down + hc_silu + up + gate-mix", _stock),
        ("hc_front (fused, zero-padded down)", _fused),
    )
    return [
        (name, _time_eager(fn, iters), _time_graph(fn, iters)) for name, fn in cases
    ]


# ---------------------------------------------------------- 4. K2 layer_post
def bench_k2_layer_post(s, device, rank, world_size, epoch, iters, peers):
    from tests.models.qwen3_5.test_amd_mono import _randn
    from tests.models.qwen4_exp.test_amd_mono import _hc_weights, _moe_weights
    from vllm.models.qwen4_exp.amd.mono import k2
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HH, HIDDEN, Layout

    lay = Layout(TP)
    gc = torch.Generator(device=device).manual_seed(41)
    gr = torch.Generator(device=device).manual_seed(410 + rank)
    core = _randn(gr, s, lay.CORE)
    gate = _randn(gr, s, lay.CORE, scale=2.0)
    state = _randn(gc, s, HH, scale=2.0)
    inj_a = _randn(gc, s, HC)
    w_o = _randn(gr, HIDDEN, lay.CORE, scale=0.02)  # bf16 o_proj: not asm_mxfp4
    hw = _hc_weights(gc)
    mw = _moe_weights(rank, device)
    out = torch.empty(s, HH, dtype=bf, device=device)
    msq_out = torch.empty(20, s, HC, dtype=torch.float32, device=device)
    key = k2.K2Build(tokens=s, tp=TP, og=True)
    scratch = torch.zeros(k2.scratch_bytes(key), dtype=torch.uint8, device=device)

    def _fused():
        epoch.add_(1)
        k2.layer_post(
            key, core=core, gate=gate, state=state, inj_a=inj_a, w_o=w_o, **hw,
            w_gate=mw["w_gate"], w_sg=mw["w_sg"], w13=mw["w13"], w13s=mw["w13s"],
            w2=mw["w2"], w2s=mw["w2s"], out=out, msq_out=msq_out, scratch=scratch,
            peers=peers.addresses, rank=rank, epoch=epoch,
            layer=k2.TAG_SLOTS // 2 + 48,
        )

    name = "layer_post (og=True, bf16 o_proj)"
    return [(name, _time_eager(_fused, iters), _time_graph(_fused, iters))]


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
    from vllm.models.qwen4_exp.amd.mono import head, k2
    from vllm.models.qwen4_exp.amd.mono.layout import MAX_TOKENS

    tp = get_tp_group()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    head_peers = PeerBuffer(head.peer_bytes(TP), tp.cpu_group, rank, world_size, device)
    k2_peers = PeerBuffer(
        k2.peer_bytes(MAX_TOKENS, TP), tp.cpu_group, rank, world_size, device
    )
    head_peers.bytes.zero_()
    k2_peers.bytes.zero_()

    sections = (
        ("1. draft_head", bench_draft_head, {"peers": head_peers}),
        ("2. draft_front", bench_draft_front, {}),
        ("3. final mixer (hc_front)", bench_mixer, {}),
        ("4. K2 layer_post (draft layer)", bench_k2_layer_post, {"peers": k2_peers}),
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

    head_peers.close()
    k2_peers.close()
    dist.barrier()


if __name__ == "__main__":
    main()
