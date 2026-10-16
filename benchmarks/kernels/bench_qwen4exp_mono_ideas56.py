# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IDEAS_5_6_DESIGN.md Stage 0c (E0: the head alone) and E6 (stage A: front
breakdown / fc_hidden off the critical path; stage B: the embedding chain)
on MI355X. Each section is its own standalone microbenchmark, independently
selectable, so a slow or failing one does not block the rest.

Not here: Stage 1 / E5 (the back kernel's tail-window prefetch) and E6.2
(``BackBuild(fh=True)``) -- both gated on results this file's sections
produce (G0, G6a) and out of scope for a probe that touches Mikko's kernels
only via no-op stamp hooks.

Usage:
    torchrun --nproc-per-node 2 benchmarks/kernels/bench_qwen4exp_mono_ideas56.py \
        [--sections e0,e60,e61,e6b0,e6b1,e6b2] [--s 1,8] [--cold] [--warm] \
        [--iters 100]

Needs MI355X, AITER, and the mono kernels; authored and syntax-checked
CPU-only (FlyDSL is not installed locally), not run here.
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

from benchmarks.kernels.bench_qwen4exp_mono_draft import (  # noqa: E402
    REPEATS,
    TP,
    WARMUP,
    _fmt,
    _time_graph,
    setup_distributed,
)

bf = torch.bfloat16


# ---------------------------------------------------------------- plumbing
def _flush_mall(device, mb: int = 1024) -> None:
    """A ``cm=0`` read of ``mb`` MiB of unrelated data: evicts whatever a
    previous launch left resident in the 256 MB Infinity Cache (LRU), so the
    next launch's weights are genuinely cold. Plain torch, not the probe
    kernel's own cache-modifier knobs -- good enough to evict, and avoids a
    JIT build just to flush."""
    junk = torch.empty(mb * 1024 * 1024 // 4, dtype=torch.int32, device=device)
    junk.add_(1)
    torch.cuda.synchronize()
    del junk


def _print_table(title: str, header: list, rows: list) -> None:
    print(f"-- {title} --")
    widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(header)
    ]
    print("  " + "  ".join(f"{h:<{w}}" for h, w in zip(header, widths)))
    for r in rows:
        print("  " + "  ".join(f"{str(v):<{w}}" for v, w in zip(r, widths)))


def _median_us(fn, iters: int) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
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
    return statistics.median(samples)


# ---------------------------------------------------------------- E0: the head alone
def section_e0(s_list, device, rank, world_size, epoch, iters, head_peers):
    """E0.a (warm/cold/prod-like, D = cold - warm), E0.b (AHEAD sweep),
    E0.c (residency for free: cm=0 vs cm=2 on a prod-like pattern),
    E0.d (loads_only floor, + the rows=8192 fixed-tail case). Gate G0 (build
    E5 only if D >= 8 us at S=1 AND (E0.c or a P1d hit)) is reported, not
    evaluated here -- it needs this section's D plus the MALL probe's P1d."""
    from tests.models.qwen3_5.test_amd_mono import _randn
    from tests.models.qwen4_exp.test_amd_mono import _lmhead_weights
    from vllm.models.qwen4_exp.amd.mono import head
    from vllm.models.qwen4_exp.amd.mono.layout import HIDDEN

    v_r = 124160
    vocab_start = rank * v_r
    CM_DEV = 0  # plain / device-scope, as E0.c's "cm=0" column

    def _case(s, rows, ahead, cm, loads_only, cold):
        q_sets = [_lmhead_weights(rank + 7 * i, device, rows) for i in range(3 if cold else 1)]
        g = torch.Generator(device=device).manual_seed(900 + s)
        x = _randn(g, s, HIDDEN, scale=0.3)
        out = torch.full((s,), -1, dtype=torch.int64, device=device)
        key = head.HeadBuild(tokens=s, tp=TP, rows=rows, ahead=ahead, cm=cm, loads_only=loads_only)
        scratch = torch.zeros(head.scratch_bytes(key), dtype=torch.uint8, device=device)
        i = 0

        def fn():
            nonlocal i
            epoch.add_(1)
            if cold:
                _flush_mall(device)
                i = (i + 1) % len(q_sets)
            w, ws = q_sets[i][2], q_sets[i][3]
            head.draft_head(
                key, x=x, w=w, ws=ws, valid=rows, vocab_start=vocab_start, out=out,
                scratch=scratch, peers=head_peers.addresses, rank=rank, epoch=epoch,
                slot=0,
            )

        return _median_us(fn, iters if not cold else min(iters, 20))

    rows = []
    for s in s_list:
        warm = _case(s, v_r, 2, 2, False, cold=False)
        cold = _case(s, v_r, 2, 2, False, cold=True)
        rows.append(("E0.a warm", s, v_r, f"{warm:.2f}", "-"))
        rows.append(("E0.a cold", s, v_r, f"{cold:.2f}", f"D={cold - warm:+.2f}"))
    if rank == 0:
        _print_table("E0.a warm/cold (us, cm=CM_NT, ahead=2)", ["case", "S", "rows", "us", "note"], rows)

    rows = []
    for s in (s_list[0], s_list[-1]):
        for ahead in (2, 3, 4, 6):
            for cm in (2, 0):
                us = _case(s, v_r, ahead, cm, False, cold=False)
                rows.append((s, ahead, cm, f"{us:.2f}"))
    if rank == 0:
        _print_table(
            "E0.b AHEAD sweep (us, warm) -- grep FlyDSL's dumped ISA for "
            "`private_segment_fixed_size` / `scratch` on the node to confirm no spill",
            ["S", "ahead", "cm", "us"], rows,
        )  # fmt: skip

    rows = []
    for s in s_list:
        cold_cm2 = _case(s, v_r, 2, 2, False, cold=True)
        cold_cm0 = _case(s, v_r, 2, 0, False, cold=True)
        rows.append((s, f"{cold_cm2:.2f}", f"{cold_cm0:.2f}", f"{cold_cm2 - cold_cm0:+.2f}"))
    if rank == 0:
        _print_table(
            "E0.c residency for free (us, cold, prod-like proxy) -- cm=0 vs cm=2",
            ["S", "us(cm=2)", "us(cm=0)", "delta"], rows,
        )  # fmt: skip

    rows = []
    for s in s_list:
        for ahead in (2, 4, 8):
            floor = _case(s, v_r, ahead, 2, True, cold=True)
            rows.append((s, v_r, ahead, f"{floor:.2f}"))
    tail = _case(1, 8192, 2, 2, True, cold=True)
    rows.append((1, 8192, 2, f"{tail:.2f} (fixed-tail probe)"))
    if rank == 0:
        _print_table("E0.d loads_only floor (us, cold)", ["S", "rows", "ahead", "us"], rows)


# ---------------------------------------------------------------- E6 stage A
def section_e60(s_list, device, rank, world_size, epoch, iters):
    """Per-pass exposed latency inside ``front.py`` via Stage 0a stamps:
    F1 (load_rrms_front), the fe pass, each of the 4 hn passes, F3
    (wait_state)."""
    from tests.models.qwen3_5.test_amd_mono import _randn
    from tests.models.qwen4_exp.test_amd_mono import _hc_weights
    from vllm.models.qwen4_exp.amd.mono import front, stamps
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HH, HIDDEN

    gc = torch.Generator(device=device).manual_seed(21)
    hw = _hc_weights(gc)
    for s in s_list:
        emb = _randn(gc, s, HIDDEN, scale=2.0)
        prev = _randn(gc, s, HH, scale=2.0)
        w_ne = _randn(gc, HIDDEN, scale=0.1)
        w_nh = _randn(gc, HH, scale=0.1)
        w_fe = _randn(gc, HIDDEN, HIDDEN, scale=0.05)
        w_fh = _randn(gc, HIDDEN, HIDDEN, scale=0.05)
        state_out = torch.empty(s, HH, dtype=bf, device=device)
        inj = torch.empty(s, HC, dtype=bf, device=device)
        x_bin = torch.empty(s, HIDDEN, dtype=bf, device=device)
        key = front.FrontBuild(tokens=s, tp=TP, trace=True)
        scratch = torch.zeros(front.scratch_bytes(key), dtype=torch.uint8, device=device)
        dbg = stamps.dbg_buffer(device)

        epoch.add_(1)
        front.draft_front(
            key, emb=emb, prev=prev, w_ne=w_ne, w_nh=w_nh, w_fe=w_fe, w_fh=w_fh, **hw,
            state_out=state_out, bin_out=x_bin, inj_out=inj, scratch=scratch,
            epoch=epoch, slot=0, dbg=dbg,
        )
        torch.cuda.synchronize()
        report = stamps.phase_report(
            dbg,
            {
                "entry->F1(load_rrms_front)": (1, 0),
                "F1->fe pass": (2, 1),
                "fe->hn0": (3, 2),
                "hn0->hn1": (4, 3),
                "hn1->hn2": (5, 4),
                "hn2->hn3": (6, 5),
                "hn3->sflag put": (7, 6),
                "sflag->wait_state": (8, 7),
                "wait_state->end": (10, 8),
                "total (entry->end)": (10, 0),
            },
        )
        if rank == 0:
            print(f"-- E6.0 front breakdown (us, S={s}) --")
            for name, (lo, med, hi) in report.items():
                print(f"  {name:<28s} min {lo:8.2f}  median {med:8.2f}  max {hi:8.2f}")


def section_e61(s_list, device, rank, world_size, epoch, iters):
    """E6.1: ``FrontBuild(pre_fh=True)`` vs stock front, graph-replayed.
    Gate G6a: build the back-kernel integration (E6.2, out of scope here)
    only if this saving is >= 10 us at S=1."""
    from tests.models.qwen3_5.test_amd_mono import _randn
    from tests.models.qwen4_exp.test_amd_mono import _hc_weights
    from vllm.models.qwen4_exp.amd.mono import front
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HH, HIDDEN

    gc = torch.Generator(device=device).manual_seed(22)
    hw = _hc_weights(gc)
    rows = []
    for s in s_list:
        emb = _randn(gc, s, HIDDEN, scale=2.0)
        prev = _randn(gc, s, HH, scale=2.0)
        w_ne = _randn(gc, HIDDEN, scale=0.1)
        w_nh = _randn(gc, HH, scale=0.1)
        w_fe = _randn(gc, HIDDEN, HIDDEN, scale=0.05)
        w_fh = _randn(gc, HIDDEN, HIDDEN, scale=0.05)
        state_out = torch.empty(s, HH, dtype=bf, device=device)
        inj = torch.empty(s, HC, dtype=bf, device=device)
        x_bin = torch.empty(s, HIDDEN, dtype=bf, device=device)

        stock_key = front.FrontBuild(tokens=s, tp=TP)
        stock_scratch = torch.zeros(
            front.scratch_bytes(stock_key), dtype=torch.uint8, device=device
        )
        pre_fh_key = front.FrontBuild(tokens=s, tp=TP, pre_fh=True)
        pre_fh_scratch = torch.zeros(
            front.scratch_bytes(pre_fh_key), dtype=torch.uint8, device=device
        )

        def _stock():
            epoch.add_(1)
            front.draft_front(
                stock_key, emb=emb, prev=prev, w_ne=w_ne, w_nh=w_nh, w_fe=w_fe,
                w_fh=w_fh, **hw, state_out=state_out, bin_out=x_bin, inj_out=inj,
                scratch=stock_scratch, epoch=epoch, slot=0,
            )

        def _pre_fh():
            epoch.add_(1)
            front.draft_front(
                pre_fh_key, emb=emb, prev=prev, w_ne=w_ne, w_nh=w_nh, w_fe=w_fe,
                w_fh=w_fh, **hw, state_out=state_out, bin_out=x_bin, inj_out=inj,
                scratch=pre_fh_scratch, epoch=epoch, slot=0,
            )

        stock_us, _ = _time_graph(_stock, iters)
        pre_fh_us, _ = _time_graph(_pre_fh, iters)
        saving = None if (stock_us is None or pre_fh_us is None) else stock_us - pre_fh_us
        rows.append(
            (s, _fmt((stock_us, None)), _fmt((pre_fh_us, None)), f"{saving:+.2f}" if saving else "n/a")
        )
    if rank == 0:
        _print_table(
            "E6.1 pre_fh saving (us, graph-replayed) -- G6a: saving >= 10 us at S=1",
            ["S", "stock", "pre_fh", "saving"], rows,
        )  # fmt: skip


# ---------------------------------------------------------------- E6 stage B
def section_e6b0(device, rank, iters):
    """E6b.0 floor: N=11 trivial kernels graph-replayed, us/kernel -- the
    true graph-replay glue cost the profiled ~4 us/kernel converts to."""
    n = 11
    t = torch.zeros(n, 8, dtype=torch.int32, device=device)

    def fn():
        for i in range(n):
            t[i].add_(1)

    us, _ = _time_graph(fn, iters)
    if rank == 0 and us is not None:
        print(f"-- E6b.0 floor: {n} trivial kernels, graph-replayed --")
        print(f"  total {us:.2f} us  ({us / n:.2f} us/kernel)")


def section_e6b1(s_list, device, rank, world_size, epoch, iters):
    """E6b.1 stock chain: ``VocabParallelEmbedding`` forward (masks +
    index_select + mul + custom AR), graph-replayed, the real
    ``embed_tokens`` shard shape (124160 x 2560 bf16 a rank, TP2)."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.models.qwen4_exp.amd.mono.layout import HIDDEN

    vocab = 248320
    with set_current_vllm_config(VllmConfig()):
        emb = VocabParallelEmbedding(vocab, HIDDEN, params_dtype=bf).to(device)
    rows = []
    for s in s_list:
        g = torch.Generator(device=device).manual_seed(930 + s)
        ids = torch.randint(0, vocab, (s,), generator=g, device=device)

        def fn():
            epoch.add_(1)
            emb(ids)

        us, _ = _time_graph(fn, iters)
        rows.append((s, _fmt((us, None))))
    if rank == 0:
        _print_table(
            "E6b.1 stock VocabParallelEmbedding forward (us, graph-replayed)",
            ["S", "us"], rows,
        )  # fmt: skip


def section_e6b2(s_list, device, rank, world_size, epoch, iters):
    """E6b.2: ``FrontBuild(emb_ids=True)`` (front gathers its own embedding
    row from a replicated table) vs the stock path's precomputed ``emb``
    tensor, graph-replayed, plus a bitwise-equality correctness check.
    Saving = E6b.1's us (section_e6b1) minus this section's delta; GO if
    >= 8 us at S=1."""
    from tests.models.qwen3_5.test_amd_mono import _randn
    from tests.models.qwen4_exp.test_amd_mono import _hc_weights
    from vllm.models.qwen4_exp.amd.mono import front
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HH, HIDDEN

    vocab_per_rank = 124160  # this rank's shard of the 248320-row table
    gc = torch.Generator(device=device).manual_seed(23)
    hw = _hc_weights(gc)
    table = _randn(gc, vocab_per_rank, HIDDEN, scale=2.0)  # this rank's replicated shard
    rows = []
    for s in s_list:
        g = torch.Generator(device=device).manual_seed(940 + s)
        ids = torch.randint(0, vocab_per_rank, (s,), generator=g, device=device).to(torch.int64)
        emb_rows = table[ids].contiguous()  # the stock path's precomputed [S, H]
        prev = _randn(gc, s, HH, scale=2.0)
        w_ne = _randn(gc, HIDDEN, scale=0.1)
        w_nh = _randn(gc, HH, scale=0.1)
        w_fe = _randn(gc, HIDDEN, HIDDEN, scale=0.05)
        w_fh = _randn(gc, HIDDEN, HIDDEN, scale=0.05)

        def _run(emb_ids: bool, emb_tensor, ids_tensor):
            key = front.FrontBuild(tokens=s, tp=TP, emb_ids=emb_ids)
            scratch = torch.zeros(front.scratch_bytes(key), dtype=torch.uint8, device=device)
            state_out = torch.empty(s, HH, dtype=bf, device=device)
            inj = torch.empty(s, HC, dtype=bf, device=device)
            x_bin = torch.empty(s, HIDDEN, dtype=bf, device=device)

            def fn():
                epoch.add_(1)
                front.draft_front(
                    key, emb=emb_tensor, prev=prev, w_ne=w_ne, w_nh=w_nh, w_fe=w_fe,
                    w_fh=w_fh, **hw, state_out=state_out, bin_out=x_bin, inj_out=inj,
                    scratch=scratch, epoch=epoch, slot=0,
                    ids=ids_tensor if emb_ids else None,
                )

            return fn, state_out

        precomp_fn, precomp_state = _run(False, emb_rows, None)
        ids_fn, ids_state = _run(True, table, ids)

        # correctness: bitwise equal to front(emb=table[ids])
        epoch_val = epoch.clone()
        precomp_fn()
        torch.cuda.synchronize()
        want = precomp_state.clone()
        epoch.copy_(epoch_val)
        ids_fn()
        torch.cuda.synchronize()
        bitwise_ok = torch.equal(want, ids_state)

        precomp_us, _ = _time_graph(precomp_fn, iters)
        ids_us, _ = _time_graph(ids_fn, iters)
        delta = None if (precomp_us is None or ids_us is None) else ids_us - precomp_us
        rows.append(
            (s, _fmt((precomp_us, None)), _fmt((ids_us, None)),
             f"{delta:+.2f}" if delta is not None else "n/a", bitwise_ok)
        )  # fmt: skip
    if rank == 0:
        _print_table(
            "E6b.2 emb_ids delta (us, graph-replayed) + bitwise check "
            "(expect ~0..+1 us, bitwise_ok=True)",
            ["S", "us(precomputed emb)", "us(emb_ids)", "delta", "bitwise_ok"], rows,
        )  # fmt: skip


# ---------------------------------------------------------------- plumbing
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sections", type=str, default="e0,e60,e61,e6b0,e6b1,e6b2")
    p.add_argument("--s", type=str, default="1,8")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--cold", action="store_true", help="unused flag, kept for the CLI shape IDEAS_5_6_DESIGN.md sketches; E0 always runs both warm and cold")  # noqa: E501
    p.add_argument("--warm", action="store_true", help="see --cold")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sections = set(args.sections.split(","))
    s_list = [int(v) for v in args.s.split(",") if v]
    rank, world_size, device = setup_distributed()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)

    head_peers = None
    if "e0" in sections:
        from vllm.distributed import get_tp_group
        from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
        from vllm.models.qwen4_exp.amd.mono import head

        tp = get_tp_group()
        head_peers = PeerBuffer(head.peer_bytes(TP), tp.cpu_group, rank, world_size, device)
        head_peers.bytes.zero_()

    try:
        if "e0" in sections:
            section_e0(s_list, device, rank, world_size, epoch, args.iters, head_peers)
        if "e60" in sections:
            section_e60(s_list, device, rank, world_size, epoch, args.iters)
        if "e61" in sections:
            section_e61(s_list, device, rank, world_size, epoch, args.iters)
        if "e6b0" in sections:
            section_e6b0(device, rank, args.iters)
        if "e6b1" in sections:
            section_e6b1(s_list, device, rank, world_size, epoch, args.iters)
        if "e6b2" in sections:
            section_e6b2(s_list, device, rank, world_size, epoch, args.iters)
    finally:
        if head_peers is not None:
            head_peers.close()
        import torch.distributed as dist

        dist.barrier()


if __name__ == "__main__":
    main()
