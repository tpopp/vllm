# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An in-kernel GPU-utilization timeline for the Qwen3.8-Flash-Next mono
decode kernels (K2's ``layer_post`` and K1's ``hc_front``), standing in for a
hardware profiler: these are persistent, mailbox-synchronized FlyDSL
kernels, so a sampling profiler's "which SM is doing what" does not map
cleanly onto "which stage is this CTA waiting on" -- the in-kernel stamps
(``vllm.models.qwen4_exp.amd.mono.stamps``, Stage 0a) answer that directly.

Runs, at the production TP=2 shapes, graph-replayed ``--graph-launches``
times with ``trace=True`` (so the dbg buffer holds the LAST replay's
timestamps -- each launch overwrites every slot it visits):

    (a) K2 alone, a GDN layer (``oq=True``, MXFP4 out_proj) and a QSA layer
        (``og=True``, bf16 o_proj with the attention gate)
    (b) K1's ``hc_front`` (``front_only=True``)

each at S in {1, 4, 8}. Per run: decodes the dbg buffer (normalized to that
launch's min "entry" stamp), groups CTAs by the role ranges the kernel code
places them in, and prints each role's median phase durations and the
fraction of wall time spent between a wait site's "before" and "after"
stamps; reports which wait chain's last CTA finishes last (the kernel's
critical path); prints an ASCII Gantt of 16 representative CTAs; and times
the same launch with ``trace=False`` to report the stamps' overhead. Also
writes the raw per-CTA arrays (every slot, every role) to .npz/.json under
``--out-dir``.

Usage:
    torchrun --nproc-per-node 2 benchmarks/kernels/bench_qwen4exp_mono_util.py \
        [--s 1,4,8] [--graph-launches 50] [--iters 200] [--out-dir /tmp/mono_util]

Needs MI355X (gfx950), AITER, and the mono kernels; this file is authored
CPU-only and not run here.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

bf = torch.bfloat16
TP = 2
WARMUP = 3
REPEATS = 5


# ---------------------------------------------------------------- timing
def _time_eager(fn, iters: int):
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


def _time_graph(fn, graph_launches: int, iters: int):
    """``graph_launches`` launches captured once, replayed until >= ``iters``
    launches are timed, median of ``REPEATS`` repeats, divided back to a
    per-launch microsecond figure. Returns (us_per_launch, graph): the graph
    is returned so the caller can decode the dbg buffer the last captured
    launch's stamps left behind (any replay leaves the same content: each
    launch overwrites every slot it visits)."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_launches):
            fn()
    torch.cuda.synchronize()
    replays = max(1, iters // graph_launches)
    samples = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (replays * graph_launches))
    return statistics.median(samples), graph


# ---------------------------------------------------------------- inputs
def make_k2_runner(key, s, device, rank, hw, mw, peers, epoch, dbg):
    from tests.models.qwen3_5.test_amd_mono import _mxfp4_linear, _randn
    from vllm.models.qwen4_exp.amd.mono import k2
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HH, HIDDEN, Layout

    lay = Layout(TP)
    gr = torch.Generator(device=device).manual_seed(4100 + rank)
    gc = torch.Generator(device=device).manual_seed(4000)
    core = _randn(gr, s, lay.CORE)
    gate = _randn(gr, s, lay.CORE, scale=2.0) if key.og else None
    state = _randn(gc, s, HH, scale=2.0)
    inj_a = _randn(gc, s, HC)
    if key.oq:
        _, _, w_o, w_os = _mxfp4_linear(gr, HIDDEN, lay.CORE)
    else:
        w_o, w_os = _randn(gr, HIDDEN, lay.CORE, scale=0.02), None
    out = torch.empty(s, HH, dtype=bf, device=device)
    msq_out = torch.empty(20, s, HC, dtype=torch.float32, device=device)
    scratch = torch.zeros(k2.scratch_bytes(key), dtype=torch.uint8, device=device)

    def run():
        epoch.add_(1)
        k2.layer_post(
            key, core=core, gate=gate, state=state, inj_a=inj_a, w_o=w_o, w_os=w_os,
            **hw, w_gate=mw["w_gate"], w_sg=mw["w_sg"], w13=mw["w13"],
            w13s=mw["w13s"], w2=mw["w2"], w2s=mw["w2s"], out=out, msq_out=msq_out,
            scratch=scratch, peers=peers.addresses, rank=rank, epoch=epoch,
            layer=k2.TAG_SLOTS // 2 + 48, dbg=dbg,
        )

    return run


def make_k1_front_runner(key, s, device, dbg):
    from tests.models.qwen3_5.test_amd_mono import _randn
    from tests.models.qwen4_exp.test_amd_mono import _hc_weights, _msq
    from vllm.models.qwen4_exp.amd.mono import k1
    from vllm.models.qwen4_exp.amd.mono.layout import HC, HIDDEN

    gc = torch.Generator(device=device).manual_seed(4200)
    state = _randn(gc, s, k1.HH, scale=2.0)
    hw = _hc_weights(gc)
    msq_in = _msq(state)
    x = torch.empty(s, HIDDEN, dtype=bf, device=device)
    inj = torch.empty(s, HC, dtype=bf, device=device)
    scratch = torch.zeros(k1.scratch_bytes(key), dtype=torch.uint8, device=device)
    epoch = torch.zeros(1, dtype=torch.int32, device=device)

    def run():
        epoch.add_(1)
        k1.hc_front(
            key, state=state, msq_in=msq_in, **hw, inj_out=inj, bin_out=x,
            scratch=scratch, epoch=epoch, layer=0, dbg=dbg,
        )

    return run


# ------------------------------------------------------------- CTA roles
def k2_roles(s: int) -> dict:
    from vllm.models.kimi_k3.amd.mono.common.plan import BLOCKS
    from vllm.models.qwen4_exp.amd.mono import k2

    return {
        "o_proj+hc_up (bid<D0)": range(0, k2.D0),
        "o_proj+hc_down+hc_up (D0<=bid<OTASKS)": range(k2.D0, k2.OTASKS),
        "router+hc_down (is_r)": range(k2.R0, k2.RT0),
        "sgate+hc_down (is_g)": range(k2.RT0, k2.RT0 + s),
        "hc_down_only": range(k2.RT0 + s, BLOCKS),
    }


def k1_front_roles() -> dict:
    from vllm.models.kimi_k3.amd.mono.common.plan import BLOCKS
    from vllm.models.qwen4_exp.amd.mono import hc, k1

    return {
        "hc_up_only (bid<D0)": range(0, k1.D0),
        "hc_down+hc_up (D0<=bid<UP_TASKS)": range(k1.D0, hc.UP_TASKS),
        "hc_down_only (bid>=UP_TASKS)": range(hc.UP_TASKS, BLOCKS),
    }


# ------------------------------------------------------------- decode / report
def _role_of(bid: int, roles: dict) -> str:
    for name, rng in roles.items():
        if bid in rng:
            return name
    return "<unassigned>"


def _norm(us: np.ndarray, entry_slot: int) -> tuple[np.ndarray, float]:
    """``us`` ([256, nslot] decoded microseconds) shifted so this launch's
    min "entry" stamp (over CTAs that actually got one; an off / idle CTA's
    entry stays 0 and is excluded) is time 0. Returns (shifted, t0).

    Caveat: 0 doubles as "this slot was never stamped" (``stamps.py``'s own
    convention); the one CTA whose entry stamp defines t0 lands exactly on 0
    after shifting, so its "entry" point is indistinguishable from unvisited
    in ``_transitions`` / ``gantt`` -- a cosmetic gap in that single CTA's
    reported sequence, not a correctness issue for the rest."""
    live = us[:, entry_slot] != 0
    t0 = float(us[live, entry_slot].min()) if live.any() else 0.0
    out = us.copy()
    shift = (out != 0) & live[:, None]  # only real stamps move; unvisited stay 0
    out[shift] -= t0
    out[~live] = 0.0
    return out, t0


def _transitions(us: np.ndarray, bid: int, slots: dict) -> list[tuple[str, str, float]]:
    """CTA ``bid``'s nonzero (label, time) pairs, slot-number order ->
    consecutive (from_label, to_label, duration_us)."""
    pairs = sorted(slots.items(), key=lambda kv: kv[1])
    visited = [(name, us[bid, slot]) for name, slot in pairs if us[bid, slot] != 0]
    return [
        (a[0], b[0], b[1] - a[1]) for a, b in zip(visited, visited[1:]) if b[1] > a[1]
    ]


def report(label: str, us: np.ndarray, slots: dict, roles: dict, wait_pairs: dict):
    """Per-role median phase durations, waiting fraction, and the critical
    path (the wait whose last-finishing CTA finishes latest)."""
    print(f"\n-- {label} --")
    wait_set = {(lo, hi) for lo, hi in wait_pairs.values()}
    by_role: dict[str, list[int]] = {}
    for bid in range(us.shape[0]):
        if us[bid, slots["entry"]] == 0:
            continue
        by_role.setdefault(_role_of(bid, roles), []).append(bid)
    for role, ctas in by_role.items():
        trans: dict[tuple[str, str], list[float]] = {}
        wait_us = {bid: 0.0 for bid in ctas}
        span_us = {bid: 0.0 for bid in ctas}
        for bid in ctas:
            tr = _transitions(us, bid, slots)
            for a, b, d in tr:
                trans.setdefault((a, b), []).append(d)
            span_us[bid] = us[bid, slots["exit"]] - us[bid, slots["entry"]]
            for a, b, d in tr:
                a_slot, b_slot = slots[a], slots[b]
                if (a_slot, b_slot) in wait_set:
                    wait_us[bid] += d
        print(f"  role: {role} ({len(ctas)} CTAs)")
        for (a, b), ds in sorted(trans.items(), key=lambda kv: slots[kv[0][0]]):
            tag = " [wait]" if (slots[a], slots[b]) in wait_set else ""
            print(f"    {a:>20s} -> {b:<20s} median {statistics.median(ds):7.3f} us"
                  f" (n={len(ds)}){tag}")
        fracs = [wait_us[b] / span_us[b] for b in ctas if span_us[b] > 0]
        if fracs:
            print(f"    waiting fraction of wall time: median "
                  f"{statistics.median(fracs) * 100:5.1f}%, max "
                  f"{max(fracs) * 100:5.1f}%")

    # critical path: for every named wait, the latest "after" (hi) stamp over
    # every CTA that reached it -- whichever wait's last CTA finishes latest
    # is the wait chain the kernel's tail is riding on.
    latest = []
    for name, (lo, hi) in wait_pairs.items():
        live = us[:, hi] != 0
        if not live.any():
            continue
        bid = int(np.argmax(np.where(live, us[:, hi], -1)))
        latest.append((us[bid, hi], name, bid))
    if latest:
        latest.sort()
        t, name, bid = latest[-1]
        print(f"  critical path: \"{name}\" wait's last CTA finishes latest"
              f" (CTA {bid}, {t:.3f} us after entry)")
        for t2, name2, bid2 in latest[-5:][::-1]:
            print(f"    {name2:>16s}: CTA {bid2:3d} finishes its wait at {t2:7.3f} us")


def gantt(us: np.ndarray, slots: dict, n: int = 16) -> str:
    """ASCII Gantt of ``n`` representative CTAs (evenly spaced 0 .. 255):
    one line a CTA, one character per ~2% of the kernel's total span, a
    letter per distinct phase label (first letter, repeated for its
    duration)."""
    live_bids = [b for b in range(us.shape[0]) if us[b, slots["entry"]] != 0]
    if not live_bids:
        return "(no live CTAs)"
    total = max(us[b, slots["exit"]] for b in live_bids) or 1.0
    width = 60
    step = max(1, us.shape[0] // n)
    lines = []
    for bid in range(0, us.shape[0], step):
        if us[bid, slots["entry"]] == 0:
            continue
        row = [" "] * width
        tr = _transitions(us, bid, slots)
        if tr:
            # fill from the first transition's start to each's end
            t = tr[0]
            start = us[bid, slots[t[0]]]
            for a, b, _ in tr:
                lo, hi = us[bid, slots[a]], us[bid, slots[b]]
                c0 = int((lo - start) / total * width)
                c1 = max(c0 + 1, int((hi - start) / total * width))
                for c in range(max(0, c0), min(width, c1)):
                    row[c] = a[0].upper()
        lines.append(f"  CTA {bid:3d} |{''.join(row)}|")
    return "\n".join(lines)


def _dump(out_dir: str, tag: str, us: np.ndarray, slots: dict, roles: dict):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{tag}.npz"), us=us)
    meta = {
        "slots": slots,
        "roles": {k: [v.start, v.stop] for k, v in roles.items()},
    }
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s", type=str, default="1,4,8")
    p.add_argument("--graph-launches", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--out-dir", type=str, default="/tmp/qwen4exp_mono_util")
    return p.parse_args()


def _bench_one(tag, key_off, key_on, s, make_runner, slots, roles, out_dir, rank,
                graph_launches, iters):
    """Time ``key_off`` (trace=False) and ``key_on`` (trace=True, its own
    compiled kernel -- ``trace`` is in the JIT cache key) eager, then
    ``key_on`` graph-replayed; decode the dbg buffer the last replay left and
    report / dump it (rank 0)."""
    from vllm.models.qwen4_exp.amd.mono import stamps

    device = torch.device("cuda", torch.cuda.current_device())
    dbg_on = stamps.dbg_buffer(device, nslot=slots["__nslot__"])
    run_off = make_runner(key_off, s, device, None)
    run_on = make_runner(key_on, s, device, dbg_on)

    us_off = _time_eager(run_off, iters // 10 or 1)
    us_on_eager = _time_eager(run_on, iters // 10 or 1)
    us_on_graph, _ = _time_graph(run_on, graph_launches, iters)
    torch.cuda.synchronize()

    if rank == 0:
        nslot = slots["__nslot__"]
        raw = stamps.decode(dbg_on, nslot=nslot)
        sm = {k: v for k, v in slots.items() if k != "__nslot__"}  # the slot map
        norm, t0 = _norm(raw, sm["entry"])
        print(f"\n=== {tag} (S={s}) ===")
        print(f"  eager: trace=False {us_off:8.2f} us, trace=True {us_on_eager:8.2f} us"
              f" (stamp overhead {us_on_eager - us_off:+6.2f} us)")
        print(f"  graph-replayed (trace=True): {us_on_graph:8.2f} us/launch")
        wait_pairs = {
            n: (sm[f"{n}_wait_lo"], sm[f"{n}_wait_hi"])
            for n in ("msq", "lora", "bin")
            if f"{n}_wait_lo" in sm
        }
        for extra, pair in (
            ("ar_attn", ("attn_ar_wait_lo", "attn_ar_wait_hi")),
            ("ar_final", ("final_ar_wait_lo", "final_ar_wait_hi")),
            ("route", ("route_wait_lo", "route_wait_hi")),
            ("sgate", ("sgate_wait_lo", "sgate_wait_hi")),
            ("inter", ("inter_wait_lo", "inter_wait_hi")),
            ("gdn_conv", ("gdn_conv_wait_lo", "gdn_conv_wait_hi")),
            ("gdn_gates", ("gdn_gates_wait_lo", "gdn_gates_wait_hi")),
            ("gdn_norm", ("gdn_norm_wait_lo", "gdn_norm_wait_hi")),
        ):
            if pair[0] in sm:
                wait_pairs[extra] = (sm[pair[0]], sm[pair[1]])
        report(tag, norm, sm, roles, wait_pairs)
        span = norm[:, sm["exit"]].max()
        print(f"\n  ASCII Gantt (t0={t0:.1f} us absolute, {span:.2f} us span):")
        print(gantt(norm, sm))
        _dump(out_dir, tag.replace(" ", "_").replace("/", "-"), norm, sm, roles)


def main() -> None:
    args = parse_args()
    s_list = [int(v) for v in args.s.split(",") if v]
    rank, world_size, device = setup_distributed()

    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.qwen4_exp.amd.mono import k1, k2
    from vllm.models.qwen4_exp.amd.mono.layout import MAX_TOKENS

    tp = get_tp_group()
    k2_peers = PeerBuffer(
        k2.peer_bytes(MAX_TOKENS, TP), tp.cpu_group, rank, world_size, device
    )
    k2_peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)

    for s in s_list:
        from tests.models.qwen4_exp.test_amd_mono import _hc_weights, _moe_weights

        hw = _hc_weights(torch.Generator(device=device).manual_seed(4000))
        mw = _moe_weights(rank, device)
        roles = k2_roles(s)
        for tag, flags in (("K2 GDN-type (oq=True)", {"oq": True}),
                            ("K2 QSA-type (og=True)", {"og": True})):
            key_off = k2.K2Build(tokens=s, tp=TP, trace=False, **flags)
            key_on = k2.K2Build(tokens=s, tp=TP, trace=True, **flags)
            slots = dict(k2.SLOTS)
            slots["__nslot__"] = k2.NSLOT

            def make(key, s_, device_, dbg_):
                return make_k2_runner(key, s_, device_, rank, hw, mw, k2_peers, epoch,
                                       dbg_)

            _bench_one(tag, key_off, key_on, s, make, slots, roles, args.out_dir,
                       rank, args.graph_launches, args.iters)
            dist.barrier()

        roles1 = k1_front_roles()
        k1key_off = k1.K1Build(tokens=s, tp=TP, self_msq=True, front_only=True,
                                trace=False)
        k1key_on = k1.K1Build(tokens=s, tp=TP, self_msq=True, front_only=True,
                               trace=True)
        slots1 = dict(k1.SLOTS)
        slots1["__nslot__"] = k1.NSLOT

        def make1(key, s_, device_, dbg_):
            return make_k1_front_runner(key, s_, device_, dbg_)

        _bench_one("K1 hc_front (front_only)", k1key_off, k1key_on, s, make1, slots1,
                   roles1, args.out_dir, rank, args.graph_launches, args.iters)
        dist.barrier()

    k2_peers.close()
    dist.barrier()


if __name__ == "__main__":
    main()
