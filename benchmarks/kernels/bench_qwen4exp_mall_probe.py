# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage 0b (P1) of IDEAS_5_6_DESIGN.md: MALL (Infinity Cache) bandwidth and
residency probe on MI355X. A single-GPU FlyDSL streaming-read kernel with the
mono kernels' geometry (grid 256, 512 threads, ``rsrc`` + ``bo.buffer_load``),
answering three open questions `S4` of the design doc flags:

    1. does a plain ``hipMalloc`` (PyTorch) allocation live in MALL at all
    2. does a ``CM_NT`` (nontemporal, what every mono weight load uses) load
       allocate into / get looked up in MALL, or does it bypass it
    3. achievable MALL bandwidth, and whether a same-pool prefetch (``cm=0`` /
       ``CM_DEV``, or a direct-to-LDS DMA) later helps a ``CM_NT`` read hit

Needs MI355X (gfx950) + AITER + FlyDSL; authored and syntax-checked CPU-only
(``python3 -m py_compile``), not run here -- see IDEAS_5_6_DESIGN.md Â§4 Stage
0b and the handoff report for why.

Usage (single GPU, no TP needed -- this is a pure bandwidth probe):
    python3 benchmarks/kernels/bench_qwen4exp_mall_probe.py \
        [--tests p1a,p1b,p1c,p1d,p1e] [--iters 6]

Output: a CSV to stdout (also written to ``runs/out_probe/mall_probe.csv``
if that directory exists), columns ``test,size_mb,mode,cm,depth,b_mb,cm_a,
cm_b,us,gbps``; ``amd-smi static --partition`` (or ``rocm-smi
--showcomputepartition --showmemorypartition``) output first, since an
address is only cached in its home stack's slice outside NPS1/SPX.
"""

import argparse
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BLOCKS = 256  # a CU each, all resident -- the mono execution model
THREADS = 512
LANE_BYTES = 16  # a dwordx4, 16 B a lane a step
STEP_BYTES = THREADS * LANE_BYTES  # 8 KB a CTA a step
GRID_STEP_BYTES = BLOCKS * STEP_BYTES  # 2 MB: one "step" across the whole grid
SENTINEL = 0x7EADBEEF
MiB = 1024 * 1024

WARM_SIZES_MB = (32, 64, 128, 170, 200, 240, 256, 320, 512, 1024)
SURVIVAL_B_MB = (0, 48, 96, 144, 192)


def _steps_for(size_mb: int) -> int:
    """Grid-wide steps to cover ``size_mb`` (rounded down to a whole step;
    ``GRID_STEP_BYTES`` = 2 MB, so this is exact for every size in
    ``WARM_SIZES_MB`` but 170 (84 steps = 168 MB) -- the probe reports the
    bytes it actually reads, not the nominal label, for GB/s."""
    return max(1, (size_mb * MiB) // GRID_STEP_BYTES)


def _mb_of(steps: int) -> float:
    return steps * GRID_STEP_BYTES / MiB


# ---------------------------------------------------------------- the kernel
@dataclass(frozen=True)
class ProbeBuild:
    """One compiled variant. ``steps`` is baked into the build (not passed as
    a runtime argument): a ``range_constexpr`` loop of up to ~512 iterations
    is simpler and safer to get right without a local FlyDSL to compile
    against than a dynamic ``while`` over a traced trip count (the mono
    kernels' own dynamic loops, e.g. ``head.run_head``, carry pipelined
    *operands* across iterations, not a completely static unroll like this
    probe wants) -- the tradeoff is one compiled kernel a (mode, cm, depth,
    steps) combination, and a bigger binary / longer compile at large
    ``steps``, acceptable for a one-off probe."""

    mode: str  # "reg" | "lds"
    cm: int  # CM_NT (2), CM_DEV (16), or 0 (plain / cached)
    depth: int  # operands kept in flight (AHEAD-style software pipeline)
    steps: int  # grid-wide steps this launch reads (GRID_STEP_BYTES each)


_BUILDS: dict = {}


def build(key: ProbeBuild):
    if key in _BUILDS:
        return _BUILDS[key]

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr import range_constexpr
    from flydsl.expr.typing import Int64, T

    from vllm.models.kimi_k3.amd.mono.common.ops import rsrc
    from vllm.models.qwen4_exp.amd.mono import stamps

    _STREAM = fx.Stream(None)
    name = f"qwen4exp_mall_probe_{key.mode}_cm{key.cm}_d{key.depth}_s{key.steps}"
    steps, depth, mode, cm = key.steps, key.depth, key.mode, key.cm
    dwords_per_step = THREADS * 4  # a CTA's step: 512 lanes * 4 dwords (16 B)

    if mode == "lds":
        ring_slots = min(4, max(1, depth))

        @fx.struct
        class Smem:
            # `ring_slots` slots x one lane's 16 B a step, never read back
            ring: fx.Array[fx.Int32, ring_slots * 4, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def probe(buf: Int64, dbg: Int64):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        c = {"tid": tid, "bid": bid, "dbg": dbg}
        stamps.stamp(c, 0)  # entry
        r_buf = rsrc(buf)
        # this CTA's contiguous slice: `steps` grid-wide rounds, bid's own
        # 8 KB window a round, this lane's 16 B inside it (dword units)
        base = bid * dwords_per_step + tid * 4

        def _load(i):
            """``i``: always a plain Python int in [0, steps), never a
            per-thread/traced value -- every lane reads step ``i`` at the
            same program point, so this needs no runtime clamp."""
            at = base + i * (BLOCKS * dwords_per_step)
            return fx.Vector(
                bo.buffer_load(r_buf, at, vec_width=4, dtype=T.i32, cache_modifier=cm)
            )  # fmt: skip

        if mode == "reg":
            ops = [_load(j) for j in range(min(depth, steps))]
            acc = fx.Int32(0)
            for i in range_constexpr(steps):
                v = ops[0]
                acc = acc ^ v[0] ^ v[3]
                if i + depth < steps:
                    ops = ops[1:] + [_load(i + depth)]
                else:
                    ops = ops[1:] + [ops[-1]]  # tail: re-touch the last tile
            # keep `acc` live without perturbing anything real: a sentinel
            # that a few hundred random dwords' xor essentially never hits
            if (tid == 0) & (acc == fx.Int32(SENTINEL)):
                bo.buffer_store(acc, r_buf, 0)
        else:
            # mode == "lds": prefetch-and-discard -- load with `cm`, store to
            # LDS, never read it back. NOTE: this is a global-load-then-
            # ptr_store round trip (the proven ``gemv.rows_to_lds`` pattern),
            # not the zero-VGPR ``raw_ptr_buffer_load_lds`` DMA Â§5.2
            # describes: that intrinsic's per-wave M0-uniform addressing
            # needs a real compiler to get right, and FlyDSL is not
            # available in this sandbox to check it against. The bytes
            # touched and the cache policy applied are the same either way,
            # which is what P1a/P1b/P1d actually measure; swap in the true
            # DMA (Â§5.2) on the node if the extra VGPR / issue-slot cost of
            # this round trip turns out to matter for the P1d comparison.
            lds = fx.SharedAllocator().allocate(Smem).peek()
            for i in range_constexpr(steps):
                v = _load(i)
                slot = i % ring_slots
                fx.ptr_store(v, lds.ring.ptr + slot * 4)
        stamps.stamp(c, 10)  # end

    @flyc.jit
    def launch(buf: Int64, dbg: Int64, stream: fx.Stream = _STREAM):
        probe(buf, dbg).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    _BUILDS[key] = launch
    return launch


def _run(key: ProbeBuild, buf, dbg) -> float:
    """One launch, synchronized: returns microseconds from the stamps buffer
    (``max(end) - min(entry)``, so anything outside the kernel -- including
    the host-side launch overhead -- doesn't pollute it)."""
    import torch

    from vllm.models.qwen4_exp.amd.mono import stamps

    f = build(key)
    dbg.zero_()
    torch.cuda.synchronize()
    f(buf.data_ptr(), dbg.data_ptr(), stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    ticks = stamps.decode(dbg)  # [256, NSLOT] us (already /100 from raw ticks)
    entry, end = ticks[:, 0], ticks[:, 10]
    live = end > 0  # a CTA this build never schedules work on (none, here)
    return float(end[live].max() - entry[live].min())


def _gbps(mb: float, us: float) -> float:
    return (mb * MiB) / (us * 1e-6) / 1e9 if us > 0 else float("nan")


def _median_run(key: ProbeBuild, buf, dbg, iters: int) -> float:
    return statistics.median(_run(key, buf, dbg) for _ in range(iters))


# ---------------------------------------------------------------- tests
def p1a_warm(device, cm: int, depth: int, iters: int, rows: list):
    """Same buffer read ``iters`` times; report the median of reads 2 ..
    iters (drop the first: first-touch / TLB warmup), every size in
    ``WARM_SIZES_MB``. Expect a GB/s knee near 256 MB **iff** this ``cm``
    allocates into MALL."""
    import torch

    from vllm.models.qwen4_exp.amd.mono import stamps

    dbg = stamps.dbg_buffer(device)
    for mb in WARM_SIZES_MB:
        steps = _steps_for(mb)
        real_mb = _mb_of(steps)
        buf = torch.empty(steps * GRID_STEP_BYTES // 4, dtype=torch.int32, device=device)
        for mode in ("reg", "lds"):
            key = ProbeBuild(mode, cm, depth, steps)
            samples = [_run(key, buf, dbg) for _ in range(max(2, iters))]
            us = statistics.median(samples[1:])
            rows.append(("p1a_warm", real_mb, mode, cm, depth, 0, 0, 0, us, _gbps(real_mb, us)))
        del buf
        torch.cuda.empty_cache()


def p1b_cold(device, cm: int, depth: int, rows: list):
    """The same sizes, rotating through 3 buffer copies (plus a 1 GiB
    ``cm=0`` flush read between) so every read is cold: HBM GB/s."""
    import torch

    from vllm.models.qwen4_exp.amd.mono import stamps

    dbg = stamps.dbg_buffer(device)
    flush_steps = _steps_for(1024)
    flush = torch.empty(flush_steps * GRID_STEP_BYTES // 4, dtype=torch.int32, device=device)
    flush_key = ProbeBuild("reg", 0, depth, flush_steps)
    for mb in WARM_SIZES_MB:
        steps = _steps_for(mb)
        real_mb = _mb_of(steps)
        copies = [
            torch.empty(steps * GRID_STEP_BYTES // 4, dtype=torch.int32, device=device)
            for _ in range(3)
        ]
        for mode in ("reg", "lds"):
            key = ProbeBuild(mode, cm, depth, steps)
            samples = []
            for i in range(3):
                _run(flush_key, flush, dbg)
                samples.append(_run(key, copies[i], dbg))
            us = statistics.median(samples)
            rows.append(("p1b_cold", real_mb, mode, cm, depth, 0, 0, 0, us, _gbps(real_mb, us)))
        del copies
        torch.cuda.empty_cache()


def p1c_survival(device, depth: int, rows: list):
    """A=170MB with cm_a, B=b MB with cm_b, A again with cm_a: A's 2nd-read
    time vs p1a (hit) / p1b (miss). b in SURVIVAL_B_MB; (cm_a, cm_b) in
    {(2,2), (0,2), (0,0), (2,0)} (CM_NT=2, CM_DEV/plain=0 here -- 0 and
    CM_DEV both bypass NT's "read once" hint; 0 is simplest to read)."""
    import torch

    from vllm.models.qwen4_exp.amd.mono import stamps

    dbg = stamps.dbg_buffer(device)
    a_steps = _steps_for(170)
    a_mb = _mb_of(a_steps)
    a_buf = torch.empty(a_steps * GRID_STEP_BYTES // 4, dtype=torch.int32, device=device)
    for cm_a, cm_b in ((2, 2), (0, 2), (0, 0), (2, 0)):
        a_key = ProbeBuild("reg", cm_a, depth, a_steps)
        for b_mb in SURVIVAL_B_MB:
            if b_mb > 0:
                b_steps = _steps_for(b_mb)
                b_buf = torch.empty(
                    b_steps * GRID_STEP_BYTES // 4, dtype=torch.int32, device=device
                )
                b_key = ProbeBuild("reg", cm_b, depth, b_steps)
            _run(a_key, a_buf, dbg)  # warm A (not timed)
            if b_mb > 0:
                _run(b_key, b_buf, dbg)
                del b_buf
                torch.cuda.empty_cache()
            us = _run(a_key, a_buf, dbg)  # A's 2nd read, timed
            rows.append(
                ("p1c_survival", a_mb, "reg", 0, depth, b_mb, cm_a, cm_b, us, _gbps(a_mb, us))
            )


def p1d_prefetch_consume(device, depth: int, rows: list):
    """A read with mode=lds + cm_x (the prefetch, discarded), then A read
    with mode=reg + cm=CM_NT (the head's own policy): does the lds prefetch
    make the later NT read hit?"""
    import torch

    from vllm.models.qwen4_exp.amd.mono import stamps

    CM_NT = 2
    dbg = stamps.dbg_buffer(device)
    a_steps = _steps_for(170)
    a_mb = _mb_of(a_steps)
    a_buf = torch.empty(a_steps * GRID_STEP_BYTES // 4, dtype=torch.int32, device=device)
    for cm_x in (0, CM_NT):
        pf_key = ProbeBuild("lds", cm_x, depth, a_steps)
        read_key = ProbeBuild("reg", CM_NT, depth, a_steps)
        _run(pf_key, a_buf, dbg)
        us = _run(read_key, a_buf, dbg)
        rows.append(
            ("p1d_prefetch", a_mb, "reg", CM_NT, depth, 0, cm_x, CM_NT, us, _gbps(a_mb, us))
        )


def p1e_floor(device, depth: int, rows: list):
    """reg mode, depth 8, 170 MB cold: a perfectly pipelined head's lower
    bound (no GEMV / row-reduce / barriers, loads only)."""
    import torch

    from vllm.models.qwen4_exp.amd.mono import stamps

    CM_NT = 2
    dbg = stamps.dbg_buffer(device)
    steps = _steps_for(170)
    mb = _mb_of(steps)
    copies = [
        torch.empty(steps * GRID_STEP_BYTES // 4, dtype=torch.int32, device=device)
        for _ in range(3)
    ]
    flush_steps = _steps_for(1024)
    flush = torch.empty(flush_steps * GRID_STEP_BYTES // 4, dtype=torch.int32, device=device)
    flush_key = ProbeBuild("reg", 0, depth, flush_steps)
    key = ProbeBuild("reg", CM_NT, depth, steps)
    samples = []
    for i in range(3):
        _run(flush_key, flush, dbg)
        samples.append(_run(key, copies[i], dbg))
    us = statistics.median(samples)
    rows.append(("p1e_floor", mb, "reg", CM_NT, depth, 0, 0, 0, us, _gbps(mb, us)))


# ---------------------------------------------------------------- plumbing
def _partition_mode() -> str:
    """``amd-smi``'s (or ``rocm-smi``'s) compute/memory partition mode: an
    address is cached only in its home HBM stack's slice outside NPS1/SPX,
    so this belongs at the top of every MALL probe log."""
    last = "amd-smi / rocm-smi not found"
    for cmd in (
        ["amd-smi", "static", "--partition"],
        ["rocm-smi", "--showcomputepartition", "--showmemorypartition"],
    ):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if out.returncode == 0:
                return f"$ {' '.join(cmd)}\n{out.stdout}"
            last = f"$ {' '.join(cmd)} -> exit {out.returncode}: {out.stderr.strip()}"
        except (OSError, subprocess.SubprocessError) as exc:
            last = f"$ {' '.join(cmd)} -> {exc!r}"
    return last


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tests", type=str, default="p1a,p1b,p1c,p1d,p1e")
    p.add_argument("--cm", type=str, default="0,2", help="CM_DEV/plain=0, CM_NT=2")
    p.add_argument("--depth", type=str, default="2,4,8")
    p.add_argument("--iters", type=int, default=6)
    p.add_argument(
        "--out", type=str, default="runs/out_probe/mall_probe.csv",
        help="also write the CSV here if the parent directory exists",
    )  # fmt: skip
    return p.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    tests = set(args.tests.split(","))
    cms = [int(v) for v in args.cm.split(",")]
    depths = [int(v) for v in args.depth.split(",")]
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    print(_partition_mode())
    rows = []
    for cm in cms:
        for depth in depths:
            if "p1a" in tests:
                p1a_warm(device, cm, depth, args.iters, rows)
            if "p1b" in tests:
                p1b_cold(device, cm, depth, rows)
    if "p1c" in tests:
        p1c_survival(device, depths[0], rows)
    if "p1d" in tests:
        p1d_prefetch_consume(device, depths[0], rows)
    if "p1e" in tests:
        p1e_floor(device, depths[-1] if len(depths) > 1 else 8, rows)

    header = "test,size_mb,mode,cm,depth,b_mb,cm_a,cm_b,us,gbps"
    lines = [header] + [",".join(str(x) for x in r) for r in rows]
    print("\n".join(lines))
    out_path = Path(args.out)
    if out_path.parent.is_dir():
        out_path.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
