# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-kernel timestamps: the instrument IDEAS_5_6_DESIGN.md's Stage 0a and
every later probe (E0, E6.0) reads phase-boundary times off. A build opts in
with ``trace=True`` and a ``dbg`` pointer; every other build's ``stamp``
calls are no-ops (``const_expr`` folds the body away at trace time, so an
off build's device code and timing are unchanged).

``stamp`` is meant to be called at the same program point on every CTA (a
"phase boundary"), not inside a per-wave / per-thread branch: it barriers
first, so every thread's view of the phase is consistent, then thread 0
alone stores the clock. Slot numbers are the caller's: see the per-kernel
tables in IDEAS_5_6_DESIGN.md Stage 0a (e.g. head.py: 0 entry, 8 after
``rows_to_lds`` + ``quant_x``, 9 after ``run_head``, 10 end).
"""

import numpy as np

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu

from vllm.models.kimi_k3.amd.mono.common.ops import memrealtime, rsrc, traced

NSLOT = 16  # stamp slots a CTA (0 .. 15); every kernel here uses < 11


@traced
def stamp(c, slot):
    """CTA-wide point: barrier, then thread 0 stores the 100 MHz realtime's
    low 31 bits (never negative as an Int32, so ``decode`` needs no sign
    fixup) to ``dbg[bid * NSLOT + slot]``. No-op unless the build passed a
    `dbg` pointer (``c.get("dbg") is not None``): a plain Python check at
    trace time (``const_expr``, as ``hc.load_rrms`` gates ``msq_in``), not a
    device-side branch -- an off build emits nothing here at all."""
    if const_expr(c.get("dbg") is not None):
        gpu.barrier()
        if c["tid"] == 0:
            bo.buffer_store(
                fx.Int32(memrealtime() & 0x7FFFFFFF),
                rsrc(c["dbg"]),
                c["bid"] * NSLOT + slot,
            )


def dbg_buffer(device, nslot: int = NSLOT):
    """A fresh ``dbg`` buffer for one launch: int32 ``[256 * nslot]``, this
    rank's device. Callers decide whether to zero it; a slot a live CTA
    never visits (e.g. one head never runs, in a fused kernel with idle
    CTAs) reads back 0, which ``decode`` reports as such -- it is the
    caller's job to know which slots a given build actually stamps."""
    import torch

    return torch.zeros(256 * nslot, dtype=torch.int32, device=device)


def decode(dbg, nslot: int = NSLOT):
    """``dbg`` (a device or CPU int32 tensor/array, ``[256 * nslot]`` or
    already ``[256, nslot]``) -> float64 microseconds, shape ``[256,
    nslot]``. Ticks are the 100 MHz clock's low 31 bits (``stamp`` masks with
    0x7FFFFFFF, so every raw value is already >= 0); 100 ticks = 1 us.

    This only converts units -- it does NOT subtract one slot from another.
    A caller comparing two slots (e.g. ``max(slot9) - min(slot8)``) should
    do the subtraction on the *raw tick* values (or compute it mod 2**31)
    before converting to microseconds if the run is long enough to wrap the
    31-bit counter (~21.47 s at 100 MHz) between the two stamps -- far past
    any one kernel launch's own phases, but worth it for a long warm-up
    loop's first vs. last iteration."""
    try:
        import torch

        if isinstance(dbg, torch.Tensor):
            arr = dbg.detach().to("cpu").numpy()
        else:
            arr = np.asarray(dbg)
    except ImportError:
        arr = np.asarray(dbg)
    arr = arr.reshape(-1, nslot)
    return arr.astype(np.float64) / 100.0


def phase_report(dbg, pairs: dict, nslot: int = NSLOT) -> dict:
    """``pairs``: name -> (hi_slot, lo_slot). Returns name -> (min, median,
    max) microseconds of ``max(hi slots) - min(lo slots)``' per-CTA raw-tick
    difference over every live CTA (one whose ``lo`` slot is nonzero, i.e.
    was actually stamped -- a CTA idle through a phase leaves both its slots
    0 and would otherwise show a spurious huge or negative gap)."""
    us = decode(dbg, nslot)
    ticks = (us * 100.0).round().astype(np.int64)
    out = {}
    for name, (hi, lo) in pairs.items():
        live = ticks[:, lo] != 0
        if not np.any(live):
            out[name] = (float("nan"), float("nan"), float("nan"))
            continue
        d = (ticks[live, hi] - ticks[live, lo]).astype(np.float64) / 100.0
        out[name] = (float(d.min()), float(np.median(d)), float(d.max()))
    return out
