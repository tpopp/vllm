# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conformance kit for fused-stage providers.

Providers run these in their own CI; vLLM runs them nightly against pinned
provider versions, so a vLLM change that breaks a provider turns a job red
instead of silently falling back to the reference path.

Levels:

* ``check_claims``: static, CPU-only. Claims reference real stages, the
  acceptance predicate is pure and total over every key vLLM can dispatch.
* ``compare_to_reference``: one span on random or recorded inputs against
  the model's own code for that span, stage by stage, with tolerances.
* ``replay_stress``: graph-capture race stress. Alternates step widths in
  one captured graph and checks every replay bit-for-bit against the first.
  (This is the test that found the scratch-sharing race in vllm#60397.)

Model-level accuracy (GPQA-D, AIME25, RULER/NIAH, per RFC #59665) stays a
per-provider release gate outside this kit.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import torch

from vllm.fused_stages.api import (
    DeploymentInfo,
    Feature,
    LayerBinding,
    Refusal,
    Regime,
    StageProvider,
    StageSpec,
    StepKey,
)

# --------------------------------------------------------------- static


def candidate_keys(
    max_tokens: int,
    query_lens: Iterable[int],
    regimes: Iterable[Regime] = (Regime.EAGER, Regime.FULL_GRAPH),
) -> list[StepKey]:
    """Every StepKey vLLM could hand a provider up to ``max_tokens`` rows."""
    keys = []
    for regime, q in itertools.product(regimes, query_lens):
        for reqs in range(1, max_tokens // q + 1):
            keys.append(StepKey(reqs * q, reqs, q, True, regime))
    for regime, n in itertools.product(regimes, range(1, max_tokens + 1)):
        keys.append(StepKey(n, 0, 0, False, regime))  # mixed / prefill
        keys.append(StepKey(n, 1, 0, False, regime, ubatched=True))
    return keys


@dataclass
class ClaimReport:
    stage: str
    accepted_layers: frozenset[int]
    accepted_keys: int
    refusal: str | None = None


def check_claims(
    provider: StageProvider,
    specs: Mapping[str, StageSpec],
    deployment: DeploymentInfo,
    layers: Mapping[str, Sequence[LayerBinding]],
    keys: Sequence[StepKey],
) -> list[ClaimReport]:
    """Static checks: claims name known stages and versions, ``accept``
    allocates nothing, and the step predicate is pure and total."""
    reports = []
    for claim in provider.claims():
        spec = specs.get(claim.stage)
        assert spec is not None, f"claim for unknown stage {claim.stage}"
        assert spec.version in claim.stage_versions, (
            f"{claim.stage}: model offers v{spec.version}, provider "
            f"{sorted(claim.stage_versions)}"
        )
        before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        try:
            acc = provider.accept(claim, spec, deployment, layers[claim.stage])
        except Refusal as err:
            reports.append(ClaimReport(claim.stage, frozenset(), 0, str(err)))
            continue
        if torch.cuda.is_available():
            assert torch.cuda.memory_allocated() == before, "accept() allocated"
        first = [acc.steps(k) for k in keys]
        second = [acc.steps(k) for k in reversed(keys)][::-1]
        assert first == second, "step predicate is not a pure function of StepKey"
        reports.append(ClaimReport(claim.stage, acc.layer_ids, sum(first)))
    return reports


def deployment_for_tests(**overrides) -> DeploymentInfo:
    """A plausible DeploymentInfo for CPU tests (MI355X, TP2, DSpark k=5)."""
    base = DeploymentInfo(
        platform="rocm",
        arch="gfx950",
        num_compute_units=256,
        tp_size=2,
        pp_size=1,
        dp_size=1,
        max_model_len=262144,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        kv_cache_dtype="fp8_ds_mla",
        block_size=256,
        spec_method="dspark",
        num_spec_tokens=5,
        regimes=Regime.EAGER | Regime.FULL_GRAPH,
        features=frozenset({Feature.ASYNC_SCHEDULING}),
        hf_config=None,
    )
    return replace(base, **overrides)


# ------------------------------------------------------------- numerics


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float
    min_cosine: float = 0.999
    """Per-row cosine floor; catches localized corruption atol/rtol hide."""
    max_bad_rows: float = 0.0
    """Fraction of rows allowed below ``min_cosine``. MoE stages need a small
    allowance: top-k routing near-ties legitimately pick different experts
    (vllm#60397 measured 26/1260 rows, all with 6th/7th score gap < 3.2e-3)."""


def _row_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.float().flatten(1), b.float().flatten(1)
    return torch.nn.functional.cosine_similarity(a, b, dim=1)


def compare_to_reference(
    spec: StageSpec,
    run_provider: Callable[[Mapping[str, torch.Tensor]], Sequence[torch.Tensor]],
    run_reference: Callable[[Mapping[str, torch.Tensor]], Sequence[torch.Tensor]],
    inputs: Mapping[str, torch.Tensor],
    tolerances: Mapping[str, Tolerance],
) -> dict[str, float]:
    """Run both on the same entry-seam inputs; compare every exit tensor.
    Returns the minimum row cosine per exit tensor."""
    got = run_provider({k: v.clone() for k, v in inputs.items()})
    want = run_reference({k: v.clone() for k, v in inputs.items()})
    result = {}
    for name, g, w in zip(spec.exit.names, got, want):
        tol = tolerances.get(name, Tolerance(1e-2, 1e-2))
        cos = _row_cosine(g, w)
        bad = (cos < tol.min_cosine).float().mean().item()
        assert bad <= tol.max_bad_rows, (
            f"{spec.id}.{name}: {bad:.2%} rows below cosine {tol.min_cosine}"
        )
        if tol.max_bad_rows == 0.0:
            torch.testing.assert_close(
                g.float(), w.float(), atol=tol.atol, rtol=tol.rtol
            )
        result[name] = cos.min().item()
    return result


# ----------------------------------------------------------- race stress


def replay_stress(
    make_step: Callable[[int], Callable[[], Sequence[torch.Tensor]]],
    widths: Sequence[int],
    replays: int = 10_000,
) -> None:
    """Capture one graph that runs ``make_step(w)()`` for every width in
    ``widths`` back to back, replay it ``replays`` times, and require every
    output of every replay to match the first replay bit for bit. Catches
    state shared across widths, missing fences and stale mailbox reads."""
    steps = [make_step(w) for w in widths]
    for s in steps:  # eager warmup (JIT, scratch)
        s()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [s() for s in steps]
    graph.replay()
    torch.cuda.synchronize()
    golden = [[t.clone() for t in o] for o in outs]
    for i in range(replays):
        graph.replay()
        if i % 64 == 63 or i == replays - 1:
            torch.cuda.synchronize()
            for w, o, g in zip(widths, outs, golden):
                for t, gt in zip(o, g):
                    if not torch.equal(t, gt):
                        raise AssertionError(f"replay {i} width {w} diverged")
