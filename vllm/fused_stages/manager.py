# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection, collective acceptance, step planning and lifecycle.

One ``FusedStageManager`` per worker, created by the model runner at the end
of ``load_model`` (so everything a provider allocates in ``create`` is counted
by memory profiling before the KV cache is sized).

Rank agreement is by construction:

1. Acceptance is all-gathered over the TP group; any refusal on any rank makes
   every rank use the reference path for that (stage, provider).
2. Per-step decisions are a pure function of the ``StepKey`` derived from the
   cudagraph ``BatchDescriptor``, which every TP rank computes identically and
   which also keys graph replay. Providers' predicates are never called with
   metadata, tensors or global state.
3. ``kernel_config.fused_stages.verify_rank_consistency`` additionally
   all-gathers a digest of every newly computed plan (debug mode).
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

import vllm.envs as envs
from vllm.fused_stages import state
from vllm.fused_stages.api import (
    REFERENCE,
    Acceptance,
    DeploymentInfo,
    Feature,
    LayerBinding,
    Refusal,
    Regime,
    StageClaim,
    StageContext,
    StageProvider,
    StageRuntime,
    StageSpec,
    StepKey,
    TPContext,
)
from vllm.fused_stages.layouts import UNTAGGED
from vllm.fused_stages.registry import discover_providers
from vllm.fused_stages.views import build_view, has_view_builder
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.fused_stages.attach import StageHandle

logger = init_logger(__name__)


# --------------------------------------------------------------- plan types


@dataclass(frozen=True)
class Assignment:
    provider_id: str
    runtime: StageRuntime
    step_view: str | None
    kv_hooks: bool
    exclusive: bool


@dataclass(frozen=True)
class StepPlan:
    key: StepKey
    assignments: Mapping[tuple[str, int], Assignment]

    @property
    def exclusive(self) -> bool:
        return any(a.exclusive for a in self.assignments.values())


@dataclass
class FusedStep:
    """Mutable per-forward state (one per ForwardContext)."""

    plan: StepPlan
    manager: FusedStageManager
    begun: set[int] = field(default_factory=set)

    def fallback(self, spec: StageSpec, reason: str) -> None:
        self.manager.counters[f"{spec.id}:fallback:{reason}"] += 1
        return None

    def ran(self, spec: StageSpec) -> None:
        self.manager.counters[f"{spec.id}:ran"] += 1


# ------------------------------------------------------------ bookkeeping


@dataclass
class _Stage:
    spec: StageSpec
    handles: dict[int, StageHandle] = field(default_factory=dict)


@dataclass
class _Selected:
    stage: _Stage
    provider: StageProvider
    claim: StageClaim
    acceptance: Acceptance
    layer_ids: frozenset[int]
    step_view: str | None
    cache_view: str | None
    kv_hooks: bool
    runtime: StageRuntime | None = None
    key_ok: dict[StepKey, bool] = field(default_factory=dict)

    def accepts(self, key: StepKey) -> bool:
        ok = self.key_ok.get(key)
        if ok is None:
            ok = bool(key.regime & self.claim.regimes)
            if key.ubatched and (
                Feature.DBO not in self.claim.supports or self.claim.exclusive_gpu
            ):
                ok = False  # enforced here, not left to each predicate
            if key.has_lora and Feature.LORA not in self.claim.supports:
                ok = False
            ok = ok and self.acceptance.steps(key)
            if self.kv_hooks and key.regime != Regime.EAGER:
                # v0: layer-wise KV-connector hooks are host calls; they are
                # not replayed by a captured graph.
                ok = False
            self.key_ok[key] = ok
        return ok


def _views_of_kind(views: Sequence[str] | frozenset[str], kind: str) -> list[str]:
    return [v for v in views if f".{kind}@" in v]


# ------------------------------------------------------------------ manager


class FusedStageManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        device: torch.device,
        uniform_decode_query_len: int,
        handles: Sequence[StageHandle],
    ) -> None:
        self.vllm_config = vllm_config
        self.config = vllm_config.kernel_config.fused_stages
        self.model = model
        self.device = device
        self.uniform_decode_query_len = uniform_decode_query_len
        self.stages: dict[str, _Stage] = {}
        for handle in handles:
            for spec in handle.stages.values():
                st = self.stages.setdefault(spec.id, _Stage(spec))
                if st.spec != spec:
                    raise ValueError(f"conflicting StageSpecs for {spec.id}")
                st.handles[handle.layer_id] = handle
        self.selected: list[_Selected] = []
        self.counters: Counter[str] = Counter()
        self.cache_generation = 0
        self._caches_bound = False
        self._warmed: set[tuple[int, StepKey]] = set()
        self._plans: dict[StepKey, StepPlan | None] = {}
        self._deployment: DeploymentInfo | None = None

    # ------------------------------------------------------------ creation

    @classmethod
    def maybe_create(
        cls,
        vllm_config: VllmConfig,
        model: nn.Module,
        device: torch.device,
        uniform_decode_query_len: int,
    ) -> FusedStageManager | None:
        """Drain the handles attached during model construction and, if any
        non-reference provider is configured, select and create providers."""
        handles = list(state._PENDING_HANDLES)
        state._PENDING_HANDLES.clear()
        if not handles:
            return None
        cfg = vllm_config.kernel_config.fused_stages
        if not any(
            p != REFERENCE
            for h in handles
            for spec in h.stages.values()
            for p in cfg.priority_for(spec.id)
        ):
            return None
        manager = cls(vllm_config, model, device, uniform_decode_query_len, handles)
        manager.initialize()
        if not manager.selected:
            return None
        state.set_active_manager(manager)
        return manager

    def initialize(self) -> None:
        dep = self._deployment = self._deployment_info()
        providers = discover_providers()
        for stage_id in sorted(self.stages):  # same order on every rank
            stage = self.stages[stage_id]
            remaining = set(stage.handles)
            for pid in self.config.priority_for(stage_id):
                if pid == REFERENCE or not remaining:
                    break
                provider = providers.get(pid)
                if provider is None:
                    self._log_refusal(stage_id, pid, "provider not installed")
                    continue
                sel = self._select(stage, provider, dep, sorted(remaining))
                if sel is not None:
                    self.selected.append(sel)
                    remaining -= sel.layer_ids

        tp = self._tp_context()
        for sel in self.selected:  # same order on every rank: collectives OK
            before = torch.cuda.memory_allocated(self.device)
            ctx = StageContext(
                deployment=dep,
                tp=tp,
                device=self.device,
                cache_dir=os.path.join(
                    envs.VLLM_CACHE_ROOT,
                    "fused_stages",
                    sel.provider.provider_id,
                    f"tp{tp.size}_rank{tp.rank}",
                ),
            )
            os.makedirs(ctx.cache_dir, exist_ok=True)
            sel.runtime = sel.provider.create(sel.claim, ctx)
            sel.runtime.bind_layers(
                [
                    sel.stage.handles[i].binding_for(sel.stage.spec)
                    for i in sorted(sel.layer_ids)
                ]
            )
            req = sel.runtime.memory_request()
            used = torch.cuda.memory_allocated(self.device) - before
            logger.info(
                "Fused stage %s -> %s %s on %d layers %s (declared %.1f MiB device "
                "+ %.1f MiB peer; torch-visible %.1f MiB)%s",
                sel.stage.spec.versioned_id,
                sel.provider.provider_id,
                sel.provider.provider_version,
                len(sel.layer_ids),
                _ranges(sel.layer_ids),
                req.device_bytes / 2**20,
                req.peer_bytes / 2**20,
                used / 2**20,
                f" [{sel.acceptance.note}]" if sel.acceptance.note else "",
            )

    def _select(
        self,
        stage: _Stage,
        provider: StageProvider,
        dep: DeploymentInfo,
        layer_ids: list[int],
    ) -> _Selected | None:
        spec = stage.spec
        claims = [c for c in provider.claims() if c.stage == spec.id]
        if not claims:
            self._log_refusal(spec.id, provider.provider_id, "no claim for stage")
            return None
        claim = claims[0]
        bindings = [stage.handles[i].binding_for(spec) for i in layer_ids]
        if not claim.draft_layers:
            bindings = [b for b in bindings if not b.draft]
            if not bindings:
                self._log_refusal(spec.id, provider.provider_id, "only draft layers")
                return None

        reason = self._generic_refusal(claim, spec, dep, bindings)
        acceptance: Acceptance | None = None
        if reason is None:
            try:
                acceptance = provider.accept(claim, spec, dep, bindings)
            except Refusal as err:
                reason = str(err) or "refused"
        local = (
            None
            if acceptance is None
            else sorted(acceptance.layer_ids & set(layer_ids))
        )

        # Collective: every TP rank must reach the same verdict.
        gathered = self._all_gather((provider.provider_version, local, reason))
        versions = {g[0] for g in gathered}
        reasons = sorted({g[2] for g in gathered if g[2] is not None})
        if len(versions) > 1:
            reasons.append(f"provider versions differ across ranks: {versions}")
        if reasons or any(g[1] is None for g in gathered):
            self._log_refusal(spec.id, provider.provider_id, "; ".join(reasons))
            return None
        common = frozenset.intersection(*(frozenset(g[1]) for g in gathered))
        if not common:
            self._log_refusal(spec.id, provider.provider_id, "accepted no layers")
            return None
        assert acceptance is not None

        views = [v for v in claim.views if v in spec.views]
        step_views = _views_of_kind(views, "step")
        cache_views = _views_of_kind(views, "cache")
        return _Selected(
            stage=stage,
            provider=provider,
            claim=claim,
            acceptance=acceptance,
            layer_ids=common,
            step_view=step_views[0] if step_views else None,
            cache_view=cache_views[0] if cache_views else None,
            kv_hooks=spec.subsumes_attention and Feature.KV_TRANSFER in dep.features,
        )

    def _generic_refusal(
        self,
        claim: StageClaim,
        spec: StageSpec,
        dep: DeploymentInfo,
        bindings: Sequence[LayerBinding],
    ) -> str | None:
        """Checks every provider would otherwise re-implement."""
        if spec.version not in claim.stage_versions:
            return f"stage version {spec.version} not in {sorted(claim.stage_versions)}"
        if Regime.COMPILED in dep.regimes:
            return "model is torch.compile'd (unsupported in v0)"
        if not (claim.regimes & dep.regimes):
            return f"no common regime (engine {dep.regimes}, claim {claim.regimes})"
        unsupported = set(dep.features) - set(claim.supports)
        if claim.exclusive_gpu:
            unsupported |= dep.features & {Feature.DBO, Feature.KV_OFFLOAD}
        if unsupported:
            return "unsupported features: " + ", ".join(
                sorted(f.value for f in unsupported)
            )
        for view in claim.views:
            if view not in spec.views:
                return f"view {view} not offered by the model"
            if not has_view_builder(view, spec.family):
                return f"no builder for view {view}"
        for b in bindings:
            for role, accepted in claim.layouts.items():
                ref = b.weights.get(role)
                layout = UNTAGGED if ref is None else ref.layout
                if layout not in accepted:
                    return (
                        f"layer {b.layer_id} role {role}: layout {layout} not in "
                        f"{sorted(accepted)}"
                    )
        return None

    # ------------------------------------------------------------ per step

    def step_key(
        self, batch_descriptor: Any, cudagraph_mode: Any, ubatch_slices: Any
    ) -> StepKey | None:
        from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
        from vllm.config import CUDAGraphMode

        if batch_descriptor is None:
            return None
        if cudagraph_mode == CUDAGraphMode.FULL:
            regime = Regime.FULL_GRAPH
        elif cudagraph_mode == CUDAGraphMode.PIECEWISE:
            regime = (
                Regime.BREAKABLE_GRAPH
                if is_breakable_cudagraph_enabled()
                else Regime.PIECEWISE_GRAPH
            )
        else:
            regime = Regime.EAGER
        num_tokens = batch_descriptor.num_tokens
        num_reqs = batch_descriptor.num_reqs or 0
        uniform = batch_descriptor.uniform and num_reqs > 0
        query_len = num_tokens // num_reqs if uniform else 0
        return StepKey(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            query_len=query_len,
            uniform_decode=uniform
            and query_len == self.uniform_decode_query_len
            and query_len * num_reqs == num_tokens,
            regime=regime,
            has_lora=batch_descriptor.has_lora,
            ubatched=ubatch_slices is not None,
        )

    def plan_step(
        self,
        batch_descriptor: Any,
        cudagraph_mode: Any,
        ubatch_slices: Any,
        attn_metadata: Any,
    ) -> StepPlan | None:
        if attn_metadata is None:  # profile / dummy runs without metadata
            return None
        key = self.step_key(batch_descriptor, cudagraph_mode, ubatch_slices)
        if key is None:
            return None
        if key in self._plans:
            return self._plans[key]
        assignments: dict[tuple[str, int], Assignment] = {}
        for sel in self.selected:
            if sel.cache_view is not None and not self._caches_bound:
                continue
            if not sel.accepts(key):
                continue
            assert sel.runtime is not None
            a = Assignment(
                provider_id=sel.provider.provider_id,
                runtime=sel.runtime,
                step_view=sel.step_view,
                kv_hooks=sel.kv_hooks,
                exclusive=sel.claim.exclusive_gpu,
            )
            for layer_id in sel.layer_ids:
                assignments.setdefault((sel.stage.spec.id, layer_id), a)
        plan = StepPlan(key, assignments) if assignments else None
        if self.config.verify_rank_consistency:
            self._verify(key, plan)
        if self._caches_bound or not any(s.cache_view for s in self.selected):
            self._plans[key] = plan  # don't memoize pre-bind decisions
        return plan

    def new_step(self, plan: StepPlan | None) -> FusedStep | None:
        return None if plan is None else FusedStep(plan, self)

    # ----------------------------------------------------------- lifecycle

    def warmup(self, capture_descs: Sequence[tuple[Any, Sequence[Any]]]) -> None:
        """Before graph capture: JIT + scratch for every key to be captured."""
        keys_by_sel: dict[int, list[StepKey]] = {}
        for mode, descs in capture_descs:
            for desc in descs:
                key = self.step_key(desc, mode, None)
                if key is None:
                    continue
                for i, sel in enumerate(self.selected):
                    if sel.accepts(key) and (i, key) not in self._warmed:
                        self._warmed.add((i, key))
                        keys_by_sel.setdefault(i, []).append(key)
        for i, keys in keys_by_sel.items():
            sel = self.selected[i]
            assert sel.runtime is not None
            start = time.perf_counter()
            sel.runtime.warmup(keys)
            logger.info(
                "Fused stage %s (%s) warmed up %d step keys in %.1f s",
                sel.stage.spec.id,
                sel.provider.provider_id,
                len(keys),
                time.perf_counter() - start,
            )

    def bind_caches(self) -> None:
        """After every ``initialize_kv_cache`` (profiling, then real)."""
        self.cache_generation += 1
        for sel in self.selected:
            if sel.cache_view is None:
                continue
            assert sel.runtime is not None
            views = {}
            for layer_id in sel.layer_ids:
                view = build_view(
                    sel.cache_view,
                    sel.stage.spec.family,
                    sel.stage.handles[layer_id].module,
                    None,
                )
                if view is None:
                    raise RuntimeError(
                        f"cache view {sel.cache_view} unavailable for layer "
                        f"{layer_id} after KV-cache allocation"
                    )
                views[layer_id] = view
            sel.runtime.bind_caches(self.cache_generation, views)
        self._caches_bound = True
        self._plans.clear()

    def unbind_caches(self) -> None:
        """Before the KV cache is freed (graphs already destroyed)."""
        if not self._caches_bound:
            return
        for sel in self.selected:
            if sel.cache_view is not None and sel.runtime is not None:
                sel.runtime.unbind_caches(self.cache_generation)
        self._caches_bound = False
        self._plans.clear()

    def rebind_layers(self) -> None:
        """After reload_weights / online weight updates."""
        for sel in self.selected:
            assert sel.runtime is not None
            sel.runtime.bind_layers(
                [
                    sel.stage.handles[i].binding_for(sel.stage.spec)
                    for i in sorted(sel.layer_ids)
                ]
            )

    def on_sleep(self, level: int) -> None:
        for sel in self.selected:
            if sel.runtime is not None:
                sel.runtime.on_sleep(level)

    def on_wake(self) -> None:
        for sel in self.selected:
            if sel.runtime is not None:
                sel.runtime.on_wake()

    def close(self) -> None:
        for sel in reversed(self.selected):
            if sel.runtime is not None:
                sel.runtime.close()
                sel.runtime = None
        if state.get_active_manager() is self:
            state.set_active_manager(None)
        if self.counters:
            logger.info("Fused-stage counters: %s", dict(self.counters))

    # ------------------------------------------------------------- helpers

    def _deployment_info(self) -> DeploymentInfo:
        from vllm.platforms import current_platform

        cfg = self.vllm_config
        pc, cc, sc = cfg.parallel_config, cfg.cache_config, cfg.scheduler_config
        mode = cfg.compilation_config.cudagraph_mode
        regimes = Regime.EAGER
        if mode is not None and mode.has_full_cudagraphs():
            regimes |= Regime.FULL_GRAPH
        if mode is not None and mode.has_piecewise_cudagraphs():
            from vllm.compilation.breakable_cudagraph import (
                is_breakable_cudagraph_enabled,
            )

            regimes |= (
                Regime.BREAKABLE_GRAPH
                if is_breakable_cudagraph_enabled()
                else Regime.PIECEWISE_GRAPH
            )
        if any(
            getattr(m, "do_not_compile", True) is False for m in self.model.modules()
        ):
            regimes |= Regime.COMPILED

        features: set[Feature] = set()
        if cfg.lora_config is not None:
            features.add(Feature.LORA)
        if cfg.kv_transfer_config is not None:
            features.add(Feature.KV_TRANSFER)
        if cc.kv_offloading_size:
            features.add(Feature.KV_OFFLOAD)
        if cfg.model_config.enable_sleep_mode:
            features.add(Feature.SLEEP_MODE)
        if cfg.weight_transfer_config is not None:
            features.add(Feature.WEIGHT_TRANSFER)
        if cc.enable_prefix_caching:
            features.add(Feature.PREFIX_CACHING)
        if pc.use_ubatching:
            features.add(Feature.DBO)
        if sc.async_scheduling:
            features.add(Feature.ASYNC_SCHEDULING)
        if envs.VLLM_BATCH_INVARIANT:
            features.add(Feature.BATCH_INVARIANT)
        if pc.pipeline_parallel_size > 1:
            features.add(Feature.PIPELINE_PARALLEL)
        if pc.enable_expert_parallel:
            features.add(Feature.EXPERT_PARALLEL)
        if pc.data_parallel_size > 1:
            features.add(Feature.DATA_PARALLEL)
        if pc.decode_context_parallel_size > 1 or pc.prefill_context_parallel_size > 1:
            features.add(Feature.CONTEXT_PARALLEL)
        if pc.enable_eplb:
            features.add(Feature.EPLB)

        props = torch.cuda.get_device_properties(self.device)
        if current_platform.is_rocm():
            arch = props.gcnArchName.split(":")[0]
        else:
            arch = f"sm_{props.major}{props.minor}"
        spec = cfg.speculative_config
        return DeploymentInfo(
            platform=current_platform.device_name,
            arch=arch,
            num_compute_units=props.multi_processor_count,
            tp_size=pc.tensor_parallel_size,
            pp_size=pc.pipeline_parallel_size,
            dp_size=pc.data_parallel_size,
            max_model_len=cfg.model_config.max_model_len,
            max_num_seqs=sc.max_num_seqs,
            max_num_batched_tokens=sc.max_num_batched_tokens,
            kv_cache_dtype=cc.cache_dtype,
            block_size=cc.block_size,
            spec_method=None if spec is None else spec.method,
            num_spec_tokens=0 if spec is None else spec.num_speculative_tokens,
            regimes=regimes,
            features=frozenset(features),
            hf_config=cfg.model_config.hf_config,
        )

    def _tp_context(self) -> TPContext:
        from vllm.distributed import get_tp_group

        g = get_tp_group()
        return TPContext(g.rank_in_group, g.world_size, g.device_group, g.cpu_group)

    def _all_gather(self, obj: Any) -> list[Any]:
        import torch.distributed as dist

        from vllm.distributed import get_tp_group

        g = get_tp_group()
        if g.world_size == 1:
            return [obj]
        out: list[Any] = [None] * g.world_size
        dist.all_gather_object(out, obj, group=g.cpu_group)
        return out

    def _verify(self, key: StepKey, plan: StepPlan | None) -> None:
        digest = hashlib.sha256(
            repr(
                (
                    key,
                    None
                    if plan is None
                    else sorted(
                        (k, a.provider_id) for k, a in plan.assignments.items()
                    ),
                )
            ).encode()
        ).hexdigest()
        digests = set(self._all_gather(digest))
        if len(digests) > 1:
            raise RuntimeError(
                f"Fused-stage plans diverge across TP ranks for {key}; an "
                "acceptance predicate is not a pure function of the StepKey."
            )

    def _log_refusal(self, stage_id: str, provider_id: str, reason: str) -> None:
        logger.info(
            "Fused stage %s: provider %s not used (%s); reference path.",
            stage_id,
            provider_id,
            reason,
        )


def _ranges(ids: frozenset[int]) -> str:
    out, run = [], []
    for i in sorted(ids):
        if run and i != run[-1] + 1:
            out.append(run)
            run = []
        run.append(i)
    if run:
        out.append(run)
    return ",".join(f"{r[0]}" if len(r) == 1 else f"{r[0]}-{r[-1]}" for r in out)
