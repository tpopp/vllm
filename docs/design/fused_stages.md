# Fused Stages (Mono Kernels / Megakernels)

!!! warning
    Example / RFC companion. Interface major version 0; nothing here is a
    stability promise yet.

A *fused stage* is a span of a model's forward pass that a single provider
implementation may replace as a whole: for example a persistent kernel that
runs an entire decoder layer (attention, TP all-reduce, MoE, all-reduce) for
small decode steps. vLLM declares the span, owns the reference
implementation (the model's own code), and decides per step which provider
runs it. Providers can live in vLLM, in a vendor library, or in any
pip-installable package.

This is [vLLM IR](vllm_ir.md)'s pattern (one declared op, several registered
implementations, priority-based selection, conformance against a reference)
at layer granularity. Layer granularity adds what an IR op never has:
weights, KV caches, attention metadata, in-kernel collectives, persistent
state and a lifecycle. Those are what this design is about.

## Why

Between 2026-10-01 and 10-08, six mono-kernel PRs targeted vLLM (DSv4.1
#60397, three MiniMax-M3 ports #59705/#59653/#60528, GLM-5 #60216, and two
Kimi-K3 variants #60683/#60793), each with its own integration. The
kernel packages import nothing from vLLM; the coupling is entirely in
per-model adapters that read 50–85 vLLM internals each (processed weight
layouts, KV byte layouts, attention metadata, private attributes) and
re-derive the same safety rules (rank agreement, graph-capture
restrictions, lifecycle). Fused stages make that coupling explicit,
versioned and testable, and let several implementations of the same span
coexist.

## Concepts

| Term | Owner | What it is |
|---|---|---|
| `StageSpec` | model (vLLM) | a named span: entry/exit `Seam`s, scope, collectives, weight roles, views |
| `Seam` | model | a program point and its exact boundary tensors (`"M"` = step rows) |
| `LayerBinding` | model | per layer: weights by role (+ layout tags), constants |
| layout tag | quant method / kernel | `"mxfp4.aiter_a4w4.separated@1"` on processed weights |
| view | attention backend / model | versioned, semantic description of KV-cache and per-step metadata |
| `StageProvider` | anyone | claims stages; `accept` (static), `create` (runtime) |
| `StageRuntime` | provider | stateful per-rank implementation with lifecycle hooks |
| `StepKey` | vLLM | provider-neutral step identity derived from the cudagraph key |

Provider code imports only `vllm.fused_stages.api` (and view dataclasses).

## How a step is planned

```text
load_model ─► attach_stages() handles drained ─► per stage, per provider in
priority order: generic checks (versions, regime, features, layouts, views)
─► provider.accept() ─► all_gather over TP ─► any refusal = reference everywhere
─► provider.create() (collective-safe order) ─► bind_layers()

create_forward_context(batch_descriptor, cudagraph_mode, ubatch_slices, md)
─► StepKey ─► memoized plan {(stage, layer) -> runtime} ─► ForwardContext.fused_step

DecoderLayer.forward: out = stages.try_run(SPEC, **seam) ; None -> reference
```

The plan is a **pure function of the StepKey**, which vLLM derives from the
same `BatchDescriptor` that keys CUDA-graph replay (for eager steps, which
replay nothing, from the TP-replicated batch shape: request count and
uniform-decode flag). Two properties follow by construction:

* **Replay safety.** A Python decision made during capture is baked into the
  graph. Because the decision depends only on the replay key, a replayed
  graph never runs a different path than the one captured.
* **Rank agreement.** Kernels with in-kernel collectives deadlock (not
  mis-compute) if TP ranks take different paths. Every rank computes the same
  key, and acceptance was all-gathered. `fused_stages.verify_rank_consistency`
  additionally all-gathers a digest of every new plan.

Anything not in the key (context length, number of index blocks...) must be
enforced statically by the provider against `DeploymentInfo`
(e.g. `max_model_len`). Providers never get per-step callbacks with
metadata.

The framework itself refuses, per key: regimes the claim does not support,
DBO-microbatched keys unless the claim supports DBO (never for exclusive
claims), LoRA keys unless supported, and non-eager keys for spans that
subsume attention while a KV connector is active (layer-wise KV hooks are
host calls, not replayed).

## Lifecycle

Both model runners drive the same hooks. Model Runner V2 is the default (and
the only runner implementing DSpark, so it is the one #60397 actually runs
on); V2 replays FULL graphs without a forward context, so the plan taken in
the capture forward context (`vllm/v1/worker/gpu/cudagraph_utils.py`) is what
every replay runs.

| vLLM event | Provider hook |
|---|---|
| end of `load_model` (weights processed) | `create`, `bind_layers`, `memory_request` (allocations here are counted by memory profiling before KV-cache sizing) |
| `initialize_kv_cache` (profiling cache, then real cache) | `bind_caches(generation, cache views)` |
| profiling cache freed | `unbind_caches(generation)` |
| first graph capture (`profile_cudagraph_memory`, again in `capture_model`) | `warmup(step keys to be captured)`, once per key: JIT + per-shape scratch, inside memory profiling |
| each planned step | `begin_step` (once per runtime), `run` per layer |
| `reload_weights`, online weight updates, wake-up | `bind_layers` again |
| sleep / wake | `on_sleep(level)` / `on_wake()` (only if SLEEP_MODE supported) |
| shutdown (after graph teardown) | `close` |

## Execution regimes

| Regime | Support in v0 |
|---|---|
| eager | yes |
| FULL graph | yes (`run` must be graph-capturable) |
| PIECEWISE (FX split) / breakable graph | allowed if the claim lists it; mono kernels decline it today because a piecewise graph is replayed for mixed batches |
| torch.compile'd model | refused; v1 registers each span as a splitting custom op |

## Versioning

* `INTERFACE_MAJOR` covers seams, step keys and the lifecycle; it should
  almost never change.
* Views (`dsv4.sparse_swa_mla.step@1`) and layout tags
  (`mxfp8.rowmajor.e8m0_32x32@1`) carry their own versions. The owner of the
  underlying internals updates the view builder in the same PR as an internal
  change; providers are unaffected unless semantics change, in which case the
  owner adds `@2` (and may keep building `@1` for a deprecation window).
* Layout tags are pinned by `tests/fused_stages/test_layout_fingerprints.py`:
  a fixed weight is pushed through the real processing function and hashed; a
  byte change fails CI until the tag version is bumped.
* Provider ids/versions (and out-of-tree distribution versions) are folded
  into the compile/graph cache hash (`KernelConfig.fused_stages.compute_hash`).

Rationale from 12 months of history (2026): processed-weight shuffle sites
changed ~31 times, the shared attention metadata class changed shape 11
times, while plugin entry points barely moved. A single global version would
bump several times a year; per-view versions confine churn to the views that
actually changed.

## Using a provider

```bash
pip install vllm-dsv41-mono   # registers entry point "rocm_mono_dsv41"
vllm serve deepseek-ai/DeepSeek-V4.1-Flash -tp 2 \
  --kernel-config '{"fused_stages": {"priority": {"deepseek_v41.*": ["rocm_mono_dsv41"]}}}'
```

Logs list, per stage, which provider took which layers, and every refusal
with its reason. `VLLM_PLUGINS` filters providers by entry point name.

## Writing a provider

See `examples/others/fused_stage_plugin_dsv41/`, which wraps the kernels of
#60397 unchanged:

```toml
[project.entry-points."vllm.fused_stages"]
rocm_mono_dsv41 = "vllm_dsv41_mono.provider:Provider"
```

```python
class Provider(StageProvider):
    provider_id = "rocm_mono_dsv41"; provider_version = "0.1.0"
    def claims(self): ...      # stages, versions, regimes, layouts, views, features
    def accept(self, claim, spec, deployment, layers) -> Acceptance: ...
    def create(self, claim, ctx) -> StageRuntime: ...
```

Run `vllm.fused_stages.testing` (`check_claims`, `compare_to_reference`,
`replay_stress`) in the provider's CI.

## Adding stages to a model

See `vllm/models/deepseek_v41/amd/fused_stages.py`: declare seams and
`StageSpec`s, a `binding(layer)` role map, and view builders; in the layer,
`attach_stages(...)` in `__init__` and `try_run(...)` at each entry seam. A
seam option such as `tp_partial_sum` is applied when
`StageHandle.configured(spec)` is true (decided from config, identically on
every rank), and the model performs the reduction itself on reference steps.

## Governance (proposal)

1. **In-tree reference providers** define the interface (e.g. DSv4.1 at
   layer + sub-layer scope, Kimi-K3 at block scope), fenced under
   `vllm/fused_stages/providers/<id>/` and importing only the API.
2. **Out-of-tree providers** via the entry point group, with CI against vLLM
   nightly using the conformance kit.
3. **A staging/contrib repo** (vLLM org, or vendor-run) with an owner, exit
   criteria and a CI job per provider; removed if unowned or red for N releases.
4. **Graduation**: shipped in vLLM images via an extra, or folded into core.

## Known limitations (v0)

* torch.compile'd models are refused (span-as-splitting-op is v1).
* No layout negotiation: a provider that needs a different weight layout
  refuses; it cannot ask vLLM to pick a compatible quant backend.
* KV-connector hooks are only honoured in eager steps for spans that subsume
  attention.
* Rank agreement is guaranteed for TP; EP/DP-attention, PP and multi-node
  spans are out of scope.
* Views are defined only for DeepSeek V4/V4.1 sparse-SWA MLA.
* Speculative-decoding drafter forwards share the target's StepKey space;
  draft layers (`LayerBinding.draft`) are only offered to claims with
  `draft_layers=True`, and such providers cannot yet tell drafter steps apart.
* Eager-only keys are never warmed up; state an eager step allocates lazily
  is not counted by memory profiling (size it in `create`).
