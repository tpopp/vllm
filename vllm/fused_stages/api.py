# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused-stage provider contract.

This is the only vLLM module a fused-stage provider (a "mono kernel", a
megakernel, or any implementation that replaces a whole span of a model's
forward) may import. vLLM pushes everything a provider needs to it as frozen
descriptors and versioned views. Providers never reach into vLLM modules,
attention metadata classes or private attributes.

Versioning:

* ``INTERFACE_MAJOR`` covers the core contract below: seams, step keys and the
  lifecycle. It is expected to almost never change.
* View types and weight-layout tags carry their own small integer versions
  (``"dsv4.sparse_swa_mla.step@1"``, ``"mxfp4.aiter_a4w4.separated@1"``), so a
  change in one attention backend or one quantization layout bumps only that
  string. See ``vllm/fused_stages/views`` and ``vllm/fused_stages/layouts.py``.

See ``docs/design/fused_stages.md`` for the design and its rationale.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

INTERFACE_MAJOR = 0

ENTRY_POINT_GROUP = "vllm.fused_stages"
"""Entry point group that out-of-tree providers register under."""

REFERENCE = "reference"
"""Reserved provider name: the model's own (multi-kernel) code path."""


# --------------------------------------------------------------- vocabulary


class Regime(enum.Flag):
    """How the step containing a span is executed. Providers declare the
    regimes they support; a step in any other regime runs the reference."""

    EAGER = enum.auto()
    FULL_GRAPH = enum.auto()
    """The whole forward is captured and replayed by its BatchDescriptor."""
    PIECEWISE_GRAPH = enum.auto()
    """FX-split piecewise graphs; replayed for mixed batches too."""
    BREAKABLE_GRAPH = enum.auto()
    """Eager Python, stream capture broken at attention / KV-cache ops
    (``VLLM_USE_BREAKABLE_CUDAGRAPH``)."""
    COMPILED = enum.auto()
    """The model's forward is torch.compile'd. Not supported in v0: a span
    would have to be a splitting custom op."""


class Feature(str, enum.Enum):
    """Engine features a provider must either support or refuse. A feature is
    listed in ``DeploymentInfo.features`` when it is enabled for this engine;
    ``accept`` refuses unless the claim lists it in ``supports``."""

    LORA = "lora"
    KV_TRANSFER = "kv_transfer"
    """A KV connector is configured. The framework calls the per-layer
    ``wait_for_layer_load`` / ``save_kv_layer`` hooks around spans that
    subsume attention, so a provider only has to tolerate them."""
    KV_OFFLOAD = "kv_offload"
    """KV blocks are copied on side streams that may overlap a span."""
    SLEEP_MODE = "sleep_mode"
    WEIGHT_TRANSFER = "weight_transfer"
    """Online weight updates (RL), which bypass ``reload_weights``."""
    PREFIX_CACHING = "prefix_caching"
    DBO = "dbo"
    """Dual-batch overlap: ``forward`` runs twice, concurrently, on two host
    threads with separate streams."""
    ASYNC_SCHEDULING = "async_scheduling"
    BATCH_INVARIANT = "batch_invariant"
    PIPELINE_PARALLEL = "pipeline_parallel"
    EXPERT_PARALLEL = "expert_parallel"
    DATA_PARALLEL = "data_parallel"
    CONTEXT_PARALLEL = "context_parallel"
    EPLB = "eplb"
    # Model-specific execution choices (e.g. DSv4's sequence-parallel layers)
    # are reported per layer in LayerBinding.consts, not here.


@dataclass(frozen=True)
class DeploymentInfo:
    """Everything static about this engine that a provider may refuse on.
    Built once by vLLM from the VllmConfig and the platform; identical on
    every rank of a TP group."""

    platform: str  # "rocm", "cuda", ...
    arch: str  # "gfx950", "gfx942", "sm_100", ...
    num_compute_units: int
    tp_size: int
    pp_size: int
    dp_size: int
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    kv_cache_dtype: str
    block_size: int
    spec_method: str | None  # "dspark", "eagle3", "mtp", None
    num_spec_tokens: int
    regimes: Regime
    """Regimes this engine may run steps in (e.g. EAGER | FULL_GRAPH)."""
    features: frozenset[Feature]
    hf_config: Any
    """The model's HF config, read-only. Constants a stage depends on are
    also passed per layer in ``LayerBinding.consts``; prefer those."""


@dataclass(frozen=True)
class StepKey:
    """Provider-neutral identity of a step, derived by vLLM from the cudagraph
    ``BatchDescriptor`` and the execution regime.

    Dispatch decisions depend ONLY on this, so (1) a captured graph never
    replays a different decision than the one taken at capture, and (2) every
    rank of a TP group decides alike: kernels that all-reduce in-kernel
    deadlock if ranks disagree.

    "Decode" here means what the attention backends classify as decode:
    every request has the uniform query length ``1 + num_spec_tokens``.
    """

    num_tokens: int
    """Padded token count (rows) of the step."""
    num_reqs: int
    query_len: int
    """Tokens per request when ``uniform``, else 0."""
    uniform_decode: bool
    regime: Regime
    has_lora: bool = False
    ubatched: bool = False
    """DBO microbatching; also listed as ``Feature.DBO`` at deploy time."""


@dataclass(frozen=True)
class TensorSpec:
    """Shape/dtype of a seam tensor. ``"M"`` is the step's row count; ints are
    fixed for a given model and TP size."""

    dims: tuple[int | str, ...]
    dtype: torch.dtype
    optional: bool = False


@dataclass(frozen=True)
class Seam:
    """A program point in a model's forward with a fixed boundary state."""

    name: str
    tensors: tuple[tuple[str, TensorSpec], ...]
    options: frozenset[str] = frozenset()
    """Semantics flags of the boundary state, e.g. ``"tp_partial_sum"``: the
    first tensor is an un-reduced tensor-parallel partial sum."""

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.tensors)


Scope = Literal["block", "sublayer", "layer", "layer_range"]


@dataclass(frozen=True)
class StageSpec:
    """A replaceable span of a model's forward, declared by the model."""

    family: str
    """Model family, e.g. ``"deepseek_v41"``."""
    name: str
    """Stage name within the family, e.g. ``"decoder_layer"``."""
    version: int
    """Bumped when the seams, the weight roles or their semantics change."""
    entry: Seam
    exit: Seam
    scope: Scope
    subsumes_attention: bool
    """The span contains attention (writes the KV cache). The framework then
    runs the KV-connector layer hooks around it."""
    collectives: tuple[str, ...] = ()
    """Collectives the reference span performs, e.g. ``("tp_all_reduce",)``.
    A provider replacing the span must perform them (in-kernel or not)."""
    views: tuple[str, ...] = ()
    """View type ids the model can build for each layer of this stage."""
    weight_roles: tuple[str, ...] = ()
    """Role names in ``LayerBinding.weights`` (part of the stage version)."""

    @property
    def id(self) -> str:
        return f"{self.family}.{self.name}"

    @property
    def versioned_id(self) -> str:
        return f"{self.id}@{self.version}"


@dataclass(frozen=True)
class WeightRef:
    """A processed weight, read in place. ``layout`` is the tag the quant
    method attached in ``process_weights_after_loading``; see
    ``vllm.fused_stages.layouts``. vLLM never copies the tensor for a
    provider; a provider that needs another layout must refuse (v0)."""

    tensor: torch.Tensor
    layout: str


@dataclass(frozen=True)
class LayerBinding:
    """One layer's weights by role and the model constants the stage needs."""

    layer_id: int
    weights: Mapping[str, WeightRef]
    consts: Mapping[str, Any]
    attention_layers: tuple[str, ...] = ()
    """vLLM attention layer names inside the span (KV-connector hooks)."""
    draft: bool = False
    """The layer belongs to a speculative-decoding drafter. Drafter forwards
    share the target's StepKey space in v0, so the framework only offers
    draft layers to claims with ``draft_layers=True``."""


@dataclass(frozen=True)
class View:
    """Base for versioned views built by vLLM. Concrete views are frozen
    dataclasses in ``vllm.fused_stages.views`` with a class-level
    ``TYPE_ID`` such as ``"dsv4.sparse_swa_mla.step@1"``."""

    TYPE_ID = "abstract@0"


# ---------------------------------------------------------- provider side


class Refusal(Exception):
    """Raised by ``accept``/``create`` to decline. Refusal is collective: if
    any rank refuses, every rank runs the reference path and the reasons are
    logged once."""


@dataclass(frozen=True)
class StageClaim:
    """What a provider offers for one stage."""

    stage: str
    """``StageSpec.id``, e.g. ``"deepseek_v41.decoder_layer"``."""
    stage_versions: frozenset[int]
    regimes: Regime
    layouts: Mapping[str, frozenset[str]]
    """Role -> acceptable layout tags. Roles not listed accept any layout."""
    views: frozenset[str] = frozenset()
    """View type ids (with version) the provider consumes."""
    supports: frozenset[Feature] = frozenset()
    """Enabled engine features this provider handles; any other enabled
    feature refuses the claim."""
    draft_layers: bool = False
    """Accept layers of a speculative-decoding drafter (see
    ``LayerBinding.draft``)."""
    exclusive_gpu: bool = True
    """A persistent kernel whose CTAs spin-wait on each other: it needs every
    CTA resident, so it cannot share the GPU with unordered concurrent work.
    Exclusive claims refuse ``Feature.DBO`` and ``Feature.KV_OFFLOAD`` even if
    listed in ``supports``."""


@dataclass(frozen=True)
class Acceptance:
    """Result of a successful ``accept``."""

    layer_ids: frozenset[int]
    steps: Callable[[StepKey], bool]
    """A PURE predicate over StepKey (no global state, no metadata, no
    tensors). vLLM evaluates it at init for every captured key and memoizes
    it for eager keys; it is never a per-step callback. Limits on anything
    not in the key (context length, number of index blocks...) must be
    enforced statically against ``DeploymentInfo``."""
    note: str = ""


@dataclass(frozen=True)
class MemoryRequest:
    """Declared device memory, for logging and validation. Runtimes are
    created at the end of model loading, so anything they allocate (torch or
    IPC) is already counted by vLLM's memory profiling before the KV cache is
    sized. The declaration lets vLLM check and report it."""

    device_bytes: int = 0
    peer_bytes: int = 0


@dataclass(frozen=True)
class TPContext:
    rank: int
    size: int
    device_group: Any  # torch.distributed.ProcessGroup (NCCL/RCCL)
    cpu_group: Any  # torch.distributed.ProcessGroup (gloo), for handle exchange


@dataclass
class StageContext:
    """Handed to ``create``."""

    deployment: DeploymentInfo
    tp: TPContext
    device: torch.device
    cache_dir: str
    """A per-rank-safe directory for JIT artifacts, owned by vLLM."""
    counters: dict[str, int] = field(default_factory=dict)
    """Exported by vLLM (fallback reasons, runs)."""


@dataclass(frozen=True)
class StepContext:
    """Per-step inputs besides the seam tensors."""

    key: StepKey
    view: View | None
    """The layer's per-step view (attention metadata), when the stage
    declares one; built by vLLM from its internal metadata."""
    stream: torch.cuda.Stream | None
    """The stream to launch on (None on CPU, in tests)."""


class StageRuntime(ABC):
    """One per (provider, stage) per rank. All state lives here, not in
    module globals. Every method except ``begin_step`` and ``run`` is called
    eagerly, outside graph capture."""

    def memory_request(self) -> MemoryRequest:
        return MemoryRequest()

    @abstractmethod
    def bind_layers(self, layers: Sequence[LayerBinding]) -> None:
        """After weights are processed; again after any weight reload or
        online weight update (the tensors may have moved)."""

    def bind_caches(self, generation: int, caches: Mapping[int, View]) -> None:  # noqa: B027 (optional hook)
        """For every KV-cache allocation: the minimal profiling cache, then
        the real one. ``caches`` maps layer id to its cache view."""

    def unbind_caches(self, generation: int) -> None:  # noqa: B027 (optional hook)
        """Before that cache is freed. Graphs that used it are already
        destroyed."""

    def warmup(self, steps: Iterable[StepKey]) -> None:  # noqa: B027 (optional hook)
        """JIT-compile and allocate per-shape scratch for the given keys.

        Called once per key, before the first graph capture (inside memory
        profiling, so allocations here are counted before the KV cache is
        sized). Eager-only keys are never warmed up: anything an eager step
        allocates lazily is NOT counted, so size such state in ``create``."""

    def begin_step(self, ctx: StepContext) -> None:  # noqa: B027 (optional hook)
        """Optional: once per planned step, before this runtime's first span
        of the step. Graph-capturable."""

    @abstractmethod
    def run(
        self,
        layer_id: int,
        ctx: StepContext,
        inputs: Mapping[str, torch.Tensor],
        outputs: Mapping[str, torch.Tensor],
    ) -> None:
        """Compute the span, writing the exit seam into ``outputs``.

        ``inputs`` and ``outputs`` are ordered like the stage's entry and exit
        seams. Output buffers are allocated by vLLM for this call (from the
        graph pool when capturing), so their addresses are graph-static.

        Graph-capturable in FULL_GRAPH: no host synchronization, no stream
        forks left unjoined, no reads of token ids (async scheduling may fill
        them late). Padding rows have ``slot_mapping == -1`` and must not
        write the KV cache; kernels may read (not write) past the requested
        rows of metadata tables. Lazy JIT is allowed only in EAGER."""

    def on_sleep(self, level: int) -> None:  # noqa: B027 (optional hook)
        """Engine sleep. Only called if the claim supports SLEEP_MODE."""

    def on_wake(self) -> None:  # noqa: B027 (optional hook)
        pass

    @abstractmethod
    def close(self) -> None:
        """Release everything. Graphs are destroyed before this is called."""


class StageProvider(ABC):
    """Registered via the ``vllm.fused_stages`` entry point group (or
    in-tree via ``vllm.fused_stages.registry``)."""

    provider_id: str
    provider_version: str
    """Folded into vLLM's compile/graph cache hash (with the provider's
    distribution version for out-of-tree providers)."""
    interface_major: int = INTERFACE_MAJOR

    @abstractmethod
    def claims(self) -> Sequence[StageClaim]: ...

    @abstractmethod
    def accept(
        self,
        claim: StageClaim,
        spec: StageSpec,
        deployment: DeploymentInfo,
        layers: Sequence[LayerBinding],
    ) -> Acceptance:
        """Static checks only; raise ``Refusal`` to decline. Weights are
        present so layouts, shapes and dtypes can be validated, but nothing
        may be allocated and no collective may be issued."""

    @abstractmethod
    def create(self, claim: StageClaim, ctx: StageContext) -> StageRuntime:
        """Build the runtime. Collectives over ``ctx.tp`` are allowed (every
        rank calls ``create`` for the same claims, in the same order)."""
