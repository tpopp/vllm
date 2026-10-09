# SPDX-License-Identifier: MIT
"""vLLM fused-stage provider for DeepSeek-V4.1-Flash decode on MI355X.

Wraps the kernels of vllm#60397 (``kernels/``, vendored unchanged by
``vendor_kernels.sh``) behind ``vllm.fused_stages.api``. Everything that was
vLLM-specific in that PR's ``mono_decode.py`` is either here (eligibility:
arch, TP, routing constants, rows per step) or in vLLM's
``vllm/models/deepseek_v41/amd/fused_stages.py`` (seams, weight roles, views).
This module imports nothing from vLLM but the API module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch

from vllm.fused_stages.api import (
    Acceptance,
    Feature,
    LayerBinding,
    MemoryRequest,
    Refusal,
    Regime,
    StageClaim,
    StageContext,
    StageProvider,
    StageRuntime,
    StepContext,
    StepKey,
)

MAX_ROWS = 48  # 8 requests x (1 + 5 DSpark drafts)
SWA_WIDTH = 128
CDNA4 = ("gfx950",)  # scaled MFMA + MX formats; a port adds an entry

# Layouts the kernels read in place (vllm.fused_stages.layouts).
EXPERTS = frozenset({"mxfp4.aiter_a4w4.separated@1"})
MXFP8 = frozenset({"mxfp8.rowmajor.e8m0_32x32@1"})
SCALE = frozenset({"scale.follows_weight@1"})
FFN_LAYOUTS = {
    "experts.w13": EXPERTS, "experts.w13_scale": SCALE,
    "experts.w2": EXPERTS, "experts.w2_scale": SCALE,
    "shared.gate_up": MXFP8, "shared.gate_up_scale": SCALE,
    "shared.down": MXFP8, "shared.down_scale": SCALE,
    "router.weight": frozenset({"bf16.rowmajor@1"}),
}  # fmt: skip
ATTN_LAYOUTS = {
    **FFN_LAYOUTS,
    **{r: MXFP8 for r in ("attn.wqkv", "attn.wq_b", "attn.wo_a", "attn.wo_b")},
    **{
        r + "_scale": SCALE
        for r in ("attn.wqkv", "attn.wq_b", "attn.wo_a", "attn.wo_b")
    },
}

SUPPORTS = frozenset(
    {Feature.ASYNC_SCHEDULING, Feature.PREFIX_CACHING, Feature.KV_TRANSFER}
)


def _routing_ok(c: Mapping) -> bool:
    return (
        c["has_shared_expert"]
        and not c["has_hash_routing"]
        and not c["has_vision_routing_bias"]  # kernels route text-only
        and c["n_routed_experts"] == 384
        and c["top_k"] == 6
        and c["scoring_func"] == "sqrtsoftplus"
        and c["routed_scaling_factor"] == 1.5
        and c["swiglu_limit"] == 10.0
        and c["renormalize"]
        and c["fused_seam_norm"]
        and not c["uses_sequence_parallel"]
        and not c["is_draft_layer"]
    )


def _whole_layer_ok(c: Mapping) -> bool:
    return (
        _routing_ok(c)
        and not c["is_first_layer"]
        and not c["has_engram"]
        and not c["has_compressor"]
        and not c["has_indexer"]
        and c["compress_ratio"] in (1, 2)
        and c["kv_cache_dtype"] == "fp8_ds_mla"
        and not c["kv_mxfp8"]
    )


def _steps(key: StepKey) -> bool:
    # A pure function of the key: decode-only, at most MAX_ROWS rows.
    return key.uniform_decode and 1 <= key.num_tokens <= MAX_ROWS and not key.ubatched


class Provider(StageProvider):
    provider_id = "rocm_mono_dsv41"
    provider_version = "0.1.0+vllm60397"

    def claims(self) -> Sequence[StageClaim]:
        common = dict(
            stage_versions=frozenset({1}),
            regimes=Regime.EAGER | Regime.FULL_GRAPH,  # PIECEWISE replays mixed
            supports=SUPPORTS,
            exclusive_gpu=True,
        )
        return [
            StageClaim(
                stage="deepseek_v41.decoder_layer",
                layouts=ATTN_LAYOUTS,
                views=frozenset(
                    {"dsv4.sparse_swa_mla.cache@1", "dsv4.sparse_swa_mla.step@1"}
                ),
                **common,
            ),
            StageClaim(
                stage="deepseek_v41.ffn_after_attention",
                layouts=FFN_LAYOUTS,
                **common,
            ),
        ]

    def accept(self, claim, spec, deployment, layers: Sequence[LayerBinding]):
        if deployment.arch not in CDNA4:
            raise Refusal(f"needs {'/'.join(CDNA4)}, got {deployment.arch}")
        if deployment.tp_size not in (2, 4):
            raise Refusal("needs tensor parallel size 2 or 4")
        try:
            import vllm_dsv41_mono.kernels.runner  # noqa: F401  (FlyDSL, AITER)
        except ImportError as err:
            raise Refusal(f"kernels unavailable: {err}") from err
        whole = claim.stage.endswith("decoder_layer")
        ok = _whole_layer_ok if whole else _routing_ok
        layer_ids = frozenset(b.layer_id for b in layers if ok(b.consts))
        if not whole:
            # v0 leaves exclusivity between a model's stages to the provider:
            # layers this provider runs whole are not also offered the FFN
            # launch (the decoder_layer hook would return first anyway).
            layer_ids = frozenset(
                b.layer_id
                for b in layers
                if ok(b.consts) and not _whole_layer_ok(b.consts)
            )
        return Acceptance(layer_ids, _steps, note="decode <= 48 rows")

    def create(self, claim, ctx: StageContext) -> StageRuntime:
        return _Runtime(claim.stage.endswith("decoder_layer"), ctx)


class _Runtime(StageRuntime):
    def __init__(self, whole: bool, ctx: StageContext) -> None:
        import os

        os.environ.setdefault("FLYDSL_RUNTIME_CACHE_DIR", ctx.cache_dir)
        from vllm_dsv41_mono.kernels.runner import DSV41MonoLayer

        self.whole = whole
        # Collective over the TP CPU group (peer-memory handle exchange); every
        # rank creates the same runtimes in the same order. Allocations here
        # are counted by vLLM's memory profiling (we run at end of load).
        self.runner = DSV41MonoLayer(
            ctx.tp.size, ctx.tp.rank, ctx.tp.cpu_group, ctx.device
        )
        self.weights: dict[int, object] = {}
        self.caches: Mapping[int, object] = {}

    def memory_request(self) -> MemoryRequest:
        from vllm_dsv41_mono.kernels.layer import peer_half_bytes

        return MemoryRequest(peer_bytes=2 * peer_half_bytes(self.runner.tp))

    def bind_layers(self, layers: Sequence[LayerBinding]) -> None:
        from vllm_dsv41_mono.kernels.attention.plan import Dims
        from vllm_dsv41_mono.kernels.runner import AttnWeights, MonoLayerWeights

        def t(b: LayerBinding, role: str) -> torch.Tensor:
            x = b.weights[role].tensor
            return x.view(torch.uint8) if role.endswith("_scale") else x

        for b in layers:
            attn = None
            if self.whole:
                attn = AttnWeights(
                    layer_id=b.layer_id,
                    wqkv=t(b, "attn.wqkv"), wqkv_scale=t(b, "attn.wqkv_scale"),
                    q_norm=t(b, "attn.q_norm"), kv_norm=t(b, "attn.kv_norm"),
                    wq_b=t(b, "attn.wq_b"), wq_b_scale=t(b, "attn.wq_b_scale"),
                    wo_a=t(b, "attn.wo_a"), wo_a_scale=t(b, "attn.wo_a_scale"),
                    wo_b=t(b, "attn.wo_b"), wo_b_scale=t(b, "attn.wo_b_scale"),
                    attn_sink=t(b, "attn.sink"), cos_sin=t(b, "attn.cos_sin"),
                    ratio=b.consts["compress_ratio"],
                )  # fmt: skip
                attn.check(Dims(self.runner.tp))
            self.weights[b.layer_id] = MonoLayerWeights(
                attn=attn,
                hc_attn_fn=t(b, "hc_attn_fn"), hc_attn_scale=t(b, "hc_attn_scale"),
                hc_attn_base=t(b, "hc_attn_base"), attn_norm=t(b, "attn_norm"),
                hc_ffn_fn=t(b, "hc_ffn_fn"), hc_ffn_scale=t(b, "hc_ffn_scale"),
                hc_ffn_base=t(b, "hc_ffn_base"), ffn_norm=t(b, "ffn_norm"),
                gate_w=t(b, "router.weight"), bias=t(b, "router.bias"),
                w13=t(b, "experts.w13"), w13_s=t(b, "experts.w13_scale"),
                w2=t(b, "experts.w2"), w2_s=t(b, "experts.w2_scale"),
                sgu=t(b, "shared.gate_up"), sgu_s=t(b, "shared.gate_up_scale"),
                sw2=t(b, "shared.down"), sw2_s=t(b, "shared.down_scale"),
            )  # fmt: skip

    def bind_caches(self, generation: int, caches: Mapping[int, object]) -> None:
        self.caches = caches

    def unbind_caches(self, generation: int) -> None:
        self.caches = {}

    def warmup(self, steps: Iterable[StepKey]) -> None:
        for key in steps:
            for ratio in (1, 2) if self.whole else (None,):
                if ratio is None:
                    self.runner.ffn_kernel(key.num_tokens)
                else:
                    self.runner.kernels(key.num_tokens, ratio)
            self.runner.scratch(key.num_tokens)  # per-width scratch, eager

    def run(
        self,
        layer_id: int,
        ctx: StepContext,
        inputs: Mapping[str, torch.Tensor],
        outputs: Mapping[str, torch.Tensor],
    ) -> None:
        w = self.weights[layer_id]
        outs = tuple(outputs.values())  # x, residual, post_mix, res_mix, pre_mix
        if not self.whole:
            self.runner.ffn(
                w,
                inputs["x"],  # wo_b's un-reduced partial sum
                inputs["residual"],
                inputs["post_mix"],
                inputs["res_mix"],
                inputs["pre_mix"],
                outs=outs,
            )
            return
        cache, view = self.caches[layer_id], ctx.view
        assert view.swa_width == SWA_WIDTH and view.comp_block_size in (
            0,
            cache.comp_block_size,
        )
        self.runner.forward(
            w,
            inputs["x"], inputs["residual"], inputs["post_mix"],
            inputs["res_mix"], inputs["pre_mix"], inputs["positions"],
            view.slot_mapping, cache.swa_records,
            view.swa_indices, view.swa_lens, view.token_to_req,
            topk_indices=view.topk_indices,
            comp_cache=cache.comp_records,
            comp_block_table=view.comp_block_table,
            outs=outs,
        )  # fmt: skip

    def close(self) -> None:
        self.weights.clear()
        self.caches = {}
        self.runner = None  # peer memory freed with it, after graph teardown
