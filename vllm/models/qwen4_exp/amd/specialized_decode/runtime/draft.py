# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.8-Flash-Next MTP drafter on the specialized decode kernels,
enabled the same way the target's is (``--monokernel``; the drafter follows
the target's switch, it adds none of its own).

``DraftDecode`` runs the MTP draft model's single decoder layer as persistent
launches: the attention hyper-connection's front and the fc_embedding /
fc_hidden projections fused (``mono.draft_front``, one launch, replacing the
stock embedding branch's plain ``torch`` ops -- optional: falls back to
them + ``front.hc_front`` when the fc weights are not a plain gatherable
bf16 linear, ``prepare_weights``), QSA attention (vLLM's, unchanged), the
o_proj block, the MLP hyper-connection's front and the MoE block
(``mono.blocks.oproj_block`` / ``mono.blocks.moe_block`` -- the draft
layer's shapes match the target's decoder layer exactly, so these are the
very same launches, not a draft variant of them), then the final mixer's
front (``front.hc_front`` again, ``use_combine=False`` so its down
projection has no injection rows -- the kernel always wants a
``layout.DOWN_ROWS``-row down, so the mixer's is zero-padded once at
weight-check time, ``prepare_weights``). Greedy draft tokens are read
straight from a fused MXFP4 lm_head GEMV + TP argmax (``mono.draft_head``)
when the draft's lm_head is an online-MXFP4 copy (``get_top_tokens``);
otherwise the proposer's own ``LogitsProcessor.get_top_tokens`` reduction
runs.

Its own epoch, scratch and peer buffers (DSv4.1's drafter precedent, #62131):
a mailbox word must never be a tag to one launch and data to another, so the
drafter's mailbox is never the target's -- built at the same eager step on
every rank, with no new switch. The one piece the two executors must agree
on is the device: both bind the GPU the parent process is already pinned to
(the same ``get_tp_group()``)."""

import os
from dataclasses import dataclass

import torch

from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.models.kimi_k3.amd.specialized_decode.runtime.peer import tp_agree
from vllm.models.qwen4_exp.amd.specialized_decode.mono import layout as L
from vllm.models.qwen4_exp.amd.specialized_decode.runtime.guards import (
    mxfp4_shuffled,
    o_proj,
)

# ``mono.blocks`` / ``mono.front`` / ``mono.draft_front`` / ``mono.draft_head``
# and ``runtime.layers`` (which imports ``blocks`` / ``front`` itself) all pull
# in FlyDSL / AITER kernel code at import time. Keep this module's own top
# free of them -- as ``specialized_decode.runtime`` (the config-stage checks)
# and ``hooks.py`` do for the target -- so ``draft_unsupported`` /
# ``draft_specialized_decode_config`` stay importable, and testable, without a
# GPU toolchain; only ``DraftDecode`` (built after the weights load, never on
# a config-only check) imports them, inside its own methods.

logger = init_logger(__name__)

NAME = "Qwen3.8-Flash-Next MTP draft (--monokernel)"
# this executor's own mailbox: distinct launches need distinct tag slots,
# not distinct from the target's (a different epoch / health / peer buffer
# entirely -- see the module docstring)
ATTN_LAYER = 0
MLP_LAYER = 1
MIXER_LAYER = 2
HEAD_LAYER = 0  # the head has its own peer buffer too


@dataclass(frozen=True)
class DraftConfig:
    max_tokens: int = L.MAX_TOKENS

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(range(1, self.max_tokens + 1))


def draft_specialized_decode_config(vllm_config) -> DraftConfig | None:
    """None for the stock draft step. Raises when ``--monokernel`` is set for
    an unsupported draft deployment, listing every reason."""
    if not vllm_config.kernel_config.monokernel:
        return None
    reasons = draft_unsupported(vllm_config)
    if reasons:
        raise ValueError(
            f"{NAME} does not support: " + "; ".join(reasons) + " (without "
            "--monokernel, the stock draft step runs)"
        )
    return DraftConfig()


def draft_unsupported(vllm_config) -> list[str]:
    """Config-only refusals (the loaded draft model is checked when the
    executor is built). Shares the target's parallelism / feature / platform
    checks (``runtime.unsupported``): the drafter runs under the same
    deployment. Adds: MTP only (the drafter is bound to the MTP draft model),
    and the draft model's own shape (one MTP layer, Qwen3.8-Flash-Next's
    hidden sizes)."""
    from vllm.models.qwen4_exp.amd.specialized_decode import runtime as target

    spec = vllm_config.speculative_config
    if spec is None or spec.method != "mtp":
        return ["needs MTP speculative decoding"]
    reasons = [
        why
        for bad, why in (
            (vllm_config.parallel_config.tensor_parallel_size != target.TP,
             f"TP {vllm_config.parallel_config.tensor_parallel_size} "
             f"(TP{target.TP} only)"),  # fmt: skip
            (vllm_config.parallel_config.pipeline_parallel_size != 1,
             "pipeline parallelism"),  # fmt: skip
            (vllm_config.parallel_config.data_parallel_size != 1,
             "data parallelism"),  # fmt: skip
            (vllm_config.parallel_config.enable_expert_parallel,
             "expert parallelism"),  # fmt: skip
            (vllm_config.parallel_config.enable_eplb, "EPLB"),
            (vllm_config.lora_config is not None, "LoRA"),
            (vllm_config.kv_transfer_config is not None, "KV transfer connectors"),
            (not vllm_config.use_v2_model_runner,
             "the V1 model runner (set VLLM_USE_V2_MODEL_RUNNER=1)"),  # fmt: skip
        )
        if bad
    ]
    draft_mc = spec.draft_model_config
    if draft_mc is None:
        reasons.append("no draft model config")
        return reasons + target.platform_reasons()
    shapes = target.text_shapes(draft_mc.hf_text_config)
    if shapes != L.SHAPES:
        reasons.append(
            f"draft model shapes {shapes} ({', '.join(L.SHAPE_NAMES)}; "
            f"Qwen3.8-Flash-Next is {L.SHAPES})"
        )
    num_mtp_layers = getattr(draft_mc.hf_text_config, "mtp_num_hidden_layers", 1)
    if num_mtp_layers != 1:
        reasons.append(f"{num_mtp_layers} MTP layers (needs exactly 1)")
    if (spec.num_speculative_tokens + 1) > L.MAX_TOKENS:
        reasons.append(f"more than {L.MAX_TOKENS - 1} speculative tokens")
    return reasons + target.platform_reasons()


def _buf(n: int, dev) -> torch.Tensor:
    return torch.zeros((n + 255) // 256 * 256, dtype=torch.uint8, device=dev)


def _fc_error(name: str, lin, tp: int) -> str | None:
    """Why ``lin`` (``fc_embedding`` / ``fc_hidden``) is unreadable as the
    fused front's full, gathered weight, else None: it must be a plain bf16
    linear (no quantization the kernel reads), its local shard
    ``[H / tp, H]`` as the module shards it."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if not isinstance(lin.quant_method, UnquantizedLinearMethod):
        return f"{name}: quantized ({type(lin.quant_method).__name__})"
    want = (L.HIDDEN // tp, L.HIDDEN)
    if lin.weight.shape != want or lin.weight.dtype != torch.bfloat16:
        return f"{name}: {tuple(lin.weight.shape)} {lin.weight.dtype}, want bf16 {want}"
    return None


def _dense_error(name: str, w: torch.Tensor) -> str | None:
    if w.dtype != torch.bfloat16:
        return f"{name} {w.dtype}, neither bf16 nor a quantized layout the kernels read"
    return None


def model_reasons(mtp, tp: int) -> list[str]:
    """Every reason the kernels cannot serve the loaded MTP draft model."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.models.qwen4_exp.amd.model import Qwen4ExpSparseMoeBlock
    from vllm.models.qwen4_exp.amd.qsa import Qwen4ExpQSAAttention

    model = mtp.model
    reasons = []
    if len(model.layers) != 1:
        reasons.append("more than one MTP layer")
        return reasons
    mixer = model.hyper_connection_mixer
    if mixer is None or mixer.use_combine:
        reasons.append("no final HC mixer")
    layer = model.layers[0]
    if layer.ple is not None:
        reasons.append("a PLE layer")
    mlp = layer.mlp
    if not isinstance(mlp, Qwen4ExpSparseMoeBlock):
        reasons.append("a dense MLP")
    elif not mlp.is_fused_shared_expert_enabled:
        reasons.append("the shared expert not fused")
    else:
        ex = mlp.experts.routed_experts
        lay = L.Layout(tp)
        backend = getattr(ex.quant_method, "mxfp4_backend", None)
        if backend is not Mxfp4MoeBackend.AITER_MXFP4_MXFP4:
            reasons.append(f"routed experts on {backend}, not AITER_MXFP4_MXFP4")
        elif not getattr(ex.w13_weight, "is_shuffled", False):
            reasons.append("routed experts not shuffled")
        elif ex.w13_weight.shape != (L.E + 1, 2 * lay.RIP, L.HIDDEN // 2):
            reasons.append(f"draft layer: w13 {tuple(ex.w13_weight.shape)}")
        elif ex.w2_weight.shape != (L.E + 1, L.HIDDEN, lay.RIP // 2):
            reasons.append(f"draft layer: w2 {tuple(ex.w2_weight.shape)}")
    for name in ("attn_hyper_connection", "mlp_hyper_connection"):
        hc = getattr(layer, name)
        if not hc.use_combine or hc.pad_size != L.DOWN_ROWS - L.LORA - L.HC:
            reasons.append(f"{name} variant")
    a = layer.self_attn
    if not isinstance(a, Qwen4ExpQSAAttention):
        reasons.append("attention without the QSA indexer")
    elif a.num_heads * a.head_dim != L.Layout(tp).CORE:
        reasons.append(f"attention width {a.num_heads * a.head_dim}")
    elif not a.attn_output_gate:
        reasons.append("attention without the output gate")
    o = o_proj(layer)
    if mxfp4_shuffled(o) is None and _dense_error("o_proj", o.weight) is not None:
        reasons.append(_dense_error("o_proj", o.weight))
    dense: dict[str, torch.Tensor] = {
        "gate": mlp.gate.weight,
        "shared_expert_gate": mlp.shared_expert_gate.weight,
    }
    for name in ("attn_hyper_connection", "mlp_hyper_connection"):
        hc = getattr(layer, name)
        dense[f"{name}.hc_norm"] = hc.hc_norm.weight
        dense[f"{name}.down"] = hc.input_mix_weight_down_block_inject.weight
        dense[f"{name}.up"] = hc.input_mix_weight_up.weight
    dense["mixer.hc_norm"] = mixer.hc_norm.weight if mixer is not None else None
    dense["mixer.up"] = mixer.input_mix_weight_up.weight if mixer is not None else None
    for name, w in dense.items():
        if w is not None:
            why = _dense_error(name, w)
            if why is not None:
                reasons.append(why)
    return reasons


def agree(reasons: list[str], group) -> None:
    """Raises on every TP rank if any rank has ``reasons``: a rank that
    refused alone would leave the others spinning in-kernel."""
    if not tp_agree(not reasons, group):
        why = "; ".join(reasons) if reasons else "another TP rank refused"
        raise ValueError(
            f"{NAME} does not support the loaded MTP draft model: {why} "
            "(without --monokernel, the stock draft step runs)"
        )


def compile_build(build, key, abi) -> None:
    """Compiles one launch's kernel without running it (FlyDSL's compile-only
    mode returns before the launch)."""
    launch = build(key)
    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        launch(*abi.zeros(), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            del os.environ["COMPILE_ONLY"]
        else:
            os.environ["COMPILE_ONLY"] = old


class DraftDecode:
    """The executor over one ``Qwen4ExpMTP`` (the MTP draft model)."""

    def __init__(self, mtp, config: DraftConfig):
        from vllm.models.kimi_k3.amd.specialized_decode.mono.abi import (
            FAILED,
            PEER_HEAD,
        )
        from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import (
            HEALTH_WORDS,
        )
        from vllm.models.kimi_k3.amd.specialized_decode.runtime.guards import FailStop
        from vllm.models.kimi_k3.amd.specialized_decode.runtime.peer import PeerBuffer
        from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks
        from vllm.models.qwen4_exp.amd.specialized_decode.mono import draft_front as DF
        from vllm.models.qwen4_exp.amd.specialized_decode.mono import draft_head as DH
        from vllm.models.qwen4_exp.amd.specialized_decode.mono import front

        tp = get_tp_group()
        agree(model_reasons(mtp, tp.world_size), tp.cpu_group)
        self.model, self.config = mtp, config
        self.tp, self.rank = tp.world_size, tp.rank_in_group
        self.layer = mtp.model.layers[0]
        dev = mtp.model.embed_tokens.weight.device

        self.epoch = torch.zeros(1, dtype=torch.int32, device=dev)
        self.health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=dev)
        self.peers = PeerBuffer(
            PEER_HEAD + blocks.peer_bytes(L.MAX_TOKENS, self.tp),
            tp.cpu_group,
            self.rank,
            self.tp,
            dev,
        )
        self.peers.bytes.zero_()
        self.peer_table = self.peers.addresses + PEER_HEAD
        self.alarm = int(self.peers.addresses[0].item()) + 4 * self.rank

        def attn_bytes(s):
            return max(
                front.scratch_bytes(front.FrontBuild(tokens=s, tp=self.tp)),
                DF.scratch_bytes(DF.FrontBuild(tokens=s, tp=self.tp)),
            )

        def mlp_bytes(s):
            moe = blocks.moe_scratch_layout(blocks.MoeBuild(tokens=s, tp=self.tp))
            return max(
                front.scratch_bytes(front.FrontBuild(tokens=s, tp=self.tp)),
                blocks.scratch_bytes(moe),
            )

        self.attn_scratch = {s: _buf(attn_bytes(s), dev) for s in config.widths}
        self.mlp_scratch = {s: _buf(mlp_bytes(s), dev) for s in config.widths}

        self.prepare_weights()

        self._head = None
        if self._head_rows is not None:
            self.head_peers = PeerBuffer(
                DH.peer_bytes(self.tp), tp.cpu_group, self.rank, self.tp, dev
            )
            self.head_peers.bytes.zero_()
            self.head_epoch = torch.zeros(1, dtype=torch.int32, device=dev)
            self.head_scratch = {
                s: _buf(DH.scratch_bytes(DH.HeadBuild(tokens=s, tp=self.tp)), dev)
                for s in config.widths
            }
            self._head = True

        for build, key, abi in self._builds():
            compile_build(build, key, abi)

        self.failstop = FailStop(
            self.health[FAILED : FAILED + 1],
            self.rank,
            self.tp,
            tp.cpu_group,
            NAME,
            "without --monokernel, the stock draft step runs",
        )
        logger.info(
            "%s: widths 1-%d, front %s, head %s; other steps run the stock "
            "draft step",
            NAME,
            config.max_tokens,
            "fused" if self._fc is not None else "stock",
            "fused" if self._head else "stock",
        )

    def _eps(self):
        mixer = self.model.model.hyper_connection_mixer
        return (
            self.layer.attn_hyper_connection.config.rms_norm_eps,
            self.layer.mlp_hyper_connection.config.rms_norm_eps,
            mixer.config.rms_norm_eps,
        )

    def _builds(self):
        from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks, front
        from vllm.models.qwen4_exp.amd.specialized_decode.mono import draft_front as DF
        from vllm.models.qwen4_exp.amd.specialized_decode.mono import draft_head as DH

        tp = self.tp
        eps_attn, eps_mlp, eps_mixer = self._eps()
        model = self.model.model
        out = []
        for s in self.config.widths:
            out.append((front.build, front.FrontBuild(tokens=s, tp=tp, self_msq=True,
                        eps=eps_attn), front.ABI))  # fmt: skip
            out.append((front.build, front.FrontBuild(tokens=s, tp=tp, self_msq=False,
                        eps=eps_mlp), front.ABI))  # fmt: skip
            out.append((front.build, front.FrontBuild(tokens=s, tp=tp, self_msq=False,
                        eps=eps_mixer), front.ABI))  # fmt: skip
            if self._fc is not None:
                out.append((DF.build, DF.FrontBuild(tokens=s, tp=tp, eps=eps_attn,
                            ne_eps=model.pre_fc_norm_embedding.variance_epsilon,
                            nh_eps=model.pre_fc_norm_hidden.variance_epsilon),
                            DF.ABI))  # fmt: skip
            out.append((blocks.build_oproj, self._oproj_key(s), blocks.OPROJ_ABI))
            out.append((blocks.build_moe, blocks.MoeBuild(tokens=s, tp=tp),
                        blocks.MOE_ABI))  # fmt: skip
            if self._head:
                out.append((DH.build, DH.HeadBuild(tokens=s, tp=tp,
                            rows=self._head_rows), DH.ABI))  # fmt: skip
        return out

    def _oproj_key(self, s: int):
        from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks

        o = o_proj(self.layer)
        shuffled = mxfp4_shuffled(o)
        return blocks.OprojBuild(
            tokens=s, tp=self.tp, oq=shuffled is not None, og=True,
            shuffled=shuffled is not False,
        )  # fmt: skip

    # ------------------------------------------------------------ weights
    def prepare_weights(self) -> None:
        """After the weights are loaded, outside any graph capture: the
        mixer's down as the kernel reads it (the down GEMM always reads
        ``layout.DOWN_ROWS`` rows; the mixer has none, the pad rows are
        zero -- only the low-rank groups ever reach the gate-mix that
        produces ``bin_out``, so the pad cannot perturb it, and the
        mixer front's own injection output is unused), the fc projections
        gathered whole on every rank if they are plain bf16 (the fused front
        needs every rank's full weight), and whether the draft's lm_head is
        an online-MXFP4 copy on the ASM FP4 GEMM's layout (the fused head
        only ever reads that one layout)."""
        mixer = self.model.model.hyper_connection_mixer
        down = mixer.input_mix_weight_down.weight
        padded = down.new_zeros(L.DOWN_ROWS, L.HH)
        padded[: down.size(0)].copy_(down)
        self._mixer_w_down = padded
        self._fc: tuple[torch.Tensor, torch.Tensor] | None = None
        model = self.model.model
        lins = (("fc_embedding", model.fc_embedding), ("fc_hidden", model.fc_hidden))
        if all(_fc_error(name, lin, self.tp) is None for name, lin in lins):
            from vllm.distributed import tensor_model_parallel_all_gather

            self._fc = tuple(
                tensor_model_parallel_all_gather(lin.weight.data, dim=0).contiguous()
                for _, lin in lins
            )
        lm_head = self.model.lm_head
        self._head_rows = None
        si = getattr(lm_head, "shard_indices", None)
        if si is not None and mxfp4_shuffled(lm_head) is True:
            rows = lm_head.weight.size(0)
            if rows % 32 == 0:
                self._head_rows = rows

    # ------------------------------------------------------------ step
    def eligible(self, input_ids, intermediate_tensors) -> bool:
        """A pure decode draft step of 1 .. ``config.max_tokens`` rows: unlike
        the target's verify steps, the drafter is bound to the MTP draft
        model alone, so every step it sees is one decode token a request."""
        if intermediate_tensors is not None or input_ids is None:
            return False
        from vllm.forward_context import in_piecewise_cudagraph

        if in_piecewise_cudagraph():
            return False
        return 1 <= input_ids.size(0) <= self.config.max_tokens

    def run(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(sample_hidden_states, multi_hidden)``, the stock
        ``Qwen4ExpMultiTokenPredictor.forward``'s contract. ``hidden_states``:
        the backbone's pre-final-mixer multi stream [S, HC H] (from the
        target's first draft step, or a prior draft step's)."""
        from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks, front
        from vllm.models.qwen4_exp.amd.specialized_decode.runtime.layers import (
            qsa_core,
        )

        s = input_ids.size(0)
        self.epoch.add_(1)
        model = self.model.model
        layer = self.layer
        attn_hc = layer.attn_hyper_connection
        x = torch.empty(s, L.HIDDEN, dtype=hidden_states.dtype, device=self.epoch.device)
        inj = torch.empty(s, L.HC, dtype=x.dtype, device=x.device)
        if self._fc is not None:
            state = self._fused_front(input_ids, hidden_states, inputs_embeds, x, inj)
        else:
            state = self._stock_front(input_ids, hidden_states, inputs_embeds)
            front.hc_front(
                front.FrontBuild(tokens=s, tp=self.tp, self_msq=True,
                                  eps=attn_hc.config.rms_norm_eps),  # fmt: skip
                state=state, msq_in=None, hc_w=attn_hc.hc_norm.weight,
                w_down=attn_hc.input_mix_weight_down_block_inject.weight,
                w_up=attn_hc.input_mix_weight_up.weight,
                inj_out=inj, bin_out=x, scratch=self.attn_scratch[s],
                epoch=self.epoch, health=self.health, alarm=self.alarm,
                layer=ATTN_LAYER,
            )  # fmt: skip

        core, gate = qsa_core(layer.self_attn, x, positions)

        mlp_hc = layer.mlp_hyper_connection
        mlp = layer.mlp
        ex = mlp.experts.routed_experts
        o = o_proj(layer)
        shuffled = mxfp4_shuffled(o)
        state2 = torch.empty_like(state)
        msq = torch.empty(blocks.NRED, s, L.HC, dtype=torch.float32, device=state.device)
        blocks.oproj_block(
            self._oproj_key(s), core=core, gate=gate, w_o=o.weight,
            w_os=o.weight_scale if shuffled is not None else None,
            state=state, inj=inj, out=state2, msq_out=msq,
            peers=self.peer_table, rank=self.rank, epoch=self.epoch,
            health=self.health, alarm=self.alarm, layer=ATTN_LAYER,
        )  # fmt: skip

        mlp_x = torch.empty_like(x)
        mlp_inj = torch.empty_like(inj)
        front.hc_front(
            front.FrontBuild(tokens=s, tp=self.tp, self_msq=False,
                              eps=mlp_hc.config.rms_norm_eps),  # fmt: skip
            state=state2, msq_in=msq, hc_w=mlp_hc.hc_norm.weight,
            w_down=mlp_hc.input_mix_weight_down_block_inject.weight,
            w_up=mlp_hc.input_mix_weight_up.weight,
            inj_out=mlp_inj, bin_out=mlp_x, scratch=self.mlp_scratch[s],
            epoch=self.epoch, health=self.health, alarm=self.alarm,
            layer=MLP_LAYER,
        )  # fmt: skip

        state3 = torch.empty_like(state2)
        msq2 = torch.empty_like(msq)
        blocks.moe_block(
            blocks.MoeBuild(tokens=s, tp=self.tp), x=mlp_x, w_gate=mlp.gate.weight,
            w_sg=mlp.shared_expert_gate.weight, w13=ex.w13_weight,
            w13s=ex.w13_weight_scale, w2=ex.w2_weight, w2s=ex.w2_weight_scale,
            state=state2, inj=mlp_inj, out=state3, msq_out=msq2,
            scratch=self.mlp_scratch[s], peers=self.peer_table, rank=self.rank,
            epoch=self.epoch, health=self.health, alarm=self.alarm,
            layer=MLP_LAYER,
        )  # fmt: skip

        mixer = model.hyper_connection_mixer
        h = torch.empty(s, L.HIDDEN, dtype=state3.dtype, device=state3.device)
        mixer_inj = torch.empty(s, L.HC, dtype=state3.dtype, device=state3.device)
        front.hc_front(
            front.FrontBuild(tokens=s, tp=self.tp, self_msq=False,
                              eps=mixer.config.rms_norm_eps),  # fmt: skip
            state=state3, msq_in=msq2, hc_w=mixer.hc_norm.weight,
            w_down=self._mixer_w_down, w_up=mixer.input_mix_weight_up.weight,
            inj_out=mixer_inj, bin_out=h, scratch=self.attn_scratch[s],
            epoch=self.epoch, health=self.health, alarm=self.alarm,
            layer=MIXER_LAYER,
        )  # fmt: skip
        return h, state3

    def _fused_front(self, input_ids, hidden_states, inputs_embeds, x, inj):
        """``mono.draft_front``: the fc projections and the attention HC
        front in one launch -> ``state`` [S, HC H]; ``x`` / ``inj`` written."""
        from vllm.models.qwen4_exp.amd.specialized_decode.mono import draft_front as DF

        model = self.model.model
        layer = self.layer
        attn_hc = layer.attn_hyper_connection
        s = x.size(0)
        if inputs_embeds is None:
            inputs_embeds = model.embed_input_ids(input_ids)
        state = torch.empty(s, L.HH, dtype=x.dtype, device=x.device)
        DF.draft_front(
            DF.FrontBuild(
                tokens=s, tp=self.tp, eps=attn_hc.config.rms_norm_eps,
                ne_eps=model.pre_fc_norm_embedding.variance_epsilon,
                nh_eps=model.pre_fc_norm_hidden.variance_epsilon,
            ),  # fmt: skip
            emb=inputs_embeds.contiguous(), prev=hidden_states.contiguous(),
            w_ne=model.pre_fc_norm_embedding.weight,
            w_nh=model.pre_fc_norm_hidden.weight,
            w_fe=self._fc[0], w_fh=self._fc[1], hc_w=attn_hc.hc_norm.weight,
            w_down=attn_hc.input_mix_weight_down_block_inject.weight,
            w_up=attn_hc.input_mix_weight_up.weight,
            state_out=state, bin_out=x, inj_out=inj,
            scratch=self.attn_scratch[s], epoch=self.epoch, health=self.health,
            alarm=self.alarm, layer=ATTN_LAYER,
        )  # fmt: skip
        return state

    def _stock_front(self, input_ids, hidden_states, inputs_embeds) -> torch.Tensor:
        """``Qwen4ExpMultiTokenPredictor.forward``'s embedding branch, eager:
        embed, the two pre-norms, fc_embedding / fc_hidden, the embedding
        residual -> ``state`` [S, HC H] bf16 contiguous."""
        model = self.model.model
        if inputs_embeds is None:
            inputs_embeds = model.embed_input_ids(input_ids)
        inputs_embeds = model.pre_fc_norm_embedding(inputs_embeds)
        inputs_embeds = model.fc_embedding(inputs_embeds)
        num_tokens = hidden_states.shape[0]
        hidden_states = hidden_states.view(num_tokens, L.HC, L.HIDDEN)
        hidden_states = model.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(
            num_tokens, L.HC, L.HIDDEN
        )
        hidden_states = model.fc_hidden(hidden_states)
        hidden_states = inputs_embeds.unsqueeze(-2) + hidden_states
        return hidden_states.flatten(-2).contiguous()

    # ------------------------------------------------------------ head
    def head_eligible(self, lm_head, logits_processor, hidden_states) -> bool:
        """Greedy draft tokens of at most ``config.max_tokens`` rows from the
        fused MXFP4 draft head (every rank decides alike: shapes and module
        state, not sampling parameters outside its control)."""
        from vllm.forward_context import in_piecewise_cudagraph

        if self._head is None or in_piecewise_cudagraph():
            return False
        if not 1 <= hidden_states.size(0) <= self.config.max_tokens:
            return False
        if logits_processor.soft_cap is not None or logits_processor.scale != 1.0:
            return False
        return (
            mxfp4_shuffled(lm_head) is True
            and lm_head.tp_size == self.tp
            and lm_head.weight.size(0) == self._head_rows
        )

    def top_tokens(self, lm_head, hidden_states) -> torch.Tensor:
        """Greedy draft tokens: ``LogitsProcessor.get_top_tokens``'s result
        from one ``mono.draft_head`` launch."""
        from vllm.models.qwen4_exp.amd.specialized_decode.mono import draft_head as DH

        s = hidden_states.size(0)
        self.head_epoch.add_(1)
        si = lm_head.shard_indices
        out = torch.empty(s, dtype=torch.int64, device=hidden_states.device)
        DH.draft_head(
            DH.HeadBuild(tokens=s, tp=self.tp, rows=lm_head.weight.size(0)),
            x=hidden_states.contiguous(), w=lm_head.weight, ws=lm_head.weight_scale,
            valid=si.org_vocab_end_index - si.org_vocab_start_index,
            vocab_start=si.org_vocab_start_index, out=out,
            scratch=self.head_scratch[s], peers=self.head_peers.addresses,
            rank=self.rank, epoch=self.head_epoch, health=self.health,
            alarm=self.alarm, layer=HEAD_LAYER,
        )  # fmt: skip
        return out

    def after_step(self) -> None:
        """Once per executed draft step. Raises on every rank once a rank
        gave up an in-kernel wait."""
        self.failstop.after_step()
