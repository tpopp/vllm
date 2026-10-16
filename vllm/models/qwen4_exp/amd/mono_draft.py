# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM side of the Qwen3.8-Flash-Next MTP mono draft step
(``VLLM_ROCM_MONO_DRAFT=1``, MI355X, TP2).

A draft step of at most ``MAX_TOKENS`` rows runs the single MTP decoder layer
(QSA attention, HC, sparse MoE) as three persistent launches on a freshly
built multi-stream state:

- the stock draft front (embed, the two pre-norms, fc_embedding / fc_hidden,
  the embedding residual), eager: a fused kernel replaces it later (``_front``)
- K1's HC front (``hc_front``) of the draft layer's attention HC -> the block
  input, then vLLM's QSA core up to the output gate
- K2 (``layer_post``): o_proj + its HC combine, the sparse MoE -> the next
  materialized state and its sums of squares
- the final mixer's mix as another ``hc_front`` launch: ``use_combine=False``
  so its down projection has no injection rows; the kernel always wants a
  336-row down (``L.DOWN_ROWS``, 16 past the 320 low-rank rows for an
  injection group), so this pads it with zero rows once at weight-check time.
  The down GEMM's injection group only ever feeds ``inj_out`` (``hc.py``'s
  ``_down_out``, group 20's rows < 4), never the gate-mix that produces
  ``bin_out``, so the padding cannot change the mixed output; this launch's
  own injection output is unused.

This is a separate ``QwenMonoDecode`` from the target's (``mono_decode``):
its own epoch, scratch and peer buffer, so its mailbox tags never alias the
target's, and the draft's own HC front / K2 / mixer-front launches use three
distinct tags within one step (the draft layer's index, that index + 1, and
K2's half of the tag space at that index).
"""

from functools import cached_property

import torch

from vllm.forward_context import get_forward_context, in_piecewise_cudagraph
from vllm.logger import init_logger
from vllm.models.qwen3_5.amd import mono_base as B
from vllm.models.qwen4_exp.amd.mono import layout as L
from vllm.models.qwen4_exp.amd.mono_decode import _qsa_core
from vllm.platforms import current_platform

logger = init_logger(__name__)


def enabled() -> bool:
    from vllm import envs

    return envs.VLLM_ROCM_MONO_DRAFT and current_platform.is_rocm()


def fused_head() -> bool:
    """Greedy draft tokens from one ``mono.head`` launch when the draft
    lm_head is an online-MXFP4 copy on the ASM layout
    (``draft_lm_head_quantization="mxfp4"``)."""
    from vllm import envs

    return envs.VLLM_ROCM_MONO_DRAFT_HEAD


def position0() -> bool:
    """Also take draft position 0 (each request's verify rows, k + 1 a request,
    spec-as-decode) when it fits ``MAX_TOKENS`` rows and runs in a FULL graph."""
    from vllm import envs

    return envs.VLLM_ROCM_MONO_DRAFT_POS0


def fused_front() -> bool:
    """The front as one ``mono.front`` launch (else stock ops + ``hc_front``)."""
    from vllm import envs

    return envs.VLLM_ROCM_MONO_DRAFT_FRONT


class DraftMonoDecode(B.QwenMonoDecode):
    """Bound to one ``Qwen4ExpMultiTokenPredictor``; built at model init on
    every rank."""

    LABEL = "Qwen3.8-Flash-Next MTP draft"
    TP_SIZES = L.TPS
    MAX_TOKENS = L.MAX_TOKENS
    BLOCKS = L.BLOCKS
    MTP = True
    ENV_NAME = "VLLM_ROCM_MONO_DRAFT"

    def __init__(self, model, vllm_config):
        super().__init__(model, vllm_config)
        self._head = None
        if not self.ok or not fused_head():
            return
        from vllm.distributed import get_tp_group
        from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
        from vllm.models.qwen4_exp.amd.mono import head

        dev = self.epoch.device
        tp = get_tp_group()
        # its own epoch, scratch and peers: a head also follows draft steps
        # that ran stock (draft position 0), so it bumps its own epoch
        self._head = {
            "epoch": torch.zeros(1, dtype=torch.int32, device=dev),
            "scratch": {
                n: torch.zeros(
                    (head.scratch_bytes(head.HeadBuild(tokens=n, tp=self.tp)) + 255)
                    // 256
                    * 256,
                    dtype=torch.uint8,
                    device=dev,
                )
                for n in self.k1_scratch
            },
            "peers": PeerBuffer(
                head.peer_bytes(self.tp), tp.cpu_group, self.rank, self.tp, dev
            ),
        }
        self._head["peers"].bytes.zero_()

    @cached_property
    def lay(self) -> L.Layout:
        return L.Layout(self.tp)

    def head_eligible(self, lm_head, logits_processor, hidden_states) -> bool:
        """``get_top_tokens`` of at most ``MAX_TOKENS`` rows on the MXFP4
        draft head (every rank decides alike: shapes and module state)."""
        if self._head is None or in_piecewise_cudagraph():
            return False
        if not 1 <= hidden_states.size(0) <= self.MAX_TOKENS:
            return False
        if logits_processor.soft_cap is not None or logits_processor.scale != 1.0:
            return False
        return B.asm_mxfp4(lm_head) and lm_head.tp_size == self.tp

    def top_tokens(self, lm_head, hidden_states) -> torch.Tensor:
        """Greedy draft tokens: ``LogitsProcessor.get_top_tokens``'s result
        from one ``mono.head`` launch."""
        from vllm.models.qwen4_exp.amd.mono import head

        s = hidden_states.size(0)
        hd = self._head
        hd["epoch"].add_(1)
        si = lm_head.shard_indices
        out = torch.empty(s, dtype=torch.int64, device=hidden_states.device)
        head.draft_head(
            head.HeadBuild(tokens=s, tp=self.tp, rows=lm_head.weight.size(0)),
            x=hidden_states.contiguous(),
            w=lm_head.weight,
            ws=lm_head.weight_scale,
            valid=si.org_vocab_end_index - si.org_vocab_start_index,
            vocab_start=si.org_vocab_start_index,
            out=out,
            scratch=hd["scratch"][s],
            peers=hd["peers"].addresses,
            rank=self.rank,
            epoch=hd["epoch"],
            slot=0,
        )
        return out

    def refusals(self, vllm_config) -> list[str]:
        """As the base's, but without the base's MTP-conditional checks: they
        read ``self.MAX_TOKENS`` as a verify step's row budget tied to
        ``spec.num_speculative_tokens`` and require the GDN AITER/Triton
        kernels a verify step's GDN core needs -- neither applies here, since
        a draft step's width is its request count (checked per step in
        ``eligible``, not statically here) and the draft layer is QSA-only,
        no GDN core at all."""
        from vllm.distributed import get_pp_group, get_tp_group
        from vllm.platforms.rocm import get_cdna_version

        pc = vllm_config.parallel_config
        why = []
        if get_cdna_version() not in B.CDNA_VERSIONS:
            why.append(f"needs CDNA{'/'.join(map(str, B.CDNA_VERSIONS))}")
        if get_tp_group().world_size not in self.TP_SIZES:
            why.append(
                f"needs tensor parallel size {'/'.join(map(str, self.TP_SIZES))}"
            )
        if get_pp_group().world_size != 1:
            why.append("needs no pipeline parallelism")
        if pc.enable_expert_parallel or pc.enable_eplb or pc.data_parallel_size > 1:
            why.append("needs tensor parallelism only: no expert or data parallelism")
        if pc.decode_context_parallel_size > 1 or pc.prefill_context_parallel_size > 1:
            why.append("needs no context parallelism")
        if vllm_config.lora_config is not None:
            why.append("needs no LoRA")
        spec = vllm_config.speculative_config
        if spec is None or spec.method != "mtp":
            why.append("needs MTP speculative decoding")
        if vllm_config.kv_transfer_config is not None:
            why.append("needs no KV connector")
        cus = current_platform.num_compute_units(
            torch.accelerator.current_device_index()
        )
        if cus < self.BLOCKS:
            why.append(f"needs {self.BLOCKS} compute units, the GPU has {cus}")
        return why

    def probe_prefix(self, model) -> str:
        """The QSA side cache's metadata key: ``self_attn.layer_name`` is a
        plain ``FlashAttentionMetadata`` (no ``num_prefills``; the merged QSA
        owner's builder subclasses ``FlashAttentionMetadataBuilder``). The
        indexer's raw-key-cache side cache is a ``QSAForwardMetadata``
        (``num_prefills`` / ``num_decodes`` / ``num_decode_tokens``), the same
        key ``qsa.py`` itself reads the request counts from."""
        return model.layers[0].self_attn.indexer.raw_key_cache.prefix

    def unsupported(self, model) -> str | None:
        from vllm.distributed import get_tp_group
        from vllm.models.qwen4_exp.amd.model import Qwen4ExpSparseMoeBlock
        from vllm.models.qwen4_exp.amd.mono import k2
        from vllm.models.qwen4_exp.amd.qsa import Qwen4ExpQSAAttention

        c = model.config
        shape = (
            c.hidden_size == L.HIDDEN
            and c.hc_count == L.HC
            and c.hc_lowrank == L.LORA
            and c.num_experts == L.E
            and c.num_experts_per_tok == L.TOPK
            and c.moe_intermediate_size == L.MOE_I
            and c.shared_expert_intermediate_size == L.MOE_I
            and c.hidden_act == "silu"
            and getattr(c, "norm_topk_prob", True)
        )
        if not shape:
            return "not the Qwen3.8-Flash-Next shape"
        if len(model.layers) != 1:
            return "more than one MTP layer"
        mixer = model.hyper_connection_mixer
        if mixer is None or mixer.use_combine:
            return "no final HC mixer"
        layer = model.layers[0]
        if layer.ple is not None:
            return "a PLE layer"
        # the draft's three launches tag the attention HC front at
        # layer.layer_idx, the mixer HC front at layer.layer_idx + 1, and K2
        # at TAG_SLOTS // 2 + layer.layer_idx: the mixer's slot must stay
        # below K2's half of the tag space
        if layer.layer_idx + 1 >= k2.TAG_SLOTS // 2:
            return "draft layer index too large for the mailbox tag space"
        mlp = layer.mlp
        if not isinstance(mlp, Qwen4ExpSparseMoeBlock):
            return "dense MLP"
        if not mlp.is_fused_shared_expert_enabled:
            return "shared expert not fused"
        for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
            if not hc.use_combine or hc.pad_size != L.DOWN_ROWS - L.LORA - L.HC:
                return "HC variant"
        a = layer.self_attn
        if not isinstance(a, Qwen4ExpQSAAttention):
            return "attention without the QSA indexer"
        core = L.Layout(get_tp_group().world_size).CORE
        if a.num_heads * a.head_dim != core:
            return "attention width"
        if not a.attn_output_gate:
            return "attention without the output gate"
        return None

    def buffer_bytes(self):
        from vllm.models.qwen4_exp.amd.mono import k1, k2

        def k1_bytes(s):
            return max(
                k1.scratch_bytes(
                    k1.K1Build(tokens=s, tp=self.tp, self_msq=m, front_only=True)
                )
                for m in (False, True)
            )

        def k2_bytes(s):
            return k2.scratch_bytes(k2.K2Build(tokens=s, tp=self.tp))

        return k1_bytes, k2_bytes, k2.peer_bytes(L.MAX_TOKENS, self.tp)

    def _fc_error(self, name: str, lin) -> str | None:
        """Why ``lin`` (``fc_embedding`` / ``fc_hidden``) is unreadable as the
        fused front's full, gathered weight, else None: it must be a plain
        bf16 linear (``ColumnParallelLinear`` with no quantization the kernel
        does not expect), its local shard ``[H / tp, H]`` as the module
        shards it."""
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        if not isinstance(lin.quant_method, UnquantizedLinearMethod):
            return f"{name}: quantized ({type(lin.quant_method).__name__})"
        want = (L.HIDDEN // self.tp, L.HIDDEN)
        if lin.weight.shape != want or lin.weight.dtype != torch.bfloat16:
            return (
                f"{name}: {tuple(lin.weight.shape)} {lin.weight.dtype}, "
                f"want bf16 {want}"
            )
        return None

    def weights_unsupported(self) -> str | None:
        lay = self.lay
        fp4s = (torch.uint8, torch.float4_e2m1fn_x2)
        layer = self.model.layers[0]
        mlp = layer.mlp
        ex = mlp.experts.routed_experts
        why = B.experts_error(ex, L.E + 1)
        if why is not None:
            return why
        if ex.w13_weight.shape != (L.E + 1, 2 * lay.RIP, L.HIDDEN // 2):
            return f"draft layer: w13 {tuple(ex.w13_weight.shape)}"
        if ex.w2_weight.shape != (L.E + 1, L.HIDDEN, lay.RIP // 2):
            return f"draft layer: w2 {tuple(ex.w2_weight.shape)}"
        dense: dict[str, torch.Tensor] = {
            "gate": mlp.gate.weight,
            "shared_expert_gate": mlp.shared_expert_gate.weight,
        }
        for name in ("attn_hyper_connection", "mlp_hyper_connection"):
            hc = getattr(layer, name)
            dense[f"{name}.hc_norm"] = hc.hc_norm.weight
            dense[f"{name}.down"] = hc.input_mix_weight_down_block_inject.weight
            dense[f"{name}.up"] = hc.input_mix_weight_up.weight
        o = B.o_proj(layer)
        if B.asm_mxfp4(o):
            if (
                o.weight.shape != (L.HIDDEN, lay.CORE // 2)
                or o.weight.dtype not in fp4s
            ):
                return f"draft layer: o_proj {tuple(o.weight.shape)}"
        else:
            dense["o_proj"] = o.weight
        mixer = self.model.hyper_connection_mixer
        dense["mixer.hc_norm"] = mixer.hc_norm.weight
        dense["mixer.up"] = mixer.input_mix_weight_up.weight
        down = mixer.input_mix_weight_down.weight
        if down.shape != (L.LORA, L.HH) or down.dtype != torch.bfloat16:
            return f"mixer down: {tuple(down.shape)}"
        why = B.dense_error(layer.layer_idx, dense)
        if why is not None:
            return why
        if getattr(self, "_mixer_w_down", None) is None:
            return "the padded mixer down not built (prepare_weights)"
        # the fused front is optional (``_front`` is the eager fallback): only
        # check it when ``prepare_weights`` actually engaged it, so an
        # unreadable fc_embedding / fc_hidden never disables the rest of the
        # draft, just the fused front
        fc = getattr(self, "_fc", None)
        if fc is not None:
            model = self.model
            for fname, lin in (
                ("fc_embedding", model.fc_embedding),
                ("fc_hidden", model.fc_hidden),
            ):
                why = self._fc_error(fname, lin)
                if why is not None:
                    return why
            if any(w.shape != (L.HIDDEN, L.HIDDEN) for w in fc):
                return f"fused front gathered weights: {[tuple(w.shape) for w in fc]}"
        return None

    def prepare_weights(self) -> None:
        """After loading, outside any graph capture: the mixer's down as the
        kernel reads it. The down GEMM always reads ``DOWN_ROWS`` rows (an
        injection group past the low-rank one); the mixer has none, so the pad
        rows are zero. Only groups below ``INJ_GROUP`` reach the gate-mix that
        produces ``bin_out`` (``hc._down_out``), so the pad cannot perturb it,
        and this launch's injection output is unused."""
        down = self.model.hyper_connection_mixer.input_mix_weight_down.weight
        padded = down.new_zeros(L.DOWN_ROWS, L.HH)
        padded[: down.size(0)].copy_(down)
        self._mixer_w_down = padded
        self._fc = None
        if fused_front():
            from vllm.distributed import tensor_model_parallel_all_gather

            model = self.model
            lins = (
                ("fc_embedding", model.fc_embedding),
                ("fc_hidden", model.fc_hidden),
            )
            if all(self._fc_error(name, lin) is None for name, lin in lins):
                # every rank's full fc weights: the kernel computes every
                # output column, so no all-gather runs in the step
                self._fc = tuple(
                    tensor_model_parallel_all_gather(
                        lin.weight.data, dim=0
                    ).contiguous()
                    for _, lin in lins
                )
                self.front_scratch = {
                    n: torch.zeros(
                        (_front_bytes(n, self.tp) + 255) // 256 * 256,
                        dtype=torch.uint8,
                        device=self.epoch.device,
                    )
                    for n in self.k1_scratch
                }

    def check_weights(self) -> None:
        """As the base's, but an unreadable weight falls back to the stock
        draft step instead of raising: unlike ``VLLM_ROCM_MONO_DECODE``'s
        explicit opt-in for the target, the draft is an optimization, and
        ``prepare_weights`` may never have run (e.g. ``--load-format
        dummy``), which the base's checks would otherwise surface as an
        ``AttributeError``, not the intended ``ValueError``."""
        if self._weights_checked:
            return
        why = self.weights_unsupported()
        if why is not None:
            logger.warning_once(
                "%s mono draft off: %s. Falling back to the stock draft step.",
                self.LABEL,
                why,
            )
            self.ok = False
        self._weights_checked = True

    def eligible(self, input_ids, intermediate_tensors) -> bool:
        """A pure decode draft step of at most ``MAX_TOKENS`` rows: unlike the
        target, this never sees a verify step (it is bound to the draft
        module only), so it needs no GDN state-index bookkeeping. Draft
        position 0 stays on stock: the QSA builder's
        ``supports_spec_as_decode`` also reports ``num_prefills == 0`` there,
        but its query_len is k + 1 (every request's whole draft chain), not
        the 1-row-a-request decode this step assumes."""
        from vllm import envs

        if not envs.VLLM_ROCM_MONO_DRAFT_LAYER:
            return False
        if not self.ok or intermediate_tensors is not None:
            return False
        if in_piecewise_cudagraph():
            return False
        if input_ids is None or not 1 <= input_ids.size(0) <= self.MAX_TOKENS:
            return False
        md = get_forward_context().attn_metadata
        if not isinstance(md, dict):
            return False
        m = md.get(self._probe)
        if m is None or m.num_prefills != 0:
            return False
        if m.num_decode_tokens != m.num_decodes and not position0():
            return False
        self.check_weights()
        return self.ok

    def forward(self, input_ids, positions, hidden_states, inputs_embeds):
        """Returns ``(sample_hidden_states, multi_hidden)``, the stock
        ``Qwen4ExpMultiTokenPredictor.forward``'s contract. ``hidden_states``:
        the backbone's pre-final-mixer multi stream [S, 4H] (from the target's
        first draft step, or a prior draft step's)."""
        from vllm.models.qwen4_exp.amd.mono import hc as mhc
        from vllm.models.qwen4_exp.amd.mono import k1, k2

        s = input_ids.size(0)
        self.first_step(s)
        self.epoch.add_(1)
        layer = self.model.layers[0]
        attn_hc = layer.attn_hyper_connection
        x = torch.empty(
            s, L.HIDDEN, dtype=hidden_states.dtype, device=self.epoch.device
        )
        inj = torch.empty(s, L.HC, dtype=x.dtype, device=x.device)
        if getattr(self, "_fc", None) is not None:
            state = self._fused_front(input_ids, hidden_states, inputs_embeds, x, inj)
        else:
            state = self._front(input_ids, hidden_states, inputs_embeds)
            k1.hc_front(
                k1.K1Build(
                    tokens=s,
                    tp=self.tp,
                    self_msq=True,
                    front_only=True,
                    eps=attn_hc.config.rms_norm_eps,
                ),
                state=state,
                msq_in=None,
                hc_w=attn_hc.hc_norm.weight,
                w_down=attn_hc.input_mix_weight_down_block_inject.weight,
                w_up=attn_hc.input_mix_weight_up.weight,
                inj_out=inj,
                bin_out=x,
                scratch=self.k1_scratch[s],
                epoch=self.epoch,
                layer=layer.layer_idx,
            )
        core, gate = _qsa_core(layer.self_attn, x, positions)

        mlp_hc = layer.mlp_hyper_connection
        mlp = layer.mlp
        ex = mlp.experts.routed_experts
        o = B.o_proj(layer)
        oq = B.asm_mxfp4(o)
        state2 = torch.empty_like(state)
        msq = torch.empty(mhc.NRED, s, L.HC, dtype=torch.float32, device=state.device)
        k2.layer_post(
            k2.K2Build(
                tokens=s,
                tp=self.tp,
                eps=mlp_hc.config.rms_norm_eps,
                oq=oq,
                og=gate is not None,
            ),
            core=core,
            gate=gate,
            state=state,
            inj_a=inj,
            w_o=o.weight,
            w_os=o.weight_scale if oq else None,
            hc_w=mlp_hc.hc_norm.weight,
            w_down=mlp_hc.input_mix_weight_down_block_inject.weight,
            w_up=mlp_hc.input_mix_weight_up.weight,
            w_gate=mlp.gate.weight,
            w_sg=mlp.shared_expert_gate.weight,
            w13=ex.w13_weight,
            w13s=ex.w13_weight_scale,
            w2=ex.w2_weight,
            w2s=ex.w2_weight_scale,
            out=state2,
            msq_out=msq,
            scratch=self.k2_scratch[s],
            peers=self.peers.addresses,
            rank=self.rank,
            epoch=self.epoch,
            layer=k2.TAG_SLOTS // 2 + layer.layer_idx,
        )

        mixer = self.model.hyper_connection_mixer
        h = torch.empty(s, L.HIDDEN, dtype=state2.dtype, device=state2.device)
        mixer_inj = torch.empty(s, L.HC, dtype=state2.dtype, device=state2.device)
        k1.hc_front(
            k1.K1Build(
                tokens=s,
                tp=self.tp,
                self_msq=False,
                front_only=True,
                eps=mixer.config.rms_norm_eps,
            ),
            state=state2,
            msq_in=msq,
            hc_w=mixer.hc_norm.weight,
            w_down=self._mixer_w_down,
            w_up=mixer.input_mix_weight_up.weight,
            inj_out=mixer_inj,
            bin_out=h,
            scratch=self.k1_scratch[s],
            epoch=self.epoch,
            layer=layer.layer_idx + 1,
        )
        return h, state2

    def _fused_front(self, input_ids, hidden_states, inputs_embeds, x, inj):
        """``mono.front``: the fc projections and the attention HC front in one
        launch -> ``state`` [S, 4H]; ``x`` / ``inj`` written."""
        from vllm.models.qwen4_exp.amd.mono import front

        model = self.model
        layer = model.layers[0]
        attn_hc = layer.attn_hyper_connection
        s = x.size(0)
        if inputs_embeds is None:
            inputs_embeds = model.embed_input_ids(input_ids)
        state = torch.empty(s, L.HH, dtype=x.dtype, device=x.device)
        front.draft_front(
            front.FrontBuild(
                tokens=s,
                tp=self.tp,
                eps=attn_hc.config.rms_norm_eps,
                ne_eps=model.pre_fc_norm_embedding.variance_epsilon,
                nh_eps=model.pre_fc_norm_hidden.variance_epsilon,
            ),
            emb=inputs_embeds.contiguous(),
            prev=hidden_states.contiguous(),
            w_ne=model.pre_fc_norm_embedding.weight,
            w_nh=model.pre_fc_norm_hidden.weight,
            w_fe=self._fc[0],
            w_fh=self._fc[1],
            hc_w=attn_hc.hc_norm.weight,
            w_down=attn_hc.input_mix_weight_down_block_inject.weight,
            w_up=attn_hc.input_mix_weight_up.weight,
            state_out=state,
            bin_out=x,
            inj_out=inj,
            scratch=self.front_scratch[s],
            epoch=self.epoch,
            slot=layer.layer_idx,
        )
        return state

    def _front(self, input_ids, hidden_states, inputs_embeds) -> torch.Tensor:
        """``Qwen4ExpMultiTokenPredictor.forward``'s embedding branch, eager:
        embed, the two pre-norms, fc_embedding / fc_hidden, the embedding
        residual -> ``state`` [S, 4H] bf16 contiguous."""
        model = self.model
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


def _front_bytes(s: int, tp: int) -> int:
    from vllm.models.qwen4_exp.amd.mono import front

    return front.scratch_bytes(front.FrontBuild(tokens=s, tp=tp))
