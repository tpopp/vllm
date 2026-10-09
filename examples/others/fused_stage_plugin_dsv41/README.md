# DeepSeek-V4.1 mono decode as an out-of-tree fused-stage provider

An example of shipping a mono kernel *outside* vLLM. It wraps the FlyDSL
kernels of [vllm#60397](https://github.com/vllm-project/vllm/pull/60397)
(adapted from ROCm/ATOM, MIT) unchanged, and plugs them into vLLM through
`vllm.fused_stages` (see `docs/design/fused_stages.md`).

```text
vllm_dsv41_mono/
  provider.py      # the only vLLM-facing code: claims, accept, runtime
  kernels/         # #60397's vllm/models/deepseek_v41/amd/mono/, vendored
pyproject.toml     # entry point: vllm.fused_stages -> rocm_mono_dsv41
vendor_kernels.sh  # populates kernels/ from the PR, rewrites one import path
```

## Build and use

```bash
git fetch https://github.com/vllm-project/vllm pull/60397/head:pr60397
./vendor_kernels.sh pr60397          # kernels/ must not import vllm
pip install -e .                     # also needs flydsl and amd-aiter

vllm serve deepseek-ai/DeepSeek-V4.1-Flash -tp 2 \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 5}' \
  --kernel-config '{"fused_stages": {"priority": {"deepseek_v41.*": ["rocm_mono_dsv41"]}}}'
```

At startup the log shows which layers each stage took, e.g.

```text
Fused stage deepseek_v41.decoder_layer@1 -> rocm_mono_dsv41 0.1.0+vllm60397 on 30 layers 3-7,9-13,...
Fused stage deepseek_v41.ffn_after_attention@1 -> rocm_mono_dsv41 0.1.0+vllm60397 on 10 layers 0-2,8,...
```

and the reason for every refusal (wrong arch, TP size, unsupported feature
such as LoRA or DBO, untagged or different weight layout).

## What moved where, compared with #60397

| #60397 (in-tree) | Here |
|---|---|
| `mono/` kernel package (8.1k lines) | `kernels/`, unchanged |
| `mono_decode.py` eligibility (CDNA4, TP 2/4, routing constants, rows ≤ 48) | `provider.py` (`accept`, `_steps`) |
| `mono_decode.py` weight extraction from module attributes | vLLM `fused_stages.py` role map + layout tags |
| `mono_decode.py` cache views / metadata reads | vLLM view builders (`dsv4.sparse_swa_mla.*@1`) |
| `VLLM_ROCM_MONO_DECODE` env var | `kernel_config.fused_stages.priority` |
| decoder-layer hooks (~20 lines) | the same hooks, generic (`try_run`) |
| lazy collective init at the first eager step | `create` at end of load (memory profiled) |

## Testing

```python
from vllm.fused_stages.testing import check_claims, compare_to_reference, replay_stress
```

* `check_claims`: static (CPU) checks of claims and predicate purity.
* `compare_to_reference`: one layer against vLLM's own code for the span
  (port of #60397's `tests/models/test_dsv41_mono_numerics.py`).
* `replay_stress`: the graph-replay race test that found the per-width
  scratch race in #60397.
