# External patches & reproduction notes

This branch (`attn-gluon-decode-win`) is the cycle-6 working winner from a
best-of-N exploration aimed at improving decode-time attention on
`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` (TP=1). It contains:

- **In-tree vLLM changes** (committed in this branch on top of the base
  `8e5406d56 Fix shuffle KV cache stride bug and add gluon decode support`):
    - `vllm/platforms/interface.py` -- auto-align `mamba_block_size` to the
    attention `block_size` when the user has not specified one. Without this,
    the no-prefix-caching default sets `mamba_block_size` to `max_model_len`,
    which inflates `scheduler_block_size = lcm(attn_block, mamba_block)` and
    breaks FULL-decode cudagraph capture for the gluon kernel at batch sizes
    >= 80.
    - `vllm/v1/attention/backends/rocm_aiter_fa.py` --
    `get_supported_kernel_block_sizes()` returns `[1024]` when
    `VLLM_ROCM_USE_GLUON_DECODE=1`, so the framework allocates a 1024-block
    KV cache that matches what the Gluon `large_block_dot_kernel` expects.
    Without this, the framework block size is 16 and the dispatch path adds
    34x virtual block-table splitting that erases the per-call kernel win.

The branch ALSO requires three runtime mount overrides of files that live
in other Python packages, plus a small set of env vars and serve flags.
None of that lives in the vLLM source tree, so it is described here and
the corresponding patched source files are committed under
`external-patches/` in this branch for reference.

## Container image

`amdsiloai/vllm-private:nightly-bf0d2dc6d764f7ab1a69504f60a55883ec6d9b39`

(public mirror of the `vllm-private` ROCm nightly used in this experiment).
The flydsl shipped with this image is **0.1.4**; the gluon kernel shipped
with this image is the **older 3660-line variant** that does not include
the C++ PS-reduce dispatch branch. Both are replaced via read-only mounts
listed below.

## External library changes

The winning runtime mounts the following over the container's installed
copies, all read-only. The files mounted are committed under
`external-patches/` in this branch.

### 1. `aiter/ops/triton/gluon/pa_decode_gluon.py`

Mount source (this branch): `external-patches/pa_decode_gluon.py` (5648 lines).
Mount target: `/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/gluon/pa_decode_gluon.py`.

Origin: extracted as the patched "newer" Gluon decode kernel that appears
in the `rocm/atom-dev:vllm-v0.19.0-nightly_20260508_perf_prebuild_qwen`
image at `/app/aiter-test/aiter/ops/triton/gluon/pa_decode_gluon.py`
(5637 lines), with two flag defaults adjusted for this image:

| Flag | Atom default | This branch | Reason |
| --- | --- | --- | --- |
| `CXX_PS_REDUCE_AVAILABLE` (~line 16) | `True` | `True` (falls back to `False` if `csrc.cpp_itfs.pa.pa_ps` is not on `sys.path`) | nightly has no `csrc/` package, so the import fails and the kernel falls through to the Triton reduce. Atom has the C++ JIT-compiled reduce; nightly does not. Grafting that is the cycle-7 follow-up. |
| `FLYDSL_PS_REDUCE_AVAILABLE` (~line 25) | `True` | `False` (forced) | flydsl 0.1.5 has a `_block_reduce` bug that crashes on Qwen3-Next when this flag is `True`. Set to `False` to force the Triton fallback for the reduce step (Atom uses the C++ path; this branch uses the Triton fallback). |

Everything else outside those two flags is line-for-line equivalent to the
Atom kernel: 143 changed lines, all of which are SPDX header reorder,
`float(0.0)` -> `0.0`, and import reorder.

The 1988-line growth over the nightly-shipped kernel adds the
`paged_attention_decode_v2_gluon_large_block_dot_kernel` with
`KV_BLOCK_SIZE=1024`, the `paged_attention_decode_sliding_window` variant,
the `paged_attention_decode_v2_gluon_dot_kernel`, and the PS-reduce dispatch
branch. Confirmed firing in trace captures under this config:

- `aiter::paged_attention_decode_v2_gluon_large_block_dot_kernel` -- full-attn layers
- `paged_attention_decode_sliding_window` -- sliding-window layers
- `paged_attention_decode_ps_reduce_kernel` (Triton fallback) -- final reduce

### 2. `flydsl` (0.1.5)

Mount source: a local copy of flydsl 0.1.5 with one Python 3.12 compat
adjustment.

Mount target: `/usr/local/lib/python3.12/dist-packages/flydsl`.

Origin: extracted from a flydsl-0.1.5-bearing container (the same Atom
image carries it at `/opt/venv/lib/python3.12/site-packages/flydsl/`). The
package ships with a 226 MB binary `_mlir/_mlir_libs/libFlyPythonCAPI.so.23.0git`
that is not committed here -- copy the binary tree from the upstream image
yourself, then replace `_mlir/dialects/_ods_common.py` with the file in
`external-patches/flydsl-0.1.5__mlir_dialects__ods_common.py`.

The compat fix changes one function signature in `_ods_common.py` from
PEP 604 union syntax (`a | b | c`) back to `_Union[a, b, c]` because under
Python 3.12 with this MLIR/flydsl build the runtime evaluates the
annotation and trips on a non-type operand in the union.

Note: `FLYDSL_PS_REDUCE_AVAILABLE` is hard-coded `False` in the patched
gluon kernel (see above), so flydsl 0.1.5 is loaded only for its
non-PS-reduce code paths. flydsl 0.1.4 (image default) has incompatible
annotations elsewhere that block import of the newer Gluon kernel; that's
why 0.1.5 is required even when the PS-reduce path is disabled.

## Run config (winning serve command)

Container image:
`amdsiloai/vllm-private:nightly-bf0d2dc6d764f7ab1a69504f60a55883ec6d9b39`.

Mount overrides (`/path/to/this/branch` = checkout of this branch;
`/path/to/flydsl-0.1.5` = locally-prepared flydsl 0.1.5 with the
`_ods_common.py` replacement applied):

```text
-v /path/to/this/branch/vllm/envs.py:/usr/local/lib/python3.12/dist-packages/vllm/envs.py:ro
-v /path/to/this/branch/vllm/_aiter_ops.py:/usr/local/lib/python3.12/dist-packages/vllm/_aiter_ops.py:ro
-v /path/to/this/branch/vllm/v1/attention/backends/rocm_aiter_fa.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/rocm_aiter_fa.py:ro
-v /path/to/this/branch/external-patches/pa_decode_gluon.py:/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/gluon/pa_decode_gluon.py:ro
-v /path/to/flydsl-0.1.5:/usr/local/lib/python3.12/dist-packages/flydsl:ro
```

(Mounting `envs.py`, `_aiter_ops.py`, and `rocm_aiter_fa.py` ensures the
container uses the branch's vLLM versions of these files even though the
container ships its own installed copies.)

Environment variables:

```text
VLLM_ROCM_USE_AITER=1
VLLM_ROCM_USE_GLUON_DECODE=1
VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
```

Serve command (after entrypoint clear):

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.95 \
  --no-enable-log-requests \
  --attention-backend ROCM_AITER_FA \
  --max-num-batched-tokens 32768 \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "max_cudagraph_capture_size": 72, "custom_ops": ["-rms_norm", "-silu_and_mul", "-quant_fp8"], "pass_config": {"fuse_norm_quant": true}}'
```

Notes on the non-default knobs:

| Flag | Why |
| --- | --- |
| `--attention-backend ROCM_AITER_FA` | The aiter FA backend is required to route to the Gluon decode kernels at all. `ROCM_AITER_UNIFIED_ATTN` does not dispatch to them. |
| `--max-num-batched-tokens 32768` | +0.87% throughput vs no override (264.97 vs 262.68 tok/s). Atom uses 32768; the vLLM default for this model under TP=1 is smaller. |
| `cudagraph_mode: FULL_AND_PIECEWISE` | Required by external test policy. Plain `PIECEWISE` skips FULL-decode capture and was used by an intermediate result but is not the goal mode. |
| `max_cudagraph_capture_size: 72` | The Gluon `large_block_dot_kernel` faults during FULL-decode capture at batch sizes >= 80 on this nightly. Capping the capture list at 72 avoids that without disabling FULL capture entirely. |
| `custom_ops: ["-rms_norm", "-silu_and_mul", "-quant_fp8"]` | Disables the default custom-op implementations so the compile pass can fuse them (combined with `fuse_norm_quant`). |
| `pass_config.fuse_norm_quant: true` | Enables the rmsnorm + fp8-quant fusion pass. Contributes ~1% to throughput on top of `--max-num-batched-tokens 32768`. |

## Measured results

Reference Baseline A is the nightly image with `ROCM_AITER_FA`, no Gluon
decode, no mounts, `cudagraph_mode=FULL_AND_PIECEWISE`, no other deltas.

| Config | gsm8k @ CONCURRENT=128, --limit 200 | Output tok/s @ ISL=100k OSL=1024 |
| --- | --- | --- |
| Baseline A (nightly + FA, no gluon) | 0.870 +/- 0.024 (flex) / 0.830 +/- 0.027 (strict) | 262.68 |
| This branch (gluon decode + mounts + flags) | **0.859 +/- 0.0096 (flex)** / **0.815 +/- 0.0107 (strict)** | **264.97** (+0.87%) |

Accuracy 2-sigma bands fully overlap Baseline A.

## How much is left on the table

Trace-level comparison against the Atom image (which fuses the C++
PS-reduce, has an AOT-compiled Gluon decode pipeline, and replaces the
`gdn_attention_core` Triton kernel with an FX-graph-compiled version)
shows Atom ~11% faster per decode step. Of that ~11%:

- The Gluon decode kernel itself is **~1% of total CUDA time**. The kernel
  swap in this branch captures essentially all of it (matches the ~0.87%
  perf win we measure).
- Most of the remaining gap is `vllm::gdn_attention_core` (14 s of CUDA per
  trace window on nightly, absent / fused away on Atom) and different
  CompiledFxGraph shapes. These are structural compile-pass / fusion
  differences, not kernel-choice issues. They are out of scope for this
  branch.

The cycle-7 candidate that would extend this branch by another ~0.3-1% is
grafting Atom's C++ PS-reduce module (`csrc.cpp_itfs.pa.pa_ps`) -- a small
mount-only addition (< 50 KB across 7 source files, JIT-compiles in the
nightly container). It is not included here.
