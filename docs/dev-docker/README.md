# vllm FP8 Latency and Throughput benchmarks with vLLM on the AMD Instinct™ MI300X accelerator

Documentation for Inferencing with vLLM on AMD Instinct™ MI300X platforms.

## Overview

vLLM is a toolkit and library for large language model (LLM) inference and serving. It deploys the PagedAttention algorithm, which reduces memory consumption and increases throughput by leveraging dynamic key and value allocation in GPU memory. vLLM also incorporates many recent LLM acceleration and quantization algorithms, such as fp8 GeMM, fp8 KV cache, continuous batching, flash attention, hip graph, tensor parallel, GPTQ, AWQ, and token speculation. In addition, AMD implements high-performance custom kernels and modules in vLLM to enhance performance further.

This documentation includes information for running the popular Llama 3.1 series models from Meta using a pre-built AMD vLLM docker image optimized for an AMD Instinct™ MI300X or MI325X accelerator. The container is publicly available at [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html)

The pre-built image includes:

- ROCm™ 6.4.1
- HipblasLT 0.15
- vLLM 0.9.1
- PyTorch 2.7

## Pull latest Docker Image

Pull the most recent validated docker image with `docker pull rocm/vllm-dev:main`

## What is New

- No need to specify the --compilation-config parameter, these options were turned on by default
- Fixed llama3.1 405b CAR issue (no longer need --disable-custom-all-reduce)
- Fixed +rms_norm custom kernel issue
- Added quick reduce (set VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP to enable. Supported modes are FP, INT8, INT6, INT4)
- Mitigated the commandr model causing GPU crash through a workaround until the driver issue is fixed

## Known Issues and Workarounds

- AITER does not support fp8 kv cache

## Performance Results

The data in the following tables is a reference point to help users validate observed performance. It should not be considered as the peak performance that can be delivered by AMD Instinct™ MI300X accelerator with vLLM. See the MLPerf section in this document for information about MLPerf 4.1 inference results. The performance numbers above were collected using the steps below.
*Note Benchmarks were run with benchmark scripts from [v0.8.5](https://github.com/vllm-project/vllm/tree/v0.8.5/benchmarks)*

### Throughput Measurements

The table below shows performance data where a local inference client is fed requests at an infinite rate and shows the throughput client-server scenario under maximum load.

| Model | Precision | TP Size | Input | Output | Num Prompts | Max Num Seqs | Throughput (tokens/s) |
|-------|-----------|---------|-------|--------|-------------|--------------|-----------------------|
| Llama 3.1 70B (amd/Llama-3.1-70B-Instruct-FP8-KV) | FP8 | 8 | 128 | 2048 | 3200 | 3200 | 12638.9  |
|       |           |         | 128   | 4096   | 1500        | 1500         | 10756.8               |
|       |           |         | 500   | 2000   | 2000        | 2000         | 10691.7               |
|       |           |         | 2048  | 2048   | 1500        | 1500         | 7354.9                |
| Llama 3.1 405B (amd/Llama-3.1-405B-Instruct-FP8-KV) | FP8 | 8 | 128 | 2048 | 1500 | 1500 | 3912.8 |
|       |           |         | 128   | 4096   | 1500        | 1500         | 3084.7                |
|       |           |         | 500   | 2000   | 2000        | 2000         | 2935.9                |
|       |           |         | 2048  | 2048   | 500         | 500          | 2191.5                |

*TP stands for Tensor Parallelism.*

Supermicro AS-8125GS-TNMR2 with 2x AMD EPYC 9554 Processors, 2.25 TiB RAM, 8x AMD Instinct MI300X (192GiB, 750W) GPUs, Ubuntu 22.04, and amdgpu driver 6.8.5

### Latency Measurements

The table below shows latency measurement, which typically involves assessing the time from when the system receives an input to when the model produces a result.

| Model | Precision | TP Size | Batch Size | Input | Output | MI300X Latency (sec) |
|-------|-----------|----------|------------|--------|---------|-------------------|
| Llama 3.1 70B (amd/Llama-3.1-70B-Instruct-FP8-KV) | FP8 | 8 | 1 | 128 | 2048 | 17.236 |
| | | | 2 | 128 | 2048 | 18.057 |
| | | | 4 | 128 | 2048 | 18.450 |
| | | | 8 | 128 | 2048 | 19.677 |
| | | | 16 | 128 | 2048 | 22.072 |
| | | | 32 | 128 | 2048 | 24.932 |
| | | | 64 | 128 | 2048 | 33.287 |
| | | | 128 | 128 | 2048 | 46.484 |
| | | | 1 | 2048 | 2048 | 17.500 |
| | | | 2 | 2048 | 2048 | 18.055 |
| | | | 4 | 2048 | 2048 | 18.858 |
| | | | 8 | 2048 | 2048 | 20.161 |
| | | | 16 | 2048 | 2048 | 22.347 |
| | | | 32 | 2048 | 2048 | 25.966 |
| | | | 64 | 2048 | 2048 | 35.324 |
| | | | 128 | 2048 | 2048 | 52.394 |
| Llama 3.1 405B (amd/Llama-3.1-405B-Instruct-FP8-KV) | FP8 | 8 | 1 | 128 | 2048 | 48.453 |
| | | | 2 | 128 | 2048 | 49.268 |
| | | | 4 | 128 | 2048 | 51.136 |
| | | | 8 | 128 | 2048 | 54.226 |
| | | | 16 | 128 | 2048 | 57.274 |
| | | | 32 | 128 | 2048 | 68.901 |
| | | | 64 | 128 | 2048 | 88.631 |
| | | | 128 | 128 | 2048 | 117.027 |
| | | | 1 | 2048 | 2048 | 48.362 |
| | | | 2 | 2048 | 2048 | 49.121 |
| | | | 4 | 2048 | 2048 | 52.347 |
| | | | 8 | 2048 | 2048 | 54.471 |
| | | | 16 | 2048 | 2048 | 57.841 |
| | | | 32 | 2048 | 2048 | 70.538 |
| | | | 64 | 2048 | 2048 | 91.452 |
| | | | 128 | 2048 | 2048 | 125.471 |

*TP stands for Tensor Parallelism.*

Supermicro AS-8125GS-TNMR2 with 2x AMD EPYC 9554 Processors, 2.25 TiB RAM, 8x AMD Instinct MI300X (192GiB, 750W) GPUs, Ubuntu 22.04, and amdgpu driver 6.8.5

## Reproducing Benchmarked Results

### Preparation - Obtaining access to models

The vllm-dev docker image should work with any model supported by vLLM.  When running with FP8, AMD has quantized models available for a variety of popular models, or you can quantize models yourself using Quark.  If needed, the vLLM benchmark scripts will automatically download models and then store them in a Hugging Face cache directory for reuse in future tests. Alternatively, you can choose to download the model to the cache (or to another directory on the system) in advance.

Many HuggingFace models, including Llama-3.1, have gated access.  You will need to set up an account at (https://huggingface.co), search for the model of interest, and request access if necessary. You will also need to create a token for accessing these models from vLLM: open your user profile (https://huggingface.co/settings/profile), select "Access Tokens", press "+ Create New Token", and create a new Read token.

### System optimization

Before running performance tests you should ensure the system is optimized according to the [ROCm Documentation](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html).  In particular, it is important to ensure that NUMA auto-balancing is disabled.

*Note: Check that NUMA balancing is properly set by inspecting the output of the command below, which should have a value of 0, with, `cat /proc/sys/kernel/numa_balancing`*

### Launch AMD vLLM Docker

Download and launch the docker.  The HF_TOKEN is required to be set (either here or after launching the container) if you want to allow vLLM to download gated models automatically; use your HuggingFace token in place of `<token>` in the command below:

```bash
docker run -it --rm --ipc=host --network=host --group-add render \
    --privileged --security-opt seccomp=unconfined \
    --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
    --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    -e HF_HOME=/data \
    -e HF_TOKEN=<token> \
    -v /data:/data \
    rocm/vllm-dev:main
```

Note: The instructions in this document use `/data` to store the models.  If you choose a different directory, you will also need to make that change to the host volume mount when launching the docker container.  For example, `-v /home/username/models:/data` in place of `-v /data:/data` would store the models in /home/username/models on the host.  Some models can be quite large; please ensure that you have sufficient disk space prior to downloading the model.  Since the model download may take a long time, you can use `tmux` or `screen` to avoid getting disconnected.

### Downloading models with huggingface-cli

If you would like want to download models directly (instead of allowing vLLM to download them automatically), you can use the huggingface-cli inside the running docker container. (remove an extra white space) Login using the token that you created earlier. (Note, it is not necessary to save it as a git credential.)

```bash
huggingface-cli login
```

You can download a model to the huggingface-cache directory using a command similar to the following (substituting the name of the model you wish to download):

```bash
sudo mkdir -p /data/huggingface-cache
sudo chmod -R a+w /data/huggingface-cache
HF_HOME=/data/huggingface-cache huggingface-cli download meta-llama/Llama-3.1-405B-Instruct --exclude "original/*"
```

Alternatively, you may wish to download the model to a specific directory, e.g. so you can quantize the model with Quark:

```bash
sudo mkdir -p /data/llama-3.1
sudo chmod -R a+w /data/llama-3.1
huggingface-cli download meta-llama/Llama-3.1-405B-Instruct --exclude "original/*" --local-dir /data/llama-3.1/Llama-3.1-405B-Instruct
```

In the benchmark commands provided later in this document, replace the model name (e.g. `amd/Llama-3.1-405B-Instruct-FP8-KV`) with the path to the model (e.g. `/data/llama-3.1/Llama-3.1-405B-Instruct`)

### Use pre-quantized models

AMD has provided [FP8-quantized versions](https://huggingface.co/collections/amd/quark-quantized-ocp-fp8-models-66db7936d18fcbaf95d4405c) of several models in order to make them easier to run on MI300X / MI325X, including:

- <https://huggingface.co/amd/Llama-3.1-8B-Instruct-FP8-KV>
- <https://huggingface.co/amd/Llama-3.1-70B-Instruct-FP8-KV>
- <https://huggingface.co/amd/Llama-3.1-405B-Instruct-FP8-KV>

Some models may be private to those who are members of <https://huggingface.co/amd>.

These FP8 quantized checkpoints were generated with AMD’s Quark Quantizer. For more information about Quark, please refer to <https://quark.docs.amd.com/latest/quark_example_torch_llm_gen.html>

### Quantize your own models

This is an optional step if you would like to quantize your own model instead of using AMD's pre-quantized models.  These instructions use Llama-3.1-405B as an example, but the commands are similar for other models.

First download the model from <https://huggingface.co/meta-llama/Llama-3.1-405B> to the /data/llama-3.1 directory as described above.

[Download and install Quark](https://quark.docs.amd.com/latest/install.html)

Run the quantization script in the example folder using the following command line:

```bash
# path to quark quantization script
export QUARK_DIR=/data/quark-0.6.0+dba9ca364/examples/torch/language_modeling/llm_ptq/quantize_quark.py
# path to Model 
export MODEL_DIR=/data/llama-3.1/Llama-3.1-405B-Instruct
python3 $QUARK_DIR \
--model_dir $MODEL_DIR \
--output_dir Llama-3.1-405B-Instruct-FP8-KV \
--kv_cache_dtype fp8 \
--quant_scheme w_fp8_a_fp8 \
--num_calib_data 128 \
--model_export quark_safetensors \
--no_weight_matrix_merge \
--multi_gpu
```

Note: the `--multi_gpu` parameter can be omitted for small models that fit on a single GPU.

## Performance testing with AMD vLLM Docker

### Performance environment variables

Some environment variables enhance the performance of the vLLM kernels on the MI300X / MI325X accelerator. See the AMD Instinct MI300X workload optimization guide for more information.

```bash
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1

```

### vLLM engine performance settings

vLLM provides a number of engine options which can be changed to improve performance.  Refer to the [vLLM Engine Args](https://docs.vllm.ai/en/stable/usage/engine_args.html) documentation for the complete list of vLLM engine options.

Below is a list of a few of the key vLLM engine arguments for performance; these can be passed to the vLLM benchmark scripts:
- **--max-model-len** : Maximum context length supported by the model instance. Can be set to a lower value than model configuration value to improve performance and gpu memory utilization.
- **--max-num-batched-tokens** : The maximum prefill size, i.e., how many prompt tokens can be packed together in a single prefill. Set to a higher value to improve prefill performance at the cost of higher gpu memory utilization. 65536 works well for LLama models.
- **--max-num-seqs** : The maximum decode batch size (default 256). Using larger values will allow more prompts to be processed concurrently, resulting in increased throughput (possibly at the expense of higher latency).  If the value is too large, there may not be enough GPU memory for the KV cache, resulting in requests getting preempted.  The optimal value will depend on the GPU memory, model size, and maximum context length.
- **--max-seq-len-to-capture** : Maximum sequence length for which Hip-graphs are captured and utilized. It's recommended to use Hip-graphs for the best decode performance. The default value of this parameter is 8K, which is lower than the large context lengths supported by recent models such as LLama. Set this parameter to max-model-len or maximum context length supported by the model for best performance.
- **--gpu-memory-utilization** : The ratio of GPU memory reserved by a vLLM instance. Default value is 0.9.  Increasing the value (potentially as high as 0.99) will increase the amount of memory available for KV cache.  When running in graph mode (i.e. not using `--enforce-eager`), it may be necessary to use a slightly smaller value of 0.92 - 0.95 to ensure adequate memory is available for the HIP graph.

### Latency Benchmark

vLLM's benchmark_latency.py script measures end-to-end latency for a specified model, input/output length, and batch size.

You can run latency tests for FP8 models with:

```bash
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
MODEL=amd/Llama-3.1-405B-Instruct-FP8-KV
BS=1
IN=128
OUT=2048
TP=8

python3 /app/vllm/benchmarks/benchmark_latency.py \
    --distributed-executor-backend mp \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --model $MODEL \
    --batch-size $BS \
    --input-len $IN \
    --output-len $OUT \
    --tensor-parallel-size $TP \
    --num-iters-warmup 3 \
    --num-iters 5

```

When measuring models with long context lengths, performance may improve by setting `--max-model-len` to a smaller value.  It is important, however, to ensure that the `--max-model-len` is at least as large as the IN + OUT token counts.

To estimate Time To First Token (TTFT) with the benchmark_latency.py script, set the OUT to 1 token.  It is also recommended to use `--enforce-eager` to get a more accurate measurement of the time that it actually takes to generate the first token.  (For a more comprehensive measurement of TTFT, use the Online Serving Benchmark.)

For additional information about the available parameters run:

```bash
/app/vllm/benchmarks/benchmark_latency.py -h
```

### Throughput Benchmark

vLLM's benchmark_throughput.py script measures offline throughput.  It can either use an input dataset or random prompts with fixed input/output lengths.

You can run throughput tests for FP8 models with:

```bash
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
MODEL=amd/Llama-3.1-405B-Instruct-FP8-KV
IN=128
OUT=2048
TP=8
PROMPTS=1500
MAX_NUM_SEQS=1500

python3 /app/vllm/benchmarks/benchmark_throughput.py \
    --distributed-executor-backend mp \
    --kv-cache-dtype fp8 \
    --dtype float16 \
    --disable-detokenize \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --model $MODEL \
    --max-model-len 8192 \
    --max-num-batched-tokens 131072 \
    --max-seq-len-to-capture 131072 \
    --input-len $IN \
    --output-len $OUT \
    --tensor-parallel-size $TP \
    --num-prompts $PROMPTS \
    --max-num-seqs $MAX_NUM_SEQS
```

For FP16 models, remove `--kv-cache-dtype fp8`.

When measuring models with long context lengths, performance may improve by setting `--max-model-len` to a smaller value (8192 in this example).  It is important, however, to ensure that the `--max-model-len` is at least as large as the IN + OUT token counts.

It is important to tune vLLM’s --max-num-seqs value to an appropriate value depending on the model and input/output lengths.  Larger values will allow vLLM to leverage more of the GPU memory for KV Cache and process more prompts concurrently.  But if the value is too large, the KV cache will reach its capacity and vLLM will have to cancel and re-process some prompts.  Suggested values for various models and configurations are listed below.

For models that fit on a single GPU, it is usually best to run with `--tensor-parallel-size 1`.  Requests can be distributed across multiple copies of vLLM running on different GPUs.  This will be more efficient than running a single copy of the model with `--tensor-parallel-size 8`.  (Note: the benchmark_throughput.py script does not include direct support for using multiple copies of vLLM)

For optimal performance, the PROMPTS value should be a multiple of the MAX_NUM_SEQS value -- for example, if MAX_NUM_SEQS=1500 then the PROMPTS value could be 1500, 3000, etc.  If PROMPTS is smaller than MAX_NUM_SEQS then there won’t be enough prompts for vLLM to maximize concurrency.

For additional information about the available parameters run:

```bash
python3 /app/vllm/benchmarks/benchmark_throughput.py -h
```

### Online Serving Benchmark

Benchmark Llama-3.1-70B with input 4096 tokens, output 512 tokens and tensor parallelism 8 as an example,

```bash
export VLLM_USE_TRITON_FLASH_ATTN=0
vllm serve amd/Llama-3.1-70B-Instruct-FP8-KV \
    --swap-space 16 \
    --disable-log-requests \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --dtype float16 \
    --max-model-len 8192 \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 65536 \
    --gpu-memory-utilization 0.99 \
    --num_scheduler-steps 10
```

For FP16 models, remove `--kv-cache-dtype fp8`. Change port (for example --port 8005) if port=8000 is currently being used by other processes.

Run client in a separate terminal. Use port_id from previous step else port-id=8000.

```bash
python /app/vllm/benchmarks/benchmark_serving.py \
    --port 8000 \
    --model amd/Llama-3.1-70B-Instruct-FP8-KV \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 512 \
    --request-rate 1 \
    --ignore-eos \
    --num-prompts 500 \
    --percentile-metrics ttft,tpot,itl,e2el
```

Once all prompts are processed, terminate the server gracefully (ctrl+c).

### Running DeepSeek-V3 and DeepSeek-R1

We have experimental support for running both DeepSeek-V3 and DeepSeek-R1 models.
*Note there are currently limitations and `--max-model-len` cannot be greater than 32768*

```bash
docker run -it --rm --ipc=host --network=host --group-add render \
    --privileged --security-opt seccomp=unconfined \
    --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
    --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    -e VLLM_USE_TRITON_FLASH_ATTN=1 \
    -e VLLM_ROCM_USE_AITER=1 \
    -e  VLLM_MLA_DISABLE=0 \
    rocm/vllm-dev:main

# Online serving
vllm serve deepseek-ai/DeepSeek-V3 \
    --disable-log-requests \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --max-model-len 131072 \
    --block-size=1 

python3 /app/vllm/benchmarks/benchmark_serving.py \
    --backend vllm \
    --model deepseek-ai/DeepSeek-V3 \
    --max-concurrency 256\
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 128 \
    --num-prompts 1000

# Offline throughput 
python3 /app/vllm/benchmarks/benchmark_throughput.py --model deepseek-ai/DeepSeek-V3 \
    --input-len <> --output-len <> --tensor-parallel-size 8 \
    --quantization fp8 --kv-cache-dtype fp8 --dtype float16 \
    --max-model-len 32768 --block-size=1 --trust-remote-code

# Offline Latency
python /app/vllm/benchmarks/benchmark_latency.py --model deepseek-ai/DeepSeek-V3 \
--tensor-parallel-size 8 --trust-remote-code --max-model-len 32768 --block-size=1 \
--batch-size <> --input-len <> --output-len <>
```

### CPX mode

Currently only CPX-NPS1 mode is supported. So ONLY tp=1 is supported in CPX mode.
But multiple instances can be started simultaneously (if needed) in CPX-NPS1 mode.

Set GPUs in CPX mode with:

```bash
rocm-smi --setcomputepartition cpx
```

Example of running Llama3.1-8B on 1 CPX-NPS1 GPU with input 4096 and output 512. As mentioned above, tp=1.

```bash
HIP_VISIBLE_DEVICES=0 \
python3 /app/vllm/benchmarks/benchmark_throughput.py \
    --max-model-len 4608 \
    --num-scheduler-steps 10 \
    --num-prompts 100 \
    --model amd/Llama-3.1-8B-Instruct-FP8-KV \
    --input-len 4096 \
    --output-len 512 \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --output-json <path/to/output.json> \
    --quantization fp8 \
    --gpu-memory-utilization 0.99
```

Set GPU to SPX mode.

```bash
rocm-smi --setcomputepartition spx
```

### Speculative Decoding

Speculative decoding is one of the key features in vLLM. It has been supported on MI300. Here below is an example of the performance benchmark w/wo speculative decoding for Llama 3.1 405B with Llama 3.1 8B as the draft model.

Without Speculative Decoding -

```bash
export VLLM_USE_TRITON_FLASH_ATTN=0
python /app/vllm/benchmarks/benchmark_latency.py --model amd/Llama-3.1-405B-Instruct-FP8-KV --max-model-len 26720 -tp 8 --batch-size 1 --input-len 1024 --output-len 128
```

With Speculative Decoding -

```bash
export VLLM_USE_TRITON_FLASH_ATTN=0
python /app/vllm/benchmarks/benchmark_latency.py --model amd/Llama-3.1-405B-Instruct-FP8-KV --max-model-len 26720 -tp 8 --batch-size 1 --input-len 1024 --output-len 128 --speculative-model amd/Llama-3.1-8B-Instruct-FP8-KV --num-speculative-tokens 5
```

You should see some performance improvement about the e2e latency.

### AITER use cases

`rocm/vllm-dev:main` image has experimental [AITER](https://github.com/ROCm/aiter) support, and can yield siginficant performance increase for some model/input/output/batch size configurations. To enable the feature make sure the following environment is set: `VLLM_ROCM_USE_AITER=1`, the default value is `0`. When building your own image follow the [Docker build steps](#Docker-manifest) using the [aiter_integration_final](https://github.com/ROCm/vllm/tree/aiter_integration_final) branch.

Some use cases include:
- amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV
- amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_RMSNORM=0
python3 /app/vllm/benchmarks/benchmark_latency.py --model amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV -tp 8 --batch-size 256 --input-len 128 --output-len 2048
```

## MMLU_PRO_Biology Accuracy Evaluation

### FP16

vllm (pretrained=models--meta-llama--Llama-3.1-405B-Instruct/snapshots/069992c75aed59df00ec06c17177e76c63296a26,dtype=float16,tensor_parallel_size=8), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 64

| Tasks |Version|    Filter    |n-shot|  Metric   |   |Value |   |Stderr|
|-------|------:|--------------|-----:|-----------|---|-----:|---|-----:|
|biology|      0|custom-extract|     5|exact_match|↑  |0.8466|±  |0.0135|

### FP8

vllm (pretrained=models--meta-llama--Llama-3.1-405B-Instruct/snapshots/069992c75aed59df00ec06c17177e76c63296a26,dtype=float16,quantization=fp8,quantized_weights_path=/llama.safetensors,tensor_parallel_size=8), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 32

| Tasks |Version|    Filter    |n-shot|  Metric   |   |Value|   |Stderr|
|-------|------:|--------------|-----:|-----------|---|----:|---|-----:|
|biology|      0|custom-extract|     5|exact_match|↑  |0.848|±  |0.0134|

## Performance

### MLPerf Performance Results

#### LLama-2-70B

Please refer to the [Benchmarking Machine Learning using ROCm and AMD GPUs: Reproducing Our MLPerf Inference Submission — ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inf-4-1/README.html) for information on reproducing MLPerf 4.1 Inference results.  Note that due to changes in vLLM, it is not possible to use these instructions with the current rocm/vllm-dev docker image. Due to recent changes in vLLM, the instructions for MLPerf 4.1 submission do not apply to the current rocm/vllm-dev docker image.

## Docker Manifest

To reproduce the release docker:

```bash
    git clone https://github.com/ROCm/vllm.git
    cd vllm
    git checkout b432b7a285aa0dcb9677380936ffa74931bb6d6f
    docker build -f docker/Dockerfile.rocm -t <your_tag> --build-arg USE_CYTHON=1 .
```

### Building AITER Image

Use AITER release candidate branch instead:

```bash
    git clone https://github.com/ROCm/vllm.git
    cd vllm
    git checkout aiter_integration_final
    docker build -f docker/Dockerfile.rocm -t <your_tag> --build-arg USE_CYTHON=1 .
```

## Changelog

20250715_aiter:
- No need to specify the --compilation-config parameter, these options were turned on by default
- Fixed llama3.1 405b CAR issue (no longer need --disable-custom-all-reduce)
- Fixed +rms_norm custom kernel issue
- Added quick reduce (set VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP to enable. Supported modes are FP, INT8, INT6, INT4)
- Mitigated the commandr model causing GPU crash through a workaround until the driver issue is fixed

20250620_aiter:
- V1 on by default (use VLLM_USE_V1=0 to override)
- Fixed detokenizers issue
- Fixed AITER MoE issues
- vLLM v0.9.1
  
20250605_aiter:
- Updated to ROCm 6.4.1 and vLLM v0.9.0.1
- AITER MHA
- IBM 3d kernel for unified attention
- Full graph capture for split attention

20250521_aiter:
- AITER V1 engine performance improvement

20250513_aiter:
- Out of memory bug fix
- PyTorch fixes
- Tunable ops fixes

20250410_aiter:
- 2-stage MoE
- MLA from AITER

20250325_aiter:
- Improved DeepSeek-V3/R1 performance
- Initial Gemma-3 enablement
- Detokenizer disablement
- Torch.compile support

20250305_aiter:
- AITER improvements
- Support for FP8 skinny GEMM

20250207_aiter:
- More performant AITER
- Bug fixes

20250205_aiter:
- [AITER](https://github.com/ROCm/aiter) support
- Performance improvement for custom paged attention
- Reduced memory overhead bug fix

20250124:
- Fix accuracy issue with 405B FP8 Triton FA
- Fixed accuracy issue with TP8

20250117:
- [Experimental DeepSeek-V3 and DeepSeek-R1 support](#running-deepseek-v3-and-deepseek-r1)
