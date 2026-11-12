# FFN Inference Latency

## A800 PCIe Results

**Table 1. Single-layer, single-token FFN latency: Dense vs. global MoC (K=1024).**

| Batch size | Dense latency (us) | MoC latency (us) | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 62.364 | **39.284** | **1.588x** |
| 2 | 57.876 | **48.688** | **1.189x** |
| 3 | 58.127 | **52.220** | **1.113x** |
| 4 | 58.263 | **57.236** | **1.018x** |

Global MoC selects K=1024 of 5464 channels per token. At batch size 1, it
reduces FFN latency by **37.01%**. The advantage decreases as batch size grows;
at batch size 4, the two methods are nearly at parity.

Each measurement replays a compiled FFN on a fixed GPU-resident input.
See [end-to-end decode results](decode.md) for full-model latency.

## Configuration

- NVIDIA A800 80GB PCIe; PyTorch 2.8.0+cu128; Triton 3.4.0; BF16.
- Hidden size 2048, FFN width 5464, sequence length 1, global K=1024.
- The same parameters and input across methods in each batch setting; seed 1234.
- Both methods reuse the decoder's FFN path and
  `torch.compile(dynamic=True, options={"cpp_wrapper": True})`, followed by
  explicit CUDA Graph capture. Dense uses standard PyTorch projections.
- Each graph contains 16 calls to the same single FFN. Each timed group replays
  the graph 7 times; CUDA event time is divided by 112 FFN calls.
- Three rounds per method, each with 30 warmup replays and 80 timed groups.
  Results use all 240 groups, with method order varied between rounds.
- Compilation, graph capture, input copies, and correctness checks are excluded
  from timing. OMP and MKL use 4 threads.
- The benchmark checks compiled outputs against eager execution and verifies CUDA
  Graph replay before and after timing ([validation details](../docs/benchmarks.md#ffn-decode-latency)).

The global path uses CUB radix selection and selected-channel Up/Down kernels.

## Run

```bash
bash scripts/run_ffn_latency.sh
```

The script measures batch sizes 1 through 4 and writes
`results/ffn_latency_b{batch}.json`. Each JSON includes every timing sample,
input and parameter hashes, correctness checks, and a separate GPU-kernel
profile. Each profile sums GPU-kernel durations across 20 graph replays
(320 FFN calls per method).

To measure a single batch size:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python benchmarks/inference/benchmark_ffn_latency.py \
  --batch-size 1 --out results/ffn_latency_b1.json
```
