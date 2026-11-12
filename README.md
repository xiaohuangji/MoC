# MoC: Mixture-of-Channels

This repository provides the PyTorch implementation of **Mixture-of-Channels (MoC)** for LLaMA-style feed-forward networks, with benchmarks and results for:

- training activation-memory reduction;
- single-layer FFN training latency;
- training-step throughput;
- C4 validation perplexity;
- single-layer FFN inference latency;
- end-to-end decode latency;
- Llama-3.1-8B fine-tuning on eight GLUE tasks.

## Method

For an input hidden state `X`, a LLaMA-style SwiGLU FFN computes a gate branch, an up branch, the activated gate, the FFN intermediate state, and the output:

```text
G = X W_gate
U = X W_up
S = SiLU(G)
Z = S * U
D = Z W_down
```

MoC uses the native SwiGLU gate branch as a token-wise channel-importance signal. For each token, Top-`K` is applied row-wise to `G` to form a binary mask `M`. The FFN intermediate states are then sparsified before the down projection:

```text
M  = TopK(G)
S' = SiLU(G) * M
Z' = S' * U
D  = Z' W_down
```

The training-memory saving comes from storing selected-channel activations instead of the full FFN intermediate state. With selected-activation checkpointing, the cheap element-wise states can be recomputed during backward.

Activation accounting for the FFN block is:

| Method | Saved FFN activations | Recomputed in backward | FFN activation cost |
| --- | --- | --- | --- |
| Dense FFN | `G, U, S, Z, D` | none | `4bs d_ffn + bsd` |
| Dense FFN + GCP | `G, U, D` | `S, Z` | `2bs d_ffn + bsd` |
| MoC | `G * M, U * M, S * M, Z * M, M, D` | none | `bs d_ffn + bsd` |
| MoC + GCP | `G * M, U * M, M, D` | `S * M, Z * M` | `0.6bs d_ffn + bsd` |

The MoC costs use the top-20% channel setting. When `d_ffn = 8d/3`, the Dense FFN cost is `11.67bsd`, MoC is `3.67bsd`, and MoC+GCP is `2.6bsd`.

The implementation includes:

- `dense`: standard LLaMA-style SwiGLU FFN;
- `moc`: MoC training path with fused helper kernels;
- `moc_gcp`: MoC with selected-activation checkpointing and fused helper kernels;
- inference kernels for global MoC and MoC 2:8.

## Installation

On Linux with a BF16-capable NVIDIA GPU (such as an A100 or A800), install
the [PyTorch 2.8 CUDA 12.8 wheel](https://pytorch.org/get-started/previous-versions/#v280)
and MoC from the repository root:

```bash
git clone https://github.com/xiaohuangji/MoC.git
cd MoC
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

Some inference kernels are compiled lazily through PyTorch C++/CUDA extension tooling on first use. Make sure `nvcc`, a compatible C++ compiler, and Ninja are available if you run the optimized inference benchmarks.

## Quick Start

**Try MoC on one GPU.** The [example](examples/quickstart.py) builds the 130M
model and runs one forward and backward pass on random tokens. No dataset,
tokenizer, or pretrained weights are needed.

```bash
python examples/quickstart.py
```

After backward finishes, the example prints `logits_shape: (2, 256, 32000)`
and a loss value.

**Reproduce an experiment.** Choose the path that matches what you want to measure:

| Experiment | Where to start |
| --- | --- |
| Training | [FFN latency](scripts/run_training_latency.sh) uses generated inputs. [Peak memory](scripts/run_memory.sh), [throughput](scripts/run_training_throughput.sh), and [validation PPL](scripts/run_ppl.sh) use [local C4 shards and a T5 tokenizer](data/README.md). |
| Inference | [Single-layer FFN latency](scripts/run_ffn_latency.sh) uses a fixed input; [end-to-end decode](scripts/run_decode.sh) uses a C4 prompt. |
| GLUE fine-tuning | Follow the [four-GPU recipe](benchmarks/finetuning/README.md) for Llama-3.1-8B Base: reconstruct K25 once on C4, reuse it for each task, then reload the selected checkpoint to evaluate on dev. The recipe includes the Dense baseline, extra dependencies, and a runnable MRPC example. |

### C4 smoke test

For a short C4 run after preparing the data, use **one training update and one
validation batch**. Without these overrides the PPL launcher runs the full
pretraining schedule.

```bash
STOP_AT_STEP=1 EVAL_TARGET_NONPAD_TOKENS=0 EVAL_MAX_BATCHES=1 SAVE_LATEST=0 \
  OUTPUT_DIR="$(mktemp -d)" bash scripts/run_ppl.sh
```

## Data

Training-memory, training-throughput, PPL, and decode benchmarks use C4 input batches. Place the raw C4 shards under:

```text
data/c4/
  train/.../*.json.gz
  val/.../*.json.gz
```

A local copy or symbolic link of the `t5-base` tokenizer should be placed under:

```text
data/tokenizer/
```

The C4 loader uses a GaLore-style preprocessing setup: a local `t5-base` tokenizer, per-document truncation/padding to the configured sequence length, shifted causal-LM labels, and `-100` labels for padding tokens. Training configs use deterministic document shuffling with seed `42`.

If the dataset or tokenizer is stored on another disk, create symbolic links at `data/c4` and `data/tokenizer`.

The FFN training and inference latency benchmarks use fixed-shape hidden-state tensors to exclude the data pipeline. Both measure one FFN; inference uses the same compilation and CUDA Graph replay settings for Dense and MoC.

## Model Presets

The default model presets are LLaMA-style decoder-only configurations. The `60m` preset uses the `d=512`, `d_ffn=1376`, `8`-layer configuration:

| Preset | Hidden size | FFN size | Layers | Heads | K |
| --- | ---: | ---: | ---: | ---: | ---: |
| 60m | 512 | 1376 | 8 | 8 | 256 |
| 130m | 768 | 2048 | 12 | 12 | 384 |
| 350m | 1024 | 2736 | 24 | 16 | 512 |
| 1b | 2048 | 5461 | 24 | 32 | 1024 |

Full C4 run configs for all four presets are provided in [configs](configs). The `1b` preset uses `24` decoder layers and `32` attention heads in the executable configuration.

## Main Results

### Training Memory

The benchmark compares peak allocated GPU memory (decimal GB) for one full
training step after warmup. Dense and MoC use the same model shape, batch size,
sequence length, precision, and optimizer.

See [results/memory.md](results/memory.md).

### Training Latency

Single-layer FFN training latency measures forward and backward time for the Standard FFN and MoC FFN under the same fixed shape.

See [results/training_latency.md](results/training_latency.md).

### Training Throughput

The training-throughput benchmark measures C4 training steps after prefetching
batches to CPU. On A800 80GB PCIe, MoC retains **95.26%** of Dense throughput at
350M and **98.10%** at 1B, with **21.26%** and **20.65%** lower peak allocated
memory in the same measurement windows. Each method uses 10 warmup steps followed
by 100 measured steps, with FP32 parameters and BF16 autocast compute.

See [results/training_speed.md](results/training_speed.md).

### C4 Validation PPL

The PPL benchmark trains a selected preset with the C4 training schedule in [configs](configs), then evaluates validation perplexity with shifted causal-LM loss and padding labels ignored. The PPL entry point uses standard BF16 mixed precision (FP32 master weights and optimizer states, BF16 autocast compute), and reports `val_ppl` as the public validation perplexity.

See [results/ppl.md](results/ppl.md). Ablations on the number of activated
channels and the Top-K position are in [results/ablations.md](results/ablations.md).

### Zero-Shot Evaluation

The C4-pretrained 1B Dense, MoC, and MoC 2:8 checkpoints are evaluated on MMLU,
ARC-Easy, ARC-Challenge, PIQA, and TruthfulQA without downstream fine-tuning.

See [results/downstream_zero_shot.md](results/downstream_zero_shot.md).

### FFN Inference Latency

Single-layer FFN latency is measured on a fixed input using the decoder's Dense and global MoC paths, with
`torch.compile` and CUDA Graph replay enabled for both methods. At batch size 1,
Dense takes **62.364 us** and MoC takes **39.284 us**, a **1.588x** speedup.
Speedups at batch sizes 2, 3, and 4 are **1.189x**, **1.113x**, and **1.018x**.

See [results/inference_ffn.md](results/inference_ffn.md).

### End-to-End Decode

In the batch-1 decode benchmark on an A800 80GB PCIe, MoC takes
**2.965 ms/token** versus **3.299 ms/token** for Dense (**1.113x** throughput).
Both use the same TorchInductor C++ wrapper.

See [results/decode.md](results/decode.md).

### GLUE Fine-Tuning

A single C4 reconstruction of Llama-3.1-8B Base serves all eight tasks. MoC K25
retains 3,584 of 14,336 FFN channels per token and scores **89.60** on average
on the development sets, **0.59 points below Dense** (90.20).

On QQP at batch 16 per GPU, K25 reduces peak memory from **29.81 to 25.19 GiB**
and retains **95.09%** of Dense throughput. At batch 32 on four A100 40 GB
GPUs, K25 completed 40 ordinary timed steps on **8/8** tasks versus **2/8** for Dense.

See the [GLUE fine-tuning results](results/finetuning.md) for the complete
tables, learning-rate grid and experiment settings.

## Repository Layout

```text
moc/                     Core modules and kernels
benchmarks/              Benchmark entry points
configs/                 Model and C4 training configs
scripts/                 One-command benchmark launchers
results/                 Concise benchmark summaries
docs/                    Method and benchmark notes
data/                    Local dataset layout, ignored by Git
```

## Notes

- Long C4 pretraining checkpoints should be stored under `data/checkpoints/`, which is ignored by Git.

## Citation

If you use this repository, please cite the MoC work.
