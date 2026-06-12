# MoC: Mixture-of-Channels

This repository provides the PyTorch implementation of **Mixture-of-Channels (MoC)** for LLaMA-style feed-forward networks, together with benchmark scripts for:

- training activation-memory reduction;
- single-layer FFN training latency;
- training-step throughput;
- C4 validation perplexity;
- single-layer FFN inference latency;
- end-to-end decode latency.

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

The recommended environment is CUDA 12.8 + PyTorch 2.8 on an NVIDIA A800/A100-class GPU.

```bash
conda env create -f environment.yml
conda activate moc
```

For a pip-only setup:

```bash
pip install -e .
```

Some inference kernels are compiled lazily through PyTorch C++/CUDA extension tooling on first use. Make sure `nvcc`, a compatible C++ compiler, and Ninja are available if you run the optimized inference benchmarks.

## Quick Start

Instantiate a small MoC model:

```python
import torch
from moc import PRESETS, build_model

config = PRESETS["130m"]
model = build_model(config, ffn_type="moc").cuda().bfloat16()
tokens = torch.randint(0, config.vocab_size, (2, 256), device="cuda")
logits, loss = model(tokens, labels=tokens)
loss.backward()
```

Run the core benchmarks:

```bash
bash scripts/run_memory.sh
bash scripts/run_training_latency.sh
bash scripts/run_training_throughput.sh
bash scripts/run_ppl.sh
bash scripts/run_ffn_latency.sh
bash scripts/run_decode.sh
```

`scripts/run_ppl.sh` starts a full C4 pretraining run by default. For a quick launch check, use `STOP_AT_STEP=1`, `EVAL_MAX_BATCHES=1`, and an output directory under `/tmp` or `data/checkpoints/`.

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

The single-layer FFN training and inference latency benchmarks use fixed-shape hidden-state tensors because they measure the FFN kernels themselves rather than the data pipeline.

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

The memory benchmark first runs one warmup AdamW step to initialize optimizer state, then resets CUDA peak-memory stats and reports `torch.cuda.max_memory_allocated()` for one measured `forward + backward + AdamW step`. Reported values use decimal GB; JSON outputs also include GiB and reserved-memory fields. This metric is not `nvidia-smi` process memory and is not `torch.cuda.max_memory_reserved()`. The key comparison is Dense FFN vs MoC under the same model shape, batch size, sequence length, dtype, and optimizer.

See [results/memory.md](results/memory.md).

### Training Latency

Single-layer FFN training latency measures forward and backward time for the Standard FFN and MoC FFN under the same fixed shape.

See [results/training_latency.md](results/training_latency.md).

### Training Throughput

The training-throughput benchmark measures C4 training-step throughput after prefetching batches to CPU, so the reported number focuses on model computation rather than gzip/tokenizer IO.

See [results/training_speed.md](results/training_speed.md).

### C4 Validation PPL

The PPL benchmark trains a selected preset with the C4 training schedule in [configs](configs), then evaluates validation perplexity with shifted causal-LM loss and padding labels ignored. The PPL entry point uses standard BF16 mixed precision (FP32 master weights and optimizer states, BF16 autocast compute), and reports `val_ppl` as the public validation perplexity.

See [results/ppl.md](results/ppl.md).

### FFN Inference Latency

Single-layer FFN latency is measured for Dense FFN, global MoC, and MoC 2:8.

See [results/inference_ffn.md](results/inference_ffn.md).

### End-to-End Decode

End-to-end decode measures the full single-token generation loop with compiled model execution.

See [results/decode.md](results/decode.md).

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

- Benchmark numbers can vary with CUDA, PyTorch, compiler, GPU clocks, and driver versions.
- Long C4 pretraining checkpoints should be stored under `data/checkpoints/`, which is ignored by Git.

## Citation

If you use this repository, please cite the MoC work.
