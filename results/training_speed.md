# Training Speed

## Scope

Training speed is measured as end-to-end training-step throughput on C4 batches.
Batches are prefetched to CPU before timing. The measured loop includes transfer
to the GPU, forward, backward, and the AdamW update; dataset loading and
tokenization are excluded. Tokens/s counts all sequence positions, including
padding.

Command:

```bash
bash scripts/run_training_throughput.sh
```

## A800 PCIe Results

Hardware and measurement setup:

- GPU: NVIDIA A800 80GB PCIe
- software: CUDA 12.8, PyTorch 2.8.0, Triton 3.4.0
- precision: BF16 mixed precision (FP32 master weights and optimizer states,
  BF16 autocast compute), matching the PPL training protocol
- sequence length: 256
- warmup: 10 training steps
- measurement: 100 training steps per method and preset

| Model | Batch | Method | Throughput (tokens/s) | Throughput retained | Peak memory (GiB) |
| :--- | ---: | :--- | ---: | ---: | ---: |
| 350M | 128 | Dense | **44,506.65** | 100.00% | 59.93 |
| 350M | 128 | MoC | 42,397.93 | 95.26% | **47.19** |
| 1B | 64 | Dense | **9,722.94** | 100.00% | 66.56 |
| 1B | 64 | MoC | 9,537.89 | 98.10% | **52.82** |

MoC retains **95.26%** of Dense throughput at 350M and **98.10%** at 1B,
while reducing total peak allocated memory by **21.26%** and **20.65%**,
respectively. Ratios are computed before rounding; bold marks the better absolute
throughput or memory value within each model pair.

Peak memory is measured with `torch.cuda.max_memory_allocated()` after warmup
initializes optimizer state and resets the peak statistics. It includes weights,
optimizer states, gradients, activations, and temporary tensors throughout the
100-step measurement window. It is not saved-activation memory, reserved memory,
or `nvidia-smi` process memory. These GiB values are separate from the decimal-GB
single-step measurements in [Training Memory](memory.md).

Each row is one 100-step timed run. These measurements do not establish
statistical significance for small throughput differences.
