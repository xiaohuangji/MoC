# Training Speed

## Scope

Training speed is measured as GPU training-step throughput on C4 batches.
Batches are prefetched to CPU before timing so gzip/tokenizer IO does not
dominate the model-throughput measurement.

Command:

```bash
bash scripts/run_training_throughput.sh
```

## A800 Results

Hardware and measurement setup:

- GPU: NVIDIA A800-SXM4-80GB
- precision: BF16 mixed precision (FP32 master weights and optimizer states,
  BF16 autocast compute), matching the PPL training protocol
- sequence length: 256
- warmup: 5 training steps
- measurement: 1,000 training steps

| Model | Batch | Dense tokens/s | MoC tokens/s | MoC/Dense |
| --- | ---: | ---: | ---: | ---: |
| 350M | 128 | 46,857 | 43,789 | 0.935x |
| 1B | 64 | 9,931 | 9,639 | 0.971x |

The optimized MoC training path keeps 1B throughput close to Dense. The 350M
result has a larger relative overhead because fixed Top-K and sparse-helper
costs are less amortized at this scale.
