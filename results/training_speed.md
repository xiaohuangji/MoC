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
- dtype: BF16
- sequence length: 256
- warmup: 5 training steps
- measurement: 1,000 training steps

| Model | Batch | Dense tokens/s | MoC tokens/s | MoC/Dense |
| --- | ---: | ---: | ---: | ---: |
| 350M | 128 | 54,672 | 51,120 | 0.935x |
| 1B | 64 | 10,706 | 10,434 | 0.975x |

The optimized MoC training path keeps 1B throughput close to Dense. The 350M
result has a larger relative overhead because fixed Top-K and sparse-helper
costs are less amortized at this scale.
