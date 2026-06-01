# Training Speed

## Scope

Training speed is measured as GPU training-step throughput on C4 batches. Batches are prefetched to CPU before timing so gzip/tokenizer IO does not dominate the model-throughput measurement.

Command:

```bash
bash scripts/run_training_throughput.sh
```

## A800 Results

| Model | MoC/Dense |
| --- | ---: |
| 350M | 0.959x |
| 1B | 0.984x |

The optimized MoC training path recovers most of the overhead introduced by channel selection while preserving the memory-saving mechanism.
