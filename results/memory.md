# Training Memory

## Scope

This benchmark compares peak training memory for Dense FFN and MoC under the same LLaMA-style model shapes.

Memory is measured with `torch.cuda.max_memory_allocated()`. The script first runs one warmup AdamW step to initialize optimizer state, then resets CUDA peak-memory stats and measures one full `forward + backward + AdamW step`. Reported values use decimal GB. This is not `nvidia-smi` process memory and not `torch.cuda.max_memory_reserved()`.

Command:

```bash
bash scripts/run_memory.sh
```

To run only one preset, set `PRESETS_TO_RUN`, for example:

```bash
PRESETS_TO_RUN="1b" bash scripts/run_memory.sh
```

## Summary

MoC stores only selected FFN channel activations instead of the full FFN intermediate state.

Representative A800 results:

| Preset | Batch | Dense | MoC | MoC+GCP |
| --- | ---: | ---: | ---: | ---: |
| 60M | 256 | 33.54 GB | 29.12 GB | 28.58 GB |
| 130M | 256 | 54.09 GB | 44.22 GB | 43.01 GB |
| 350M | 128 | 58.99 GB | 45.71 GB | 44.10 GB |
| 1B | 64 | 60.54 GB | 47.39 GB | 45.78 GB |

Dense to MoC removes the dense FFN intermediate activations from the backward cache. MoC+GCP further reduces selected element-wise states by recomputing them in backward.

For gradient checkpointing:

| Method | Peak Memory | Relative Throughput |
| --- | ---: | ---: |
| FFN+GCP | 45.46 GB | 1.000x |
| MoC+GCP | 47.88 GB | 1.211x |

The absolute peak memory depends on allocator state, PyTorch version, and optimizer implementation. The primary comparison is the Dense-vs-MoC difference under the same run configuration.
