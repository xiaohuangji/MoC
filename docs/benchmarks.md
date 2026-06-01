# Benchmark Notes

## Environment

The benchmark runs use:

- NVIDIA A800-SXM4-80GB;
- CUDA 12.8;
- PyTorch 2.8.0;
- Triton 3.4.0;
- BF16.

## Dataset

Training-memory, training-throughput, PPL, and decode benchmarks use C4 input batches from `data/c4/`.

The C4 loader uses a GaLore-style preprocessing setup:

- local `t5-base` tokenizer files under `data/tokenizer/`;
- per-document truncation/padding to the configured sequence length;
- shifted causal-LM labels;
- `-100` labels for padding tokens;
- deterministic bounded document shuffling with seed `42` for training configs.

The single-layer FFN training and inference latency benchmarks use fixed-shape hidden-state tensors. This keeps the timing focused on the FFN kernels rather than tokenizer, embedding, attention, or dataset loading.

## Model Presets

The `60m` preset uses the `d=512`, `d_ffn=1376`, `8`-layer configuration. The other public presets are `130m`, `350m`, and `1b`, all defined in `moc/config.py`.

The C4 run configs in `configs/` cover all four presets:

| Config | Preset | Total batch | Micro batch | Sequence length | Steps | Approx. allocated tokens | Learning rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama_60m_c4.yaml` | `60m` | 512 | 256 | 256 | 11,000 | 1.44B | 2.5e-3 |
| `llama_130m_c4.yaml` | `130m` | 512 | 256 | 256 | 22,000 | 2.88B | 2.5e-3 |
| `llama_350m_c4.yaml` | `350m` | 512 | 128 | 256 | 59,510 | 7.80B | 1.0e-3 |
| `llama_1b_c4.yaml` | `1b` | 512 | 64 | 256 | 99,945 | 13.10B | 6.0e-4 |

The `1b` executable preset is `24` decoder layers and `32` attention heads.

## Timing

Single-layer training latency is measured with CUDA events. Forward timing includes autograd graph construction; backward timing is measured after an untimed forward pass.

Training-throughput numbers prefetch C4 batches to CPU before timing, then measure the GPU training step. This keeps the reported number focused on model computation.

Inference latency is measured with CUDA events after warmup.

## PPL

PPL runs train the selected preset on C4 with the schedule in `configs/`, then evaluate validation perplexity with the batch-mean validation protocol. The JSON output also includes token-weighted PPL. The training entry point is:

```bash
CONFIG=configs/llama_60m_c4.yaml FFN_TYPE=moc bash scripts/run_ppl.sh
```

Use `STOP_AT_STEP` and `EVAL_MAX_BATCHES` for short smoke runs.

## Memory Metric

Training-memory benchmarks run one warmup AdamW step to initialize optimizer state, then reset CUDA peak-memory stats and report `torch.cuda.max_memory_allocated()` for one measured `forward + backward + AdamW step`. This is different from `nvidia-smi` process memory and from `torch.cuda.max_memory_reserved()`.
