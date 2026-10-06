# Benchmark Notes

## Environment

Unless a result page states otherwise, the benchmark runs use:

- NVIDIA A800-SXM4-80GB;
- CUDA 12.8;
- PyTorch 2.8.0;
- Triton 3.4.0;
- BF16.

Training throughput and end-to-end decode use NVIDIA A800 80GB PCIe.
PPL and 8B fine-tuning use A100-PCIE-40GB as specified on their result pages.

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
| `llama_60m_c4.yaml` | `60m` | 512 | 256 | 256 | 10,000 | 1.31B | 2.5e-3 |
| `llama_130m_c4.yaml` | `130m` | 512 | 256 | 256 | 20,000 | 2.62B | 2.5e-3 |
| `llama_350m_c4.yaml` | `350m` | 512 | 128 | 256 | 60,000 | 7.86B | 1.0e-3 |
| `llama_1b_c4.yaml` | `1b` | 512 | 16 | 256 | 100,000 | 13.11B | 6.0e-4 |

The `1b` executable preset is `24` decoder layers and `32` attention heads.

## Timing

Single-layer training latency is measured with CUDA events. Forward timing includes autograd graph construction; backward timing is measured after an untimed forward pass.

Training-throughput runs prefetch C4 batches to CPU before timing. The timed loop
includes host-to-device transfer, forward, backward, and the AdamW update, but
excludes dataset loading and tokenization. Tokens/s is the total number of
sequence positions, including padding, divided by elapsed wall time after CUDA
synchronization.

The reported A800 80GB PCIe results use FP32 parameters and optimizer states with
BF16 autocast, sequence length 256, batch128 for 350M and batch64 for 1B. Each
method and preset uses 10 warmup steps followed by one 100-step timed run. MoC
retains 95.26% and 98.10% of Dense throughput, respectively. See
[training speed results](../results/training_speed.md).

Inference latency is measured with CUDA events after warmup.

### End-to-End Decode

The decode benchmark uses an A800 80GB PCIe, BF16, 24 layers, hidden size 2048,
FFN width 5464, and 16 attention heads. Global MoC selects K=1024 channels.
Dense and MoC both use `torch.compile(dynamic=True, options={"cpp_wrapper": True})`.
The CUDA/C++ extension and TorchInductor require a CUDA toolkit and C++ compiler.

The default run measures three rounds, each with 8 warmup and 30 timed
128-token generations, batch size 1, and a fixed 128-token C4 prompt. Latency
is measured in an independent process for each round, with fixed model initialization. Reported latency
is the mean of the per-round medians; throughput is the reciprocal of that
latency. Prefill and compilation are not timed. Random weights make this a
performance measurement rather than a quality evaluation. See [decode results](../results/decode.md).

## PPL

PPL runs train the selected preset on C4 with the schedule in `configs/`, then report validation perplexity as `val_ppl`. The PPL entry point uses standard BF16 mixed precision: FP32 master weights and optimizer states with BF16 autocast compute; it does not expose FP16 or pure-BF16-parameter training modes. The training entry point is:

```bash
CONFIG=configs/llama_60m_c4.yaml FFN_TYPE=moc bash scripts/run_ppl.sh
```

Use `STOP_AT_STEP` and `EVAL_MAX_BATCHES` for short smoke runs.

## Memory Metric

The separate [8B fine-tuning benchmark](finetuning.md) uses Commonsense170K, a
Llama tokenizer, and A100-PCIE-40GB measurements. Its result page reports **GiB**,
not the decimal GB used by the C4 training-memory table. Fine-tuning has separate
steady-state training, full train/evaluation-process, and reconstruction windows;
these peaks must not be interchanged or called saved-activation memory.

Training-memory benchmarks run one warmup AdamW step to initialize optimizer state, then reset CUDA peak-memory stats and report `torch.cuda.max_memory_allocated()` for one measured `forward + backward + AdamW step`. This is different from `nvidia-smi` process memory and from `torch.cuda.max_memory_reserved()`.

The training-throughput table separately reports allocated-memory peaks in GiB
over its 100 measured steps, after resetting peak statistics following 10 warmup
steps. These are total training-memory peaks, not saved FFN activations, and must
not be substituted for the separate single-step memory benchmark's GB values.
