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

The FFN training and inference latency benchmarks use fixed-shape hidden-state tensors, excluding tokenizer, embedding, attention, and dataset loading. Both measure one FFN; inference uses the same compilation and CUDA Graph replay settings for Dense and MoC.

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

### FFN Decode Latency

The FFN benchmark reuses the decoder's existing Dense and global MoC paths with
`torch.compile(dynamic=True, options={"cpp_wrapper": True})` and explicit CUDA
Graph replay for both methods. It measures one FFN with a fixed GPU-resident
single-token input, sweeping batch sizes 1 through 4. Parameters and inputs
match across methods; Dense retains standard PyTorch projections.

Each graph contains 16 calls to the same FFN. Three rounds each use 30 warmup
replays and 80 groups of 7 replays. Elapsed time is divided by 112 FFN calls per
group, and all 240 groups contribute to the mean. Compilation,
capture, input copies, and correctness checks are outside the timed region.
OMP and MKL use 4 threads.

`--batch-size` selects one batch size. `--profile` records a separate GPU-kernel
breakdown over 20 graph replays, normalized by 320 FFN calls. To benchmark the
grouped 2:8 path, use `--methods moc_2_8`. See [FFN inference results](../results/inference_ffn.md)
for the latency table.

On the first input, the benchmark checks the compiled FFN output against eager
execution (`rtol=0.03`, `atol=0.005`). Before timing, CUDA Graph replay must
match the compiled output bitwise on each of four inputs. After timing, the
benchmark checks the replay output again against the compiled output for the
first input.

The separate CUDA unit test [`test_selected_projection_bits`](../benchmarks/inference/test_decode.py)
compares the optimized global MoC operator with the selection, Up, and Down
operators run in sequence. It checks Top-K indices, selected activations, and
projected outputs for exact agreement.

### End-to-End Decode

The decode benchmark uses an A800 80GB PCIe, BF16, 24 layers, hidden size 2048,
FFN width 5464, and 16 attention heads. Global MoC selects K=1024 channels.
Dense and MoC both use `torch.compile(dynamic=True, options={"cpp_wrapper": True})`.
The CUDA/C++ extension and TorchInductor require a CUDA toolkit and C++ compiler.

The default run measures three rounds, each with 8 warmup and 30 timed
128-token generations, batch size 1, and a fixed 128-token C4 prompt. Latency
is measured in an independent process for each round, with fixed model initialization. Reported latency
is the mean of the per-round medians; throughput is the reciprocal of that
latency. Prefill and compilation are not timed. See the
[decode results and configuration](../results/decode.md).

## PPL

PPL runs train the selected preset on C4 with the schedule in `configs/`, then
report validation perplexity as `val_ppl`. Dense and MoC use BF16 autocast
with FP32 trainable parameters and AdamW optimizer states. To run the benchmark:

```bash
CONFIG=configs/llama_60m_c4.yaml FFN_TYPE=moc bash scripts/run_ppl.sh
```

For one training update and one validation batch, use the [C4 smoke test](../README.md#c4-smoke-test).

## Memory Metric

Peak allocated GPU memory is the maximum number of bytes occupied by PyTorch
tensors during the measurement window. Both training benchmarks reset CUDA peak
statistics after warmup and read `torch.cuda.max_memory_allocated()`:

- [Training memory](../results/memory.md): one AdamW warmup step, then one full
  training step (forward, backward, and optimizer update). Values are in decimal GB.
- [Training throughput](../results/training_speed.md): 10 warmup steps, then
  100 timed training steps. Values for peak memory are in GiB.

The [GLUE fine-tuning results](../results/finetuning.md) report per-GPU peaks
on four A100 40 GB GPUs and separately derive the memory saved on FFN
activations retained for backward.
