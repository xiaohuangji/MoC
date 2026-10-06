# End-to-End Decode

## A800 PCIe Results

| Method | Latency (ms/token) | Throughput (token/s) | Speedup |
| --- | ---: | ---: | ---: |
| Dense | 3.299 | 303.09 | 1.000x |
| MoC | **2.965** | **337.28** | **1.113x** |

MoC increases end-to-end decode throughput by **11.28%**. Both methods use
the same TorchInductor C++ wrapper with dynamic compilation.

## Configuration

- NVIDIA A800 80GB PCIe; PyTorch 2.8.0+cu128; Triton 3.4.0; BF16.
- 24 layers, hidden size 2048, FFN width 5464, 16 attention heads, vocabulary 32000.
- Batch size 1; 128 prompt tokens and 128 generated tokens; global MoC K=1024.
- Randomly initialized weights and a fixed C4 validation prompt; this is a latency benchmark, not a quality evaluation.
- `torch.compile(dynamic=True, options={"cpp_wrapper": True})` for both methods; no whole-model CUDA Graph replay.
- Three rounds per method, each with 8 warmup generations and 30 timed generations.
  Each round runs in a fresh process with fixed model initialization.
  Latency is the mean of the three per-round medians; throughput is its reciprocal.
  OMP and MKL use 4 threads.

The timed loop includes embedding, attention, KV updates, FFN, final normalization,
LM head, and argmax. Prefill, compilation, and warmup are excluded. CUDA events
measure the complete 128-token generation loop.

## Run

```bash
bash scripts/run_decode.sh
```

The output is `results/decode.json`, including per-round samples and aggregate
latency, throughput, and allocated-memory peaks. To include grouped MoC 2:8:

```bash
python benchmarks/inference/benchmark_decode.py \
  --methods dense global_moc moc_2_8 --out results/decode.json
```
