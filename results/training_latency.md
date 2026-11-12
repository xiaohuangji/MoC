# Single-Layer FFN Training Latency

## Scope

This benchmark measures one SwiGLU FFN block in training mode. Forward timing includes autograd graph construction. Backward timing is measured after an untimed forward pass.

Command:

```bash
bash scripts/run_training_latency.sh
```

## A800 PCIe Results

Shape: `batch=64`, `seq_len=256`, `hidden=2048`, `d_ffn=5461`, `K=1024`, `bf16`.

Hardware: NVIDIA A800 80GB PCIe; PyTorch 2.8.0, CUDA 12.8, Triton 3.4.0.
Parameters and computation use BF16. Each method runs once. Forward and backward
latencies are timed separately with CUDA events and averaged over 30 iterations
each after 10 warmup iterations.
Total is the sum of the unrounded forward and backward means; optimizer updates
and data loading are excluded.

| Method | Forward (ms) | Backward (ms) | Total (ms) | Standard / Method |
| --- | ---: | ---: | ---: | ---: |
| Standard FFN | 16.700 | 27.066 | 43.766 | 100.00% |
| MoC | 18.067 | 26.917 | 44.984 | 97.29% |

MoC retains **97.29%** of Standard FFN throughput for this forward-plus-backward
workload. The ratio is Standard total latency divided by method total latency,
computed before rounding.

The MoC path uses the same selected-channel training implementation exposed by `ffn_type="moc"`.
