# Single-Layer FFN Training Latency

## Scope

This benchmark measures one SwiGLU FFN block in training mode. Forward timing includes autograd graph construction. Backward timing is measured after an untimed forward pass.

Command:

```bash
bash scripts/run_training_latency.sh
```

## A800 Results

Shape: `batch=64`, `seq_len=256`, `hidden=2048`, `d_ffn=5461`, `K=1024`, `bf16`.

| Method | Forward (ms) | Backward (ms) | Total (ms) | Standard / Method |
| --- | ---: | ---: | ---: | ---: |
| Standard FFN | 16.697 | 26.975 | 43.670 | 100.00% |
| MoC | 18.148 | 27.312 | 45.465 | 96.05% |

The MoC path uses the same selected-channel training implementation exposed by `ffn_type="moc"`.
