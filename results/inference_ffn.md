# Single-Layer FFN Inference Latency

## Scope

This benchmark measures one FFN layer with `hidden=2048`, `intermediate=5464`, global `K=1024`, and BF16 input on A800.

Command:

```bash
bash scripts/run_ffn_latency.sh
```

## A800 Results

| Method | Total | Speedup |
| --- | ---: | ---: |
| Dense FFN | 86.605 us | 1.000x |
| MoC | 62.322 us | 1.390x |
| MoC 2:8 | 55.868 us | 1.550x |

The two MoC rows are measured separately: `MoC` uses global Top-K channel selection, while `MoC 2:8` uses grouped top-2-of-8 channel selection.
