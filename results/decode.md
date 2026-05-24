# End-to-End Decode

## Scope

This benchmark measures full single-token decode with random initialized weights and C4 prompt token IDs.

Command:

```bash
bash scripts/run_decode.sh
```

## A800 Results

| Method | Latency | Throughput | Speedup |
| --- | ---: | ---: | ---: |
| Dense | 4.096 ms/token | 244.1 tok/s | 1.000x |
| MoC | 4.024 ms/token | 248.5 tok/s | 1.018x |
| MoC 2:8 | 3.621 ms/token | 276.2 tok/s | 1.131x |

The single-layer FFN benchmark shows the clearest MoC speedup. In end-to-end decode, global MoC is slightly faster than Dense, while MoC 2:8 gives a stronger system-level speedup.
