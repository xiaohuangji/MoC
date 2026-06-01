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
| Dense | 4.310 ms/token | 232.0 tok/s | 1.000x |
| MoC | 4.003 ms/token | 249.8 tok/s | 1.077x |
| MoC 2:8 | 3.586 ms/token | 278.9 tok/s | 1.202x |

The single-layer FFN benchmark shows the clearest MoC speedup. In end-to-end decode, global MoC stays close to Dense, while MoC 2:8 gives a stronger system-level speedup.
