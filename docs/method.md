# Method Notes

MoC applies channel selection inside the FFN block of a LLaMA-style Transformer.

For an input hidden state `X`, the dense SwiGLU FFN computes:

```text
G = X W_gate
U = X W_up
S = SiLU(G)
Z = S * U
D = Z W_down
```

MoC uses the native SwiGLU gate branch as a token-wise channel-importance signal:

```text
M  = TopK(G)
S' = SiLU(G) * M
Z' = S' * U
D  = Z' W_down
```

This keeps the model architecture close to the dense FFN while reducing activation memory in training and enabling sparse FFN inference kernels.

## Activation Accounting

| Method | Saved FFN activations | Recomputed in backward | FFN activation cost |
| --- | --- | --- | --- |
| Dense FFN | `G, U, S, Z, D` | none | `4bs d_ffn + bsd` |
| Dense FFN + GCP | `G, U, D` | `S, Z` | `2bs d_ffn + bsd` |
| MoC | `G * M, U * M, S * M, Z * M, M, D` | none | `bs d_ffn + bsd` |
| MoC + GCP | `G * M, U * M, M, D` | `S * M, Z * M` | `0.6bs d_ffn + bsd` |

The MoC costs use the top-20% channel setting. When `d_ffn = 8d/3`, the Dense FFN cost is `11.67bsd`, MoC is `3.67bsd`, and MoC+GCP is `2.6bsd`.

The implementation provides:

- a standard dense FFN baseline;
- memory-efficient MoC training;
- selected-activation checkpointing for MoC;
- optimized MoC inference paths for global Top-K and MoC 2:8.
