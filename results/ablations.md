# Ablations

All ablation runs use the same C4 PPL protocol as [ppl.md](ppl.md): standard BF16 mixed precision (FP32 master weights, BF16 autocast compute), `total_batch_size=512`, `seq_len=256`, seed `0`, the paper token budget for each preset, and an evaluation target of approximately 10M non-padding tokens. Only the ablated factor changes between rows.

## Number of activated channels (K)

MoC keeps the Top-`K` channels of the gate branch per token. Larger `K` retains more channels (less sparsity) and lowers perplexity with diminishing returns. Measured on the 130M preset (`d_ffn=2048`, 20,000 steps / 2.62B tokens).

| K | K / d_ffn | MoC PPL |
| ---: | ---: | ---: |
| 256 | 12.5% | 22.03 |
| 384 | 18.75% | 21.76 |
| 512 | 25.0% | 21.74 |

`K=384` is the default for the 130M preset. The trend matches the monotonic improvement with diminishing returns reported in the paper.

The default 130M config fixes `K=0.5·d_model=384`; the `K=256` and `K=512` rows use the explicit-field configs [configs/llama_130m_c4_k256.yaml](../configs/llama_130m_c4_k256.yaml) and [configs/llama_130m_c4_k512.yaml](../configs/llama_130m_c4_k512.yaml):

```bash
CONFIG=configs/llama_130m_c4_k256.yaml FFN_TYPE=moc bash scripts/run_ppl.sh
CONFIG=configs/llama_130m_c4_k512.yaml FFN_TYPE=moc bash scripts/run_ppl.sh
```

## Top-K position (before vs after SiLU)

MoC selects channels by the gate **pre-activation** `G` (before SiLU). The alternative is to select after the activation, by the magnitude of the activated output `|SiLU(G)|`. Selecting before the activation yields better perplexity at both scales, consistent with the paper.

| Top-K position | 60M PPL | 130M PPL |
| --- | ---: | ---: |
| Before SiLU (`moc`, default) | 29.11 | 21.76 |
| After SiLU (`moc_post_silu_abs`) | 29.82 | 22.27 |

```bash
# before SiLU (main method)
CONFIG=configs/llama_130m_c4.yaml FFN_TYPE=moc               bash scripts/run_ppl.sh
# after SiLU (magnitude of the activated output)
CONFIG=configs/llama_130m_c4.yaml FFN_TYPE=moc_post_silu_abs bash scripts/run_ppl.sh
```

"After SiLU" selects channels by `|SiLU(G)|`, the magnitude of the activated output, following the paper's appendix definition.
