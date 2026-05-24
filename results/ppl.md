# C4 Validation PPL

## Scope

This benchmark trains the LLaMA-style presets on local C4 and reports validation perplexity. The data pipeline uses local `t5-base` tokenizer files under `data/tokenizer/`, per-document truncation/padding to sequence length `256`, shifted causal-LM loss, and `-100` labels for padding tokens.

Command:

```bash
CONFIG=configs/llama_60m_c4.yaml FFN_TYPE=moc bash scripts/run_ppl.sh
```

For a short smoke run:

```bash
CONFIG=configs/llama_60m_c4.yaml \
FFN_TYPE=moc \
OUTPUT_DIR=/tmp/moc_ppl_smoke \
STOP_AT_STEP=1 \
LOG_EVERY=1 \
EVAL_MAX_BATCHES=1 \
EVAL_TARGET_NONPAD_TOKENS=1 \
SAVE_EVERY=0 \
SAVE_LATEST=0 \
bash scripts/run_ppl.sh
```

## A800 Results

| Preset | Training Tokens | Steps | Dense PPL | MoC PPL |
| --- | ---: | ---: | ---: | ---: |
| 60M | 1.3B | 9,919 | 31.42 | 31.63 |
| 130M | 2.6B | 19,837 | 29.44 | 27.32 |
| 350M | 7.8B | - | - | - |
| 1B | 13.1B | - | - | - |

The MoC row uses the same public `ffn_type="moc"` training path as the memory and training-throughput benchmarks.
The completed 60M and 130M runs evaluate approximately 10M non-padding validation tokens.
