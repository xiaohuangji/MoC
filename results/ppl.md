# C4 Validation PPL

## Scope

This benchmark trains the LLaMA-style presets on local C4 and reports validation perplexity. The data pipeline uses local `t5-base` tokenizer files under `data/tokenizer/`, per-document truncation/padding to sequence length `256`, shifted causal-LM loss, and `-100` labels for padding tokens.

The reported PPL is the `val_ppl` field produced by the training script.

## Training Protocol

Dense and MoC use BF16 mixed precision for PPL training. Trainable parameters
and AdamW optimizer states stay in FP32. MoC uses BF16 matrix multiplications
in the backward pass and accumulates weight gradients in FP32, matching the
gradient precision of the Dense baseline's `nn.Linear` under autocast.

The C4 schedule keeps `total_batch_size=512`, `seq_len=256`, seed `0`, deterministic data shuffling with seed `42`, and the configured cosine scheduler horizon for each preset. The reported evaluation targets approximately 10M non-padding validation tokens.

The completed rows were each trained on a single A100-PCIE-40GB GPU; the micro batch size per row is given in the Notes column (gradient accumulation keeps `total_batch_size=512`).

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

## Results

| Preset | Training Tokens | Steps | Dense PPL | MoC PPL | MoC 2:8 PPL | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 60M | 1.31B | 10,000 | 28.71 | 29.11 | 29.28 | BF16 mixed precision (FP32 master weights), train micro batch 128, eval micro batch 256 |
| 130M | 2.62B | 20,000 | 21.47 | 21.76 | 21.85 | BF16 mixed precision (FP32 master weights), train micro batch 128, eval micro batch 256 |
| 350M | 7.86B | 60,000 | 16.02 | 16.18 | 16.29 | BF16 mixed precision (FP32 master weights), train micro batch 64, eval micro batch 64 |
| 1B | 13.11B | 100,000 | 13.00 | 13.16 | 13.21 | BF16 mixed precision (FP32 master weights), train micro batch 16, eval micro batch 64 |

The MoC row uses the same public `ffn_type="moc"` training path as the memory and training-throughput benchmarks.
Completed rows evaluate approximately 10M non-padding validation tokens.
