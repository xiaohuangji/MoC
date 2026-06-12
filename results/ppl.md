# C4 Validation PPL

## Scope

This benchmark trains the LLaMA-style presets on local C4 and reports validation perplexity. The data pipeline uses local `t5-base` tokenizer files under `data/tokenizer/`, per-document truncation/padding to sequence length `256`, shifted causal-LM loss, and `-100` labels for padding tokens.

The reported PPL is the `val_ppl` field produced by the training script.

## Training Protocol

PPL runs use standard BF16 mixed precision: trainable parameters and AdamW optimizer states stay in FP32, and all forward/backward compute runs under BF16 autocast. The MoC backward keeps its GEMMs in the BF16 compute dtype and accumulates weight gradients in FP32, matching the gradient precision the dense baseline gets from `nn.Linear` under autocast. The public entry point does not expose FP16 or pure-BF16-parameter training modes.

The C4 schedule keeps `total_batch_size=512`, `seq_len=256`, seed `0`, deterministic data shuffling with seed `42`, and the configured cosine scheduler horizon for each preset. The reported evaluation targets approximately 10M non-padding validation tokens.

The completed 60M and 130M rows below used `training.micro_batch_size=128` on A100 40GB and `evaluation.micro_batch_size=256`.

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
| 350M | 7.8B | - | - | - | - | Not run |
| 1B | 13.1B | - | - | - | - | Not run |

The MoC row uses the same public `ffn_type="moc"` training path as the memory and training-throughput benchmarks.
Completed rows evaluate approximately 10M non-padding validation tokens.
