#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/llama_60m_c4.yaml}"
FFN_TYPE="${FFN_TYPE:-moc}"
OUTPUT_DIR="${OUTPUT_DIR:-data/checkpoints/ppl_$(basename "$CONFIG" .yaml)_${FFN_TYPE}}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
SAVE_LATEST="${SAVE_LATEST:-1}"
SAVE_FINAL="${SAVE_FINAL:-0}"

ARGS=(
  benchmarks/ppl/train_c4.py
  --config "$CONFIG"
  --ffn-type "$FFN_TYPE"
  --output-dir "$OUTPUT_DIR"
  --num-workers "$NUM_WORKERS"
  --save-every "$SAVE_EVERY"
)

if [[ -n "${MAX_STEPS:-}" ]]; then
  ARGS+=(--max-steps "$MAX_STEPS")
fi

if [[ -n "${STOP_AT_STEP:-}" ]]; then
  ARGS+=(--stop-at-step "$STOP_AT_STEP")
fi

if [[ -n "${RESUME_FROM:-}" ]]; then
  ARGS+=(--resume-from "$RESUME_FROM")
fi

if [[ -n "${LOG_EVERY:-}" ]]; then
  ARGS+=(--log-every "$LOG_EVERY")
fi

if [[ -n "${EVAL_MAX_BATCHES:-}" ]]; then
  ARGS+=(--eval-max-batches "$EVAL_MAX_BATCHES")
fi

if [[ -n "${EVAL_TARGET_NONPAD_TOKENS:-}" ]]; then
  ARGS+=(--eval-target-nonpad-tokens "$EVAL_TARGET_NONPAD_TOKENS")
fi

if [[ "$SAVE_LATEST" == "1" ]]; then
  ARGS+=(--save-latest)
fi

if [[ "$SAVE_FINAL" == "1" ]]; then
  ARGS+=(--save-final)
fi

python "${ARGS[@]}"
