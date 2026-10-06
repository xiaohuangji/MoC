#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
STAGE="${STAGE:-train}"
CONFIG="${CONFIG:-configs/llama31_8b_commonsense.yaml}"
METHOD="${METHOD:-moc}"
if [[ "$STAGE" == reconstruct ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-data/checkpoints/commonsense_reconstruction}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-results/raw/finetuning/${STAGE}_${METHOD}}"
fi
args=(--config "$CONFIG")

case "$STAGE" in
  prepare)
    exec "$PYTHON" benchmarks/finetuning/prepare_data.py "${args[@]}" "$@"
    ;;
  reconstruct)
    exec "$PYTHON" benchmarks/finetuning/reconstruct.py "${args[@]}" --output-dir "$OUTPUT_DIR" "$@"
    ;;
  train|evaluate|resources)
    args+=(--output-dir "$OUTPUT_DIR" --method "$METHOD")
    if [[ "$METHOD" == moc ]]; then
      OVERLAY="${OVERLAY:-data/checkpoints/commonsense_reconstruction/overlay}"
    fi
    [[ -z "${OVERLAY:-}" ]] || args+=(--overlay "$OVERLAY")
    [[ -z "${CHECKPOINT_POLICY:-}" ]] || args+=(--checkpoint-policy "$CHECKPOINT_POLICY")
    [[ -z "${MICRO_BATCH_SIZE:-}" ]] || args+=(--micro-batch-size "$MICRO_BATCH_SIZE")
    if [[ "$STAGE" == evaluate ]]; then
      : "${ADAPTER:?Set ADAPTER to the final adapter directory}"
      args+=(--evaluate-adapter "$ADAPTER")
    fi
    if [[ "$STAGE" == resources ]]; then
      : "${CHECKPOINT_POLICY:?Set CHECKPOINT_POLICY for the resource measurement}"
      : "${MICRO_BATCH_SIZE:?Set MICRO_BATCH_SIZE for the resource measurement}"
      exec "$PYTHON" benchmarks/finetuning/benchmark_resources.py "${args[@]}" "$@"
    fi
    exec "$PYTHON" benchmarks/finetuning/train_commonsense.py "${args[@]}" "$@"
    ;;
  *)
    printf 'Unknown STAGE: %s\n' "$STAGE" >&2
    exit 2
    ;;
esac
