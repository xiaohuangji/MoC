#!/usr/bin/env bash
set -euo pipefail

PRESETS_TO_RUN="${PRESETS_TO_RUN:-60m 130m 350m 1b}"
METHODS="${METHODS:-dense,moc,moc_gcp}"
OUT_DIR="${OUT_DIR:-results}"

mkdir -p "$OUT_DIR"

for preset in $PRESETS_TO_RUN; do
  python benchmarks/memory/benchmark_memory.py \
    --preset "$preset" \
    --methods "$METHODS" \
    --out "$OUT_DIR/memory_${preset}.json"
done
