#!/usr/bin/env bash
set -euo pipefail

PRESET="${PRESET:-1b}"
METHODS="${METHODS:-dense,moc}"
PARAM_DTYPE="${PARAM_DTYPE:-fp32}"
MEASURE_STEPS="${MEASURE_STEPS:-1000}"
OUT="${OUT:-results/training_throughput.json}"

python benchmarks/training/benchmark_training_throughput.py \
  --preset "$PRESET" \
  --methods "$METHODS" \
  --param-dtype "$PARAM_DTYPE" \
  --measure-steps "$MEASURE_STEPS" \
  --out "$OUT"
