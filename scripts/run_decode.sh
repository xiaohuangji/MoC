#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

python benchmarks/inference/benchmark_decode.py \
  --device cuda \
  --mode full \
  --methods dense global_moc \
  --rounds 3 \
  --warmup-runs 8 \
  --measure-runs 30 \
  --out results/decode.json
