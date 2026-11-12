#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

for batch in 1 2 3 4; do
  python benchmarks/inference/benchmark_ffn_latency.py \
    --batch-size "${batch}" \
    --profile \
    --out "results/ffn_latency_b${batch}.json"
done
