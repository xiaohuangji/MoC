#!/usr/bin/env bash
set -euo pipefail

python benchmarks/training/benchmark_training_latency.py \
  --out results/training_latency.json \
  "$@"
