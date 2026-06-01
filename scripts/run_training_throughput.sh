#!/usr/bin/env bash
set -euo pipefail

python benchmarks/training/benchmark_training_throughput.py \
  --preset 1b \
  --methods dense,moc \
  --out results/training_throughput.json
