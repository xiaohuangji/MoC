#!/usr/bin/env bash
set -euo pipefail

python benchmarks/inference/benchmark_decode.py \
  --device cuda \
  --mode full \
  --execution-scope compiled \
  --measure-runs 30 \
  --out results/decode.json
