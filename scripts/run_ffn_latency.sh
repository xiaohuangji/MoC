#!/usr/bin/env bash
set -euo pipefail

python benchmarks/inference/benchmark_ffn_latency.py \
  --out results/ffn_latency.json
