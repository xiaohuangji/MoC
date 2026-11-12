"""Single-layer FFN training latency benchmark."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moc.ffn import build_ffn  # noqa: E402


HIDDEN = 2048
INTERMEDIATE = 5461
TOPK = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-layer FFN training latency")
    parser.add_argument("--out", default="results/training_latency.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p10_ms": ordered[int(0.10 * (len(ordered) - 1))],
        "p90_ms": ordered[int(0.90 * (len(ordered) - 1))],
    }


def clear_grads(module: torch.nn.Module, x: torch.Tensor) -> None:
    module.zero_grad(set_to_none=True)
    x.grad = None


def measure_forward(module: torch.nn.Module, x: torch.Tensor, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        clear_grads(module, x)
        y = module(x)
        del y
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        clear_grads(module, x)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y = module(x)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
        del y
    return summarize(samples)


def measure_backward(
    module: torch.nn.Module,
    x: torch.Tensor,
    grad_out: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict:
    for _ in range(warmup):
        clear_grads(module, x)
        y = module(x)
        y.backward(grad_out)
        del y
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        clear_grads(module, x)
        y = module(x)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y.backward(grad_out)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
        del y
    return summarize(samples)


def measure_method(args: argparse.Namespace, method: str, dtype: torch.dtype) -> dict:
    device = torch.device(args.device)
    ffn_type = {"Standard_FFN": "dense", "MoC": "moc"}[method]
    module = build_ffn(HIDDEN, INTERMEDIATE, ffn_type=ffn_type, k=TOPK)
    module = module.to(device=device, dtype=dtype).train()

    x = torch.randn(
        args.batch_size,
        args.seq_len,
        HIDDEN,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    grad_out = torch.randn_like(x)

    forward = measure_forward(module, x, args.warmup, args.measure_iters)
    backward = measure_backward(module, x, grad_out, args.warmup, args.measure_iters)
    total_ms = forward["mean_ms"] + backward["mean_ms"]

    del module, x, grad_out
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return {
        "method": method,
        "ffn_type": ffn_type,
        "forward_ms": forward,
        "backward_ms": backward,
        "total_mean_ms": total_ms,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    if device.type != "cuda":
        raise SystemExit("This benchmark uses CUDA events and requires a CUDA device.")

    if args.smoke:
        args.batch_size = min(args.batch_size, 2)
        args.seq_len = min(args.seq_len, 16)
        args.warmup = min(args.warmup, 1)
        args.measure_iters = min(args.measure_iters, 2)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    dtype = dtype_from_name(args.dtype)
    rows = {
        name: measure_method(args, name, dtype)
        for name in ("Standard_FFN", "MoC")
    }
    dense_total = rows["Standard_FFN"]["total_mean_ms"]
    for row in rows.values():
        row["standard_over_method_pct"] = 100.0 * dense_total / row["total_mean_ms"]

    payload = {
        "benchmark": "single_layer_ffn_training_latency",
        "shape": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "topk": TOPK,
            "dtype": str(dtype),
        },
        "rows": rows,
        "run_config": {
            "warmup": args.warmup,
            "measure_iters": args.measure_iters,
            "seed": args.seed,
        },
        "timing_method": {
            "timer": "torch.cuda.Event elapsed_time",
            "forward": "training-mode forward pass with autograd graph construction",
            "backward": "backward pass timed after an untimed forward pass",
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Single-layer FFN training latency")
    for name in ("Standard_FFN", "MoC"):
        row = rows[name]
        print(
            f"  {name:12s}: forward={row['forward_ms']['mean_ms']:.3f} ms, "
            f"backward={row['backward_ms']['mean_ms']:.3f} ms, "
            f"total={row['total_mean_ms']:.3f} ms, "
            f"Standard/Method={row['standard_over_method_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
