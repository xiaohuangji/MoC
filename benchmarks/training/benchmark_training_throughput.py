"""Training-throughput benchmark for Dense and MoC models."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from moc.config import PRESETS, count_parameters  # noqa: E402
from moc.data import build_dataloader  # noqa: E402
from moc.model import build_model  # noqa: E402


DEFAULT_BATCH = {"350m": 128, "1b": 64}
DEFAULT_LR = {"350m": 1.0e-3, "1b": 6.0e-4}


def cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def make_c4_batches(batch_size: int, seq_len: int, count: int):
    loader = build_dataloader(
        "train",
        batch_size=batch_size,
        seq_len=seq_len,
        shuffle=True,
        shuffle_seed=42,
    )
    batches = []
    for idx, batch in enumerate(loader):
        if idx >= count:
            break
        batches.append((batch["input_ids"].cpu(), batch["labels"].cpu()))
    if len(batches) < count:
        raise RuntimeError(f"C4 loader produced only {len(batches)} batches, expected {count}.")
    return batches


def run_method(args, method: str, batches):
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config = PRESETS[args.preset]
    ffn_type = {"dense": "dense", "moc": "moc", "moc_2_8": "moc_2_8"}[method]

    cleanup(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.param_dtype == "fp32":
        # Mixed precision: FP32 parameters and optimizer states, autocast compute.
        model = build_model(config, ffn_type=ffn_type).to(device=device)
    else:
        model = build_model(config, ffn_type=ffn_type).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr if args.lr is not None else DEFAULT_LR.get(args.preset, 1.0e-3),
        betas=(0.9, 0.999),
        weight_decay=0.0,
        fused=False,
    )

    autocast_enabled = args.param_dtype == "fp32" and device.type == "cuda"

    def one_step(ids_cpu, labels_cpu):
        ids = ids_cpu.to(device, non_blocking=True)
        labels = labels_cpu.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            _, loss = model(ids, labels=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for ids, labels in batches[: args.warmup_steps]:
        one_step(ids, labels)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    measured = batches[args.warmup_steps : args.warmup_steps + args.measure_steps]
    start = time.perf_counter()
    for ids, labels in measured:
        one_step(ids, labels)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    del model, optimizer
    cleanup(device)

    tokens = args.batch_size * args.seq_len * args.measure_steps
    return {
        "method": method,
        "preset": args.preset,
        "parameters": count_parameters(config),
        "dtype": args.dtype,
        "param_dtype": args.param_dtype,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "measure_steps": args.measure_steps,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
        "peak_memory_gb": peak / 1e9,
        "peak_memory_gib": peak / 1024**3,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="350m")
    parser.add_argument("--methods", default="dense,moc")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--param-dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="bf16: parameters cast to --dtype (no autocast); "
        "fp32: FP32 parameters/optimizer states with --dtype autocast compute.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--out", default="results/training_throughput.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.batch_size = args.batch_size or DEFAULT_BATCH.get(args.preset, 64)
    total_batches = args.warmup_steps + args.measure_steps

    batches = make_c4_batches(args.batch_size, args.seq_len, total_batches)

    rows = []
    for method in [item.strip() for item in args.methods.split(",") if item.strip()]:
        rows.append(run_method(args, method, batches))

    payload = {
        "benchmark": "training_throughput",
        "data": "c4",
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
