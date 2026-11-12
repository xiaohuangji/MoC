"""Training-memory benchmark for Dense, MoC, and MoC+GCP."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from moc.config import PRESETS, count_parameters  # noqa: E402
from moc.data import build_dataloader  # noqa: E402
from moc.model import build_model  # noqa: E402


DEFAULT_BATCH = {"60m": 256, "130m": 256, "350m": 128, "1b": 64}
DEFAULT_LR = {"60m": 2.5e-3, "130m": 2.5e-3, "350m": 1.0e-3, "1b": 6.0e-4}
METHOD_TO_FFN = {
    "dense": "dense",
    "moc": "moc",
    "moc_gcp": "moc_gcp",
}


def cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def make_c4_batch(batch_size: int, seq_len: int) -> dict:
    loader = build_dataloader(
        "train",
        batch_size=batch_size,
        seq_len=seq_len,
        shuffle=True,
        shuffle_seed=42,
    )
    try:
        return next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("C4 loader did not produce a batch.") from exc


def run_one(args, method: str, batch: dict) -> dict:
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config = PRESETS[args.preset]
    ffn_type = METHOD_TO_FFN[method]

    cleanup(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = build_model(config, ffn_type=ffn_type).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DEFAULT_LR.get(args.preset, 1.0e-3),
        betas=(0.9, 0.999),
        weight_decay=0.0,
        fused=False,
    )

    ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)

    def one_step() -> None:
        _, loss = model(ids, labels=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(args.warmup_steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    one_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
    else:
        peak = 0
        reserved = 0

    del ids, labels, model, optimizer
    cleanup(device)

    return {
        "method": method,
        "preset": args.preset,
        "parameters": count_parameters(config),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "peak_memory_gb": peak / 1e9,
        "peak_memory_gib": peak / 1024**3,
        "reserved_memory_gb": reserved / 1e9,
        "reserved_memory_gib": reserved / 1024**3,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="1b")
    parser.add_argument("--methods", default="dense,moc,moc_gcp")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--out", default="results/memory.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.batch_size = args.batch_size or DEFAULT_BATCH[args.preset]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    batch = make_c4_batch(args.batch_size, args.seq_len)
    payload = {
        "benchmark": "training_memory",
        "data": "c4",
        "rows": [run_one(args, method, batch) for method in methods],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
