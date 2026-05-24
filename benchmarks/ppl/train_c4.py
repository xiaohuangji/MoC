"""C4 pre-training and validation PPL benchmark."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from moc.config import PRESETS, MoCConfig, count_parameters  # noqa: E402
from moc.model import build_model  # noqa: E402


def list_c4_shards(split: str) -> list[str]:
    if split not in ("train", "val"):
        raise ValueError(f"split must be train or val, got {split!r}")
    base = REPO_ROOT / "data" / "c4" / split
    paths = sorted(base.glob("*/*.json.gz")) + sorted(base.glob("*.json.gz"))
    out = sorted({str(path) for path in paths})
    if not out:
        raise FileNotFoundError(f"No C4 {split} shards found under {base}")
    return out


def load_local_tokenizer(seq_len: int):
    tokenizer_dir = REPO_ROOT / "data" / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), model_max_length=seq_len)
    if "T5" not in type(tokenizer).__name__:
        raise ValueError(
            "data/tokenizer must contain local t5-base tokenizer files "
            f"(got {type(tokenizer).__name__})."
        )
    if tokenizer.pad_token_id is None:
        raise ValueError(f"tokenizer under {tokenizer_dir} must define pad_token_id")
    return tokenizer


class C4DocumentDataset(IterableDataset):
    """Emit fixed-length C4 documents with padding labels masked to -100."""

    def __init__(
        self,
        split: str,
        tokenizer,
        seq_len: int,
        seed: int,
        shuffle_docs: bool,
        shuffle_buffer_docs: int,
        max_docs: int | None,
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.shuffle_docs = shuffle_docs
        self.shuffle_buffer_docs = shuffle_buffer_docs
        self.max_docs = max_docs

    def _iter_texts(self, worker_id: int, num_workers: int) -> Iterator[str]:
        rng = random.Random(self.seed + worker_id)
        shards = list_c4_shards(self.split)
        if self.shuffle_docs:
            rng.shuffle(shards)
        shards = shards[worker_id::num_workers]

        emitted = 0
        buffer: list[str] = []
        for shard in shards:
            with gzip.open(shard, "rt", encoding="utf-8") as handle:
                for line in handle:
                    text = json.loads(line).get("text", "")
                    if not text.strip():
                        continue
                    if self.shuffle_docs and self.shuffle_buffer_docs > 1:
                        buffer.append(text)
                        if len(buffer) < self.shuffle_buffer_docs:
                            continue
                        idx = rng.randrange(len(buffer))
                        text = buffer.pop(idx)
                    yield text
                    emitted += 1
                    if self.max_docs is not None and emitted >= self.max_docs:
                        return

        if self.shuffle_docs and buffer:
            rng.shuffle(buffer)
            for text in buffer:
                yield text
                emitted += 1
                if self.max_docs is not None and emitted >= self.max_docs:
                    return

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        for text in self._iter_texts(worker_id, num_workers):
            encoded = self.tokenizer(
                text,
                max_length=self.seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0).to(torch.long)
            attention_mask = encoded["attention_mask"].squeeze(0).to(torch.long)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            yield {"input_ids": input_ids, "labels": labels}


def build_loader(
    split: str,
    tokenizer,
    batch_size: int,
    seq_len: int,
    num_workers: int,
    cfg: dict,
) -> DataLoader:
    dataset = C4DocumentDataset(
        split=split,
        tokenizer=tokenizer,
        seq_len=seq_len,
        seed=int(cfg.get("shuffle_seed", 42)),
        shuffle_docs=bool(cfg.get("shuffle_docs", True)),
        shuffle_buffer_docs=int(cfg.get("shuffle_buffer_docs", 1000)),
        max_docs=cfg.get("max_docs"),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


def build_model_config(model_cfg: dict) -> tuple[str, MoCConfig]:
    if "preset" in model_cfg:
        preset = model_cfg["preset"]
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Available: {sorted(PRESETS)}")
        return model_cfg.get("name", preset), PRESETS[preset]

    required = ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads"]
    missing = [key for key in required if key not in model_cfg]
    if missing:
        raise ValueError(f"missing explicit model field(s): {missing}")
    kwargs = {
        "hidden_size": int(model_cfg["hidden_size"]),
        "intermediate_size": int(model_cfg["intermediate_size"]),
        "num_hidden_layers": int(model_cfg["num_hidden_layers"]),
        "num_attention_heads": int(model_cfg["num_attention_heads"]),
        "vocab_size": int(model_cfg.get("vocab_size", 32000)),
        "max_seq_len": int(model_cfg.get("max_seq_len", 256)),
        "rope_theta": float(model_cfg.get("rope_theta", 10000.0)),
        "rms_norm_eps": float(model_cfg.get("rms_norm_eps", 1e-6)),
        "tie_word_embeddings": bool(model_cfg.get("tie_word_embeddings", False)),
        "k": int(model_cfg["k"]) if "k" in model_cfg else None,
    }
    return model_cfg.get("name", "custom"), MoCConfig(**kwargs)


def cosine_warmup_schedule(total_steps: int, warmup_steps: int, min_lr_ratio: float):
    warmup_steps = max(1, warmup_steps)
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")

    def fn(step: int):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return fn


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate(model, loader, device, dtype, max_batches: int, target_nonpad_tokens: int) -> dict:
    model.eval()
    total_loss = 0.0
    total_loss_tokens = 0
    total_nonpad_tokens = 0
    batches = 0
    for batch in loader:
        if batches >= max_batches:
            break
        ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device.type == "cuda")):
            logits = model(ids)
            loss = causal_lm_loss(logits, labels)
        loss_tokens = int((labels[:, 1:] != -100).sum().item())
        nonpad_tokens = int((labels != -100).sum().item())
        total_loss += loss.item() * loss_tokens
        total_loss_tokens += loss_tokens
        total_nonpad_tokens += nonpad_tokens
        batches += 1
        if target_nonpad_tokens > 0 and total_nonpad_tokens >= target_nonpad_tokens:
            break
    model.train()
    mean_loss = total_loss / max(total_loss_tokens, 1)
    return {
        "val_loss": mean_loss,
        "val_ppl": math.exp(min(mean_loss, 20.0)),
        "eval_loss_tokens": total_loss_tokens,
        "eval_nonpad_tokens": total_nonpad_tokens,
        "eval_batches": batches,
    }


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            **payload,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, device) -> int:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if state.get("rng_state") is not None:
        torch.set_rng_state(state["rng_state"].cpu())
    if state.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"].cpu())
    return int(state["step"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ffn-type", required=True, choices=["dense", "moc", "moc_gcp"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stop-at-step", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--save-latest", action="store_true")
    parser.add_argument("--save-final", action="store_true")
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--eval-target-nonpad-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_name, model_config = build_model_config(cfg["model"])
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluation"]

    total_batch_size = int(train_cfg["total_batch_size"])
    micro_batch_size = int(train_cfg["micro_batch_size"])
    seq_len = int(train_cfg["seq_len"])
    if total_batch_size % micro_batch_size != 0:
        raise ValueError("total_batch_size must be divisible by micro_batch_size")
    grad_accum_steps = total_batch_size // micro_batch_size
    max_steps = args.max_steps or int(train_cfg["max_steps"])
    train_until_step = args.stop_at_step if args.stop_at_step is not None else max_steps
    if train_until_step <= 0 or train_until_step > max_steps:
        raise ValueError("--stop-at-step must be in the range [1, --max-steps]")
    warmup_steps = int(train_cfg.get("warmup_steps", max_steps * float(train_cfg.get("warmup_ratio", 0.1))))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))

    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tokenizer = load_local_tokenizer(seq_len)
    train_loader = build_loader(
        "train",
        tokenizer,
        batch_size=micro_batch_size,
        seq_len=seq_len,
        num_workers=args.num_workers,
        cfg=data_cfg,
    )
    val_cfg = {
        "shuffle_docs": eval_cfg.get("shuffle_docs", True),
        "shuffle_seed": eval_cfg.get("shuffle_seed", 42),
        "shuffle_buffer_docs": eval_cfg.get("shuffle_buffer_docs", 1000),
        "max_docs": eval_cfg.get("max_docs"),
    }
    val_loader = build_loader(
        "val",
        tokenizer,
        batch_size=int(eval_cfg.get("micro_batch_size", micro_batch_size)),
        seq_len=seq_len,
        num_workers=max(1, args.num_workers // 2),
        cfg=val_cfg,
    )

    model = build_model(model_config, ffn_type=args.ffn_type).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.999))),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        fused=False,
    )
    scheduler = LambdaLR(optimizer, cosine_warmup_schedule(max_steps, warmup_steps, min_lr_ratio))

    step = 0
    if args.resume_from is not None and args.resume_from.exists():
        step = load_checkpoint(args.resume_from, model, optimizer, scheduler, device)
    if step > train_until_step:
        raise ValueError(f"resume step {step} is greater than target step {train_until_step}")

    run_config = {
        "model_name": model_name,
        "model_config": asdict(model_config),
        "ffn_type": args.ffn_type,
        "parameters": count_parameters(model_config),
        "dtype": args.dtype,
        "training": train_cfg,
        "data": {
            **data_cfg,
            "train_shards": len(list_c4_shards("train")),
            "val_shards": len(list_c4_shards("val")),
            "tokenizer_path": str(REPO_ROOT / "data" / "tokenizer"),
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "tokenizer_pad_token_id": tokenizer.pad_token_id,
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
        },
        "evaluation": eval_cfg,
        "schedule_total_steps": max_steps,
        "train_until_step": train_until_step,
        "loss": "shifted causal LM loss; padding labels are -100",
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log_path = args.output_dir / "train.log"
    log_handle = open(log_path, "a", encoding="utf-8", buffering=1)

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        log_handle.write(line + "\n")

    checkpoint_payload = {
        "ffn_type": args.ffn_type,
        "model_name": model_name,
        "model_config": asdict(model_config),
        "run_config": run_config,
    }

    def handle_sigint(_signum, _frame):
        log("caught SIGINT; saving latest.pt before exit")
        save_checkpoint(args.output_dir / "latest.pt", model, optimizer, scheduler, step, checkpoint_payload)
        log_handle.close()
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    log(f"model={model_name} ffn_type={args.ffn_type} params={count_parameters(model_config)/1e6:.2f}M")
    log(f"device={device} dtype={args.dtype} total_batch={total_batch_size} micro_batch={micro_batch_size}")
    log(f"max_steps={max_steps} train_until_step={train_until_step} lr={train_cfg['learning_rate']}")

    train_iter = iter(train_loader)
    log_every = args.log_every or int(train_cfg.get("log_every", 50))
    metrics_path = args.output_dir / "metrics.jsonl"
    t0 = time.time()
    loss_accum = 0.0
    nonpad_accum = 0

    while step < train_until_step:
        optimizer.zero_grad(set_to_none=True)
        micro_loss = 0.0
        micro_nonpad = 0
        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device.type == "cuda")):
                logits = model(ids)
                loss = causal_lm_loss(logits, labels)
            (loss / grad_accum_steps).backward()
            micro_loss += loss.item() / grad_accum_steps
            micro_nonpad += int((labels != -100).sum().item())

        grad_clip = float(train_cfg.get("grad_clip", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1
        loss_accum += micro_loss
        nonpad_accum += micro_nonpad

        if step % log_every == 0:
            elapsed = time.time() - t0
            allocated_slots = total_batch_size * seq_len * log_every
            avg_loss = loss_accum / log_every
            row = {
                "event": "train",
                "step": step,
                "max_steps": max_steps,
                "train_until_step": train_until_step,
                "loss": avg_loss,
                "ppl": math.exp(min(avg_loss, 20.0)),
                "lr": scheduler.get_last_lr()[0],
                "allocated_tokens_per_second": allocated_slots / max(elapsed, 1e-6),
                "nonpad_tokens_per_second": nonpad_accum / max(elapsed, 1e-6),
                "nonpad_fraction": nonpad_accum / max(allocated_slots, 1),
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0,
                "time": time.time(),
            }
            append_jsonl(metrics_path, row)
            log(
                f"step {step}/{train_until_step} schedule_total={max_steps} "
                f"loss={avg_loss:.4f} ppl={row['ppl']:.2f} lr={row['lr']:.2e} "
                f"slot_tok/s={row['allocated_tokens_per_second']:.0f} mem={row['peak_memory_gib']:.2f}GiB"
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            loss_accum = 0.0
            nonpad_accum = 0

        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(args.output_dir / "latest.pt", model, optimizer, scheduler, step, checkpoint_payload)
            log(f"saved latest.pt @ step {step}")

    eval_max_batches = args.eval_max_batches or int(eval_cfg.get("max_batches", 200))
    eval_target_tokens = (
        args.eval_target_nonpad_tokens
        if args.eval_target_nonpad_tokens is not None
        else int(eval_cfg.get("target_nonpad_tokens", 0))
    )
    eval_result = evaluate(model, val_loader, device, dtype, eval_max_batches, eval_target_tokens)
    final_payload = {
        "run_config": run_config,
        "step": step,
        "schedule_total_steps": max_steps,
        "train_until_step": train_until_step,
        **eval_result,
    }
    (args.output_dir / "final_eval.json").write_text(
        json.dumps(final_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(
        f"final eval: loss={eval_result['val_loss']:.4f} ppl={eval_result['val_ppl']:.2f} "
        f"loss_tokens={eval_result['eval_loss_tokens']} nonpad_tokens={eval_result['eval_nonpad_tokens']}"
    )
    if args.save_latest:
        save_checkpoint(args.output_dir / "latest.pt", model, optimizer, scheduler, step, checkpoint_payload)
        log("saved latest.pt")
    if args.save_final:
        save_checkpoint(args.output_dir / "final.pt", model, optimizer, scheduler, step, checkpoint_payload)
        log("saved final.pt")
    log_handle.close()


if __name__ == "__main__":
    main()
