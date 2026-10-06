"""Fixed-K Llama commonsense fine-tuning and canonical-answer evaluation."""
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, GenerationConfig

from common import cuda_device, load_config, parser, read_json, source_hashes, write_json
from data import evaluation_rows, loaders
from evaluation import evaluate_task, evaluate_validation_loss
from loss import backward_batch
from model import build_model, inference_mode, load_trained_adapter, save_adapter
from overlay import verify_overlay_weights
from protocol import TASKS


class WallBudget:
    """Training deadline only; final save/evaluation time must be added separately."""
    def __init__(self, seconds, warmup_steps, clock=time.monotonic):
        if seconds <= 0 or warmup_steps < 1:
            raise ValueError("Invalid wall-clock budget")
        self.clock, self.start = clock, clock()
        self.deadline = self.start + seconds
        self.warmup_steps, self.warmup_end = warmup_steps, None

    def exhausted(self):
        return self.clock() >= self.deadline

    def factor(self, step):
        now = self.clock()
        if now >= self.deadline:
            return 0.
        if step < self.warmup_steps:
            return step / self.warmup_steps
        if self.warmup_end is None:
            self.warmup_end = now
        return max(0., (self.deadline - now) / (self.deadline - self.warmup_end))


def arguments(argv=None):
    p = parser(__doc__)
    p.add_argument("--method", choices=("dense", "moc"), required=True)
    p.add_argument("--overlay", type=Path)
    p.add_argument("--checkpoint-policy", choices=("block", "attention", "none"))
    p.add_argument("--micro-batch-size", type=int, choices=(4, 8, 16))
    p.add_argument("--stop-at-step", type=int, help="Smoke limit; does not shorten the LR horizon")
    p.add_argument("--eval-limit", type=int, help="Smoke-only rows per task")
    p.add_argument("--evaluate-adapter", type=Path, help="Evaluate a saved model, not optimizer resume")
    p.add_argument("--wall-budget-seconds", type=float, help="Dense equal-time control; includes setup, excludes final evaluation")
    args = p.parse_args(argv)
    if args.method == "dense" and args.overlay:
        p.error("Dense does not use a sparse overlay")
    if args.method == "moc" and not args.overlay:
        p.error("MoC requires --overlay from the reconstruction stage")
    if args.eval_limit is not None and args.eval_limit <= 0:
        p.error("--eval-limit must be positive")
    if args.wall_budget_seconds is not None and (args.method != "dense" or args.evaluate_adapter or args.stop_at_step):
        p.error("Wall-budget mode is a separate Dense training control")
    if args.evaluate_adapter and args.stop_at_step:
        p.error("Evaluation-only mode has no stop-at-step")
    args.checkpoint_policy = args.checkpoint_policy or ("attention" if args.method == "moc" else "block")
    args.micro_batch_size = args.micro_batch_size or (8 if args.method == "moc" else 16)
    return args


def main():
    started = time.monotonic()
    args = arguments()
    cfg = load_config(args.config)
    training = cfg["training"]
    horizon = training["steps"]
    stop = args.stop_at_step or horizon
    if args.stop_at_step is not None and not 0 < args.stop_at_step <= horizon:
        raise ValueError("Smoke step must be within the configured LR horizon")
    budget = WallBudget(args.wall_budget_seconds, training["warmup_steps"]) if args.wall_budget_seconds else None
    if args.wall_budget_seconds is not None and args.wall_budget_seconds <= 0:
        raise ValueError("Wall budget must be positive")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "predictions").mkdir()
    device = cuda_device(cfg["seed"])
    torch.set_num_threads(4)
    train_loader, val_loader, cache = loaders(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_dir"], local_files_only=True)
    tokenizer.pad_token_id, tokenizer.padding_side, tokenizer.truncation_side = 0, "left", "right"
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(cfg, args.method, device, overlay=args.overlay, policy=args.checkpoint_policy)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if sum(p.numel() for _, p in named) != 301989888:
        raise RuntimeError("Unexpected trainable LoRA parameter count")
    manifest = read_json(args.overlay / "manifest.json") if args.overlay else None
    write_json(out / "run_config.json", dict(config=cfg, method=args.method,
        overlay=str(args.overlay.resolve()) if args.overlay else None,
        activation_storage="selected" if args.method == "moc" else "dense", checkpoint_policy=args.checkpoint_policy,
        micro_batch_size=args.micro_batch_size, logical_batch_size=16,
        lr_schedule="wall_time_after_step_warmup" if budget else "linear_steps",
        schedule_total_steps=None if budget else horizon, wall_budget_seconds=args.wall_budget_seconds,
        stop_at_step=args.stop_at_step, evaluation_limit=args.eval_limit,
        evaluate_adapter=str(args.evaluate_adapter) if args.evaluate_adapter else None,
        source_files=source_hashes(), token_cache=cache,
        gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__,
        frozen_parameter_dtype="bfloat16", trainable_parameter_dtype="float32",
        checkpoint_selection="fixed_final_state", checkpoints_are_strict_resume=False))

    def record(value):
        value = dict(value, elapsed_seconds=time.monotonic() - started)
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, allow_nan=False) + "\n")
        print(json.dumps(value), flush=True)

    step, consumed, order_hash = 0, 0, hashlib.sha256()
    final_val = None
    if args.evaluate_adapter:
        meta = load_trained_adapter(model, args.evaluate_adapter, cfg, args.method, args.overlay)
        step = meta["step"]
    else:
        params = [p for _, p in named]
        optimizer = torch.optim.AdamW(params, lr=training["learning_rate"], betas=(.9, .999), eps=1e-8, weight_decay=0.)
        def linear(current):
            warmup = training["warmup_steps"]
            return current / max(1, warmup) if current < warmup else max(0., (horizon - current) / max(1, horizon - warmup))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, budget.factor if budget else linear)
        initial = evaluate_validation_loss(model, val_loader, device)
        record(dict(event="initial_validation", step=0, val_loss=initial))
        if budget and budget.exhausted():
            raise RuntimeError("Budget exhausted by setup; no training result")
        rolling, count, epoch, finished = 0., 0, 0, False
        while not finished:
            epoch += 1
            model.train()
            for batch in train_loader:
                order_hash.update(batch["source_indices"].numpy().tobytes())
                consumed += len(batch["input_ids"])
                optimizer.zero_grad(set_to_none=True)
                loss = backward_batch(model, batch, device, args.micro_batch_size, diagnostics=args.method == "dense")
                gate = torch.stack([p.grad.float().norm().square() for n, p in named
                                    if ".gate_proj." in n and p.grad is not None]).sum().sqrt()
                if step == 0 and (not torch.isfinite(gate) or gate <= 0):
                    raise RuntimeError("Gate LoRA has no finite nonzero gradient")
                norm = torch.nn.utils.clip_grad_norm_(params, 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite gradient norm")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                rolling, count = rolling + loss, count + 1
                if step == 1 or step % training["logging_every"] == 0:
                    record(dict(event="train", epoch=epoch, step=step, loss=rolling / count,
                        grad_norm=float(norm), gate_grad_norm=float(gate), lr=scheduler.get_last_lr()[0],
                        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30))
                    rolling, count = 0., 0
                if step % training["save_every"] == 0:
                    save_adapter(model, out / f"adapter_step_{step}", cfg, args.method, args.overlay, step)
                if step % training["validation_every"] == 0:
                    record(dict(event="validation", step=step, val_loss=evaluate_validation_loss(model, val_loader, device)))
                finished = budget.exhausted() if budget else step >= stop
                if finished:
                    break
        final_val = evaluate_validation_loss(model, val_loader, device)
        save_adapter(model, out / "adapter", cfg, args.method, args.overlay, step)
        # Evaluation retains the live optimizer, matching the measured process-peak scope.
    if manifest:
        verify_overlay_weights(model.base_model.model.model.layers, manifest)
    inference_mode(model)
    generation = GenerationConfig(temperature=.1, top_p=.75, top_k=40,
        num_beams=cfg["evaluation"]["num_beams"], do_sample=False,
        max_new_tokens=cfg["evaluation"]["max_new_tokens"], pad_token_id=0, eos_token_id=tokenizer.eos_token_id)
    record(dict(event="evaluation_started", step=step, data_order_sha256=order_hash.hexdigest()))
    results = {}
    for task in TASKS:
        results[task] = evaluate_task(model, tokenizer, task, evaluation_rows(cfg["data_dir"], task, args.eval_limit),
                                     out / "predictions" / f"{task}.jsonl", device, generation)
        record(dict(event="task_evaluated", result=results[task]))
    result = dict(method=args.method, step=step, seed=cfg["seed"], average_accuracy=sum(r["accuracy"] for r in results.values()) / len(TASKS),
        tasks=results, checkpoint_selection="fixed_final_state", validation_loss=final_val,
        smoke=bool(args.eval_limit or args.stop_at_step), decoding="canonical_legal_answer_prefix_constraint")
    write_json(out / "final_eval.json", result)
    write_json(out / "summary.json", dict(**result, complete=True, consumed_train_examples=consumed,
        data_order_sha256=order_hash.hexdigest() if not args.evaluate_adapter else None,
        entry_elapsed_seconds=time.monotonic() - started,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
        peak_scope="model loading through final evaluation; allocator allocated bytes, not FFN activation bytes"))
    (out / "DONE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
