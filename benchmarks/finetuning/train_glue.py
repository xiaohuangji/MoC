"""Train one fresh supervised LoRA and task head on Dense or a shared K25 overlay."""
import argparse
from pathlib import Path

TASK_NAMES = ("cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--task", choices=TASK_NAMES, required=True)
    p.add_argument("--method", choices=("dense", "k25"), required=True)
    p.add_argument("--overlay", type=Path)
    p.add_argument("--output", type=Path, required=True, help="New directory for this single run")
    p.add_argument("--glue-dir", type=Path, help="Optional local copy of the pinned GLUE parquet tree")
    p.add_argument("--lr", required=True, choices=("2e-5", "5e-5", "1e-4", "3e-4"))
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if (a.method == "k25") != (a.overlay is not None):
        p.error("K25 requires --overlay; Dense must not use one")

    import torch.distributed as dist
    from transformers import AutoTokenizer, TrainingArguments
    from checkpoint import adapter_identity, frozen_identity, probe
    from distributed import four_gpus
    from glue_data import TASKS, collate, load_split
    from glue_model import build
    from glue_trainer import GlueTrainer
    from io_utils import CONFIG, sha, write
    from metrics import SCORERS

    task, out = a.task, a.output
    count, _, epochs, _, _, metric = TASKS[task]
    with four_gpus(a.base) as (rank, device):
        if rank == 0:
            out.mkdir(exist_ok=False, parents=True)
        dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(a.base, local_files_only=True)
        train = load_split(task, "train", tokenizer, a.glue_dir)
        dev = load_split(task, "validation", tokenizer, a.glue_dir)
        model = build(a.base, task, device, overlay=a.overlay, seed=a.seed)
        frozen = frozen_identity(model)
        args = TrainingArguments(output_dir=str(out), num_train_epochs=epochs,
            per_device_train_batch_size=8, per_device_eval_batch_size=8, gradient_accumulation_steps=1,
            learning_rate=float(a.lr), lr_scheduler_type="linear", warmup_ratio=0.06, weight_decay=0.01,
            max_grad_norm=1.0, optim="adamw_torch", bf16=True, gradient_checkpointing=False,
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
            metric_for_best_model="eval_" + metric, greater_is_better=True, logging_steps=10,
            logging_nan_inf_filter=False, report_to=[], seed=a.seed, data_seed=a.seed,
            remove_unused_columns=False, label_names=["labels"], ddp_find_unused_parameters=False,
            ddp_timeout=3600, dataloader_num_workers=0)

        def compute(pred):
            values = pred.predictions.reshape(-1) if task == "stsb" else pred.predictions.argmax(-1)
            return SCORERS[task](pred.label_ids.tolist(), values.tolist())

        trainer = GlueTrainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
                              data_collator=collate, compute_metrics=compute)
        trainer.train(resume_from_checkpoint=None)
        assert trainer.state.global_step == ((count + 31) // 32) * epochs
        curve = [e for e in trainer.state.log_history if "eval_loss" in e]
        assert len(curve) == epochs
        best = min(curve, key=lambda e: (-e["eval_" + metric], e["epoch"]))
        assert Path(trainer.state.best_model_checkpoint).name == f"checkpoint-{best['step']}"
        if frozen_identity(model) != frozen:
            raise RuntimeError("Frozen base parameters changed during supervised fine-tuning")
        actual = probe(model, dev, device)
        if rank == 0:
            selected = out / "selected"
            model.save_pretrained(selected, safe_serialization=True, save_embedding_layers=False)
            write(out / "selected.json", dict(task=task, method=a.method, lr=a.lr, seed=a.seed,
                  gcp=False, selected_step=best["step"], selected_epoch=best["epoch"],
                  trainer_metric=best["eval_" + metric], sources_sha256=sha(CONFIG),
                  overlay_sha256=sha(a.overlay / "OVERLAY.json") if a.overlay else None,
                  adapter_sha256=sha(selected / "adapter_model.safetensors"),
                  tensors=adapter_identity(model), probe=actual))
        dist.barrier()


if __name__ == "__main__":
    main()
