"""One fixed configuration on one GPU; six warmups and three 16-step windows."""
import hashlib
import json
import statistics
import time

import torch
import torch.nn.functional as F

from common import cuda_device, load_config, parser, source_hashes, write_json
from data import loaders
from loss import backward_batch, weighted_causal_loss
from model import build_model
from selected_lora import install, uninstall


def parity(model, named, batch, selected):
    """Compare the full CE/reference graph to chunked CE and optional selected storage."""
    blocks = model.base_model.model.model.layers
    if selected:
        for block in blocks:
            uninstall(block.mlp)
    def run(chunked):
        torch.manual_seed(98128)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            if chunked:
                loss, _ = weighted_causal_loss(output.logits, batch["labels"], batch["response_mask"], 16)
            else:
                targets = batch["labels"][:, 1:].contiguous()
                weights = targets.ne(-100).float() * (1 + 15 * batch["response_mask"][:, 1:].float())
                values = output.logits[:, :-1].float().contiguous()
                losses = F.cross_entropy(values.view(-1, values.shape[-1]), targets.view(-1), reduction="none", ignore_index=-100).view_as(weights)
                loss = (losses * weights).sum() / weights.sum()
        rng = torch.cuda.get_rng_state().clone()
        loss.backward()
        if not torch.equal(rng, torch.cuda.get_rng_state()):
            raise RuntimeError("Backward changed the forward RNG stream")
        return float(loss.detach()), {n: p.grad.detach().cpu().clone() for n, p in named}
    value, reference = run(False)
    repeat, gradients = run(False)
    if repeat != value or any(not torch.equal(reference[n], gradients[n]) for n in reference):
        raise RuntimeError("Reference graph is not repeatable")
    del gradients
    if selected:
        for block in blocks:
            install(block.mlp)
    changed, gradients = run(True)
    errors = {}
    for name in reference:
        old, new = reference[name].float(), gradients[name].float()
        norm = float(old.norm())
        errors[name] = dict(relative_l2=float((old - new).norm()) / max(norm, 1e-12),
                           cosine=float(F.cosine_similarity(old.flatten(), new.flatten(), dim=0)) if norm else 1.)
    result = dict(reference_loss=value, optimized_loss=changed, gradients=errors,
                  passed=value == changed and all(e["relative_l2"] <= .01 and e["cosine"] >= .9999 for e in errors.values()))
    model.zero_grad(set_to_none=True)
    if not result["passed"]:
        raise RuntimeError(f"Loss/gradient parity failed: {result}")
    return result


def main(args):
    started = time.monotonic()
    cfg = load_config(args.config)
    device = cuda_device(cfg["seed"])
    torch.set_num_threads(4)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    loader, _, _ = loaders(cfg, pin_memory=False)
    iterator = iter(loader)
    batches = [next(iterator) for _ in range(54)]
    digest = hashlib.sha256()
    for batch in batches:
        for key in ("source_indices", "input_ids", "attention_mask", "labels", "response_mask"):
            digest.update(batch[key].numpy().tobytes())
    model = build_model(cfg, args.method, device, overlay=args.overlay, policy=args.checkpoint_policy)
    model.train()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    write_json(out / "run_config.json", dict(config=cfg, method=args.method,
        activation_storage="selected" if args.method == "moc" else "dense",
        overlay=str(args.overlay) if args.overlay else None, checkpoint_policy=args.checkpoint_policy,
        micro_batch_size=args.micro_batch_size, logical_batch_size=16, batch_sha256=digest.hexdigest(),
        warmup_steps=6, repetitions=3, measured_steps_per_repetition=16,
        gpu=torch.cuda.get_device_name(device), source_files=source_hashes(),
        scope="training updates only; excludes loading, parity, reconstruction, saving and task evaluation"))
    probe = {k: v[:2].to(device) for k, v in batches[0].items() if k != "source_indices"}
    write_json(out / "parity.json", parity(model, named, probe, args.method == "moc"))
    del probe
    torch.cuda.empty_cache()
    params = [p for _, p in named]
    optimizer = torch.optim.AdamW(params, lr=cfg["training"]["learning_rate"], betas=(.9, .999), eps=1e-8, weight_decay=0.)
    def step(index):
        torch.manual_seed(20260929 + index)
        optimizer.zero_grad(set_to_none=True)
        value = backward_batch(model, batches[index], device, args.micro_batch_size, diagnostics=args.method == "dense")
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if args.method == "moc":
            parts = [p.grad.float().square().sum() for n, p in named if ".gate_proj." in n]
            # One host transfer, retaining the original Python FP64 summation order.
            gate = sum(torch.stack(parts).cpu().tolist()) ** .5
        else:
            gate = sum(float(p.grad.float().square().sum()) for n, p in named if ".gate_proj." in n) ** .5
        if not torch.isfinite(norm) or not gate > 0:
            raise RuntimeError("Invalid norm or gate gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = dict(step=index + 1, loss=value, grad_norm=float(norm), gate_grad_norm=gate,
                      nonpad_tokens=int(batches[index]["attention_mask"].sum()))
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        return record["nonpad_tokens"]
    for index in range(6):
        step(index)
    resident = torch.cuda.memory_allocated(device) / 2**30
    repetitions = []
    for repeat in range(3):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        tick = time.monotonic()
        tokens = sum(step(index) for index in range(6 + repeat * 16, 6 + (repeat + 1) * 16))
        torch.cuda.synchronize(device)
        seconds = time.monotonic() - tick
        repetitions.append(dict(seconds=seconds, nonpad_tokens=tokens, tokens_per_second=tokens / seconds,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30))
        write_json(out / "repetitions.json", repetitions)
        print(json.dumps(repetitions[-1]), flush=True)
    write_json(out / "summary.json", dict(complete=True, repetitions=repetitions, batch_sha256=digest.hexdigest(),
        initialized_resident_gib=resident, median_tokens_per_second=statistics.median(r["tokens_per_second"] for r in repetitions),
        max_peak_allocated_gib=max(r["peak_allocated_gib"] for r in repetitions),
        entry_elapsed_seconds=time.monotonic() - started))
    (out / "DONE").write_text("complete\n", encoding="utf-8")


def arguments(argv=None):
    from pathlib import Path
    p = parser(__doc__)
    p.add_argument("--method", choices=("dense", "moc"), required=True)
    p.add_argument("--overlay", type=Path)
    p.add_argument("--checkpoint-policy", choices=("block", "attention", "none"), required=True)
    p.add_argument("--micro-batch-size", type=int, choices=(4, 8, 16), required=True)
    options = p.parse_args(argv)
    if options.method == "dense" and options.overlay:
        p.error("Dense cannot use an overlay")
    if options.method == "moc" and not options.overlay:
        p.error("MoC requires --overlay from the reconstruction stage")
    return options


if __name__ == "__main__":
    options = arguments()
    if options.output_dir.exists():
        raise FileExistsError(f"Output already exists: {options.output_dir}")
    try:
        main(options)
    except torch.cuda.OutOfMemoryError as error:
        if options.output_dir.is_dir():
            write_json(options.output_dir / "oom.json", dict(error=str(error), scope="This fixed resource configuration did not fit"))
        raise
