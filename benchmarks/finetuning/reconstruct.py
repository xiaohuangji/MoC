"""Sequential, train-only Dense-block reconstruction into fixed-K BF16 FFNs."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import cuda_device, load_config, parser, source_hashes, write_json
from data import verify_cache
from model import build_model, convert
from overlay import BASE_NAMES, save_layer
from protocol import file_sha256
from reconstruction import StreamingTensor, autocast_for, fit, measure, tensor_digest
from trajectory import cache_dense, capture_pair, correction_target, load_target, measure_block


def reconstruction_rows(train_indices, smoke=False):
    order = np.random.default_rng(20260920).permutation(train_indices)
    train = np.concatenate((order[:512], order[640:2176], order[2432:8576]))
    validation = order[512:640]
    if len(train) != 8192 or len(validation) != 128 or np.intersect1d(train, validation).size:
        raise ValueError("Invalid reconstruction split")
    return (train[:8], validation[:4]) if smoke else (train, validation)


def main():
    started = time.monotonic()
    p = parser(__doc__)
    p.add_argument("--smoke", action="store_true", help="One layer / four updates; produces an incomplete overlay")
    args = p.parse_args()
    cfg = load_config(args.config)
    rec = cfg["reconstruction"]
    verify_cache(cfg)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = cuda_device(20260920)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(cfg, "dense", device, policy="none", expand=False)
    model.requires_grad_(False).eval()
    layers = model.base_model.model.model.layers
    layer_ids = [0] if args.smoke else list(range(len(layers)))
    directory = Path(cfg["cache_dir"])
    ids = np.load(directory / "token_ids.npy", mmap_mode="r")
    lengths = np.load(directory / "lengths.npy", mmap_mode="r")
    train_rows, val_rows = reconstruction_rows(np.load(directory / "train_indices.npy"), args.smoke)
    train_limit = 128 if args.smoke else rec["train_token_limit"]
    val_limit = 128 if args.smoke else rec["validation_token_limit"]
    steps = 4 if args.smoke else rec["steps_per_layer"]
    write_json(out / "run_config.json", dict(config=cfg, smoke=args.smoke, source_files=source_hashes(),
        gpu=torch.cuda.get_device_name(device), layers=layer_ids, steps_per_layer=steps,
        source_adapter_required=True, rank_during_reconstruction=32,
        teacher="original Dense complete block outputs", target="teacher block output minus current student residual",
        cache_storage="CPU RAM capture, sequential disk write, CPU correction targets", strict_resume=False))
    cache_root = out / "teacher_cache"
    teacher = cache_dense(model, layers, ids, lengths,
        [("train", train_rows, train_limit, 20260920), ("validation", val_rows, val_limit, 20260921)], layer_ids, cache_root)
    convert(layers, cfg["k"])
    protected = {n: tensor_digest(t) for n, t in model.named_parameters()
                 if not (".mlp." in n and n.endswith(".base_layer.weight"))}
    saved, results, peak = [], [], torch.cuda.max_memory_allocated(device)
    for index in layer_ids:
        model.requires_grad_(False).eval()
        module = layers[index].mlp
        x, residual, cap = capture_pair(model, layers[index], ids, lengths, train_rows, train_limit, 20260920, storage_device="cpu")
        v, vres, vcap = capture_pair(model, layers[index], ids, lengths, val_rows, val_limit, 20260921, storage_device="cpu")
        for name, capture in (("train", cap), ("validation", vcap)):
            if capture["selection_sha256"] != teacher["groups"][name]["selection_sha256"]:
                raise RuntimeError("Teacher/student token order mismatch")
        dense_y = load_target(cache_root, teacher, "train", index, "cpu")
        dense_v = load_target(cache_root, teacher, "validation", index, "cpu")
        y, vy = correction_target(dense_y, residual), correction_target(dense_v, vres)
        x, v, y, vy, vres, dense_v = (StreamingTensor(t, device) for t in (x, v, y, vy, vres, dense_v))
        del residual, dense_y
        original = {n: tensor_digest(t) for n, t in module.named_parameters() if n in BASE_NAMES}
        frozen = {n: tensor_digest(t) for n, t in module.named_parameters() if n not in BASE_NAMES}
        before = measure(module, v, vy)
        block_before = measure_block(module, v, vres, dense_v)
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(name in BASE_NAMES)
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        peak = max(peak, torch.cuda.max_memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        def record(row):
            value = dict(row, layer=index, elapsed_seconds=time.monotonic() - started)
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, allow_nan=False) + "\n")
            print(json.dumps(value), flush=True)
        last = fit(module, x, y, steps=steps, lr=rec["learning_rate"], batch_tokens=rec["batch_tokens"], seed=20260920 + index, record=record)
        after = measure(module, v, vy)
        block_after = measure_block(module, v, vres, dense_v)
        if frozen != {n: tensor_digest(t) for n, t in module.named_parameters() if n not in BASE_NAMES}:
            raise RuntimeError("Reconstruction changed a frozen adapter")
        with torch.no_grad(), autocast_for(v):
            before_export = module(v[:256]).clone()
        for name, parameter in module.named_parameters():
            parameter.grad = None
            parameter.requires_grad_(False)
            if name in BASE_NAMES:
                parameter.data = parameter.data.to(torch.bfloat16)
        with torch.no_grad(), autocast_for(v):
            if not torch.equal(before_export, module(v[:256])):
                raise RuntimeError("BF16 export changed the forward result")
        saved.append(save_layer(module, index, out / "overlay", original))
        result = dict(layer=index, before=before, after=after, block_before=block_before, block_after=block_after,
                      last_train_event=last, capture=cap, validation_capture=vcap, exported=saved[-1])
        results.append(result)
        write_json(out / f"layer_{index:02d}.json", result)
        peak = max(peak, torch.cuda.max_memory_allocated(device))
        del x, v, y, vy, vres, dense_v, before_export
        gc.collect()
        torch.cuda.empty_cache()
    if protected != {n: tensor_digest(t) for n, t in model.named_parameters()
                     if not (".mlp." in n and n.endswith(".base_layer.weight"))}:
        raise RuntimeError("Attention, embeddings or adapter weights changed during reconstruction")
    write_json(out / "overlay/manifest.json", dict(complete=not args.smoke, format="base_ffn_bf16_v1",
        k=cfg["k"], selector="raw_gate", source_adapter_sha256=cfg["source_adapter_sha256"],
        model_config_sha256=file_sha256(Path(cfg["model_dir"]) / "config.json"), layers=saved,
        requires_original_adapter=True, not_strict_resume_checkpoint=True))
    write_json(out / "summary.json", dict(complete=True, full_overlay=not args.smoke, layers=results,
        teacher_cache_seconds=teacher["cache_seconds"], teacher_cache_bytes=teacher["bytes"],
        entry_elapsed_seconds=time.monotonic() - started,
        peak_allocated_gib=max(peak, torch.cuda.max_memory_allocated(device)) / 2**30,
        peak_scope="from model loading, including fresh teacher cache and all reconstructed layers"))
    (out / "DONE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
