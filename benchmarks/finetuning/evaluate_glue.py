"""Independently reload a selected run and score the official GLUE development split."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True, help="Output of train_glue")
    p.add_argument("--overlay", type=Path, help="Same overlay as training, required for K25")
    p.add_argument("--glue-dir", type=Path)
    a = p.parse_args()

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from checkpoint import adapter_identity, load_adapter, probe
    from distributed import four_gpus
    from glue_data import collate, load_split
    from glue_model import build
    from io_utils import CONFIG, read, sha, write
    from metrics import SCORERS

    selected = read(a.run / "selected.json")
    task = selected["task"]
    adapter = a.run / "selected"
    if (selected["method"] not in ("dense", "k25")
            or (a.overlay is not None) != (selected["method"] == "k25")
            or (sha(a.overlay / "OVERLAY.json") if a.overlay else None) != selected["overlay_sha256"]
            or sha(CONFIG) != selected["sources_sha256"]
            or sha(adapter / "adapter_model.safetensors") != selected["adapter_sha256"]):
        raise ValueError("Base/data recipe, overlay or selected adapter identity mismatch")
    with four_gpus(a.base) as (rank, device):
        tokenizer = AutoTokenizer.from_pretrained(a.base, local_files_only=True)
        dev = load_split(task, "validation", tokenizer, a.glue_dir)
        model = build(a.base, task, device, overlay=a.overlay, seed=selected["seed"]).eval()
        load_adapter(model, adapter)
        if adapter_identity(model) != selected["tensors"] or probe(model, dev, device) != selected["probe"]:
            raise ValueError("Reloaded head/adapter or probe logits differ from training")
        records = []
        shard = dev[rank::4]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(shard), 8):
                group = shard[start:start + 8]
                batch = collate(group)
                logits = model(**{k: batch[k].to(device) for k in ("input_ids", "attention_mask")}).logits.float()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite dev logits")
                values = logits.flatten() if task == "stsb" else logits.argmax(-1)
                records.extend((rank + 4 * (start + j), r["sample_id"], r["labels"], v)
                               for j, (r, v) in enumerate(zip(group, values.tolist())))
        gathered = [None] * 4
        dist.all_gather_object(gathered, records)
        if rank == 0:
            ordered = sorted(item for chunk in gathered for item in chunk)
            assert [(pos, i, g) for pos, i, g, _ in ordered] == [
                (pos, r["sample_id"], r["labels"]) for pos, r in enumerate(dev)]
            scores = SCORERS[task]([r[2] for r in ordered], [r[3] for r in ordered])
            report = dict(task=task, split="validation", examples=len(dev),
                          selected_step=selected["selected_step"], metrics=scores)
            write(a.run / "independent_dev.json", report)
            print(report)


if __name__ == "__main__":
    main()
