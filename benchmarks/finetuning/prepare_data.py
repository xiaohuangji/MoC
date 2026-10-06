"""Prepare the pinned Commonsense170K cache without changing its truncation policy."""
from pathlib import Path

import numpy as np

from common import CONFIG, load_config, write_json
from data import verify_data, verify_cache
from protocol import file_sha256, generate_training_prompt, load_json_list, validate_training_rows


def answer_mask(text, prefix, token_ids, offsets, eos_id, max_length):
    if not text.startswith(prefix) or len(text) <= len(prefix) or not token_ids:
        raise ValueError("Invalid training prompt")
    intact = max(end for _, end in offsets) >= len(text)
    mask = [intact and end > len(prefix) and end > begin for begin, end in offsets]
    token_ids = list(token_ids)
    if token_ids[-1] != eos_id and len(token_ids) < max_length:
        token_ids.append(eos_id)
        mask.append(bool(intact))
    if intact and not any(mask):
        raise ValueError("Complete answer has no response tokens")
    return token_ids, mask, intact


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIG)
    args = p.parse_args()
    cfg = load_config(args.config)
    manifest = verify_data(cfg["data_dir"])
    directory = Path(cfg["cache_dir"])
    if directory.exists():
        verify_cache(cfg)
        print(f"Cache already complete and verified: {directory}")
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_dir"], local_files_only=True)
    if not tokenizer.is_fast or tokenizer.eos_token_id is None:
        raise ValueError("A fast tokenizer with EOS is required for exact response boundaries")
    tokenizer.pad_token_id = 0
    tokenizer.padding_side, tokenizer.truncation_side = "left", "right"
    source = Path(cfg["data_dir"]) / manifest["training"]["path"]
    rows = load_json_list(source)
    validate_training_rows(rows)
    if len(rows) != manifest["training"]["rows"]:
        raise ValueError("Training row count mismatch")
    directory.mkdir(parents=True, exist_ok=False)
    width = cfg["data"]["max_length"]
    token_ids = np.lib.format.open_memmap(directory / "token_ids.npy", mode="w+", dtype=np.int32, shape=(len(rows), width))
    lengths = np.lib.format.open_memmap(directory / "lengths.npy", mode="w+", dtype=np.uint16, shape=(len(rows),))
    masks = np.lib.format.open_memmap(directory / "response_mask.npy", mode="w+", dtype=np.bool_, shape=token_ids.shape)
    token_ids[:] = -1
    masks[:] = False
    complete = np.zeros(len(rows), dtype=np.bool_)
    for start in range(0, len(rows), 512):
        group = rows[start:start+512]
        texts = [generate_training_prompt(row) for row in group]
        prefixes = [generate_training_prompt({**row, "output": ""}) for row in group]
        encoded = tokenizer(texts, truncation=True, max_length=width, padding=False, return_offsets_mapping=True)
        for offset, (text, prefix, ids, spans) in enumerate(zip(texts, prefixes, encoded["input_ids"], encoded["offset_mapping"])):
            ids, mask, intact = answer_mask(text, prefix, ids, spans, tokenizer.eos_token_id, width)
            row = start + offset
            lengths[row], complete[row] = len(ids), intact
            token_ids[row, :len(ids)] = ids
            masks[row, :len(mask)] = mask
        print(f"Tokenized {min(start+512, len(rows))}/{len(rows)}", flush=True)
    token_ids.flush()
    lengths.flush()
    masks.flush()
    permutation = np.random.default_rng(cfg["data"]["split_seed"]).permutation(len(rows))
    val, train = permutation[:120].astype(np.int64), permutation[120:].astype(np.int64)
    np.save(directory / "train_indices.npy", train)
    np.save(directory / "val_indices.npy", val)
    del token_ids, lengths, masks
    metadata = dict(complete=True, source_sha256=file_sha256(source), source_rows=len(rows), **cfg["data"],
        train_rows=len(train), validation_rows=len(val), padding_side="left", truncation_side="right",
        train_complete_answers=int(complete[train].sum()),
        validation_complete_answers=int(complete[val].sum()),
        files={p.name: file_sha256(p) for p in directory.glob("*.npy")},
        eos_token_id=tokenizer.eos_token_id, tokenizer_class=type(tokenizer).__name__,
        tokenizer_files={name: file_sha256(Path(cfg["model_dir"]) / name)
                         for name in ("tokenizer.json", "tokenizer_config.json")},
        response_policy="All original tokens supervised. Only complete answer spans and appended EOS receive extra weight.")
    write_json(directory / "metadata.json", metadata)
    verify_cache(cfg)


if __name__ == "__main__":
    main()
