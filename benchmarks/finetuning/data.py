"""Cached token rows and the original left-padding collator."""
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from common import read_json
from protocol import file_sha256, load_json_list, validate_evaluation_rows


MANIFEST = Path(__file__).with_name("data_manifest.json")


def verify_data(data_dir):
    manifest = read_json(MANIFEST)
    for record in [manifest["training"], *manifest["evaluations"].values()]:
        path = Path(data_dir) / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Dataset identity mismatch: {path}")
    return manifest


def verify_cache(cfg):
    manifest = verify_data(cfg["data_dir"])
    directory = Path(cfg["cache_dir"])
    meta = read_json(directory / "metadata.json")
    if meta["source_sha256"] != manifest["training"]["sha256"] or not meta["complete"]:
        raise ValueError("Incomplete cache or changed dataset")
    for key in ("max_length", "validation_size", "split_seed"):
        if meta[key] != cfg["data"][key]:
            raise ValueError(f"Cache configuration mismatch: {key}")
    required = {"token_ids.npy", "lengths.npy", "train_indices.npy", "val_indices.npy", "response_mask.npy"}
    if set(meta["files"]) != required:
        raise ValueError("Missing cache checksums")
    for name, digest in meta["files"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError(f"Cache checksum mismatch: {name}")
    if set(meta["tokenizer_files"]) != {"tokenizer.json", "tokenizer_config.json"}:
        raise ValueError("Missing tokenizer checksums")
    for name, digest in meta["tokenizer_files"].items():
        if file_sha256(Path(cfg["model_dir"]) / name) != digest:
            raise ValueError(f"Tokenizer identity changed: {name}")
    train = np.load(directory / "train_indices.npy", allow_pickle=False)
    val = np.load(directory / "val_indices.npy", allow_pickle=False)
    expected = np.random.default_rng(cfg["data"]["split_seed"]).permutation(170420)
    if not np.array_equal(val, expected[:120]) or not np.array_equal(train, expected[120:]):
        raise ValueError("Unexpected or overlapping train/validation split")
    ids = np.load(directory / "token_ids.npy", mmap_mode="r", allow_pickle=False)
    lengths = np.load(directory / "lengths.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(directory / "response_mask.npy", mmap_mode="r", allow_pickle=False)
    if (ids.shape != (170420, 256) or mask.shape != ids.shape or lengths.shape != (170420,)
            or ids.dtype != np.int32 or mask.dtype != np.bool_ or np.any(lengths < 2) or np.any(lengths > 256)):
        raise ValueError("Invalid token-cache schema")
    return meta


def loaders(cfg, *, pin_memory=True):
    metadata = verify_cache(cfg)
    directory = Path(cfg["cache_dir"])
    train = PackedTokenDataset(directory, "train")
    val = PackedTokenDataset(directory, "val")
    common = dict(num_workers=0, pin_memory=pin_memory, collate_fn=LeftPadCollator(0))
    return (DataLoader(train, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(cfg["seed"]), **common),
            DataLoader(val, batch_size=16, shuffle=False, **common), metadata)


def evaluation_rows(data_dir, task, limit=None):
    record = read_json(MANIFEST)["evaluations"][task]
    path = Path(data_dir) / record["path"]
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"Evaluation dataset hash mismatch: {task}")
    rows = load_json_list(path)
    validate_evaluation_rows(task, rows)
    if len(rows) != record["rows"]:
        raise ValueError("Evaluation row count changed")
    if limit is not None and limit <= 0:
        raise ValueError("Evaluation limit must be positive")
    return rows[:limit] if limit else rows

class PackedTokenDataset(Dataset):
    def __init__(self, cache_dir: Path, split: str, limit: int | None = None, boundary_dir: Path | None = None):
        if split not in ("train", "val"):
            raise ValueError(f"unsupported split: {split}")
        self.token_ids = np.load(cache_dir / "token_ids.npy", mmap_mode="r")
        self.lengths = np.load(cache_dir / "lengths.npy", mmap_mode="r")
        self.indices = np.load(cache_dir / f"{split}_indices.npy", mmap_mode="r")
        self.response_masks = np.load((boundary_dir or cache_dir) / "response_mask.npy", mmap_mode="r")
        if limit is not None:
            if limit <= 0:
                raise ValueError("dataset limit must be positive")
            self.indices = self.indices[:limit]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        row = int(self.indices[index])
        length = int(self.lengths[row])
        ids = torch.from_numpy(np.asarray(self.token_ids[row, :length], dtype=np.int64))
        return {"input_ids": ids, "response_mask": torch.from_numpy(np.array(self.response_masks[row, :length])), "source_index": row}


class LeftPadCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, items: list[dict]) -> dict:
        width = max(int(item["input_ids"].numel()) for item in items)
        width = math.ceil(width / self.pad_to_multiple_of) * self.pad_to_multiple_of
        input_ids = torch.full(
            (len(items), width), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(items), width), dtype=torch.long)
        labels = torch.full((len(items), width), -100, dtype=torch.long)
        response_mask = torch.zeros((len(items), width), dtype=torch.bool)
        for row, item in enumerate(items):
            ids = item["input_ids"]
            length = ids.numel()
            input_ids[row, -length:] = ids
            attention_mask[row, -length:] = 1
            labels[row, -length:] = ids
            response_mask[row, -length:] = item["response_mask"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "response_mask": response_mask,
            "source_indices": torch.tensor([item["source_index"] for item in items], dtype=torch.long),
        }
