"""C4 streaming dataloader for MoC benchmarks and pre-training.

Loads C4 (English) JSON shards from disk and tokenizes them with a local
``t5-base`` tokenizer stored under ``data/tokenizer``. Documents are truncated
or padded independently, then shifted into causal-LM input/label pairs. Padding
labels are set to ``-100`` so they are ignored by cross entropy.

Layout expected on disk:
  <repo_root>/data/c4/train/<shard-dir>/c4-train.NNNNN-of-01024.json.gz
  <repo_root>/data/c4/val/<shard-dir>/c4-validation.NNNNN-of-00008.json.gz
"""
import gzip
import glob
import json
import os
import random
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_root() -> Path:
    return _repo_root() / "data"


def resolve_tokenizer_path() -> Path:
    return _repo_root() / "data" / "tokenizer"


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(str(resolve_tokenizer_path()))
    if "T5" not in type(tokenizer).__name__:
        raise ValueError(
            "data/tokenizer must contain local t5-base tokenizer files "
            f"(got {type(tokenizer).__name__})."
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def list_shards(split: str = "train", shard_range: Optional[tuple] = None) -> List[str]:
    data_root = resolve_data_root()
    if split == "train":
        patterns = [
            str(data_root / "c4" / "train" / "*" / "*.json.gz"),
            str(data_root / "c4" / "train" / "*.json.gz"),
        ]
    elif split == "val":
        patterns = [
            str(data_root / "c4" / "val" / "*" / "*.json.gz"),
            str(data_root / "c4" / "val" / "*.json.gz"),
        ]
    else:
        raise ValueError(f"split must be 'train'|'val', got {split!r}")

    paths: List[str] = []
    for p in patterns:
        paths.extend(glob.glob(p))
    paths = sorted(set(paths))

    if shard_range is not None:
        lo, hi = shard_range
        out = []
        for p in paths:
            fname = os.path.basename(p)
            try:
                num = int(fname.split(".")[1].split("-")[0])
            except (ValueError, IndexError):
                continue
            if lo <= num <= hi:
                out.append(p)
        return sorted(out)
    return paths


def _emit_shuffled_buffer(buffer: list[str], rng: random.Random) -> Iterator[str]:
    rng.shuffle(buffer)
    while buffer:
        yield buffer.pop()


def iter_documents(
    shard_paths: List[str],
    max_docs: Optional[int] = None,
    shuffle_docs: bool = False,
    shuffle_seed: int = 42,
    shuffle_buffer_docs: int = 1000,
) -> Iterator[str]:
    count = 0
    rng = random.Random(shuffle_seed)
    buffer: list[str] = []
    for path in shard_paths:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if max_docs is not None and count >= max_docs:
                    if shuffle_docs and buffer:
                        yield from _emit_shuffled_buffer(buffer, rng)
                    return
                doc = json.loads(line)
                text = doc.get("text", "")
                if text.strip():
                    count += 1
                    if shuffle_docs:
                        buffer.append(text)
                        if len(buffer) >= shuffle_buffer_docs:
                            yield from _emit_shuffled_buffer(buffer, rng)
                    else:
                        yield text
    if shuffle_docs and buffer:
        yield from _emit_shuffled_buffer(buffer, rng)


class C4ShardDataset(IterableDataset):
    """Streaming C4 dataset with local t5-base tokenization."""

    def __init__(self, split: str = "train", seq_len: int = 256,
                 shard_range: Optional[tuple] = None,
                 max_docs: Optional[int] = None,
                 max_tokens: Optional[int] = None,
                 tokenizer=None,
                 shuffle_docs: bool = False,
                 shuffle_seed: int = 42,
                 shuffle_buffer_docs: int = 1000):
        self.split = split
        self.seq_len = seq_len
        self.shard_range = shard_range
        self.max_docs = max_docs
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer or load_tokenizer()
        self.shuffle_docs = shuffle_docs
        self.shuffle_seed = shuffle_seed
        self.shuffle_buffer_docs = shuffle_buffer_docs

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        shards = list_shards(self.split, self.shard_range)
        if worker_info is not None:
            shards = shards[worker_info.id::worker_info.num_workers]

        emitted_tokens = 0
        for text in iter_documents(
            shards,
            max_docs=self.max_docs,
            shuffle_docs=self.shuffle_docs,
            shuffle_seed=self.shuffle_seed,
            shuffle_buffer_docs=self.shuffle_buffer_docs,
        ):
            encoded = self.tokenizer(
                text,
                max_length=self.seq_len + 1,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            ids = encoded["input_ids"][0]
            mask = encoded["attention_mask"][0]
            input_ids = ids[:-1].clone()
            labels = ids[1:].clone()
            label_mask = mask[1:].bool()
            labels[~label_mask] = -100
            yield {"input_ids": input_ids.long(), "labels": labels.long()}
            emitted_tokens += int(label_mask.sum().item())
            if self.max_tokens is not None and emitted_tokens >= self.max_tokens:
                return


def build_dataloader(split: str = "train", batch_size: int = 32,
                     seq_len: int = 256, num_workers: int = 2,
                     shard_range: Optional[tuple] = None,
                     max_docs: Optional[int] = None,
                     max_tokens: Optional[int] = None,
                     tokenizer=None,
                     shuffle: bool = False,
                     shuffle_seed: int = 42,
                     shuffle_buffer_docs: int = 1000) -> DataLoader:
    dataset = C4ShardDataset(
        split=split, seq_len=seq_len, shard_range=shard_range,
        max_docs=max_docs, max_tokens=max_tokens, tokenizer=tokenizer,
        shuffle_docs=shuffle, shuffle_seed=shuffle_seed,
        shuffle_buffer_docs=shuffle_buffer_docs,
    )
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )

