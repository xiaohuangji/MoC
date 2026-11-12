"""Official GLUE inputs, original packing and dynamic right padding."""
from pathlib import Path
import torch
from io_utils import CONFIG, read, sha

# train count, dev count, epochs, first field, optional second field, selection metric
TASKS = {
    "cola": (8551, 1043, 3, "sentence", None, "matthews_correlation"),
    "sst2": (67349, 872, 3, "sentence", None, "accuracy"),
    "mrpc": (3668, 408, 20, "sentence1", "sentence2", "combined"),
    "stsb": (5749, 1500, 3, "sentence1", "sentence2", "combined"),
    "qqp": (363846, 40430, 3, "question1", "question2", "combined"),
    "mnli": (392702, 19647, 3, "premise", "hypothesis", "combined"),
    "qnli": (104743, 5463, 3, "question", "sentence", "accuracy"),
    "rte": (2490, 277, 3, "sentence1", "sentence2", "accuracy"),
}


def pack(first, second, bos, eos, limit=128):
    a, b = list(first), None if second is None else list(second)
    budget = limit - (2 if b is None else 3)
    original = len(a)+(0 if b is None else len(b))
    while len(a)+(0 if b is None else len(b)) > budget:
        if b is not None and len(b) > len(a): b.pop()
        else: a.pop()
    ids = [bos]+a+[eos]
    if b is not None: ids += b+[eos]
    return ids, original > budget


def collate(rows):
    width = max(len(r['input_ids']) for r in rows)
    ids = torch.zeros((len(rows), width), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        n = len(row['input_ids'])
        ids[i, :n] = torch.tensor(row['input_ids'])
        mask[i, :n] = 1
    return dict(input_ids=ids, attention_mask=mask,
                labels=torch.tensor([r['labels'] for r in rows]),
                sample_id=torch.tensor([r['sample_id'] for r in rows]))

def load_split(task, split, tokenizer, directory=None):
    from huggingface_hub import hf_hub_download
    from pyarrow.parquet import ParquetFile

    if split not in ("train", "validation"):
        raise ValueError("Only train and official development splits are supported")
    config = read(CONFIG)
    train, dev, _, first, second, _ = TASKS[task]
    parts = (["validation_matched", "validation_mismatched"]
             if task == "mnli" and split == "validation" else [split])
    rows = []
    for subset in parts:
        name = subset + "-00000-of-00001.parquet"
        path = (Path(directory) / task / name if directory else hf_hub_download(
            "nyu-mll/glue", task + "/" + name, repo_type="dataset", revision=config["glue_revision"]))
        if sha(path) != config["glue"][task][name]:
            raise ValueError(f"Unexpected GLUE source {task}/{name}")
        for batch in ParquetFile(path).iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                a = tokenizer.encode(row[first], add_special_tokens=False)
                b = None if second is None else tokenizer.encode(row[second], add_special_tokens=False)
                ids, _ = pack(a, b, tokenizer.bos_token_id, tokenizer.eos_token_id)
                sample_id = len(rows) if len(parts) == 2 else row["idx"]
                rows.append(dict(input_ids=ids, labels=row["label"], source_idx=row["idx"],
                                 sample_id=sample_id, subset=subset))
    if len(rows) != (train if split == "train" else dev):
        raise ValueError("GLUE split size mismatch")
    return rows
