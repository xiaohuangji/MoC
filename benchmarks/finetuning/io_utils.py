import hashlib
import json
from pathlib import Path

import torch


CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'llama31_8b_glue_sources.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n', encoding='utf-8')
    temp.replace(path)


def sha(p, source=False):
    if source:
        return hashlib.sha256(Path(p).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''): h.update(chunk)
    return h.hexdigest()


def object_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensor_sha(tensor):
    tensor=tensor.detach()
    digest=hashlib.sha256(str((tuple(tensor.shape),tensor.dtype)).encode())
    for chunk in tensor.split(256):
        digest.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_base(base):
    for filename, digest in read(CONFIG)["base"].items():
        if sha(Path(base) / filename) != digest:
            raise ValueError(f"Unexpected base model/tokenizer: {filename}")
