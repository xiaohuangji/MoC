"""Reconstruct the fixed C4 partitions locally; never use downstream data."""
import gzip
import json
from pathlib import Path
import random

from transformers import AutoTokenizer
from dedup import Deduplicator, document_sha, reservoir, window_sha
from io_utils import CONFIG, object_sha, read, sha, write


def window(line, text, shard, tokenizer, generator):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < 126:
        return None
    offset = generator.randrange(len(tokens)-125)
    ids = [128000]+tokens[offset:offset+126]+[128001]
    return dict(input_ids=ids, labels=0, label_is_unused_placeholder=True,
        text_sha256=document_sha(text), window_sha256=window_sha(ids), shard=shard,
        source_line=line, window_start=offset, source_text_tokens=len(tokens))


def replay_prefix(reservoirs, tokenizer):
    groups = {'fit': [], 'heldout': [], 'audit': []}
    dedup = Deduplicator(); tails = []
    for shard, candidates in enumerate(reservoirs):
        generator = random.Random(4300+shard); accepted = []
        for offset, (line, text) in enumerate(candidates):
            row = window(line, text, shard, tokenizer, generator)
            if row is None or not dedup.accept(row['text_sha256'], row['input_ids']):
                continue
            accepted.append(row)
            if len(accepted) == 1280:
                tails.append((offset+1, generator.getstate()))
                break
        assert len(accepted) == 1280
        generator.shuffle(accepted)
        groups['fit'].extend(accepted[:1024]); groups['heldout'].extend(accepted[1024:1152])
        groups['audit'].extend(accepted[1152:])
    for index, rows in enumerate(groups.values()):
        random.Random(4400+index).shuffle(rows)
    for index, row in enumerate(row for rows in groups.values() for row in rows):
        row['sample_id'] = index
    return groups, tails, dedup


def extend(reservoirs, tokenizer, tails, dedup):
    added = []
    for shard, candidates in enumerate(reservoirs):
        offset, state = tails[shard]
        generator = random.Random(); generator.setstate(state)
        accepted = []
        for line, text in candidates[offset:]:
            row = window(line, text, shard, tokenizer, generator)
            if row is None or not dedup.accept(row['text_sha256'], row['input_ids']):
                continue
            accepted.append(row)
            if len(accepted) == 1024:
                break
        assert len(accepted) == 1024, 'Fixed reservoir insufficient; no automatic resampling'
        added.extend(accepted)
    random.Random(4500).shuffle(added)
    for index, row in enumerate(added, 10240):
        row['sample_id'] = index
    return added


def prepare(base, directory, output):
    config = read(CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    reservoirs = []
    for i, (name, digest) in enumerate(config["c4_shards"]):
        source = Path(directory) / name
        if sha(source) != digest:
            raise ValueError(f"Unexpected C4 shard: {name}")
        with gzip.open(source, "rt", encoding="utf8") as stream:
            samples, _, _ = reservoir(map(json.loads, stream), 4096, random.Random(4200 + i))
        reservoirs.append(samples)
    groups, tails, dedup = replay_prefix(reservoirs, tokenizer)
    groups["fit"].extend(extend(reservoirs, tokenizer, tails, dedup))
    if {k: object_sha(v) for k, v in groups.items()} != config["partitions"]:
        raise ValueError("C4 window membership, order or tokenization differs")
    write(output, groups)
