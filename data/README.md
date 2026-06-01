# Data Layout

Large datasets and checkpoints are not tracked by Git.

Place raw C4 shards under:

```text
data/c4/
  train/.../*.json.gz
  val/.../*.json.gz
```

Place a local copy or symbolic link of the `t5-base` tokenizer under:

```text
data/tokenizer/
```

The C4 loader checks that this is a T5 tokenizer. It uses
per-document truncation/padding, shifted causal-LM labels, and `-100` labels
for padding tokens.

If the dataset or tokenizer is stored on another disk, create symbolic links at
`data/c4` and `data/tokenizer`.

Long-running C4 checkpoints should be stored under `data/checkpoints/`.
