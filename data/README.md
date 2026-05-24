# Data Layout

Large datasets and checkpoints are not tracked by Git.

Place C4 under:

```text
data/c4/
  train/
  val/
```

The C4 folders should contain the raw `json.gz` shards.

Place a local copy or symbolic link of the `t5-base` tokenizer under:

```text
data/tokenizer/
```

The C4 loader checks that this is a T5 tokenizer. It uses
per-document truncation/padding, shifted causal-LM labels, and `-100` labels
for padding tokens.

If the dataset is stored on another disk, create a symbolic link at `data/c4`.
