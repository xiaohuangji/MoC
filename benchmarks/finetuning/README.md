# Reproducing the GLUE runs

Run from a Linux checkout on one node with four A100 40 GB GPUs. Install
PyTorch 2.8.0 (CUDA 12.8 in the experiments), a matching FlashAttention 2
build, then `pip install -e '.[finetuning]'`. The reconstruction uses Linux
shared memory via memfd and /proc; plan for a high-memory host (the original
job used a 320 GiB host-memory limit). Ranks must share files and be able to
open each other's /proc file descriptors. Disable Python assertion stripping.

Provide a local Llama-3.1-8B **Base**, not Instruct or a previous adapter.
The filenames and content hashes of the model/tokenizer, the eight English
C4 gzip shards, and the official GLUE parquet files are pinned in
[the input specification](../../configs/llama31_8b_glue_sources.json).
Put the eight C4 shards directly in one directory. GLUE downloads only the
requested train/dev files from the pinned nyu-mll/glue revision, or pass
`--glue-dir DIR` to both task commands with local files in `DIR/<task>/`.
Keep all outputs outside this checkout. Each new run requires a new directory.

```bash
BASE=/models/Llama-3.1-8B
C4=/datasets/c4/en
RUNS=/scratch/moc-glue

# Build once, reuse the same overlay on all eight tasks. No GLUE input.
bash scripts/run_reconstruct_c4.sh --base "$BASE" --c4-dir "$C4" --output "$RUNS/k25"

# One task and one rate per invocation, all four GPUs on the same run.
bash scripts/run_train_glue.sh --base "$BASE" --task mrpc --lr 1e-4 --method k25 \
  --overlay "$RUNS/k25" --output "$RUNS/mrpc-k25-1e-4"
bash scripts/run_evaluate_glue.sh --base "$BASE" --overlay "$RUNS/k25" \
  --run "$RUNS/mrpc-k25-1e-4"

# Dense uses the same supervised recipe but no overlay.
bash scripts/run_train_glue.sh --base "$BASE" --task mrpc --lr 1e-4 --method dense \
  --output "$RUNS/mrpc-dense-1e-4"
bash scripts/run_evaluate_glue.sh --base "$BASE" --run "$RUNS/mrpc-dense-1e-4"
```

Reconstruction uses the original reservoir and deduplication, 16,384 fit
windows and two disjoint 1,024-window partitions. Only fit and heldout enter
the 32-layer fit; the heldout relative block MSE selects among step 0 and
every 256 steps up to 8,192. Four ranks capture activations and validate;
rank zero updates the three FFN weights with the original shard shapes.
Only a complete run writes `OVERLAY.json`; keep it with its 32 layer files.

The supervised entry point trains either the original Dense baseline or K25
with the shared C4 overlay. Both start with a fresh LoRA adapter and task head
(default seed 42), with gradient checkpointing off. All eight tasks and the
independent evaluation use the same compressed-saved-input operator for K25.

For each learning rate, Trainer selects the epoch with the highest development
score, taking the earliest epoch in an exact tie.
`selected.json` binds the adapter, head, overlay and reference logits.
Independent evaluation reloads the selected checkpoint in fresh processes,
verifies these identities, and scores the full development split using the
original strided rank assignment and dynamic batches. For each model and task,
choose the learning rate with the highest score from `independent_dev.json`.
The file reports metrics on their native scale; multiply by 100 for the results table.

Small CPU contracts (no model download or GPU):

```bash
python -m unittest discover -s benchmarks/finetuning -p 'test_glue.py'
```
