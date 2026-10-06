# Commonsense Fine-Tuning

This benchmark converts a common Dense-SFT Llama-3.1-8B checkpoint to fixed-low-K
MoC, reconstructs its FFNs, and continues LoRA training. It is separate from C4
pretraining and the custom `moc/` training kernels. Results are in
[results/finetuning.md](../results/finetuning.md).

The result table distinguishes three routes:

| Route | FFN initialization before downstream LoRA | Reconstruction |
| --- | --- | --- |
| Dense | Unchanged Dense FFNs | None |
| Direct MoC | Dense weights with fixed-K selection enabled | None |
| Reconstructed MoC | Fixed-K FFNs fitted to original Dense block outputs | Sequential FFN fitting |

All three start from the same already eight-task-SFT Dense adapter and receive
10240 downstream updates. Direct MoC transfers those weights without fitting
them to sparse execution first. Reconstructed MoC performs that fitting after
the architecture switch and before downstream training; it is not a separately
pretrained MoC base model. Their macro accuracies are 80.80%, 73.14%, and 77.02%,
respectively, with reconstruction cost reported separately.

The launcher implements `dense` and `moc`; `moc` selects **Reconstructed MoC**,
including compact activation storage, and requires its FFN overlay. Direct MoC
is the no-reconstruction control in the result table, not a fallback when that
overlay is absent.

## Prerequisites

Use a separate environment with CUDA 12.8 / PyTorch 2.8.0 and the fine-tuning extras:

```bash
pip install -e '.[finetuning]'
```

The measured stack uses Transformers 5.10.2, PEFT 0.20.0 and Accelerate 1.14.0.
Select one available GPU with `CUDA_VISIBLE_DEVICES`. The scripts neither find
idle GPUs nor start multi-GPU jobs. The measured device was an A100-PCIE-40GB.
Reconstruction also needs substantial host RAM and disk: the main teacher arrays
alone use about 64 GiB; teacher files use about 67.8 GiB and the final FFN overlay
about 10.5 GiB. These are not whole-process host-RAM limits; allow extra headroom.

Obtain Meta-Llama-3.1-8B **Base**, including its original tokenizer, under its model
license. Use the training JSON and eight evaluation JSONs from
[LLM-Adapters revision 8166572](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/816657208af4db747803f87ba40a4c71383fed7a).
The expected files and SHA256 checksums are in
[data_manifest.json](../benchmarks/finetuning/data_manifest.json).

```text
data/models/Meta-Llama-3.1-8B-base/    HF model, config and tokenizer files
data/commonsense/commonsense_170k.json
data/commonsense/eval/<task>/test.json
data/checkpoints/commonsense_source/adapter/
  adapter_config.json
  adapter_model.safetensors
```

**The common source adapter is required and is not distributed in this repository.**
It is an already eight-task-fine-tuned Dense rank32/alpha64 adapter with gate LoRA,
not raw Llama Base. Its weight-file SHA256 is
`f13af507efe3526bb74e3f3c369ca25dbc711dba69589dd62cfbd346c3028f78`.
The entry verifies this identity before loading. Until this artifact is supplied,
a fresh clone cannot independently reproduce the reported starting point.
Substituting another SFT checkpoint produces a different experiment, even if the
architecture and hyperparameters match. There is no implicit fresh-Base fallback.

Paths are configurable in [llama31_8b_commonsense.yaml](../configs/llama31_8b_commonsense.yaml);
relative paths resolve from the repository root. Model loading is local-only;
launching a training command never downloads model weights.

## Method

For every token and every FFN, select the largest **raw gate** values with
`torch.topk(..., largest=True, sorted=False)`. With intermediate width 14336,
all 32 layers use K2048 (14.2857%), both during downstream training and evaluation.
There is no K annealing, Dense/MoC output blending, or test-time K increase.

Reconstruction first caches the original Dense **complete block outputs** on
training-only rows. Layers are then fitted sequentially, using the current sparse
student prefix. The regression target for the current FFN is:

```text
target = original_dense_block_output - current_student_pre_ffn_residual
```

Only the current gate/up/down base matrices are updated. Attention, embeddings,
and the original rank32 adapter remain frozen; dropout is disabled in this phase.
Use 8192 training rows, up to 262144 sampled token positions per layer, 128
disjoint training-only calibration rows, and 4096 updates per layer. The objective
is relative MSE, token minibatch 1024, LR 3e-5 with 10% warmup and linear decay.
Current-layer FP32 master matrices are optimized with BF16 compute and exported
as BF16. CPU-resident captures are transferred one minibatch at a time.

Downstream training freezes the reconstructed base matrices and trains LoRA on
q/k/v/gate/up/down. The original rank32 factors are preserved while rank96 is
added, with new B factors zero-initialized. Split rank32/rank96 matrix operations
are retained for numerical consistency. Total rank is 128, alpha256, dropout0.05,
with 301,989,888 trainable parameters. Base weights are BF16; LoRA masters are
FP32 with BF16 autocast. Gate LoRA is trainable.

MoC keeps native PEFT gate/up backward and stores compact int16 indices and
selected values. Index copies operate directly on int16 indices. Split-rank LoRA
shares the BF16 input conversion while retaining independent backward casts and
uses views for its B inputs. Down-projection backward reconstructs temporary
full-width BF16/FP32 inputs, gathers each branch's gradient, and then accumulates
the compact gradients with the original BF16 rounding. Native GEMMs, dropout and
RNG replay are preserved. These are one fixed implementation, not optional
algorithm variants; Dense retains its original split-rank implementation.

This is **not** an all-sparse GEMM implementation and does not shrink frozen model
weights. Both methods share chunked cross-entropy. MoC skips unused loss
diagnostics; neither that change nor general LoRA implementation improvements
should be attributed exclusively to the MoC algorithm.

## Data And Evaluation

The pinned mixture contains 170420 rows. A seed42 permutation selects 120 internal
validation rows; the remaining 170300 form training. Training uses right
truncation to 256 and left padding to a multiple of 8. Complete answer spans and
appended EOS receive weight16; other supervised tokens receive weight1. Truncated
or missing answers receive no extra weight. This retains the original cutoff
limitation rather than silently changing the dataset.

MoC and Dense use seed42, logical batch16, 10240 optimizer updates, AdamW
(betas0.9/0.999, epsilon1e-8, weight decay0), LR3e-5, warmup128 and linear decay,
with gradient clipping at 1.0. Accumulation is normalized by weighted supervised
tokens, including a partial last batch. Micro-batch size changes can still change
dropout and floating-point trajectories. The final step is selected in advance;
internal validation loss is logged, not used to select the best test result.

Evaluation uses the exact upstream instruction templates with **project-specific
canonical-answer constrained beam generation**: beam4, batch1, max-new-tokens32,
no evaluation prompt truncation. Report per-task accuracy and its unweighted
eight-task macro average on all 22419 rows. Labels are not supplied to generation.
The preserved candidate vocabulary includes `answer1` through `answer5` for the
SocialIQA/ARC/OpenBookQA formats; it is not dynamically reduced per question.

This is an adapted LLM-Adapters/NVIDIA-style recipe, **not an unmodified NVIDIA
trainer, likelihood-ranking evaluation, or official leaderboard submission**.
Upstream files named `test.json` do not imply every task uses a hidden official
test set (WinoGrande here has 1267 public validation examples). These benchmarks
were used during method development; the single-seed table is not an independent
blind confirmation. Internal validation row separation also does not guarantee
unique question text across the mixture.

## Commands

Prepare and verify the token cache, then reconstruct once:

```bash
STAGE=prepare bash scripts/run_finetuning.sh
CUDA_VISIBLE_DEVICES=0 STAGE=reconstruct \
  OUTPUT_DIR=data/checkpoints/commonsense_reconstruction bash scripts/run_finetuning.sh
```

Run MoC and its Dense baseline separately on an available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 METHOD=dense OUTPUT_DIR=results/raw/finetuning/dense \
  bash scripts/run_finetuning.sh
CUDA_VISIBLE_DEVICES=0 METHOD=moc \
  OVERLAY=data/checkpoints/commonsense_reconstruction/overlay \
  OUTPUT_DIR=results/raw/finetuning/moc bash scripts/run_finetuning.sh
```

Defaults are block checkpointing/micro16 for Dense and attention-only
checkpointing/micro8 for MoC. Both use logical16. The launcher defaults to MoC
and the reconstruction directory shown above. MoC requires a complete FFN overlay;
there is no fallback to an unreconstructed model and no activation-storage switch.
Outputs include `run_config.json`, `metrics.jsonl`, `final_eval.json`, `summary.json`,
per-task prediction JSONL files, and adapter weights. Existing output directories
are refused. Periodic adapters are model snapshots, **not strict optimizer/RNG
resume checkpoints**.

Smoke checks use separate output directories. Add `--stop-at-step 4 --eval-limit 2`
to a training command; the LR horizon remains 10240 and the output is marked smoke.
For reconstruction, add `--smoke`; its one-layer overlay is explicitly incomplete
and rejected by full-model loading.

Reload a final model for evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 STAGE=evaluate METHOD=moc \
  OVERLAY=data/checkpoints/commonsense_reconstruction/overlay \
  ADAPTER=results/raw/finetuning/moc/adapter \
  OUTPUT_DIR=results/raw/finetuning/reloaded bash scripts/run_finetuning.sh
```

A reconstructed model requires the original Base, common source adapter,
**32-layer BF16 FFN overlay**, final rank128 adapter, and its `moc_config.json`.
The loader verifies identities and restores the split-rank layout. A standalone
`PeftModel.from_pretrained` call on the final adapter is not sufficient.

## Resource Measurement

The resource table separates Dense fine-tuning, one-time FFN reconstruction,
and MoC fine-tuning with reconstructed weights. Only the fine-tuning rows share
a matched training configuration and define the reported memory-saving and
throughput ratios. Reconstruction reports its own peak, including model loading
and fresh teacher capture; its single-layer fitting throughput is not comparable
to full-model training. Its GPU-hours are included separately in the total-cost
table. Phase peaks are not summed, and the reported downstream memory saving
is not a measured full-pipeline saving.

To measure the resource-table configuration, use `STAGE=resources`, fixed
`CHECKPOINT_POLICY=attention`, and `MICRO_BATCH_SIZE=8`. For example:

```bash
CUDA_VISIBLE_DEVICES=0 STAGE=resources METHOD=moc \
  OVERLAY=data/checkpoints/commonsense_reconstruction/overlay \
  CHECKPOINT_POLICY=attention MICRO_BATCH_SIZE=8 \
  OUTPUT_DIR=results/raw/finetuning/resource_moc_attention8 bash scripts/run_finetuning.sh
```

For the primary comparison, compare `METHOD=moc` with `METHOD=dense` (no overlay)
at the **same** `CHECKPOINT_POLICY` and `MICRO_BATCH_SIZE`. For the command above,
the matched Dense control is:

```bash
CUDA_VISIBLE_DEVICES=0 STAGE=resources METHOD=dense \
  CHECKPOINT_POLICY=attention MICRO_BATCH_SIZE=8 \
  OUTPUT_DIR=results/raw/finetuning/resource_dense_attention8 bash scripts/run_finetuning.sh
```

Keep `OVERLAY` unset for Dense if it was previously exported in the shell.
The reported aggregates use three runs per method, each with a distinct output
directory. Keep attention-only checkpointing, micro8 and logical batch16 fixed;
the entry uses two accumulation passes. The reported Dense and optimized MoC
runs come from separate measurement batches, not a new interleaved paired suite.
Micro batches 4 and 16 are supported but were not remeasured with this optimized
implementation.

Here GCP is activation recomputation, not saving model files. Attention-only
checkpointing is distinct from the paper's FFN element-wise GCP, which preserves
gate/up activations and recomputes SiLU/products. That specific Dense/MoC pairing
is not separately measured by the present LoRA resource suite.
The entry also supports `none` and `block` checkpoint policies, but these are not the
settings of the reported resource table. MoC always retains its compact-storage
method and associated backward recomputation. Resource switches do not select
different MoC algorithms. Do not mix checkpoint policies or micro batches within
a controlled Dense/MoC pair.

Run configurations serially on the same GPU. The entry checks whole-model reference gradients before timing,
prefetches 54 identical logical batches, warms up six updates, and measures three
16-update windows. Each invocation reports median nonpadding-token throughput
and the maximum `torch.cuda.max_memory_allocated()` across those windows, in
**GiB**. Across three invocations, use the mean of the per-run medians and the
maximum peak across all runs. OOMs are
recorded as failures, not discarded or automatically retried at smaller batches.
Compute memory saving as `100 * (1 - MoC_peak / Dense_peak)` and throughput
retained as `100 * MoC_token_per_s / Dense_token_per_s` from unrounded aggregate
measurements. Do not assign ratios to OOM cases. The three timing windows are
not independent seeds or confidence intervals. Short resource probes do not
establish per-configuration downstream quality; the quality-run settings are
reported separately.

For the separate equal-total-time Dense control, use `CHECKPOINT_POLICY=none`,
`MICRO_BATCH_SIZE=4` and `--wall-budget-seconds <training_budget>` with `STAGE=train`.
Subtract a separately measured evaluation tail from the desired total cost.
The deadline includes model setup, training, intermediate validation and saves;
final saving/evaluation is extra. LR warmup is step-based, followed by wall-time
decay, so this is not the 10240-step scheduler. Measure actual process occupancy:
a wall-clock deadline cannot promise exact total GPU-time matching on another host.

## Verification

```bash
python -m unittest discover -s benchmarks/finetuning -p test_finetuning.py -v
```

Tests cover raw-gate selection, actual PEFT forward/gradient/RNG behavior with
checkpointing, rank expansion and adapter reload, chunked loss, token-weighted
accumulation, reconstruction capture/order, frozen weights, and overlay identity.
They also check the learning-rate config contract, isolation of MoC-only changes,
diagnostic-free loss equivalence, down-projection output/gradient/Adam/RNG bitwise
equivalence, and compact CUDA copies. CUDA-only cases skip when CUDA is unavailable.
CPU tests do not replace BF16 CUDA/8B smoke validation. The result table records
completed measured runs; the cleaned repository entry must be revalidated on GPU
before claiming a new full reproduction. No weights, private paths or raw
experiment-control logs are distributed with the code.
