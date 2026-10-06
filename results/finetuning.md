# Commonsense Fine-Tuning

Llama-3.1-8B Base, A100-PCIE-40GB, CUDA 12.8, PyTorch 2.8.0, Transformers 5.10.2,
PEFT 0.20.0. **MoC** denotes the complete method: sequential FFN reconstruction,
fixed raw-gate K2048/14336 (14.2857%) in all 32 FFNs, compact activation storage,
and LoRA fine-tuning. See [the recipe and commands](../docs/finetuning.md).

## Quality

Both methods start from the **same already eight-task-SFT Dense rank32 adapter**,
then receive 10240 downstream updates (163840 example presentations), seed42 and
logical batch16. They use the same data order, answer-weight16 objective and
rank128 LoRA on q/k/v/gate/up/down. MoC additionally requires the reconstruction
phase described below. There is no K annealing or Dense output fusion.

**Table 1. Downstream accuracy (%), evaluated at the fixed final step.**

| Task | Examples | Dense | MoC |
| --- | ---: | ---: | ---: |
| BoolQ | 3270 | 72.54 | 69.27 |
| PIQA | 1838 | 84.93 | 81.88 |
| SocialIQA | 1954 | 78.76 | 77.58 |
| HellaSwag | 10042 | 91.23 | 86.26 |
| WinoGrande | 1267 | 82.87 | 80.11 |
| ARC-Challenge | 1172 | 70.73 | 64.16 |
| ARC-Easy | 2376 | 84.13 | 78.70 |
| OpenBookQA | 500 | 81.20 | 78.20 |
| **Macro average** | **22419** | **80.80** | **77.02** |

MoC is **3.78 percentage points below Dense**. Generation uses beam4, batch1,
max-new-tokens32 and canonical-answer constraints; invalid answers are zero.
Macro average is unweighted over tasks. This is an adapted recipe, not an
official leaderboard result. Training retains right truncation at 256.
Dense uses block checkpointing/micro16; MoC uses attention checkpointing/micro8,
accumulated to 16. These are single-seed development results, not an independent
blind test.

This quality table records the completed full training runs. The subsequent
implementation optimizations in Table 2 passed short exact-state checks but have
not been used for a new full eight-task accuracy run.

## Training Resources

**Table 2. Training memory and throughput.**

| Method | Peak memory (GiB) | Throughput (token/s) | Memory saved | Throughput retained |
| :--- | ---: | ---: | ---: | ---: |
| Dense | 36.36 | **1339.33** | 0.00% | 100.00% |
| MoC | **26.81** | 1309.91 | 26.28% | 97.80% |

*Shared setup: A100-PCIE-40GB, micro batch8, logical batch16, sequence length256,
attention-only activation checkpointing. Lower memory and higher throughput are
better; bold marks the best absolute value in each metric. Relative columns use
Dense as the reference.*

Micro batch8 uses two accumulation passes per update. Both use the same model dimensions, BF16 base
and compute / FP32 LoRA masters, rank128 LoRA targets, chunked cross-entropy,
optimizer settings and identical prefetched data. MoC uses the same reconstructed,
fixed-K2048 method and compact activation storage throughout. Attention-only
checkpointing is not the paper's FFN element-wise GCP, which retains gate/up
activations and recomputes SiLU/products; this table does not measure that pairing.

Each of three runs per method uses six warmup updates to initialize optimizer
states before three consecutive 16-update measurement windows. Peak memory is the
maximum **allocated GiB** across all runs and windows, including weights, optimizer
states, gradients, activations and temporary tensors. Throughput is the mean of
the three runs' median nonpadding token/s. These
are training-loop measurements, not isolated kernel timings; loading, correctness
prechecks, reconstruction, checkpoint saves and task evaluation are excluded.
Peak is neither pure FFN activation storage, reserved memory, nor `nvidia-smi`
process memory.

At micro batch8, MoC reduces total peak memory by **26.28%** and retains
**97.80%** of Dense throughput. Dense remains the original implementation; only
MoC uses shared input casts, compact index copies and independent down-projection
gradient edges. General LoRA implementation improvements contribute to this
comparison and are not exclusively algorithmic MoC gains. The updated
implementation has not been remeasured at micro batches 4 or 16.
These findings apply to attention-only checkpointing and do not establish an
advantage over every checkpoint policy. Full-block checkpointing can eliminate
the additional memory saving.

Memory saving is `100 * (1 - MoC_peak / Dense_peak)`; throughput retained is
`100 * MoC_token_per_s / Dense_token_per_s`, computed before rounding. Both
statistics include all three runs, not the fastest MoC and slowest Dense runs.

Dense and MoC were measured on the same GPU with the same data and configuration,
but in **separate measurement batches**, not newly interleaved paired runs.
Cross-batch environmental variation is not ruled out. Repeated timing runs are
not independent model seeds; no confidence interval or significance claim is
inferred. These 54-update resource probes do not remeasure task accuracy
for every configuration. Table 1 retains its stated complete-training settings;
it is not an identical-micro-batch quality ablation.

## Reconstruction And Total Cost

Reconstruction uses training-only data: 8192 rows, up to 262144 token positions
per layer, and 4096 fitting steps per FFN. Its complete process requires
**18.24 GiB** GPU peak and about **2.81 GPU-hours**, including fresh teacher
capture. Teacher cache files occupy 67.8 GiB; main CPU arrays occupy about 64 GiB
plus additional buffers; exported base FFN weights occupy about 10.5 GiB.

**Table 3. Full-process costs of the completed quality runs (excluding reconstruction).**

| Downstream method | Updates | Full process GPU-hours | Full process peak GiB |
| --- | ---: | ---: | ---: |
| Dense | 10240 | 8.2250 | 24.29 |
| MoC | 10240 | 7.9089 | 27.80 |

These process figures include loading, saves, training and all eight evaluations
under the quality-run settings. They exclude reconstruction and the common source
Dense SFT. They are not the steady-state training windows in Table 2: the
quality-run Dense uses block/micro16 and MoC uses attention/micro8. This is a
record of the actual complete runs, not a matched-GCP/micro resource comparison.
The full-process figures retain the implementation used for those completed
quality runs; they are not extrapolated from the later optimized throughput.

Including fresh reconstruction, MoC costs **10.7159 GPU-hours**, about **30.28%
more** than same-step Dense. A separate near-equal-total-time Dense control costs
**10.7292 GPU-hours**, completes 17918 updates and scores **80.67%**, versus MoC's
**77.02%**. The measured budget differs by about 0.124%; this control uses wall-time
LR decay rather than the fixed-step schedule.

The shared earlier Dense SFT cost is excluded from all these incremental figures;
its available log gives a lower bound of **14.21 GPU-hours**. This is not a from-Base
training-cost comparison and not a claim of overall time saving.

## Reproducibility

Both methods' 22419 predictions were checked for row order, labels and correct
counts. The shared training-order SHA256 is
`b2256c4d7b88d88e3c32f7cca6e5a5a2768d61d86cb66316bc5135691d5977bf`.
The [data manifest](../benchmarks/finetuning/data_manifest.json) pins the source
files. The common source adapter is an explicit, currently unbundled prerequisite;
its SHA256 and model reload requirements are documented in the recipe.

The selected optimized implementation passed CUDA operator checks and four-update
8B gradient/parameter/Adam/RNG and fresh-reload bitwise checks. This bounds the
verified trajectory; it is not a guarantee about arbitrary long training runs.
The packaged entry passed local regression and source-consistency checks but has
not yet been rerun on an 8B GPU job. These tables record completed experiments,
not a new execution of that entry.
