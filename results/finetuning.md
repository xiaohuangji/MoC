# Commonsense Fine-Tuning

Llama-3.1-8B Base, A100-PCIE-40GB, CUDA 12.8, PyTorch 2.8.0, Transformers 5.10.2,
PEFT 0.20.0. The comparison includes **Dense**, **Direct MoC**, and
**Reconstructed MoC**. Both MoC routes use fixed raw-gate K2048/14336 (14.2857%)
in all 32 FFNs. See [the recipe and commands](../docs/finetuning.md).

- **Dense:** continue LoRA fine-tuning without changing the FFNs.
- **Direct MoC:** retain the Dense weights, switch the FFNs to fixed-K MoC, and
  immediately begin LoRA fine-tuning, without reconstruction.
- **Reconstructed MoC:** switch to fixed-K MoC, sequentially fit its FFNs to the
  original Dense block outputs, then perform LoRA fine-tuning with compact
  activation storage.

## Quality

All three routes start from the **same already eight-task-SFT Dense rank32 adapter**,
then receive 10240 downstream updates (163840 example presentations), seed42 and
logical batch16. They use the same data order, answer-weight16 objective and
rank128 LoRA on q/k/v/gate/up/down. Only Reconstructed MoC requires the additional
reconstruction phase described below. Neither MoC route is pretrained from
scratch; neither uses K annealing or Dense output fusion.

**Table 1. Downstream accuracy (%), evaluated at the fixed final step.**

| Task | Examples | Dense | Direct MoC | Reconstructed MoC |
| --- | ---: | ---: | ---: | ---: |
| BoolQ | 3270 | 72.54 | 66.94 | 69.27 |
| PIQA | 1838 | 84.93 | 80.20 | 81.88 |
| SocialIQA | 1954 | 78.76 | 76.15 | 77.58 |
| HellaSwag | 10042 | 91.23 | 75.97 | 86.26 |
| WinoGrande | 1267 | 82.87 | 76.95 | 80.11 |
| ARC-Challenge | 1172 | 70.73 | 59.98 | 64.16 |
| ARC-Easy | 2376 | 84.13 | 74.54 | 78.70 |
| OpenBookQA | 500 | 81.20 | 74.40 | 78.20 |
| **Macro average** | **22419** | **80.80** | **73.14** | **77.02** |

Reconstructed MoC improves on Direct MoC by **3.88 percentage points** and is
**3.78 percentage points below Dense**. The downstream update budget is matched,
but reconstruction adds compute, so this is not an equal-total-cost gain.
Generation uses beam4, batch1,
max-new-tokens32 and canonical-answer constraints; invalid answers are zero.
Macro average is unweighted over tasks. This is an adapted recipe, not an
official leaderboard result. Training retains right truncation at 256.
Dense and Direct MoC use block checkpointing/micro16; Reconstructed MoC uses
attention checkpointing/micro8, accumulated to 16. Different micro batches and
implementations can change dropout and floating-point trajectories, so this is
not an identical-implementation ablation of reconstruction alone. These are
single-seed development results, not an independent blind test.

This quality table records the completed full training runs. The subsequent
implementation optimizations in Table 2 passed short exact-state checks but have
not been used for a new full eight-task accuracy run.

## Training Resources

**Table 2. GPU memory by phase and downstream training throughput.**

| Phase | Peak memory (GiB) | Throughput (token/s) | Memory saved | Throughput retained |
| :--- | ---: | ---: | ---: | ---: |
| Dense fine-tuning | 36.36 | **1339.33** | 0.00% | 100.00% |
| FFN reconstruction | 18.24 | n/a | n/a | n/a |
| MoC fine-tuning | **26.81** | 1309.91 | **26.28%** | 97.80% |

*All phases use A100-PCIE-40GB; peaks are allocated GPU memory.
The two fine-tuning rows share micro batch8,
logical batch16, sequence length256 and attention-only activation checkpointing;
MoC uses reconstructed weights. Bold marks the better fine-tuning values.
Reconstruction is sequential single-FFN fitting, not full-model fine-tuning;
its peak includes model loading, fresh teacher capture and all fitted layers.
Its token/s and relative savings are not comparable (`n/a`).*

Reconstruction costs about **2.81 GPU-hours**, separate from downstream training.
The **26.28%** saving applies only to the matched fine-tuning rows, not to the
complete pipeline. Sequential phase peaks are not additive; a full-pipeline
claim must cover all stages, including loading and evaluation. Direct MoC has
not been separately measured under this resource setup.

Micro batch8 uses two accumulation passes per update. The two fine-tuning rows use
the same model dimensions, BF16 base and compute / FP32 LoRA masters, rank128
LoRA targets, chunked cross-entropy,
optimizer settings and identical prefetched data. Reconstructed MoC uses the same
fixed-K2048 method and compact activation storage throughout. Attention-only
checkpointing is not the paper's FFN element-wise GCP, which retains gate/up
activations and recomputes SiLU/products; this table does not measure that pairing.

For the fine-tuning rows, each of three runs per method uses six warmup updates
to initialize optimizer states before three consecutive 16-update measurement
windows. Peak memory is the
maximum **allocated GiB** across all runs and windows, including weights, optimizer
states, gradients, activations and temporary tensors. Throughput is the mean of
the three runs' median nonpadding token/s. These
are training-loop measurements, not isolated kernel timings; loading, correctness
prechecks, reconstruction, checkpoint saves and task evaluation are excluded.
Peak is neither pure FFN activation storage, reserved memory, nor `nvidia-smi`
process memory.

At micro batch8, Reconstructed MoC reduces training-loop peak memory by **26.28%** and
retains **97.80%** of Dense throughput. Dense remains the original implementation;
only Reconstructed MoC uses shared input casts, compact index copies and
independent down-projection gradient edges. General LoRA implementation improvements contribute to this
comparison and are not exclusively algorithmic MoC gains. The updated
implementation has not been remeasured at micro batches 4 or 16.
These findings apply to attention-only checkpointing and do not establish an
advantage over every checkpoint policy. Full-block checkpointing can eliminate
the additional memory saving.

Memory saving is `100 * (1 - MoC_peak / Dense_peak)`; throughput retained is
`100 * MoC_token_per_s / Dense_token_per_s`, with MoC referring to the reconstructed
route and ratios computed before rounding. Both statistics include all three
runs, not the fastest MoC and slowest Dense runs.

Dense and Reconstructed MoC were measured on the same GPU with the same data
and configuration, but in **separate measurement batches**, not newly interleaved
paired runs.
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

**Table 3. Actual costs of the completed quality runs, including reconstruction (GPU-hours).**

| Method | Downstream updates | Reconstruction | Downstream process | Total |
| --- | ---: | ---: | ---: | ---: |
| Dense | 10240 | 0.0000 | 8.2250 | 8.2250 |
| Direct MoC | 10240 | 0.0000 | 8.7481 | 8.7481 |
| Reconstructed MoC | 10240 | 2.8070 | 7.9089 | 10.7159 |

Downstream process time includes loading, saves, training and all eight evaluations
under the quality-run settings. Reconstruction includes fresh teacher capture;
total cost is their sum. All rows exclude the common source Dense SFT.
Reconstruction uses 4096 single-layer fitting updates per FFN, which are not
full-model updates and are not added to the downstream update count.
These times are not the steady-state training windows in Table 2: the
quality-run Dense and Direct MoC use block/micro16, while Reconstructed MoC uses
attention/micro8. This records the actual complete runs, not a matched-GCP/micro
resource comparison.
The full-process figures retain the implementation used for those completed
quality runs; they are not extrapolated from the later optimized throughput.

Direct MoC has no reconstruction phase; its incremental full-process cost is
the **8.7481 GPU-hours** shown above. Including fresh reconstruction,
Reconstructed MoC costs **10.7159 GPU-hours**, about **30.28%
more** than same-step Dense. A separate near-equal-total-time Dense control costs
**10.7292 GPU-hours**, completes 17918 updates and scores **80.67%**, versus
Reconstructed MoC's **77.02%**. The measured budget differs by about 0.124%; this control uses wall-time
LR decay rather than the fixed-step schedule.

The shared earlier Dense SFT cost is excluded from all these incremental figures;
its available log gives a lower bound of **14.21 GPU-hours**. This is not a from-Base
training-cost comparison and not a claim of overall time saving.

## Reproducibility

For each of the three routes, all 22419 predictions were checked for row order,
labels and correct counts. The shared training-order SHA256 is
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
