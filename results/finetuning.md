# GLUE Fine-Tuning

Llama-3.1-8B Base, original Dense vs. MoC K25. K25 retains 3,584 of 14,336
FFN channels per token. One C4 reconstruction serves all eight tasks.

[Quality](#development-set-quality) | [QQP case study](#qqp-case-study) |
[Sequence length](#sequence-length) | [All task resources](#resources-by-task) |
[Learning-rate grid](#learning-rate-sensitivity) | [Settings and metrics](#experiment-settings)

## Development-Set Quality

**Table 1. Eight-task GLUE development scores (seed 42).** Both methods use the same
four-rate grid. Bold marks the higher score; differences are K25 minus Dense,
computed before rounding.

| Method | CoLA | SST-2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE | Avg. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | **70.85** | **97.02** | 91.79 | **92.79** | **91.33** | **91.98** | **95.92** | 89.89 | **90.20** |
| MoC K25 | 67.20 | 96.90 | **91.93** | 92.73 | 91.01 | 91.26 | 95.55 | **90.25** | 89.60 |
| Difference (K25 - Dense) | -3.65 | -0.11 | +0.13 | -0.06 | -0.32 | -0.73 | -0.37 | +0.36 | -0.59 |

Avg. is the unweighted mean of the eight task scores.

## Fine-Tuning Resources

**Expected savings in activations saved for backward.** Let $d=4096$ be the
hidden width, $h=14336$ the FFN width, $k=3584$ the selected width, and $r=16$
the LoRA rank. Per layer and token position, Dense stores two-byte BF16 states
and one-byte dropout masks. K25 reduces four BF16 states from width $h$ to $k$
and adds one width-$k$ vector of two-byte indices. In bytes:

$$
\begin{aligned}
M_{\mathrm{Dense}} &= 2(4h+2d+3r)+(2d+h) = 153{,}696,\\
\Delta M &= 2\cdot4(h-k)-2k = 78{,}848,\\
\frac{\Delta M}{M_{\mathrm{Dense}}} &= \frac{78{,}848}{153{,}696} \approx 51.3\%.
\end{aligned}
$$

The layer count and number of token positions cancel in the ratio. Thus K25
saves **51.3% of FFN saved activation memory** for this configuration. FFNs
account for about 60.46% of all Dense saved activation memory at batch 16;
using that share gives **about 31% model-wide** ($0.6046 \times 51.3\%$).
This estimates saved activations, not the peak memory of a fine-tuning step.

### QQP Case Study

**Table 2. QQP at three per-GPU batch sizes.** QQP had the largest peak
memory reduction at batch 16, where all eight tasks have a Dense comparison.
Peaks are per GPU; throughput is relative to Dense. All task results below
refer to 40 ordinary timed steps on four A100 40 GB GPUs.

| Batch/GPU | Dense peak | K25 peak | Memory saved | Throughput |
| :--- | ---: | ---: | ---: | ---: |
| 8 | 21.74 GiB | 19.60 GiB | 9.82% | 91.01% |
| 16 | 29.81 GiB | 25.19 GiB | 15.51% | 95.09% |
| 32 | OOM | 35.48 GiB | &mdash; | &mdash; |

### Sequence Length

**Table 3. Peak GPU memory and throughput on fixed-length, unpadded C4 prefixes (batch 8 per GPU).**

| Seq. length | Dense peak | K25 peak | Memory saved | Throughput |
| :--- | ---: | ---: | ---: | ---: |
| 32 | 16.80 GiB | 16.23 GiB | 3.38% | 87.21% |
| 128 | 22.72 GiB | 20.44 GiB | 10.05% | 95.33% |
| 256 | 30.69 GiB | 26.05 GiB | 15.10% | 95.25% |
| 512 | OOM | 37.36 GiB | &mdash; | &mdash; |

From 32 to 256 tokens per sequence, the memory saving grows from **0.57 to
4.63 GiB per GPU**. From 128 to 256 tokens, K25 retains about **95%** of Dense
throughput at both lengths. At 512 tokens, Dense ran out of memory in both
trials. K25 completed both at **37.36 GiB** per GPU and averaged **13,172
effective tokens/s** across the four GPUs.

## Resources by Task

**Table 4. All eight GLUE tasks, grouped by per-GPU batch size.**
The columns have the same units and baseline as Table 2. OOM denotes an
out-of-memory failure before completion of the timed steps.

### Batch 8

| Task | Dense peak | K25 peak | Memory saved | Throughput |
| :--- | ---: | ---: | ---: | ---: |
| CoLA | 16.58 GiB | 16.05 GiB | 3.20% | 90.36% |
| SST-2 | 18.27 GiB | 17.23 GiB | 5.69% | 90.06% |
| MRPC | 21.13 GiB | 19.23 GiB | 9.00% | 93.76% |
| STS-B | 20.61 GiB | 18.85 GiB | 8.54% | 91.33% |
| QQP | 21.74 GiB | 19.60 GiB | 9.82% | 91.01% |
| MNLI | 22.36 GiB | 20.08 GiB | 10.21% | 92.97% |
| QNLI | 22.37 GiB | 20.06 GiB | 10.32% | 95.09% |
| RTE | 22.58 GiB | 20.28 GiB | 10.21% | 96.14% |

### Batch 16

| Task | Dense peak | K25 peak | Memory saved | Throughput |
| :--- | ---: | ---: | ---: | ---: |
| CoLA | 20.37 GiB | 18.65 GiB | 8.42% | 89.65% |
| SST-2 | 22.01 GiB | 19.79 GiB | 10.10% | 94.26% |
| MRPC | 27.82 GiB | 23.88 GiB | 14.16% | 95.73% |
| STS-B | 27.73 GiB | 23.76 GiB | 14.31% | 93.86% |
| QQP | 29.81 GiB | 25.19 GiB | 15.51% | 95.09% |
| MNLI | 29.94 GiB | 25.30 GiB | 15.48% | 95.01% |
| QNLI | 30.03 GiB | 25.41 GiB | 15.40% | 94.13% |
| RTE | 30.29 GiB | 25.66 GiB | 15.27% | 94.64% |

### Batch 32

| Task | Dense peak | K25 peak | Memory saved | Throughput |
| :--- | ---: | ---: | ---: | ---: |
| CoLA | 25.86 GiB | 22.44 GiB | 13.24% | 95.30% |
| SST-2 | 29.29 GiB | 24.82 GiB | 15.26% | 95.01% |
| MRPC | OOM | 33.01 GiB | &mdash; | &mdash; |
| STS-B | OOM | 34.68 GiB | &mdash; | &mdash; |
| QQP | OOM | 35.48 GiB | &mdash; | &mdash; |
| MNLI | OOM | 35.71 GiB | &mdash; | &mdash; |
| QNLI | OOM | 35.99 GiB | &mdash; | &mdash; |
| RTE | OOM | 36.44 GiB | &mdash; | &mdash; |

At batch 32, K25 completed the timed steps on **8/8** tasks; Dense completed
them on **2/8**. Dashes indicate that Dense ran out of memory, so no relative
comparison is reported.

## Learning-Rate Sensitivity

**Table 5. Eight development-set scores at each learning rate.** Bold marks
the maximum for each method and task, including exact ties.

### Original Dense

| Learning rate | CoLA | SST-2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2e-5 | 65.85 | 96.90 | 89.59 | 89.35 | 90.65 | **91.98** | 95.59 | 75.45 |
| 5e-5 | 67.06 | **97.02** | 91.56 | 91.49 | 91.25 | 91.85 | 95.66 | 85.92 |
| 1e-4 | 68.14 | 96.67 | 91.78 | 92.05 | **91.33** | 91.33 | **95.92** | 87.73 |
| 3e-4 | **70.85** | 96.44 | **91.79** | **92.79** | 84.61 | 38.72 | 61.18 | **89.89** |
| Best per task | 70.85 | 97.02 | 91.79 | 92.79 | 91.33 | 91.98 | 95.92 | 89.89 |

### Task-Specific Reconstruction

| Learning rate | CoLA | SST-2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2e-5 | 65.72 | **96.33** | 90.95 | 89.57 | 90.35 | **91.20** | 95.24 | 75.45 |
| 5e-5 | 65.14 | **96.33** | **92.20** | 91.12 | 90.98 | 91.05 | **95.53** | 85.56 |
| 1e-4 | 68.21 | 96.22 | 92.02 | 91.63 | **91.09** | 90.85 | 95.24 | 87.00 |
| 3e-4 | **69.03** | 95.87 | 90.78 | **92.36** | 67.87 | 88.96 | 62.27 | **88.45** |
| Best per task | 69.03 | 96.33 | 92.20 | 92.36 | 91.09 | 91.20 | 95.53 | 88.45 |

### Shared C4 Reconstruction

| Learning rate | CoLA | SST-2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2e-5 | 66.68 | **96.90** | 89.32 | 90.09 | 90.19 | 91.25 | 95.42 | 76.90 |
| 5e-5 | **67.20** | 96.44 | 91.04 | 90.74 | 90.69 | **91.26** | 95.41 | 84.84 |
| 1e-4 | 66.34 | 96.33 | **91.93** | 91.77 | **91.01** | 90.72 | **95.55** | 85.92 |
| 3e-4 | 64.36 | 95.18 | 91.01 | **92.73** | 73.28 | 35.33 | 93.57 | **90.25** |
| Best per task | 67.20 | 96.90 | 91.93 | 92.73 | 91.01 | 91.26 | 95.55 | 90.25 |

Best-per-task macro means are **90.20** (Dense), **89.53** (task-specific K25)
and **89.60** (shared C4 K25).

## Experiment Settings

### Model and Data

The Dense baseline is Llama-3.1-8B Base. MoC K25 keeps the top 3,584 of 14,336
FFN channels per token, selected by the raw gate in each of the 32 layers.
A single reconstruction on 16,384 C4 windows of 128 tokens, with 8,192 fitting
steps per layer, provides the starting weights for all eight GLUE tasks. No
task-specific reconstruction or additional language-model fine-tuning is used
for this route. The learning-rate grid also includes task-specific reconstruction as a baseline.

Each task starts a fresh supervised LoRA adapter and task head. LoRA uses rank
16, alpha 32 and dropout 0.1 on the four attention and three FFN projections.
The eight tasks are fine-tuned independently. Model weights and task examples
are not distributed with these results.

### Evaluation

Dense and both K25 routes use the same four learning rates: `2e-5`, `5e-5`,
`1e-4`, and `3e-4`. For each learning rate, Trainer selects the epoch with the
highest development score, taking the earliest epoch in an exact tie. We
independently reload each selected checkpoint, score it on the development set,
and choose the highest-scoring learning rate for each model and task. The mean
weights all eight tasks equally.

| Task | Development metric |
| :--- | :--- |
| CoLA | Matthews correlation |
| SST-2, QNLI, RTE | Accuracy |
| MRPC, QQP | Mean of accuracy and F1 |
| STS-B | Mean of Pearson and Spearman correlations |
| MNLI | Mean of matched and mismatched accuracy |
