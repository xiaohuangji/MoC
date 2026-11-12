# Zero-Shot Evaluation

## Scope

Table 3 evaluates the C4-pretrained 1B Dense, MoC, and MoC 2:8 checkpoints without downstream fine-tuning.

## Evaluation Setup

- Model: 24 layers, hidden size 2048, FFN size 5461, 32 attention heads; MoC uses K=1024.
- Hardware: NVIDIA A800 80GB PCIe.
- Software: PyTorch 2.8.0+cu128, Transformers 4.55.4, lm-evaluation-harness 0.4.9.2.
- Precision: BF16 parameters and RoPE, TF32 disabled.
- Protocol: zero-shot, context length 256, 20,245 examples and 79,914 candidate requests per model.
- Execution: Dense and MoC use reference FFNs; MoC 2:8 uses the native grouped kernel.

All models use identical tokenized inputs; 6,225 requests are truncated to the context limit. MMLU aggregates all subject examples by sample count. Scores were verified offline from saved per-example likelihoods.

## Results

| Model | MMLU | ARC-Easy | ARC-Challenge | PIQA | TruthfulQA | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | **23.02** | 46.59 | **26.28** | 70.62 | **38.32** | **40.97** |
| MoC | 22.98 | 46.00 | 26.11 | 71.16 | 35.69 | 40.39 |
| MoC 2:8 | 22.92 | **46.63** | 24.74 | **71.38** | 37.80 | 40.70 |

<sub>Scores are percentages. ARC-Easy, ARC-Challenge, and PIQA use acc_norm; MMLU uses accuracy and TruthfulQA uses MC2. Avg. is the unweighted mean of the five unrounded scores.</sub>
