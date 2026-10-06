"""Evaluate all tasks with the recorded canonical-answer beam protocol."""
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import GenerationConfig
from protocol import (ANSWER_OPTIONS, allowed_next_answer_tokens, canonical_answer_responses,
                      extract_answer, generate_evaluation_prompt)

@torch.no_grad()
def evaluate_validation_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    weighted_loss = 0.0
    examples = 0
    for batch in loader:
        batch_size = int(batch["input_ids"].shape[0])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=batch["input_ids"].to(device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                labels=batch["labels"].to(device, non_blocking=True),
                use_cache=False,
                return_dict=True,
            )
        if not torch.isfinite(output.loss):
            raise RuntimeError("non-finite validation loss")
        weighted_loss += float(output.loss) * batch_size
        examples += batch_size
    model.train()
    return weighted_loss / examples


@torch.no_grad()
def evaluate_task(
    model: nn.Module,
    tokenizer,
    task: str,
    rows: list[dict],
    output_path: Path,
    device: torch.device,
    generation_config: GenerationConfig,
) -> dict:
    model.eval()
    correct = 0
    invalid = 0
    started = time.time()
    with output_path.open("w", encoding="utf-8") as output_handle:
        for index, row in enumerate(rows):
            prompt = generate_evaluation_prompt(row["instruction"], row.get("input"))
            encoded = tokenizer(prompt, return_tensors="pt", padding=True)
            prompt_length = int(encoded["input_ids"].shape[1])
            answer_sequences = tuple(
                tuple(tokenizer(response, add_special_tokens=False).input_ids)
                for response in canonical_answer_responses(task)
            )
            if any(not sequence for sequence in answer_sequences):
                raise RuntimeError(f"empty constrained answer sequence for {task}")
            if len(set(answer_sequences)) != len(answer_sequences):
                raise RuntimeError(f"non-unique constrained answer sequence for {task}")
            required_new_tokens = max(map(len, answer_sequences))
            if generation_config.max_new_tokens < required_new_tokens:
                raise ValueError(
                    f"max_new_tokens={generation_config.max_new_tokens} is too small "
                    f"for {task}; need at least {required_new_tokens}"
                )

            def prefix_allowed_tokens_fn(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
                suffix = tuple(int(token) for token in input_ids[prompt_length:].tolist())
                return allowed_next_answer_tokens(
                    answer_sequences, suffix, tokenizer.eos_token_id
                )

            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=generation_config.max_new_tokens,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                )
            response = tokenizer.decode(
                generated.sequences[0, prompt_length:], skip_special_tokens=True
            ).strip()
            prediction = extract_answer(task, response)
            if prediction not in ANSWER_OPTIONS[task]:
                raise RuntimeError(
                    f"constrained decoding produced an invalid {task} answer: {response!r}"
                )
            label = row["answer"]
            flag = prediction == label
            correct += int(flag)
            invalid += int(not prediction)
            output_handle.write(
                json.dumps(
                    {
                        "index": index,
                        "answer": label,
                        "prediction": prediction,
                        "correct": flag,
                        "response": response,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (index + 1) % 250 == 0:
                output_handle.flush()
                print(
                    f"eval task={task} {index + 1}/{len(rows)} "
                    f"accuracy={correct / (index + 1):.6f}",
                    flush=True,
                )
    return {
        "task": task,
        "accuracy": correct / len(rows),
        "correct": correct,
        "total": len(rows),
        "invalid": invalid,
        "decoding": "canonical_legal_answer_prefix_constraint",
        "elapsed_seconds": time.time() - started,
        "predictions_path": output_path.name,
    }
