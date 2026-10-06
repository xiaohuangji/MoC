"""Commonsense task formats and constrained-answer evaluation helpers.

Prompt templates adapted from Apache-2.0 LLM-Adapters; see THIRD_PARTY_NOTICES.md.
Modified: task validation and constrained generation are specific to this benchmark.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


NVIDIA_DORA_COMMIT = "b293c179558d4b1e0b1ec1426e30c1f59ce72f80"
LLM_ADAPTERS_SOURCE_COMMIT = "816657208af4db747803f87ba40a4c71383fed7a"
TRAINING_REFERENCE = (
    "https://github.com/NVlabs/DoRA/blob/"
    f"{NVIDIA_DORA_COMMIT}/commonsense_reasoning/finetune.py"
)
EVALUATION_REFERENCE = (
    "https://github.com/NVlabs/DoRA/blob/"
    f"{NVIDIA_DORA_COMMIT}/commonsense_reasoning/commonsense_evaluate.py"
)

TASKS = (
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
)

ANSWER_OPTIONS = {
    "boolq": ("true", "false"),
    "piqa": ("solution1", "solution2"),
    "social_i_qa": ("answer1", "answer2", "answer3", "answer4", "answer5"),
    "hellaswag": ("ending1", "ending2", "ending3", "ending4"),
    "winogrande": ("option1", "option2"),
    "ARC-Challenge": ("answer1", "answer2", "answer3", "answer4", "answer5"),
    "ARC-Easy": ("answer1", "answer2", "answer3", "answer4", "answer5"),
    "openbookqa": ("answer1", "answer2", "answer3", "answer4", "answer5"),
}
ANSWER_PATTERNS = {
    task: re.compile(
        r"(?<![a-z0-9_])(?:" + "|".join(map(re.escape, options)) + r")(?![a-z0-9_])",
        flags=re.IGNORECASE,
    )
    for task, options in ANSWER_OPTIONS.items()
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_list(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"expected a non-empty JSON list: {path}")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"expected JSON objects in {path}")
    return payload


def validate_training_rows(rows: list[dict]) -> None:
    required = {"instruction", "input", "output", "answer"}
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"training row {index} is missing {sorted(missing)}")
        if not isinstance(row["instruction"], str) or not row["instruction"]:
            raise ValueError(f"training row {index} has no instruction")
        if not isinstance(row["input"], str):
            raise ValueError(f"training row {index} has a non-string input")
        if not isinstance(row["output"], str) or not row["output"]:
            raise ValueError(f"training row {index} has no output")


def validate_evaluation_rows(task: str, rows: list[dict]) -> None:
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    for index, row in enumerate(rows):
        if not isinstance(row.get("instruction"), str) or not row["instruction"]:
            raise ValueError(f"{task} row {index} has no instruction")
        if "answer" not in row:
            raise ValueError(f"{task} row {index} has no answer")


def generate_training_prompt(row: dict) -> str:
    """Preserve the LLM-Adapters training prompt, including whitespace."""
    if row["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n"
            f"                ### Instruction:\n                {row['instruction']}\n                \n"
            f"                ### Input:\n                {row['input']}\n                \n"
            f"                ### Response:\n                {row['output']}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.  \n\n"
        f"                ### Instruction:\n                {row['instruction']}\n                \n"
        f"                ### Response:\n                {row['output']}"
    )


def generate_evaluation_prompt(instruction: str, input_text: str | None = None) -> str:
    """Preserve the LLM-Adapters evaluation prompt."""
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input_text}

                ### Response:
                """
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n"
        f"                ### Instruction:\n                {instruction}\n\n"
        "                ### Response:\n                "
    )


def extract_answer(task: str, response: str) -> str:
    pattern = ANSWER_PATTERNS[task]
    matches = pattern.findall(response.strip())
    return matches[0].lower() if matches else ""


def canonical_answer_responses(task: str) -> tuple[str, ...]:
    return tuple(f"the correct answer is {answer}" for answer in ANSWER_OPTIONS[task])


def allowed_next_answer_tokens(
    answer_sequences: tuple[tuple[int, ...], ...],
    generated_suffix: tuple[int, ...],
    eos_token_id: int,
) -> list[int]:
    allowed = set()
    for sequence in answer_sequences:
        if sequence[: len(generated_suffix)] != generated_suffix:
            continue
        if len(generated_suffix) == len(sequence):
            allowed.add(int(eos_token_id))
        else:
            allowed.add(int(sequence[len(generated_suffix)]))
    return sorted(allowed) if allowed else [int(eos_token_id)]
