"""Configuration, provenance and output helpers shared by the fine-tuning entries."""
import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import yaml

from protocol import file_sha256

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/llama31_8b_commonsense.yaml"
TARGETS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj")
MODEL_IDENTITY = dict(model_type="llama", hidden_size=4096, intermediate_size=14336,
                      num_hidden_layers=32, vocab_size=128256)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path=CONFIG):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {"model_dir", "data_dir", "cache_dir", "source_adapter", "source_adapter_sha256",
                "seed", "k", "training", "data", "evaluation", "reconstruction"}
    if not isinstance(cfg, dict) or set(cfg) != expected:
        raise ValueError("Unexpected fine-tuning configuration fields")
    for key in ("model_dir", "data_dir", "cache_dir", "source_adapter"):
        value = Path(cfg[key]).expanduser()
        cfg[key] = str(value if value.is_absolute() else ROOT / value)
    if cfg["k"] != 2048 or cfg["training"]["batch_size"] != 16:
        raise ValueError("This recipe uses fixed K2048 and logical batch16")
    if cfg["data"]["max_length"] != 256 or cfg["data"]["validation_size"] != 120:
        raise ValueError("This recipe uses length256 and 120 training-only validation rows")
    if cfg["training"]["answer_weight"] != 16:
        raise ValueError("This recipe uses answer weight16")
    for section in ("training", "reconstruction"):
        rate = cfg[section].get("learning_rate")
        if "lr" in cfg[section] or isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"{section}.learning_rate must be a finite positive number; 'lr' is not supported")
    for key in ("steps", "validation_every", "logging_every", "save_every"):
        if cfg["training"][key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if not 0 <= cfg["training"]["warmup_steps"] < cfg["training"]["steps"]:
        raise ValueError("Invalid warmup or scheduler horizon")
    if not re.fullmatch(r"[0-9a-f]{64}", cfg["source_adapter_sha256"]):
        raise ValueError("Set the SHA256 of the common Dense SFT adapter explicitly")
    return cfg


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", type=Path, default=CONFIG)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def cuda_device(seed):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA device is required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    return device


def check_source(cfg):
    directory = Path(cfg["source_adapter"])
    digest = file_sha256(directory / "adapter_model.safetensors")
    if digest != cfg["source_adapter_sha256"]:
        raise ValueError("Source adapter SHA256 mismatch; do not silently substitute a different start")
    adapter = read_json(directory / "adapter_config.json")
    if (set(adapter["target_modules"]) != set(TARGETS) or adapter["r"] != 32
            or adapter["lora_alpha"] != 64 or adapter["lora_dropout"] != .05):
        raise ValueError("Expected the common rank32/alpha64 Dense SFT adapter with gate LoRA")
    for name in ("use_dora", "use_rslora", "rank_pattern", "alpha_pattern"):
        if adapter.get(name):
            raise ValueError(f"Unsupported source adapter option: {name}")
    return digest


def source_hashes():
    return {p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")}
