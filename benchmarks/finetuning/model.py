"""Llama/PEFT loading and fixed raw-gate MoC conversion; no Dense blending."""
import types
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from common import MODEL_IDENTITY, check_source, read_json, write_json
from lora import expand_rank, optimize_moc_lora
from overlay import load_overlay
from protocol import file_sha256
from selected_lora import install, uninstall


class FixedKMLP(nn.Module):
    def __init__(self, original, k):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = original.gate_proj, original.up_proj, original.down_proj
        self.act_fn = original.act_fn
        self.intermediate_size = original.gate_proj.out_features
        if not 0 < k <= self.intermediate_size <= 32768:
            raise ValueError("Invalid K or width for compact int16 indices")
        self.current_k = self.target_k = k
        self.selector = "raw_gate"

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        if self.current_k == self.intermediate_size:
            return self.down_proj(self.act_fn(gate) * up)
        with torch.no_grad():
            indices = torch.topk(gate, self.current_k, dim=-1, largest=True, sorted=False).indices
        selected = self.act_fn(gate.gather(-1, indices)) * up.gather(-1, indices)
        return self.down_proj(torch.zeros_like(gate).scatter(-1, indices, selected))


def convert(layers, k):
    before = {id(p) for layer in layers for p in layer.parameters()}
    for layer in layers:
        if isinstance(layer.mlp, FixedKMLP):
            raise ValueError("MLP has already been converted")
        layer.mlp = FixedKMLP(layer.mlp, k)
    if before != {id(p) for layer in layers for p in layer.parameters()}:
        raise RuntimeError("MoC conversion changed parameter identities")


def attention_checkpoint(model):
    for block in model.base_model.model.model.layers:
        original = block.self_attn.forward
        def call(self, *args, _original=original, **kwargs):
            if self.training and torch.is_grad_enabled():
                if not args and "hidden_states" in kwargs:
                    args = (kwargs.pop("hidden_states"),)
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)
        block.self_attn.forward = types.MethodType(call, block.self_attn)


def build_model(cfg, method, device, *, overlay=None, policy="block", expand=True):
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    if method not in ("dense", "moc") or policy not in ("block", "attention", "none"):
        raise ValueError("Invalid method/checkpoint policy")
    if method == "dense" and overlay:
        raise ValueError("Dense cannot use a sparse overlay")
    if method == "moc" and not overlay:
        raise ValueError("MoC requires the reconstructed FFN overlay")
    check_source(cfg)
    config = AutoConfig.from_pretrained(cfg["model_dir"], local_files_only=True)
    if {key: getattr(config, key) for key in MODEL_IDENTITY} != MODEL_IDENTITY:
        raise ValueError("Expected Meta-Llama-3.1-8B Base dimensions")
    model = AutoModelForCausalLM.from_pretrained(cfg["model_dir"], torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation="sdpa")
    model.requires_grad_(False)
    model.config.use_cache = False
    if method == "moc":
        convert(model.model.layers, cfg["k"])
    if policy == "block":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(model, cfg["source_adapter"], is_trainable=True, local_files_only=True)
    stored = load_file(str(Path(cfg["source_adapter"]) / "adapter_model.safetensors"))
    actual = get_peft_model_state_dict(model)
    if stored.keys() != actual.keys() or any(not torch.equal(stored[n], actual[n].detach().cpu().to(stored[n].dtype)) for n in stored):
        raise RuntimeError("Source adapter did not load exactly")
    del stored, actual
    if expand:
        expand_rank(model, 128)
        if method == "moc":
            optimize_moc_lora(model)
    model.enable_input_require_grads()
    model.to(device)
    if policy == "attention":
        attention_checkpoint(model)
    if overlay:
        load_overlay(model.base_model.model.model.layers, Path(overlay), cfg["source_adapter_sha256"],
                     file_sha256(Path(cfg["model_dir"]) / "config.json"))
    if method == "moc":
        for block in model.base_model.model.model.layers:
            install(block.mlp)
    return model


def inference_mode(model):
    model.gradient_checkpointing_disable()
    for block in model.base_model.model.model.layers:
        if hasattr(block.mlp, "_reference_forward"):
            uninstall(block.mlp)
    model.config.use_cache = True
    model.eval()


def save_adapter(model, directory, cfg, method, overlay, step):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    directory = Path(directory)
    model.save_pretrained(directory, safe_serialization=True)
    saved = load_file(str(directory / "adapter_model.safetensors"))
    actual = get_peft_model_state_dict(model)
    if saved.keys() != actual.keys() or any(not torch.equal(saved[n], actual[n].detach().cpu()) for n in saved):
        raise RuntimeError("Adapter save/read mismatch")
    write_json(directory / "moc_config.json", dict(
        format="moc_commonsense_adapter_v1", method=method, k=cfg["k"], rank=128, split_rank=32,
        source_adapter_sha256=cfg["source_adapter_sha256"],
        model_config_sha256=file_sha256(Path(cfg["model_dir"]) / "config.json"),
        overlay_manifest_sha256=file_sha256(Path(overlay) / "manifest.json") if overlay else None,
        adapter_sha256=file_sha256(directory / "adapter_model.safetensors"),
        step=step, strict_resume=False))


def load_trained_adapter(model, directory, cfg, method, overlay):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    directory = Path(directory)
    meta = read_json(directory / "moc_config.json")
    expected = dict(format="moc_commonsense_adapter_v1", method=method, k=cfg["k"], rank=128, split_rank=32,
        source_adapter_sha256=cfg["source_adapter_sha256"],
        model_config_sha256=file_sha256(Path(cfg["model_dir"]) / "config.json"),
        overlay_manifest_sha256=file_sha256(Path(overlay) / "manifest.json") if overlay else None,
        adapter_sha256=file_sha256(directory / "adapter_model.safetensors"))
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Adapter/base/overlay/layout identity mismatch")
    weights = load_file(str(directory / "adapter_model.safetensors"))
    if weights.keys() != get_peft_model_state_dict(model).keys():
        raise ValueError("Adapter tensor names do not match the split-rank model")
    set_peft_model_state_dict(model, weights)
    loaded = get_peft_model_state_dict(model)
    if any(not torch.equal(weights[n], loaded[n].detach().cpu()) for n in weights):
        raise RuntimeError("Adapter reload was not exact")
    return meta
