"""End-to-end decode benchmark."""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import traceback
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moc.inference.inference_ffn import InferenceMoCSwiGLUFFN  # noqa: E402
from moc.inference.optimized_global_moc_ops import (  # noqa: E402
    ensure_native_ops_ready,
    make_optimized_global_moc_graph_runner,
)
from moc.data import build_dataloader  # noqa: E402
import moc.inference.triton_fused_ffn_kernels  # noqa: F401,E402
import moc.inference.triton_torch28_op  # noqa: F401,E402


HIDDEN = 2048
INTERMEDIATE = 5464
NUM_HEADS = 16
HEAD_DIM = HIDDEN // NUM_HEADS
NUM_LAYERS_FULL = 24
NUM_LAYERS_SMOKE = 2
PROMPT_LEN = 128
GEN_LEN_FULL = 128
GEN_LEN_SMOKE = 8
VOCAB_SIZE = 32000
GLOBAL_K = 1024
GROUPED_A = 2
GROUPED_B = 8
MOC_2_8_K = INTERMEDIATE * GROUPED_A // GROUPED_B

ROW_SPECS = {
    "dense": {
        "row": "dense",
        "label": "Dense",
        "selection": "dense",
        "ffn_mode": "dense_baseline",
        "k": GLOBAL_K,
        "grouped_a": None,
        "grouped_b": None,
    },
    "global_moc": {
        "row": "global_moc",
        "label": "Global MoC",
        "selection": "global_topk",
        "ffn_mode": "moc_inference_optimized_global_after_gate_native",
        "k": GLOBAL_K,
        "grouped_a": None,
        "grouped_b": None,
    },
    "moc_2_8": {
        "row": "moc_2_8",
        "label": "MoC 2:8",
        "selection": "grouped_top2_of_8",
        "ffn_mode": "moc_inference_torch28_balanced",
        "k": MOC_2_8_K,
        "grouped_a": GROUPED_A,
        "grouped_b": GROUPED_B,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end decode benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup-runs", type=int, default=8)
    parser.add_argument("--measure-runs", type=int, default=15)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--execution-scope",
        choices=["compiled", "whole_graph", "system_graph", "single_layer_methods"],
        default="compiled",
        help=(
            "compiled uses torch.compile over the whole decode step; "
            "whole_graph captures the whole decode step as one fixed-shape CUDA Graph; "
            "system_graph uses fixed-shape per-layer CUDA Graph alternate benchmark scopes; "
            "single_layer_methods inserts the single-layer FFN execution methods into decode."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        norm = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def precompute_rope(dim: int, max_len: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int) -> torch.Tensor:
    length = x.shape[2]
    c = cos[offset:offset + length].unsqueeze(0).unsqueeze(0)
    s = sin[offset:offset + length].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class StaticKVAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, max_seq: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.max_seq = max_seq
        self.qkv_proj = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

        cos, sin = precompute_rope(self.head_dim, max_seq)
        self.register_buffer("rope_cos", cos.to(device=device, dtype=dtype), persistent=False)
        self.register_buffer("rope_sin", sin.to(device=device, dtype=dtype), persistent=False)
        self.register_buffer(
            "causal_mask",
            torch.full((max_seq, max_seq), float("-inf"), device=device, dtype=dtype).triu(1),
            persistent=False,
        )
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def allocate_cache(self, device: str, dtype: torch.dtype) -> None:
        self.k_cache = torch.zeros(1, self.heads, self.max_seq, self.head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(1, self.heads, self.max_seq, self.head_dim, device=device, dtype=dtype)

    def reset_cache(self) -> None:
        if self.k_cache is not None:
            self.k_cache.zero_()
            self.v_cache.zero_()

    def forward_prompt(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        qkv = self.qkv_proj(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q, self.rope_cos, self.rope_sin, 0)
        k = apply_rope(k, self.rope_cos, self.rope_sin, 0)
        self.k_cache[:, :, :length, :].copy_(k)
        self.v_cache[:, :, :length, :].copy_(v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(y.transpose(1, 2).reshape(batch, length, hidden))

    def forward_step_static(self, x: torch.Tensor, pos_tensor: torch.Tensor) -> torch.Tensor:
        batch, _, hidden = x.shape
        qkv = self.qkv_proj(x).reshape(batch, 1, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos_row = self.rope_cos.index_select(0, pos_tensor.view(1))
        sin_row = self.rope_sin.index_select(0, pos_tensor.view(1))
        cos_b = cos_row.unsqueeze(0).unsqueeze(0)
        sin_b = sin_row.unsqueeze(0).unsqueeze(0)

        q1, q2 = q[..., 0::2], q[..., 1::2]
        q = torch.stack([q1 * cos_b - q2 * sin_b, q1 * sin_b + q2 * cos_b], dim=-1).flatten(-2)
        k1, k2 = k[..., 0::2], k[..., 1::2]
        k = torch.stack([k1 * cos_b - k2 * sin_b, k1 * sin_b + k2 * cos_b], dim=-1).flatten(-2)

        self.k_cache.index_copy_(2, pos_tensor.view(1), k)
        self.v_cache.index_copy_(2, pos_tensor.view(1), v)
        mask_row = self.causal_mask.index_select(0, pos_tensor.view(1))
        attn_mask = mask_row.unsqueeze(0).unsqueeze(0)
        y = F.scaled_dot_product_attention(q, self.k_cache, self.v_cache, attn_mask=attn_mask, is_causal=False)
        return self.out_proj(y.transpose(1, 2).reshape(batch, 1, hidden))


class DecoderLayer(nn.Module):
    def __init__(self, ffn_kind: str, device: str, dtype: torch.dtype, max_seq: int):
        super().__init__()
        if ffn_kind not in ROW_SPECS:
            raise ValueError(f"Unknown ffn_kind: {ffn_kind}")
        self.ffn_kind = ffn_kind
        spec = ROW_SPECS[ffn_kind]
        self.attn_norm = RMSNorm(HIDDEN)
        self.attn = StaticKVAttention(HIDDEN, NUM_HEADS, max_seq, device, dtype)
        self.ffn_norm = RMSNorm(HIDDEN)
        ffn_kwargs = {
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "k": spec["k"],
        }
        if spec["grouped_a"] is not None:
            ffn_kwargs.update({"grouped_a": spec["grouped_a"], "grouped_b": spec["grouped_b"]})
        self.ffn = InferenceMoCSwiGLUFFN(**ffn_kwargs)
        self.decode_layer_runner: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        self.single_layer_ffn_runner: Callable[[torch.Tensor], torch.Tensor] | None = None

    def allocate_cache(self, device: str, dtype: torch.dtype) -> None:
        self.attn.allocate_cache(device, dtype)

    def reset_cache(self) -> None:
        self.attn.reset_cache()

    def freeze_for_compile(self) -> None:
        if self.ffn_kind == "global_moc":
            ensure_native_ops_ready()
        self.ffn.freeze_for_compile(device=self.ffn.down_proj.weight.device)

    def prepare_system_graph_runner(self) -> None:
        self.freeze_for_compile()
        device = self.ffn.gate_proj.weight.device
        dtype = self.ffn.gate_proj.weight.dtype
        x_buf = torch.empty(1, 1, HIDDEN, device=device, dtype=dtype)
        pos_buf = torch.zeros((), dtype=torch.int64, device=device)

        def path() -> torch.Tensor:
            return self.forward_step_static(x_buf, pos_buf)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                path()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y_buf = path()

        def runner(x_in: torch.Tensor, pos_in: torch.Tensor) -> torch.Tensor:
            if x_in.shape != x_buf.shape:
                raise ValueError(f"layer graph runner expects {tuple(x_buf.shape)}, got {tuple(x_in.shape)}")
            x_buf.copy_(x_in)
            pos_buf.copy_(pos_in)
            graph.replay()
            return y_buf

        self.decode_layer_runner = runner

    def prepare_single_layer_ffn_runner(self, compile_mode: str) -> None:
        self.freeze_for_compile()
        if self.ffn_kind == "dense":
            mode = "dense_baseline"

            def call(x_in: torch.Tensor) -> torch.Tensor:
                return self.ffn(x_in, mode=mode)

            compiled = torch.compile(call, mode=compile_mode, dynamic=False)
            x_probe = torch.empty(1, HIDDEN, device=self.ffn.gate_proj.weight.device, dtype=self.ffn.gate_proj.weight.dtype)
            for _ in range(5):
                compiled(x_probe)
            torch.cuda.synchronize()
            self.single_layer_ffn_runner = compiled
        elif self.ffn_kind == "global_moc":
            self.single_layer_ffn_runner = make_optimized_global_moc_graph_runner(self.ffn, batch_size=1)
        elif self.ffn_kind == "moc_2_8":
            self.single_layer_ffn_runner = self.ffn.make_v14_graph_runner(
                batch_size=1,
                gate_up_BLOCK_B=16,
                gate_up_BLOCK_G=16,
                gate_up_BLOCK_H=128,
                down_block_k=128,
                down_block_h=16,
            )

    def _ffn_call(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        flat = x.reshape(batch * length, hidden)
        if self.single_layer_ffn_runner is not None and flat.shape[0] == 1:
            out = self.single_layer_ffn_runner(flat)
        else:
            out = self.ffn(flat, mode=ROW_SPECS[self.ffn_kind]["ffn_mode"])
        return out.reshape(batch, length, hidden)

    def forward_prompt(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward_prompt(self.attn_norm(x))
        x = x + self._ffn_call(self.ffn_norm(x))
        return x

    def forward_step_static(self, x: torch.Tensor, pos_tensor: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward_step_static(self.attn_norm(x), pos_tensor)
        x = x + self._ffn_call(self.ffn_norm(x))
        return x

    def forward_step_system_graph(self, x: torch.Tensor, pos_tensor: torch.Tensor) -> torch.Tensor:
        if self.decode_layer_runner is None:
            raise RuntimeError("final algorithm layer runner has not been prepared")
        return self.decode_layer_runner(x, pos_tensor)


class DecoderModel(nn.Module):
    def __init__(self, num_layers: int, ffn_kind: str, device: str, dtype: torch.dtype, max_seq: int):
        super().__init__()
        self.ffn_kind = ffn_kind
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.layers = nn.ModuleList(
            [DecoderLayer(ffn_kind, device, dtype, max_seq) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB_SIZE, bias=False)

    def allocate_cache(self, device: str, dtype: torch.dtype) -> None:
        for layer in self.layers:
            layer.allocate_cache(device, dtype)

    def reset_cache(self) -> None:
        for layer in self.layers:
            layer.reset_cache()

    def freeze_for_compile(self) -> None:
        if self.ffn_kind == "global_moc":
            ensure_native_ops_ready()
        if self.ffn_kind != "dense":
            for layer in self.layers:
                layer.freeze_for_compile()

    def prepare_system_graph_runners(self) -> None:
        for layer in self.layers:
            layer.prepare_system_graph_runner()

    def prepare_single_layer_ffn_runners(self, compile_mode: str) -> None:
        for layer in self.layers:
            layer.prepare_single_layer_ffn_runner(compile_mode)

    def forward_prompt(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        for layer in self.layers:
            x = layer.forward_prompt(x)
        return self.lm_head(self.final_norm(x))

    def forward_step_static(self, token_ids: torch.Tensor, pos_tensor: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        for layer in self.layers:
            x = layer.forward_step_static(x, pos_tensor)
        return self.lm_head(self.final_norm(x))

    def forward_step_system_graph(self, token_ids: torch.Tensor, pos_tensor: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        for layer in self.layers:
            x = layer.forward_step_system_graph(x, pos_tensor)
        return self.lm_head(self.final_norm(x))


@torch.no_grad()
def run_one_decode(
    model: DecoderModel,
    prompt_token_ids: torch.Tensor,
    gen_len: int,
    step_callable: Callable[[torch.Tensor, int], torch.Tensor],
) -> float:
    model.reset_cache()
    prompt_logits = model.forward_prompt(prompt_token_ids)
    cur_token_ids = prompt_logits[:, -1:, :].argmax(dim=-1)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for step in range(gen_len):
        logits = step_callable(cur_token_ids, PROMPT_LEN + step)
        cur_token_ids = logits.argmax(dim=-1)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def summarize(samples: list[float]) -> dict:
    return {
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "all": samples,
    }


def measure_decode(
    model: DecoderModel,
    prompt_token_ids: torch.Tensor,
    gen_len: int,
    warmup_runs: int,
    measure_runs: int,
    step_callable: Callable[[torch.Tensor, int], torch.Tensor],
) -> dict:
    for _ in range(warmup_runs):
        run_one_decode(model, prompt_token_ids, gen_len, step_callable)

    samples = [
        run_one_decode(model, prompt_token_ids, gen_len, step_callable)
        for _ in range(measure_runs)
    ]
    median_ms = statistics.median(samples)
    return {
        "total_generation_ms": summarize(samples),
        "latency_ms_per_token": median_ms / gen_len,
        "throughput_tok_per_sec": gen_len / (median_ms / 1000.0),
        "gen_len": gen_len,
        "prompt_len": PROMPT_LEN,
        "batch_size": 1,
    }


def build_compiled_step(
    model: DecoderModel,
    device: str,
    compile_mode: str,
    prompt_token_ids: torch.Tensor,
) -> tuple[Callable[[torch.Tensor, int], torch.Tensor], list[dict]]:
    model.freeze_for_compile()
    pos_tensor = torch.zeros((), dtype=torch.int64, device=device)
    attempts: list[dict] = []
    attempt = {"mode": compile_mode, "dynamic": True, "phase": "probe", "status": None}
    try:
        compiled = torch.compile(model.forward_step_static, mode=compile_mode, dynamic=True)

        def step(token_ids: torch.Tensor, pos: int) -> torch.Tensor:
            pos_tensor.fill_(pos)
            return compiled(token_ids, pos_tensor)

        with torch.no_grad():
            model.reset_cache()
            _ = model.forward_prompt(prompt_token_ids)
            probe_token_ids = torch.randint(0, VOCAB_SIZE, (1, 1), device=device, dtype=torch.int64)
            for probe_pos in range(PROMPT_LEN, PROMPT_LEN + 4):
                _ = step(probe_token_ids, probe_pos)
            torch.cuda.synchronize()
        attempt["status"] = "OK"
        attempts.append(attempt)
        return step, attempts
    except Exception as exc:
        attempt.update(
            {
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
        )
        attempts.append(attempt)
        raise


def build_system_graph_step(
    model: DecoderModel,
    ffn_kind: str,
    device: str,
    compile_mode: str,
    prompt_token_ids: torch.Tensor,
) -> tuple[Callable[[torch.Tensor, int], torch.Tensor], list[dict], str]:
    model.prepare_system_graph_runners()
    pos_tensor = torch.zeros((), dtype=torch.int64, device=device)
    attempts = [
        {
            "mode": "system_graph",
            "dynamic": False,
            "phase": "prepare_layer_graph_runners",
            "status": "OK",
        }
    ]

    def step(token_ids: torch.Tensor, pos: int) -> torch.Tensor:
        pos_tensor.fill_(pos)
        return model.forward_step_system_graph(token_ids, pos_tensor)

    with torch.no_grad():
        model.reset_cache()
        _ = model.forward_prompt(prompt_token_ids)
        probe_token_ids = torch.randint(0, VOCAB_SIZE, (1, 1), device=device, dtype=torch.int64)
        for probe_pos in range(PROMPT_LEN, PROMPT_LEN + 4):
            _ = step(probe_token_ids, probe_pos)
        torch.cuda.synchronize()

    if ffn_kind == "dense":
        optimization = "dense_layer_cuda_graph_alternate"
    elif ffn_kind == "global_moc":
        optimization = "global_moc_cub_layer_cuda_graph_alternate"
    elif ffn_kind == "moc_2_8":
        optimization = "moc2_8_balanced_v14_layer_cuda_graph_alternate"
    else:
        optimization = "system_graph"
    return step, attempts, optimization


def build_whole_graph_step(
    model: DecoderModel,
    ffn_kind: str,
    device: str,
    prompt_token_ids: torch.Tensor,
) -> tuple[Callable[[torch.Tensor, int], torch.Tensor], list[dict], str]:
    model.freeze_for_compile()
    token_buf = torch.empty(1, 1, dtype=torch.int64, device=device)
    pos_buf = torch.zeros((), dtype=torch.int64, device=device)
    token_buf.zero_()
    pos_buf.fill_(PROMPT_LEN)

    attempts = [
        {
            "mode": "whole_graph",
            "dynamic": False,
            "phase": "capture_full_decode_step",
            "status": None,
        }
    ]

    def path() -> torch.Tensor:
        return model.forward_step_static(token_buf, pos_buf)

    try:
        with torch.no_grad():
            model.reset_cache()
            _ = model.forward_prompt(prompt_token_ids)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    path()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                logits_buf = path()
            torch.cuda.synchronize()

        def step(token_ids: torch.Tensor, pos: int) -> torch.Tensor:
            token_buf.copy_(token_ids)
            pos_buf.fill_(pos)
            graph.replay()
            return logits_buf

        with torch.no_grad():
            model.reset_cache()
            _ = model.forward_prompt(prompt_token_ids)
            probe_token_ids = torch.randint(0, VOCAB_SIZE, (1, 1), device=device, dtype=torch.int64)
            for probe_pos in range(PROMPT_LEN, PROMPT_LEN + 4):
                _ = step(probe_token_ids, probe_pos)
            torch.cuda.synchronize()

        attempts[0]["status"] = "OK"
    except Exception as exc:
        attempts[0].update(
            {
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
        )
        raise

    if ffn_kind == "dense":
        optimization = "dense_whole_decode_cuda_graph_alternate"
    elif ffn_kind == "global_moc":
        optimization = "global_moc_cub_whole_decode_cuda_graph_alternate"
    elif ffn_kind == "moc_2_8":
        optimization = "moc2_8_balanced_v14_whole_decode_cuda_graph_alternate"
    else:
        optimization = "whole_graph"
    return step, attempts, optimization


def build_single_layer_methods_step(
    model: DecoderModel,
    ffn_kind: str,
    device: str,
    compile_mode: str,
    prompt_token_ids: torch.Tensor,
) -> tuple[Callable[[torch.Tensor, int], torch.Tensor], list[dict], str]:
    model.prepare_single_layer_ffn_runners(compile_mode)
    pos_tensor = torch.zeros((), dtype=torch.int64, device=device)
    attempts = [
        {
            "mode": "single_layer_methods",
            "dynamic": False,
            "phase": "prepare_single_layer_ffn_ffn_runners",
            "status": "OK",
        }
    ]

    def step(token_ids: torch.Tensor, pos: int) -> torch.Tensor:
        pos_tensor.fill_(pos)
        return model.forward_step_static(token_ids, pos_tensor)

    with torch.no_grad():
        model.reset_cache()
        _ = model.forward_prompt(prompt_token_ids)
        probe_token_ids = torch.randint(0, VOCAB_SIZE, (1, 1), device=device, dtype=torch.int64)
        for probe_pos in range(PROMPT_LEN, PROMPT_LEN + 4):
            _ = step(probe_token_ids, probe_pos)
        torch.cuda.synchronize()

    if ffn_kind == "dense":
        optimization = f"single_layer_ffn_dense_ffn_torch_compile_{compile_mode}_dynamic_false"
    elif ffn_kind == "global_moc":
        optimization = "single_layer_ffn_global_moc_cub_ffn_cuda_graph"
    elif ffn_kind == "moc_2_8":
        optimization = "single_layer_ffn_moc2_8_balanced_v14_ffn_cuda_graph"
    else:
        optimization = "single_layer_methods"
    return step, attempts, optimization


def build_and_measure_row(
    ffn_kind: str,
    num_layers: int,
    gen_len: int,
    device: str,
    dtype: torch.dtype,
    warmup_runs: int,
    measure_runs: int,
    compile_mode: str,
    execution_scope: str,
    prompt_token_ids_cpu: torch.Tensor,
) -> dict:
    spec = ROW_SPECS[ffn_kind]
    label = spec["row"]
    print(f"[{label}] building {num_layers}-layer model ...", flush=True)
    model: DecoderModel | None = None
    try:
        model = DecoderModel(num_layers, ffn_kind, device, dtype, PROMPT_LEN + gen_len).to(device=device, dtype=dtype)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        model.allocate_cache(device, dtype)
        prompt_token_ids = prompt_token_ids_cpu.to(device=device, dtype=torch.int64, non_blocking=True)
        if execution_scope == "compiled":
            step_callable, compile_attempts = build_compiled_step(
                model, device, compile_mode, prompt_token_ids
            )
            optimization = f"torch_compile_{compile_mode}_dynamic"
        elif execution_scope == "single_layer_methods":
            step_callable, compile_attempts, optimization = build_single_layer_methods_step(
                model, ffn_kind, device, compile_mode, prompt_token_ids
            )
        elif execution_scope == "whole_graph":
            step_callable, compile_attempts, optimization = build_whole_graph_step(
                model, ffn_kind, device, prompt_token_ids
            )
        else:
            step_callable, compile_attempts, optimization = build_system_graph_step(
                model, ffn_kind, device, compile_mode, prompt_token_ids
            )
        result = measure_decode(model, prompt_token_ids, gen_len, warmup_runs, measure_runs, step_callable)
        result.update(
            {
                "row": label,
                "label": spec["label"],
                "ffn_kind": ffn_kind,
                "selection": spec["selection"],
                "ffn_mode": spec["ffn_mode"],
                "k": spec["k"],
                "grouped_a": spec["grouped_a"],
                "grouped_b": spec["grouped_b"],
                "optimization": optimization,
                "execution_scope": execution_scope,
                "num_layers": num_layers,
                "status": "OK",
                "cuda_memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
                "compile_attempts": compile_attempts,
            }
        )
        print(
            f"[{label}] OK per_token_ms={result['latency_ms_per_token']:.4f} "
            f"throughput={result['throughput_tok_per_sec']:.1f} tok/s",
            flush=True,
        )
        return result
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        print(f"[{label}] FAILED {msg}", flush=True)
        return {
            "row": label,
            "label": spec["label"],
            "ffn_kind": ffn_kind,
            "selection": spec["selection"],
            "ffn_mode": spec["ffn_mode"],
            "k": spec["k"],
            "grouped_a": spec["grouped_a"],
            "grouped_b": spec["grouped_b"],
            "optimization": (
                f"torch_compile_{compile_mode}_dynamic"
                if execution_scope == "compiled"
                else execution_scope
            ),
            "execution_scope": execution_scope,
            "status": "FAILED",
            "failure_message": msg,
            "traceback": traceback.format_exc(limit=5),
        }
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def compare_pair(dense_row: dict, moc_row: dict) -> dict:
    if dense_row.get("status") != "OK" or moc_row.get("status") != "OK":
        return {
            "speedup_dense_over_moc": None,
            "moc_faster_than_dense": None,
            "dense_status": dense_row.get("status"),
            "moc_status": moc_row.get("status"),
        }
    dense_latency = dense_row["latency_ms_per_token"]
    moc_latency = moc_row["latency_ms_per_token"]
    return {
        "dense_latency_ms_per_token": dense_latency,
        "moc_latency_ms_per_token": moc_latency,
        "speedup_dense_over_moc": dense_latency / moc_latency,
        "moc_faster_than_dense": moc_latency < dense_latency,
    }


def load_c4_prompt(prompt_len: int) -> torch.Tensor:
    loader = build_dataloader("val", batch_size=1, seq_len=prompt_len, num_workers=0, shuffle=False)
    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("C4 validation loader did not produce a prompt batch.") from exc
    return batch["input_ids"].cpu()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    dtype = torch.bfloat16
    num_layers = NUM_LAYERS_SMOKE if args.mode == "smoke" else NUM_LAYERS_FULL
    gen_len = GEN_LEN_SMOKE if args.mode == "smoke" else GEN_LEN_FULL
    warmup_runs = 1 if args.mode == "smoke" else args.warmup_runs
    measure_runs = 2 if args.mode == "smoke" else args.measure_runs
    prompt_token_ids_cpu = load_c4_prompt(PROMPT_LEN)

    rows = {
        ROW_SPECS["dense"]["row"]: build_and_measure_row(
            "dense",
            num_layers,
            gen_len,
            args.device,
            dtype,
            warmup_runs,
            measure_runs,
            args.compile_mode,
            args.execution_scope,
            prompt_token_ids_cpu,
        ),
        ROW_SPECS["global_moc"]["row"]: build_and_measure_row(
            "global_moc",
            num_layers,
            gen_len,
            args.device,
            dtype,
            warmup_runs,
            measure_runs,
            args.compile_mode,
            args.execution_scope,
            prompt_token_ids_cpu,
        ),
        ROW_SPECS["moc_2_8"]["row"]: build_and_measure_row(
            "moc_2_8",
            num_layers,
            gen_len,
            args.device,
            dtype,
            warmup_runs,
            measure_runs,
            args.compile_mode,
            args.execution_scope,
            prompt_token_ids_cpu,
        ),
    }
    global_moc_pair = compare_pair(rows["dense"], rows["global_moc"])
    moc_2_8_pair = compare_pair(rows["dense"], rows["moc_2_8"])

    payload = {
        "benchmark": "end_to_end_decode",
        "version": f"a800_cuda128_{args.execution_scope}",
        "mode": args.mode,
        "data": "c4",
        "shape": {
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "num_layers": num_layers,
            "vocab_size": VOCAB_SIZE,
            "has_token_embedding": True,
            "has_lm_head": True,
            "embedding_tied_with_lm_head": False,
            "prompt_len": PROMPT_LEN,
            "gen_len": gen_len,
            "max_seq": PROMPT_LEN + gen_len,
            "batch_size": 1,
            "global_topk_k": GLOBAL_K,
            "moc_2_8_k": MOC_2_8_K,
            "grouped_a": GROUPED_A,
            "grouped_b": GROUPED_B,
        },
        "rows": rows,
        "pairs": {
            "dense_vs_global_moc": global_moc_pair,
            "dense_vs_moc_2_8": moc_2_8_pair,
        },
        "alignment_status": (
            "MOC_FASTER_THAN_DENSE"
            if global_moc_pair.get("moc_faster_than_dense")
            else "MOC_NOT_FASTER_THAN_DENSE"
        ),
        "measurement_scope": args.execution_scope,
        "run_config": {
            "warmup_runs": warmup_runs,
            "measure_runs": measure_runs,
            "compile_mode": args.compile_mode,
            "execution_scope": args.execution_scope,
            "seed": args.seed,
        },
        "timing_method": {
            "loop_contains": [
                "embedding lookup",
                "forward_step per layer",
                "in-place KV update",
                "SDPA over full cache + mask",
                "final_norm + lm_head",
                "argmax over vocab",
            ],
            "loop_excludes": [
                "host-side .item() / .tolist() / .cpu()",
            ],
            "kv_cache": "preallocated [1, num_heads, max_seq, head_dim] per layer",
            "timer": "torch.cuda.Event elapsed_time",
        },
        "notes": [
            "Random weights; C4 prompt token IDs; latency-only benchmark.",
            (
                f"execution_scope=compiled: all rows use torch.compile(mode='{args.compile_mode}', dynamic=True) over the whole decode step."
                if args.execution_scope == "compiled"
                else (
                    "execution_scope=single_layer_methods: each row inserts the single-layer FFN execution method into decode."
                    if args.execution_scope == "single_layer_methods"
                    else (
                        "execution_scope=whole_graph: each row captures the whole single-token decode step as one fixed-shape CUDA Graph alternate."
                        if args.execution_scope == "whole_graph"
                        else "execution_scope=system_graph: each row uses fixed-shape per-layer CUDA Graph alternate benchmark scopes."
                    )
                )
            ),
            "global_moc uses ordinary global Top-K channel selection.",
            "moc_2_8 uses grouped top-2-of-8 channel selection.",
            "compiled is the default benchmark scope.",
        ],
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"Decode benchmark ({args.execution_scope})")
    for name, row in rows.items():
        if row.get("status") == "OK":
            print(
                f"  {name:16s}: {row['latency_ms_per_token']:.3f} ms/token, "
                f"{row['throughput_tok_per_sec']:.1f} tok/s"
            )
        else:
            print(f"  {name:16s}: {row.get('status')} {row.get('failure_message', '')}")
    if global_moc_pair.get("speedup_dense_over_moc") is not None:
        print(f"  global MoC speedup: {global_moc_pair['speedup_dense_over_moc']:.3f}x")
    if moc_2_8_pair.get("speedup_dense_over_moc") is not None:
        print(f"  MoC 2:8 speedup:    {moc_2_8_pair['speedup_dense_over_moc']:.3f}x")


if __name__ == "__main__":
    main()

