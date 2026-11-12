"""End-to-end BF16 decode benchmark."""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
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
        "ffn_mode": "moc_inference_grouped_top2of8",
        "k": MOC_2_8_K,
        "grouped_a": GROUPED_A,
        "grouped_b": GROUPED_B,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end BF16 decode benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--out", required=True)
    parser.add_argument("--methods", nargs="+", choices=list(ROW_SPECS), default=["dense", "global_moc"])
    parser.add_argument("--warmup-runs", type=int, default=8)
    parser.add_argument("--measure-runs", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.warmup_runs < 0 or args.measure_runs < 1 or args.rounds < 1:
        parser.error("warmup must be nonnegative; measure-runs and rounds must be positive")
    if len(args.methods) != len(set(args.methods)):
        parser.error("methods must not contain duplicates")
    return args


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
        from moc.inference.inference_ffn import InferenceMoCSwiGLUFFN

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
            import moc.inference.triton_grouped_moc_ops  # noqa: F401

            ffn_kwargs.update({"grouped_a": spec["grouped_a"], "grouped_b": spec["grouped_b"]})
        self.ffn = InferenceMoCSwiGLUFFN(**ffn_kwargs)

    def allocate_cache(self, device: str, dtype: torch.dtype) -> None:
        self.attn.allocate_cache(device, dtype)

    def reset_cache(self) -> None:
        self.attn.reset_cache()

    def freeze_for_compile(self) -> None:
        self.ffn.freeze_for_compile(device=self.ffn.down_proj.weight.device)

    def _ffn_call(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        flat = x.reshape(batch * length, hidden)
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
            from moc.inference.optimized_global_moc_ops import ensure_native_ops_ready

            ensure_native_ops_ready()
        if self.ffn_kind != "dense":
            for layer in self.layers:
                layer.freeze_for_compile()

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
    prompt_token_ids: torch.Tensor,
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    model.freeze_for_compile()
    pos_tensor = torch.zeros((), dtype=torch.int64, device=device)
    compiled = torch.compile(model.forward_step_static, dynamic=True, options={"cpp_wrapper": True})

    def step(token_ids: torch.Tensor, pos: int) -> torch.Tensor:
        pos_tensor.fill_(pos)
        return compiled(token_ids, pos_tensor)

    with torch.no_grad():
        model.reset_cache()
        model.forward_prompt(prompt_token_ids)
        probe = torch.randint(0, VOCAB_SIZE, (1, 1), device=device, dtype=torch.int64)
        for pos in range(PROMPT_LEN, PROMPT_LEN + 4):
            step(probe, pos)
        torch.cuda.synchronize()
    return step


@torch.no_grad()
def build_and_measure_row(
    ffn_kind: str,
    num_layers: int,
    gen_len: int,
    device: str,
    dtype: torch.dtype,
    warmup_runs: int,
    measure_runs: int,
    prompt_token_ids_cpu: torch.Tensor,
) -> dict:
    spec = ROW_SPECS[ffn_kind]
    print(f"[{ffn_kind}] building {num_layers}-layer model ...", flush=True)
    model = None
    step = None
    try:
        model = DecoderModel(num_layers, ffn_kind, device, dtype, PROMPT_LEN + gen_len).to(device=device, dtype=dtype).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.allocate_cache(device, dtype)
        prompt = prompt_token_ids_cpu.to(device=device, dtype=torch.int64, non_blocking=True)
        step = build_compiled_step(model, device, prompt)
        for _ in range(warmup_runs):
            run_one_decode(model, prompt, gen_len, step)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        result = measure_decode(model, prompt, gen_len, 0, measure_runs, step)
        result.update(
            row=ffn_kind, label=spec["label"], ffn_kind=ffn_kind, selection=spec["selection"],
            ffn_mode=spec["ffn_mode"], k=spec["k"], grouped_a=spec["grouped_a"], grouped_b=spec["grouped_b"],
            status="OK", peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        )
        print(f"[{ffn_kind}] {result['latency_ms_per_token']:.4f} ms/token", flush=True)
        return result
    except Exception as exc:
        return {
            "row": ffn_kind, "status": "FAILED", "failure_message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=5),
        }
    finally:
        del step, model
        gc.collect()
        torch.cuda.empty_cache()


def aggregate_rounds(rounds: list[dict]) -> dict:
    if not rounds:
        raise ValueError("At least one round is required")
    if any(row.get("status") != "OK" for row in rounds):
        return {"row": rounds[0]["row"], "status": "FAILED", "rounds": rounds}
    latency = statistics.mean(row["latency_ms_per_token"] for row in rounds)
    first = rounds[0]
    return {
        **{key: first[key] for key in ("row", "label", "ffn_kind", "selection", "ffn_mode", "k", "grouped_a", "grouped_b")},
        "status": "OK",
        "latency_ms_per_token": latency,
        "throughput_tok_per_sec": 1000.0 / latency,
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rounds),
        "rounds": rounds,
    }


def _measure_process(connection, settings: dict, rng_state: dict, prompt: torch.Tensor) -> None:
    try:
        device = torch.device(settings["device"])
        torch.cuda.set_device(device if device.index is not None else 0)
        torch.manual_seed(settings["seed"])
        torch.cuda.manual_seed_all(settings["seed"])
        torch.set_rng_state(torch.tensor(rng_state["cpu"], dtype=torch.uint8))
        if rng_state["cuda"] is not None:
            torch.cuda.set_rng_state(torch.tensor(rng_state["cuda"], dtype=torch.uint8), device)
        torch.set_float32_matmul_precision("high")
        result = build_and_measure_row(
            settings["method"], settings["num_layers"], settings["gen_len"], settings["device"],
            torch.bfloat16, settings["warmup"], settings["measures"], prompt,
        )
        result["gpu"] = torch.cuda.get_device_name(device)
        next_state = {"cpu": torch.get_rng_state().tolist(),
                      "cuda": torch.cuda.get_rng_state(device).tolist()}
        connection.send((result, next_state))
    except Exception as exc:
        connection.send(({"row": settings["method"], "status": "FAILED",
                          "failure_message": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc(limit=5)}, rng_state))
    finally:
        connection.close()


def run_measurement(settings: dict, rng_state: dict, prompt: torch.Tensor) -> tuple[dict, dict]:
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_measure_process, args=(send, settings, rng_state, prompt))
    process.start()
    send.close()
    try:
        result = receive.recv()
    except EOFError:
        result = ({"row": settings["method"], "status": "FAILED",
                   "failure_message": "Measurement process exited without a result"}, rng_state)
    except BaseException:
        process.terminate()
        raise
    finally:
        receive.close()
        process.join()
    if process.exitcode != 0:
        result[0].update(status="FAILED", process_exitcode=process.exitcode)
    return result


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
    from moc.data import build_dataloader

    loader = build_dataloader("val", batch_size=1, seq_len=prompt_len, num_workers=0, shuffle=False)
    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("C4 validation loader did not produce a prompt batch.") from exc
    return batch["input_ids"].cpu()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("A CUDA device is required")
    torch.manual_seed(args.seed)
    num_layers = NUM_LAYERS_SMOKE if args.mode == "smoke" else NUM_LAYERS_FULL
    gen_len = GEN_LEN_SMOKE if args.mode == "smoke" else GEN_LEN_FULL
    warmup = 1 if args.mode == "smoke" else args.warmup_runs
    measures = 2 if args.mode == "smoke" else args.measure_runs
    count = 1 if args.mode == "smoke" else args.rounds
    prompt = load_c4_prompt(PROMPT_LEN)
    results = {method: [] for method in args.methods}
    rng_states = {}
    next_state = {"cpu": torch.get_rng_state().tolist(), "cuda": None}
    orders = []
    for round_index in range(count):
        order = args.methods if round_index % 3 == 0 else (
            list(reversed(args.methods)) if round_index % 3 == 1 else args.methods[1:] + args.methods[:1]
        )
        orders.append(order)
        for method in order:
            rng_states.setdefault(method, next_state)
            print(f"Round {round_index + 1}/{count}", flush=True)
            result, next_state = run_measurement(
                {"method": method, "num_layers": num_layers, "gen_len": gen_len, "device": args.device,
                 "warmup": warmup, "measures": measures, "seed": args.seed}, rng_states[method], prompt,
            )
            results[method].append(result)
    rows = {method: aggregate_rounds(values) for method, values in results.items()}
    pairs = {
        f"dense_vs_{method}": compare_pair(rows["dense"], row)
        for method, row in rows.items() if method != "dense" and "dense" in rows
    }
    payload = {
        "benchmark": "end_to_end_decode",
        "mode": args.mode,
        "data": "c4",
        "shape": {
            "hidden": HIDDEN, "intermediate": INTERMEDIATE, "num_heads": NUM_HEADS, "head_dim": HEAD_DIM,
            "num_layers": num_layers, "vocab_size": VOCAB_SIZE, "prompt_len": PROMPT_LEN,
            "gen_len": gen_len, "max_seq": PROMPT_LEN + gen_len, "batch_size": 1,
            "has_token_embedding": True, "has_lm_head": True, "embedding_tied_with_lm_head": False,
            "global_topk_k": GLOBAL_K, "moc_2_8_k": MOC_2_8_K, "grouped_a": GROUPED_A, "grouped_b": GROUPED_B,
        },
        "rows": rows,
        "pairs": pairs,
        "run_config": {
            "rounds": count, "round_order": orders, "warmup_runs": warmup, "measure_runs": measures,
            "seed": args.seed, "execution_scope": "compiled", "compile_mode": "default",
            "dynamic": True, "cpp_wrapper": True, "methods": args.methods,
            "isolated_process_per_round": True,
        },
        "timing_method": {
            "timer": "torch.cuda.Event elapsed_time",
            "aggregation": "mean of per-round median latency; maximum allocated peak across rounds",
            "loop_contains": ["embedding", "attention", "KV updates", "FFN", "final norm", "LM head", "argmax"],
            "loop_excludes": ["prefill", "warmup", "compilation", "host token copies"],
        },
        "notes": ["Random weights; C4 prompt token IDs; latency-only benchmark."],
        "device": args.device,
        "gpu": next((row["gpu"] for values in results.values() for row in values if "gpu" in row), None),
        "dtype": "bf16",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\nEnd-to-end decode")
    for method, row in rows.items():
        if row["status"] == "OK":
            print(f"  {method:16s}: {row['latency_ms_per_token']:.3f} ms/token, {row['throughput_tok_per_sec']:.1f} tok/s")
        else:
            print(f"  {method:16s}: FAILED")
    for name, pair in pairs.items():
        if pair.get("speedup_dense_over_moc") is not None:
            print(f"  {name}: {pair['speedup_dense_over_moc']:.3f}x")
    if any(row["status"] != "OK" for row in rows.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
