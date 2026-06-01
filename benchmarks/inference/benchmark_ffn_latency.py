"""Single-layer FFN inference latency benchmark."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moc.inference.inference_ffn import InferenceMoCSwiGLUFFN  # noqa: E402
from moc.inference.optimized_global_moc_ops import load_optimized_global_moc_extension  # noqa: E402
import moc.inference.triton_fused_ffn_kernels  # noqa: F401,E402


HIDDEN = 2048
INTERMEDIATE = 5464
GLOBAL_K = 1024
GROUPED_A = 2
GROUPED_B = 8
MOC_2_8_K = INTERMEDIATE * GROUPED_A // GROUPED_B
MOC2_8_BALANCED_GRAPH_CONFIG = {
    "gate_up_BLOCK_B": 16,
    "gate_up_BLOCK_G": 16,
    "gate_up_BLOCK_H": 128,
    "down_block_k": 128,
    "down_block_h": 16,
}

ROWS = {
    "Standard_FFN": {
        "kind": "dense",
        "selection": "dense",
        "mode": "dense_baseline",
        "runner": "torch_compile",
        "k": GLOBAL_K,
        "topk_sorted": None,
        "grouped_a": None,
        "grouped_b": None,
    },
    "MoC": {
        "kind": "moc_global",
        "selection": "global_topk",
        "mode": "moc_inference_optimized_global_graph",
        "runner": "global_cuda_graph",
        "k": GLOBAL_K,
        "topk_sorted": False,
        "grouped_a": None,
        "grouped_b": None,
    },
    "MoC_2_8": {
        "kind": "moc_2_8",
        "selection": "grouped_top2_of_8",
        "mode": "moc_inference_moc2_8_graph",
        "runner": "moc2_8_cuda_graph",
        "k": MOC_2_8_K,
        "topk_sorted": False,
        "grouped_a": GROUPED_A,
        "grouped_b": GROUPED_B,
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-layer FFN latency benchmark")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--groups", type=int, default=80)
    parser.add_argument("--group-iters", type=int, default=100)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "min_us": min(samples),
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "p10_us": ordered[int(0.10 * (len(ordered) - 1))],
        "p90_us": ordered[int(0.90 * (len(ordered) - 1))],
        "num_groups": len(samples),
    }


def build_ffn(row: dict, device: str, dtype: torch.dtype) -> InferenceMoCSwiGLUFFN:
    kwargs = {
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "k": row["k"],
    }
    if row["grouped_a"] is not None:
        kwargs.update({"grouped_a": row["grouped_a"], "grouped_b": row["grouped_b"]})
    module = InferenceMoCSwiGLUFFN(**kwargs).to(device=device, dtype=dtype).eval()
    for param in module.parameters():
        param.requires_grad_(False)
    module.freeze_for_compile(device=torch.device(device))
    return module


def compile_call(module: InferenceMoCSwiGLUFFN, x: torch.Tensor, mode: str, compile_mode: str):
    def call() -> torch.Tensor:
        return module(x, mode=mode)

    compiled = torch.compile(call, mode=compile_mode, dynamic=False)
    for _ in range(5):
        compiled()
    torch.cuda.synchronize()
    return compiled


def make_fixed_global_moc_graph_runner(module: InferenceMoCSwiGLUFFN, x: torch.Tensor):
    ext = load_optimized_global_moc_extension()
    if module.config.grouped:
        raise ValueError("global MoC graph runner requires ungrouped MoC")
    module._ensure_down_weight_t()
    x = x.contiguous()
    up_w = module.up_proj.weight.contiguous()
    down_t = module._down_weight_t.contiguous()

    def path():
        gate = module.gate_proj(x).contiguous()
        topk_vals, topk_idx = ext.cub_topk_bf16_512x11(gate, module.config.k)
        sparse_z = ext.selected_up_silu_bf16(
            x,
            topk_vals.contiguous(),
            topk_idx.contiguous(),
            up_w,
        )
        return ext.selected_down_bf16_h32_k16(
            sparse_z.contiguous(),
            topk_idx.contiguous(),
            down_t,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            path()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = path()

    def replay():
        graph.replay()
        return y

    return replay


def build_runner(module: InferenceMoCSwiGLUFFN, x: torch.Tensor, row: dict, compile_mode: str):
    runner_kind = row["runner"]
    if runner_kind == "torch_compile":
        return compile_call(module, x, row["mode"], compile_mode), f"torch_compile_{compile_mode}_dynamic_false"
    if runner_kind == "global_cuda_graph":
        runner = make_fixed_global_moc_graph_runner(module, x)
        return runner, "global_topk_cub_selected_projection_cuda_graph"
    if runner_kind == "moc2_8_cuda_graph":
        runner = module.make_moc2_8_graph_runner(
            batch_size=x.shape[0],
            **MOC2_8_BALANCED_GRAPH_CONFIG,
        )
        return lambda: runner(x), "grouped_top2of8_triton_cuda_graph"
    raise ValueError(f"unknown runner={runner_kind}")


@torch.no_grad()
def measure_us(fn, warmup: int, groups: int, group_iters: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(groups):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(group_iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / group_iters)
    return summarize(samples)


@torch.no_grad()
def check_correctness(module: InferenceMoCSwiGLUFFN, x: torch.Tensor, mode: str) -> dict | None:
    if mode == "dense_baseline":
        return None
    ref = module(x, mode="masked_reference")
    got = module(x, mode=mode)
    diff = (ref - got).abs()
    denom = ref.abs().clamp_min(1e-6)
    return {
        "reference": "masked_reference with the same selection rule",
        "max_abs_error": float(diff.max().item()),
        "max_rel_error": float((diff / denom).max().item()),
    }


@torch.no_grad()
def check_runner_correctness(
    module: InferenceMoCSwiGLUFFN,
    x: torch.Tensor,
    fn,
    row: dict,
) -> dict | None:
    if row["runner"] == "torch_compile":
        return check_correctness(module, x, row["mode"])
    ref = module(x, mode="masked_reference")
    got = fn()
    torch.cuda.synchronize()
    diff = (ref - got).abs()
    denom = ref.abs().clamp_min(1e-6)
    return {
        "reference": "masked_reference with the same selection rule",
        "max_abs_error": float(diff.max().item()),
        "max_rel_error": float((diff / denom).max().item()),
    }


def measure_row(row_name: str, args: argparse.Namespace, dtype: torch.dtype) -> dict:
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    row = ROWS[row_name]
    module = build_ffn(row, args.device, dtype)
    x = torch.randn(1, HIDDEN, device=args.device, dtype=dtype)
    mode = row["mode"]
    fn, optimization = build_runner(module, x, row, args.compile_mode)
    correctness = check_runner_correctness(module, x, fn, row)
    stats = measure_us(fn, args.warmup, args.groups, args.group_iters)
    return {
        "row_name": row_name,
        "kind": row["kind"],
        "selection": row["selection"],
        "mode": mode,
        "runner": row["runner"],
        "k": row["k"],
        "topk_sorted": row["topk_sorted"],
        "grouped_a": row["grouped_a"],
        "grouped_b": row["grouped_b"],
        "optimization": optimization,
        "latency": stats,
        "correctness": correctness,
        "cuda_memory_allocated_mb": torch.cuda.memory_allocated(args.device) / (1024 ** 2),
        "status": "OK",
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    if args.smoke:
        args.warmup = 3
        args.groups = 3
        args.group_iters = 5

    dtype = torch.bfloat16
    rows = {name: measure_row(name, args, dtype) for name in ROWS}
    dense_mean = rows["Standard_FFN"]["latency"]["mean_us"]
    for name, row in rows.items():
        row["speedup_vs_standard_mean"] = dense_mean / row["latency"]["mean_us"]
        row["speedup_vs_standard_median"] = (
            rows["Standard_FFN"]["latency"]["median_us"] / row["latency"]["median_us"]
        )

    payload = {
        "benchmark": "single_layer_ffn_latency",
        "version": "a800_cuda128",
        "shape": {
            "batch_size": 1,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
        "global_topk_k": GLOBAL_K,
        "moc_2_8_k": MOC_2_8_K,
        "grouped_a": GROUPED_A,
        "grouped_b": GROUPED_B,
        "moc_2_8_graph_config": MOC2_8_BALANCED_GRAPH_CONFIG,
        "dtype": str(dtype),
    },
        "rows": rows,
        "run_config": {
            "warmup": args.warmup,
            "groups": args.groups,
            "group_iters": args.group_iters,
            "compile_mode": args.compile_mode,
            "seed": args.seed,
        },
        "timing_method": {
            "timer": "torch.cuda.Event elapsed_time",
            "per_sample": "Each sample times group_iters repeated FFN calls and divides by group_iters.",
            "comparison_scope": (
                "Dense uses torch.compile; global MoC and MoC 2:8 use fixed-shape "
                "CUDA Graph runners for the optimized selected-channel paths."
            ),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Single-layer FFN latency")
    for name in ROWS:
        row = rows[name]
        print(
            f"  {name:12s}: mean={row['latency']['mean_us']:.3f} us, "
            f"speedup={row['speedup_vs_standard_mean']:.3f}x, "
            f"selection={row['selection']}"
        )


if __name__ == "__main__":
    main()
