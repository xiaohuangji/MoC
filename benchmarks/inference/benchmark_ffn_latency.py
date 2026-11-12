"""Single-token FFN latency with compilation and CUDA Graph replay."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.inference.benchmark_decode import (  # noqa: E402
    DecoderLayer, GLOBAL_K, GROUPED_A, GROUPED_B, HIDDEN, INTERMEDIATE, ROW_SPECS,
)


class SingleFFN(torch.nn.Module):
    forward = DecoderLayer._ffn_call

    def __init__(self, method: str):
        super().__init__()
        from moc.inference.inference_ffn import InferenceMoCSwiGLUFFN

        if method == "global_moc":
            from moc.inference.optimized_global_moc_ops import ensure_native_ops_ready

            ensure_native_ops_ready()
        elif method == "moc_2_8":
            import moc.inference.triton_grouped_moc_ops  # noqa: F401

        self.ffn_kind = method
        row = ROW_SPECS[method]
        kwargs = dict(hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, k=row["k"])
        if row["grouped_a"] is not None:
            kwargs.update(grouped_a=GROUPED_A, grouped_b=GROUPED_B)
        self.ffn = InferenceMoCSwiGLUFFN(**kwargs)

    def freeze_for_compile(self):
        self.ffn.freeze_for_compile()


CAPTURE_CALLS = 16


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", choices=list(ROW_SPECS),
                        default=["dense", "global_moc"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--groups", type=int, default=80)
    parser.add_argument("--replays-per-group", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Save a separate GPU-kernel breakdown")
    args = parser.parse_args(argv)
    if args.warmup < 0 or min(args.groups, args.replays_per_group, args.rounds, args.batch_size) < 1:
        parser.error("warmup must be nonnegative; groups, replays-per-group, rounds and batch-size must be positive")
    if len(args.methods) != len(set(args.methods)):
        parser.error("methods must not contain duplicates")
    if args.smoke:
        args.warmup, args.groups, args.replays_per_group, args.rounds = 3, 3, 2, 1
    return args


def summarize(samples):
    return dict(mean_us=statistics.mean(samples), median_us=statistics.median(samples),
                std_us=statistics.stdev(samples) if len(samples) > 1 else 0.0,
                num_groups=len(samples), all_us=samples)


def digest(tensor):
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def compile_call(module):
    module.freeze_for_compile()
    return torch.compile(module.forward, dynamic=True, options={"cpp_wrapper": True})


def assert_exact(actual, expected):
    if not torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
        raise AssertionError("Output differs from the compiled reference")


@torch.no_grad()
def capture_call(fn, x):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn(x)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(CAPTURE_CALLS):
            output = fn(x)
    return graph, output


def profile_window_kernels(events):
    window, = (e for e in events if e.get("name") == "ffn_profile_measure"
               and e.get("cat") == "user_annotation" and e.get("ph") == "X")
    begin, end = window["ts"], window["ts"] + window["dur"]
    kernels = sorted((e for e in events if e.get("cat") == "kernel"
                      and begin <= e["ts"] and e["ts"] + e["dur"] <= end), key=lambda e: e["ts"])
    return begin, end, kernels


@torch.no_grad()
def profile_kernels(graph, method, args):
    """Record kernel durations separately from the uninstrumented latency samples."""
    repeats = 20
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        graph.replay()
        torch.cuda.synchronize()
        with torch.profiler.record_function("ffn_profile_measure"):
            for _ in range(repeats):
                graph.replay()
            torch.cuda.synchronize()
    path = Path(args.out).with_suffix(f".{method}.trace.json")
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_text(encoding="utf-8"))["traceEvents"]
    begin, end, kernels = profile_window_kernels(events)
    calls = repeats * CAPTURE_CALLS
    # GEMM algorithms may launch different numbers of kernels for different batches.
    if not kernels or len(kernels) % calls:
        raise RuntimeError(f"Incomplete profiler capture: {len(kernels)} kernels for {calls} calls; inspect {path}")
    names = {}
    for kernel in kernels:
        names.setdefault(kernel["name"], []).append(kernel["dur"])
    return dict(scope="profiled GPU kernels; excludes host/launch gaps", repeats=repeats,
                window_start_us=begin, window_end_us=end,
                ffn_calls=calls, kernels_per_call=len(kernels) // calls,
                total_kernel_us_per_call=sum(e["dur"] for e in kernels) / calls,
                kernels=[dict(name=name, count=len(values),
                              us_per_call=sum(values) / calls)
                         for name, values in names.items()], trace_file=path.name)


@torch.no_grad()
def measure_row(method, args):
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    # Each method and round starts with identical parameters and input.
    torch.manual_seed(args.seed)
    module = SingleFFN(method).to(device=args.device, dtype=torch.bfloat16).eval()
    module.requires_grad_(False)
    x = torch.randn(args.batch_size, 1, HIDDEN, device=args.device, dtype=torch.bfloat16)
    parameter_hashes = {name: digest(value) for name, value in module.state_dict().items()}
    fn = compile_call(module)
    values = [x.clone()] + [torch.randn_like(x) for _ in range(3)]
    eager = module(x).clone()
    expected = []
    for value in values:
        x.copy_(value)
        expected.append(fn(x).clone())
    torch.testing.assert_close(expected[0], eager, rtol=0.03, atol=0.005)
    x.copy_(values[0])
    graph, output = capture_call(fn, x)
    for value, reference in zip(values, expected):
        x.copy_(value)
        graph.replay()
        torch.cuda.synchronize()
        assert_exact(output, reference)
    x.copy_(values[0])
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.groups):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.replays_per_group):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / (args.replays_per_group * CAPTURE_CALLS))
    assert_exact(output, expected[0])
    assert_exact(x, values[0])
    row = dict(method=method, batch_size=args.batch_size, selection=ROW_SPECS[method]["selection"],
               k=None if method == "dense" else ROW_SPECS[method]["k"],
               latency=summarize(samples), input_sha256=digest(x),
               parameter_sha256=parameter_hashes,
               correctness=dict(compiled_vs_eager_max_abs=float((expected[0]-eager).abs().max()),
                                rtol=0.03, atol=0.005, graph_changed_inputs_exact=len(values),
                                post_timing_exact=True))
    if args.profile and args.profile_round:
        row["kernel_profile"] = profile_kernels(graph, method, args)
    print(f"{method}: {row['latency']['mean_us']:.3f} us", flush=True)
    return row


def aggregate(rounds):
    methods = list(rounds[0])
    if any(set(r) != set(methods) for r in rounds):
        raise ValueError("Every round must contain all requested methods")
    rows = {}
    identity = None
    for method in methods:
        samples = []
        for round_result in rounds:
            row = round_result[method]
            current = (row["input_sha256"], row["parameter_sha256"])
            if identity is None:
                identity = current
            if current != identity:
                raise ValueError("Input or parameters differ across methods/rounds")
            samples.extend(row["latency"]["all_us"])
        rows[method] = dict(latency=summarize(samples))
    if "dense" in rows:
        dense = rows["dense"]["latency"]["mean_us"]
        for row in rows.values():
            row["speedup_vs_dense"] = dense / row["latency"]["mean_us"]
    return rows


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    torch.set_float32_matmul_precision("high")
    payload = dict(benchmark="single_layer_ffn_latency", status="RUNNING",
                   run_config=vars(args).copy(),
                   shape=dict(batch_size=args.batch_size, seq_len=1, hidden=HIDDEN,
                              intermediate=INTERMEDIATE, global_topk_k=GLOBAL_K, dtype="bf16"),
                   execution=dict(compile_mode="default", dynamic=True, cpp_wrapper=True,
                                  cuda_graph=True, capture_calls=CAPTURE_CALLS,
                                  ffn_path="DecoderLayer._ffn_call"),
                   timing_method="CUDA Event; all-group mean per FFN call; fixed GPU-resident input",
                   environment=dict(gpu=torch.cuda.get_device_name(args.device),
                                    torch_version=torch.__version__, cuda_version=torch.version.cuda),
                   rounds=[])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for round_index in range(args.rounds):
            args.profile_round = round_index == args.rounds - 1
            methods = args.methods if round_index % 2 == 0 else list(reversed(args.methods))
            round_result = {}
            payload["rounds"].append(round_result)
            for method in methods:
                round_result[method] = measure_row(method, args)
                out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        payload["rows"] = aggregate(payload["rounds"])
        payload["status"] = "DONE"
    except BaseException as error:
        payload.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
