"""Shared optimized global-MoC CUDA operators.

These operators are the reusable version of the the single-layer benchmark optimized
ordinary MoC path: CUB BF16 global Top-K, custom CUDA selected Up+SiLU, and
custom CUDA selected Down.  They are registered as ``torch.library.custom_op``
so end-to-end decode benchmarks can call the same algorithm from inside
``torch.compile`` without importing code from another experiment directory.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXT = None
_NATIVE_FAKE_REGISTERED = False


def load_optimized_global_moc_extension(build_dir: str | Path | None = None):
    """Load the C++/CUDA extension for optimized ordinary global MoC."""
    global _EXT
    if _EXT is not None:
        return _EXT

    source_dir = Path(__file__).resolve().parent / "cuda"
    if build_dir is None:
        build_dir = Path.home() / ".cache" / "moc" / "optimized_global_moc_cuda"
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    _EXT = load(
        name="moc_global_moc_ext",
        sources=[
            str(source_dir / "optimized_global_moc_ext.cpp"),
            str(source_dir / "optimized_global_moc_ext.cu"),
        ],
        build_directory=str(build_dir),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _EXT


def _register_native_fake_ops() -> None:
    global _NATIVE_FAKE_REGISTERED
    if _NATIVE_FAKE_REGISTERED:
        return

    @torch.library.register_fake("moc_native::optimized_global_topk_bf16")
    def _native_topk_fake(
        scores: torch.Tensor,
        k: int,
    ) -> list[torch.Tensor]:
        batch = scores.shape[0]
        values = torch.empty(batch, k, device=scores.device, dtype=scores.dtype)
        indices = torch.empty(batch, k, device=scores.device, dtype=torch.int64)
        return [values, indices]

    @torch.library.register_fake("moc_native::optimized_global_selected_up_silu_bf16")
    def _native_selected_up_fake(
        x: torch.Tensor,
        topk_vals: torch.Tensor,
        topk_idx: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> torch.Tensor:
        del topk_idx, up_weight
        return torch.empty_like(topk_vals, dtype=x.dtype)

    @torch.library.register_fake("moc_native::optimized_global_selected_down_bf16")
    def _native_selected_down_fake(
        sparse_z: torch.Tensor,
        topk_idx: torch.Tensor,
        down_weight_t: torch.Tensor,
    ) -> torch.Tensor:
        del topk_idx
        batch = sparse_z.shape[0]
        hidden = down_weight_t.shape[1]
        return torch.empty(batch, hidden, device=sparse_z.device, dtype=sparse_z.dtype)

    @torch.library.register_fake("moc_native::optimized_global_after_gate_bf16")
    def _native_after_gate_fake(
        x: torch.Tensor,
        gate_scores: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight_t: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        del gate_scores, up_weight, k
        batch = x.shape[0]
        hidden = down_weight_t.shape[1]
        return torch.empty(batch, hidden, device=x.device, dtype=x.dtype)

    _NATIVE_FAKE_REGISTERED = True


def ensure_native_ops_ready() -> None:
    """Load direct C++ dispatcher ops and register fake kernels for compile."""
    load_optimized_global_moc_extension()
    _register_native_fake_ops()


def make_optimized_global_moc_graph_runner(module, batch_size: int = 1):
    """Capture the optimized ordinary global-MoC FFN path in a CUDA Graph.

    The returned callable accepts a fixed batch-size tensor and replays the
    same CUB/CUDA path used by the single-layer benchmark's final ordinary MoC row.
    """
    ext = load_optimized_global_moc_extension()
    if module.config.grouped:
        raise ValueError("optimized global MoC graph runner requires ungrouped MoC")
    module._ensure_down_weight_t()

    hidden = module.config.hidden_size
    k = module.config.k
    device = module.gate_proj.weight.device
    dtype = module.gate_proj.weight.dtype
    x_buf = torch.empty(batch_size, hidden, device=device, dtype=dtype)
    up_w = module.up_proj.weight.contiguous()
    down_t = module._down_weight_t.contiguous()

    def path():
        gate = module.gate_proj(x_buf).contiguous()
        topk_vals, topk_idx = ext.cub_topk_bf16_512x11(gate, k)
        sparse_z = ext.selected_up_silu_bf16(
            x_buf,
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
        y_buf = path()

    def runner(x_in: torch.Tensor) -> torch.Tensor:
        x_2d, original_shape = module._flatten_decode_input(x_in)
        if x_2d.shape[0] != batch_size:
            raise ValueError(
                f"runner built for batch_size={batch_size}, got {x_2d.shape[0]}"
            )
        x_buf.copy_(x_2d)
        graph.replay()
        return module._restore_output_shape(y_buf, original_shape)

    return runner


@torch.library.custom_op("moc::optimized_global_topk_bf16", mutates_args=())
def optimized_global_topk_bf16(
    scores: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-shape BF16 global Top-K via CUB.

    Args:
        scores: ``[B, intermediate]`` BF16 CUDA gate scores.
        k: number of globally selected channels.
    """
    ext = load_optimized_global_moc_extension()
    return ext.cub_topk_bf16_512x11(scores.contiguous(), int(k))


@optimized_global_topk_bf16.register_fake
def _optimized_global_topk_bf16_fake(
    scores: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = scores.shape[0]
    values = torch.empty(batch, k, device=scores.device, dtype=scores.dtype)
    indices = torch.empty(batch, k, device=scores.device, dtype=torch.int64)
    return values, indices


@torch.library.custom_op("moc::optimized_global_selected_up_silu_bf16", mutates_args=())
def optimized_global_selected_up_silu_bf16(
    x: torch.Tensor,
    topk_vals: torch.Tensor,
    topk_idx: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute selected Up projection and SiLU for ordinary global MoC."""
    ext = load_optimized_global_moc_extension()
    return ext.selected_up_silu_bf16(
        x.contiguous(),
        topk_vals.contiguous(),
        topk_idx.contiguous(),
        up_weight.contiguous(),
    )


@optimized_global_selected_up_silu_bf16.register_fake
def _optimized_global_selected_up_silu_bf16_fake(
    x: torch.Tensor,
    topk_vals: torch.Tensor,
    topk_idx: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    del topk_idx, up_weight
    return torch.empty_like(topk_vals, dtype=x.dtype)


@torch.library.custom_op("moc::optimized_global_selected_down_bf16", mutates_args=())
def optimized_global_selected_down_bf16(
    sparse_z: torch.Tensor,
    topk_idx: torch.Tensor,
    down_weight_t: torch.Tensor,
) -> torch.Tensor:
    """Compute selected Down projection for ordinary global MoC."""
    ext = load_optimized_global_moc_extension()
    return ext.selected_down_bf16_h32_k16(
        sparse_z.contiguous(),
        topk_idx.contiguous(),
        down_weight_t.contiguous(),
    )


@optimized_global_selected_down_bf16.register_fake
def _optimized_global_selected_down_bf16_fake(
    sparse_z: torch.Tensor,
    topk_idx: torch.Tensor,
    down_weight_t: torch.Tensor,
) -> torch.Tensor:
    del topk_idx
    batch = sparse_z.shape[0]
    hidden = down_weight_t.shape[1]
    return torch.empty(batch, hidden, device=sparse_z.device, dtype=sparse_z.dtype)

