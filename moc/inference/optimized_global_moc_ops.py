"""BF16 global Top-K and selected-projection CUDA operators."""
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

    @torch.library.register_fake("moc_native::optimized_global_after_gate_bf16_out")
    def _native_after_gate_fake(
        x: torch.Tensor,
        gate_scores: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight_t: torch.Tensor,
        idx: torch.Tensor,
        sparse_z: torch.Tensor,
        out: torch.Tensor,
        k: int,
    ) -> None:
        return None

    _NATIVE_FAKE_REGISTERED = True


def ensure_native_ops_ready() -> None:
    """Load the CUDA dispatcher op and its compilation metadata."""
    load_optimized_global_moc_extension()
    _register_native_fake_ops()


def optimized_global_after_gate_bf16(
    x: torch.Tensor,
    gate_scores: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight_t: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Evaluate selected projections with compiler-visible temporary storage."""
    idx = torch.empty((x.shape[0], k), device=x.device, dtype=torch.int32)
    sparse_z = torch.empty((x.shape[0], k), device=x.device, dtype=x.dtype)
    out = torch.empty_like(x)
    torch.ops.moc_native.optimized_global_after_gate_bf16_out(
        x, gate_scores, up_weight, down_weight_t, idx, sparse_z, out, k,
    )
    return out

