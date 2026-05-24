"""Optional RAFT/cuVS Top-K wrapper.

This module loads a pre-built `raft_select_k_ext.so` when available. The main
public benchmarks do not require this extension; it is kept as an optional
backend for users who want to compare Top-K implementations.

Public surface:
- has_raft_topk()
- load_raft_topk_extension()
- raft_topk(scores, k, sorted=False, algo=4) -> (values, indices)

The extension exposes `select_k_fp32_i64(scores, k, select_min, sorted, algo)`.
It requires fp32 scores and returns int64 indices, matching `torch.topk` on
float32 inputs.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import torch

_EXT = None
_LOAD_ERROR: Optional[str] = None
_LOAD_PATH: Optional[Path] = None

V08_ROOT = Path(__file__).resolve().parents[1]
_PRIMARY_SO = V08_ROOT / "cpp" / "build" / "raft_select_k_ext.so"
_FALLBACK_SO = Path.home() / ".cache" / "moc" / "raft_select_k_ext.so"


def _try_load(so_path: Path):
    # Module name MUST match TORCH_EXTENSION_NAME baked into the .so symbols
    # (`PyInit_raft_select_k_ext`); using a different name yields ImportError.
    spec = importlib.util.spec_from_file_location("raft_select_k_ext", str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_raft_topk_extension():
    """Load the RAFT/cuVS select_k extension.

    Returns the loaded module on success, or None on failure. The first call
    caches the result; subsequent calls reuse it.
    """
    global _EXT, _LOAD_ERROR, _LOAD_PATH
    if _EXT is not None:
        return _EXT
    # torch must be initialized for libc10/libtorch symbols.
    _ = torch  # noqa: F841

    candidates = [_PRIMARY_SO, _FALLBACK_SO]
    last_err: Optional[str] = None
    for cand in candidates:
        if not cand.exists():
            last_err = f"{cand} not found"
            continue
        try:
            _EXT = _try_load(cand)
            _LOAD_PATH = cand
            return _EXT
        except Exception as ex:
            last_err = f"{cand}: {type(ex).__name__}: {ex}"
            continue
    _LOAD_ERROR = last_err or "no candidate path"
    return None


def has_raft_topk() -> bool:
    return load_raft_topk_extension() is not None


def load_path() -> Optional[str]:
    load_raft_topk_extension()
    return None if _LOAD_PATH is None else str(_LOAD_PATH)


def load_error() -> Optional[str]:
    return _LOAD_ERROR


def raft_topk(
    scores: torch.Tensor,
    k: int,
    sorted: bool = False,
    algo: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise Top-K via cuvs::selection::select_k.

    scores must be a 2D CUDA tensor. fp32 is used directly; bf16/fp16 is cast
    to fp32 (extension only exposes fp32 signature). Returns
    (values, indices) where values is fp32 and indices is int64.
    select_min=False, so we get top-K largest scores, mirroring
    `torch.topk(largest=True)`.

    SelectAlgo (cuvs/raft) enum:
      0=kAuto, 1=kRadix8bits, 2=kRadix11bits, 3=kRadix11bitsExtraPass,
      4=kWarpAuto, 5=kWarpImmediate, 6=kWarpFiltered.
    Warpsort variants (4/5/6) only support K<=256; for K=1024 use 0..3.

    sorted=False is the cuVS default and the fastest setting; use sorted=True
    only if downstream order must match `torch.topk`'s descending order.
    """
    ext = load_raft_topk_extension()
    if ext is None:
        raise RuntimeError(
            f"RAFT Top-K extension unavailable: {_LOAD_ERROR}"
        )
    if scores.dim() != 2:
        raise ValueError(f"scores must be 2D, got {scores.dim()}D")
    if not scores.is_cuda:
        raise ValueError("scores must be a CUDA tensor")
    s = scores
    if s.dtype != torch.float32:
        s = s.to(torch.float32)
    if not s.is_contiguous():
        s = s.contiguous()
    return ext.select_k_fp32_i64(s, int(k), False, bool(sorted), int(algo))


# =====================================================================
# torch.library custom_op wrapper so that moc_inference_v26 (which calls
# RAFT in its top-K stage) can be traced by torch.compile / Inductor.
# Inductor cannot trace into the raw .so call; the custom_op makes it
# an opaque CUDA op with a register_fake shape signature. The wrapper
# does NOT mutate any input.
# =====================================================================


@torch.library.custom_op("moc::raft_topk_fp32", mutates_args=())
def raft_topk_fp32_op(
    scores_fp32: torch.Tensor, k: int, algo: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """scores_fp32: [B, N] float32 contiguous CUDA. Returns (vals_fp32 [B,K],
    idx_int64 [B,K]) 鈥?RAFT WarpSort/Radix select_k of the top-K largest.
    """
    ext = load_raft_topk_extension()
    if ext is None:
        raise RuntimeError(
            f"RAFT Top-K extension unavailable: {_LOAD_ERROR}"
        )
    return ext.select_k_fp32_i64(
        scores_fp32, int(k), False, False, int(algo),
    )


@raft_topk_fp32_op.register_fake
def _raft_topk_fp32_fake(
    scores_fp32: torch.Tensor, k: int, algo: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = scores_fp32.shape[0]
    return (
        torch.empty(B, k, device=scores_fp32.device, dtype=torch.float32),
        torch.empty(B, k, device=scores_fp32.device, dtype=torch.int64),
    )

