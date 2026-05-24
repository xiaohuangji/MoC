"""a800_cuda128 — Wrap v_14 Triton kernels using torch.library.triton_op (torch 2.8+).

Why: v_22's `torch.library.custom_op(mutates_args=())` makes Inductor treat
the wrapped Triton kernel as an opaque call boundary. When MoC FFN appears
inside `torch.compile(reduce-overhead)` 24 times in a transformer decoder,
each call eats Python dispatch + CUDA Graph capture overhead that doesn't
get fused with surrounding ops.

`torch.library.triton_op` (torch 2.8 GA) lets Inductor *inline* the Triton
kernel call into the compiled graph, removing the opaque-box boundary while
still capturing the kernel correctly under CUDA Graph mode.

We re-wrap v_14's two kernels (`fused_gate_top2of8_selected_up_silu` and
`selected_down_from_sparse_z`) with `triton_op`. Kernel bodies are unchanged;
only the Python registration changes.

Notes:
  - triton_op requires the wrapped function to call the @triton.jit kernel
    via `torch.library.wrap_triton(...)`.
  - Output shapes and dtypes must be statically determinable from input
    shapes (we provide `register_fake`).
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from .triton_fused_ffn_kernels import (
    _fused_gate_top2_of_8_selected_up_silu_kernel,
    _selected_down_from_sparse_z_kernel,
)


# v_14 swept defaults (same as v_22 wrappers)
_BLOCK_B = 16
_BLOCK_G = 16
_BLOCK_H = 64
_DOWN_BLOCK_K = 64
_DOWN_BLOCK_H = 32

# the single-layer benchmark final MoC 2:8 alignment config.
_BALANCED_BLOCK_B = 16
_BALANCED_BLOCK_G = 16
_BALANCED_BLOCK_H = 128
_BALANCED_DOWN_BLOCK_K = 128
_BALANCED_DOWN_BLOCK_H = 16


# =============================================================================
# Kernel A: fused gate + top-2 + selected_up + silu
# =============================================================================


@torch.library.triton_op("moc::fused_gate_top2_up_silu_torch28", mutates_args={})
def fused_gate_top2_up_silu_torch28(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """v_14 Kernel A wrapped via torch.library.triton_op (Inductor-inlinable)."""
    if not (x.is_cuda and gate_weight.is_cuda and up_weight.is_cuda):
        raise ValueError("inputs must be CUDA")
    if x.dtype != gate_weight.dtype or x.dtype != up_weight.dtype:
        raise ValueError("dtype mismatch")
    if not gate_weight.is_contiguous():
        gate_weight = gate_weight.contiguous()
    if not up_weight.is_contiguous():
        up_weight = up_weight.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()

    B, H = x.shape
    I_g, _ = gate_weight.shape
    if I_g % 8 != 0:
        raise ValueError(f"intermediate I={I_g} must be divisible by 8")

    groups = I_g // 8
    K = groups * 2

    topk_idx = torch.empty(B, K, device=x.device, dtype=torch.int64)
    sparse_z = torch.empty(B, K, device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(B, _BLOCK_B), triton.cdiv(groups, _BLOCK_G))
    torch.library.wrap_triton(_fused_gate_top2_of_8_selected_up_silu_kernel)[grid](
        x, gate_weight, up_weight, topk_idx, sparse_z,
        B, H, groups,
        x.stride(0), x.stride(1),
        gate_weight.stride(0), gate_weight.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        topk_idx.stride(0), topk_idx.stride(1),
        sparse_z.stride(0), sparse_z.stride(1),
        BLOCK_B=_BLOCK_B, BLOCK_G=_BLOCK_G, BLOCK_H=_BLOCK_H,
    )
    return topk_idx, sparse_z


@fused_gate_top2_up_silu_torch28.register_fake
def _fused_gate_top2_up_silu_torch28_fake(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = x.shape[0]
    I = gate_weight.shape[0]
    K = (I // 8) * 2
    topk_idx = torch.empty(B, K, device=x.device, dtype=torch.int64)
    sparse_z = torch.empty(B, K, device=x.device, dtype=x.dtype)
    return topk_idx, sparse_z


# =============================================================================
# Kernel B: selected down from sparse_z
# =============================================================================


@torch.library.triton_op("moc::selected_down_torch28", mutates_args={})
def selected_down_torch28(
    sparse_z: torch.Tensor,
    idx: torch.Tensor,
    w_down_t: torch.Tensor,
) -> torch.Tensor:
    """v_14 Kernel B wrapped via torch.library.triton_op."""
    assert sparse_z.is_cuda and idx.is_cuda and w_down_t.is_cuda
    assert sparse_z.dtype == torch.bfloat16 and w_down_t.dtype == torch.bfloat16
    assert idx.dtype == torch.int64

    B, K = sparse_z.shape
    I, H = w_down_t.shape
    out = torch.empty(B, H, device=sparse_z.device, dtype=torch.bfloat16)

    grid = (B, triton.cdiv(H, _DOWN_BLOCK_H))
    torch.library.wrap_triton(_selected_down_from_sparse_z_kernel)[grid](
        sparse_z, idx, w_down_t, out,
        B, K, H, I,
        sparse_z.stride(0), sparse_z.stride(1),
        idx.stride(0), idx.stride(1),
        w_down_t.stride(0), w_down_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_K=_DOWN_BLOCK_K, BLOCK_H=_DOWN_BLOCK_H,
    )
    return out


@selected_down_torch28.register_fake
def _selected_down_torch28_fake(
    sparse_z: torch.Tensor,
    idx: torch.Tensor,
    w_down_t: torch.Tensor,
) -> torch.Tensor:
    B = sparse_z.shape[0]
    H = w_down_t.shape[1]
    return torch.empty(B, H, device=sparse_z.device, dtype=sparse_z.dtype)


@torch.library.triton_op("moc::fused_gate_top2_up_silu_balanced_torch28", mutates_args={})
def fused_gate_top2_up_silu_balanced_torch28(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """v_14 Kernel A with the the single-layer benchmark balanced A800 block config."""
    if not (x.is_cuda and gate_weight.is_cuda and up_weight.is_cuda):
        raise ValueError("inputs must be CUDA")
    if x.dtype != gate_weight.dtype or x.dtype != up_weight.dtype:
        raise ValueError("dtype mismatch")
    if not gate_weight.is_contiguous():
        gate_weight = gate_weight.contiguous()
    if not up_weight.is_contiguous():
        up_weight = up_weight.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()

    B, H = x.shape
    I_g, _ = gate_weight.shape
    if I_g % 8 != 0:
        raise ValueError(f"intermediate I={I_g} must be divisible by 8")

    groups = I_g // 8
    K = groups * 2
    topk_idx = torch.empty(B, K, device=x.device, dtype=torch.int64)
    sparse_z = torch.empty(B, K, device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(B, _BALANCED_BLOCK_B), triton.cdiv(groups, _BALANCED_BLOCK_G))
    torch.library.wrap_triton(_fused_gate_top2_of_8_selected_up_silu_kernel)[grid](
        x, gate_weight, up_weight, topk_idx, sparse_z,
        B, H, groups,
        x.stride(0), x.stride(1),
        gate_weight.stride(0), gate_weight.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        topk_idx.stride(0), topk_idx.stride(1),
        sparse_z.stride(0), sparse_z.stride(1),
        BLOCK_B=_BALANCED_BLOCK_B,
        BLOCK_G=_BALANCED_BLOCK_G,
        BLOCK_H=_BALANCED_BLOCK_H,
    )
    return topk_idx, sparse_z


@fused_gate_top2_up_silu_balanced_torch28.register_fake
def _fused_gate_top2_up_silu_balanced_torch28_fake(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del up_weight
    B = x.shape[0]
    I = gate_weight.shape[0]
    K = (I // 8) * 2
    topk_idx = torch.empty(B, K, device=x.device, dtype=torch.int64)
    sparse_z = torch.empty(B, K, device=x.device, dtype=x.dtype)
    return topk_idx, sparse_z


@torch.library.triton_op("moc::selected_down_balanced_torch28", mutates_args={})
def selected_down_balanced_torch28(
    sparse_z: torch.Tensor,
    idx: torch.Tensor,
    w_down_t: torch.Tensor,
) -> torch.Tensor:
    """v_14 Kernel B with the the single-layer benchmark balanced A800 block config."""
    assert sparse_z.is_cuda and idx.is_cuda and w_down_t.is_cuda
    assert sparse_z.dtype == torch.bfloat16 and w_down_t.dtype == torch.bfloat16
    assert idx.dtype == torch.int64

    B, K = sparse_z.shape
    I, H = w_down_t.shape
    out = torch.empty(B, H, device=sparse_z.device, dtype=torch.bfloat16)
    grid = (B, triton.cdiv(H, _BALANCED_DOWN_BLOCK_H))
    torch.library.wrap_triton(_selected_down_from_sparse_z_kernel)[grid](
        sparse_z, idx, w_down_t, out,
        B, K, H, I,
        sparse_z.stride(0), sparse_z.stride(1),
        idx.stride(0), idx.stride(1),
        w_down_t.stride(0), w_down_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_K=_BALANCED_DOWN_BLOCK_K,
        BLOCK_H=_BALANCED_DOWN_BLOCK_H,
    )
    return out


@selected_down_balanced_torch28.register_fake
def _selected_down_balanced_torch28_fake(
    sparse_z: torch.Tensor,
    idx: torch.Tensor,
    w_down_t: torch.Tensor,
) -> torch.Tensor:
    del idx
    B = sparse_z.shape[0]
    H = w_down_t.shape[1]
    return torch.empty(B, H, device=sparse_z.device, dtype=sparse_z.dtype)

