"""Grouped MoC Triton operators used by the inference benchmarks.

The functions in this file wrap the final grouped top-2-of-8 kernels with
`torch.library.triton_op` so they can be used inside `torch.compile`.
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton

from .triton_fused_ffn_kernels import (
    _fused_gate_top2_of_8_selected_up_silu_kernel,
    _selected_down_from_sparse_z_kernel,
)


_GROUPED_BLOCK_B = 16
_GROUPED_BLOCK_G = 16
_GROUPED_BLOCK_H = 128
_GROUPED_DOWN_BLOCK_K = 128
_GROUPED_DOWN_BLOCK_H = 16


@torch.library.triton_op("moc::fused_grouped_top2_up_silu", mutates_args={})
def fused_grouped_top2_up_silu(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused gate projection, grouped top-2 selection, selected up, and SiLU."""
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

    batch, hidden = x.shape
    intermediate, _ = gate_weight.shape
    if intermediate % 8 != 0:
        raise ValueError(f"intermediate={intermediate} must be divisible by 8")

    groups = intermediate // 8
    k = groups * 2
    topk_idx = torch.empty(batch, k, device=x.device, dtype=torch.int64)
    sparse_z = torch.empty(batch, k, device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(batch, _GROUPED_BLOCK_B), triton.cdiv(groups, _GROUPED_BLOCK_G))
    torch.library.wrap_triton(_fused_gate_top2_of_8_selected_up_silu_kernel)[grid](
        x,
        gate_weight,
        up_weight,
        topk_idx,
        sparse_z,
        batch,
        hidden,
        groups,
        x.stride(0),
        x.stride(1),
        gate_weight.stride(0),
        gate_weight.stride(1),
        up_weight.stride(0),
        up_weight.stride(1),
        topk_idx.stride(0),
        topk_idx.stride(1),
        sparse_z.stride(0),
        sparse_z.stride(1),
        BLOCK_B=_GROUPED_BLOCK_B,
        BLOCK_G=_GROUPED_BLOCK_G,
        BLOCK_H=_GROUPED_BLOCK_H,
    )
    return topk_idx, sparse_z


@fused_grouped_top2_up_silu.register_fake
def _fused_grouped_top2_up_silu_fake(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del up_weight
    batch = x.shape[0]
    intermediate = gate_weight.shape[0]
    k = (intermediate // 8) * 2
    topk_idx = torch.empty(batch, k, device=x.device, dtype=torch.int64)
    sparse_z = torch.empty(batch, k, device=x.device, dtype=x.dtype)
    return topk_idx, sparse_z


@torch.library.triton_op("moc::selected_grouped_down", mutates_args={})
def selected_grouped_down(
    sparse_z: torch.Tensor,
    idx: torch.Tensor,
    w_down_t: torch.Tensor,
) -> torch.Tensor:
    """Selected down projection for the grouped MoC sparse intermediate."""
    assert sparse_z.is_cuda and idx.is_cuda and w_down_t.is_cuda
    assert sparse_z.dtype == torch.bfloat16 and w_down_t.dtype == torch.bfloat16
    assert idx.dtype == torch.int64

    batch, k = sparse_z.shape
    intermediate, hidden = w_down_t.shape
    out = torch.empty(batch, hidden, device=sparse_z.device, dtype=torch.bfloat16)
    grid = (batch, triton.cdiv(hidden, _GROUPED_DOWN_BLOCK_H))
    torch.library.wrap_triton(_selected_down_from_sparse_z_kernel)[grid](
        sparse_z,
        idx,
        w_down_t,
        out,
        batch,
        k,
        hidden,
        intermediate,
        sparse_z.stride(0),
        sparse_z.stride(1),
        idx.stride(0),
        idx.stride(1),
        w_down_t.stride(0),
        w_down_t.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_K=_GROUPED_DOWN_BLOCK_K,
        BLOCK_H=_GROUPED_DOWN_BLOCK_H,
    )
    return out


@selected_grouped_down.register_fake
def _selected_grouped_down_fake(
    sparse_z: torch.Tensor,
    idx: torch.Tensor,
    w_down_t: torch.Tensor,
) -> torch.Tensor:
    del idx
    batch = sparse_z.shape[0]
    hidden = w_down_t.shape[1]
    return torch.empty(batch, hidden, device=sparse_z.device, dtype=sparse_z.dtype)
