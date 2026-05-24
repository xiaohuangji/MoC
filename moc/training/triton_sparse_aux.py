"""Triton auxiliary kernels for the MoC training path.

These kernels fuse the sparse helper work around the dense GEMMs:
selected-up gather, SiLU, sparse-to-dense scatter, and the matching backward
gather/scatter. Top-K and the dense GEMMs still use PyTorch/cuBLAS.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


DEFAULT_BLOCK = 256


@triton.jit
def _forward_sparse_aux_kernel(
    topk_vals,
    topk_idx16,
    u_full,
    u_sparse,
    s_sparse,
    z_sparse,
    z_full,
    n_elements: tl.constexpr,
    k_size: tl.constexpr,
    inter_size: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n_elements
    row = offs // k_size
    idx = tl.load(topk_idx16 + offs, mask=mask, other=0).to(tl.int32)
    g = tl.load(topk_vals + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_full + row * inter_size + idx, mask=mask, other=0.0)
    s = g * tl.sigmoid(g)
    z = s * u
    tl.store(u_sparse + offs, u, mask=mask)
    tl.store(s_sparse + offs, s, mask=mask)
    tl.store(z_sparse + offs, z, mask=mask)
    tl.store(z_full + row * inter_size + idx, z, mask=mask)


@triton.jit
def _scatter_sparse_to_dense_kernel(
    sparse,
    topk_idx16,
    dense,
    n_elements: tl.constexpr,
    k_size: tl.constexpr,
    inter_size: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n_elements
    row = offs // k_size
    idx = tl.load(topk_idx16 + offs, mask=mask, other=0).to(tl.int32)
    val = tl.load(sparse + offs, mask=mask, other=0.0)
    tl.store(dense + row * inter_size + idx, val, mask=mask)


@triton.jit
def _backward_sparse_aux_kernel(
    topk_idx16,
    g_sparse,
    u_sparse,
    s_sparse,
    grad_z_full,
    grad_g_full,
    grad_u_full,
    n_elements: tl.constexpr,
    k_size: tl.constexpr,
    inter_size: tl.constexpr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n_elements
    row = offs // k_size
    idx = tl.load(topk_idx16 + offs, mask=mask, other=0).to(tl.int32)
    gz = tl.load(grad_z_full + row * inter_size + idx, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_sparse + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_sparse + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_sparse + offs, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    silu_deriv = sig * (1.0 + g * (1.0 - sig))
    grad_g = u * gz * silu_deriv
    grad_u = s * gz
    tl.store(grad_g_full + row * inter_size + idx, grad_g, mask=mask)
    tl.store(grad_u_full + row * inter_size + idx, grad_u, mask=mask)


def _check_inputs(*tensors: torch.Tensor) -> None:
    if not all(t.is_cuda for t in tensors):
        raise ValueError("Triton MoC training kernels require CUDA tensors.")


def _shape_meta(topk_vals: torch.Tensor, inter_size: int) -> tuple[int, int, tuple[int, ...]]:
    k_size = topk_vals.shape[-1]
    rows = topk_vals.numel() // k_size
    dense_shape = (*topk_vals.shape[:-1], inter_size)
    return rows, k_size, dense_shape


def sparse_forward_aux(
    topk_vals: torch.Tensor,
    topk_idx16: torch.Tensor,
    u_full: torch.Tensor,
    inter_size: int,
    block: int = DEFAULT_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return u_sparse, s_sparse, z_sparse, z_full for MoC forward."""
    _check_inputs(topk_vals, topk_idx16, u_full)
    if topk_idx16.dtype != torch.int16:
        raise ValueError("topk_idx16 must be torch.int16.")
    rows, k_size, dense_shape = _shape_meta(topk_vals, inter_size)
    u_sparse = torch.empty_like(topk_vals)
    s_sparse = torch.empty_like(topk_vals)
    z_sparse = torch.empty_like(topk_vals)
    z_full = torch.empty(dense_shape, device=u_full.device, dtype=u_full.dtype)
    z_full.zero_()
    n_elements = rows * k_size
    grid = (triton.cdiv(n_elements, block),)
    _forward_sparse_aux_kernel[grid](
        topk_vals,
        topk_idx16,
        u_full,
        u_sparse,
        s_sparse,
        z_sparse,
        z_full,
        n_elements,
        k_size,
        inter_size,
        block,
    )
    return u_sparse, s_sparse, z_sparse, z_full


def scatter_sparse_to_dense(
    sparse: torch.Tensor,
    topk_idx16: torch.Tensor,
    inter_size: int,
    block: int = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Scatter a sparse [*, K] tensor into a dense [*, inter_size] tensor."""
    _check_inputs(sparse, topk_idx16)
    if topk_idx16.dtype != torch.int16:
        raise ValueError("topk_idx16 must be torch.int16.")
    rows, k_size, dense_shape = _shape_meta(sparse, inter_size)
    dense = torch.empty(dense_shape, device=sparse.device, dtype=sparse.dtype)
    dense.zero_()
    n_elements = rows * k_size
    grid = (triton.cdiv(n_elements, block),)
    _scatter_sparse_to_dense_kernel[grid](
        sparse,
        topk_idx16,
        dense,
        n_elements,
        k_size,
        inter_size,
        block,
    )
    return dense


def sparse_backward_aux(
    topk_idx16: torch.Tensor,
    g_sparse: torch.Tensor,
    u_sparse: torch.Tensor,
    s_sparse: torch.Tensor,
    grad_z_full: torch.Tensor,
    inter_size: int,
    block: int = DEFAULT_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dense grad_g and grad_u for MoC backward."""
    _check_inputs(topk_idx16, g_sparse, u_sparse, s_sparse, grad_z_full)
    if topk_idx16.dtype != torch.int16:
        raise ValueError("topk_idx16 must be torch.int16.")
    rows, k_size, dense_shape = _shape_meta(g_sparse, inter_size)
    grad_g = torch.empty(dense_shape, device=grad_z_full.device, dtype=grad_z_full.dtype)
    grad_u = torch.empty(dense_shape, device=grad_z_full.device, dtype=grad_z_full.dtype)
    grad_g.zero_()
    grad_u.zero_()
    n_elements = rows * k_size
    grid = (triton.cdiv(n_elements, block),)
    _backward_sparse_aux_kernel[grid](
        topk_idx16,
        g_sparse,
        u_sparse,
        s_sparse,
        grad_z_full,
        grad_g,
        grad_u,
        n_elements,
        k_size,
        inter_size,
        block,
    )
    return grad_g, grad_u

