"""Triton auxiliary kernels for the MoC training path.

These kernels fuse the sparse helper work around the dense GEMMs:
selected-up gather, SiLU, sparse-to-dense scatter, and the matching backward
gather/scatter. Top-K and dense GEMMs are handled by PyTorch/cuBLAS.
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
    round_bf16: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n_elements
    row = offs // k_size
    idx = tl.load(topk_idx16 + offs, mask=mask, other=0).to(tl.int32)
    g = tl.load(topk_vals + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_full + row * inter_size + idx, mask=mask, other=0.0)
    s = g * tl.sigmoid(g)
    if round_bf16:
        s_store = s.to(tl.bfloat16)
        z = (s_store * u).to(tl.bfloat16)
    else:
        s_store = s
        z = s * u
    tl.store(u_sparse + offs, u, mask=mask)
    tl.store(s_sparse + offs, s_store, mask=mask)
    tl.store(z_sparse + offs, z, mask=mask)
    tl.store(z_full + row * inter_size + idx, z, mask=mask)


@triton.jit
def _gather_dense_to_sparse_kernel(
    dense,
    topk_idx16,
    sparse,
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
    val = tl.load(dense + row * inter_size + idx, mask=mask, other=0.0)
    tl.store(sparse + offs, val, mask=mask)


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
def _gather_mul_dense_to_sparse_kernel(
    dense,
    topk_idx16,
    u_sparse,
    s_sparse,
    grad_s_sparse,
    grad_u_sparse,
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
    gz = tl.load(dense + row * inter_size + idx, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_sparse + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_sparse + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(grad_s_sparse + offs, u * gz, mask=mask)
    tl.store(grad_u_sparse + offs, s * gz, mask=mask)


@triton.jit
def _scatter_two_sparse_to_dense_kernel(
    sparse_a,
    sparse_b,
    topk_idx16,
    dense_a,
    dense_b,
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
    val_a = tl.load(sparse_a + offs, mask=mask, other=0.0)
    val_b = tl.load(sparse_b + offs, mask=mask, other=0.0)
    dense_offs = row * inter_size + idx
    tl.store(dense_a + dense_offs, val_a, mask=mask)
    tl.store(dense_b + dense_offs, val_b, mask=mask)


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
    round_bf16: tl.constexpr,
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
    if round_bf16:
        g_b = g.to(tl.bfloat16)
        gz_b = gz.to(tl.bfloat16)
        u_b = u.to(tl.bfloat16)
        s_b = s.to(tl.bfloat16)
        grad_s = (u_b * gz_b).to(tl.bfloat16)
        grad_u = (s_b * gz_b).to(tl.bfloat16)
        sig = (1.0 / (1.0 + tl.exp(-g))).to(tl.bfloat16)
        one_minus_sig = (1.0 - sig).to(tl.bfloat16)
        g_term = (g_b * one_minus_sig).to(tl.bfloat16)
        inner = (1.0 + g_term).to(tl.bfloat16)
        silu_deriv = (sig * inner).to(tl.bfloat16)
        grad_g = (grad_s * silu_deriv).to(tl.bfloat16)
    else:
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
    round_bf16 = topk_vals.dtype == torch.bfloat16
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
        round_bf16,
    )
    return u_sparse, s_sparse, z_sparse, z_full


def gather_dense_to_sparse(
    dense: torch.Tensor,
    topk_idx16: torch.Tensor,
    block: int = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Gather a dense [*, inter_size] tensor into sparse [*, K] with int16 indices."""
    _check_inputs(dense, topk_idx16)
    if topk_idx16.dtype != torch.int16:
        raise ValueError("topk_idx16 must be torch.int16.")
    k_size = topk_idx16.shape[-1]
    rows = topk_idx16.numel() // k_size
    if dense.numel() % rows != 0:
        raise ValueError(
            f"dense shape {tuple(dense.shape)} is incompatible with "
            f"topk_idx16 shape {tuple(topk_idx16.shape)}"
        )
    inter_size = dense.numel() // rows
    sparse = torch.empty(topk_idx16.shape, device=dense.device, dtype=dense.dtype)
    n_elements = rows * k_size
    grid = (triton.cdiv(n_elements, block),)
    _gather_dense_to_sparse_kernel[grid](
        dense,
        topk_idx16,
        sparse,
        n_elements,
        k_size,
        inter_size,
        block,
    )
    return sparse


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


def gather_mul_dense_to_sparse(
    dense: torch.Tensor,
    topk_idx16: torch.Tensor,
    u_sparse: torch.Tensor,
    s_sparse: torch.Tensor,
    block: int = DEFAULT_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather dense grad_z and form grad_s/grad_u sparse tensors."""
    _check_inputs(dense, topk_idx16, u_sparse, s_sparse)
    if topk_idx16.dtype != torch.int16:
        raise ValueError("topk_idx16 must be torch.int16.")
    k_size = topk_idx16.shape[-1]
    rows = topk_idx16.numel() // k_size
    if dense.numel() % rows != 0:
        raise ValueError(
            f"dense shape {tuple(dense.shape)} is incompatible with "
            f"topk_idx16 shape {tuple(topk_idx16.shape)}"
        )
    inter_size = dense.numel() // rows
    grad_s_sparse = torch.empty(topk_idx16.shape, device=dense.device, dtype=dense.dtype)
    grad_u_sparse = torch.empty(topk_idx16.shape, device=dense.device, dtype=dense.dtype)
    n_elements = rows * k_size
    grid = (triton.cdiv(n_elements, block),)
    _gather_mul_dense_to_sparse_kernel[grid](
        dense,
        topk_idx16,
        u_sparse,
        s_sparse,
        grad_s_sparse,
        grad_u_sparse,
        n_elements,
        k_size,
        inter_size,
        block,
    )
    return grad_s_sparse, grad_u_sparse


def scatter_two_sparse_to_dense(
    sparse_a: torch.Tensor,
    sparse_b: torch.Tensor,
    topk_idx16: torch.Tensor,
    inter_size: int,
    block: int = DEFAULT_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter two sparse [*, K] tensors into dense [*, inter_size] tensors."""
    _check_inputs(sparse_a, sparse_b, topk_idx16)
    if topk_idx16.dtype != torch.int16:
        raise ValueError("topk_idx16 must be torch.int16.")
    rows, k_size, dense_shape = _shape_meta(sparse_a, inter_size)
    dense_a = torch.empty(dense_shape, device=sparse_a.device, dtype=sparse_a.dtype)
    dense_b = torch.empty(dense_shape, device=sparse_b.device, dtype=sparse_b.dtype)
    dense_a.zero_()
    dense_b.zero_()
    n_elements = rows * k_size
    grid = (triton.cdiv(n_elements, block),)
    _scatter_two_sparse_to_dense_kernel[grid](
        sparse_a,
        sparse_b,
        topk_idx16,
        dense_a,
        dense_b,
        n_elements,
        k_size,
        inter_size,
        block,
    )
    return dense_a, dense_b


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
    if g_sparse.dtype in (torch.bfloat16, torch.float16):
        # Correctness-first fallback: PyTorch low-precision elementwise ops
        # materialize BF16/FP16 intermediates between the SiLU-gradient steps.
        # A fused Triton expression is mathematically close but not bitwise
        # equivalent, and the small gradient differences can drift during PPL
        # training. Index movement still uses int16-aware Triton helpers so the
        # path avoids PyTorch's int64 gather/scatter boundary.
        grad_s_sparse, grad_u_sparse = gather_mul_dense_to_sparse(
            grad_z_full,
            topk_idx16,
            u_sparse,
            s_sparse,
            block=block,
        )
        sig = torch.sigmoid(g_sparse)
        silu_deriv = sig * (1.0 + g_sparse * (1.0 - sig))
        grad_g_sparse = grad_s_sparse * silu_deriv
        grad_g, grad_u = scatter_two_sparse_to_dense(
            grad_g_sparse,
            grad_u_sparse,
            topk_idx16,
            inter_size,
            block=block,
        )
        return grad_g, grad_u
    rows, k_size, dense_shape = _shape_meta(g_sparse, inter_size)
    grad_g = torch.empty(dense_shape, device=grad_z_full.device, dtype=grad_z_full.dtype)
    grad_u = torch.empty(dense_shape, device=grad_z_full.device, dtype=grad_z_full.dtype)
    grad_g.zero_()
    grad_u.zero_()
    n_elements = rows * k_size
    grid = (triton.cdiv(n_elements, block),)
    round_bf16 = g_sparse.dtype == torch.bfloat16
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
        round_bf16,
    )
    return grad_g, grad_u
