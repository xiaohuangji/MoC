from __future__ import annotations

import torch
import triton
import triton.language as tl


# ============================================================================
# Selected-down: y[b,h] = sum_k sparse_z[b,k] * W_down_t[idx[b,k], h]
# Grid: (B, ceil(H/BLOCK_H)). Inner loop over K. No [B,K,H] materialization.
# ============================================================================

@triton.jit
def _selected_down_gather_dot_kernel(
    sparse_z_ptr, idx_ptr, w_ptr, out_ptr,
    B, K, H, I,
    s_zn, s_zk,
    s_in, s_ik,
    s_wi, s_wh,
    s_on, s_oh,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        z_vec = tl.load(sparse_z_ptr + pid_b * s_zn + k_offs * s_zk,
                        mask=k_mask, other=0.0)
        idx_vec = tl.load(idx_ptr + pid_b * s_in + k_offs * s_ik,
                          mask=k_mask, other=0).to(tl.int64)
        w_tile = tl.load(
            w_ptr + idx_vec[:, None] * s_wi + h_offs[None, :] * s_wh,
            mask=k_mask[:, None] & h_mask[None, :], other=0.0,
        )
        acc += tl.sum(z_vec[:, None].to(tl.float32) * w_tile.to(tl.float32), axis=0)

    tl.store(out_ptr + pid_b * s_on + h_offs * s_oh,
             acc.to(tl.bfloat16), mask=h_mask)


def selected_down_gather_dot(sparse_z: torch.Tensor, idx: torch.Tensor,
                              w_down_t: torch.Tensor,
                              BLOCK_K: int = 32, BLOCK_H: int = 64) -> torch.Tensor:
    """Compute y[b,h] = sum_k sparse_z[b,k] * w_down_t[idx[b,k], h] without materializing [B,K,H]."""
    assert sparse_z.is_cuda and idx.is_cuda and w_down_t.is_cuda
    assert sparse_z.dtype == torch.bfloat16 and w_down_t.dtype == torch.bfloat16
    assert idx.dtype == torch.int64
    B, K = sparse_z.shape
    I, H = w_down_t.shape
    out = torch.empty(B, H, device=sparse_z.device, dtype=torch.bfloat16)

    grid = (B, triton.cdiv(H, BLOCK_H))
    _selected_down_gather_dot_kernel[grid](
        sparse_z, idx, w_down_t, out,
        B, K, H, I,
        sparse_z.stride(0), sparse_z.stride(1),
        idx.stride(0), idx.stride(1),
        w_down_t.stride(0), w_down_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
    )
    return out


# ============================================================================
# Selected-up: sparse_up[b,k] = sum_h x[b,h] * W_up[idx[b,k], h]
# Grid: (B, ceil(K/BLOCK_K)). Inner loop over H. No [B,K,H] materialization.
# Each program processes one batch element and a block of K output positions.
# ============================================================================

@triton.jit
def _selected_up_gather_dot_kernel(
    x_ptr, idx_ptr, w_ptr, out_ptr,
    B, K, H, I,
    s_xn, s_xh,
    s_in, s_ik,
    s_wi, s_wh,
    s_on, s_ok,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    idx_vec = tl.load(idx_ptr + pid_b * s_in + k_offs * s_ik,
                      mask=k_mask, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for h_start in range(0, H, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offs < H

        x_vec = tl.load(x_ptr + pid_b * s_xn + h_offs * s_xh,
                        mask=h_mask, other=0.0)
        w_tile = tl.load(
            w_ptr + idx_vec[:, None] * s_wi + h_offs[None, :] * s_wh,
            mask=k_mask[:, None] & h_mask[None, :], other=0.0,
        )
        acc += tl.sum(x_vec[None, :].to(tl.float32) * w_tile.to(tl.float32), axis=1)

    tl.store(out_ptr + pid_b * s_on + k_offs * s_ok,
             acc.to(tl.bfloat16), mask=k_mask)


def selected_up_gather_dot(x: torch.Tensor, idx: torch.Tensor,
                            w_up: torch.Tensor,
                            BLOCK_K: int = 32, BLOCK_H: int = 64) -> torch.Tensor:
    """Compute sparse_up[b,k] = sum_h x[b,h] * w_up[idx[b,k], h] without materializing [B,K,H].

    Args:
        x: [B, H] bf16
        idx: [B, K] int64
        w_up: [I, H] bf16 contiguous (i.e. up_proj.weight directly)
    Returns:
        out: [B, K] bf16
    """
    assert x.is_cuda and idx.is_cuda and w_up.is_cuda
    assert x.dtype == torch.bfloat16 and w_up.dtype == torch.bfloat16
    assert idx.dtype == torch.int64
    B, H = x.shape
    I, Hw = w_up.shape
    Bi, K = idx.shape
    assert Hw == H and Bi == B
    out = torch.empty(B, K, device=x.device, dtype=torch.bfloat16)

    grid = (B, triton.cdiv(K, BLOCK_K))
    _selected_up_gather_dot_kernel[grid](
        x, idx, w_up, out,
        B, K, H, I,
        x.stride(0), x.stride(1),
        idx.stride(0), idx.stride(1),
        w_up.stride(0), w_up.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
    )
    return out


# ============================================================================
# Fused silu(topk_vals) * sparse_up + selected-down  (v_07 next-best fusion)
# y[b,h] = sum_k silu(topk_vals[b,k]) * sparse_up[b,k] * W_down_t[idx[b,k], h]
# Avoids writing/reading sparse_z in HBM and avoids one extra elementwise launch.
# selected-up still produces sparse_up [B,K] in HBM (kept separate; full fusion
# would require shared-memory or single-program H reduction, infeasible here).
# Grid: (B, ceil(H/BLOCK_H)). Inner loop over K. No [B,K,H] materialization.
# ============================================================================

@triton.jit
def _fused_silu_selected_down_kernel(
    topk_vals_ptr, sparse_up_ptr, idx_ptr, w_ptr, out_ptr,
    B, K, H, I,
    s_tn, s_tk,
    s_un, s_uk,
    s_in, s_ik,
    s_wi, s_wh,
    s_on, s_oh,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load topk_vals[b, k] and sparse_up[b, k]
        tv = tl.load(topk_vals_ptr + pid_b * s_tn + k_offs * s_tk,
                     mask=k_mask, other=0.0).to(tl.float32)
        su = tl.load(sparse_up_ptr + pid_b * s_un + k_offs * s_uk,
                     mask=k_mask, other=0.0).to(tl.float32)

        # Fused silu * sparse_up in registers (no HBM write of sparse_z)
        # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        sig = 1.0 / (1.0 + tl.exp(-tv))
        silu = tv * sig
        z = silu * su  # [BLOCK_K] fp32

        # Load idx and W_down_t tile, accumulate
        idx_vec = tl.load(idx_ptr + pid_b * s_in + k_offs * s_ik,
                          mask=k_mask, other=0).to(tl.int64)
        w_tile = tl.load(
            w_ptr + idx_vec[:, None] * s_wi + h_offs[None, :] * s_wh,
            mask=k_mask[:, None] & h_mask[None, :], other=0.0,
        )
        acc += tl.sum(z[:, None] * w_tile.to(tl.float32), axis=0)

    tl.store(out_ptr + pid_b * s_on + h_offs * s_oh,
             acc.to(tl.bfloat16), mask=h_mask)


def fused_silu_selected_down(topk_vals: torch.Tensor, sparse_up: torch.Tensor,
                              idx: torch.Tensor, w_down_t: torch.Tensor,
                              BLOCK_K: int = 64, BLOCK_H: int = 32) -> torch.Tensor:
    """Compute y[b,h] = sum_k silu(topk_vals[b,k]) * sparse_up[b,k] * w_down_t[idx[b,k], h]
    without materializing [B,K,H] and without writing sparse_z = silu(topk_vals)*sparse_up to HBM.

    Args:
        topk_vals: [B, K] bf16 (raw gate values, pre-SiLU)
        sparse_up: [B, K] bf16 (already gathered from selected_up_gather_dot)
        idx: [B, K] int64
        w_down_t: [I, H] bf16 contiguous (down_proj.weight.t().contiguous())
    Returns:
        out: [B, H] bf16
    """
    assert topk_vals.is_cuda and sparse_up.is_cuda and idx.is_cuda and w_down_t.is_cuda
    assert topk_vals.dtype == torch.bfloat16 and sparse_up.dtype == torch.bfloat16
    assert w_down_t.dtype == torch.bfloat16
    assert idx.dtype == torch.int64
    B, K = topk_vals.shape
    Bu, Ku = sparse_up.shape
    Bi, Ki = idx.shape
    I, H = w_down_t.shape
    assert Bu == B and Ku == K
    assert Bi == B and Ki == K
    out = torch.empty(B, H, device=topk_vals.device, dtype=torch.bfloat16)

    grid = (B, triton.cdiv(H, BLOCK_H))
    _fused_silu_selected_down_kernel[grid](
        topk_vals, sparse_up, idx, w_down_t, out,
        B, K, H, I,
        topk_vals.stride(0), topk_vals.stride(1),
        sparse_up.stride(0), sparse_up.stride(1),
        idx.stride(0), idx.stride(1),
        w_down_t.stride(0), w_down_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
    )
    return out

