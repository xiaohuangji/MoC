"""Grouped MoC fused gate/up and selected-down Triton kernels.

The grouped top-2-of-8 path uses two kernels:

    [Kernel A] fused_gate_top2of8_selected_up_silu
        -> computes gate and up projections for each group, selects top-2
           gate channels, applies SiLU on the selected gate values, and writes
           only topk_idx[B, K] plus sparse_z[B, K].
    [Kernel B] selected_down_from_sparse_z
        -> reads sparse_z[B, K] and topk_idx[B, K], then reduces against the
           transposed down-projection weight.

The MoC 2:8 math is preserved: selection is performed on raw gate values
before SiLU, then SiLU is applied only to the selected gate values.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ============================================================================
# Kernel A: fused gate + top2-of-8 + selected_up + silu.
# The post-selection step folds silu(topk_val) * up_val into one
# in-register multiplication and writes only sparse_z (and idx).
# ============================================================================

@triton.jit
def _fused_gate_top2_of_8_selected_up_silu_kernel(
    x_ptr, gw_ptr, uw_ptr, i_ptr, sz_ptr,
    B, H, groups,
    s_xb, s_xh,
    s_gwi, s_gwh,
    s_uwi, s_uwh,
    s_ib, s_ik,
    s_szb, s_szk,
    BLOCK_B: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    eight = tl.arange(0, 8)
    h_arange = tl.arange(0, BLOCK_H)

    b_mask = b_offs < B
    g_mask = g_offs < groups

    chan = g_offs[:, None] * 8 + eight[None, :]
    chan_flat = tl.reshape(chan, (BLOCK_G * 8,))
    chan_mask_flat = tl.reshape(
        g_mask[:, None] & tl.full((1, 8), True, tl.int1),
        (BLOCK_G * 8,),
    )

    acc_gate = tl.zeros((BLOCK_B, BLOCK_G * 8), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_B, BLOCK_G * 8), dtype=tl.float32)

    for h_start in range(0, H, BLOCK_H):
        h_offs = h_start + h_arange
        h_mask = h_offs < H

        x_ptrs = x_ptr + b_offs[:, None] * s_xb + h_offs[None, :] * s_xh
        x_tile = tl.load(
            x_ptrs,
            mask=b_mask[:, None] & h_mask[None, :],
            other=0.0,
        )

        gw_ptrs = gw_ptr + chan_flat[:, None] * s_gwi + h_offs[None, :] * s_gwh
        gw_tile = tl.load(
            gw_ptrs,
            mask=chan_mask_flat[:, None] & h_mask[None, :],
            other=0.0,
        )

        uw_ptrs = uw_ptr + chan_flat[:, None] * s_uwi + h_offs[None, :] * s_uwh
        uw_tile = tl.load(
            uw_ptrs,
            mask=chan_mask_flat[:, None] & h_mask[None, :],
            other=0.0,
        )

        gw_t = tl.trans(gw_tile)
        uw_t = tl.trans(uw_tile)
        acc_gate += tl.dot(x_tile, gw_t, out_dtype=tl.float32)
        acc_up += tl.dot(x_tile, uw_t, out_dtype=tl.float32)

    out_dtype = sz_ptr.dtype.element_ty

    # Quantize gate accumulator BEFORE top-2 reduce, matching the benchmark path
    g_quant = tl.reshape(acc_gate.to(out_dtype), (BLOCK_B, BLOCK_G, 8))
    g_f = g_quant.to(tl.float32)

    NEG_INF = -float('inf')
    g_mask_3d = b_mask[:, None, None] & g_mask[None, :, None]
    g_f = tl.where(g_mask_3d, g_f, tl.full(g_f.shape, NEG_INF, tl.float32))

    max1_val = tl.max(g_f, axis=-1)
    max1_mask = (g_f == max1_val[:, :, None])
    BIG = tl.full((BLOCK_B, BLOCK_G, 8), 8, tl.int32)
    b_candidate1 = tl.where(max1_mask, eight[None, None, :].to(tl.int32), BIG)
    max1_b = tl.min(b_candidate1, axis=-1)

    not_top1 = (eight[None, None, :] != max1_b[:, :, None])
    g_f_masked = tl.where(not_top1, g_f, tl.full(g_f.shape, NEG_INF, tl.float32))
    max2_val = tl.max(g_f_masked, axis=-1)
    max2_mask = (g_f_masked == max2_val[:, :, None])
    b_candidate2 = tl.where(max2_mask, eight[None, None, :].to(tl.int32), BIG)
    max2_b = tl.min(b_candidate2, axis=-1)

    # Pull selected up values from acc_up via one-hot mask reduce
    u_quant = tl.reshape(acc_up.to(out_dtype), (BLOCK_B, BLOCK_G, 8))
    u_f = u_quant.to(tl.float32)
    sel1 = (eight[None, None, :] == max1_b[:, :, None]).to(tl.float32)
    sel2 = (eight[None, None, :] == max2_b[:, :, None]).to(tl.float32)
    up1_val = tl.sum(u_f * sel1, axis=-1)
    up2_val = tl.sum(u_f * sel2, axis=-1)

    # Apply SiLU to the selected top-K gate values, fuse with up:
    # sparse_z = silu(top_val) * up_val  in fp32 registers
    sig1 = 1.0 / (1.0 + tl.exp(-max1_val))
    sig2 = 1.0 / (1.0 + tl.exp(-max2_val))
    sz1 = (max1_val * sig1) * up1_val
    sz2 = (max2_val * sig2) * up2_val

    global1 = (g_offs[None, :] * 8 + max1_b).to(tl.int64)
    global2 = (g_offs[None, :] * 8 + max2_b).to(tl.int64)

    out_k_a = g_offs[None, :] * 2 + 0
    out_k_b = g_offs[None, :] * 2 + 1
    b_idx = b_offs[:, None]
    write_mask = b_mask[:, None] & g_mask[None, :]

    sz_dtype = sz_ptr.dtype.element_ty

    tl.store(
        i_ptr + b_idx * s_ib + out_k_a * s_ik,
        global1, mask=write_mask,
    )
    tl.store(
        i_ptr + b_idx * s_ib + out_k_b * s_ik,
        global2, mask=write_mask,
    )
    tl.store(
        sz_ptr + b_idx * s_szb + out_k_a * s_szk,
        sz1.to(sz_dtype), mask=write_mask,
    )
    tl.store(
        sz_ptr + b_idx * s_szb + out_k_b * s_szk,
        sz2.to(sz_dtype), mask=write_mask,
    )


def fused_gate_top2of8_selected_up_silu(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    grouped_b: int = 8,
    BLOCK_B: int = 16,
    BLOCK_G: int = 16,
    BLOCK_H: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single Triton kernel: gate_proj + top-2-of-8 + selected_up + SiLU.

    Returns:
        topk_idx:  [B, K] int64 selected channel indices.
        sparse_z:  [B, K] same dtype as x; sparse_z[b,k] = silu(gate[idx])*up[idx].

    It writes sparse_z[B, K] plus topk_idx[B, K] and avoids materializing full gate/up intermediates.
    """
    if grouped_b != 8:
        raise ValueError(f"only grouped_b=8 is supported, got {grouped_b}")
    if x.dim() != 2 or gate_weight.dim() != 2 or up_weight.dim() != 2:
        raise ValueError("x, gate_weight, up_weight must all be 2D")
    if not (x.is_cuda and gate_weight.is_cuda and up_weight.is_cuda):
        raise ValueError("inputs must be CUDA")
    if not (x.dtype == gate_weight.dtype == up_weight.dtype):
        raise ValueError(
            f"dtype mismatch: x={x.dtype}, gate={gate_weight.dtype}, up={up_weight.dtype}"
        )
    B, H = x.shape
    I_g, Hg = gate_weight.shape
    I_u, Hu = up_weight.shape
    if not (Hg == H and Hu == H):
        raise ValueError(
            f"H mismatch: x={x.shape}, gate={gate_weight.shape}, up={up_weight.shape}"
        )
    if I_g != I_u:
        raise ValueError(f"gate I={I_g} != up I={I_u}")
    if I_g % 8 != 0:
        raise ValueError(f"intermediate I={I_g} must be divisible by 8")
    if not gate_weight.is_contiguous():
        gate_weight = gate_weight.contiguous()
    if not up_weight.is_contiguous():
        up_weight = up_weight.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()

    groups = I_g // 8
    K = groups * 2

    topk_idx = torch.empty(B, K, device=x.device, dtype=torch.int64)
    sparse_z = torch.empty(B, K, device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(groups, BLOCK_G))
    _fused_gate_top2_of_8_selected_up_silu_kernel[grid](
        x, gate_weight, up_weight, topk_idx, sparse_z,
        B, H, groups,
        x.stride(0), x.stride(1),
        gate_weight.stride(0), gate_weight.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        topk_idx.stride(0), topk_idx.stride(1),
        sparse_z.stride(0), sparse_z.stride(1),
        BLOCK_B=BLOCK_B, BLOCK_G=BLOCK_G, BLOCK_H=BLOCK_H,
    )
    return topk_idx, sparse_z


# ============================================================================
# Kernel B: selected_down_from_sparse_z. Strictly simpler than the earlier fused-silu down design: no topk_vals input and no inline SiLU in the down kernel.
# ============================================================================

@triton.jit
def _selected_down_from_sparse_z_kernel(
    sz_ptr, idx_ptr, w_ptr, out_ptr,
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

        z_vec = tl.load(
            sz_ptr + pid_b * s_zn + k_offs * s_zk,
            mask=k_mask, other=0.0,
        ).to(tl.float32)                                     # [BLOCK_K]
        idx_vec = tl.load(
            idx_ptr + pid_b * s_in + k_offs * s_ik,
            mask=k_mask, other=0,
        ).to(tl.int64)                                       # [BLOCK_K]
        w_tile = tl.load(
            w_ptr + idx_vec[:, None] * s_wi + h_offs[None, :] * s_wh,
            mask=k_mask[:, None] & h_mask[None, :], other=0.0,
        )                                                    # [BLOCK_K, BLOCK_H]
        acc += tl.sum(z_vec[:, None] * w_tile.to(tl.float32), axis=0)

    tl.store(
        out_ptr + pid_b * s_on + h_offs * s_oh,
        acc.to(tl.bfloat16), mask=h_mask,
    )


def selected_down_from_sparse_z(
    sparse_z: torch.Tensor,
    idx: torch.Tensor,
    w_down_t: torch.Tensor,
    BLOCK_K: int = 64,
    BLOCK_H: int = 32,
) -> torch.Tensor:
    """y[b,h] = sum_k sparse_z[b,k] * w_down_t[idx[b,k], h]

    Args:
        sparse_z:  [B, K] bf16 (already silu-applied in upstream Kernel A).
        idx:       [B, K] int64 selected channel ids.
        w_down_t:  [I, H] bf16 contiguous (down_proj.weight.t().contiguous()).
    """
    assert sparse_z.is_cuda and idx.is_cuda and w_down_t.is_cuda
    assert sparse_z.dtype == torch.bfloat16 and w_down_t.dtype == torch.bfloat16
    assert idx.dtype == torch.int64
    B, K = sparse_z.shape
    I, H = w_down_t.shape
    out = torch.empty(B, H, device=sparse_z.device, dtype=torch.bfloat16)

    grid = (B, triton.cdiv(H, BLOCK_H))
    _selected_down_from_sparse_z_kernel[grid](
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
# Kernel B variant 2: split-K + atomic accumulate.
# At bs=1 the down kernel launches B * ceil(H/BLOCK_H) programs. Splitting K
# into partial chunks multiplies parallelism and reduces per-program serial work.
#
# Each program now handles 1 batch row x BLOCK_H output cols x 1 K-shard.
# Partial sums are atomically added into a shared fp32 output buffer; a tiny
# downstream cast kernel converts fp32 -> bf16 in the user's output tensor.
# ============================================================================

@triton.jit
def _selected_down_splitk_partial_kernel(
    sz_ptr, idx_ptr, w_ptr, out_fp32_ptr,
    B, K, H, I, K_SHARD,
    s_zn, s_zk,
    s_in, s_ik,
    s_wi, s_wh,
    s_on, s_oh,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    k_start_global = pid_s * K_SHARD
    k_end_global = tl.minimum(k_start_global + K_SHARD, K)

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k_start in range(k_start_global, k_end_global, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < k_end_global

        z_vec = tl.load(
            sz_ptr + pid_b * s_zn + k_offs * s_zk,
            mask=k_mask, other=0.0,
        ).to(tl.float32)
        idx_vec = tl.load(
            idx_ptr + pid_b * s_in + k_offs * s_ik,
            mask=k_mask, other=0,
        ).to(tl.int64)
        w_tile = tl.load(
            w_ptr + idx_vec[:, None] * s_wi + h_offs[None, :] * s_wh,
            mask=k_mask[:, None] & h_mask[None, :], other=0.0,
        )
        acc += tl.sum(z_vec[:, None] * w_tile.to(tl.float32), axis=0)

    # Atomic add into the shared fp32 output for this (b, h_offs) tile
    out_ptrs = out_fp32_ptr + pid_b * s_on + h_offs * s_oh
    tl.atomic_add(out_ptrs, acc, mask=h_mask)


@triton.jit
def _cast_fp32_to_bf16_kernel(
    in_ptr, out_ptr, B, H,
    s_in, s_ih, s_on, s_oh,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    v = tl.load(in_ptr + pid_b * s_in + h_offs * s_ih, mask=h_mask, other=0.0)
    tl.store(out_ptr + pid_b * s_on + h_offs * s_oh, v.to(tl.bfloat16), mask=h_mask)


def selected_down_from_sparse_z_splitk(
    sparse_z: torch.Tensor,
    idx: torch.Tensor,
    w_down_t: torch.Tensor,
    BLOCK_K: int = 64,
    BLOCK_H: int = 32,
    SPLITS: int = 4,
) -> torch.Tensor:
    """Split-K variant of selected_down_from_sparse_z.

    For bs=1 decode the standard kernel only launches B * ceil(H/BLOCK_H)
    programs which underutilizes RTX 5090's ~170 SMs. SPLITS-way K split
    multiplies parallelism by SPLITS at the cost of one atomic_add per
    partial and one tiny fp32->bf16 cast pass.
    """
    assert sparse_z.is_cuda and idx.is_cuda and w_down_t.is_cuda
    assert sparse_z.dtype == torch.bfloat16 and w_down_t.dtype == torch.bfloat16
    assert idx.dtype == torch.int64
    B, K = sparse_z.shape
    I, H = w_down_t.shape

    K_SHARD = (K + SPLITS - 1) // SPLITS
    actual_splits = (K + K_SHARD - 1) // K_SHARD

    out_fp32 = torch.zeros(B, H, device=sparse_z.device, dtype=torch.float32)
    out = torch.empty(B, H, device=sparse_z.device, dtype=torch.bfloat16)

    grid_partial = (B, triton.cdiv(H, BLOCK_H), actual_splits)
    _selected_down_splitk_partial_kernel[grid_partial](
        sparse_z, idx, w_down_t, out_fp32,
        B, K, H, I, K_SHARD,
        sparse_z.stride(0), sparse_z.stride(1),
        idx.stride(0), idx.stride(1),
        w_down_t.stride(0), w_down_t.stride(1),
        out_fp32.stride(0), out_fp32.stride(1),
        BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
    )

    grid_cast = (B, triton.cdiv(H, BLOCK_H))
    _cast_fp32_to_bf16_kernel[grid_cast](
        out_fp32, out, B, H,
        out_fp32.stride(0), out_fp32.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_H=BLOCK_H,
    )
    return out


# =====================================================================
# Direct-output wrappers for the grouped MoC kernels. They take preallocated
# output buffers from the caller so CUDA Graph capture contains only kernel
# launches.
# =====================================================================


def fused_gate_top2of8_selected_up_silu_into(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    topk_idx_out: torch.Tensor,
    sparse_z_out: torch.Tensor,
    grouped_b: int = 8,
    BLOCK_B: int = 16,
    BLOCK_G: int = 16,
    BLOCK_H: int = 64,
) -> None:
    """Direct-output variant of fused_gate_top2of8_selected_up_silu.
    Writes into caller-provided topk_idx_out [B, K] int64 and
    sparse_z_out [B, K] same dtype as x."""
    if grouped_b != 8:
        raise ValueError(f"only grouped_b=8 is supported, got {grouped_b}")
    B, H = x.shape
    I = gate_weight.shape[0]
    if I % 8 != 0:
        raise ValueError(f"intermediate I={I} must be divisible by 8")
    groups = I // 8
    K = groups * 2
    if topk_idx_out.shape != (B, K) or sparse_z_out.shape != (B, K):
        raise ValueError(
            f"out shapes must be ({B},{K}); got "
            f"idx={tuple(topk_idx_out.shape)}, sz={tuple(sparse_z_out.shape)}"
        )
    if not gate_weight.is_contiguous():
        gate_weight = gate_weight.contiguous()
    if not up_weight.is_contiguous():
        up_weight = up_weight.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()

    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(groups, BLOCK_G))
    _fused_gate_top2_of_8_selected_up_silu_kernel[grid](
        x, gate_weight, up_weight, topk_idx_out, sparse_z_out,
        B, H, groups,
        x.stride(0), x.stride(1),
        gate_weight.stride(0), gate_weight.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        topk_idx_out.stride(0), topk_idx_out.stride(1),
        sparse_z_out.stride(0), sparse_z_out.stride(1),
        BLOCK_B=BLOCK_B, BLOCK_G=BLOCK_G, BLOCK_H=BLOCK_H,
    )


def selected_down_from_sparse_z_into(
    sparse_z: torch.Tensor,
    idx: torch.Tensor,
    w_down_t: torch.Tensor,
    out: torch.Tensor,
    BLOCK_K: int = 128,
    BLOCK_H: int = 16,
) -> None:
    """Direct-output variant of selected_down_from_sparse_z.
    Writes into caller-provided out [B, H] bf16."""
    B, K = sparse_z.shape
    I, H = w_down_t.shape
    if out.shape != (B, H):
        raise ValueError(f"out shape must be ({B},{H}); got {tuple(out.shape)}")

    grid = (B, triton.cdiv(H, BLOCK_H))
    _selected_down_from_sparse_z_kernel[grid](
        sparse_z, idx, w_down_t, out,
        B, K, H, I,
        sparse_z.stride(0), sparse_z.stride(1),
        idx.stride(0), idx.stride(1),
        w_down_t.stride(0), w_down_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
    )

