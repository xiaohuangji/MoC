"""v_11 group-wise MoC_{2:8} Up&Down Triton kernels.

Idea:
    For MoC_{2:8} the intermediate dimension is partitioned into groups of 8
    contiguous channels. Within each group exactly 2 channels are selected by
    Top-K. Instead of doing dynamic gather on individual selected rows of
    W_up / W_down, we load each group's 8 contiguous rows in one coalesced
    burst and apply a per-group lane mask in registers. That keeps the
    load-side memory access aligned to 8-channel boundaries while still only
    producing the 2 selected lanes' values.

We provide two kernels:

1. `selected_up_group8`:
       sparse_up[b, 2g + j] = sum_h x[b, h] * W_up[g*8 + lane_j[b, g], h]
   The kernel computes the 8 dot products per group (BLOCK_G groups per
   program), then masks by `lane_j` to write 2 selected scores. This trades
   4x compute for contiguous 8-row weight loads.

2. `fused_silu_selected_down_group8`:
       y[b, h] = sum_g sum_{j in {0,1}} silu(tv[b, 2g+j]) * up[b, 2g+j]
                                       * W_down_t[g*8 + lane_j[b, g], h]
   For each group we load 8 contiguous W_down_t rows. We construct an
   in-register length-8 vector `w_per_lane` = (lane_j == k) ? sz_j : 0,
   then reduce. NO z_full[B, I] tensor is materialized.

Neither kernel materializes [B, K, H] or [B, I]. The selector plus the two
group8 kernels make up a sparse Up&Down path that only touches active channels
while preserving 8-lane coalesced loads.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ============================================================================
# Kernel A: selected_up_group8
# Each program: 1 batch row x BLOCK_G groups (= 8 * BLOCK_G channels).
# Loads W_up[g*8 .. g*8+7, h_tile] contiguously per group, computes 8 dot
# products, then masks to 2 selected lanes per group via topk_idx.
# Grid: (B, ceil(groups / BLOCK_G)).
# ============================================================================

@triton.jit
def _selected_up_group8_kernel(
    x_ptr, idx_ptr, w_ptr, out_ptr,
    B, H, groups,
    s_xn, s_xh,
    s_in, s_ik,
    s_wi, s_wh,
    s_on, s_ok,
    BLOCK_G: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)         # [BLOCK_G]
    g_mask = g_offs < groups
    eight = tl.arange(0, 8)                                  # [8]

    # Channel offsets [BLOCK_G, 8] -> [BLOCK_G * 8]
    chan = g_offs[:, None] * 8 + eight[None, :]              # [BLOCK_G, 8]
    chan_flat = tl.reshape(chan, (BLOCK_G * 8,))
    chan_mask_flat = tl.reshape(
        g_mask[:, None] & tl.full((1, 8), True, tl.int1),
        (BLOCK_G * 8,),
    )

    acc = tl.zeros((BLOCK_G * 8,), dtype=tl.float32)

    # H tiling
    for h_start in range(0, H, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offs < H

        x_vec = tl.load(
            x_ptr + pid_b * s_xn + h_offs * s_xh,
            mask=h_mask, other=0.0,
        )                                                    # [BLOCK_H]

        w_tile = tl.load(
            w_ptr + chan_flat[:, None] * s_wi + h_offs[None, :] * s_wh,
            mask=chan_mask_flat[:, None] & h_mask[None, :],
            other=0.0,
        )                                                    # [BLOCK_G*8, BLOCK_H]

        # acc[c] += sum_h x[h] * w[c, h]
        acc += tl.sum(x_vec[None, :].to(tl.float32) * w_tile.to(tl.float32), axis=1)

    # Reshape and select
    acc_grp = tl.reshape(acc, (BLOCK_G, 8))                  # [BLOCK_G, 8]

    # Load topk_idx for each group's 2 selected positions
    k0 = g_offs * 2 + 0
    k1 = g_offs * 2 + 1
    idx0 = tl.load(idx_ptr + pid_b * s_in + k0 * s_ik,
                   mask=g_mask, other=0).to(tl.int64)        # [BLOCK_G]
    idx1 = tl.load(idx_ptr + pid_b * s_in + k1 * s_ik,
                   mask=g_mask, other=0).to(tl.int64)

    lane0 = (idx0 - g_offs.to(tl.int64) * 8).to(tl.int32)    # [BLOCK_G]
    lane1 = (idx1 - g_offs.to(tl.int64) * 8).to(tl.int32)

    m0 = (eight[None, :] == lane0[:, None])                  # [BLOCK_G, 8]
    m1 = (eight[None, :] == lane1[:, None])

    score0 = tl.sum(tl.where(m0, acc_grp, tl.zeros_like(acc_grp)), axis=-1)
    score1 = tl.sum(tl.where(m1, acc_grp, tl.zeros_like(acc_grp)), axis=-1)

    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + pid_b * s_on + k0 * s_ok,
             score0.to(out_dtype), mask=g_mask)
    tl.store(out_ptr + pid_b * s_on + k1 * s_ok,
             score1.to(out_dtype), mask=g_mask)


def selected_up_group8(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    w_up: torch.Tensor,
    BLOCK_G: int = 16,
    BLOCK_H: int = 64,
) -> torch.Tensor:
    """Compute sparse_up[b, 2g+j] = dot(x[b], W_up[g*8+lane_j[b,g], :])
    where lane_j is derived from topk_idx mod 8. Loads 8 contiguous W_up rows
    per group for coalesced access, masks to 2 selected lanes via topk_idx.

    Args:
        x: [B, H] bf16
        topk_idx: [B, K] int64, K = 2 * groups
        w_up: [I, H] bf16, I = 8 * groups, contiguous (nn.Linear weight)
    Returns:
        out: [B, K] bf16
    """
    assert x.is_cuda and topk_idx.is_cuda and w_up.is_cuda
    assert x.dtype == torch.bfloat16 and w_up.dtype == torch.bfloat16
    assert topk_idx.dtype == torch.int64
    B, H = x.shape
    I, Hw = w_up.shape
    assert Hw == H
    assert I % 8 == 0
    groups = I // 8
    Bi, K = topk_idx.shape
    assert Bi == B and K == 2 * groups

    out = torch.empty(B, K, device=x.device, dtype=torch.bfloat16)

    grid = (B, triton.cdiv(groups, BLOCK_G))
    _selected_up_group8_kernel[grid](
        x, topk_idx, w_up, out,
        B, H, groups,
        x.stride(0), x.stride(1),
        topk_idx.stride(0), topk_idx.stride(1),
        w_up.stride(0), w_up.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_G=BLOCK_G, BLOCK_H=BLOCK_H,
    )
    return out


# ============================================================================
# Kernel B: fused_silu_selected_down_group8
# Avoids z_full[B, I]. For each group g, loads W_down_t[g*8..g*8+7, h_tile]
# contiguously, builds a length-8 in-register weight vector that is non-zero
# at exactly the 2 selected lanes, then reduces.
# Grid: (B, ceil(H / BLOCK_H)).
# ============================================================================

@triton.jit
def _fused_silu_selected_down_group8_kernel(
    topk_vals_ptr, sparse_up_ptr, idx_ptr, w_ptr, out_ptr,
    B, H, groups,
    s_tn, s_tk,
    s_un, s_uk,
    s_in, s_ik,
    s_wi, s_wh,
    s_on, s_oh,
    BLOCK_G: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    eight = tl.arange(0, 8)

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for g_start in range(0, groups, BLOCK_G):
        g_offs = g_start + tl.arange(0, BLOCK_G)             # [BLOCK_G]
        g_mask = g_offs < groups

        k0 = g_offs * 2 + 0
        k1 = g_offs * 2 + 1

        tv0 = tl.load(topk_vals_ptr + pid_b * s_tn + k0 * s_tk,
                      mask=g_mask, other=0.0).to(tl.float32) # [BLOCK_G]
        tv1 = tl.load(topk_vals_ptr + pid_b * s_tn + k1 * s_tk,
                      mask=g_mask, other=0.0).to(tl.float32)
        su0 = tl.load(sparse_up_ptr + pid_b * s_un + k0 * s_uk,
                      mask=g_mask, other=0.0).to(tl.float32)
        su1 = tl.load(sparse_up_ptr + pid_b * s_un + k1 * s_uk,
                      mask=g_mask, other=0.0).to(tl.float32)
        idx0 = tl.load(idx_ptr + pid_b * s_in + k0 * s_ik,
                       mask=g_mask, other=0).to(tl.int64)
        idx1 = tl.load(idx_ptr + pid_b * s_in + k1 * s_ik,
                       mask=g_mask, other=0).to(tl.int64)

        # SiLU * sparse_up = z, in fp32 register
        z0 = (tv0 / (1.0 + tl.exp(-tv0))) * su0              # [BLOCK_G]
        z1 = (tv1 / (1.0 + tl.exp(-tv1))) * su1

        # Per-group local lane in 0..7
        lane0 = (idx0 - g_offs.to(tl.int64) * 8).to(tl.int32)
        lane1 = (idx1 - g_offs.to(tl.int64) * 8).to(tl.int32)

        m0 = (eight[None, :] == lane0[:, None]).to(tl.float32)  # [BLOCK_G, 8]
        m1 = (eight[None, :] == lane1[:, None]).to(tl.float32)
        gm = g_mask[:, None].to(tl.float32)
        # w_per_lane[g, j] = (j==lane0)*z0 + (j==lane1)*z1, masked invalid groups to 0
        w_per_lane = (m0 * z0[:, None] + m1 * z1[:, None]) * gm  # [BLOCK_G, 8]
        w_flat = tl.reshape(w_per_lane, (BLOCK_G * 8,))

        # Channel offsets [BLOCK_G * 8]
        chan = g_offs[:, None] * 8 + eight[None, :]          # [BLOCK_G, 8]
        chan_flat = tl.reshape(chan, (BLOCK_G * 8,))
        chan_mask_flat = tl.reshape(
            g_mask[:, None] & tl.full((1, 8), True, tl.int1),
            (BLOCK_G * 8,),
        )

        # Load contiguous W_down_t rows for these groups
        w_tile = tl.load(
            w_ptr + chan_flat[:, None] * s_wi + h_offs[None, :] * s_wh,
            mask=chan_mask_flat[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)                                      # [BLOCK_G*8, BLOCK_H]

        acc += tl.sum(w_flat[:, None] * w_tile, axis=0)

    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + pid_b * s_on + h_offs * s_oh,
             acc.to(out_dtype), mask=h_mask)


def fused_silu_selected_down_group8(
    topk_vals: torch.Tensor,
    sparse_up: torch.Tensor,
    topk_idx: torch.Tensor,
    w_down_t: torch.Tensor,
    BLOCK_G: int = 16,
    BLOCK_H: int = 32,
) -> torch.Tensor:
    """Compute y[b, h] = sum_k silu(topk_vals[b,k]) * sparse_up[b,k] * w_down_t[idx[b,k], h]
    using group-wise contiguous 8-row W_down_t loads, without z_full[B, I].

    Args:
        topk_vals: [B, K] bf16, K = 2*groups
        sparse_up: [B, K] bf16
        topk_idx:  [B, K] int64, channel ids in [0, I=8*groups)
        w_down_t:  [I, H] bf16 contiguous (down_proj.weight.t().contiguous())
    Returns:
        out: [B, H] bf16
    """
    assert topk_vals.is_cuda and sparse_up.is_cuda and topk_idx.is_cuda and w_down_t.is_cuda
    assert topk_vals.dtype == torch.bfloat16
    assert sparse_up.dtype == torch.bfloat16
    assert w_down_t.dtype == torch.bfloat16
    assert topk_idx.dtype == torch.int64
    B, K = topk_vals.shape
    I, H = w_down_t.shape
    assert I % 8 == 0
    groups = I // 8
    assert K == 2 * groups
    assert sparse_up.shape == (B, K) and topk_idx.shape == (B, K)

    out = torch.empty(B, H, device=topk_vals.device, dtype=torch.bfloat16)

    grid = (B, triton.cdiv(H, BLOCK_H))
    _fused_silu_selected_down_group8_kernel[grid](
        topk_vals, sparse_up, topk_idx, w_down_t, out,
        B, H, groups,
        topk_vals.stride(0), topk_vals.stride(1),
        sparse_up.stride(0), sparse_up.stride(1),
        topk_idx.stride(0), topk_idx.stride(1),
        w_down_t.stride(0), w_down_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_G=BLOCK_G, BLOCK_H=BLOCK_H,
    )
    return out

