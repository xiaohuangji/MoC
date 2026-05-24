"""v_09 fused gate_proj + grouped top-2-of-8 selector (Triton).

Single Triton kernel that:
1. Computes gate scores g[b, i] = sum_h x[b, h] * gate_w[i, h] for the
   channel range owned by this program, tiled over H with fp32 accumulation.
2. For each group of 8 consecutive channels, picks the top-2 scores
   (argmax + second-argmax) directly in registers.
3. Writes ONLY (topk_vals[B, K], topk_idx[B, K]) where K = groups * 2.

The full gate matrix [B, I] is never materialized in HBM. Each program
handles BLOCK_B rows x BLOCK_G groups (i.e. 8 * BLOCK_G channels).

Shape requirements:
- intermediate I must be divisible by 8 (groups = I / 8).
- gate_w must be contiguous [I, H] bf16 (matches nn.Linear weight layout).
- x must be [B, H] bf16.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gate_top2_of_8_kernel(
    x_ptr, w_ptr, v_ptr, i_ptr,
    B, H, groups,
    s_xb, s_xh,
    s_wi, s_wh,
    s_vb, s_vk,
    s_ib, s_ik,
    BLOCK_B: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Each program covers [BLOCK_B rows] x [BLOCK_G groups (= 8*BLOCK_G channels)].

    Inner H loop accumulates the matmul into a [BLOCK_B, BLOCK_G, 8] fp32
    score tile, then top-2-of-8 is computed per (row, group).
    """
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)        # [BLOCK_B]
    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)        # [BLOCK_G]
    eight = tl.arange(0, 8)                                 # [8]
    h_arange = tl.arange(0, BLOCK_H)                        # [BLOCK_H]

    b_mask = b_offs < B
    g_mask = g_offs < groups

    # Channel offsets [BLOCK_G, 8] -> flat [BLOCK_G * 8]
    chan = g_offs[:, None] * 8 + eight[None, :]             # [BLOCK_G, 8]
    chan_flat = tl.reshape(chan, (BLOCK_G * 8,))            # [BLOCK_G*8]
    chan_mask_flat = tl.reshape(
        g_mask[:, None] & tl.full((1, 8), True, tl.int1),
        (BLOCK_G * 8,),
    )

    # Accumulator [BLOCK_B, BLOCK_G * 8] in fp32
    acc = tl.zeros((BLOCK_B, BLOCK_G * 8), dtype=tl.float32)

    # H tiling
    for h_start in range(0, H, BLOCK_H):
        h_offs = h_start + h_arange
        h_mask = h_offs < H

        # x tile [BLOCK_B, BLOCK_H] bf16
        x_ptrs = x_ptr + b_offs[:, None] * s_xb + h_offs[None, :] * s_xh
        x_tile = tl.load(
            x_ptrs,
            mask=b_mask[:, None] & h_mask[None, :],
            other=0.0,
        )

        # w tile [BLOCK_G*8, BLOCK_H] bf16; rows are 8*BLOCK_G channels
        w_ptrs = w_ptr + chan_flat[:, None] * s_wi + h_offs[None, :] * s_wh
        w_tile = tl.load(
            w_ptrs,
            mask=chan_mask_flat[:, None] & h_mask[None, :],
            other=0.0,
        )

        # Accumulate: x [B, H] @ w.T [H, C] = [B, C]
        # tl.dot signature: dot(a, b) does a @ b. We have a=[B,H] and want
        # result [B, C], so b should be [H, C]. We have w_tile=[C, H], so use
        # transpose.
        acc += tl.dot(x_tile, tl.trans(w_tile), out_dtype=tl.float32)

    # Reshape to [BLOCK_B, BLOCK_G, 8] for per-group top-2.
    # Quantize fp32 acc to bf16 BEFORE the top-2 reduce so values match
    # torch's `(x @ gate_w.t()).view(...).topk(2)` exactly (where the
    # bf16 GEMM rounds the fp32 tensor-core output to bf16 before topk
    # sees it). Without this, our fp32-precision tie-breaking differs from
    # torch.topk on ulp-close pairs.
    out_dtype = v_ptr.dtype.element_ty
    g_quant = tl.reshape(acc.to(out_dtype), (BLOCK_B, BLOCK_G, 8))
    g_f = g_quant.to(tl.float32)

    # Mask out-of-bounds groups so they cannot win argmax
    NEG_INF = -float('inf')
    g_mask_3d = b_mask[:, None, None] & g_mask[None, :, None]
    g_f = tl.where(g_mask_3d, g_f, tl.full(g_f.shape, NEG_INF, tl.float32))

    # top-1
    max1_val = tl.max(g_f, axis=-1)                         # [BLOCK_B, BLOCK_G]
    max1_mask = (g_f == max1_val[:, :, None])
    BIG = tl.full((BLOCK_B, BLOCK_G, 8), 8, tl.int32)
    b_candidate1 = tl.where(max1_mask, eight[None, None, :].to(tl.int32), BIG)
    max1_b = tl.min(b_candidate1, axis=-1)                  # [BLOCK_B, BLOCK_G] int32

    # top-2 (mask out top-1 slot, then max again)
    not_top1 = (eight[None, None, :] != max1_b[:, :, None])
    g_f_masked = tl.where(not_top1, g_f, tl.full(g_f.shape, NEG_INF, tl.float32))
    max2_val = tl.max(g_f_masked, axis=-1)
    max2_mask = (g_f_masked == max2_val[:, :, None])
    b_candidate2 = tl.where(max2_mask, eight[None, None, :].to(tl.int32), BIG)
    max2_b = tl.min(b_candidate2, axis=-1)

    # Global channel idx
    global1 = (g_offs[None, :] * 8 + max1_b).to(tl.int64)   # [BLOCK_B, BLOCK_G]
    global2 = (g_offs[None, :] * 8 + max2_b).to(tl.int64)

    # Output K-axis layout: [..group0_top1, group0_top2, group1_top1, ...]
    out_k_a = (g_offs[None, :] * 2 + 0)                     # [1, BLOCK_G]
    out_k_b = (g_offs[None, :] * 2 + 1)

    # Broadcast b axis
    b_idx = b_offs[:, None]                                 # [BLOCK_B, 1]

    write_mask = b_mask[:, None] & g_mask[None, :]

    v_dtype = v_ptr.dtype.element_ty
    tl.store(
        v_ptr + b_idx * s_vb + out_k_a * s_vk,
        max1_val.to(v_dtype), mask=write_mask,
    )
    tl.store(
        v_ptr + b_idx * s_vb + out_k_b * s_vk,
        max2_val.to(v_dtype), mask=write_mask,
    )
    tl.store(
        i_ptr + b_idx * s_ib + out_k_a * s_ik,
        global1, mask=write_mask,
    )
    tl.store(
        i_ptr + b_idx * s_ib + out_k_b * s_ik,
        global2, mask=write_mask,
    )


def fused_gate_top2of8(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    grouped_b: int = 8,
    BLOCK_B: int = 16,
    BLOCK_G: int = 16,
    BLOCK_H: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused gate_proj + grouped top-2-of-8 selector.

    Args:
        x: [B, H] bf16/fp16, decode input.
        gate_weight: [I, H] bf16/fp16, contiguous (matches nn.Linear weight).
        grouped_b: must be 8 (top-2-of-8 specialization).
    Returns:
        topk_vals: [B, K] same dtype as x, K = (I / 8) * 2.
        topk_idx:  [B, K] int64, global channel ids in [0, I).
    """
    if grouped_b != 8:
        raise ValueError(f"only grouped_b=8 is supported, got {grouped_b}")
    if x.dim() != 2 or gate_weight.dim() != 2:
        raise ValueError("x and gate_weight must be 2D")
    if x.shape[1] != gate_weight.shape[1]:
        raise ValueError(f"H mismatch: x={x.shape}, gate_w={gate_weight.shape}")
    if not (x.is_cuda and gate_weight.is_cuda):
        raise ValueError("x and gate_weight must be CUDA tensors")
    if gate_weight.dtype != x.dtype:
        raise ValueError(f"dtype mismatch: x={x.dtype}, gate_w={gate_weight.dtype}")
    B, H = x.shape
    I = gate_weight.shape[0]
    if I % 8 != 0:
        raise ValueError(f"intermediate I={I} must be divisible by 8")
    if not gate_weight.is_contiguous():
        gate_weight = gate_weight.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()

    groups = I // 8
    K = groups * 2

    topk_vals = torch.empty(B, K, device=x.device, dtype=x.dtype)
    topk_idx = torch.empty(B, K, device=x.device, dtype=torch.int64)

    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(groups, BLOCK_G))
    _fused_gate_top2_of_8_kernel[grid](
        x, gate_weight, topk_vals, topk_idx,
        B, H, groups,
        x.stride(0), x.stride(1),
        gate_weight.stride(0), gate_weight.stride(1),
        topk_vals.stride(0), topk_vals.stride(1),
        topk_idx.stride(0), topk_idx.stride(1),
        BLOCK_B=BLOCK_B, BLOCK_G=BLOCK_G, BLOCK_H=BLOCK_H,
    )
    return topk_vals, topk_idx

