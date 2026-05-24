"""v_13 fused gate_proj + grouped top-2-of-8 + selected_up Triton kernel.

Single Triton kernel for the MoC 2:8 decode path:

1. Loads `x` once per H tile and reuses it for two GEMM accumulators.
2. Loads BOTH `gate_weight[g*8:(g+1)*8, h_tile]` and
   `up_weight[g*8:(g+1)*8, h_tile]` for the same 8 contiguous channels per
   group, so each H-tile pays one x-load instead of two (one for the v_09
   gate selector kernel, one for the v_07 selected_up_gather_dot).
3. Accumulates per-channel gate scores and up scores in fp32 separately,
   shape [BLOCK_B, BLOCK_G * 8].
4. Quantizes the gate accumulator to the output dtype before the top-2
   reduce, matching v_09's tie/round behavior so the resulting topk
   indices are bit-identical to v_09's path.
5. Inside each 8-channel group, picks the top-2 indices by gate, gathers
   the corresponding two `acc_up` lanes via a one-hot mask reduce, and
   writes only [B, K] sized topk_vals / topk_idx / sparse_up.

Never materializes `gate_full[B,I]`, `up_full[B,I]`, or `z_full[B,I]`. The
8-lane up accumulators live entirely in registers and the 6 unselected
lanes are dropped.

Shape requirements:
- intermediate I must be divisible by 8 (groups = I / 8).
- gate_weight and up_weight must be contiguous [I, H] bf16/fp16 (matches
  nn.Linear weight layout).
- x must be [B, H] same dtype.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gate_top2_of_8_selected_up_kernel(
    x_ptr, gw_ptr, uw_ptr, v_ptr, i_ptr, su_ptr,
    B, H, groups,
    s_xb, s_xh,
    s_gwi, s_gwh,
    s_uwi, s_uwh,
    s_vb, s_vk,
    s_ib, s_ik,
    s_sub, s_suk,
    BLOCK_B: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)        # [BLOCK_B]
    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)        # [BLOCK_G]
    eight = tl.arange(0, 8)                                 # [8]
    h_arange = tl.arange(0, BLOCK_H)                        # [BLOCK_H]

    b_mask = b_offs < B
    g_mask = g_offs < groups

    # Channel offsets [BLOCK_G, 8] -> flat [BLOCK_G * 8]
    chan = g_offs[:, None] * 8 + eight[None, :]
    chan_flat = tl.reshape(chan, (BLOCK_G * 8,))
    chan_mask_flat = tl.reshape(
        g_mask[:, None] & tl.full((1, 8), True, tl.int1),
        (BLOCK_G * 8,),
    )

    # Two fp32 accumulators sharing the same x-load
    acc_gate = tl.zeros((BLOCK_B, BLOCK_G * 8), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_B, BLOCK_G * 8), dtype=tl.float32)

    # H tiling
    for h_start in range(0, H, BLOCK_H):
        h_offs = h_start + h_arange
        h_mask = h_offs < H

        # x_tile reused across both gate and up dot products
        x_ptrs = x_ptr + b_offs[:, None] * s_xb + h_offs[None, :] * s_xh
        x_tile = tl.load(
            x_ptrs,
            mask=b_mask[:, None] & h_mask[None, :],
            other=0.0,
        )                                                    # [BLOCK_B, BLOCK_H]

        gw_ptrs = gw_ptr + chan_flat[:, None] * s_gwi + h_offs[None, :] * s_gwh
        gw_tile = tl.load(
            gw_ptrs,
            mask=chan_mask_flat[:, None] & h_mask[None, :],
            other=0.0,
        )                                                    # [BLOCK_G*8, BLOCK_H]

        uw_ptrs = uw_ptr + chan_flat[:, None] * s_uwi + h_offs[None, :] * s_uwh
        uw_tile = tl.load(
            uw_ptrs,
            mask=chan_mask_flat[:, None] & h_mask[None, :],
            other=0.0,
        )

        # Two GEMM accumulators
        gw_t = tl.trans(gw_tile)                             # [BLOCK_H, BLOCK_G*8]
        uw_t = tl.trans(uw_tile)
        acc_gate += tl.dot(x_tile, gw_t, out_dtype=tl.float32)
        acc_up += tl.dot(x_tile, uw_t, out_dtype=tl.float32)

    out_dtype = v_ptr.dtype.element_ty

    # ---- Top-2 selection by gate (matches v_09 quantize-before-topk) ----
    g_quant = tl.reshape(acc_gate.to(out_dtype), (BLOCK_B, BLOCK_G, 8))
    g_f = g_quant.to(tl.float32)

    NEG_INF = -float('inf')
    g_mask_3d = b_mask[:, None, None] & g_mask[None, :, None]
    g_f = tl.where(g_mask_3d, g_f, tl.full(g_f.shape, NEG_INF, tl.float32))

    max1_val = tl.max(g_f, axis=-1)                          # [BLOCK_B, BLOCK_G]
    max1_mask = (g_f == max1_val[:, :, None])
    BIG = tl.full((BLOCK_B, BLOCK_G, 8), 8, tl.int32)
    b_candidate1 = tl.where(max1_mask, eight[None, None, :].to(tl.int32), BIG)
    max1_b = tl.min(b_candidate1, axis=-1)                   # [BLOCK_B, BLOCK_G] int32

    not_top1 = (eight[None, None, :] != max1_b[:, :, None])
    g_f_masked = tl.where(not_top1, g_f, tl.full(g_f.shape, NEG_INF, tl.float32))
    max2_val = tl.max(g_f_masked, axis=-1)
    max2_mask = (g_f_masked == max2_val[:, :, None])
    b_candidate2 = tl.where(max2_mask, eight[None, None, :].to(tl.int32), BIG)
    max2_b = tl.min(b_candidate2, axis=-1)

    # ---- Gather corresponding sparse_up values via one-hot mask reduce ----
    # Match the v_07 selected_up_gather_dot semantics: fp32 acc -> bf16 store.
    u_quant = tl.reshape(acc_up.to(out_dtype), (BLOCK_B, BLOCK_G, 8))
    u_f = u_quant.to(tl.float32)
    # Multiply per-lane up acc by 0/1 mask of selected lane, then sum across the
    # 8-lane axis to extract the top-1's up value.
    sel1 = (eight[None, None, :] == max1_b[:, :, None]).to(tl.float32)
    sel2 = (eight[None, None, :] == max2_b[:, :, None]).to(tl.float32)
    up1_val = tl.sum(u_f * sel1, axis=-1)                    # [BLOCK_B, BLOCK_G]
    up2_val = tl.sum(u_f * sel2, axis=-1)

    # ---- Output addressing ----
    global1 = (g_offs[None, :] * 8 + max1_b).to(tl.int64)
    global2 = (g_offs[None, :] * 8 + max2_b).to(tl.int64)

    out_k_a = g_offs[None, :] * 2 + 0
    out_k_b = g_offs[None, :] * 2 + 1
    b_idx = b_offs[:, None]
    write_mask = b_mask[:, None] & g_mask[None, :]

    v_dtype = v_ptr.dtype.element_ty
    su_dtype = su_ptr.dtype.element_ty

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
    tl.store(
        su_ptr + b_idx * s_sub + out_k_a * s_suk,
        up1_val.to(su_dtype), mask=write_mask,
    )
    tl.store(
        su_ptr + b_idx * s_sub + out_k_b * s_suk,
        up2_val.to(su_dtype), mask=write_mask,
    )


def fused_gate_top2of8_selected_up(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    grouped_b: int = 8,
    BLOCK_B: int = 16,
    BLOCK_G: int = 16,
    BLOCK_H: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused gate_proj + grouped top-2-of-8 selector + selected_up Triton kernel.

    Args:
        x: [B, H] bf16/fp16 CUDA tensor.
        gate_weight: [I, H] bf16/fp16 CUDA, contiguous (nn.Linear weight).
        up_weight: [I, H] bf16/fp16 CUDA, contiguous.
        grouped_b: must be 8.
    Returns:
        topk_vals: [B, K] gate scores at selected top-2 lanes per group.
        topk_idx:  [B, K] int64 global channel indices.
        sparse_up: [B, K] up scores at the same top-2 lanes per group.

    The K dimension follows v_09 layout: [g0_top1, g0_top2, g1_top1, g1_top2, ...].
    K = (I / 8) * 2.

    The kernel never materializes gate_full, up_full, or z_full.
    """
    if grouped_b != 8:
        raise ValueError(f"only grouped_b=8 is supported, got {grouped_b}")
    if x.dim() != 2 or gate_weight.dim() != 2 or up_weight.dim() != 2:
        raise ValueError("x, gate_weight, up_weight must all be 2D")
    if not (x.is_cuda and gate_weight.is_cuda and up_weight.is_cuda):
        raise ValueError("x, gate_weight, up_weight must be CUDA")
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

    topk_vals = torch.empty(B, K, device=x.device, dtype=x.dtype)
    topk_idx = torch.empty(B, K, device=x.device, dtype=torch.int64)
    sparse_up = torch.empty(B, K, device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(groups, BLOCK_G))
    _fused_gate_top2_of_8_selected_up_kernel[grid](
        x, gate_weight, up_weight, topk_vals, topk_idx, sparse_up,
        B, H, groups,
        x.stride(0), x.stride(1),
        gate_weight.stride(0), gate_weight.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        topk_vals.stride(0), topk_vals.stride(1),
        topk_idx.stride(0), topk_idx.stride(1),
        sparse_up.stride(0), sparse_up.stride(1),
        BLOCK_B=BLOCK_B, BLOCK_G=BLOCK_G, BLOCK_H=BLOCK_H,
    )
    return topk_vals, topk_idx, sparse_up

