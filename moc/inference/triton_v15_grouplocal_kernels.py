"""v_15 group-local fused gate+top2+selected_up+SiLU + group-local selected-down.

Why group-local
---------------
v_14 already removed the `topk_vals` and `sparse_up` HBM round trips, but it
still wrote `topk_idx[B, K]` as int64 (64 bits per selected channel) and the
v_14 lean down kernel re-read this index in every H block. For 2:8 grouped
sparsity with coalesced per-group access, the natural boundary between kernels
is therefore not "global int64 channel id"
but "per-group local lane in [0, 8)", which fits in a single byte (or 3 bits).

v_15 changes the boundary format to a group-local representation:

- Kernel A still computes the 8-way fp32 dot of `x` against `gate_weight`
  and `up_weight` per group, picks top-2 by gate, applies SiLU on the selected
  gate values, multiplies by the corresponding up accumulators, and writes:
    * `local_idx_uint8[B, groups, 2]`  (uint8, value in 0..7)
    * `sparse_z[B, groups, 2]`         (bf16, group-major contiguous layout)
- Kernel B (group-local selected-down) loops directly over `groups`, loads
  the two `local_idx` bytes and the two `sparse_z` bf16 values per group,
  reconstructs the global row as `group * 8 + local`, and accumulates two
  rows of `down_weight_t` per group into the fp32 H accumulator.

Memory accounting at bs=1, K=1366, groups=683, H=2048:
  v_14 boundary:  topk_idx int64 [B,K]   = 1366*8 = 10936 B
                  sparse_z bf16  [B,K]   = 1366*2 =  2732 B
  v_15 boundary:  local_idx u8  [B,G,2]  =  683*2 =  1366 B
                  sparse_z bf16 [B,G,2]  =  683*4 =  2732 B
Total boundary bytes drop from 13.7 KB to 4.1 KB at bs=1; per H block re-read
also drops accordingly. SiLU placement, MoC_{2:8} top-2 semantics, and
quantize-before-topk tie behavior are preserved bit-for-bit relative to
v_13/v_14.

A reconstruction helper turns `local_idx` into the v_14 global int64
`topk_idx` for benchmarks that compare bit-exactness against v_14.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# =====================================================================
# Kernel A: fused gate+top2+selected_up+SiLU, GROUP-LOCAL output.
#
# Layout note: outputs are written in group-major order. For a single batch
# row b, the layout is identical to a (groups, 2) tile flattened to K=2*groups,
# with K[2*g+0] = top-1 lane and K[2*g+1] = top-2 lane within group g. The
# v_14 sparse_z layout already used this same flattening, so v_15 keeps the
# same K-axis convention (g0_top1, g0_top2, g1_top1, ...).
# =====================================================================

@triton.jit
def _v15_fused_gate_top2_of_8_selected_up_silu_grouplocal_kernel(
    x_ptr, gw_ptr, uw_ptr, lidx_ptr, sz_ptr,
    B, H, groups,
    s_xb, s_xh,
    s_gwi, s_gwh,
    s_uwi, s_uwh,
    s_lib, s_lik,
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

    sz_dtype = sz_ptr.dtype.element_ty

    g_quant = tl.reshape(acc_gate.to(sz_dtype), (BLOCK_B, BLOCK_G, 8))
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

    u_quant = tl.reshape(acc_up.to(sz_dtype), (BLOCK_B, BLOCK_G, 8))
    u_f = u_quant.to(tl.float32)
    sel1 = (eight[None, None, :] == max1_b[:, :, None]).to(tl.float32)
    sel2 = (eight[None, None, :] == max2_b[:, :, None]).to(tl.float32)
    up1_val = tl.sum(u_f * sel1, axis=-1)
    up2_val = tl.sum(u_f * sel2, axis=-1)

    sig1 = 1.0 / (1.0 + tl.exp(-max1_val))
    sig2 = 1.0 / (1.0 + tl.exp(-max2_val))
    sz1 = (max1_val * sig1) * up1_val
    sz2 = (max2_val * sig2) * up2_val

    out_k_a = g_offs[None, :] * 2 + 0
    out_k_b = g_offs[None, :] * 2 + 1
    b_idx = b_offs[:, None]
    write_mask = b_mask[:, None] & g_mask[None, :]

    li_dtype = lidx_ptr.dtype.element_ty

    tl.store(
        lidx_ptr + b_idx * s_lib + out_k_a * s_lik,
        max1_b.to(li_dtype), mask=write_mask,
    )
    tl.store(
        lidx_ptr + b_idx * s_lib + out_k_b * s_lik,
        max2_b.to(li_dtype), mask=write_mask,
    )
    tl.store(
        sz_ptr + b_idx * s_szb + out_k_a * s_szk,
        sz1.to(sz_dtype), mask=write_mask,
    )
    tl.store(
        sz_ptr + b_idx * s_szb + out_k_b * s_szk,
        sz2.to(sz_dtype), mask=write_mask,
    )


def fused_gate_top2of8_selected_up_silu_grouplocal(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    grouped_b: int = 8,
    BLOCK_B: int = 16,
    BLOCK_G: int = 16,
    BLOCK_H: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """v_15 group-local Kernel A.

    Returns:
        local_idx_uint8: [B, K] uint8 with K = 2*groups, values in 0..7,
            laid out as [g0_top1, g0_top2, g1_top1, g1_top2, ...].
        sparse_z:        [B, K] same dtype as x; sparse_z[b, 2g+j] =
            silu(top_gate_b_g_j) * up_b_g_j.

    Compared to v_14 fused_gate_top2of8_selected_up_silu:
      - Replaces int64 topk_idx [B, K] (K=1366 -> 10.9 KB at bs=1) with
        uint8 local_idx [B, K] (1.4 KB at bs=1). Same K layout.
      - Same MoC_{2:8} semantics: Top-K on raw gate before SiLU; SiLU only
        on selected; quantize-before-topk for tie compatibility with v_09.
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

    local_idx = torch.empty(B, K, device=x.device, dtype=torch.uint8)
    sparse_z = torch.empty(B, K, device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(groups, BLOCK_G))
    _v15_fused_gate_top2_of_8_selected_up_silu_grouplocal_kernel[grid](
        x, gate_weight, up_weight, local_idx, sparse_z,
        B, H, groups,
        x.stride(0), x.stride(1),
        gate_weight.stride(0), gate_weight.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        local_idx.stride(0), local_idx.stride(1),
        sparse_z.stride(0), sparse_z.stride(1),
        BLOCK_B=BLOCK_B, BLOCK_G=BLOCK_G, BLOCK_H=BLOCK_H,
    )
    return local_idx, sparse_z


def reconstruct_global_idx(local_idx: torch.Tensor) -> torch.Tensor:
    """Reconstruct v_14-style global int64 topk_idx [B, K] from group-local
    uint8 local_idx [B, K].

    For K[2*g + j] the global channel id is `g*8 + local_idx[2*g + j]`.
    """
    B, K = local_idx.shape
    assert K % 2 == 0
    groups = K // 2
    g_axis = torch.arange(groups, device=local_idx.device, dtype=torch.int64) * 8
    g_axis = g_axis.repeat_interleave(2)  # [K]
    return (g_axis[None, :] + local_idx.to(torch.int64)).contiguous()


# =====================================================================
# Kernel B: group-local selected-down. Loops over groups, reads two local
# lanes per group, reconstructs row = group*8 + local, accumulates against
# down_weight_t. Never reads int64 indices.
# =====================================================================

@triton.jit
def _v15_selected_down_grouplocal_kernel(
    sz_ptr, lidx_ptr, w_ptr, out_ptr,
    B, K, H, I, groups,
    s_zn, s_zk,
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

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for g_start in range(0, groups, BLOCK_G):
        g_offs = g_start + tl.arange(0, BLOCK_G)
        g_mask = g_offs < groups

        k0 = g_offs * 2 + 0
        k1 = g_offs * 2 + 1

        z0 = tl.load(
            sz_ptr + pid_b * s_zn + k0 * s_zk,
            mask=g_mask, other=0.0,
        ).to(tl.float32)
        z1 = tl.load(
            sz_ptr + pid_b * s_zn + k1 * s_zk,
            mask=g_mask, other=0.0,
        ).to(tl.float32)

        l0 = tl.load(
            lidx_ptr + pid_b * s_in + k0 * s_ik,
            mask=g_mask, other=0,
        ).to(tl.int64)
        l1 = tl.load(
            lidx_ptr + pid_b * s_in + k1 * s_ik,
            mask=g_mask, other=0,
        ).to(tl.int64)

        row0 = g_offs.to(tl.int64) * 8 + l0
        row1 = g_offs.to(tl.int64) * 8 + l1

        w0 = tl.load(
            w_ptr + row0[:, None] * s_wi + h_offs[None, :] * s_wh,
            mask=g_mask[:, None] & h_mask[None, :], other=0.0,
        ).to(tl.float32)
        w1 = tl.load(
            w_ptr + row1[:, None] * s_wi + h_offs[None, :] * s_wh,
            mask=g_mask[:, None] & h_mask[None, :], other=0.0,
        ).to(tl.float32)

        acc += tl.sum(z0[:, None] * w0, axis=0)
        acc += tl.sum(z1[:, None] * w1, axis=0)

    tl.store(
        out_ptr + pid_b * s_on + h_offs * s_oh,
        acc.to(tl.bfloat16), mask=h_mask,
    )


def selected_down_grouplocal(
    sparse_z: torch.Tensor,
    local_idx: torch.Tensor,
    w_down_t: torch.Tensor,
    BLOCK_G: int = 16,
    BLOCK_H: int = 16,
) -> torch.Tensor:
    """v_15 group-local selected-down.

    y[b, h] = sum_g sum_{j in {0,1}} sparse_z[b, 2*g+j] * w_down_t[g*8 + local_idx[b, 2*g+j], h]

    Args:
        sparse_z:  [B, K] bf16 (group-major K=2*groups, j-flat-within-group).
        local_idx: [B, K] uint8, lanes in 0..7.
        w_down_t:  [I, H] bf16 contiguous (down_proj.weight.t().contiguous()).
    """
    assert sparse_z.is_cuda and local_idx.is_cuda and w_down_t.is_cuda
    assert sparse_z.dtype == torch.bfloat16 and w_down_t.dtype == torch.bfloat16
    assert local_idx.dtype == torch.uint8
    B, K = sparse_z.shape
    I, H = w_down_t.shape
    assert I % 8 == 0
    groups = I // 8
    assert K == 2 * groups
    out = torch.empty(B, H, device=sparse_z.device, dtype=torch.bfloat16)

    grid = (B, triton.cdiv(H, BLOCK_H))
    _v15_selected_down_grouplocal_kernel[grid](
        sparse_z, local_idx, w_down_t, out,
        B, K, H, I, groups,
        sparse_z.stride(0), sparse_z.stride(1),
        local_idx.stride(0), local_idx.stride(1),
        w_down_t.stride(0), w_down_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_G=BLOCK_G, BLOCK_H=BLOCK_H,
    )
    return out


# =====================================================================
# v_16 direct-output wrappers. Same kernel bodies as the v_15 wrappers
# above; the only difference is that the wrappers do NOT allocate output
# tensors with `torch.empty(...)`. They take preallocated output buffers
# from the caller and only compute the launch grid + invoke the JIT
# kernel.
#
# Why this matters: when v_15's `make_v15_graph_runner` captured the
# wrappers above, the wrappers internally created fresh tensors per
# call. CUDA Graph capture assigns those allocations to the graph's own
# memory pool, so replay does not re-allocate; but the *captured graph
# content* still bundled the wrappers' temporaries plus a `copy_` into
# persistent buffers. That made the v_15 graph timing path different
# from what the report described as "two kernels writing into fixed
# buffers". v_16 uses these direct-output wrappers inside the captured
# region so the graph contains exactly the two kernel launches and
# nothing else.
# =====================================================================


def fused_gate_top2of8_selected_up_silu_grouplocal_into(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    local_idx_out: torch.Tensor,
    sparse_z_out: torch.Tensor,
    grouped_b: int = 8,
    BLOCK_B: int = 16,
    BLOCK_G: int = 16,
    BLOCK_H: int = 64,
) -> None:
    """Direct-output variant: writes into caller-provided buffers."""
    if grouped_b != 8:
        raise ValueError(f"only grouped_b=8 is supported, got {grouped_b}")
    if not (x.is_cuda and gate_weight.is_cuda and up_weight.is_cuda):
        raise ValueError("inputs must be CUDA")
    if not (x.dtype == gate_weight.dtype == up_weight.dtype):
        raise ValueError("dtype mismatch among x/gate/up")
    if x.dtype != sparse_z_out.dtype:
        raise ValueError(
            f"sparse_z_out dtype {sparse_z_out.dtype} must match x dtype {x.dtype}"
        )
    if local_idx_out.dtype != torch.uint8:
        raise ValueError(f"local_idx_out must be uint8, got {local_idx_out.dtype}")
    B, H = x.shape
    I_g, Hg = gate_weight.shape
    I_u, Hu = up_weight.shape
    if not (Hg == H and Hu == H):
        raise ValueError("H mismatch")
    if I_g != I_u or I_g % 8 != 0:
        raise ValueError("intermediate I mismatch or not divisible by 8")
    groups = I_g // 8
    K = groups * 2
    if local_idx_out.shape != (B, K) or sparse_z_out.shape != (B, K):
        raise ValueError(
            f"out shapes must be ({B},{K}); got "
            f"local_idx={tuple(local_idx_out.shape)}, sparse_z={tuple(sparse_z_out.shape)}"
        )

    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(groups, BLOCK_G))
    _v15_fused_gate_top2_of_8_selected_up_silu_grouplocal_kernel[grid](
        x, gate_weight, up_weight, local_idx_out, sparse_z_out,
        B, H, groups,
        x.stride(0), x.stride(1),
        gate_weight.stride(0), gate_weight.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        local_idx_out.stride(0), local_idx_out.stride(1),
        sparse_z_out.stride(0), sparse_z_out.stride(1),
        BLOCK_B=BLOCK_B, BLOCK_G=BLOCK_G, BLOCK_H=BLOCK_H,
    )


def selected_down_grouplocal_into(
    sparse_z: torch.Tensor,
    local_idx: torch.Tensor,
    w_down_t: torch.Tensor,
    out: torch.Tensor,
    BLOCK_G: int = 32,
    BLOCK_H: int = 16,
) -> None:
    """Direct-output variant: writes into caller-provided `out` buffer."""
    if not (sparse_z.is_cuda and local_idx.is_cuda and w_down_t.is_cuda and out.is_cuda):
        raise ValueError("inputs must be CUDA")
    if sparse_z.dtype != torch.bfloat16 or w_down_t.dtype != torch.bfloat16:
        raise ValueError("sparse_z and w_down_t must be bf16")
    if local_idx.dtype != torch.uint8:
        raise ValueError("local_idx must be uint8")
    if out.dtype != torch.bfloat16:
        raise ValueError("out must be bf16")
    B, K = sparse_z.shape
    I, H = w_down_t.shape
    if I % 8 != 0:
        raise ValueError("I must be divisible by 8")
    groups = I // 8
    if K != 2 * groups:
        raise ValueError(f"sparse_z K={K} must equal 2*groups={2*groups}")
    if out.shape != (B, H):
        raise ValueError(f"out shape must be ({B},{H}); got {tuple(out.shape)}")

    grid = (B, triton.cdiv(H, BLOCK_H))
    _v15_selected_down_grouplocal_kernel[grid](
        sparse_z, local_idx, w_down_t, out,
        B, K, H, I, groups,
        sparse_z.stride(0), sparse_z.stride(1),
        local_idx.stride(0), local_idx.stride(1),
        w_down_t.stride(0), w_down_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_G=BLOCK_G, BLOCK_H=BLOCK_H,
    )

