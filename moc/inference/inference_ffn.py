from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import optimized_global_moc_ops  # noqa: F401
from .triton_inference_kernels import (
    selected_down_gather_dot,
    selected_up_gather_dot,
    fused_silu_selected_down,
)
from .raft_topk import has_raft_topk, raft_topk
from .triton_gate_topk_kernels import fused_gate_top2of8
from .triton_group8_kernels import (
    selected_up_group8,
    fused_silu_selected_down_group8,
)
from .triton_gate_up_kernels import fused_gate_top2of8_selected_up
from .triton_fused_ffn_kernels import (
    fused_gate_top2of8_selected_up_silu,
    selected_down_from_sparse_z,
    selected_down_from_sparse_z_splitk,
    fused_gate_top2of8_selected_up_silu_into,
    selected_down_from_sparse_z_into,
)
from .triton_v15_grouplocal_kernels import (
    fused_gate_top2of8_selected_up_silu_grouplocal,
    selected_down_grouplocal,
    reconstruct_global_idx,
    fused_gate_top2of8_selected_up_silu_grouplocal_into,
    selected_down_grouplocal_into,
)


@dataclass(frozen=True)
class InferenceMoCConfig:
    hidden_size: int
    intermediate_size: int
    k: int
    grouped_a: Optional[int] = None
    grouped_b: Optional[int] = None

    @property
    def grouped(self) -> bool:
        return self.grouped_a is not None and self.grouped_b is not None


class InferenceMoCSwiGLUFFN(nn.Module):
    """Inference-only MoC FFN benchmark helper.

    This module keeps the training paths in src/ffn.py untouched.
    It exposes three forward modes using shared weights:

    - dense_baseline: standard dense SwiGLU FFN
    - masked_reference: dense MoC masking reference that materializes full U/Z
    - moc_inference: global Top-K inference path that computes full gate,
      then gathers only selected up/down channels for the remaining FFN work
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        k: int,
        grouped_a: Optional[int] = None,
        grouped_b: Optional[int] = None,
    ):
        super().__init__()
        if k < 1 or k > intermediate_size:
            raise ValueError(f"k={k} must be in [1, {intermediate_size}]")
        if (grouped_a is None) != (grouped_b is None):
            raise ValueError("grouped_a and grouped_b must be set together")
        if grouped_a is not None:
            if intermediate_size % grouped_b != 0:
                raise ValueError(
                    f"intermediate_size={intermediate_size} must be divisible by grouped_b={grouped_b}"
                )
            expected_k = intermediate_size * grouped_a // grouped_b
            if k != expected_k:
                raise ValueError(
                    f"grouped MoC requires k=intermediate_size*a/b={expected_k}, got {k}"
                )

        self.config = InferenceMoCConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            k=k,
            grouped_a=grouped_a,
            grouped_b=grouped_b,
        )
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self._down_weight_t: torch.Tensor | None = None
        self._down_weight_t_source_ptr: int | None = None
        # v_22: frozen flag prevents in-forward Python-state mutation under
        # torch.compile. Set to True via freeze_for_compile() before compile.
        self._compile_frozen: bool = False
        # v_10: grouped index base cache. Avoids per-call torch.arange in
        # _select_channels for grouped MoC. Re-created when (device, groups,
        # grouped_b) changes.
        self._grouped_base: torch.Tensor | None = None
        self._grouped_base_key: tuple | None = None

    def _get_grouped_base(self, device: torch.device, groups: int, grouped_b: int) -> torch.Tensor:
        key = (device, groups, grouped_b)
        if self._grouped_base is None or self._grouped_base_key != key:
            if getattr(self, "_compile_frozen", False):
                # Frozen: the cache should already match the call. If not, the
                # caller built the model on a different device than what was
                # passed to freeze_for_compile().
                if self._grouped_base is not None:
                    return self._grouped_base
            self._grouped_base = (
                torch.arange(groups, device=device, dtype=torch.int64) * grouped_b
            ).view(1, groups, 1)
            self._grouped_base_key = key
        return self._grouped_base

    def _ensure_down_weight_t(self) -> None:
        if getattr(self, "_compile_frozen", False):
            return
        source_ptr = self.down_proj.weight.data_ptr()
        if self._down_weight_t is None or self._down_weight_t_source_ptr != source_ptr:
            self._down_weight_t = self.down_proj.weight.detach().t().contiguous()
            self._down_weight_t_source_ptr = source_ptr

    def freeze_for_compile(self, device: torch.device | None = None) -> None:
        """Pre-allocate caches and freeze them so torch.compile sees no
        Python-level mutable state inside forward.

        After this call, every subsequent forward path will skip the
        cache-update branch of `_ensure_down_weight_t` and `_get_grouped_base`,
        making the Triton paths compatible with `torch.compile` + CUDAGraphs.
        """
        # Force fresh cached transposed down weight.
        self._compile_frozen = False
        self._ensure_down_weight_t()
        # Pre-warm grouped index base on the requested device.
        if device is None:
            device = self.down_proj.weight.device
        if self.config.grouped:
            groups = self.config.intermediate_size // self.config.grouped_b
            _ = self._get_grouped_base(device, groups, self.config.grouped_b)
        self._compile_frozen = True

    def _flatten_decode_input(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.dim() == 2:
            return x, tuple(x.shape)
        if x.dim() == 3 and x.shape[1] == 1:
            return x[:, 0, :], tuple(x.shape)
        raise ValueError(
            f"expected decode shape [batch, hidden] or [batch, 1, hidden], got {tuple(x.shape)}"
        )

    def _restore_output_shape(self, y_2d: torch.Tensor, original_shape: tuple[int, ...]) -> torch.Tensor:
        if len(original_shape) == 2:
            return y_2d
        return y_2d.unsqueeze(1)

    def _select_channels(self, gate_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.grouped:
            return torch.topk(
                gate_values,
                self.config.k,
                dim=-1,
                largest=True,
                sorted=False,
            )

        grouped_a = self.config.grouped_a
        grouped_b = self.config.grouped_b
        groups = self.config.intermediate_size // grouped_b
        grouped_gate = gate_values.view(gate_values.shape[0], groups, grouped_b)
        topk_vals, local_idx = torch.topk(grouped_gate, grouped_a, dim=-1)
        base = self._get_grouped_base(gate_values.device, groups, grouped_b)
        topk_idx = (local_idx + base).reshape(gate_values.shape[0], self.config.k)
        return topk_vals.reshape(gate_values.shape[0], self.config.k), topk_idx

    # v_08: experimental RAFT/cuVS Top-K. Used only by
    # moc_inference_raft_topk_fused_updown; default _select_channels stays on
    # torch.topk so all v_06/v_07 modes keep their existing semantics. cuVS only
    # exposes fp32 select_k and only supports K<=256 for warpsort algos
    # (4/5/6); for K=1024 we use radix-based algos (0=auto / 1 / 2 / 3).
    RAFT_TOPK_ALGO_DEFAULT = 0  # kAuto

    def _select_channels_raft(
        self, gate_values: torch.Tensor, algo: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RAFT/cuVS Top-K backend. Falls back to torch.topk on small grouped
        per-group selections (since RAFT works on a 2D matrix and grouped
        per-group K=2-of-8 is faster as torch.topk on the [B*groups, b] view).
        For ungrouped 1.3B K=1024 we route through raft_topk on fp32.

        Returns (topk_vals, topk_idx) in the input dtype/int64 like torch.topk.
        Note that values are returned in fp32 from RAFT regardless of
        gate dtype; we cast back to gate dtype to keep downstream semantics.
        """
        algo = self.RAFT_TOPK_ALGO_DEFAULT if algo is None else algo
        if not self.config.grouped:
            scores_fp32 = gate_values if gate_values.dtype == torch.float32 else gate_values.float()
            vals_fp32, idx = raft_topk(scores_fp32, self.config.k, sorted=False, algo=algo)
            vals = vals_fp32 if gate_values.dtype == torch.float32 else vals_fp32.to(gate_values.dtype)
            return vals, idx

        # Grouped: reshape to [B, groups, grouped_b] then 2D-flatten for RAFT
        # would require per-group K=grouped_a small select. Warpsort works for
        # K<=256, so a single combined RAFT call doesn't apply directly — keep
        # torch.topk for grouped path and document.
        return self._select_channels(gate_values)

    def dense_baseline(self, x: torch.Tensor) -> torch.Tensor:
        x_2d, original_shape = self._flatten_decode_input(x)
        y_2d = self.down_proj(F.silu(self.gate_proj(x_2d)) * self.up_proj(x_2d))
        return self._restore_output_shape(y_2d, original_shape)

    def masked_reference(self, x: torch.Tensor) -> torch.Tensor:
        x_2d, original_shape = self._flatten_decode_input(x)
        gate_full = self.gate_proj(x_2d)
        up_full = self.up_proj(x_2d)
        topk_vals, topk_idx = self._select_channels(gate_full)
        sparse_up = torch.gather(up_full, -1, topk_idx)
        sparse_z = F.silu(topk_vals) * sparse_up

        z_full = torch.zeros_like(gate_full)
        z_full.scatter_(-1, topk_idx, sparse_z)
        y_2d = self.down_proj(z_full)
        return self._restore_output_shape(y_2d, original_shape)

    def moc_inference(self, x: torch.Tensor) -> torch.Tensor:
        x_2d, original_shape = self._flatten_decode_input(x)
        gate_full = self.gate_proj(x_2d)
        topk_vals, topk_idx = self._select_channels(gate_full)

        selected_up = self.up_proj.weight[topk_idx]
        sparse_up = torch.einsum("bh,bkh->bk", x_2d, selected_up)
        sparse_z = F.silu(topk_vals) * sparse_up

        selected_down = self.down_proj.weight.t()[topk_idx]
        y_2d = torch.einsum("bk,bkh->bh", sparse_z, selected_down)
        return self._restore_output_shape(y_2d, original_shape)

    def moc_inference_bmm(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy dead-end from v_04. Kept for comparison only."""
        self._ensure_down_weight_t()
        x_2d, original_shape = self._flatten_decode_input(x)
        gate_full = self.gate_proj(x_2d)
        topk_vals, topk_idx = self._select_channels(gate_full)

        sel_up = self.up_proj.weight[topk_idx].contiguous()
        sparse_up = torch.bmm(x_2d.unsqueeze(1), sel_up.transpose(1, 2)).squeeze(1)
        sparse_z = F.silu(topk_vals) * sparse_up

        sel_down = self._down_weight_t[topk_idx].contiguous()
        y_2d = torch.bmm(sparse_z.unsqueeze(1), sel_down).squeeze(1)
        return self._restore_output_shape(y_2d, original_shape)

    def moc_inference_down_triton(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_down_weight_t()
        x_2d, original_shape = self._flatten_decode_input(x)
        gate_full = self.gate_proj(x_2d)
        topk_vals, topk_idx = self._select_channels(gate_full)

        selected_up = self.up_proj.weight[topk_idx]
        sparse_up = torch.einsum("bh,bkh->bk", x_2d, selected_up)
        sparse_z = F.silu(topk_vals) * sparse_up

        y_2d = selected_down_gather_dot(sparse_z, topk_idx, self._down_weight_t)
        return self._restore_output_shape(y_2d, original_shape)

    # v_07: defaults updated to the stable 1.3B bs=1 best block configs from v_06 sweep.
    # selected-down sweep showed (BLOCK_K=64, BLOCK_H=32) was best across bs=1/2/4 (22.39/22.17/22.11 us);
    # selected-up sweep showed bs=1/4 best at (64,128) and bs=2 best at (16,64) — they cluster at
    # ~22.3-22.5 us, so we pick (64,128) as the stable default since the bs=1 primary case prefers it.
    DOWN_BLOCK_K_DEFAULT = 64
    DOWN_BLOCK_H_DEFAULT = 32
    UP_BLOCK_K_DEFAULT = 64
    UP_BLOCK_H_DEFAULT = 128

    def moc_inference_updown_triton(
        self,
        x: torch.Tensor,
        down_block_k: int | None = None,
        down_block_h: int | None = None,
        up_block_k: int | None = None,
        up_block_h: int | None = None,
    ) -> torch.Tensor:
        self._ensure_down_weight_t()
        x_2d, original_shape = self._flatten_decode_input(x)
        gate_full = self.gate_proj(x_2d)
        topk_vals, topk_idx = self._select_channels(gate_full)

        sparse_up = selected_up_gather_dot(
            x_2d, topk_idx, self.up_proj.weight,
            BLOCK_K=up_block_k or self.UP_BLOCK_K_DEFAULT,
            BLOCK_H=up_block_h or self.UP_BLOCK_H_DEFAULT,
        )
        sparse_z = F.silu(topk_vals) * sparse_up

        y_2d = selected_down_gather_dot(
            sparse_z, topk_idx, self._down_weight_t,
            BLOCK_K=down_block_k or self.DOWN_BLOCK_K_DEFAULT,
            BLOCK_H=down_block_h or self.DOWN_BLOCK_H_DEFAULT,
        )
        return self._restore_output_shape(y_2d, original_shape)

    def moc_inference_fused_updown_triton(
        self,
        x: torch.Tensor,
        down_block_k: int | None = None,
        down_block_h: int | None = None,
        up_block_k: int | None = None,
        up_block_h: int | None = None,
    ) -> torch.Tensor:
        """v_07 next-best fusion: selected-up Triton, then fused silu+selected-down kernel.

        sparse_up [B,K] is still written to HBM (full 3-stage fusion would require
        single-program H reduction, infeasible at H=2048). The intermediate
        sparse_z = silu(topk_vals) * sparse_up is computed inside the down kernel
        in registers and never written/read in HBM.
        """
        self._ensure_down_weight_t()
        x_2d, original_shape = self._flatten_decode_input(x)
        gate_full = self.gate_proj(x_2d)
        topk_vals, topk_idx = self._select_channels(gate_full)

        sparse_up = selected_up_gather_dot(
            x_2d, topk_idx, self.up_proj.weight,
            BLOCK_K=up_block_k or self.UP_BLOCK_K_DEFAULT,
            BLOCK_H=up_block_h or self.UP_BLOCK_H_DEFAULT,
        )
        y_2d = fused_silu_selected_down(
            topk_vals, sparse_up, topk_idx, self._down_weight_t,
            BLOCK_K=down_block_k or self.DOWN_BLOCK_K_DEFAULT,
            BLOCK_H=down_block_h or self.DOWN_BLOCK_H_DEFAULT,
        )
        return self._restore_output_shape(y_2d, original_shape)

    def moc_inference_raft_topk_fused_updown(
        self,
        x: torch.Tensor,
        algo: int | None = None,
        down_block_k: int | None = None,
        down_block_h: int | None = None,
        up_block_k: int | None = None,
        up_block_h: int | None = None,
    ) -> torch.Tensor:
        """v_08 experimental: RAFT/cuVS Top-K + v_07 fused selected-up + fused
        silu+selected-down. Only ungrouped MoC; grouped falls back to
        torch.topk inside _select_channels_raft.

        Raises if the RAFT extension is not loadable.
        """
        if not has_raft_topk():
            raise RuntimeError(
                "RAFT/cuVS extension is not available; cannot run "
                "moc_inference_raft_topk_fused_updown."
            )
        self._ensure_down_weight_t()
        x_2d, original_shape = self._flatten_decode_input(x)
        gate_full = self.gate_proj(x_2d)
        topk_vals, topk_idx = self._select_channels_raft(gate_full, algo=algo)

        sparse_up = selected_up_gather_dot(
            x_2d, topk_idx, self.up_proj.weight,
            BLOCK_K=up_block_k or self.UP_BLOCK_K_DEFAULT,
            BLOCK_H=up_block_h or self.UP_BLOCK_H_DEFAULT,
        )
        y_2d = fused_silu_selected_down(
            topk_vals, sparse_up, topk_idx, self._down_weight_t,
            BLOCK_K=down_block_k or self.DOWN_BLOCK_K_DEFAULT,
            BLOCK_H=down_block_h or self.DOWN_BLOCK_H_DEFAULT,
        )
        return self._restore_output_shape(y_2d, original_shape)

    def moc_inference_optimized_global_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """Ordinary global Top-K MoC with the the single-layer benchmark optimized kernels.

        This path preserves the non-2:8 MoC semantics: rank all FFN channels,
        select the global top-K set, compute selected Up+SiLU, and compute
        selected Down.  The underlying CUDA kernels are registered as
        ``torch.library.custom_op`` so this path can be used inside
        ``torch.compile`` for end-to-end decode benchmarks.
        """
        if self.config.grouped:
            raise ValueError("moc_inference_optimized_global_cuda requires ungrouped global MoC")
        self._ensure_down_weight_t()
        x_2d, original_shape = self._flatten_decode_input(x)
        gate_full = self.gate_proj(x_2d).contiguous()
        topk_vals, topk_idx = torch.ops.moc.optimized_global_topk_bf16.default(
            gate_full,
            self.config.k,
        )
        sparse_z = torch.ops.moc.optimized_global_selected_up_silu_bf16.default(
            x_2d.contiguous(),
            topk_vals,
            topk_idx,
            self.up_proj.weight,
        )
        y_2d = torch.ops.moc.optimized_global_selected_down_bf16.default(
            sparse_z,
            topk_idx,
            self._down_weight_t,
        )
        return self._restore_output_shape(y_2d, original_shape)

    def moc_inference_optimized_global_native(self, x: torch.Tensor) -> torch.Tensor:
        """Ordinary global Top-K MoC using direct C++ dispatcher ops.

        This is equivalent to `moc_inference_optimized_global_cuda` but avoids
        the Python `torch.library.custom_op` wrapper at runtime. It is intended
        for end-to-end `torch.compile` decode experiments where custom-op
        boundary overhead is visible.
        """
        if self.config.grouped:
            raise ValueError("moc_inference_optimized_global_native requires ungrouped global MoC")
        self._ensure_down_weight_t()
        x_2d, original_shape = self._flatten_decode_input(x)
        gate_full = self.gate_proj(x_2d).contiguous()
        topk_vals, topk_idx = torch.ops.moc_native.optimized_global_topk_bf16.default(
            gate_full,
            self.config.k,
        )
        sparse_z = torch.ops.moc_native.optimized_global_selected_up_silu_bf16.default(
            x_2d.contiguous(),
            topk_vals,
            topk_idx,
            self.up_proj.weight,
        )
        y_2d = torch.ops.moc_native.optimized_global_selected_down_bf16.default(
            sparse_z,
            topk_idx,
            self._down_weight_t,
        )
        return self._restore_output_shape(y_2d, original_shape)

    def moc_inference_optimized_global_after_gate_native(self, x: torch.Tensor) -> torch.Tensor:
        """Ordinary global MoC with Top-K/selected projections behind one native op."""
        if self.config.grouped:
            raise ValueError("moc_inference_optimized_global_after_gate_native requires ungrouped global MoC")
        self._ensure_down_weight_t()
        x_2d, original_shape = self._flatten_decode_input(x)
        gate_full = self.gate_proj(x_2d).contiguous()
        y_2d = torch.ops.moc_native.optimized_global_after_gate_bf16.default(
            x_2d.contiguous(),
            gate_full,
            self.up_proj.weight,
            self._down_weight_t,
            self.config.k,
        )
        return self._restore_output_shape(y_2d, original_shape)

    def moc_inference_fused_gate_top2of8_fused_updown(
        self,
        x: torch.Tensor,
        gate_BLOCK_B: int = 16,
        gate_BLOCK_G: int = 16,
        gate_BLOCK_H: int = 64,
        down_block_k: int | None = None,
        down_block_h: int | None = None,
        up_block_k: int | None = None,
        up_block_h: int | None = None,
    ) -> torch.Tensor:
        """v_09 experimental: fused gate_proj + grouped top-2-of-8 selector
        Triton kernel, then v_07 fused selected-up + fused silu+selected-down.

        The fused gate+selector kernel never materializes gate_full [B,I];
        only the [B, K] selected (values, indices) cross HBM. K = (I/8)*2.

        Requires grouped MoC config with grouped_a=2, grouped_b=8 and
        intermediate_size divisible by 8.
        """
        if not (self.config.grouped and self.config.grouped_a == 2 and self.config.grouped_b == 8):
            raise ValueError(
                "moc_inference_fused_gate_top2of8_fused_updown requires "
                "grouped MoC with grouped_a=2, grouped_b=8."
            )
        self._ensure_down_weight_t()
        x_2d, original_shape = self._flatten_decode_input(x)

        topk_vals, topk_idx = fused_gate_top2of8(
            x_2d, self.gate_proj.weight, grouped_b=8,
            BLOCK_B=gate_BLOCK_B, BLOCK_G=gate_BLOCK_G, BLOCK_H=gate_BLOCK_H,
        )

        sparse_up = selected_up_gather_dot(
            x_2d, topk_idx, self.up_proj.weight,
            BLOCK_K=up_block_k or self.UP_BLOCK_K_DEFAULT,
            BLOCK_H=up_block_h or self.UP_BLOCK_H_DEFAULT,
        )
        y_2d = fused_silu_selected_down(
            topk_vals, sparse_up, topk_idx, self._down_weight_t,
            BLOCK_K=down_block_k or self.DOWN_BLOCK_K_DEFAULT,
            BLOCK_H=down_block_h or self.DOWN_BLOCK_H_DEFAULT,
        )
        return self._restore_output_shape(y_2d, original_shape)

    def moc_inference_fused_gate_top2of8_dense_updown(
        self,
        x: torch.Tensor,
        gate_BLOCK_B: int = 16,
        gate_BLOCK_G: int = 16,
        gate_BLOCK_H: int = 64,
    ) -> torch.Tensor:
        """Hybrid top-2-of-8 selector with dense cuBLAS up/down projections.

        This variant is useful when comparing the cost of channel selection
        against dense projection kernels. It materializes the full [B, I]
        intermediate tensors, so the sparse kernels remain the default path for
        MoC inference benchmarks.

        The selector still uses fused_gate_top2of8, so gate_full [B, I] is never
        materialized. Requires grouped_a=2, grouped_b=8 and I % 8 == 0.
        """
        if not (self.config.grouped and self.config.grouped_a == 2 and self.config.grouped_b == 8):
            raise ValueError(
                "moc_inference_fused_gate_top2of8_dense_updown requires "
                "grouped MoC with grouped_a=2, grouped_b=8."
            )
        if self.config.intermediate_size % 8 != 0:
            raise ValueError(
                f"intermediate_size={self.config.intermediate_size} must be divisible by 8"
            )
        x_2d, original_shape = self._flatten_decode_input(x)

        # selector: fused gate + grouped top-2-of-8, no gate_full materialize
        topk_vals, topk_idx = fused_gate_top2of8(
            x_2d, self.gate_proj.weight, grouped_b=8,
            BLOCK_B=gate_BLOCK_B, BLOCK_G=gate_BLOCK_G, BLOCK_H=gate_BLOCK_H,
        )

        # dense up: full [B, I] tensor-core path
        up_full = self.up_proj(x_2d)
        sparse_up = torch.gather(up_full, -1, topk_idx)
        sparse_z = F.silu(topk_vals) * sparse_up

        # scatter back to dense [B, I] for tensor-core down
        z_full = torch.zeros_like(up_full)
        z_full.scatter_(-1, topk_idx, sparse_z)

        # dense down: full [B, I] -> [B, H] tensor-core path
        y_2d = self.down_proj(z_full)
        return self._restore_output_shape(y_2d, original_shape)

    # v_11 group8 default block sizes. BLOCK_G=16 covers 16 groups (128 channels)
    # per program; BLOCK_H_UP=64 / BLOCK_H_DOWN=32 mirror the v_07/v_09 choices.
    G8_UP_BLOCK_G_DEFAULT = 16
    G8_UP_BLOCK_H_DEFAULT = 64
    G8_DOWN_BLOCK_G_DEFAULT = 16
    G8_DOWN_BLOCK_H_DEFAULT = 32

    def moc_inference_fused_gate_top2of8_group8_updown(
        self,
        x: torch.Tensor,
        gate_BLOCK_B: int = 16,
        gate_BLOCK_G: int = 16,
        gate_BLOCK_H: int = 64,
        up_block_g: int | None = None,
        up_block_h: int | None = None,
        down_block_g: int | None = None,
        down_block_h: int | None = None,
    ) -> torch.Tensor:
        """v_11: fused gate+top-2-of-8 selector (v_09) +
        group-wise 8-lane Triton selected_up + group-wise 8-lane fused
        silu+selected_down. NO `gate_full`, NO `up_full`, NO `z_full`
        materialization.

        Both Up and Down kernels iterate over MoC_{2:8} groups and load each
        group's 8 contiguous W rows in one coalesced burst, then mask to the
        2 selected lanes via topk_idx. This trades 4x compute for aligned
        contiguous IO for `MoC_{2:8}`.

        Requires grouped_a=2, grouped_b=8 and intermediate_size % 8 == 0.
        """
        if not (self.config.grouped and self.config.grouped_a == 2 and self.config.grouped_b == 8):
            raise ValueError(
                "moc_inference_fused_gate_top2of8_group8_updown requires "
                "grouped MoC with grouped_a=2, grouped_b=8."
            )
        if self.config.intermediate_size % 8 != 0:
            raise ValueError(
                f"intermediate_size={self.config.intermediate_size} must be divisible by 8"
            )
        self._ensure_down_weight_t()
        x_2d, original_shape = self._flatten_decode_input(x)

        topk_vals, topk_idx = fused_gate_top2of8(
            x_2d, self.gate_proj.weight, grouped_b=8,
            BLOCK_B=gate_BLOCK_B, BLOCK_G=gate_BLOCK_G, BLOCK_H=gate_BLOCK_H,
        )

        sparse_up = selected_up_group8(
            x_2d, topk_idx, self.up_proj.weight,
            BLOCK_G=up_block_g or self.G8_UP_BLOCK_G_DEFAULT,
            BLOCK_H=up_block_h or self.G8_UP_BLOCK_H_DEFAULT,
        )

        y_2d = fused_silu_selected_down_group8(
            topk_vals, sparse_up, topk_idx, self._down_weight_t,
            BLOCK_G=down_block_g or self.G8_DOWN_BLOCK_G_DEFAULT,
            BLOCK_H=down_block_h or self.G8_DOWN_BLOCK_H_DEFAULT,
        )
        return self._restore_output_shape(y_2d, original_shape)

    # v_13 fused gate+top2+selected_up defaults. Selected by the small
    # block sweep documented in docs/fused_gate_up_report.md.
    GUP_BLOCK_B_DEFAULT = 16
    GUP_BLOCK_G_DEFAULT = 16
    GUP_BLOCK_H_DEFAULT = 64

    def moc_inference_fused_gate_top2of8_fused_gate_up_down(
        self,
        x: torch.Tensor,
        gate_up_BLOCK_B: int | None = None,
        gate_up_BLOCK_G: int | None = None,
        gate_up_BLOCK_H: int | None = None,
        down_block_k: int | None = None,
        down_block_h: int | None = None,
    ) -> torch.Tensor:
        """v_13 fastest path: a single Triton kernel that
        produces (topk_vals, topk_idx, sparse_up) in one launch by sharing
        the H-tile x-load across gate and up GEMMs over the same 8-channel
        groups; followed by v_07/v_09 fused_silu_selected_down for the
        post-selection half.

        Compared to v_09's `moc_inference_fused_gate_top2of8_fused_updown`
        this saves:
          - 1 kernel launch (selected_up_gather_dot is folded in)
          - 1 [B, K] topk_idx HBM round-trip (idx stays in registers)
          - 1 x[B, H] HBM read (x is loaded once per H tile, reused for both
            gate and up dot products)
        Cost: each program now also accumulates 8 up lanes per group then
        keeps only the 2 selected — same 4x compute trade as v_09's
        fused gate selector, applied to up_proj.

        Never materializes gate_full, up_full, or z_full.
        Requires grouped_a=2, grouped_b=8, intermediate % 8 == 0.
        """
        if not (self.config.grouped and self.config.grouped_a == 2 and self.config.grouped_b == 8):
            raise ValueError(
                "moc_inference_fused_gate_top2of8_fused_gate_up_down requires "
                "grouped MoC with grouped_a=2, grouped_b=8."
            )
        if self.config.intermediate_size % 8 != 0:
            raise ValueError(
                f"intermediate_size={self.config.intermediate_size} must be divisible by 8"
            )
        self._ensure_down_weight_t()
        x_2d, original_shape = self._flatten_decode_input(x)

        topk_vals, topk_idx, sparse_up = fused_gate_top2of8_selected_up(
            x_2d, self.gate_proj.weight, self.up_proj.weight, grouped_b=8,
            BLOCK_B=gate_up_BLOCK_B or self.GUP_BLOCK_B_DEFAULT,
            BLOCK_G=gate_up_BLOCK_G or self.GUP_BLOCK_G_DEFAULT,
            BLOCK_H=gate_up_BLOCK_H or self.GUP_BLOCK_H_DEFAULT,
        )
        y_2d = fused_silu_selected_down(
            topk_vals, sparse_up, topk_idx, self._down_weight_t,
            BLOCK_K=down_block_k or self.DOWN_BLOCK_K_DEFAULT,
            BLOCK_H=down_block_h or self.DOWN_BLOCK_H_DEFAULT,
        )
        return self._restore_output_shape(y_2d, original_shape)

    # v_14 fused gate+top2+selected_up+silu defaults. The front-half kernel
    # default reuses v_13's swept (BLOCK_G=16, BLOCK_H=64) at BLOCK_B=16.
    GUSILU_BLOCK_B_DEFAULT = 16
    GUSILU_BLOCK_G_DEFAULT = 16
    GUSILU_BLOCK_H_DEFAULT = 64
    # v_15 alignment: switch lean Down default to v_14 full benchmark sweep
    # winner (BLOCK_K=128, BLOCK_H=16). The previous (64, 32) default was
    # never the swept best; v_14 reports were already using (128, 16) for
    # component timings and that is the value v_15 must ship as the actual
    # forward-path default so e2e and component numbers share one block.
    V14_DOWN_BLOCK_K_DEFAULT = 128
    V14_DOWN_BLOCK_H_DEFAULT = 16
    V14_DOWN_SPLITS_DEFAULT = 4

    def moc_inference_fused_gate_top2of8_fused_gate_up_silu_down(
        self,
        x: torch.Tensor,
        gate_up_BLOCK_B: int | None = None,
        gate_up_BLOCK_G: int | None = None,
        gate_up_BLOCK_H: int | None = None,
        down_block_k: int | None = None,
        down_block_h: int | None = None,
    ) -> torch.Tensor:
        """v_14: fuses SiLU into the v_13 gate+up kernel and
        runs a leaner selected-down kernel that reads only (idx, sparse_z).

        Vs v_13 the front half writes one fewer [B, K] bf16 buffer
        (topk_vals); the down half no longer reads topk_vals or applies
        SiLU inline. MoC_{2:8} semantics preserved (Top-K on raw gate
        BEFORE SiLU; SiLU applied to selected top-K vals only).

        No gate_full / up_full / z_full materialization.
        """
        if not (self.config.grouped and self.config.grouped_a == 2 and self.config.grouped_b == 8):
            raise ValueError(
                "moc_inference_fused_gate_top2of8_fused_gate_up_silu_down requires "
                "grouped MoC with grouped_a=2, grouped_b=8."
            )
        if self.config.intermediate_size % 8 != 0:
            raise ValueError(
                f"intermediate_size={self.config.intermediate_size} must be divisible by 8"
            )
        self._ensure_down_weight_t()
        x_2d, original_shape = self._flatten_decode_input(x)

        topk_idx, sparse_z = fused_gate_top2of8_selected_up_silu(
            x_2d, self.gate_proj.weight, self.up_proj.weight, grouped_b=8,
            BLOCK_B=gate_up_BLOCK_B or self.GUSILU_BLOCK_B_DEFAULT,
            BLOCK_G=gate_up_BLOCK_G or self.GUSILU_BLOCK_G_DEFAULT,
            BLOCK_H=gate_up_BLOCK_H or self.GUSILU_BLOCK_H_DEFAULT,
        )
        y_2d = selected_down_from_sparse_z(
            sparse_z, topk_idx, self._down_weight_t,
            BLOCK_K=down_block_k or self.V14_DOWN_BLOCK_K_DEFAULT,
            BLOCK_H=down_block_h or self.V14_DOWN_BLOCK_H_DEFAULT,
        )
        return self._restore_output_shape(y_2d, original_shape)

    def moc_inference_fused_gate_top2of8_fused_gate_up_silu_down_splitk(
        self,
        x: torch.Tensor,
        gate_up_BLOCK_B: int | None = None,
        gate_up_BLOCK_G: int | None = None,
        gate_up_BLOCK_H: int | None = None,
        down_block_k: int | None = None,
        down_block_h: int | None = None,
        down_splits: int | None = None,
    ) -> torch.Tensor:
        """v_14 split-K variant: same Kernel A as the non-split mode, plus
        a split-K selected-down (atomic-add partial sums + fp32->bf16 cast)
        to multiply parallelism beyond the few dozen programs B*ceil(H/BLOCK_H)
        gives at bs=1.
        """
        if not (self.config.grouped and self.config.grouped_a == 2 and self.config.grouped_b == 8):
            raise ValueError(
                "moc_inference_fused_gate_top2of8_fused_gate_up_silu_down_splitk requires "
                "grouped MoC with grouped_a=2, grouped_b=8."
            )
        if self.config.intermediate_size % 8 != 0:
            raise ValueError(
                f"intermediate_size={self.config.intermediate_size} must be divisible by 8"
            )
        self._ensure_down_weight_t()
        x_2d, original_shape = self._flatten_decode_input(x)

        topk_idx, sparse_z = fused_gate_top2of8_selected_up_silu(
            x_2d, self.gate_proj.weight, self.up_proj.weight, grouped_b=8,
            BLOCK_B=gate_up_BLOCK_B or self.GUSILU_BLOCK_B_DEFAULT,
            BLOCK_G=gate_up_BLOCK_G or self.GUSILU_BLOCK_G_DEFAULT,
            BLOCK_H=gate_up_BLOCK_H or self.GUSILU_BLOCK_H_DEFAULT,
        )
        y_2d = selected_down_from_sparse_z_splitk(
            sparse_z, topk_idx, self._down_weight_t,
            BLOCK_K=down_block_k or self.V14_DOWN_BLOCK_K_DEFAULT,
            BLOCK_H=down_block_h or self.V14_DOWN_BLOCK_H_DEFAULT,
            SPLITS=down_splits or self.V14_DOWN_SPLITS_DEFAULT,
        )
        return self._restore_output_shape(y_2d, original_shape)

    # v_15 group-local defaults. Kernel A reuses the v_14 swept block; the
    # group-local Down kernel is brand new, defaults documented in the v_15
    # benchmark report after a small (BLOCK_G, BLOCK_H) sweep.
    V15_GUSILU_BLOCK_B_DEFAULT = 16
    V15_GUSILU_BLOCK_G_DEFAULT = 16
    V15_GUSILU_BLOCK_H_DEFAULT = 64
    # v_16 fix: align direct-mode default Down block with the v_15 full
    # benchmark sweep best (32, 16). Calling the v_15 mode without explicit
    # block now uses the same path the report numbers were collected on.
    V15_DOWN_BLOCK_G_DEFAULT = 32
    V15_DOWN_BLOCK_H_DEFAULT = 16

    def moc_inference_fused_gate_top2of8_group_local_silu_down(
        self,
        x: torch.Tensor,
        gate_up_BLOCK_B: int | None = None,
        gate_up_BLOCK_G: int | None = None,
        gate_up_BLOCK_H: int | None = None,
        down_block_g: int | None = None,
        down_block_h: int | None = None,
    ) -> torch.Tensor:
        """v_15 group-local path: writes uint8 local lanes (in-group 0..7)
        and group-major sparse_z, then runs a group-local selected-down that
        reconstructs `row = group * 8 + local` per group.

        Compared to v_14, the cross-kernel boundary drops from
        (int64 topk_idx + bf16 sparse_z) to (uint8 local_idx + bf16 sparse_z),
        cutting the int64 boundary traffic to 1/8 in bytes and aligning the
        Down kernel inner loop with the 2:8 group structure.

        Same MoC_{2:8} math: Top-K on raw gate before SiLU; SiLU only on
        selected lanes. Reconstructed global idx is bit-exact with v_14.
        """
        if not (self.config.grouped and self.config.grouped_a == 2 and self.config.grouped_b == 8):
            raise ValueError(
                "moc_inference_fused_gate_top2of8_group_local_silu_down requires "
                "grouped MoC with grouped_a=2, grouped_b=8."
            )
        if self.config.intermediate_size % 8 != 0:
            raise ValueError(
                f"intermediate_size={self.config.intermediate_size} must be divisible by 8"
            )
        self._ensure_down_weight_t()
        x_2d, original_shape = self._flatten_decode_input(x)

        local_idx, sparse_z = fused_gate_top2of8_selected_up_silu_grouplocal(
            x_2d, self.gate_proj.weight, self.up_proj.weight, grouped_b=8,
            BLOCK_B=gate_up_BLOCK_B or self.V15_GUSILU_BLOCK_B_DEFAULT,
            BLOCK_G=gate_up_BLOCK_G or self.V15_GUSILU_BLOCK_G_DEFAULT,
            BLOCK_H=gate_up_BLOCK_H or self.V15_GUSILU_BLOCK_H_DEFAULT,
        )
        y_2d = selected_down_grouplocal(
            sparse_z, local_idx, self._down_weight_t,
            BLOCK_G=down_block_g or self.V15_DOWN_BLOCK_G_DEFAULT,
            BLOCK_H=down_block_h or self.V15_DOWN_BLOCK_H_DEFAULT,
        )
        return self._restore_output_shape(y_2d, original_shape)

    # ---------------------------------------------------------------------
    # v_15 allocation-free CUDA Graph replay path. For fixed shape decode
    # (bs, hidden, intermediate, grouped_b, K) the two kernels can be
    # captured into a CUDA Graph so per-step Python dispatch and
    # allocation overhead disappear at steady state.
    # ---------------------------------------------------------------------
    _v15_graph_state: dict | None = None

    def make_v15_graph_runner(
        self,
        batch_size: int,
        gate_up_BLOCK_B: int | None = None,
        gate_up_BLOCK_G: int | None = None,
        gate_up_BLOCK_H: int | None = None,
        down_block_g: int | None = None,
        down_block_h: int | None = None,
    ):
        """Build a (capture-once / replay-many) runner for v_15 group-local
        decode at a fixed batch size. Returns a callable
        `runner(x_in: Tensor[batch_size, hidden]) -> Tensor[batch_size, hidden]`.

        The returned function:
          - copies `x_in` into the captured input buffer in place;
          - replays the captured graph;
          - returns a view of the captured output buffer.

        Re-using the same runner across calls amortizes allocation and Python
        dispatch overhead. The same shape and same block sizes must be used
        for replay; calling with a different batch size requires building a
        new runner.
        """
        if not (self.config.grouped and self.config.grouped_a == 2 and self.config.grouped_b == 8):
            raise ValueError(
                "v_15 graph runner requires grouped MoC with grouped_a=2, grouped_b=8."
            )
        if self.config.intermediate_size % 8 != 0:
            raise ValueError("intermediate_size must be divisible by 8")
        self._ensure_down_weight_t()

        H = self.config.hidden_size
        I = self.config.intermediate_size
        groups = I // 8
        K = groups * 2
        device = self.gate_proj.weight.device
        dtype = self.gate_proj.weight.dtype

        bb = gate_up_BLOCK_B or self.V15_GUSILU_BLOCK_B_DEFAULT
        bg = gate_up_BLOCK_G or self.V15_GUSILU_BLOCK_G_DEFAULT
        bh = gate_up_BLOCK_H or self.V15_GUSILU_BLOCK_H_DEFAULT
        dbg = down_block_g or self.V15_DOWN_BLOCK_G_DEFAULT
        dbh = down_block_h or self.V15_DOWN_BLOCK_H_DEFAULT

        # Persistent captured buffers
        x_buf = torch.empty(batch_size, H, device=device, dtype=dtype)
        local_idx_buf = torch.empty(batch_size, K, device=device, dtype=torch.uint8)
        sparse_z_buf = torch.empty(batch_size, K, device=device, dtype=dtype)
        y_buf = torch.empty(batch_size, H, device=device, dtype=dtype)

        # ---- Warm-up under a fresh stream so capture sees the right ops ----
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                local_idx_buf, sparse_z_buf = self._v15_graph_run_inplace_kernels(
                    x_buf, local_idx_buf, sparse_z_buf, y_buf,
                    bb, bg, bh, dbg, dbh,
                )
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._v15_graph_run_inplace_kernels(
                x_buf, local_idx_buf, sparse_z_buf, y_buf,
                bb, bg, bh, dbg, dbh,
            )

        state = {
            "graph": graph,
            "x_buf": x_buf,
            "y_buf": y_buf,
            "local_idx_buf": local_idx_buf,
            "sparse_z_buf": sparse_z_buf,
            "batch_size": batch_size,
            "block": dict(BB=bb, BG=bg, BH=bh, DBG=dbg, DBH=dbh),
        }
        self._v15_graph_state = state

        def runner(x_in: torch.Tensor) -> torch.Tensor:
            x_2d, original_shape = self._flatten_decode_input(x_in)
            if x_2d.shape[0] != batch_size:
                raise ValueError(
                    f"runner built for batch_size={batch_size}, got x.shape[0]={x_2d.shape[0]}"
                )
            x_buf.copy_(x_2d)
            graph.replay()
            return self._restore_output_shape(y_buf, original_shape)

        return runner

    def _v15_graph_run_inplace_kernels(
        self,
        x_buf: torch.Tensor,
        local_idx_buf: torch.Tensor,
        sparse_z_buf: torch.Tensor,
        y_buf: torch.Tensor,
        bb: int, bg: int, bh: int, dbg: int, dbh: int,
    ):
        """v_16 hardened: invokes the v_15 kernels via direct-output wrappers
        so the captured graph contains EXACTLY two kernel launches that
        write into the persistent buffers. No `torch.empty` and no `copy_`
        inside the captured region.
        """
        fused_gate_top2of8_selected_up_silu_grouplocal_into(
            x_buf, self.gate_proj.weight, self.up_proj.weight,
            local_idx_out=local_idx_buf, sparse_z_out=sparse_z_buf,
            grouped_b=8, BLOCK_B=bb, BLOCK_G=bg, BLOCK_H=bh,
        )
        selected_down_grouplocal_into(
            sparse_z_buf, local_idx_buf, self._down_weight_t,
            out=y_buf, BLOCK_G=dbg, BLOCK_H=dbh,
        )
        return local_idx_buf, sparse_z_buf

    # ---------------------------------------------------------------------
    # v_16 fair graph runners. Build the equivalent capture / replay path
    # for dense FFN and for the v_14 fused-gate-up-silu-down mode, so all
    # three graph modes can be compared head-to-head under the same
    # "captured fixed-shape decode" optimization class.
    # ---------------------------------------------------------------------

    def make_dense_graph_runner(self, batch_size: int):
        """Capture dense FFN `down_proj(silu(gate_proj(x)) * up_proj(x))` as
        a CUDA Graph for fixed batch size. This is the FAIR baseline for
        the v_15 graph runner.
        """
        self._ensure_down_weight_t()  # not strictly needed for dense; harmless
        H = self.config.hidden_size
        device = self.gate_proj.weight.device
        dtype = self.gate_proj.weight.dtype

        x_buf = torch.empty(batch_size, H, device=device, dtype=dtype)
        y_buf = torch.empty(batch_size, H, device=device, dtype=dtype)

        def _run_dense_into(x_in: torch.Tensor, y_out: torch.Tensor) -> None:
            g = self.gate_proj(x_in)
            u = self.up_proj(x_in)
            z = F.silu(g) * u
            y_tmp = self.down_proj(z)
            y_out.copy_(y_tmp)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _run_dense_into(x_buf, y_buf)
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_dense_into(x_buf, y_buf)

        def runner(x_in: torch.Tensor) -> torch.Tensor:
            x_2d, original_shape = self._flatten_decode_input(x_in)
            if x_2d.shape[0] != batch_size:
                raise ValueError(
                    f"dense runner built for batch_size={batch_size}, got {x_2d.shape[0]}"
                )
            x_buf.copy_(x_2d)
            graph.replay()
            return self._restore_output_shape(y_buf, original_shape)

        return runner

    def make_v14_graph_runner(
        self,
        batch_size: int,
        gate_up_BLOCK_B: int | None = None,
        gate_up_BLOCK_G: int | None = None,
        gate_up_BLOCK_H: int | None = None,
        down_block_k: int | None = None,
        down_block_h: int | None = None,
    ):
        """v_17 hardened: capture the v_14 fused gate+up+silu + lean down
        path as a CUDA Graph using direct-output wrappers. The captured
        region contains exactly two Triton kernel launches writing into
        persistent (topk_idx_buf, sparse_z_buf, y_buf). No torch.empty
        or copy_ inside the captured region.

        This is the primary MoC 2:8 inference graph runner. It preserves
        strict MoC_{2:8} semantics: Top-K on
        raw gate before SiLU, SiLU only on selected top-2 channels.
        """
        if not (self.config.grouped and self.config.grouped_a == 2 and self.config.grouped_b == 8):
            raise ValueError(
                "v_14 graph runner requires grouped MoC with grouped_a=2, grouped_b=8."
            )
        self._ensure_down_weight_t()
        H = self.config.hidden_size
        I = self.config.intermediate_size
        groups = I // 8
        K = groups * 2
        device = self.gate_proj.weight.device
        dtype = self.gate_proj.weight.dtype

        bb = gate_up_BLOCK_B or self.GUSILU_BLOCK_B_DEFAULT
        bg = gate_up_BLOCK_G or self.GUSILU_BLOCK_G_DEFAULT
        bh = gate_up_BLOCK_H or self.GUSILU_BLOCK_H_DEFAULT
        dbk = down_block_k or self.V14_DOWN_BLOCK_K_DEFAULT
        dbh = down_block_h or self.V14_DOWN_BLOCK_H_DEFAULT

        x_buf = torch.empty(batch_size, H, device=device, dtype=dtype)
        topk_idx_buf = torch.empty(batch_size, K, device=device, dtype=torch.int64)
        sparse_z_buf = torch.empty(batch_size, K, device=device, dtype=dtype)
        y_buf = torch.empty(batch_size, H, device=device, dtype=dtype)

        def _run_v14_direct() -> None:
            fused_gate_top2of8_selected_up_silu_into(
                x_buf, self.gate_proj.weight, self.up_proj.weight,
                topk_idx_out=topk_idx_buf, sparse_z_out=sparse_z_buf,
                grouped_b=8, BLOCK_B=bb, BLOCK_G=bg, BLOCK_H=bh,
            )
            selected_down_from_sparse_z_into(
                sparse_z_buf, topk_idx_buf, self._down_weight_t,
                out=y_buf, BLOCK_K=dbk, BLOCK_H=dbh,
            )

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _run_v14_direct()
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_v14_direct()

        def runner(x_in: torch.Tensor) -> torch.Tensor:
            x_2d, original_shape = self._flatten_decode_input(x_in)
            if x_2d.shape[0] != batch_size:
                raise ValueError(
                    f"v14 graph runner built for batch_size={batch_size}, got {x_2d.shape[0]}"
                )
            x_buf.copy_(x_2d)
            graph.replay()
            return self._restore_output_shape(y_buf, original_shape)

        return runner

    # Public alias for the v14 graph runner.
    make_moc_v14_graph_runner = make_v14_graph_runner

    # v_22: compile-friendly path. Uses Triton kernels wrapped as
    # `torch.library.custom_op` (defined in triton_fused_ffn_kernels.py) so
    # Inductor treats them as opaque ops and CUDAGraphs do not record any
    # Python-state mutation. Requires `freeze_for_compile()` to have been
    # called before `torch.compile`, so that `self._down_weight_t` is stable.
    def moc_inference_compile_friendly(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.config.grouped and self.config.grouped_a == 2 and self.config.grouped_b == 8):
            raise ValueError(
                "moc_inference_compile_friendly requires grouped MoC with "
                "grouped_a=2, grouped_b=8."
            )
        if self.config.intermediate_size % 8 != 0:
            raise ValueError(
                f"intermediate_size={self.config.intermediate_size} must be divisible by 8"
            )
        x_2d, original_shape = self._flatten_decode_input(x)
        topk_idx, sparse_z = torch.ops.moc.fused_gate_top2_up_silu_v22.default(
            x_2d, self.gate_proj.weight, self.up_proj.weight,
        )
        y_2d = torch.ops.moc.selected_down_v22.default(
            sparse_z, topk_idx, self._down_weight_t,
        )
        return self._restore_output_shape(y_2d, original_shape)

    # a800_cuda128: Same v_14 Triton kernels wrapped with torch.library.triton_op
    # (torch 2.8+), which lets Inductor INLINE the kernel call into the
    # compiled graph instead of treating it as an opaque custom_op boundary.
    # Targets the 0.38 ms/tok MoC compile regression that's stuck since
    # full_model; if Inductor can inline, graph break count drops and per-layer
    # dispatch overhead shrinks.
    def moc_inference_torch28(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.config.grouped and self.config.grouped_a == 2 and self.config.grouped_b == 8):
            raise ValueError(
                "moc_inference_torch28 requires grouped MoC with grouped_a=2, grouped_b=8."
            )
        if self.config.intermediate_size % 8 != 0:
            raise ValueError(
                f"intermediate_size={self.config.intermediate_size} must be divisible by 8"
            )
        x_2d, original_shape = self._flatten_decode_input(x)
        topk_idx, sparse_z = torch.ops.moc.fused_gate_top2_up_silu_torch28.default(
            x_2d, self.gate_proj.weight, self.up_proj.weight,
        )
        y_2d = torch.ops.moc.selected_down_torch28.default(
            sparse_z, topk_idx, self._down_weight_t,
        )
        return self._restore_output_shape(y_2d, original_shape)

    def moc_inference_torch28_balanced(self, x: torch.Tensor) -> torch.Tensor:
        """MoC 2:8 torch.compile path with the the single-layer benchmark balanced blocks."""
        if not (self.config.grouped and self.config.grouped_a == 2 and self.config.grouped_b == 8):
            raise ValueError(
                "moc_inference_torch28_balanced requires grouped MoC with grouped_a=2, grouped_b=8."
            )
        if self.config.intermediate_size % 8 != 0:
            raise ValueError(
                f"intermediate_size={self.config.intermediate_size} must be divisible by 8"
            )
        x_2d, original_shape = self._flatten_decode_input(x)
        topk_idx, sparse_z = torch.ops.moc.fused_gate_top2_up_silu_balanced_torch28.default(
            x_2d, self.gate_proj.weight, self.up_proj.weight,
        )
        y_2d = torch.ops.moc.selected_down_balanced_torch28.default(
            sparse_z, topk_idx, self._down_weight_t,
        )
        return self._restore_output_shape(y_2d, original_shape)

    def forward(self, x: torch.Tensor, mode: str = "moc_inference") -> torch.Tensor:
        if mode == "dense_baseline":
            return self.dense_baseline(x)
        if mode == "masked_reference":
            return self.masked_reference(x)
        if mode == "moc_inference":
            return self.moc_inference(x)
        if mode == "moc_inference_bmm":
            return self.moc_inference_bmm(x)
        if mode == "moc_inference_down_triton":
            return self.moc_inference_down_triton(x)
        if mode == "moc_inference_updown_triton":
            return self.moc_inference_updown_triton(x)
        if mode == "moc_inference_fused_updown_triton":
            return self.moc_inference_fused_updown_triton(x)
        if mode == "moc_inference_raft_topk_fused_updown":
            return self.moc_inference_raft_topk_fused_updown(x)
        if mode == "moc_inference_optimized_global_cuda":
            return self.moc_inference_optimized_global_cuda(x)
        if mode == "moc_inference_optimized_global_native":
            return self.moc_inference_optimized_global_native(x)
        if mode == "moc_inference_optimized_global_after_gate_native":
            return self.moc_inference_optimized_global_after_gate_native(x)
        if mode == "moc_inference_fused_gate_top2of8_fused_updown":
            return self.moc_inference_fused_gate_top2of8_fused_updown(x)
        if mode == "moc_inference_fused_gate_top2of8_dense_updown":
            return self.moc_inference_fused_gate_top2of8_dense_updown(x)
        if mode == "moc_inference_fused_gate_top2of8_group8_updown":
            return self.moc_inference_fused_gate_top2of8_group8_updown(x)
        if mode == "moc_inference_fused_gate_top2of8_fused_gate_up_down":
            return self.moc_inference_fused_gate_top2of8_fused_gate_up_down(x)
        if mode == "moc_inference_fused_gate_top2of8_fused_gate_up_silu_down":
            return self.moc_inference_fused_gate_top2of8_fused_gate_up_silu_down(x)
        if mode == "moc_inference_fused_gate_top2of8_fused_gate_up_silu_down_splitk":
            return self.moc_inference_fused_gate_top2of8_fused_gate_up_silu_down_splitk(x)
        if mode == "moc_inference_fused_gate_top2of8_group_local_silu_down":
            return self.moc_inference_fused_gate_top2of8_group_local_silu_down(x)
        if mode == "moc_inference_compile_friendly":
            return self.moc_inference_compile_friendly(x)
        if mode == "moc_inference_torch28":
            return self.moc_inference_torch28(x)
        if mode == "moc_inference_torch28_balanced":
            return self.moc_inference_torch28_balanced(x)
        raise ValueError(f"unknown mode={mode}")

