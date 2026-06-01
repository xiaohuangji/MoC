from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import optimized_global_moc_ops  # noqa: F401
from .triton_fused_ffn_kernels import (
    fused_gate_top2of8_selected_up_silu_into,
    selected_down_from_sparse_z_into,
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
    - moc_inference_optimized_global_after_gate_native: optimized global Top-K
      MoC path used by the decode benchmark
    - moc_inference_grouped_top2of8: optimized grouped top-2-of-8 MoC path
      used by the decode benchmark
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
        # Frozen flag prevents in-forward Python-state mutation under
        # torch.compile. Set to True via freeze_for_compile() before compile.
        self._compile_frozen: bool = False
        # Grouped index base cache. Avoids per-call torch.arange in
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

    # Fused gate+top2+selected-up+SiLU defaults for grouped MoC.
    GUSILU_BLOCK_B_DEFAULT = 16
    GUSILU_BLOCK_G_DEFAULT = 16
    GUSILU_BLOCK_H_DEFAULT = 64
    GROUPED_DOWN_BLOCK_K_DEFAULT = 128
    GROUPED_DOWN_BLOCK_H_DEFAULT = 16

    def make_moc2_8_graph_runner(
        self,
        batch_size: int,
        gate_up_BLOCK_B: int | None = None,
        gate_up_BLOCK_G: int | None = None,
        gate_up_BLOCK_H: int | None = None,
        down_block_k: int | None = None,
        down_block_h: int | None = None,
    ):
        """Capture the fused gate+up+SiLU plus selected-down
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
                "graph runner requires grouped MoC with grouped_a=2, grouped_b=8."
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
        dbk = down_block_k or self.GROUPED_DOWN_BLOCK_K_DEFAULT
        dbh = down_block_h or self.GROUPED_DOWN_BLOCK_H_DEFAULT

        x_buf = torch.empty(batch_size, H, device=device, dtype=dtype)
        topk_idx_buf = torch.empty(batch_size, K, device=device, dtype=torch.int64)
        sparse_z_buf = torch.empty(batch_size, K, device=device, dtype=dtype)
        y_buf = torch.empty(batch_size, H, device=device, dtype=dtype)

        def _run_moc2_8_direct() -> None:
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
                _run_moc2_8_direct()
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_moc2_8_direct()

        def runner(x_in: torch.Tensor) -> torch.Tensor:
            x_2d, original_shape = self._flatten_decode_input(x_in)
            if x_2d.shape[0] != batch_size:
                raise ValueError(
                    f"MoC 2:8 graph runner built for batch_size={batch_size}, got {x_2d.shape[0]}"
                )
            x_buf.copy_(x_2d)
            graph.replay()
            return self._restore_output_shape(y_buf, original_shape)

        return runner

    def moc_inference_grouped_top2of8(self, x: torch.Tensor) -> torch.Tensor:
        """MoC 2:8 torch.compile path with the final grouped kernel blocks."""
        if not (self.config.grouped and self.config.grouped_a == 2 and self.config.grouped_b == 8):
            raise ValueError(
                "moc_inference_grouped_top2of8 requires grouped MoC with grouped_a=2, grouped_b=8."
            )
        if self.config.intermediate_size % 8 != 0:
            raise ValueError(
                f"intermediate_size={self.config.intermediate_size} must be divisible by 8"
            )
        x_2d, original_shape = self._flatten_decode_input(x)
        topk_idx, sparse_z = torch.ops.moc.fused_grouped_top2_up_silu.default(
            x_2d, self.gate_proj.weight, self.up_proj.weight,
        )
        y_2d = torch.ops.moc.selected_grouped_down.default(
            sparse_z, topk_idx, self._down_weight_t,
        )
        return self._restore_output_shape(y_2d, original_shape)

    def forward(self, x: torch.Tensor, mode: str = "auto") -> torch.Tensor:
        if mode == "auto":
            mode = (
                "moc_inference_grouped_top2of8"
                if self.config.grouped
                else "moc_inference_optimized_global_after_gate_native"
            )
        if mode == "dense_baseline":
            return self.dense_baseline(x)
        if mode == "masked_reference":
            return self.masked_reference(x)
        if mode == "moc_inference_optimized_global_after_gate_native":
            return self.moc_inference_optimized_global_after_gate_native(x)
        if mode == "moc_inference_grouped_top2of8":
            return self.moc_inference_grouped_top2of8(x)
        raise ValueError(f"unknown mode={mode}")
