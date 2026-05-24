"""Mixture-of-Channels SwiGLU FFN.

The public factory exposes three variants:

  dense             - standard LLaMA-style SwiGLU FFN.
  moc               - MoC training path with fused sparse helper kernels.
  moc_gcp           - checkpointed MoC with fused sparse helper kernels.

MoC saves activation memory by keeping only the selected channel activations
needed by backward instead of the full intermediate FFN state.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_triton_training_kernels():
    try:
        from .training import triton_sparse_aux
    except Exception as exc:  # pragma: no cover - depends on CUDA/Triton install.
        raise RuntimeError(
            "MoC fused helper kernels require the optional Triton training module."
        ) from exc
    return triton_sparse_aux


def _pack_topk_idx(topk_idx: torch.Tensor, dtype, max_allowed_idx: int | None = None):
    if dtype is None:
        return topk_idx
    info = torch.iinfo(dtype)
    if max_allowed_idx is not None:
        min_idx = 0
        max_idx = max_allowed_idx
    else:
        max_idx = int(topk_idx.max().item())
        min_idx = int(topk_idx.min().item())
    if min_idx < info.min or max_idx > info.max:
        raise ValueError(
            f"topk_idx range [{min_idx}, {max_idx}] cannot be stored as {dtype}"
        )
    return topk_idx.to(dtype)


class StandardSwiGLUFFN(nn.Module):
    """Standard LLaMA-style SwiGLU FFN."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _MoCFunction(torch.autograd.Function):
    """MoC autograd path that keeps selected activations for backward.

    """

    @staticmethod
    def forward(ctx, x, gate_w, up_w, down_w, k, index_save_dtype=None, use_triton_aux=False):
        inter = gate_w.shape[0]
        if use_triton_aux:
            if index_save_dtype != torch.int16:
                raise ValueError("MoC fused helper kernels require compact saved indices.")
            if not x.is_cuda:
                raise ValueError("MoC fused helper kernels require CUDA tensors.")
            triton_kernels = _load_triton_training_kernels()
        else:
            triton_kernels = None

        g = x @ gate_w.t()
        # MoC only needs the selected channel set; the selected channels do not
        # need to be ordered by gate value.
        topk_vals, topk_idx = torch.topk(g, k, dim=-1, largest=True, sorted=False)
        del g
        u = x @ up_w.t()
        saved_topk_idx = _pack_topk_idx(topk_idx, index_save_dtype, inter - 1)
        if use_triton_aux:
            u_sparse, s_sparse, z_sparse, z_full = triton_kernels.sparse_forward_aux(
                topk_vals,
                saved_topk_idx,
                u,
                inter,
            )
        else:
            u_sparse = torch.gather(u, -1, topk_idx)
            s_sparse = F.silu(topk_vals)
            z_sparse = s_sparse * u_sparse

            z_full = torch.zeros(*x.shape[:-1], inter, device=x.device, dtype=z_sparse.dtype)
            z_full.scatter_(-1, topk_idx, z_sparse)
        del u
        output = z_full @ down_w.t()
        del z_full

        ctx.save_for_backward(
            x, saved_topk_idx, topk_vals, u_sparse, s_sparse, z_sparse,
            gate_w, up_w, down_w,
        )
        ctx.k = k
        ctx.inter = inter
        ctx.use_triton_aux = use_triton_aux
        return output

    @staticmethod
    def backward(ctx, grad_out):
        (x, topk_idx, g_sparse, u_sparse, s_sparse, z_sparse,
         gate_w, up_w, down_w) = ctx.saved_tensors
        use_triton_aux = ctx.use_triton_aux
        if use_triton_aux:
            triton_kernels = _load_triton_training_kernels()
            topk_idx64 = None
        else:
            triton_kernels = None
            topk_idx64 = topk_idx.to(torch.int64)
        k, inter = ctx.k, ctx.inter

        if use_triton_aux:
            z_full = triton_kernels.scatter_sparse_to_dense(z_sparse, topk_idx, inter)
        else:
            z_full = torch.zeros(*x.shape[:-1], inter, device=x.device, dtype=z_sparse.dtype)
            z_full.scatter_(-1, topk_idx64, z_sparse)
        z_2d = z_full.reshape(-1, inter)
        go_2d = grad_out.reshape(-1, grad_out.shape[-1])
        grad_down_w = go_2d.t() @ z_2d
        del z_2d, z_full

        grad_z_full = grad_out @ down_w
        x_2d = x.reshape(-1, x.shape[-1])
        if use_triton_aux:
            grad_g, grad_u = triton_kernels.sparse_backward_aux(
                topk_idx,
                g_sparse,
                u_sparse,
                s_sparse,
                grad_z_full,
                inter,
            )
            del grad_z_full
        else:
            grad_z_sparse = torch.gather(grad_z_full, -1, topk_idx64)
            del grad_z_full

            grad_s_sparse = u_sparse * grad_z_sparse
            grad_u_sparse = s_sparse * grad_z_sparse
            del grad_z_sparse

            sig = torch.sigmoid(g_sparse)
            silu_deriv = sig * (1.0 + g_sparse * (1.0 - sig))
            del sig
            grad_g_sparse = grad_s_sparse * silu_deriv
            del grad_s_sparse, silu_deriv

            grad_g = x.new_zeros(*x.shape[:-1], inter)
            grad_g.scatter_(-1, topk_idx64, grad_g_sparse)
            del grad_g_sparse
            grad_u = x.new_zeros(*x.shape[:-1], inter)
            grad_u.scatter_(-1, topk_idx64, grad_u_sparse)
            del grad_u_sparse

        grad_gate_w = grad_g.reshape(-1, inter).t() @ x_2d
        grad_x_g = grad_g @ gate_w
        del grad_g

        grad_up_w = grad_u.reshape(-1, inter).t() @ x_2d
        grad_x_g.reshape(-1, grad_x_g.shape[-1]).addmm_(
            grad_u.reshape(-1, inter),
            up_w,
        )
        del grad_u

        return grad_x_g, grad_gate_w, grad_up_w, grad_down_w, None, None, None


class _MoCGCPFunction(torch.autograd.Function):
    """MoC autograd path with selected-activation checkpointing.

    """

    @staticmethod
    def forward(ctx, x, gate_w, up_w, down_w, k, index_save_dtype=None, use_triton_aux=False):
        inter = gate_w.shape[0]
        if use_triton_aux:
            if index_save_dtype != torch.int16:
                raise ValueError("MoC+GCP fused helper kernels require compact saved indices.")
            if not x.is_cuda:
                raise ValueError("MoC+GCP fused helper kernels require CUDA tensors.")
            triton_kernels = _load_triton_training_kernels()
        else:
            triton_kernels = None

        g = x @ gate_w.t()
        # MoC only needs the selected channel set; the selected channels do not
        # need to be ordered by gate value.
        topk_vals, topk_idx = torch.topk(g, k, dim=-1, largest=True, sorted=False)
        del g
        u = x @ up_w.t()
        saved_topk_idx = _pack_topk_idx(topk_idx, index_save_dtype, inter - 1)
        if use_triton_aux:
            u_sparse, _s_sparse, _z_sparse, z_full = triton_kernels.sparse_forward_aux(
                topk_vals,
                saved_topk_idx,
                u,
                inter,
            )
            del _s_sparse, _z_sparse
        else:
            u_sparse = torch.gather(u, -1, topk_idx)
            s_sparse = F.silu(topk_vals)
            z_sparse = s_sparse * u_sparse
            del s_sparse

            z_full = torch.zeros(*x.shape[:-1], inter, device=x.device, dtype=z_sparse.dtype)
            z_full.scatter_(-1, topk_idx, z_sparse)
            del z_sparse
        del u
        output = z_full @ down_w.t()
        del z_full

        ctx.save_for_backward(x, saved_topk_idx, topk_vals, u_sparse, gate_w, up_w, down_w)
        ctx.k = k
        ctx.inter = inter
        ctx.use_triton_aux = use_triton_aux
        return output

    @staticmethod
    def backward(ctx, grad_out):
        x, topk_idx, g_sparse, u_sparse, gate_w, up_w, down_w = ctx.saved_tensors
        use_triton_aux = ctx.use_triton_aux
        if use_triton_aux:
            triton_kernels = _load_triton_training_kernels()
            topk_idx64 = None
        else:
            triton_kernels = None
            topk_idx64 = topk_idx.to(torch.int64)
        k, inter = ctx.k, ctx.inter

        s_sparse = F.silu(g_sparse)
        z_sparse = s_sparse * u_sparse

        if use_triton_aux:
            z_full = triton_kernels.scatter_sparse_to_dense(z_sparse, topk_idx, inter)
        else:
            z_full = torch.zeros(*x.shape[:-1], inter, device=x.device, dtype=z_sparse.dtype)
            z_full.scatter_(-1, topk_idx64, z_sparse)
        del z_sparse
        z_2d = z_full.reshape(-1, inter)
        go_2d = grad_out.reshape(-1, grad_out.shape[-1])
        grad_down_w = go_2d.t() @ z_2d
        del z_2d, z_full

        grad_z_full = grad_out @ down_w
        if use_triton_aux:
            grad_g, grad_u = triton_kernels.sparse_backward_aux(
                topk_idx,
                g_sparse,
                u_sparse,
                s_sparse,
                grad_z_full,
                inter,
            )
            del s_sparse, grad_z_full
        else:
            grad_z_sparse = torch.gather(grad_z_full, -1, topk_idx64)
            del grad_z_full

            grad_s_sparse = u_sparse * grad_z_sparse
            grad_u_sparse = s_sparse * grad_z_sparse
            del grad_z_sparse, s_sparse

            sig = torch.sigmoid(g_sparse)
            silu_deriv = sig * (1.0 + g_sparse * (1.0 - sig))
            del sig
            grad_g_sparse = grad_s_sparse * silu_deriv
            del grad_s_sparse, silu_deriv

            grad_g = x.new_zeros(*x.shape[:-1], inter)
            grad_g.scatter_(-1, topk_idx64, grad_g_sparse)
            del grad_g_sparse

            grad_u = x.new_zeros(*x.shape[:-1], inter)
            grad_u.scatter_(-1, topk_idx64, grad_u_sparse)
            del grad_u_sparse

        x_2d = x.reshape(-1, x.shape[-1])
        grad_gate_w = grad_g.reshape(-1, inter).t() @ x_2d
        grad_x_g = grad_g @ gate_w
        del grad_g

        grad_up_w = grad_u.reshape(-1, inter).t() @ x_2d
        grad_x_g.reshape(-1, grad_x_g.shape[-1]).addmm_(
            grad_u.reshape(-1, inter),
            up_w,
        )
        del grad_u

        return grad_x_g, grad_gate_w, grad_up_w, grad_down_w, None, None, None


class MoCSwiGLUFFN(nn.Module):
    """MoC SwiGLU FFN with selectable backward strategy.

    mode='moc'     : save sparse selected activations.
    mode='moc_gcp' : recompute selected activations in backward.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, k: int,
                 mode: str = "moc", index_save_dtype=None):
        super().__init__()
        if mode not in ("moc", "moc_gcp", "moc_triton_aux", "moc_gcp_triton_aux"):
            raise ValueError(
                f"Unknown mode '{mode}'. Use 'moc', 'moc_gcp', "
                "'moc_triton_aux', or 'moc_gcp_triton_aux'."
            )
        self.k = k
        self.mode = mode
        self.index_save_dtype = index_save_dtype
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        if self.mode in ("moc_gcp", "moc_gcp_triton_aux"):
            return _MoCGCPFunction.apply(
                x,
                self.gate_proj.weight,
                self.up_proj.weight,
                self.down_proj.weight,
                self.k,
                self.index_save_dtype,
                self.mode == "moc_gcp_triton_aux",
            )
        return _MoCFunction.apply(
            x,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
            self.k,
            self.index_save_dtype,
            self.mode == "moc_triton_aux",
        )


def build_ffn(hidden_size: int, intermediate_size: int,
              ffn_type: str = "dense", k=None):
    """Factory for Dense FFN, MoC, and MoC+GCP variants."""
    if ffn_type == "dense":
        return StandardSwiGLUFFN(hidden_size, intermediate_size)
    if ffn_type in ("moc", "moc_gcp"):
        if k is None:
            raise ValueError("MoC FFN requires `k`")
        mode = {
            "moc": "moc_triton_aux",
            "moc_gcp": "moc_gcp_triton_aux",
        }[ffn_type]
        return MoCSwiGLUFFN(
            hidden_size,
            intermediate_size,
            k,
            mode=mode,
            index_save_dtype=torch.int16,
        )
    raise ValueError(f"Unknown ffn_type '{ffn_type}'")

