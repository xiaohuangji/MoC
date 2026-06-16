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


def _to_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype == dtype:
        return tensor
    return tensor.to(dtype)


def _amp_matmul(a: torch.Tensor, b: torch.Tensor, compute_dtype: torch.dtype | None,
                out_dtype: torch.dtype | None = None) -> torch.Tensor:
    if compute_dtype in (torch.bfloat16, torch.float16) and a.is_cuda and b.is_cuda:
        with torch.autocast("cuda", dtype=compute_dtype):
            out = a @ b
    else:
        out = a @ b
    if out_dtype is not None:
        return _to_dtype(out, out_dtype)
    return out


def _moc_2_8_effective_k(intermediate_size: int) -> int:
    """Return the structure-determined K' for MoC 2:8 selection."""
    full_groups, tail = divmod(intermediate_size, 8)
    return full_groups * 2 + min(2, tail)


def _select_moc_2_8(g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-2 per consecutive 8-channel group from pre-SiLU gates.

    The final tail group, when present, contributes top-min(2, tail) channels.
    Returned indices are flattened channel indices and selected entries are not
    globally sorted.
    """
    intermediate_size = g.shape[-1]
    full_groups, tail = divmod(intermediate_size, 8)
    prefix_shape = g.shape[:-1]
    value_parts = []
    index_parts = []

    if full_groups:
        grouped = g[..., : full_groups * 8].reshape(*prefix_shape, full_groups, 8)
        values, local_idx = torch.topk(grouped, 2, dim=-1, largest=True, sorted=False)
        bases = torch.arange(
            full_groups,
            device=g.device,
            dtype=local_idx.dtype,
        ) * 8
        view_shape = (1,) * len(prefix_shape) + (full_groups, 1)
        flat_idx = local_idx + bases.view(view_shape)
        value_parts.append(values.reshape(*prefix_shape, full_groups * 2))
        index_parts.append(flat_idx.reshape(*prefix_shape, full_groups * 2))

    if tail:
        tail_k = min(2, tail)
        values, local_idx = torch.topk(
            g[..., full_groups * 8:],
            tail_k,
            dim=-1,
            largest=True,
            sorted=False,
        )
        value_parts.append(values)
        index_parts.append(local_idx + full_groups * 8)

    if len(value_parts) == 1:
        return value_parts[0], index_parts[0]
    return torch.cat(value_parts, dim=-1), torch.cat(index_parts, dim=-1)


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
        # Backward GEMMs run in the saved-activation compute dtype (the autocast
        # dtype used during forward); only the resulting weight gradients are
        # cast to the parameter dtype. This matches the gradient precision the
        # dense baseline gets from nn.Linear under autocast.
        compute_dtype = (
            g_sparse.dtype if g_sparse.dtype in (torch.bfloat16, torch.float16) else None
        )
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
        weight_dtype = down_w.dtype
        z_2d = z_full.reshape(-1, inter)
        go_2d = grad_out.reshape(-1, grad_out.shape[-1])
        grad_down_w = _amp_matmul(go_2d.t(), z_2d, compute_dtype, weight_dtype)
        del z_2d, z_full

        grad_z_full = _amp_matmul(grad_out, down_w, compute_dtype, g_sparse.dtype)
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

            grad_g = torch.zeros(*x.shape[:-1], inter, device=x.device, dtype=grad_g_sparse.dtype)
            grad_g.scatter_(-1, topk_idx64, grad_g_sparse)
            del grad_g_sparse
            grad_u = torch.zeros(*x.shape[:-1], inter, device=x.device, dtype=grad_u_sparse.dtype)
            grad_u.scatter_(-1, topk_idx64, grad_u_sparse)
            del grad_u_sparse

        grad_gate_w = _amp_matmul(
            grad_g.reshape(-1, inter).t(),
            x_2d,
            compute_dtype,
            gate_w.dtype,
        )
        grad_x_g = _amp_matmul(grad_g, gate_w, compute_dtype, x.dtype)
        del grad_g

        grad_up_w = _amp_matmul(
            grad_u.reshape(-1, inter).t(),
            x_2d,
            compute_dtype,
            up_w.dtype,
        )
        grad_x_u = _amp_matmul(grad_u, up_w, compute_dtype, x.dtype)
        grad_x_g = _to_dtype(grad_x_g + grad_x_u, x.dtype)
        del grad_x_u
        del grad_u

        return _to_dtype(grad_x_g, x.dtype), grad_gate_w, grad_up_w, grad_down_w, None, None, None


class _MoC28Function(torch.autograd.Function):
    """MoC with structured 2:8 selection; reuses the MoC backward path."""

    @staticmethod
    def forward(ctx, x, gate_w, up_w, down_w, k, index_save_dtype=None, use_triton_aux=False):
        inter = gate_w.shape[0]
        if use_triton_aux:
            if index_save_dtype != torch.int16:
                raise ValueError("MoC 2:8 fused helper kernels require compact saved indices.")
            if not x.is_cuda:
                raise ValueError("MoC 2:8 fused helper kernels require CUDA tensors.")
            triton_kernels = _load_triton_training_kernels()
        else:
            triton_kernels = None

        g = x @ gate_w.t()
        topk_vals, topk_idx = _select_moc_2_8(g)
        k = topk_idx.shape[-1]
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

    backward = staticmethod(_MoCFunction.backward)


class _MoCPostSiluFunction(torch.autograd.Function):
    """MoC with post-SiLU top-K selection; reuses the MoC backward path.

    Indices are the top-K largest |SiLU(g)| (magnitude of the activated output);
    the values handed to the shared sparse machinery are the pre-SiLU gate
    values at those indices.
    """

    @staticmethod
    def forward(ctx, x, gate_w, up_w, down_w, k, index_save_dtype=None,
                use_triton_aux=False):
        inter = gate_w.shape[0]
        if use_triton_aux:
            if index_save_dtype != torch.int16:
                raise ValueError("MoC post-SiLU fused helper kernels require compact saved indices.")
            if not x.is_cuda:
                raise ValueError("MoC post-SiLU fused helper kernels require CUDA tensors.")
            triton_kernels = _load_triton_training_kernels()
        else:
            triton_kernels = None

        g = x @ gate_w.t()
        score = F.silu(g).abs()
        _, topk_idx = torch.topk(score, k, dim=-1, largest=True, sorted=False)
        del score
        topk_vals = torch.gather(g, -1, topk_idx)
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

    backward = staticmethod(_MoCFunction.backward)


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
        weight_dtype = down_w.dtype
        z_2d = _to_dtype(z_full.reshape(-1, inter), weight_dtype)
        go_2d = _to_dtype(grad_out.reshape(-1, grad_out.shape[-1]), weight_dtype)
        grad_down_w = go_2d.t() @ z_2d
        del z_2d, z_full

        grad_z_full = _to_dtype(grad_out, weight_dtype) @ down_w
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

        x_2d = _to_dtype(x.reshape(-1, x.shape[-1]), gate_w.dtype)
        grad_g = _to_dtype(grad_g, gate_w.dtype)
        grad_gate_w = grad_g.reshape(-1, inter).t() @ x_2d
        grad_x_g = grad_g @ gate_w
        del grad_g

        grad_u = _to_dtype(grad_u, up_w.dtype)
        grad_up_w = grad_u.reshape(-1, inter).t() @ x_2d
        grad_x_g.reshape(-1, grad_x_g.shape[-1]).addmm_(
            grad_u.reshape(-1, inter),
            up_w,
        )
        del grad_u

        return _to_dtype(grad_x_g, x.dtype), grad_gate_w, grad_up_w, grad_down_w, None, None, None


class MoCSwiGLUFFN(nn.Module):
    """MoC SwiGLU FFN with selectable backward strategy.

    mode='moc'     : save sparse selected activations.
    mode='moc_gcp' : recompute selected activations in backward.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, k: int | None = None,
                 mode: str = "moc", index_save_dtype=None):
        super().__init__()
        if mode not in (
            "moc",
            "moc_gcp",
            "moc_triton_aux",
            "moc_gcp_triton_aux",
            "moc_2_8",
            "moc_2_8_triton_aux",
            "moc_post_silu_abs",
            "moc_post_silu_abs_triton_aux",
        ):
            raise ValueError(
                f"Unknown mode '{mode}'. Use 'moc', 'moc_gcp', "
                "'moc_triton_aux', 'moc_gcp_triton_aux', "
                "'moc_2_8', 'moc_2_8_triton_aux', "
                "'moc_post_silu_abs', or 'moc_post_silu_abs_triton_aux'."
            )
        self.mode = mode
        self.requested_k = k
        if mode.startswith("moc_2_8"):
            # Structured 2:8 selection determines K' from the layout; any
            # externally supplied k is recorded but ignored.
            self.k_effective = _moc_2_8_effective_k(intermediate_size)
        else:
            if k is None:
                raise ValueError("MoC FFN requires `k` outside moc_2_8 modes.")
            self.k_effective = k
        self.k = self.k_effective
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
        if self.mode in ("moc_2_8", "moc_2_8_triton_aux"):
            return _MoC28Function.apply(
                x,
                self.gate_proj.weight,
                self.up_proj.weight,
                self.down_proj.weight,
                self.k,
                self.index_save_dtype,
                self.mode == "moc_2_8_triton_aux",
            )
        if self.mode in ("moc_post_silu_abs", "moc_post_silu_abs_triton_aux"):
            return _MoCPostSiluFunction.apply(
                x,
                self.gate_proj.weight,
                self.up_proj.weight,
                self.down_proj.weight,
                self.k,
                self.index_save_dtype,
                self.mode == "moc_post_silu_abs_triton_aux",
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
    """Factory for Dense FFN, MoC, MoC+GCP, and MoC 2:8 variants.

    For ffn_type "moc_2_8" the external `k` argument is ignored and the module
    exposes the structure-determined K' as `k_effective`.
    """
    if ffn_type == "dense":
        return StandardSwiGLUFFN(hidden_size, intermediate_size)
    if ffn_type in ("moc", "moc_gcp", "moc_2_8", "moc_post_silu_abs"):
        if k is None and ffn_type != "moc_2_8":
            raise ValueError("MoC FFN requires `k`")
        mode = {
            "moc": "moc_triton_aux",
            "moc_gcp": "moc_gcp_triton_aux",
            "moc_2_8": "moc_2_8_triton_aux",
            "moc_post_silu_abs": "moc_post_silu_abs_triton_aux",
        }[ffn_type]
        return MoCSwiGLUFFN(
            hidden_size,
            intermediate_size,
            k,
            mode=mode,
            index_save_dtype=torch.int16,
        )
    raise ValueError(f"Unknown ffn_type '{ffn_type}'")

