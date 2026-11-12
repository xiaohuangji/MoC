"""Shared raw-gate routing and compact index movement for MoC training.

Indices must come from raw_topk (unique, in range). Copy helpers deliberately
do not validate tensor values on the host or synchronize CUDA in the hot path.
"""
import torch


def raw_topk(gate, k):
    if not 0 < k <= gate.shape[-1]:
        raise ValueError("K must be in [1, intermediate_size]")
    return torch.topk(gate, k, dim=-1, largest=True, sorted=False)


def pack_indices(indices, dtype, max_allowed_idx=None):
    if dtype is None:
        return indices
    info = torch.iinfo(dtype)
    if max_allowed_idx is None:
        lo, hi = int(indices.min().item()), int(indices.max().item())
    else:
        lo, hi = 0, max_allowed_idx
    if lo < info.min or hi > info.max:
        raise ValueError(f"topk_idx range [{lo}, {hi}] cannot be stored as {dtype}")
    return indices.to(dtype)


def _check(value, indices, width, compact):
    if value.device != indices.device or value.shape[:-1] != indices.shape[:-1]:
        raise ValueError("Values and indices must have matching rows and device")
    if indices.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("Indices must be signed integers")
    if not 0 < indices.shape[-1] <= width:
        raise ValueError("Invalid selected width")
    if compact and value.shape != indices.shape:
        raise ValueError("Selected values and indices must have the same shape")
    if width - 1 > torch.iinfo(indices.dtype).max:
        raise ValueError("Channel width cannot be represented by the index dtype")


def gather(value, indices):
    """Copy selected values; use gather_selected for differentiable calls."""
    _check(value, indices, value.shape[-1], False)
    if value.is_cuda and indices.dtype == torch.int16 and indices.numel():
        from .triton_sparse_aux import gather_dense_to_sparse
        return gather_dense_to_sparse(value.contiguous(), indices.contiguous())
    return value.gather(-1, indices.long())


def scatter(value, indices, width):
    """Zero-fill unselected channels; indices must be unique within each row."""
    _check(value, indices, width, True)
    if value.is_cuda and indices.dtype == torch.int16 and indices.numel():
        from .triton_sparse_aux import scatter_sparse_to_dense
        return scatter_sparse_to_dense(value.contiguous(), indices.contiguous(), width)
    return value.new_zeros((*value.shape[:-1], width)).scatter_(-1, indices.long(), value)


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, indices):
        ctx.width = value.shape[-1]
        ctx.save_for_backward(indices)
        return gather(value, indices)

    @staticmethod
    def backward(ctx, grad):
        indices, = ctx.saved_tensors
        return scatter(grad, indices, ctx.width), None


class _Scatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, indices, width):
        ctx.save_for_backward(indices)
        return scatter(value, indices, width)

    @staticmethod
    def backward(ctx, grad):
        indices, = ctx.saved_tensors
        return gather(grad, indices), None, None


gather_selected = _Gather.apply
scatter_selected = _Scatter.apply


def selected_product(gate, up, k, activation):
    if gate.shape != up.shape:
        raise ValueError("Gate and up projections must have the same shape")
    with torch.no_grad():
        indices = pack_indices(raw_topk(gate, k).indices, torch.int16, gate.shape[-1] - 1)
    product = activation(gather_selected(gate, indices)) * gather_selected(up, indices)
    return product, indices
