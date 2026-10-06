"""Store selected activations and replay only the down-projection backward."""

from contextlib import contextmanager
import types

import torch
from torch.autograd.function import once_differentiable
from projection_replay import projection_replay, validate_down


def scatter(value, indices, width):
    if value.is_cuda:
        from compact_kernels import scatter as implementation
        return implementation(value, indices, width)
    return value.new_zeros((*value.shape[:-1], width)).scatter_(-1, indices.long(), value)


def gather(value, indices):
    if value.is_cuda:
        from compact_kernels import gather as implementation
        return implementation(value, indices)
    return value.gather(-1, indices.long())


def _prepare(ctx, x, module, params):
    expected = tuple(p for p in module.parameters() if p.requires_grad)
    if len(expected) != len(params) or any(a is not b for a, b in zip(expected, params)):
        raise ValueError("Trainable projection parameters changed")
    ctx.module = module
    ctx.params = expected
    ctx.training = tuple((m, m.training) for m in module.modules())
    ctx.device = x.device
    ctx.amp = torch.is_autocast_enabled(x.device.type)
    ctx.amp_dtype = torch.get_autocast_dtype(x.device.type)
    ctx.cpu_rng = torch.get_rng_state()
    ctx.cuda_rng = torch.cuda.get_rng_state(x.device) if x.is_cuda else None
    ctx.versions = tuple((p, p._version) for p in module.parameters())
    ctx.set_materialize_grads(False)


@contextmanager
def _replay(ctx):
    if any(m.training != state for m, state in ctx.training):
        raise RuntimeError("Projection mode changed before backward")
    if any(p._version != version for p, version in ctx.versions):
        raise RuntimeError("Projection weight modified before backward")
    devices = [ctx.device.index] if ctx.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(ctx.cpu_rng)
        if ctx.cuda_rng is not None:
            torch.cuda.set_rng_state(ctx.cuda_rng, ctx.device)
        with torch.enable_grad(), torch.autocast(ctx.device.type, enabled=ctx.amp, dtype=ctx.amp_dtype):
            yield


def _gradients(output, x, params, grad_output):
    targets = (x,) + params if x.requires_grad else params
    if not targets:
        return None, ()
    values = torch.autograd.grad(output, targets, grad_output, allow_unused=False)
    return (values[0], values[1:]) if x.requires_grad else (None, values)


class _SelectedEdge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, selected, buffer, indices):
        assert selected.dtype == torch.bfloat16 and indices.dtype == torch.int16
        ctx.save_for_backward(indices)
        ctx.set_materialize_grads(False)
        return buffer

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        if grad is None:
            return None, None, None
        indices, = ctx.saved_tensors
        return gather(grad, indices).to(torch.bfloat16), None, None


def expand_edges(selected, indices, width):
    # Release each full gradient independently; accumulate only compact BF16 values.
    with torch.no_grad():
        dense = scatter(selected, indices, width)
        expanded = dense.float()
    return (_SelectedEdge.apply(selected, dense, indices),
            _SelectedEdge.apply(selected, expanded, indices))


class _SelectedDown(torch.autograd.Function):
    @staticmethod
    def forward(ctx, selected, indices, module, width, *params):
        validate_down(module)
        _prepare(ctx, selected, module, params)
        ctx.width = width
        output = module(scatter(selected, indices, width))
        ctx.save_for_backward(selected, indices)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if grad_output is None:
            return (None,) * (4 + len(ctx.params))
        saved, indices = ctx.saved_tensors
        with _replay(ctx):
            selected = saved.detach().requires_grad_(ctx.needs_input_grad[0])
            dense, expanded = expand_edges(selected, indices, ctx.width)
            output = projection_replay(ctx.module, dense, expanded)
            del dense, expanded
            grad_selected, gradients = _gradients(output, selected, ctx.params, grad_output)
        return grad_selected, None, None, None, *gradients


def _params(module):
    return tuple(p for p in module.parameters() if p.requires_grad)


class _CompactGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, indices):
        ctx.width = value.shape[-1]
        ctx.save_for_backward(indices)
        return gather(value, indices)

    @staticmethod
    def backward(ctx, grad):
        indices, = ctx.saved_tensors
        return scatter(grad, indices, ctx.width), None


def selected_forward(module, x):
    if module.selector != "raw_gate" or module.current_k != module.target_k:
        raise ValueError("Only fixed raw-gate selection is supported")
    # Preserve the native gate/up gradient accumulation graph; gather only needs indices.
    full_gate = module.gate_proj(x)
    full_up = module.up_proj(x)
    with torch.no_grad():
        indices = torch.topk(full_gate, module.target_k, dim=-1, largest=True, sorted=False).indices.to(torch.int16)
    gate = _CompactGather.apply(full_gate, indices)
    up = _CompactGather.apply(full_up, indices)
    product = module.act_fn(gate) * up
    return _SelectedDown.apply(product, indices, module.down_proj, module.intermediate_size,
                               *_params(module.down_proj))


def install(module):
    if hasattr(module, "_reference_forward"):
        raise ValueError("Already installed")
    if not all(hasattr(module, name) for name in
               ("selector", "current_k", "target_k", "intermediate_size", "act_fn")):
        raise TypeError("Expected a fixed-K MoC MLP")
    for name in ("gate_proj", "up_proj", "down_proj"):
        projection = getattr(module, name)
        if hasattr(projection, "base_layer") and any(p.requires_grad for p in projection.base_layer.parameters()):
            raise ValueError("This path requires frozen base weights")
    module._reference_forward = module.forward
    module.forward = types.MethodType(selected_forward, module)
    return module


def uninstall(module):
    module.forward = module._reference_forward
    del module._reference_forward
