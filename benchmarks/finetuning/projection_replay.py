"""Replay only trainable factors; a frozen affine map needs no forward in VJP."""
import torch
from torch import nn


class _FrozenLinearVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        if weight.requires_grad or weight.ndim != 2:
            raise ValueError('Only frozen non-quantized linear weights are supported')
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        ctx.input_dtype = x.dtype
        ctx.save_for_backward(weight.to(dtype))
        return torch.zeros((*x.shape[:-1], weight.shape[0]), device=x.device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad):
        weight, = ctx.saved_tensors
        with torch.autocast(grad.device.type, enabled=False):
            result = grad.reshape(-1, grad.shape[-1]).mm(weight)
        return result.reshape(*grad.shape[:-1], weight.shape[1]).to(ctx.input_dtype), None


def validate_down(module):
    assert isinstance(module.base_layer, nn.Linear) and module.base_layer.bias is None
    assert not module.base_layer.weight.requires_grad
    assert module.base_layer.weight.dtype == torch.bfloat16
    assert not module.disable_adapters and not module.merged and not module.lora_variant
    assert getattr(module, 'cast_input_dtype_enabled', True)
    assert len(module.active_adapters) == 1
    name = module.active_adapters[0]
    a, b = module.lora_A[name], module.lora_B[name]
    assert a.weight.dtype == b.weight.dtype == torch.float32
    assert a.out_features == b.in_features == 128
    assert a.bias is b.bias is None
    return name


def projection_replay(module, dense, expanded):
    """VJP surrogate using separate native base and LoRA input-gradient edges."""
    name = validate_down(module)
    result = _FrozenLinearVJP.apply(dense, module.base_layer.weight)
    result_dtype = result.dtype
    a, b = module.lora_A[name], module.lora_B[name]
    value = module.lora_dropout[name](expanded)
    result = result + b(a(value)) * module.scaling[name]
    return result.to(result_dtype)
