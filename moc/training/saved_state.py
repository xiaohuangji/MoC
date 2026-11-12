"""Lossless saved-input compression for the fixed-K MoC down projection.

The native PEFT forward/backward and dropout remain unchanged. Only known
full-width differentiable inputs inside this projection are packed. Weights,
masks, low-rank states, and outputs are not compressed.
"""
import types
import weakref

import torch
from moc.training.selection import gather, scatter, selected_product, scatter_selected


class SavedInputCodec:
    def __init__(self, indices, width, parameters, stats):
        self.indices = indices.reshape(-1, indices.shape[-1])
        self.width = width
        self.parameter_storage = {p.untyped_storage().data_ptr() for p in parameters}
        self.stats = stats

    def pack(self, tensor):
        shape = tuple(tensor.shape)
        sparse_input = (
            tensor.is_floating_point() and tensor.requires_grad and tensor.ndim >= 2
            and shape[-1] == self.width and tensor.stride(-1) == 1
            and tensor.numel() == self.indices.shape[0] * self.width
            and tensor.untyped_storage().data_ptr() not in self.parameter_storage
        )
        if sparse_input:
            with torch.no_grad():
                values = gather(tensor.reshape(-1, self.width), self.indices)
            self.stats['packed'] += 1
            self.stats['logical_input_bytes'] += tensor.numel() * tensor.element_size()
            self.stats['logical_packed_bytes'] += values.numel() * values.element_size()
            return ('selected', values, self.indices, shape, weakref.ref(tensor), tensor._version)
        return ('native', tensor.detach(), tensor._version)

    @staticmethod
    def unpack(packed):
        if packed[0] == 'native':
            _, tensor, version = packed
            if tensor._version != version:
                raise RuntimeError('Saved native tensor mutated before backward')
            return tensor
        _, values, indices, shape, original, version = packed
        source = original()
        if source is not None and source._version != version:
            raise RuntimeError('Saved sparse input mutated before backward')
        return scatter(values, indices, shape[-1]).reshape(shape)


def packed_forward(module, x):
    if module.selector != 'raw_gate' or module.current_k != module.target_k:
        raise ValueError('Only fixed raw-gate MoC is supported')
    gate, up = module.gate_proj(x), module.up_proj(x)
    product, indices = selected_product(gate, up, module.target_k, module.act_fn)
    codec = SavedInputCodec(indices, module.intermediate_size,
                            module.down_proj.parameters(), module._moc_saved_stats)
    with torch.autograd.graph.saved_tensors_hooks(codec.pack, codec.unpack):
        return module.down_proj(scatter_selected(product, indices, module.intermediate_size))


def install(module):
    if hasattr(module, '_moc_saved_original') or getattr(module, '_moc_replay_down', False):
        raise ValueError('Duplicate or replay adapter')
    if module.selector != 'raw_gate' or module.current_k != module.target_k:
        raise ValueError('Only fixed raw-gate MoC is supported')
    projection = module.down_proj
    if not hasattr(projection, 'base_layer') or projection.base_layer.weight.requires_grad:
        raise ValueError('Expected frozen-base LoRA down projection')
    if projection.merged or projection.disable_adapters or len(projection.active_adapters) != 1:
        raise ValueError('Expected one active unmerged adapter')
    name = projection.active_adapters[0]
    if getattr(projection, 'use_dora', {}).get(name, False):
        raise ValueError('DoRA is unsupported')
    if not all(isinstance(m, torch.nn.Linear) and m.bias is None for m in (
            projection.base_layer, projection.lora_A[name], projection.lora_B[name])):
        raise ValueError('Only bias-free linear projections are supported')
    module._moc_saved_original = module.forward
    module._moc_saved_stats = dict(packed=0, logical_input_bytes=0, logical_packed_bytes=0)
    module.forward = types.MethodType(packed_forward, module)
    return module
