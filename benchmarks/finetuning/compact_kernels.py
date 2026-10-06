"""Copy-only CUDA kernels for compact int16 activation indices."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather(values, indices, result, count: tl.constexpr, width: tl.constexpr,
            k: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < count
    column = tl.load(indices + offsets, active, 0).to(tl.int32)
    value = tl.load(values + offsets // k * width + column, active, 0)
    tl.store(result + offsets, value, active)


@triton.jit
def _scatter(values, indices, result, count: tl.constexpr, width: tl.constexpr,
             k: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < count
    column = tl.load(indices + offsets, active, 0).to(tl.int32)
    value = tl.load(values + offsets, active, 0)
    tl.store(result + offsets // k * width + column, value, active)


def gather(values, indices):
    assert values.is_cuda and indices.dtype == torch.int16
    assert values.shape[:-1] == indices.shape[:-1] and values.shape[-1] <= 32768
    values, indices = values.contiguous(), indices.contiguous()
    result = values.new_empty(indices.shape)
    count, k = indices.numel(), indices.shape[-1]
    _gather[(triton.cdiv(count, 256),)](values, indices, result, count, values.shape[-1], k, 256)
    return result


def scatter(values, indices, width):
    assert values.is_cuda and indices.dtype == torch.int16 and values.shape == indices.shape
    assert width <= 32768
    values, indices = values.contiguous(), indices.contiguous()
    result = values.new_zeros((*values.shape[:-1], width))
    count, k = indices.numel(), indices.shape[-1]
    _scatter[(triton.cdiv(count, 256),)](values, indices, result, count, width, k, 256)
    return result
