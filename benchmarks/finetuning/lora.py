"""Preserve rank32 operations when expanding to rank128 LoRA."""
import math
import torch
import torch.nn.functional as F


class SplitA(torch.nn.Linear):
    def forward(self, x):
        assert self.bias is None
        return torch.cat((F.linear(x, self.weight[:32].contiguous()),
                          F.linear(x, self.weight[32:].contiguous())), dim=-1)


class SplitB(torch.nn.Linear):
    def forward(self, x):
        assert self.bias is None
        return (F.linear(x[..., :32].contiguous(), self.weight[:, :32].contiguous())
                + F.linear(x[..., 32:].contiguous(), self.weight[:, 32:].contiguous()))


class _AttachCast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, buffer):
        ctx.source_dtype = source.dtype
        return buffer

    @staticmethod
    def backward(ctx, grad):
        return grad.to(ctx.source_dtype), None


class _MoCSplitA(SplitA):
    def forward(self, x):
        assert self.bias is None
        device = x.device.type
        enabled = torch.is_autocast_enabled(device)
        dtype = torch.get_autocast_dtype(device) if enabled else x.dtype
        if enabled and x.dtype == torch.float32 and dtype == torch.bfloat16:
            # Share storage, not the cast VJP: the original sum occurs in FP32.
            with torch.no_grad():
                buffer = x.to(dtype)
            left = _AttachCast.apply(x, buffer)
            right = _AttachCast.apply(x, buffer)
        else:
            left = right = x
        return torch.cat((F.linear(left, self.weight[:32].contiguous()),
                          F.linear(right, self.weight[32:].contiguous())), dim=-1)


class _MoCSplitB(SplitB):
    def forward(self, x):
        assert self.bias is None
        return F.linear(x[..., :32], self.weight[:, :32]) + F.linear(x[..., 32:], self.weight[:, 32:])


def optimize_moc_lora(model):
    """Install the measured layout on this MoC instance, never on Dense classes."""
    count = 0
    for module in model.modules():
        if not hasattr(module, 'lora_A') or 'default' not in module.lora_A:
            continue
        a, b = module.lora_A['default'], module.lora_B['default']
        if type(a) is not SplitA or type(b) is not SplitB:
            raise ValueError('Expected rank128 split LoRA before MoC optimization')
        a.__class__, b.__class__ = _MoCSplitA, _MoCSplitB
        count += 1
    if count != 192:
        raise ValueError('Expected six LoRA projections in each of 32 layers')
    return count


def install_split(model):
    count = 0
    for module in model.modules():
        if not hasattr(module, 'lora_A') or 'default' not in module.lora_A:
            continue
        a, b = module.lora_A['default'], module.lora_B['default']
        assert type(a) is torch.nn.Linear and type(b) is torch.nn.Linear
        assert a.out_features == b.in_features and a.out_features == 128
        a.__class__, b.__class__ = SplitA, SplitB
        count += 1
    assert count == 192
    return count



def expanded_pair(a, b, rank):
    old_rank, features = a.shape
    if b.shape[1] != old_rank or rank <= old_rank:
        raise ValueError('incompatible or non-expanding rank')
    new_a = a.new_empty((rank, features))
    torch.nn.init.kaiming_uniform_(new_a, a=math.sqrt(5))
    new_b = b.new_zeros((b.shape[0], rank))
    new_a[:old_rank].copy_(a)
    new_b[:, :old_rank].copy_(b)
    return new_a, new_b


@torch.no_grad()
def expand_rank(model, rank, adapter='default'):
    if rank != 128:
        raise ValueError('This recipe expands rank32 to rank128')
    config = model.peft_config[adapter]
    assert config.r == 32 and config.lora_alpha == 64
    assert not config.rank_pattern and not config.alpha_pattern
    assert not config.use_rslora and not config.use_dora
    modules = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260924)
        for name, module in model.named_modules():
            if not hasattr(module, 'lora_A') or adapter not in module.lora_A:
                continue
            a, b = module.lora_A[adapter], module.lora_B[adapter]
            assert a.weight.device.type == b.weight.device.type == 'cpu'
            assert a.weight.dtype == b.weight.dtype == torch.float32
            assert a.bias is None and b.bias is None and module.r[adapter] == 32
            assert module.scaling[adapter] == 2 and not module.merged
            expanded_a, expanded_b = expanded_pair(a.weight, b.weight, rank)
            next_a = torch.nn.Linear(a.in_features, rank, bias=False, dtype=a.weight.dtype)
            next_b = torch.nn.Linear(rank, b.out_features, bias=False, dtype=b.weight.dtype)
            next_a.weight.copy_(expanded_a)
            next_b.weight.copy_(expanded_b)
            assert torch.equal(next_a.weight[:32], a.weight)
            assert torch.equal(next_b.weight[:, :32], b.weight)
            assert not torch.count_nonzero(next_b.weight[:, 32:])
            module.lora_A[adapter], module.lora_B[adapter] = next_a, next_b
            module.r[adapter], module.lora_alpha[adapter] = rank, 2*rank
            assert module.scaling[adapter] == 2
            modules.append(name)
    assert len(modules) == 192
    config.r, config.lora_alpha = rank, 2*rank
    assert install_split(model) == len(modules)
    return {'old_rank': 32, 'new_rank': rank, 'alpha': 2*rank, 'scaling': 2,
            'modules': modules, 'old_factors_exact': True, 'added_B_initially_zero': True,
            'added_A_initialization': 'Kaiming uniform, seed20260924',
            'algebraically_identical_initial_function': True,
            'bf16_full_model_initial_equivalence_requires_runtime_check': True,
            'original_rank32_gemm_preserved_by_split': True,
            'split_forward_additional_calls_and_buffers_must_be_counted': True}
