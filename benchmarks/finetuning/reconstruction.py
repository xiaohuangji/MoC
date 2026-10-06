"""Local FFN reconstruction utilities; no task-test selection or Dense blending."""
import hashlib
from contextlib import nullcontext

import torch
import torch.nn.functional as F


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def autocast_for(tensor):
    return torch.autocast('cuda', dtype=torch.bfloat16) if tensor.is_cuda else nullcontext()


def relative_mse(prediction, target):
    prediction, target = prediction.float(), target.detach().float()
    energy = target.square().mean()
    if not torch.isfinite(energy) or energy <= 1e-12:
        raise ValueError('invalid or near-zero teacher energy')
    return (prediction - target).square().mean() / energy


@torch.no_grad()
def measure(module, inputs, targets, batch_tokens=1024):
    numerator = denominator = cosine = 0.0
    for start in range(0, len(inputs), batch_tokens):
        x, y = inputs[start:start+batch_tokens], targets[start:start+batch_tokens]
        with autocast_for(x):
            prediction = module(x)
        if not torch.isfinite(prediction).all():
            raise ValueError('nonfinite prediction')
        numerator += float((prediction.float()-y.float()).square().sum())
        denominator += float(y.float().square().sum())
        cosine += float(F.cosine_similarity(prediction.float(), y.float(), dim=-1).sum())
    if denominator <= 1e-12 or not len(inputs):
        raise ValueError('empty/zero-energy target')
    return {'relative_mse': numerator/denominator, 'mean_token_cosine': cosine/len(inputs),
            'tokens': len(inputs), 'error_energy': numerator, 'target_energy': denominator}


def fit(module, inputs, targets, *, steps, lr, batch_tokens=1024, seed=20260920, record=None):
    module.eval()  # Disable adapter dropout while fitting a deterministic teacher function.
    trainable = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    valid = {n for n, _ in trainable} == {
        f'{proj}.base_layer.weight' for proj in ('gate_proj', 'up_proj', 'down_proj')}
    if not valid:
        raise ValueError('Only the current gate/up/down base weights may be trainable')
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=lr, weight_decay=0, eps=1e-8, foreach=False)
    generator = torch.Generator(device=inputs.device).manual_seed(seed)
    last = None
    for step in range(1, steps+1):
        ids = torch.randint(len(inputs), (min(batch_tokens, len(inputs)),), generator=generator, device=inputs.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_for(inputs):
            prediction = module(inputs[ids])
        loss = relative_mse(prediction, targets[ids])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        if not torch.isfinite(norm) or any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in trainable):
            raise ValueError('invalid trainable gradients')
        gate_norm = sum(float(p.grad.square().sum()) for n, p in trainable if 'gate_proj' in n)**.5
        warmup = max(1, steps//10)
        factor = step/warmup if step <= warmup else (steps-step+1)/max(1, steps-warmup)
        for group in optimizer.param_groups:
            group['lr'] = lr * factor
        optimizer.step()
        last = {'step': step, 'loss': float(loss.detach()), 'grad_norm': float(norm), 'gate_grad_norm': gate_norm,
                'lr': optimizer.param_groups[0]['lr']}
        if record and (step in (1, 2, steps) or step % 32 == 0):
            record(last)
    return last


class StreamingTensor:
    def __init__(self, tensor, device):
        if tensor.device.type != 'cpu' or tensor.requires_grad:
            raise ValueError('Captured storage must be detached CPU data')
        self.storage = tensor
        self.device = torch.device(device)
        self.is_cuda = self.device.type == 'cuda'
        self.requires_grad = False

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.to('cpu')
        return self.storage[index].to(self.device)

    def detach(self):
        return self.storage.detach()
