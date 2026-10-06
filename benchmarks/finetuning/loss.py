"""Same weighted causal objective, chunked FP32 CE with BF16 logits retained."""
import torch
import torch.nn.functional as F


class _ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, weights, chunk_tokens):
        batch, sequence, vocabulary = logits.shape
        losses = torch.empty_like(targets, dtype=torch.float32)
        count = batch * (sequence - 1)
        for start in range(0, count, chunk_tokens):
            ids = torch.arange(start, min(start+chunk_tokens, count), device=logits.device)
            rows, columns = ids // (sequence-1), ids % (sequence-1)
            values = logits[rows, columns].float()
            losses.view(-1)[start:start+len(ids)] = F.cross_entropy(
                values, targets.reshape(-1)[start:start+len(ids)], ignore_index=-100, reduction='none')
        denominator = weights.sum()
        loss = (losses * weights).sum() / denominator
        ctx.save_for_backward(logits, targets, weights, denominator)
        ctx.chunk_tokens = chunk_tokens
        ctx.mark_non_differentiable(losses)
        return loss, losses

    @staticmethod
    def backward(ctx, grad_loss, _grad_losses):
        logits, targets, weights, denominator = ctx.saved_tensors
        batch, sequence, vocabulary = logits.shape
        result = torch.empty_like(logits)
        result[:, -1].zero_()
        grad_weights = (grad_loss / denominator) * weights
        count = batch * (sequence-1)
        for start in range(0, count, ctx.chunk_tokens):
            ids = torch.arange(start, min(start+ctx.chunk_tokens, count), device=logits.device)
            rows, columns = ids // (sequence-1), ids % (sequence-1)
            with torch.enable_grad(), torch.autocast(logits.device.type, enabled=False):
                values = logits[rows, columns].detach().float().requires_grad_()
                losses = F.cross_entropy(values, targets.reshape(-1)[start:start+len(ids)],
                                         ignore_index=-100, reduction='none')
                gradient, = torch.autograd.grad(losses, values, grad_weights.reshape(-1)[start:start+len(ids)])
            result[rows, columns] = gradient.to(logits.dtype)
        return result, None, None, None


def weighted_causal_loss(logits, labels, response_mask, answer_weight, chunk_tokens=256, *, diagnostics=True):
    if answer_weight < 1 or chunk_tokens < 1:
        raise ValueError('Invalid weighting or chunk size')
    targets = labels[:, 1:].contiguous()
    valid = targets.ne(-100)
    response = response_mask[:, 1:].bool() & valid
    if not valid.any():
        raise ValueError('empty supervised batch')
    weights = valid.float() + response.float() * (answer_weight-1)
    loss, losses = _ChunkedCE.apply(logits, targets, weights, chunk_tokens)
    if not diagnostics:
        return loss, {}
    return loss, {'unweighted_loss': float(losses.detach().sum()/valid.sum()),
                  'answer_loss': float(losses.detach()[response].mean()) if response.any() else None,
                  'answer_tokens': int(response.sum()), 'supervised_tokens': int(valid.sum())}


def denominator(batch):
    valid = batch['labels'][:, 1:].ne(-100)
    response = batch['response_mask'][:, 1:].bool() & valid
    return int(valid.sum()) + 15 * int(response.sum())


def micro_batches(batch, size):
    total = len(batch['input_ids'])
    if not 0 < total <= 16 or size <= 0:
        raise ValueError('Invalid logical or physical batch size')
    weight = denominator(batch)
    if weight <= 0:
        raise ValueError('Empty supervised logical batch')
    for start in range(0, total, size):
        part = {k: v[start:start+size] for k, v in batch.items() if k != 'source_indices'}
        part_weight = denominator(part)
        if part_weight:
            yield part, part_weight / weight


def backward_batch(model, batch, device, micro_size, *, diagnostics=True):
    total_loss = 0.
    for part, factor in micro_batches(batch, micro_size):
        part = {k: v.to(device, non_blocking=True) for k, v in part.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16):
            output = model(input_ids=part['input_ids'], attention_mask=part['attention_mask'],
                           use_cache=False, return_dict=True)
            loss, _ = weighted_causal_loss(output.logits, part['labels'], part['response_mask'], 16,
                                          diagnostics=diagnostics)
        scaled = loss * factor
        if not torch.isfinite(scaled):
            raise RuntimeError('Nonfinite training loss')
        scaled.backward()
        total_loss += float(scaled.detach())
        del output, loss, scaled, part
    return total_loss
