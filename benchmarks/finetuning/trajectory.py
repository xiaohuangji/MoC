"""Cache original Dense block outputs; fit sparse FFNs to residual corrections."""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from reconstruction import autocast_for, tensor_digest


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def token_plan(lengths, rows, limit, seed):
    total = sum(int(lengths[row]) for row in rows)
    indices = (torch.randperm(total, generator=torch.Generator().manual_seed(seed))[:limit]
               if total > limit else torch.arange(total))
    return indices, total


def batches(token_ids, lengths, rows, device):
    for start in range(0, len(rows), 4):
        batch = rows[start:start+4]
        width = max(int(lengths[r]) for r in batch)
        ids = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            length = int(lengths[row])
            ids[i, -length:] = torch.as_tensor(np.array(token_ids[row, :length]), device=device, dtype=torch.long)
            mask[i, -length:] = 1
        yield ids, mask


@torch.no_grad()
def cache_dense(model, layers, token_ids, lengths, groups, layer_ids, root):
    root.mkdir(parents=True, exist_ok=False)
    width = model.config.hidden_size
    device = next(model.parameters()).device
    started = time.monotonic()
    manifest = {'meaning': 'original Dense complete block outputs, before any sparse conversion',
                'dtype': 'bfloat16 stored as uint16 bits', 'width': width, 'groups': {}}
    for name, rows, limit, seed in groups:
        selected, total = token_plan(lengths, rows, limit, seed)
        directory = root / name
        directory.mkdir()
        maps = {i: np.empty((len(selected), width), dtype=np.uint16) for i in layer_ids}
        visits = torch.zeros(len(selected), dtype=torch.int32)
        current_mask = None
        destination, positions = None, None
        residuals, ffn_values = {}, {}
        exact_block_calls = {i: 0 for i in layer_ids}
        def make_residual_hook(index):
            def hook(_module, args):
                residuals[index] = args[0]
            return hook
        def make_ffn_hook(index):
            def hook(_module, _args, value):
                ffn_values[index] = value
            return hook
        def make_hook(index):
            def hook(_module, _args, output):
                value = output[0] if isinstance(output, tuple) else output
                assert value.dtype == torch.bfloat16 and value.shape[:2] == current_mask.shape
                assert torch.equal(value, residuals.pop(index) + ffn_values.pop(index))
                exact_block_calls[index] += 1
                chosen = value[current_mask.bool()][positions].detach().contiguous().cpu()
                maps[index][destination.numpy()] = chosen.view(torch.uint16).numpy()
            return hook
        handles = []
        for i in layer_ids:
            handles += [layers[i].post_attention_layernorm.register_forward_pre_hook(make_residual_hook(i)),
                        layers[i].mlp.register_forward_hook(make_ffn_hook(i)),
                        layers[i].register_forward_hook(make_hook(i))]
        offset = 0
        try:
            for ids, current_mask in batches(token_ids, lengths, rows, device):
                count = int(current_mask.sum())
                destination = torch.where((selected >= offset) & (selected < offset + count))[0]
                positions = (selected[destination] - offset).to(device)
                with autocast_for(ids):
                    model(input_ids=ids, attention_mask=current_mask, use_cache=False)
                visits[destination] += 1
                offset += count
                if offset == total or sum(exact_block_calls.values()) % (128*len(layer_ids)) == 0:
                    progress = dict(group=name, nonpad_tokens=offset, total_nonpad_tokens=total,
                                    phase='forward_capture', elapsed_seconds=time.monotonic()-started)
                    (root/'progress.json').write_text(json.dumps(progress)+'\n')
                    print('TEACHER_PROGRESS', json.dumps(progress), flush=True)
        finally:
            for handle in handles:
                handle.remove()
        assert offset == total and torch.all(visits == 1)
        files = {}
        for index in list(maps):
            array = maps.pop(index)
            path = directory / f'layer_{index:02d}.bin'
            with path.open('wb') as handle:
                array.tofile(handle)
                handle.flush()
                os.fsync(handle.fileno())
            del array
            files[str(index)] = {'file': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                                 'sha256': file_digest(path)}
            progress = dict(group=name, phase='sequential_write_and_hash', last_layer=index,
                            elapsed_seconds=time.monotonic()-started)
            (root/'progress.json').write_text(json.dumps(progress)+'\n')
            print('TEACHER_PROGRESS', json.dumps(progress), flush=True)
        del maps
        manifest['groups'][name] = {'source_rows': [int(r) for r in rows], 'tokens': len(selected),
            'nonpad_tokens_before_subsample': total, 'selection_sha256': tensor_digest(selected),
            'seed': seed, 'limit': limit, 'files': files, 'every_selected_token_written_once': True,
            'native_shape_residual_sum_exact_calls': exact_block_calls}
    manifest['cache_seconds'] = time.monotonic() - started
    manifest['bytes'] = sum(f['bytes'] for g in manifest['groups'].values() for f in g['files'].values())
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def load_target(root, manifest, group, index, device):
    info = manifest['groups'][group]
    record = info['files'][str(index)]
    path = root / record['file']
    assert path.stat().st_size == record['bytes'] and file_digest(path) == record['sha256']
    data = np.memmap(path, mode='r', dtype=np.uint16, shape=(info['tokens'], manifest['width']))
    return torch.from_numpy(np.array(data, copy=True)).view(torch.bfloat16).to(device)


class CaptureComplete(Exception):
    pass


@torch.no_grad()
def capture_pair(model, block, token_ids, lengths, rows, max_tokens, seed, storage_device=None):
    inputs, residuals = [], []
    current_mask = None
    def residual_hook(_module, args):
        residuals.append(args[0][current_mask.bool()].detach().cpu())
    def input_hook(_module, args):
        inputs.append(args[0][current_mask.bool()].detach().cpu())
        raise CaptureComplete
    handles = [block.post_attention_layernorm.register_forward_pre_hook(residual_hook),
               block.mlp.register_forward_pre_hook(input_hook)]
    started = time.monotonic()
    device = next(model.parameters()).device
    try:
        for ids, current_mask in batches(token_ids, lengths, rows, device):
            try:
                with autocast_for(ids):
                    model(input_ids=ids, attention_mask=current_mask, use_cache=False)
            except CaptureComplete:
                continue
            raise RuntimeError('failed to capture MLP input and pre-MLP residual')
    finally:
        for handle in handles:
            handle.remove()
    selected, total = token_plan(lengths, rows, max_tokens, seed)
    x, residual = torch.cat(inputs), torch.cat(residuals)
    assert len(x) == len(residual) == total
    destination = device if storage_device is None else storage_device
    x, residual = x[selected].to(destination), residual[selected].to(destination)
    assert not x.requires_grad and not residual.requires_grad
    return x, residual, {'source_rows': [int(r) for r in rows], 'nonpad_tokens_before_subsample': total,
        'selected_tokens': len(x), 'input_sha256': tensor_digest(x), 'residual_sha256': tensor_digest(residual),
        'selection_sha256': tensor_digest(selected), 'capture_seconds': time.monotonic()-started}


def correction_target(dense_block_output, student_residual):
    return (dense_block_output.float() - student_residual.float()).detach()


@torch.no_grad()
def measure_block(module, inputs, residuals, dense_outputs):
    error = energy = cosine = 0.0
    for start in range(0, len(inputs), 1024):
        x = inputs[start:start+1024]
        target = dense_outputs[start:start+1024].float()
        with autocast_for(x):
            prediction = residuals[start:start+1024] + module(x)
        prediction = prediction.float()
        assert torch.isfinite(prediction).all()
        error += float((prediction-target).square().sum())
        energy += float(target.square().sum())
        cosine += float(F.cosine_similarity(prediction, target, dim=-1).sum())
    assert energy > 1e-12
    return {'relative_mse': error/energy, 'mean_token_cosine': cosine/len(inputs), 'tokens': len(inputs)}
