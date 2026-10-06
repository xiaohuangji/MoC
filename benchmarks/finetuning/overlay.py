"""Explicit base-weight sidecar; a PEFT adapter alone is not this model."""
import hashlib
import json
from pathlib import Path

import torch

from reconstruction import tensor_digest

BASE_NAMES = {f'{projection}.base_layer.weight' for projection in ('gate_proj', 'up_proj', 'down_proj')}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_layer(module, index, directory, original):
    directory.mkdir(exist_ok=True)
    tensors = {n: p.detach().cpu().clone() for n, p in module.named_parameters() if n in BASE_NAMES}
    assert set(tensors) == BASE_NAMES and set(original) == BASE_NAMES
    assert all(t.dtype == torch.bfloat16 and torch.isfinite(t).all() for t in tensors.values())
    path = directory / f'layer_{index:02d}.pt'
    torch.save(tensors, path)
    restored = torch.load(path, weights_only=True)
    assert tensors.keys() == restored.keys() and all(torch.equal(tensors[n], restored[n]) for n in tensors)
    return {'layer': index, 'file': path.name, 'file_sha256': file_sha(path), 'bytes': path.stat().st_size,
            'original': dict(original), 'exported': {n: tensor_digest(t) for n, t in tensors.items()}}


@torch.no_grad()
def apply_layer(module, directory, entry):
    path = directory / entry['file']
    assert Path(entry['file']).name == entry['file']
    assert file_sha(path) == entry['file_sha256']
    tensors = torch.load(path, map_location='cpu', weights_only=True)
    params = dict(module.named_parameters())
    assert set(tensors) == BASE_NAMES == set(entry['original']) == set(entry['exported'])
    # Validate everything before changing any weight in this layer.
    for name, tensor in tensors.items():
        p = params[name]
        assert tensor.shape == p.shape and tensor.dtype == p.dtype == torch.bfloat16
        assert torch.isfinite(tensor).all()
        assert tensor_digest(p) == entry['original'][name]
        assert tensor_digest(tensor) == entry['exported'][name]
    for name, tensor in tensors.items():
        params[name].copy_(tensor)
        assert tensor_digest(params[name]) == entry['exported'][name]


def load_overlay(layers, directory, source_adapter_hash, model_config_hash, expected_layers=32):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['complete'] and manifest['format'] == 'base_ffn_bf16_v1'
    assert manifest['k'] == 2048 and manifest['selector'] == 'raw_gate'
    assert manifest['source_adapter_sha256'] == source_adapter_hash
    assert manifest['model_config_sha256'] == model_config_hash
    assert [r['layer'] for r in manifest['layers']] == list(range(expected_layers))
    for entry in manifest['layers']:
        apply_layer(layers[entry['layer']].mlp, directory, entry)
    return manifest


def verify_overlay_weights(layers, manifest):
    for entry in manifest['layers']:
        params = dict(layers[entry['layer']].mlp.named_parameters())
        for name in BASE_NAMES:
            p = params[name]
            assert not p.requires_grad and p.dtype == torch.bfloat16
            assert tensor_digest(p) == entry['exported'][name]
