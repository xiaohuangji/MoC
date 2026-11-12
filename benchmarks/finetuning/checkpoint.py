"""Strict, independent PEFT adapter and sequence-head reload."""
from pathlib import Path
import torch
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from glue_data import collate
from io_utils import tensor_sha


def adapter_identity(model):
    return {n: tensor_sha(t) for n, t in get_peft_model_state_dict(model).items()}


def frozen_identity(model):
    # Compare values, not tensor version counters (DDP initialization broadcasts).
    return {n: tensor_sha(p) for n, p in model.named_parameters() if not p.requires_grad}


def load_adapter(model, path):
    state = load_file(str(Path(path)/'adapter_model.safetensors'))
    expected = get_peft_model_state_dict(model)
    assert state.keys() == expected.keys(), 'Adapter/head keys mismatch'
    assert all(state[k].shape == expected[k].shape and torch.isfinite(state[k]).all() for k in state)
    # PEFT rewrites modules_to_save keys in the supplied mapping.
    set_peft_model_state_dict(model, dict(state))
    actual = get_peft_model_state_dict(model)
    assert all(torch.equal(actual[k].detach().cpu(), state[k]) for k in state)


def probe(model, rows, device):
    mode = model.training
    model.eval()
    batch = collate(rows[:4])
    batch.pop('sample_id'); batch.pop('labels')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(**{k: v.to(device) for k, v in batch.items()}).logits.float()
    assert torch.isfinite(logits).all()
    model.train(mode)
    return dict(logits=logits.detach().cpu().tolist(), sha256=tensor_sha(logits))
