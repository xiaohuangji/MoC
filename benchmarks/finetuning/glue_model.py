"""Original Llama weights, raw-gate sparsity, and fresh seven-target standard LoRA."""
from pathlib import Path
import copy
import torch
from torch.nn import functional as F
from transformers import LlamaForSequenceClassification, set_seed
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from peft import LoraConfig, get_peft_model
from io_utils import CONFIG, read, sha
from moc.training.saved_state import install

TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def last_valid(mask):
    assert mask.ndim == 2 and bool(mask.any(dim=1).all())
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    return positions.masked_fill(~mask.bool(), -1).max(dim=1).values


class Classifier(LlamaForSequenceClassification):
    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        kwargs.pop('return_dict', None)
        kwargs.pop('use_cache', None)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            inputs_embeds=inputs_embeds, use_cache=False, return_dict=True, **kwargs)
        hidden = output.last_hidden_state
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_valid(attention_mask)]
        logits = self.score(pooled)
        loss = None
        if labels is not None:
            loss = (F.mse_loss(logits.float().squeeze(-1), labels.float().reshape(-1))
                    if logits.shape[-1] == 1 else F.cross_entropy(logits.float(), labels))
        return SequenceClassifierOutputWithPast(loss=loss, logits=logits,
            hidden_states=output.hidden_states, attentions=output.attentions)


def build(base, task, device, *, overlay=None, seed=42, tiny_config=None):
    set_seed(seed)
    labels = {"stsb": 1, "mnli": 3}.get(task, 2)
    if tiny_config is None:
        model = Classifier.from_pretrained(base, local_files_only=True, num_labels=labels,
            torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        assert (model.config.hidden_size, model.config.intermediate_size,
                len(model.model.layers), model.config.vocab_size) == (4096, 14336, 32, 128256)
    else:
        config = copy.deepcopy(tiny_config)
        config.num_labels = labels
        model = Classifier(config)
    model.config.pad_token_id = 0
    model.config.use_cache = False
    # Reinitialize the small head with its own generator, identical across models.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261021)
        head = torch.randn(model.score.weight.shape, dtype=torch.float32)*0.02
    model.score.weight.data = head
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, use_rslora=False,
        bias='none', target_modules=TARGETS, modules_to_save=['score'], task_type='SEQ_CLS')
    model = get_peft_model(model, config)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for name, p in trainable.items():
        assert 'lora_' in name or 'score.modules_to_save.' in name, name
        p.data = p.data.float()
        assert torch.isfinite(p).all()
        if '.lora_B.' in name: assert torch.count_nonzero(p) == 0

    if overlay is not None:
        apply_overlay(model, Path(overlay))
    return model.to(device)


def apply_overlay(model, directory):
    layers = model.base_model.model.model.layers
    manifest = read(directory / "OVERLAY.json")
    if not (manifest["passed"] and manifest["generic"] and manifest["method"] == "k25"
            and len(manifest["layers"]) == len(layers)):
        raise ValueError("Not a complete shared K25 overlay")
    if len(layers) == 32:
        if not (manifest.get("original_dense_start") and manifest.get("labels_used") is False
                and manifest.get("fit_windows") == 16384
                and all(e.get("passed") and e.get("steps") == 8192 and e.get("global_token_batch") == 1024
                        for e in manifest["layers"])):
            raise ValueError("Not the shared 16384-window / 8192-step reconstruction recipe")
        if "partition_sha256" in manifest and manifest["partition_sha256"] != read(CONFIG)["partitions"]:
            raise ValueError("Reconstruction partition identity mismatch")
    for i, (block, entry) in enumerate(zip(layers, manifest["layers"])):
        mlp = block.mlp
        width = mlp.up_proj.base_layer.out_features
        k = width // 4
        filename = f"layer_{i:02d}.pt"
        if (entry["layer"], entry["k"], entry["filename"]) != (i, k, filename):
            raise ValueError(f"Unexpected overlay layer {i}")
        path = directory / filename
        if sha(path) != entry["file_sha256"]:
            raise ValueError(f"Wrong overlay weights for layer {i}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if set(state) != {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError(f"Incomplete overlay layer {i}")
        with torch.no_grad():
            for name, weight in state.items():
                dest = getattr(mlp, name).base_layer.weight
                if weight.shape != dest.shape or weight.dtype != dest.dtype:
                    raise ValueError(f"Overlay shape/dtype mismatch: {i}.{name}")
                dest.copy_(weight)
        mlp.selector = "raw_gate"
        mlp.current_k = mlp.target_k = k
        mlp.intermediate_size = width
        install(mlp)
