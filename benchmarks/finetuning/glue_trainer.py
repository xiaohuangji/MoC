"""Keep the original four-rank best-checkpoint synchronization and head reload."""
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import Trainer

from checkpoint import load_adapter


class GlueTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, *args, **kwargs):
        inputs.pop("sample_id", None)
        return super().compute_loss(model, inputs, *args, **kwargs)

    def training_step(self, model, inputs, *args, **kwargs):
        result = super().training_step(model, inputs, *args, **kwargs)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite supervised loss")
        return result

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        # In HF 4.51 a non-saving rank may check existence before rank zero writes.
        if dist.is_initialized():
            dist.barrier()
        if self.state.best_global_step is not None:
            folder = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.best_global_step}"
            if not (folder / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(folder)
            self.state.best_model_checkpoint = str(folder)
        if dist.is_initialized():
            state = (self.state.best_model_checkpoint, self.state.best_metric)
            states = [None] * dist.get_world_size()
            dist.all_gather_object(states, state)
            if any(s != state for s in states):
                raise RuntimeError("Ranks disagree on the selected checkpoint")
        if self.args.should_save:
            path = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
            self.state.save_to_json(str(path / "trainer_state.json"))
        if dist.is_initialized():
            dist.barrier()

    def _load_best_model(self):
        load_adapter(self.model, self.state.best_model_checkpoint)
