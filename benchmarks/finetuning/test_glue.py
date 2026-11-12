from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, TrainingArguments

from checkpoint import adapter_identity, load_adapter
from dedup import Deduplicator, document_sha
from glue_data import TASKS, collate, pack
from glue_model import build, last_valid
from glue_trainer import GlueTrainer
from io_utils import sha, write
from metrics import SCORERS
from recon_fit import ReconstructionMLP, loss_from_parts, positions, required_indices, sample_groups
from moc.training.saved_state import packed_forward
from recon_capture import capture, initial
from recon_storage import RAMRecords


class GlueContracts(unittest.TestCase):
    def test_data_and_metrics(self):
        ids, truncated = pack([10, 11, 12], [20, 21, 22], 1, 2, limit=8)
        self.assertEqual((ids, truncated), ([1, 10, 11, 2, 20, 21, 22, 2], True))
        batch = collate([dict(input_ids=[1, 2], labels=1.25, sample_id=1),
                         dict(input_ids=[1, 3, 2], labels=3.5, sample_id=2)])
        self.assertEqual(last_valid(batch["attention_mask"]).tolist(), [1, 2])
        self.assertTrue(batch["labels"].is_floating_point())
        self.assertEqual(SCORERS.keys(), TASKS.keys())
        for task in ("sst2", "qnli", "rte"):
            self.assertEqual(SCORERS[task]([1, 0], [1, 1])["accuracy"], .5)
        for task in ("mrpc", "qqp"):
            self.assertAlmostEqual(SCORERS[task]([1, 0], [1, 1])["combined"], (.5 + 2/3)/2)
        self.assertEqual(SCORERS["mnli"]([0, 1, 2], [0, 1, 1], 2)["combined"], .5)
        self.assertEqual(SCORERS["cola"]([1, 0], [0, 1])["matthews_correlation"], -1.)
        self.assertAlmostEqual(SCORERS["stsb"]([1., 1., 3.], [2., 2., 5.])["combined"], 1.)

    def test_reconstruction_contracts(self):
        dedup = Deduplicator()
        tokens = list(range(128))
        self.assertTrue(dedup.accept(document_sha("a b"), tokens))
        self.assertFalse(dedup.accept(document_sha("a  b"), tokens))
        self.assertFalse(dedup.accept(document_sha("another"), tokens[:64] + [321] + tokens[65:]))
        counts = [12, 13, 11, 14]
        g = torch.Generator().manual_seed(20261022)
        cached = [required_indices(counts, 0, r, 8, 10) for r in range(4)]
        for _ in range(8):
            chunks = sample_groups(counts, g, 10)
            self.assertEqual(sum(map(len, chunks)), 10)
            for ids, required in zip(chunks, cached):
                self.assertTrue(torch.equal(required[positions(required, ids)], ids))
        n = torch.tensor([2., 5.], requires_grad=True)
        loss = loss_from_parts(list(n), list(torch.tensor([3., 7.])))
        loss.backward()
        torch.testing.assert_close(n.grad, torch.tensor([.1, .1]))

    def test_overlay_operator_and_independent_reload(self):
        cfg = LlamaConfig(hidden_size=16, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=2, vocab_size=32)
        device = torch.device("cpu")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dense = build(None, "stsb", device, tiny_config=cfg)
            entries = []
            for i, block in enumerate(dense.base_model.model.model.layers):
                path = root / f"layer_{i:02d}.pt"
                torch.save({n: getattr(block.mlp, n).base_layer.weight.detach() for n in
                            ("gate_proj", "up_proj", "down_proj")}, path)
                entries.append(dict(layer=i, filename=path.name, file_sha256=sha(path), k=16))
            write(root / "OVERLAY.json", dict(passed=True, generic=True, method="k25", layers=entries))
            model = build(None, "stsb", device, overlay=root, tiny_config=cfg)
            self.assertTrue(all(b.mlp.forward.__func__ is packed_forward for b in model.base_model.model.model.layers))
            self.assertEqual(adapter_identity(dense), adapter_identity(model))
            # Compare against the native PEFT backward with nonzero LoRA B and dropout.
            for name, p in model.named_parameters():
                if ".lora_B." in name:
                    p.data.add_(.01)
            batch = dict(input_ids=torch.tensor([[1, 3, 2]]), attention_mask=torch.ones(1, 3),
                         labels=torch.tensor([1.25]))
            native = build(None, "stsb", device, tiny_config=cfg)
            for block in native.base_model.model.model.layers:
                block.mlp = ReconstructionMLP(block.mlp)
            native.load_state_dict(model.state_dict())
            model.train()
            native.train()
            for m in (native, model):
                torch.manual_seed(14)
                out = m(**batch)
                torch.testing.assert_close(out.loss, (out.logits.squeeze() - 1.25).square())
                out.loss.backward()
            for (name, a), (_, b) in zip(native.named_parameters(), model.named_parameters()):
                if a.requires_grad:
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
            self.assertTrue(all(b.mlp._moc_saved_stats["packed"] for b in model.base_model.model.model.layers))
            model.save_pretrained(root / "selected", save_embedding_layers=False)
            rebuilt = build(None, "stsb", device, overlay=root, tiny_config=cfg)
            self.assertTrue(all(b.mlp.forward.__func__ is packed_forward for b in rebuilt.base_model.model.model.layers))
            load_adapter(rebuilt, root / "selected")
            torch.testing.assert_close(rebuilt.eval()(**batch).logits, model.eval()(**batch).logits,
                                       atol=0, rtol=0)
            entries[0]["file_sha256"] = "bad"
            write(root / "OVERLAY.json", dict(passed=True, generic=True, method="k25", layers=entries))
            with self.assertRaisesRegex(ValueError, "weights"):
                build(None, "stsb", device, overlay=root, tiny_config=cfg)

    def test_rolling_boundaries_follow_sparse_student_and_dense_teacher(self):
        cfg = LlamaConfig(hidden_size=16, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=2, vocab_size=32)
        device = torch.device("cpu")
        model = build(None, "mnli", device, tiny_config=cfg).eval()
        rows = [dict(input_ids=[1, 3, 5, 2], labels=0, sample_id=0)]
        records = initial(model, rows, device, RAMRecords())
        blocks = model.base_model.model.model.layers
        batch = collate(rows)
        def hidden():
            with torch.no_grad():
                return model.base_model.model.model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]).last_hidden_state
        gold = hidden()
        # Substitute only the Linux memfd allocation, not the block capture or replay.
        alloc = lambda _d, _n, shape, dtype: (None, torch.empty(shape, dtype=dtype))
        with patch("recon_storage.mapped", side_effect=alloc):
            capture(blocks[0], records, device)
            blocks[0].mlp = ReconstructionMLP(blocks[0].mlp)
            (x, residual, target), _ = capture(blocks[1], records, device, previous_block=blocks[0])
        norm = model.base_model.model.model.norm
        torch.testing.assert_close(norm(target).reshape_as(gold), gold)
        torch.testing.assert_close(norm(residual + blocks[1].mlp(x)).reshape_as(gold), hidden())

    def test_trainer_checkpoint_tie(self):
        cfg = LlamaConfig(hidden_size=16, intermediate_size=64, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=2, vocab_size=32)
        rows = [dict(input_ids=[1, 3+i, 2], labels=i, sample_id=i) for i in range(2)]
        with TemporaryDirectory() as tmp:
            model = build(None, "mrpc", torch.device("cpu"), tiny_config=cfg)
            args = TrainingArguments(output_dir=tmp, use_cpu=True, num_train_epochs=2,
                per_device_train_batch_size=2, per_device_eval_batch_size=2, label_names=["labels"],
                remove_unused_columns=False, save_strategy="epoch", eval_strategy="epoch",
                metric_for_best_model="combined", greater_is_better=True, load_best_model_at_end=True,
                report_to=[], disable_tqdm=True)
            trainer = GlueTrainer(model=model, args=args, train_dataset=rows, eval_dataset=rows,
                                  data_collator=collate, compute_metrics=lambda _: {"combined": .5})
            trainer.train()
            selected = Path(trainer.state.best_model_checkpoint)
            self.assertEqual(selected.name, "checkpoint-1")
            other = build(None, "mrpc", torch.device("cpu"), tiny_config=cfg)
            load_adapter(other, selected)
            self.assertEqual(adapter_identity(other), adapter_identity(model))


if __name__ == "__main__":
    unittest.main()
