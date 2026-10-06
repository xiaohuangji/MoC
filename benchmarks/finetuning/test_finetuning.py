"""Fine-tuning regression tests, with optional CUDA coverage for compact copies."""
import copy
import contextlib
import inspect
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear
from transformers import LlamaConfig, LlamaForCausalLM

from common import TARGETS, load_config, write_json
from data import LeftPadCollator
from lora import SplitA, SplitB, _MoCSplitA, _MoCSplitB, expand_rank, optimize_moc_lora
from loss import denominator, micro_batches, weighted_causal_loss
from model import FixedKMLP, attention_checkpoint, build_model, convert, load_trained_adapter, save_adapter
from overlay import BASE_NAMES, load_overlay, save_layer
from prepare_data import answer_mask
from protocol import (TASKS, allowed_next_answer_tokens, canonical_answer_responses,
                      extract_answer, generate_training_prompt)
from reconstruct import reconstruction_rows
from reconstruction import StreamingTensor, fit, tensor_digest
from selected_lora import _SelectedDown, install, uninstall
from train_commonsense import WallBudget
from train_commonsense import arguments as training_arguments
from benchmark_resources import arguments as resource_arguments
from trajectory import cache_dense, capture_pair, correction_target, load_target


def tiny_model(layers=32, sparse=True):
    torch.manual_seed(93)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, attention_dropout=0., use_cache=False)
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model.requires_grad_(False)
    if sparse:
        convert(model.model.layers, 8)
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=.05,
                                           target_modules=list(TARGETS), task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_B" in name:
                value.uniform_(-.02, .02)
    return model


class MathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_bits_equal(self, a, b):
        self.assertEqual((a.shape, a.dtype), (b.shape, b.dtype))
        self.assertTrue(torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),
                                    b.detach().contiguous().reshape(-1).view(torch.uint8)))

    def test_learning_rate_contract(self):
        cfg = load_config()
        for section in ("training", "reconstruction"):
            self.assertEqual(cfg[section]["learning_rate"], 3e-5)
            for value in (0., -1., float("nan"), float("inf"), True, "3e-5"):
                changed = copy.deepcopy(cfg)
                changed[section]["learning_rate"] = value
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "config.yaml"
                    path.write_text(yaml.safe_dump(changed))
                    with self.assertRaisesRegex(ValueError, "learning_rate"):
                        load_config(path)
            changed = copy.deepcopy(cfg)
            changed[section]["lr"] = changed[section].pop("learning_rate")
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "config.yaml"
                path.write_text(yaml.safe_dump(changed))
                with self.assertRaisesRegex(ValueError, "learning_rate"):
                    load_config(path)

    def test_loss_diagnostics_do_not_change_math(self):
        torch.manual_seed(14)
        logits = torch.randn(2, 7, 23, dtype=torch.bfloat16, requires_grad=True)
        labels = torch.randint(23, (2, 7))
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[:, -2:] = True
        original, stats = weighted_causal_loss(logits, labels, mask, 16)
        fast, empty = weighted_causal_loss(logits, labels, mask, 16, diagnostics=False)
        self.assertTrue(stats)
        self.assertEqual(empty, {})
        self.assert_bits_equal(original, fast)
        self.assert_bits_equal(torch.autograd.grad(original, logits)[0], torch.autograd.grad(fast, logits)[0])

    def test_moc_optimization_does_not_modify_dense(self):
        dense = tiny_model(sparse=False)
        expand_rank(dense, 128)
        moc = copy.deepcopy(dense)
        before = {n: p.detach().clone() for n, p in dense.named_parameters()}
        self.assertEqual(optimize_moc_lora(moc), 192)
        for model, a_type, b_type in ((dense, SplitA, SplitB), (moc, _MoCSplitA, _MoCSplitB)):
            for module in model.modules():
                if hasattr(module, "lora_A") and "default" in module.lora_A:
                    self.assertIs(type(module.lora_A["default"]), a_type)
                    self.assertIs(type(module.lora_B["default"]), b_type)
            for name, p in model.named_parameters():
                self.assert_bits_equal(before[name], p)
        ids = torch.arange(12).view(2, 6)
        for model in (dense, moc):
            model.eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            self.assert_bits_equal(dense(input_ids=ids).logits, moc(input_ids=ids).logits)

    def check_down_updates(self, device):
        for training in (True, False):
            torch.manual_seed(17)
            base = torch.nn.Linear(256, 128, bias=False, dtype=torch.bfloat16, device=device)
            base.requires_grad_(False)
            config = LoraConfig(r=128, lora_alpha=256, lora_dropout=.05)
            reference = Linear(base, adapter_name="default", config=config,
                               r=128, lora_alpha=256, lora_dropout=.05)
            reference.lora_A["default"].to(dtype=torch.float32)
            reference.lora_B["default"].to(dtype=torch.float32)
            reference.lora_A["default"].__class__ = SplitA
            reference.lora_B["default"].__class__ = SplitB
            with torch.no_grad():
                reference.lora_B["default"].weight.normal_(0, .01)
            candidate = copy.deepcopy(reference)
            candidate.lora_A["default"].__class__ = _MoCSplitA
            candidate.lora_B["default"].__class__ = _MoCSplitB
            self.assertEqual(reference.lora_dropout["default"].p, .05)
            indices = torch.randn(17, 256, device=device).topk(32, sorted=False).indices.to(torch.int16)
            data = torch.randn(17, 32, dtype=torch.bfloat16, device=device)
            incoming = torch.randn(17, 128, dtype=torch.bfloat16, device=device)
            states = []
            for module in (reference, candidate):
                module.train(training)
                params = [p for p in module.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(params, lr=3e-5, weight_decay=0.)
                for step in range(3):
                    torch.manual_seed(42 + step)
                    optimizer.zero_grad(set_to_none=True)
                    x = data.clone().requires_grad_()
                    with torch.autocast(device, dtype=torch.bfloat16):
                        if module is reference:
                            dense = x.new_zeros((17, 256)).scatter(-1, indices.long(), x)
                            y = module(dense)
                        else:
                            y = _SelectedDown.apply(x, indices, module, 256, *params)
                    y.backward(incoming)
                    optimizer.step()
                    values = [y.detach(), x.grad, torch.get_rng_state()]
                    if device == "cuda":
                        values.append(torch.cuda.get_rng_state())
                    for p in params:
                        values.extend((p.grad, p.detach(), *optimizer.state[p].values()))
                    if module is reference:
                        states.append([v.clone() for v in values])
                    else:
                        for old, new in zip(states[step], values):
                            self.assert_bits_equal(old, new)

    def test_down_updates_are_bitwise_exact_cpu(self):
        self.check_down_updates("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_down_updates_are_bitwise_exact_cuda(self):
        self.check_down_updates("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_compact_cuda_copies(self):
        from compact_kernels import gather, scatter
        for rows, width, k in ((17, 256, 32), (256, 14336, 2048)):
            indices = torch.randn(rows, width, device="cuda").topk(k, sorted=False).indices.to(torch.int16)
            for dtype in (torch.bfloat16, torch.float32):
                x = torch.randn(rows, width, device="cuda", dtype=dtype)
                compact = gather(x, indices)
                self.assert_bits_equal(compact, x.gather(-1, indices.long()))
                self.assert_bits_equal(scatter(compact, indices, width),
                                       torch.zeros_like(x).scatter(-1, indices.long(), compact))

    def test_public_method_is_the_complete_moc_recipe(self):
        base = ["--output-dir", "unused-test-output"]
        dense = training_arguments(base + ["--method", "dense"])
        moc = training_arguments(base + ["--method", "moc", "--overlay", "overlay"])
        self.assertEqual((dense.checkpoint_policy, dense.micro_batch_size), ("block", 16))
        self.assertEqual((moc.checkpoint_policy, moc.micro_batch_size), ("attention", 8))
        self.assertFalse(hasattr(moc, "selected_activations"))
        self.assertNotIn("selected", inspect.signature(build_model).parameters)
        resource_base = base + ["--checkpoint-policy", "attention", "--micro-batch-size", "8"]
        resource_arguments(resource_base + ["--method", "moc", "--overlay", "overlay"])
        for parse, prefix in ((training_arguments, base), (resource_arguments, resource_base)):
            for invalid in (["--method", "moc"],
                            ["--method", "dense", "--overlay", "overlay"],
                            ["--method", "moc", "--overlay", "overlay", "--selected-activations"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    parse(prefix + invalid)
                self.assertEqual(raised.exception.code, 2)
        with self.assertRaisesRegex(ValueError, "requires the reconstructed"):
            build_model({}, "moc", "cpu")

    def test_signed_gate_topk_and_parameter_identity(self):
        module = tiny_model(1).base_model.model.model.layers[0].mlp
        x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
        module.eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            g, u = module.gate_proj(x), module.up_proj(x)
            indices = g.topk(8, dim=-1, sorted=False).indices
            z = F.silu(g.gather(-1, indices)) * u.gather(-1, indices)
            expected = module.down_proj(torch.zeros_like(g).scatter(-1, indices, z))
            self.assertTrue(torch.equal(module(x), expected))
            module.current_k = 32
            self.assertTrue(torch.equal(module(x), module.down_proj(F.silu(g) * u)))
        dense = tiny_model(1, sparse=False).base_model.model.model.layers
        old = {id(p) for p in dense.parameters()}
        convert(dense, 8)
        self.assertEqual(old, {id(p) for p in dense.parameters()})
        with self.assertRaises(ValueError):
            convert(dense, 8)

    def test_chunked_ce_full_and_micro_weighting(self):
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(7)
            logits = torch.randn(12, 7, 23, dtype=dtype, requires_grad=True)
            labels = torch.randint(23, (12, 7))
            labels[:3, :4] = -100
            mask = torch.zeros_like(labels, dtype=torch.bool)
            mask[:5, -2:] = True
            targets = labels[:, 1:]
            weights = targets.ne(-100).float() * (1 + 15 * mask[:, 1:].float())
            ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, 23), targets.reshape(-1), ignore_index=-100, reduction="none")
            reference = (ce.view_as(weights) * weights).sum() / weights.sum()
            expected, = torch.autograd.grad(reference, logits)
            value, _ = weighted_causal_loss(logits, labels, mask, 16, chunk_tokens=11)
            actual, = torch.autograd.grad(value, logits)
            torch.testing.assert_close(value, reference, rtol=0, atol=1e-6)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            batch = dict(input_ids=labels, labels=labels, response_mask=mask, source_indices=torch.arange(12))
            combined = 0.
            start = 0
            for part, factor in micro_batches(batch, 8):
                size = len(part["labels"])
                result, _ = weighted_causal_loss(logits[start:start+size], part["labels"], part["response_mask"], 16)
                combined += factor * result
                start += size
            torch.testing.assert_close(combined, value, rtol=1e-6, atol=1e-6)
            self.assertEqual(denominator(batch), int(weights.sum()))
        with self.assertRaises(ValueError):
            list(micro_batches(dict(input_ids=torch.ones(17, 2)), 8))

    def test_selected_peft_replay_rng_and_gradients(self):
        for policy in ("none", "attention", "block"):
            reference = tiny_model()
            expand_rank(reference, 128)
            selected = copy.deepcopy(reference)
            optimize_moc_lora(selected)
            for model in (reference, selected):
                model.enable_input_require_grads()
                if policy == "attention":
                    attention_checkpoint(model)
                elif policy == "block":
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.train()
            for block in selected.base_model.model.model.layers:
                install(block.mlp)
            ids = torch.arange(12).view(2, 6)
            def step(model):
                model.zero_grad(set_to_none=True)
                torch.manual_seed(101)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    output = model(input_ids=ids, use_cache=False).logits
                    loss, _ = weighted_causal_loss(output, ids, torch.ones_like(ids, dtype=torch.bool), 16)
                rng = torch.get_rng_state().clone()
                loss.backward()
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                return output.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
            old, old_grad = step(reference)
            new, new_grad = step(selected)
            self.assertTrue(torch.equal(old, new), policy)
            self.assertEqual(len(old_grad), 384)
            for name in old_grad:
                torch.testing.assert_close(old_grad[name], new_grad[name], rtol=.02, atol=3e-4, msg=f"{policy}:{name}")
            for block in selected.base_model.model.model.layers:
                uninstall(block.mlp)

    def test_rank_expansion_and_adapter_roundtrip(self):
        model = tiny_model()
        model.eval()
        ids = torch.arange(12).view(2, 6)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            before = model(input_ids=ids).logits
            expand_rank(model, 128)
            optimize_moc_lora(model)
            after = model(input_ids=ids).logits
        self.assertTrue(torch.equal(before, after))
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)
        for update in range(4):
            torch.manual_seed(400 + update)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = model(input_ids=ids, use_cache=False).logits
                loss, _ = weighted_causal_loss(logits, ids, torch.ones_like(ids, dtype=torch.bool), 16)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            self.assertTrue(torch.isfinite(norm))
            optimizer.step()
        model.eval()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("{}")
            cfg = dict(k=2048, source_adapter_sha256="a" * 64, model_dir=str(root))
            save_adapter(model, root / "adapter", cfg, "moc", None, 4)
            reloaded = tiny_model()
            expand_rank(reloaded, 128)
            optimize_moc_lora(reloaded)
            load_trained_adapter(reloaded, root / "adapter", cfg, "moc", None)
            for (name, old), (name2, new) in zip(model.named_parameters(), reloaded.named_parameters()):
                self.assertEqual(name, name2)
                self.assertTrue(torch.equal(old, new), name)
            reloaded.eval()
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                self.assertTrue(torch.equal(model(input_ids=ids).logits, reloaded(input_ids=ids).logits))
            with self.assertRaises(ValueError):
                load_trained_adapter(reloaded, root / "adapter", cfg, "dense", None)

    def test_reconstruction_frozen_weights_and_overlay(self):
        module = tiny_model(1).base_model.model.model.layers[0].mlp
        module.requires_grad_(False).eval()
        original_module = copy.deepcopy(module)
        original = {n: tensor_digest(p) for n, p in module.named_parameters() if n in BASE_NAMES}
        frozen = {n: tensor_digest(p) for n, p in module.named_parameters() if n not in BASE_NAMES}
        # CPU fit uses FP32 throughout; production fits FP32 masters under CUDA BF16 autocast.
        module.float()
        frozen = {n: tensor_digest(p) for n, p in module.named_parameters() if n not in BASE_NAMES}
        for name, value in module.named_parameters():
            value.requires_grad_(name in BASE_NAMES)
        x = torch.randn(16, 16)
        with torch.no_grad():
            y = module(x) * .9
        fit(module, StreamingTensor(x, "cpu"), StreamingTensor(y, "cpu"), steps=4, lr=3e-5, batch_tokens=8)
        self.assertEqual(frozen, {n: tensor_digest(p) for n, p in module.named_parameters() if n not in BASE_NAMES})
        for name, value in module.named_parameters():
            value.requires_grad_(False)
            if name in BASE_NAMES:
                value.data = value.data.bfloat16()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = save_layer(module, 0, root, original)
            manifest = dict(format="base_ffn_bf16_v1", complete=True, k=2048, selector="raw_gate",
                            source_adapter_sha256="a", model_config_sha256="b", layers=[entry])
            write_json(root / "manifest.json", manifest)
            holder = torch.nn.Module()
            holder.mlp = original_module
            load_overlay([holder], root, "a", "b", expected_layers=1)
            for name, value in holder.mlp.named_parameters():
                if name in BASE_NAMES:
                    self.assertEqual(tensor_digest(value), entry["exported"][name])
            manifest["complete"] = False
            write_json(root / "manifest.json", manifest)
            with self.assertRaises(AssertionError):
                load_overlay([holder], root, "a", "b", expected_layers=1)

    def test_teacher_student_token_alignment(self):
        model = tiny_model(2, sparse=False).requires_grad_(False).eval()
        layers = model.base_model.model.model.layers
        ids = np.arange(18, dtype=np.int32).reshape(3, 6)
        lengths = np.array([6, 4, 5])
        rows = np.arange(3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            with torch.autocast("cpu", dtype=torch.bfloat16):
                cache = cache_dense(model, layers, ids, lengths, [("train", rows, 10, 7)], [0, 1], root)
                convert(layers, 8)
                inputs, residual, capture = capture_pair(model, layers[0], ids, lengths, rows, 10, 7, "cpu")
            self.assertEqual(capture["selection_sha256"], cache["groups"]["train"]["selection_sha256"])
            targets = load_target(root, cache, "train", 0, "cpu")
            corrected = correction_target(targets, residual)
            self.assertEqual(corrected.dtype, torch.float32)
            torch.testing.assert_close(corrected + residual.float(), targets.float(), rtol=0, atol=0)
            self.assertEqual(len(inputs), 10)

    def test_prompt_boundaries_padding_and_labels(self):
        text, prefix = "prompt answer", "prompt "
        ids, mask, intact = answer_mask(text, prefix, [1, 2, 3], [(0, 0), (0, 7), (7, 13)], 9, 8)
        self.assertTrue(intact)
        self.assertEqual(ids, [1, 2, 3, 9])
        self.assertEqual(mask, [False, False, True, True])
        _, mask2, intact2 = answer_mask(text, prefix, [1, 2, 3], [(0, 0), (0, 7), (7, 10)], 9, 3)
        self.assertFalse(intact2)
        self.assertFalse(any(mask2))
        collated = LeftPadCollator(0)([dict(input_ids=torch.tensor(ids), response_mask=torch.tensor(mask), source_index=11)])
        self.assertEqual(collated["input_ids"].shape, (1, 8))
        self.assertTrue(collated["labels"][0, :4].eq(-100).all())
        self.assertEqual(int(collated["response_mask"].sum()), 2)
        prompt = generate_training_prompt(dict(instruction="Q", input="", output="A"))
        self.assertTrue(prompt.endswith("### Response:\n                A"))
        for task in TASKS:
            for answer in canonical_answer_responses(task):
                self.assertTrue(extract_answer(task, answer))
        self.assertEqual(allowed_next_answer_tokens(((3, 4), (3, 5)), (3,), 9), [4, 5])

    def test_splits_schedule_and_config(self):
        train, validation = reconstruction_rows(np.arange(170300))
        self.assertEqual((len(train), len(validation)), (8192, 128))
        self.assertFalse(np.intersect1d(train, validation).size)
        cfg = load_config()
        self.assertEqual(cfg["training"]["steps"], 10240)
        now = [0.]
        schedule = WallBudget(100., 2, clock=lambda: now[0])
        self.assertEqual(schedule.factor(0), 0.)
        now[0] = 10.
        self.assertEqual(schedule.factor(1), .5)
        now[0] = 20.
        self.assertEqual(schedule.factor(2), 1.)
        now[0] = 60.
        self.assertEqual(schedule.factor(3), .5)
        now[0] = 100.
        self.assertTrue(schedule.exhausted())
        self.assertEqual(schedule.factor(4), 0.)


if __name__ == "__main__":
    unittest.main()
