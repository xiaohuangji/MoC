"""Decode aggregation, compilation settings, and CUDA projection checks."""
from __future__ import annotations

import importlib.util
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

MODULE_PATH = Path(__file__).with_name("benchmark_decode.py")
spec = importlib.util.spec_from_file_location("benchmark_decode", MODULE_PATH)
decode = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decode)


class DecodeTests(unittest.TestCase):
    def parse(self, *arguments):
        with patch.object(sys, "argv", ["benchmark_decode.py", "--out", "decode.json", *arguments]):
            return decode.parse_args()

    def test_defaults(self):
        args = self.parse()
        self.assertEqual(args.methods, ["dense", "global_moc"])
        self.assertEqual((args.rounds, args.warmup_runs, args.measure_runs), (3, 8, 30))

    def test_grouped_method(self):
        self.assertEqual(self.parse("--methods", "dense", "moc_2_8").methods, ["dense", "moc_2_8"])

    def test_invalid_counts(self):
        for flag, value in [("--rounds", "0"), ("--measure-runs", "0"), ("--warmup-runs", "-1")]:
            with self.subTest(flag=flag), patch("sys.stderr"), self.assertRaises(SystemExit):
                self.parse(flag, value)

    def test_duplicate_methods(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            self.parse("--methods", "dense", "dense")

    def row(self, latency, peak):
        return dict(row="global_moc", label="MoC", ffn_kind="global_moc", selection="global_topk",
                    ffn_mode="moc_inference_optimized_global_after_gate_native", k=1024,
                    grouped_a=None, grouped_b=None, status="OK", latency_ms_per_token=latency,
                    peak_allocated_bytes=peak)

    def test_aggregate_uses_every_round(self):
        rounds = [self.row(2.0, 100), self.row(5.0, 300), self.row(8.0, 200)]
        result = decode.aggregate_rounds(rounds)
        self.assertEqual(result["latency_ms_per_token"], 5.0)
        self.assertEqual(result["throughput_tok_per_sec"], 200.0)
        self.assertEqual(result["peak_allocated_bytes"], 300)
        self.assertEqual(result["rounds"], rounds)

    def test_failed_round_is_not_discarded(self):
        rounds = [self.row(2.0, 100), {"row": "global_moc", "status": "FAILED"}]
        result = decode.aggregate_rounds(rounds)
        self.assertEqual(result["status"], "FAILED")
        self.assertNotIn("latency_ms_per_token", result)
        self.assertEqual(result["rounds"], rounds)

    def test_empty_rounds(self):
        with self.assertRaises(ValueError):
            decode.aggregate_rounds([])

    def test_comparison(self):
        result = decode.compare_pair(self.row(3.0, 100), self.row(2.0, 200))
        self.assertEqual(result["speedup_dense_over_moc"], 1.5)
        self.assertIsNone(decode.compare_pair(self.row(3.0, 100), {"status": "FAILED"})["speedup_dense_over_moc"])

    def test_compile_settings(self):
        model = Mock()
        model.forward_step_static = lambda tokens, position: tokens + position
        with patch.object(torch, "compile", return_value=model.forward_step_static) as compile_fn:
            with patch.object(torch.cuda, "synchronize"):
                step = decode.build_compiled_step(model, "cpu", torch.zeros((1, 128), dtype=torch.int64))
            compile_fn.assert_called_once_with(model.forward_step_static, dynamic=True, options={"cpp_wrapper": True})
        self.assertEqual(step(torch.tensor([[1]]), 7).item(), 8)
        model.freeze_for_compile.assert_called_once()

    def test_rounds_restore_model_rng(self):
        seen = []

        def measure(settings, rng_state, prompt):
            method = settings["method"]
            torch.set_rng_state(torch.tensor(rng_state["cpu"], dtype=torch.uint8))
            seen.append((method, torch.rand(4)))
            row = self.row(2.0 if method == "global_moc" else 3.0, 100)
            row.update(row=method, ffn_kind=method)
            return row, {"cpu": torch.get_rng_state().tolist(), "cuda": None}

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / "decode.json"
            stack.enter_context(patch.object(sys, "argv", ["benchmark_decode.py", "--out", str(output)]))
            for name, value in [("is_available", True), ("manual_seed_all", None)]:
                stack.enter_context(patch.object(torch.cuda, name, return_value=value))
            stack.enter_context(patch.object(decode, "load_c4_prompt", return_value=torch.zeros((1, 128), dtype=torch.int64)))
            stack.enter_context(patch.object(decode, "run_measurement", side_effect=measure))
            stack.enter_context(patch("builtins.print"))
            with torch.random.fork_rng(devices=[]):
                decode.main()
            result = json.loads(output.read_text())
        self.assertEqual([method for method, _ in seen], ["dense", "global_moc", "global_moc", "dense", "global_moc", "dense"])
        for method in ("dense", "global_moc"):
            values = [value for name, value in seen if name == method]
            self.assertTrue(all(torch.equal(values[0], value) for value in values))
        self.assertEqual(result["run_config"]["rounds"], 3)
        self.assertTrue(result["run_config"]["cpp_wrapper"])
        self.assertTrue(result["run_config"]["isolated_process_per_round"])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from moc.inference import optimized_global_moc_ops as ops

        ops.ensure_native_ops_ready()
        cls.ops = ops
        cls.extension = ops.load_optimized_global_moc_extension()

    def test_selected_projection_bits(self):
        torch.manual_seed(9123)
        shapes = [(1, 2048, 5464, 1024), (128, 2048, 5464, 1024),
                  (2, 1024, 2736, 512), (1, 513, 1376, 256),
                  (1, 2048, 5464, 1000), (1, 2048, 5464, 1536)]
        with torch.no_grad():
            for batch, hidden, intermediate, k in shapes:
                up = torch.randn(intermediate, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
                down = torch.randn_like(up) * 0.02
                x = torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16)
                for case in ("random", "ties", "zero"):
                    with self.subTest(shape=(batch, hidden, intermediate, k), case=case):
                        gate = torch.randn(batch, intermediate, device="cuda", dtype=torch.bfloat16)
                        if case == "ties":
                            gate.copy_(torch.randint(-2, 3, gate.shape, device="cuda"))
                        if case == "zero":
                            gate.zero_()
                        values, indices = self.extension.cub_topk_bf16_512x11(gate, k)
                        z = self.extension.selected_up_silu_bf16(x, values, indices, up)
                        expected = self.extension.selected_down_bf16_h32_k16(z, indices, down)
                        actual = self.ops.optimized_global_after_gate_bf16(x, gate, up, down, k)
                        self.assertTrue(torch.equal(expected.view(torch.int16), actual.view(torch.int16)))

    def test_empty_batch(self):
        x = torch.empty(0, 64, device="cuda", dtype=torch.bfloat16)
        gate = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.empty(128, 64, device="cuda", dtype=torch.bfloat16)
        result = self.ops.optimized_global_after_gate_bf16(x, gate, weight, weight, 32)
        self.assertEqual(result.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
