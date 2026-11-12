import copy
import unittest
from unittest.mock import Mock, patch

from benchmarks.inference import benchmark_ffn_latency as bench


class FFNLatencyTests(unittest.TestCase):
    def test_decode_path_and_defaults(self):
        args = bench.parse_args(["--out", "unused.json"])
        self.assertEqual(args.methods, ["dense", "global_moc"])
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(bench.CAPTURE_CALLS, 16)
        self.assertEqual((args.warmup, args.groups, args.replays_per_group, args.rounds), (30, 80, 7, 3))
        self.assertIs(bench.SingleFFN.forward, bench.DecoderLayer._ffn_call)

    def test_compile_matches_decode(self):
        module = Mock()
        with patch.object(bench.torch, "compile", return_value="compiled") as compile_fn:
            self.assertEqual(bench.compile_call(module), "compiled")
        module.freeze_for_compile.assert_called_once_with()
        compile_fn.assert_called_once_with(module.forward, dynamic=True, options={"cpp_wrapper": True})

    def test_optional_grouped_path(self):
        args = bench.parse_args(["--out", "unused.json", "--methods", "moc_2_8", "--smoke"])
        self.assertEqual(args.methods, ["moc_2_8"])
        self.assertEqual((args.groups, args.replays_per_group, args.rounds), (3, 2, 1))

    def test_argument_validation(self):
        for extra in (["--groups", "0"], ["--rounds", "0"], ["--warmup", "-1"], ["--batch-size", "0"],
                      ["--replays-per-group", "0"],
                      ["--methods", "dense", "dense"]):
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                bench.parse_args(["--out", "unused.json", *extra])

    def test_exact_check_uses_bits(self):
        value = bench.torch.tensor([0.0, 1.0], dtype=bench.torch.bfloat16)
        bench.assert_exact(value, value.clone())
        with self.assertRaises(AssertionError):
            bench.assert_exact(value, bench.torch.tensor([-0.0, 1.0], dtype=value.dtype))

    def test_profile_window_handles_cpu_and_gpu_annotations(self):
        events = [dict(name="ffn_profile_measure", cat="user_annotation", ph="X", ts=100, dur=100),
                  dict(name="ffn_profile_measure", cat="gpu_user_annotation", ph="X", ts=120, dur=60),
                  dict(name="warmup", cat="kernel", ts=90, dur=5),
                  dict(name="first", cat="kernel", ts=110, dur=5),
                  dict(name="last", cat="kernel", ts=190, dur=5)]
        begin, end, kernels = bench.profile_window_kernels(events)
        self.assertEqual((begin, end), (100, 200))
        self.assertEqual([e["name"] for e in kernels], ["first", "last"])

    def fixture(self):
        def row(samples):
            return dict(latency=dict(all_us=samples), input_sha256="input",
                        parameter_sha256={"weight": "weight"})
        return [dict(dense=row([10, 12]), global_moc=row([8, 9])),
                dict(global_moc=row([7, 16]), dense=row([12, 14]))]

    def test_all_samples_aggregate(self):
        rows = bench.aggregate(self.fixture())
        self.assertEqual(rows["dense"]["latency"]["mean_us"], 12)
        self.assertEqual(rows["global_moc"]["latency"]["mean_us"], 10)
        self.assertEqual(rows["global_moc"]["speedup_vs_dense"], 1.2)
        self.assertEqual(rows["global_moc"]["latency"]["all_us"], [8, 9, 7, 16])

    def test_missing_method_or_different_inputs_fail(self):
        original = self.fixture()
        missing = copy.deepcopy(original)
        del missing[1]["dense"]
        different = copy.deepcopy(original)
        different[1]["dense"]["input_sha256"] = "different"
        for invalid in (missing, different):
            with self.assertRaises(ValueError):
                bench.aggregate(invalid)


@unittest.skipUnless(bench.torch.cuda.is_available(), "CUDA required")
class FFNGraphTests(unittest.TestCase):
    def test_capture_replays_with_changed_input(self):
        x = bench.torch.randn(2, 1, 64, device="cuda", dtype=bench.torch.bfloat16)
        fn = bench.torch.compile(lambda value: value * 2, dynamic=True, options={"cpp_wrapper": True})
        graph, output = bench.capture_call(fn, x)
        for _ in range(4):
            x.normal_()
            expected = fn(x).clone()
            graph.replay()
            bench.torch.cuda.synchronize()
            bench.assert_exact(output, expected)


if __name__ == "__main__":
    unittest.main()
