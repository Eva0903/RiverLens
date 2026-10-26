"""Regression: cache scaling must equal the original shared trainer, not relax it."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from jinjiang.interpretation_precision_fix import (
    VERSION, fixed_inference, load_predictor_fixed, prediction_diagnostics,
    restore_scaler, source_inverse_outputs, source_scaled_test,
)
from jinjiang.models import _scale_x


class ArithmeticTests(unittest.TestCase):
    def test_matches_original_scale_values_dtype_and_tensor_layout(self):
        rng = np.random.default_rng(2026)
        x = (rng.normal(size=(60, 12, 8)) * 2 + 10000).astype(np.float32)
        x = x[:, :, list(range(8))]  # same feature-list selection as common_station_pool
        train, test = np.arange(40), np.array([51, 42, 58, 48, 59, 46, 40])
        original_scaled, xs = _scale_x(x, train)
        expected = original_scaled[test]
        incoming = x[test]
        before = incoming.copy()
        fixed = source_scaled_test(incoming, restore_scaler(xs.mean_, xs.scale_))
        np.testing.assert_array_equal(fixed, expected)
        np.testing.assert_array_equal(incoming, before)
        self.assertEqual(fixed.dtype, expected.dtype)
        self.assertEqual(fixed.strides, expected.strides)
        # A C-order perturbed array must retain the same inference tensor layout.
        fixed_c = source_scaled_test(incoming.copy(order="C"), restore_scaler(xs.mean_, xs.scale_))
        np.testing.assert_array_equal(fixed_c, expected)
        self.assertEqual(fixed_c.strides, expected.strides)

    def test_matches_inverse_api_in_original_concatenate_order(self):
        rng = np.random.default_rng(4)
        y = (.003 + rng.normal(size=(100, 6)) * .0004).astype(np.float32)
        ys = StandardScaler().fit(y[:70])
        z = rng.normal(size=(23, 6)).astype(np.float32)
        expected = ys.inverse_transform(z).astype(np.float32)
        actual = source_inverse_outputs([z[:8], z[8:16], z[16:]], restore_scaler(ys.mean_, ys.scale_))
        np.testing.assert_array_equal(actual, expected)

    def test_reproduces_old_precision_defect_and_fixes_it_without_new_tolerance(self):
        rng = np.random.default_rng(11)
        x = (10000 + rng.normal(size=(80, 6, 3)) * .05).astype(np.float32)
        x = x[:, :, [0, 1, 2]]
        original, xs = _scale_x(x, np.arange(60))
        test = np.arange(60, 80)
        expected = .001 + .01 * original[test, -1, 0, None]
        legacy = x[test].astype(np.float32, copy=True)
        legacy -= xs.mean_; legacy /= xs.scale_
        legacy_prediction = .001 + .01 * legacy[:, -1, 0, None]
        fixed = source_scaled_test(x[test], restore_scaler(xs.mean_, xs.scale_))
        fixed_prediction = .001 + .01 * fixed[:, -1, 0, None]
        # Older sklearn may already match the manual operation; skip this single
        # reproduction on those versions while retaining API parity tests above.
        if np.array_equal(legacy_prediction, expected):
            self.skipTest("Installed sklearn has older scaler rounding; API parity is still checked.")
        self.assertFalse(prediction_diagnostics(legacy_prediction, expected)["passed"])
        np.testing.assert_array_equal(fixed_prediction, expected)
        self.assertTrue(prediction_diagnostics(fixed_prediction, expected)["passed"])

    def test_restoration_never_refits_and_rejects_invalid_stats(self):
        with patch.object(StandardScaler, "fit", side_effect=AssertionError("No scaler refit")):
            scaler = restore_scaler(np.array([2., 3.]), np.array([1., 2.]))
            np.testing.assert_array_equal(scaler.transform(np.array([[2., 3.]], np.float32)), [[0., 0.]])
        with self.assertRaises(ValueError):
            restore_scaler([0.], [0.])


class AdapterTests(unittest.TestCase):
    def fake_core(self, temp):
        writes = []
        def verify(actual, expected):
            if not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=1e-5, atol=1e-7):
                raise ValueError("original predictions differ")
        def csv(path, frame):
            writes.append((str(path), frame.copy()))
        core = SimpleNamespace(load_predictor=lambda *a: "old", verify_prediction=verify, csv_write=csv,
            json_write=lambda p,v: p.write_text(json.dumps(v)), __file__=__file__)
        class Runner:
            def __init__(self):
                self.output = Path(temp); self.guards = []
            def _guard(self, name, value): self.guards.append((name, value))
            def recover(self, *a): raise AssertionError("Training is forbidden in this regression")
            def run(self):
                predictor = core.load_predictor(self.output / "checkpoints/C_NH3N_seed42", "cpu")
                core.verify_prediction(predictor(None), np.ones((3, 6), np.float32))
                core.csv_write(self.output / "attributions/C_NH3N_seed42/importance.csv", pd.DataFrame({"score":[1.]}))
                core.csv_write(self.output / "importance_summary.csv", pd.DataFrame({"score":[2.]}))
                return "cached_only_complete"
        core.InterpretationRunner = Runner
        return core, writes

    def test_cached_only_adapter_diagnostics_guards_and_method_restoration(self):
        with tempfile.TemporaryDirectory() as temp:
            core, writes = self.fake_core(temp)
            original_run, original_verify, original_loader = core.InterpretationRunner.run, core.verify_prediction, core.load_predictor
            with patch("jinjiang.interpretation_precision_fix.load_predictor_fixed", return_value=lambda x: np.ones((3,6), np.float32)):
                with fixed_inference(core):
                    runner = core.InterpretationRunner()
                    self.assertEqual(runner.run(), "cached_only_complete")
                    self.assertTrue(runner.guards[0][1]["original_tolerance_unchanged"])
            self.assertIs(core.InterpretationRunner.run, original_run)
            self.assertIs(core.verify_prediction, original_verify)
            self.assertIs(core.load_predictor, original_loader)
            check = json.loads((Path(temp) / "inference_precision_checks/C_NH3N_seed42_cached_inference.json").read_text())
            self.assertTrue(check["passed"]); self.assertEqual(check["n_failed"], 0)
            self.assertEqual(writes[0][1].inference_implementation.iloc[0], VERSION)
            self.assertEqual(writes[1][1].inference_implementation.iloc[0], "legacy_manual_scaling_v1")

    def test_real_mismatch_still_raises_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            core, writes = self.fake_core(temp)
            with patch("jinjiang.interpretation_precision_fix.load_predictor_fixed", return_value=lambda x: np.ones((3,6), np.float32) + .01):
                with self.assertRaisesRegex(ValueError, "original predictions differ"):
                    with fixed_inference(core): core.InterpretationRunner().run()
            self.assertEqual(writes, [])
            check = json.loads((Path(temp) / "inference_precision_checks/C_NH3N_seed42_cached_inference.json").read_text())
            self.assertFalse(check["passed"]); self.assertEqual(check["n_failed"], 18)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Optional tiny CPU Torch checkpoint parity test")
    def test_actual_cpu_checkpoint_matches_shared_training_predictions(self):
        import torch
        from torch.nn.modules.module import register_module_forward_pre_hook
        from jinjiang.models import fit_torch_predict
        rng = np.random.default_rng(41)
        x = (rng.normal(size=(40, 12, 5)) + 40).astype(np.float32)
        x = x[:, :, list(range(5))]
        y = np.repeat((.02 + .001 * x[:, -1, 0, None]), 6, axis=1)
        split = dict(train=np.arange(25), validation=np.arange(25,32), test=np.arange(32,40))
        training = dict(hidden_size=4, num_layers=1, dropout=0., batch_size=8, max_epochs=2, patience=2,
                        learning_rate=.001, device="cpu", deterministic="strict")
        captured = {}
        def observe(module, args):
            if type(module).__name__ == "Recurrent": captured["model"] = module
        handle = register_module_forward_pre_hook(observe)
        try: result = fit_torch_predict("lstm", x, y, split, 42, training)
        finally: handle.remove()
        xs = StandardScaler().fit(x[split["train"]].reshape(-1,5))
        ys = StandardScaler().fit(y[split["train"]])
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            torch.save(captured["model"].state_dict(), d / "model.pt")
            np.savez(d / "scalers.npz", x_mean=xs.mean_, x_scale=xs.scale_, y_mean=ys.mean_, y_scale=ys.scale_)
            (d / "metadata.json").write_text(json.dumps(dict(training=training, input_columns=list("abcde"),
                case=dict(parameter_count=result.parameter_count))))
            actual = load_predictor_fixed(d, "cpu")(x[split["test"]])
            np.testing.assert_allclose(actual, result.prediction, rtol=1e-5, atol=1e-7)


if __name__ == "__main__": unittest.main()
