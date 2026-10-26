"""Portable deterministic workflow tests; no torch/GPU required."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from jinjiang.data import PreparedDataset, compute_split_boundaries
from jinjiang.models import PredictionResult
from jinjiang.window_horizon_random import RandomWindowRunner
from jinjiang.window_training_diagnostics import (
    TASKS, TrainingDiagnosticRunner, task_plan, validate_diagnostic_config,
    verify_prefix, replace_tasks,
)


def fixture():
    index = pd.date_range("2022-01-01", periods=240, freq="4h")
    i = np.arange(len(index))
    frame = pd.DataFrame({"A_DO": 8 + i / 1000, "B_DO": 9 + i / 1000,
                          "C_CODMN": 1 + i / 1000, "C_TP": .01 + i / 100000}, index=index)
    return PreparedDataset(frame.copy(), frame.copy(),
        {s: [c for c in frame if c.startswith(s + "_")] for s in "ABC"},
        {"A": {"DO": "A_DO"}, "B": {"DO": "B_DO"}, "C": {"CODMN": "C_CODMN", "TP": "C_TP"}},
        [], compute_split_boundaries(index, .7, .15), pd.DataFrame())


def fake_fit(architecture, x, y, split, seed, training):
    target = "CODMN" if y.mean() > .5 else "TP"
    cap = (target, x.shape[1], seed) in {(t, w, s) for t, w, s, _ in TASKS["cap"]}
    history, stale, best, best_epoch = [], 0, float("inf"), 0
    for epoch in range(1, training["max_epochs"] + 1):
        if cap:
            loss = 2 / (min(epoch, 1026) + 2)
        elif epoch <= 10:
            loss = 2 / (epoch + 2)
        elif epoch < 50:
            loss = 2 / 12
        else:
            loss = .1
        history.append(dict(epoch=epoch, train_loss=loss * .5, validation_loss=loss))
        if loss < best - 1e-8:
            best, best_epoch, stale = loss, epoch, 0
        else:
            stale += 1
            if stale >= training["patience"]:
                break
    return PredictionResult(y[split["test"]] + best / 100, history, best_epoch, 58002)


class PureTests(unittest.TestCase):
    def test_task_plans_are_exact_and_do_not_expand_cartesian_products(self):
        for check in ["cap", "patience"]:
            plan = task_plan(check)
            self.assertEqual(len(plan), 7)
            self.assertEqual(plan.duplicated(["target", "lookback_steps", "seed"]).sum(), 0)
        self.assertEqual(sum(task_plan("patience").diagnostic_role == "higher_validation_error"), 4)

    def test_prefix_accepts_exact_extension(self):
        old = pd.DataFrame(dict(epoch=[1, 2], train_loss=[2., 1.], validation_loss=[3., 2.]))
        new = pd.concat([old, pd.DataFrame(dict(epoch=[3], train_loss=[.5], validation_loss=[1.]))])
        self.assertEqual(verify_prefix(old, new)["shared_epochs"], 2)

    def test_prefix_rejects_changed_losses_missing_epochs_and_premature_stop(self):
        old = pd.DataFrame(dict(epoch=[1, 2], train_loss=[2., 1.], validation_loss=[3., 2.]))
        changed = old.copy(); changed.loc[0, "validation_loss"] += .1
        for new in [changed, old.iloc[:1], old.iloc[1:]]:
            with self.assertRaises(ValueError):
                verify_prefix(old, new)

    def test_smoke_prefix_can_stop_after_two_epochs(self):
        old = pd.DataFrame(dict(epoch=[1, 2, 3], train_loss=[3., 2., 1.], validation_loss=[4., 3., 2.]))
        self.assertTrue(verify_prefix(old, old.iloc[:2], smoke=True)["prefix_matches"])

    def test_task_replacement_uses_semantic_identity_and_preserves_unrelated_rows(self):
        a = pd.DataFrame([dict(target="TP", model="lstm", lookback_steps=54, seed=7, value=1),
                          dict(target="TP", model="lstm", lookback_steps=66, seed=7, value=2)])
        b = a.iloc[:1].copy(); b["value"] = 99
        merged = replace_tasks(a, b)
        self.assertEqual(sorted(merged.value), [2, 99])


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.store = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.store.name)
        cls.template = json.loads((cls.root / "configs/window_training_cap_gpu.json").read_text())
        cls.reference_config = deepcopy(cls.template)
        for key in ["diagnostic_check", "diagnostic_protocol", "baseline_results_dir"]:
            cls.reference_config.pop(key)
        cls.reference_config["_project_root"] = str(cls.root)
        cls.reference_config["training"].update(max_epochs=1024, patience=30)
        cls.reference_config["results_dir"] = str(cls.folder / "reference")
        with redirect_stdout(io.StringIO()), patch("jinjiang.runner.prepare_dataset", return_value=fixture()), \
             patch.object(RandomWindowRunner, "_environment_manifest", return_value={}), \
             patch("jinjiang.window_horizon_random.render_figures"), \
             patch("jinjiang.window_horizon_random.fit_torch_predict", side_effect=fake_fit):
            cls.reference = RandomWindowRunner(cls.reference_config).run()

    @classmethod
    def tearDownClass(cls):
        cls.store.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = deepcopy(self.template)
        self.config.update(_project_root=str(self.root), baseline_results_dir=str(self.reference), results_dir=self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_check(self, config=None, **kwargs):
        with redirect_stdout(io.StringIO()), patch("jinjiang.runner.prepare_dataset", return_value=fixture()), \
             patch.object(RandomWindowRunner, "_environment_manifest", return_value={}), \
             patch("jinjiang.window_horizon_random.fit_torch_predict", side_effect=fake_fit) as fitted:
            out = TrainingDiagnosticRunner(config or self.config, **kwargs).run()
        return out, fitted.call_count

    def test_cap_runs_exactly_seven_lstm_tasks_and_merges_full_grid(self):
        out, calls = self.run_check()
        self.assertEqual(calls, 7)
        self.assertEqual(json.loads((out / "run_status.json").read_text())["run_status"], "complete")
        self.assertEqual(len(pd.read_csv(out / "metrics.csv")), 7 * 18)
        merged = out / "cap_merged_full_grid"
        self.assertEqual(pd.read_csv(merged / "metrics.csv").task_id.nunique(), 170)
        self.assertEqual(len(pd.read_csv(merged / "metrics_mean_sd.csv")), 612)
        self.assertEqual(len(pd.read_csv(merged / "validation_window_selection.csv")), 38)
        self.assertEqual(len(pd.read_csv(out / "paired_task_summary.csv")), 7)

    def test_patience_runs_only_seven_tasks_and_never_publishes_partial_window_ranking(self):
        self.config.update(diagnostic_check="patience")
        self.config["training"]["patience"] = 100
        out, calls = self.run_check()
        self.assertEqual(calls, 7)
        self.assertFalse((out / "cap_merged_full_grid").exists())
        self.assertFalse((out / "validation_window_selection.csv").exists())
        paired = pd.read_csv(out / "paired_task_summary.csv")
        self.assertTrue(paired.best_after_original_stop.all())

    def test_audit_does_not_train(self):
        out, calls = self.run_check(max_tasks=0)
        self.assertEqual(calls, 0)
        self.assertEqual(len(pd.read_csv(out / "sample_manifest.csv")), (240 - 84 - 18 + 1) * 2)

    def test_partial_resume_skips_completed_tasks_and_rebuilds_summary(self):
        out, calls = self.run_check(max_tasks=2)
        self.assertEqual(calls, 2)
        self.assertFalse((out / "cap_merged_full_grid").exists())
        _, calls = self.run_check()
        self.assertEqual(calls, 5)
        _, calls = self.run_check()
        self.assertEqual(calls, 0)

    def test_missing_diagnostic_output_reopens_only_that_task(self):
        out, _ = self.run_check()
        d = pd.read_csv(out / "model_diagnostics.csv")
        d.iloc[:-1].to_csv(out / "model_diagnostics.csv", index=False)
        _, calls = self.run_check()
        self.assertEqual(calls, 1)
        self.assertEqual(len(pd.read_csv(out / "metrics.csv")), 126)

    def test_reference_mutation_is_rejected_on_resume(self):
        self.run_check(max_tasks=1)
        path = self.reference / "run_status.json"
        original = path.read_bytes()
        try:
            value = json.loads(original); value["unexpected"] = True
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "fingerprints"):
                self.run_check()
        finally:
            path.write_bytes(original)

    def test_source_drift_rejected_before_training(self):
        path = self.reference / "source_fingerprints.json"
        original = path.read_bytes()
        try:
            value = json.loads(original); value["models.py"] = "changed"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "source_fingerprints"):
                self.run_check()
        finally:
            path.write_bytes(original)

    def test_changed_data_or_split_support_is_rejected(self):
        path = self.reference / "sample_manifest.csv"
        original = path.read_bytes()
        try:
            manifest = pd.read_csv(path)
            manifest.loc[0, "issue_time"] = "2099-01-01 00:00:00"
            manifest.to_csv(path, index=False)
            with self.assertRaises(AssertionError):
                self.run_check()
        finally:
            path.write_bytes(original)

    def test_smoke_runs_one_task_two_epochs_without_full_grid_merge(self):
        self.config["smoke_test"] = True
        self.config["training"]["max_epochs"] = 2
        out, calls = self.run_check()
        self.assertEqual(calls, 1)
        self.assertEqual(pd.read_csv(out / "model_diagnostics.csv").epochs_run.iloc[0], 2)
        self.assertFalse((out / "cap_merged_full_grid").exists())

    def test_wrong_budget_or_protocol_rejected(self):
        for field, value in [("max_epochs", 1024), ("patience", 100)]:
            config = deepcopy(self.config); config["training"][field] = value
            with self.assertRaises(ValueError):
                validate_diagnostic_config(config)
        config = deepcopy(self.config); config["split_seed"] = 7
        with self.assertRaises(ValueError):
            validate_diagnostic_config(config)

    def test_ctrl_c_then_resume_restarts_only_unfinished_training(self):
        count = [0]
        def interrupted(*args, **kwargs):
            count[0] += 1
            if count[0] == 2:
                raise KeyboardInterrupt()
            return fake_fit(*args, **kwargs)
        with redirect_stdout(io.StringIO()), patch("jinjiang.runner.prepare_dataset", return_value=fixture()), \
             patch.object(RandomWindowRunner, "_environment_manifest", return_value={}), \
             patch("jinjiang.window_horizon_random.fit_torch_predict", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                TrainingDiagnosticRunner(self.config).run()
        out = Path(self.temp.name) / "03_window_horizon"
        self.assertEqual(json.loads((out / "run_status.json").read_text())["completed_tasks"], 1)
        _, calls = self.run_check()
        self.assertEqual(calls, 6)

    def test_loss_prefix_drift_is_not_marked_complete(self):
        def changed(*args, **kwargs):
            result = fake_fit(*args, **kwargs)
            result.history[0]["validation_loss"] += .1
            return result
        with redirect_stdout(io.StringIO()), patch("jinjiang.runner.prepare_dataset", return_value=fixture()), \
             patch.object(RandomWindowRunner, "_environment_manifest", return_value={}), \
             patch("jinjiang.window_horizon_random.fit_torch_predict", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "prefix differs"):
                TrainingDiagnosticRunner(self.config).run()
        out = Path(self.temp.name) / "03_window_horizon"
        status = json.loads((out / "run_status.json").read_text())
        self.assertEqual(status["completed_tasks"], 0)
        self.assertEqual(status["unresolved_failed_tasks"], 1)

    def test_unintended_learning_rate_change_is_rejected(self):
        self.config["training"]["learning_rate"] = .002
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            self.run_check()

    def test_merge_failure_is_not_reported_as_complete(self):
        with patch.object(TrainingDiagnosticRunner, "_merge_cap", side_effect=ValueError("merge failed")):
            with self.assertRaisesRegex(ValueError, "merge failed"):
                self.run_check()
        out = Path(self.temp.name) / "03_window_horizon"
        self.assertEqual(json.loads((out / "run_status.json").read_text())["run_status"], "partial")
        _, calls = self.run_check()
        self.assertEqual(calls, 0)
        self.assertEqual(json.loads((out / "run_status.json").read_text())["run_status"], "complete")


if __name__ == "__main__":
    unittest.main()
