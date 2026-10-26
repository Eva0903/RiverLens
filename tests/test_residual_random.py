"""Matched-backbone, validation isolation, twenty-fit counting and atomic resume."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from jinjiang.data import PreparedDataset, compute_split_boundaries
from jinjiang.evaluation import calculate_metric_rows, prediction_frame
from jinjiang.residual_random import (
    FEATURES, OUTPUTS, RandomResidualRunner, fit_residual_once, validate_config,
)
from jinjiang.station_rotation_random import common_station_pool, FIXED_WINDOWS


def config(smoke=False):
    name = "residual_random_smoke_gpu.json" if smoke else "residual_random_gpu.json"
    return json.loads((Path(__file__).resolve().parents[1] / "configs" / name).read_text(encoding="utf-8"))


def fixture():
    index = pd.date_range("2022-01-01", periods=210, freq="4h")
    x = np.arange(len(index), dtype=np.float32) / 100
    frame = pd.DataFrame({f"{s}_{t}": x + j + k / 10
                          for j, s in enumerate("ABC") for k, t in enumerate(FIXED_WINDOWS)}, index=index)
    frame["W_TEM"], frame["W_RAIN_LAG_4H"] = x + 3, x + 4
    return PreparedDataset(frame.copy(), frame.copy(),
        {s: [f"{s}_{t}" for t in FIXED_WINDOWS] for s in "ABC"},
        {s: {t: f"{s}_{t}" for t in FIXED_WINDOWS} for s in "ABC"},
        ["W_TEM", "W_RAIN_LAG_4H"], compute_split_boundaries(index, .8, .1), pd.DataFrame())


def fake_reference(runner):
    metrics, predictions = [], []
    for target in runner.config["targets"]:
        seq, split, arms = common_station_pool(runner.data, runner.config, "C", target)
        for seed in runner.config["training"]["seeds"]:
            metadata = dict(task_id=f"old04|{target}|{seed}", experiment="04_meteorology", subexperiment="old",
                fold=0, split_scheme=runner.config["baseline_protocol"], target_station="C", target=target,
                feature_set=FEATURES, model="lstm", seed=seed, lookback_steps=FIXED_WINDOWS[target],
                lookback_days=FIXED_WINDOWS[target]/6, best_epoch=2, parameter_count=100,
                result_origin="reused_04", n_test=len(split["test"]))
            prediction = seq.y[split["test"]] + .2
            metrics.extend(calculate_metric_rows(seq.y[split["test"]], prediction, metadata))
            predictions.append(prediction_frame(seq.target_start_times[split["test"]], seq.y[split["test"]], prediction, metadata))
    runner.reused_metrics = pd.DataFrame(metrics)
    runner.reused_predictions = pd.concat(predictions, ignore_index=True)
    runner.reused_diagnostics = runner.reused_metrics.drop_duplicates("task_id")
    runner.reference_verified = True


def fake_fit(seq, split, seed, training):
    truth = seq.y[split["test"]]
    forecasts = dict(persistence=seq.persistence[split["test"]], seasonal_naive=seq.seasonal_naive[split["test"]],
        validation_selected_naive=seq.persistence[split["test"]], residual_lstm=truth+.02, gated_residual_lstm=truth+.01)
    history = [dict(epoch=1, train_loss=.3, validation_loss=.4), dict(epoch=2, train_loss=.2, validation_loss=.3)]
    gates = pd.DataFrame(dict(horizon_step=np.arange(1, 7), horizon_hours=np.arange(1, 7)*4,
        selected_baseline=["persistence"]*6, gate=[.5]*6, validation_naive_MAE=[.1]*6,
        validation_residual_MAE=[.08]*6, validation_gated_MAE=[.05]*6))
    return forecasts, history, 2, 100, gates


class ProtocolTests(unittest.TestCase):
    def test_fixed_twenty_fits_and_no_old_upgrade_knobs(self):
        c = config(); validate_config(c)
        self.assertEqual(len(c["targets"])*len(c["training"]["seeds"]), 20)
        for key, value in [("target_station", "A"), ("gate_grid_size", 31), ("refit_train_validation", True),
                           ("append_calendar_features", True), ("split_seed", 99)]:
            bad = deepcopy(c); bad[key] = value
            with self.assertRaises(ValueError): validate_config(bad)
        for key, value in [("max_epochs", 2048), ("weight_decay", .0001), ("hidden_size", 32)]:
            bad = deepcopy(c); bad["training"][key] = value
            with self.assertRaises(ValueError): validate_config(bad)
        validate_config(config(True))
        bad = config(True); bad["results_dir"] = c["results_dir"]
        with self.assertRaises(ValueError): validate_config(bad)

    def test_one_core_call_validation_gates_and_test_label_isolation(self):
        c = config()
        seq, split, arms = common_station_pool(fixture(), c, "C", "DO")
        seq = arms[FEATURES]
        calls = []
        def core(architecture, x, residual_y, inference_split, seed, training):
            self.assertEqual(architecture, "lstm")
            np.testing.assert_array_equal(inference_split["train"], split["train"])
            np.testing.assert_array_equal(inference_split["validation"], split["validation"])
            np.testing.assert_array_equal(inference_split["test"], np.r_[split["validation"],split["test"]])
            self.assertFalse(training["compute_shap"])
            calls.append(residual_y.copy())
            # Wrong-signed validation residuals must switch the gate to zero.
            prediction = np.full((len(inference_split["test"]),6), -.1, dtype=np.float32)
            return SimpleNamespace(prediction=prediction,history=[dict(epoch=1,train_loss=.2,validation_loss=.3)],best_epoch=1,parameter_count=100)
        with patch("jinjiang.residual_random.fit_torch_predict", side_effect=core):
            forecast, _, _, _, gates = fit_residual_once(seq, split, 42, c["training"])
            altered = deepcopy(seq); altered.y[split["test"]] += 1000
            forecast2, _, _, _, gates2 = fit_residual_once(altered, split, 42, c["training"])
        self.assertEqual(len(calls), 2)  # one model fit per invocation, not one per forecast
        np.testing.assert_array_equal(gates.gate, np.zeros(6))
        pd.testing.assert_frame_equal(gates, gates2)
        for name in OUTPUTS:
            np.testing.assert_array_equal(forecast[name],forecast2[name])
        np.testing.assert_array_equal(forecast["gated_residual_lstm"],forecast["validation_selected_naive"])
        np.testing.assert_array_equal(calls[0][split["train"]],calls[1][split["train"]])
        np.testing.assert_array_equal(calls[0][split["validation"]],calls[1][split["validation"]])


class WorkflowTests(unittest.TestCase):
    def test_twenty_fit_resume_commit_repair_and_matched_comparisons(self):
        calls = []
        def fit(*args):
            calls.append(args[2]); return fake_fit(*args)
        with tempfile.TemporaryDirectory() as temp:
            c = config(); c.update(_project_root=temp, data_dir="unused", results_dir="results")
            with patch("jinjiang.residual_random.prepare_dataset", return_value=fixture()), \
                 patch("jinjiang.residual_random.prepare_random_pool", side_effect=lambda d,c:d), \
                 patch.object(RandomResidualRunner, "_load_reference", fake_reference), \
                 patch("jinjiang.residual_random.fit_residual_once", side_effect=fit), redirect_stdout(io.StringIO()):
                out = RandomResidualRunner(deepcopy(c),max_tasks=3).run()
                self.assertEqual(len(calls),3)
                partial = pd.read_csv(out/"residual_gain_summary.csv")
                self.assertTrue((partial.status=="partial_paired_seeds").all())
                RandomResidualRunner(deepcopy(c)).run()
                self.assertEqual(len(calls),20)
                self.assertEqual(len(pd.read_csv(out/"metrics.csv")),600)
                self.assertEqual(len(pd.read_csv(out/"combined_metrics.csv")),720)
                summary = pd.read_csv(out/"residual_gain_summary.csv")
                self.assertEqual(len(summary),20)
                self.assertTrue((summary.n_paired_seeds==5).all())
                np.testing.assert_allclose(summary[summary.contrast=="gated_vs_lstm"].MAE_reduction_percent,95,rtol=1e-4)
                self.assertEqual(len(pd.read_csv(out/"gate_diagnostics.csv")),120)
                self.assertEqual(len(pd.read_csv(out/"model_diagnostics.csv")),20)
                RandomResidualRunner(deepcopy(c)).run()
                self.assertEqual(len(calls),20)
                # Completion marker alone cannot hide truncated forecast rows.
                file = out/"tasks/C_DO_seed42/predictions.csv"
                frame=pd.read_csv(file);frame.iloc[:-1].to_csv(file,index=False)
                RandomResidualRunner(deepcopy(c)).run()
                self.assertEqual(len(calls),21)
                status=json.loads((out/"run_status.json").read_text())
                self.assertEqual(status["run_status"],"complete")
                self.assertEqual(status["completed_new_lstm_fits"],20)
                self.assertEqual(status["reused_ordinary_lstm_fits"],20)
                # No marker means no complete task, even if CSV files survived.
                (out/"tasks/C_TP_seed99/commit.json").unlink()
                RandomResidualRunner(deepcopy(c)).run()
                self.assertEqual(len(calls),22)

    def test_parameter_count_mismatch_refuses_completion(self):
        def mismatch(*args):
            f,h,e,p,g=fake_fit(*args); return f,h,e,p+1,g
        with tempfile.TemporaryDirectory() as temp:
            c=config();c.update(_project_root=temp,data_dir="unused",results_dir="results")
            with patch("jinjiang.residual_random.prepare_dataset", return_value=fixture()), \
                 patch("jinjiang.residual_random.prepare_random_pool", side_effect=lambda d,c:d), \
                 patch.object(RandomResidualRunner,"_load_reference",fake_reference), \
                 patch("jinjiang.residual_random.fit_residual_once",side_effect=mismatch), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,"parameter counts"):
                    RandomResidualRunner(c,max_tasks=1).run()
                status=json.loads((Path(temp)/"results/06_residual_random/run_status.json").read_text())
                self.assertEqual(status["completed_new_lstm_fits"],0)
                self.assertEqual(status["unresolved_failed_tasks"],1)

    def test_configuration_change_blocks_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            c=config(True);c.update(_project_root=temp,data_dir="unused",results_dir="smoke")
            with patch("jinjiang.residual_random.prepare_dataset", return_value=fixture()), \
                 patch("jinjiang.residual_random.prepare_random_pool", side_effect=lambda d,c:d), redirect_stdout(io.StringIO()):
                RandomResidualRunner(deepcopy(c),max_tasks=0).run()
                c["project_name"] += " changed"
                with self.assertRaisesRegex(RuntimeError,"run_config.json"):
                    RandomResidualRunner(c,max_tasks=0)


if __name__ == "__main__":
    unittest.main()
