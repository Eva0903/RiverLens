"""Protocol, station labels, pairing and interruption recovery without a GPU."""
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
from sklearn.model_selection import train_test_split

from jinjiang.data import PreparedDataset, compute_split_boundaries
from jinjiang.station_rotation_random import (
    ARMS, FIXED_WINDOWS, SEEDS, RandomStationRotationRunner,
    common_station_pool, validate_config, validate_reference_config, verify_task_tables,
)


def config():
    path = Path(__file__).resolve().parents[1] / "configs/station_rotation_random_gpu.json"
    return json.loads(path.read_text(encoding="utf-8"))


def fixture(missing=False):
    index = pd.date_range("2022-01-01", periods=240, freq="4h")
    i = np.arange(len(index), dtype=float)
    frame = pd.DataFrame({f"{s}_{t}": i / 100 + j + k / 10
                          for j, s in enumerate("ABC") for k, t in enumerate(FIXED_WINDOWS)}, index=index)
    frame["W_TEM"] = i / 100 + 3
    frame["W_RAIN_LAG_4H"] = i / 100 + 4
    if missing:
        frame.loc[index[120], "W_TEM"] = np.nan
    return PreparedDataset(frame.copy(), frame.copy(),
        {s: [f"{s}_{t}" for t in FIXED_WINDOWS] for s in "ABC"},
        {s: {t: f"{s}_{t}" for t in FIXED_WINDOWS} for s in "ABC"},
        ["W_TEM", "W_RAIN_LAG_4H"], compute_split_boundaries(index, .8, .1), pd.DataFrame())


class ProtocolTests(unittest.TestCase):
    def test_exactly_80_new_fits_and_fixed_settings(self):
        c = config(); validate_config(c)
        self.assertEqual(len(c["target_stations"]) * len(c["targets"]) * len(ARMS) * len(SEEDS), 80)
        for change in ["budget", "window", "split", "station", "seed"]:
            c = config()
            if change == "budget": c["training"]["max_epochs"] = 2048
            elif change == "window": c["target_default_windows"]["DO"] = 18
            elif change == "split": c["split"]["train_fraction"] = .7
            elif change == "station": c["target_stations"].append("C")
            else: c["training"]["seeds"] = [42]
            with self.assertRaises(ValueError): validate_config(c)

    def test_union_mask_station_target_and_exact_legacy_split(self):
        data = fixture(True)
        for station in "ABC":
            seq, split, arms = common_station_pool(data, config(), station, "DO")
            self.assertEqual(seq.target_column, f"{station}_DO")
            target_positions = data.raw_frame.index.get_indexer(seq.target_start_times)
            np.testing.assert_allclose(seq.y[:, 0], data.raw_frame[f"{station}_DO"].iloc[target_positions])
            train, temp = train_test_split(np.arange(len(seq.x)), test_size=.2, random_state=42)
            validation, test = train_test_split(temp, test_size=.5, random_state=42)
            for name, expected in [("train", train), ("validation", validation), ("test", test)]:
                np.testing.assert_array_equal(split[name], expected)
            for arm in arms.values():
                np.testing.assert_array_equal(arm.y, seq.y)
                np.testing.assert_array_equal(arm.split_labels, seq.split_labels)
                np.testing.assert_array_equal(arm.target_start_times, seq.target_start_times)
                self.assertTrue(np.isfinite(arm.x).all())
            self.assertTrue((seq.issue_times < seq.target_start_times).all())
            self.assertEqual([len(a.feature_columns) for a in arms.values()], [12, 14])
        self.assertLess(len(common_station_pool(data, config(), "A", "DO")[0].x),
                        len(common_station_pool(fixture(), config(), "A", "DO")[0].x))

    def test_reference_config_rejects_different_budget_and_preprocessing(self):
        c = config()
        ref = deepcopy(c)
        ref.update(target_station="C", meteorology_model="lstm", meteorology_protocol="meteorology_random_fixed_budget_v1")
        validate_reference_config(c, ref)
        for section, key, value in [("training", "patience", 50), ("missing_data", "max_water_gap_steps", 2)]:
            bad = deepcopy(ref); bad[section][key] = value
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                validate_reference_config(c, bad)

    def test_smoke_is_separate_and_cannot_be_formal(self):
        c = config()
        c.update(smoke_only=True, target_stations=["A"], targets=["DO"], results_dir="results/smoke")
        c["training"].update(seeds=[42], max_epochs=2, patience=1)
        validate_config(c)
        c["results_dir"] = "results/station_rotation_random_gpu"
        with self.assertRaises(ValueError): validate_config(c)


class WorkflowTests(unittest.TestCase):
    def test_80_fit_resume_and_repair_of_truncated_completed_task(self):
        calls = []
        def fit(runner, method, x, sequence, split, seed):
            calls.append((sequence.target_column, method.feature_set, seed))
            history = [dict(epoch=1, train_loss=.3, validation_loss=.4),
                       dict(epoch=2, train_loss=.2, validation_loss=.3)]
            delta = .02 if method.feature_set == ARMS[0] else .01
            return sequence.y[split["test"]] + delta, history, 2, 100, None
        def reference(runner):
            # Reference validation itself is covered by the real-data audit.
            runner.reference_verified = True
        with tempfile.TemporaryDirectory() as temp:
            c = config(); c.update(_project_root=temp, data_dir="unused", results_dir="results")
            with patch("jinjiang.runner.prepare_dataset", return_value=fixture()), \
                 patch("jinjiang.station_rotation_random.prepare_random_pool", side_effect=lambda data, cfg: data), \
                 patch.object(RandomStationRotationRunner, "_load_reference", reference), \
                 patch.object(RandomStationRotationRunner, "_fit_method", fit), redirect_stdout(io.StringIO()):
                r = RandomStationRotationRunner(deepcopy(c), max_tasks=3); out = r.run()
                self.assertEqual(len(calls), 3)
                self.assertEqual(json.loads((out / "run_status.json").read_text())["run_status"], "partial")
                RandomStationRotationRunner(deepcopy(c)).run()
                self.assertEqual(len(calls), 80)
                self.assertTrue(all(task[0][0] in "AB" for task in calls))
                m = pd.read_csv(out / "metrics.csv")
                p = pd.read_csv(out / "predictions.csv")
                h = pd.read_csv(out / "training_history.csv")
                self.assertEqual(len(m), 480)
                self.assertEqual(len(verify_task_tables(m, p, h)), 80)
                self.assertFalse(m.duplicated(["task_id", "horizon_step"]).any())
                summary = pd.read_csv(out / "station_rotation_gain_summary.csv")
                self.assertEqual(len(summary), 8)
                self.assertTrue((summary.n_paired_seeds == 5).all())
                np.testing.assert_allclose(summary.MAE_reduction_percent, 50)
                RandomStationRotationRunner(deepcopy(c)).run()
                self.assertEqual(len(calls), 80)
                # Simulate a CSV losing its last lead despite a completion marker.
                m.iloc[:-1].to_csv(out / "metrics.csv", index=False)
                RandomStationRotationRunner(deepcopy(c)).run()
                self.assertEqual(len(calls), 81)
                self.assertEqual(len(pd.read_csv(out / "metrics.csv")), 480)
                # An interrupted uncommitted task must not survive repair.
                stray = pd.read_csv(out / "metrics.csv").iloc[:1].copy(); stray["task_id"] = "uncommitted"
                stray.to_csv(out / "metrics.csv", mode="a", header=False, index=False)
                RandomStationRotationRunner(deepcopy(c), max_tasks=0).run()
                self.assertEqual(len(pd.read_csv(out / "metrics.csv")), 480)
                self.assertEqual(len(calls), 81)
                status = json.loads((out / "run_status.json").read_text())
                self.assertEqual(status["run_status"], "complete")
                self.assertEqual(status["expected_new_lstm_tasks"], 80)

    def test_data_fingerprint_change_blocks_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            c = config(); c.update(_project_root=temp, data_dir="unused", results_dir="results")
            with patch("jinjiang.runner.prepare_dataset", return_value=fixture()), \
                 patch("jinjiang.station_rotation_random.prepare_random_pool", side_effect=lambda data, cfg: data), \
                 patch.object(RandomStationRotationRunner, "_load_reference"), redirect_stdout(io.StringIO()):
                runner = RandomStationRotationRunner(deepcopy(c), max_tasks=0)
                runner.run()
                (Path(temp) / "results/05_robustness/data_fingerprints.json").write_text('{"changed": "data"}')
                with self.assertRaisesRegex(RuntimeError, "data_fingerprints"):
                    RandomStationRotationRunner(deepcopy(c), max_tasks=0)


if __name__ == "__main__":
    unittest.main()
