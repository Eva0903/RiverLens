"""05: repeat the fixed 04 meteorological contrast at target stations A and B."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from .baseline_random import PROTOCOL as SPLIT_PROTOCOL, SPLITS, legacy_random_split, prepare_random_pool, random_support_tables
from .baseline_within_split import training_mase_scale
from .config import resolve_path
from .data import feature_set_columns, make_sequences
from .evaluation import calculate_metric_rows, mean_sd_table, prediction_frame
from .meteorology_random import FIXED_WINDOWS, paired_seed_gains, paired_calendar_gains
from .runner import ExperimentRunner, Method, _append_csv, _read_csv_resilient, _upsert_csv

PROTOCOL = "station_rotation_random_fixed_budget_v1"
ARMS = ("all_stations", "all_stations_weather_rain_lags")
SEEDS = [42, 2026, 3407, 7, 99]
SOURCES = ["meteorology_random.py", "baseline_random.py", "baseline_within_split.py",
           "runner.py", "data.py", "models.py", "evaluation.py", "config.py"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(c):
    if c.get("station_rotation_protocol") != PROTOCOL or c.get("baseline_protocol") != SPLIT_PROTOCOL:
        raise ValueError("Use configs/station_rotation_random_gpu.json.")
    if c.get("frequency") != "4h" or c.get("forecast_horizon_steps") != 6:
        raise ValueError("05 fixes 4h frequency and six output steps.")
    if c.get("target_default_windows") != FIXED_WINDOWS or c.get("split_seed") != 42:
        raise ValueError("Keep 03 windows DO84/NH3N78/CODMN48/TP66 and split_seed=42.")
    if (c["split"]["train_fraction"], c["split"]["validation_fraction"]) != (.8, .1):
        raise ValueError("05 uses the unchanged 80/10/10 random-window split.")
    if c["missing_data"]["interpolation_policy"] != "causal_ffill":
        raise ValueError("Use causal_ffill input preprocessing.")
    if c.get("interpretability", {}).get("enabled", False):
        raise ValueError("SHAP is disabled for this experiment.")
    t = c["training"]
    expected = dict(hidden_size=64, num_layers=2, dropout=.1, batch_size=32,
                    learning_rate=.001, device="cuda", deterministic="strict")
    if any(t.get(k) != v for k, v in expected.items()):
        raise ValueError("Keep the completed 04 LSTM architecture, optimizer and deterministic CUDA settings.")
    smoke = c.get("smoke_only", False)
    stations, targets, seeds, epochs, patience = (["A"], ["DO"], [42], 2, 1) if smoke else (
        ["A", "B"], list(FIXED_WINDOWS), SEEDS, 512, 100)
    if (c.get("target_stations"), c.get("targets"), t["seeds"], t["max_epochs"], t["patience"]) != (
            stations, targets, seeds, epochs, patience):
        raise ValueError("Formal 05 requires A/B, four targets, five seeds, 512 epochs and patience=100; use the separate smoke config for 2 epochs.")
    if smoke and "smoke" not in c["results_dir"].lower():
        raise ValueError("Smoke output must use a separate results_dir containing 'smoke'.")
    if not smoke and "smoke" in c["results_dir"].lower():
        raise ValueError("Formal results_dir must not contain 'smoke'.")
    if c["bootstrap"] != dict(block_length_steps=6, replicates=2000, seed=2026):
        raise ValueError("Keep the completed 04 bootstrap settings.")


def common_station_pool(data, config, station, target):
    """Build one union-feature validity mask; select columns only after splitting."""
    columns = feature_set_columns(data, ARMS[1], station, target)
    sequence = make_sequences(data, columns, data.target_columns[station][target],
                              int(config["target_default_windows"][target]), 6)
    split = legacy_random_split(len(sequence.x), .8, .1, config["split_seed"])
    labels = np.empty(len(sequence.x), dtype=object)
    for name in SPLITS:
        labels[split[name]] = name
    sequence = replace(sequence, split_labels=labels)
    arms = {}
    for arm in ARMS:
        selected = feature_set_columns(data, arm, station, target)
        positions = [columns.index(column) for column in selected]
        arms[arm] = replace(sequence, x=sequence.x[:, :, positions], feature_columns=selected)
    if len(arms[ARMS[1]].feature_columns) <= len(arms[ARMS[0]].feature_columns):
        raise ValueError("The meteorological arm must add weather/rain-history columns.")
    return sequence, split, arms


def pool_tables(data, config, station, target, protocol_hash):
    sequence, split, arms = common_station_pool(data, config, station, target)
    support_id = f"{protocol_hash}|{station}|{target}|{config['target_default_windows'][target]}|6"
    audit, manifest = random_support_tables(data, sequence, split, dict(
        support_id=support_id, protocol_hash=protocol_hash, target_station=station,
        target=target, baseline_protocol=SPLIT_PROTOCOL, station_rotation_protocol=PROTOCOL))
    groups = pd.DataFrame([dict(
        support_id=support_id, target_station=station, target=target, feature_set=arm,
        model="lstm", n_features=len(seq.feature_columns), feature_columns="|".join(seq.feature_columns),
        common_samples=True, identical_split_indices=True,
        **{f"n_{name}": len(split[name]) for name in SPLITS},
        **{f"{name}_ordered_indices_sha256": hashlib.sha256(np.asarray(split[name], dtype="<i8").tobytes()).hexdigest()
           for name in SPLITS}) for arm, seq in arms.items()])
    return sequence, split, arms, audit, manifest, groups


def validate_reference_config(current, reference):
    for key in ["frequency", "forecast_horizon_steps", "target_default_windows", "targets",
                "split", "split_seed", "missing_data", "training", "baseline_protocol", "bootstrap"]:
        if reference.get(key) != current.get(key):
            raise ValueError(f"C reuse rejected: 04/05 configuration differs at {key}.")
    if reference.get("target_station") != "C" or reference.get("meteorology_model") != "lstm":
        raise ValueError("C reuse requires the completed 04 C-station LSTM run.")
    if reference.get("meteorology_protocol") != "meteorology_random_fixed_budget_v1":
        raise ValueError("C reuse requires the fixed-budget random-window 04 protocol.")


def verify_task_tables(metrics, predictions, history, expected_ids=None):
    """Accept tasks only with all six leads, all origins, and a saved best epoch."""
    valid = set()
    if metrics.empty or predictions.empty or history.empty:
        return valid
    for task, m in metrics.groupby("task_id"):
        p = predictions[predictions.task_id == task]
        h = history[history.task_id == task]
        if expected_ids is not None and task not in expected_ids:
            continue
        if len(m) != 6 or set(m.horizon_step) != set(range(1, 7)) or h.empty:
            continue
        if m[["n_test", "best_epoch", "MAE", "RMSE"]].isna().any().any() or h.epoch.isna().any():
            continue
        if m.n_test.nunique() != 1 or m.best_epoch.nunique() != 1:
            continue
        n = int(m.n_test.iloc[0])
        if len(p) != n * 6 or p.duplicated(["target_start_time", "horizon_step"]).any():
            continue
        counts = p.groupby("horizon_step").size()
        if set(counts.index) != set(range(1, 7)) or not (counts == n).all():
            continue
        starts = [set(g.target_start_time.astype(str)) for _, g in p.groupby("horizon_step")]
        if any(s != starts[0] for s in starts[1:]):
            continue
        if not np.isfinite(m[["MAE", "RMSE"]].to_numpy()).all() or not np.isfinite(p[["y_true", "y_pred"]].to_numpy()).all():
            continue
        epochs = set(h.epoch.astype(int))
        if epochs != set(range(1, max(epochs) + 1)) or int(m.best_epoch.iloc[0]) not in epochs:
            continue
        valid.add(task)
    return valid


class RandomStationRotationRunner(ExperimentRunner):
    def __init__(self, config, **kwargs):
        validate_config(config)
        if kwargs.get("overwrite") or kwargs.get("model_override"):
            raise ValueError("Use a new results_dir for protocol changes; no overwrite/model override.")
        self.reference_verified = False
        self.reused_metrics = pd.DataFrame()
        self.reused_predictions = pd.DataFrame()
        self.reused_diagnostics = pd.DataFrame()
        try:
            super().__init__(config, "05_robustness", **kwargs)
            self.data = prepare_random_pool(self.data, config)
            audit = self.data.audit.drop(columns=["splitwise_missing_cells_after_imputation"], errors="ignore").copy()
            audit.to_csv(self.output / "data_audit.csv", index=False, encoding="utf-8-sig")
            self.data_hashes = {p.name: digest(p) for p in sorted(resolve_path(config, config["data_dir"]).glob("*.xlsx"))}
            self.source_hashes = {n: digest(Path(__file__).with_name(n)) for n in SOURCES + ["station_rotation_random.py"]}
            self._guard("data_fingerprints.json", self.data_hashes)
            self._guard("source_fingerprints.json", self.source_hashes)
            if not config.get("smoke_only", False):
                self._load_reference()
        except BaseException:
            if getattr(self, "_lock_handle", None) is not None:
                self._release_run_lock()
            raise

    def _guard(self, filename, payload):
        path = self.output / filename
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(f"{filename}: data, source or reference changed; use a new results_dir.")
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _load_reference(self):
        ref = resolve_path(self.config, self.config["reference_04_dir"])
        names = ["run_config.json", "run_status.json", "data_fingerprints.json", "source_fingerprints.json",
                 "completed_tasks.txt", "metrics.csv", "predictions.csv", "training_history.csv",
                 "model_diagnostics.csv", "sample_manifest.csv", "input_groups.csv"]
        for name in names:
            if not (ref / name).is_file():
                raise FileNotFoundError(f"C reuse requires {ref / name}; set reference_04_dir to your completed 04 directory.")
        read = lambda n: json.loads((ref / n).read_text(encoding="utf-8-sig"))
        validate_reference_config(self.config, read("run_config.json"))
        status = read("run_status.json")
        if status.get("run_status") != "complete" or status.get("completed_tasks") != 120 or status.get("unresolved_failed_tasks") != 0:
            raise ValueError("C reuse rejected: 04 must be complete (120 tasks, zero unresolved failures).")
        if read("data_fingerprints.json") != self.data_hashes:
            raise ValueError("C reuse rejected: raw data fingerprints differ from 04.")
        for name, value in read("source_fingerprints.json").items():
            if self.source_hashes.get(name) != value:
                raise ValueError(f"C reuse rejected: source fingerprint mismatch for {name}. Restore the source used for 04.")
        if set(read("source_fingerprints.json")) != set(SOURCES):
            raise ValueError("C reuse rejected: incomplete 04 source fingerprint manifest.")
        select = lambda f: f[(f.target_station == "C") & (f.model == "lstm") & f.feature_set.isin(ARMS)].copy()
        m = select(pd.read_csv(ref / "metrics.csv"))
        p = select(pd.read_csv(ref / "predictions.csv", parse_dates=["target_start_time", "valid_time"]))
        h = select(pd.read_csv(ref / "training_history.csv"))
        completed = set((ref / "completed_tasks.txt").read_text(encoding="utf-8").splitlines())
        valid = verify_task_tables(m, p, h, completed)
        expected_keys = {(t, a, s) for t in FIXED_WINDOWS for a in ARMS for s in SEEDS}
        keys = set(m[["target", "feature_set", "seed"]].itertuples(index=False, name=None))
        if len(valid) != 40 or valid != set(m.task_id) or keys != expected_keys or len(m) != 240:
            raise ValueError("C reuse rejected: expected 40 intact paired LSTM tasks / 240 metric rows.")
        d = pd.read_csv(ref / "model_diagnostics.csv")
        d = d[d.task_id.isin(valid)].copy()
        if len(d) != 40 or d.task_id.nunique() != 40:
            raise ValueError("C reuse rejected: missing/duplicate model diagnostics.")
        old_manifest = pd.read_csv(ref / "sample_manifest.csv")
        old_groups = pd.read_csv(ref / "input_groups.csv")
        for target in FIXED_WINDOWS:
            seq, split, arms, audit, manifest, groups = pool_tables(self.data, self.config, "C", target, self.protocol_hash)
            previous = old_manifest[old_manifest.target == target]
            columns = ["sample_index", "data_split", "input_start_time", "issue_time", "target_start_time", "target_end_time"]
            left = manifest[columns].sort_values("sample_index").reset_index(drop=True).astype(str)
            right = previous[columns].sort_values("sample_index").reset_index(drop=True).astype(str)
            if not left.equals(right):
                raise ValueError(f"C reuse rejected: sample timestamps or split assignments differ for {target}.")
            for arm in ARMS:
                a = groups[groups.feature_set == arm].iloc[0]
                b = old_groups[(old_groups.target == target) & (old_groups.feature_set == arm) & (old_groups.model == "lstm")]
                fields = ["feature_columns", "n_train", "n_validation", "n_test"] + [f"{s}_ordered_indices_sha256" for s in SPLITS]
                if len(b) != 1 or any(str(a[k]) != str(b.iloc[0][k]) for k in fields):
                    raise ValueError(f"C reuse rejected: feature/split order differs for {target}/{arm}.")
            true = prediction_frame(seq.target_start_times[split["test"]], seq.y[split["test"]], seq.y[split["test"]], {})
            for task, task_p in p[p.target == target].groupby("task_id"):
                joined = true.merge(task_p, on=["target_start_time", "horizon_step"], suffixes=("_now", "_04"), validate="one_to_one")
                # 04 writes float32 predictions/labels with pandas' short decimal representation.
                if len(joined) != len(true) or len(task_p) != len(true) or not np.array_equal(joined.y_true_now.to_numpy(dtype=np.float32), joined.y_true_04.to_numpy(dtype=np.float32)):
                    raise ValueError(f"C reuse rejected: test labels/support differ for {task}.")
                for lead, g in task_p.groupby("horizon_step"):
                    row = m[(m.task_id == task) & (m.horizon_step == lead)].iloc[0]
                    error = g.y_pred.to_numpy(dtype=np.float32) - g.y_true.to_numpy(dtype=np.float32)
                    if not np.allclose([row.MAE, row.RMSE], [np.abs(error).mean(), np.sqrt((error ** 2).mean())], rtol=1e-6, atol=1e-9):
                        raise ValueError(f"C reuse rejected: metrics/predictions disagree for {task}.")
                    if not np.allclose(g.absolute_error, np.abs(error), rtol=1e-6, atol=1e-9):
                        raise ValueError(f"C reuse rejected: saved absolute errors disagree for {task}.")
            for frame, filename in [(audit, "sample_support_audit.csv"), (manifest, "sample_manifest.csv"), (groups, "input_groups.csv")]:
                _upsert_csv(frame, self.output / filename, "support_id")
        provenance = dict(source_directory=str(ref), source_experiment="04_meteorology", reused_lstm_tasks=40,
                          raw_data_verified=True, code_verified=True, common_samples_verified=True,
                          files_sha256={name: digest(ref / name) for name in names})
        self._guard("reference_04_provenance.json", provenance)
        for frame in [m, p, d]:
            frame["target_station"] = "C"
            frame["result_origin"] = "reused_04"
            frame["source_experiment"] = "04_meteorology"
            frame["source_protocol_hash"] = frame.task_id.astype(str).str.split("|", regex=False).str[0]
        self.reused_metrics, self.reused_predictions, self.reused_diagnostics = m, p, d
        self.reference_verified = True
        print("[05] Verified C reference: 40 LSTM tasks reused; no C training.", flush=True)

    def _repair_partial_outputs(self):
        # Run before the parent's presence-only repair; partial lead/CSV writes must rerun.
        if self.completed:
            read = lambda p: _read_csv_resilient(p) if p.exists() else pd.DataFrame()
            self.completed &= verify_task_tables(read(self.metrics_path), read(self.predictions_path), read(self.history_path), self.completed)
        super()._repair_partial_outputs()

    def run(self):
        try:
            self._train_new_stations()
            self._postprocess()
            return self.output
        except BaseException:
            self._write_run_status()
            raise
        finally:
            self._release_run_lock()

    def _train_new_stations(self):
        for station in self.config["target_stations"]:
            for target in self.config["targets"]:
                seq, split, arms, audit, manifest, groups = pool_tables(self.data, self.config, station, target, self.protocol_hash)
                for frame, filename in [(audit, "sample_support_audit.csv"), (manifest, "sample_manifest.csv"), (groups, "input_groups.csv")]:
                    _upsert_csv(frame, self.output / filename, "support_id")
                lookback = self.config["target_default_windows"][target]
                print(f"[05] {station}/{target}: " + ", ".join(f"{s}={len(split[s])}" for s in SPLITS), flush=True)
                scale = training_mase_scale(seq, split["train"])
                for arm, selected in arms.items():
                    for seed in self.config["training"]["seeds"]:
                        task = "|".join(map(str, [self.protocol_hash, self.experiment, PROTOCOL, station, target, arm, "lstm", seed, lookback, 6]))
                        if task in self.completed:
                            continue
                        if self.max_tasks is not None and self.tasks_run >= self.max_tasks:
                            continue
                        metadata = dict(task_id=task, protocol_hash=self.protocol_hash, experiment=self.experiment,
                            subexperiment="target_station_rotation", fold=0, split_scheme=SPLIT_PROTOCOL,
                            target_station=station, target=target, target_column=seq.target_column,
                            lookback_steps=lookback, lookback_days=lookback / 6, forecast_horizon_steps=6,
                            feature_set=arm, model="lstm", seed=seed, n_features=len(selected.feature_columns),
                            n_train=len(split["train"]), n_validation=len(split["validation"]), n_test=len(split["test"]),
                            result_origin="new_05", source_experiment=self.experiment, source_protocol_hash=self.protocol_hash)
                        try:
                            prediction, history, epoch, parameters, _ = self._fit_method(Method(arm, "lstm"), selected.x, selected, split, seed)
                            if prediction.shape != seq.y[split["test"]].shape or not np.isfinite(prediction).all() or not history:
                                raise ValueError("Expected finite predictions matching test labels, and nonempty LSTM training history.")
                            metadata.update(best_epoch=epoch, parameter_count=parameters)
                            _append_csv(pd.DataFrame(calculate_metric_rows(seq.y[split["test"]], prediction, metadata, scale)), self.metrics_path)
                            _append_csv(prediction_frame(seq.target_start_times[split["test"]], seq.y[split["test"]], prediction, metadata), self.predictions_path)
                            history_frame = pd.DataFrame(history)
                            for k, v in metadata.items():
                                history_frame[k] = v
                            _append_csv(history_frame, self.history_path)
                            self._mark_completed(task)
                            self.tasks_run += 1
                            print(f"  done {station}/{target}/{arm}/seed={seed}, best_epoch={epoch}", flush=True)
                        except Exception as exc:
                            self.failed[task] = f"{type(exc).__name__}: {exc}"
                            self._write_failed()
                            with self.error_path.open("a", encoding="utf-8") as handle:
                                handle.write(task + "\n" + traceback.format_exc() + "\n")
                            if not self.continue_on_error:
                                raise

    def _postprocess(self):
        self._repair_partial_outputs()
        metrics = _read_csv_resilient(self.metrics_path) if self.metrics_path.exists() else pd.DataFrame()
        predictions = _read_csv_resilient(self.predictions_path, parse_dates=["target_start_time", "valid_time"]) if self.predictions_path.exists() else pd.DataFrame()
        diagnostics = pd.DataFrame()
        if self.history_path.exists():
            h = _read_csv_resilient(self.history_path)
            keys = ["task_id", "target_station", "target", "feature_set", "model", "seed", "lookback_steps", "best_epoch"]
            diagnostics = h.groupby(keys, dropna=False).agg(epochs_run=("epoch", "max")).reset_index()
            diagnostics["max_epochs"] = self.config["training"]["max_epochs"]
            diagnostics["patience"] = self.config["training"]["patience"]
            diagnostics["reached_epoch_cap"] = diagnostics.epochs_run >= diagnostics.max_epochs
            diagnostics["result_origin"] = "new_05"
            diagnostics.to_csv(self.output / "model_diagnostics.csv", index=False, encoding="utf-8-sig")
        def combine(new, old):
            frames = [f for f in [new, old] if not f.empty]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        merged = combine(metrics, self.reused_metrics)
        preds = combine(predictions, self.reused_predictions)
        diag = combine(diagnostics, self.reused_diagnostics)
        if not merged.empty:
            # Original task IDs/protocol hashes remain intact; reporting groups use 05.
            merged["experiment"] = self.experiment
            merged["subexperiment"] = "target_station_rotation"
            merged.to_csv(self.output / "combined_metrics.csv", index=False, encoding="utf-8-sig")
            preds.to_csv(self.output / "combined_predictions.csv", index=False, encoding="utf-8-sig")
            diag.to_csv(self.output / "combined_model_diagnostics.csv", index=False, encoding="utf-8-sig")
            mean_sd_table(merged).to_csv(self.output / "metrics_mean_sd.csv", index=False, encoding="utf-8-sig")
            gain_rows, block_rows = [], []
            for station, frame in merged.groupby("target_station"):
                gains = paired_seed_gains(frame, "lstm")
                gains["target_station"] = station
                gain_rows.append(gains)
                blocks = paired_calendar_gains(preds[preds.target_station == station], "lstm", self.config["training"]["seeds"], self.config["bootstrap"])
                blocks["target_station"] = station
                block_rows.append(blocks)
            gains = pd.concat(gain_rows, ignore_index=True)
            blocks = pd.concat(block_rows, ignore_index=True)
            gains.to_csv(self.output / "paired_seed_horizon.csv", index=False, encoding="utf-8-sig")
            blocks.to_csv(self.output / "paired_block_bootstrap.csv", index=False, encoding="utf-8-sig")
            summaries = []
            for (station, target), group in (gains.groupby(["target_station", "target"]) if not gains.empty else []):
                complete = group.groupby("seed").filter(lambda g: len(g) == 6 and set(g.horizon_hours) == {4, 8, 12, 16, 20, 24})
                if complete.empty:
                    continue
                seed_mean = complete.groupby("seed")[["MAE_reference", "MAE_augmented", "RMSE_reduction", "R2_gain"]].mean()
                percent = 100 * (1 - seed_mean.MAE_augmented / seed_mean.MAE_reference)
                summaries.append(dict(target_station=station, target=target, n_paired_seeds=len(seed_mean),
                    expected_seeds=len(self.config["training"]["seeds"]),
                    status="complete" if len(seed_mean) == len(self.config["training"]["seeds"]) else "partial_paired_seeds",
                    MAE_reference=seed_mean.MAE_reference.mean(), MAE_augmented=seed_mean.MAE_augmented.mean(),
                    MAE_reduction_percent=100 * (1 - seed_mean.MAE_augmented.mean() / seed_mean.MAE_reference.mean()),
                    seed_MAE_reduction_percent_mean=percent.mean(), seed_MAE_reduction_percent_sd=percent.std(ddof=1),
                    n_seeds_with_positive_reduction=int((percent > 0).sum()),
                    RMSE_reduction=seed_mean.RMSE_reduction.mean(), R2_gain=seed_mean.R2_gain.mean(),
                    horizon_scope="equal_mean_of_six_horizons", result_origin="reused_04" if station == "C" else "new_05"))
            pd.DataFrame(summaries).to_csv(self.output / "station_rotation_gain_summary.csv", index=False, encoding="utf-8-sig")
        self._write_run_status()

    def _write_run_status(self):
        super()._write_run_status()
        path = self.output / "run_status.json"
        status = json.loads(path.read_text(encoding="utf-8"))
        expected = len(self.config["target_stations"]) * len(self.config["targets"]) * len(ARMS) * len(self.config["training"]["seeds"])
        smoke = self.config.get("smoke_only", False)
        complete = len(self.completed) == expected and not self.failed and (smoke or self.reference_verified)
        status.update(station_rotation_protocol=PROTOCOL, expected_new_lstm_tasks=expected,
            completed_new_lstm_tasks=len(self.completed), reused_C_lstm_tasks=40 if self.reference_verified else 0,
            expected_combined_tasks=expected + (0 if smoke else 40), reference_C_verified=self.reference_verified,
            run_status=("smoke_complete" if smoke else "complete") if complete else (
                "audit_only" if self.max_tasks == 0 and not self.completed else "partial"),
            smoke_only=smoke, target_windows_steps=FIXED_WINDOWS, max_epochs=self.config["training"]["max_epochs"],
            patience=self.config["training"]["patience"], split_seed=42,
            evaluation_scope="fixed_random_window_holdout_not_temporally_independent",
            interpretation="target_station_rotation_with_separate_models_not_unseen_station_transfer")
        path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
