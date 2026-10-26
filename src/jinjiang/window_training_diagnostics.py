"""Paired, explicitly selected 03 training diagnostics; no changes to 01--03."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from .config import resolve_path
from .baseline_random import SPLITS, PROTOCOL as SPLIT_PROTOCOL, random_support_tables
from .baseline_within_split import training_mase_scale
from .evaluation import calculate_metric_rows, mean_sd_table, prediction_frame
from .runner import ExperimentRunner, _append_csv, _read_csv_resilient, _upsert_csv
from .window_horizon_random import (
    RandomWindowRunner, assign_pool, fit_with_validation, validate_config,
    select_windows, describe_selected_test_windows, GRIDS,
)

TASKS = {
    "cap": [
        ("CODMN", 24, 99, "epoch_cap"), ("CODMN", 48, 99, "epoch_cap"),
        ("CODMN", 72, 99, "epoch_cap"), ("CODMN", 84, 2026, "epoch_cap"),
        ("TP", 30, 42, "epoch_cap"), ("TP", 36, 42, "epoch_cap"),
        ("TP", 66, 2026, "epoch_cap"),
    ],
    "patience": [
        ("CODMN", 48, 42, "higher_validation_error"),
        ("CODMN", 48, 3407, "higher_validation_error"),
        ("CODMN", 48, 2026, "same_window_control"),
        ("TP", 66, 99, "higher_validation_error"),
        ("TP", 66, 42, "same_window_control"),
        ("TP", 54, 3407, "higher_validation_error"),
        ("TP", 54, 2026, "same_window_control"),
    ],
}
KEY = ["target", "lookback_steps", "seed"]
REFERENCE_FILES = [
    "run_config.json", "run_status.json", "data_fingerprints.json",
    "source_fingerprints.json", "sample_manifest.csv", "model_diagnostics.csv",
    "metrics.csv", "validation_metrics.csv", "training_history.csv",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_plan(check):
    return pd.DataFrame(TASKS[check], columns=KEY + ["diagnostic_role"])


def validate_diagnostic_config(config):
    validate_config(config)
    check = config.get("diagnostic_check")
    if check not in TASKS:
        raise ValueError("diagnostic_check must be cap or patience.")
    if config["window_phase"] != "main" or config["window_grid"] != GRIDS["main"]:
        raise ValueError("Keep the original full main grid and 84-step common support.")
    if config["targets"] != ["CODMN", "TP"] or config.get("window_model", "lstm") != "lstm":
        raise ValueError("Diagnostics require CODMN/TP and the original LSTM.")
    if config["training"]["seeds"] != [42, 2026, 3407, 7, 99]:
        raise ValueError("Keep the original five-seed definition; tasks are filtered separately.")
    smoke = config.get("smoke_test", False)
    if config["training"]["max_epochs"] != (2 if smoke else 2048):
        raise ValueError("Use 2048 epochs for formal diagnostics, or --smoke for two epochs.")
    if config["training"]["patience"] != (30 if check == "cap" else 100):
        raise ValueError("Use patience 30 for cap and 100 for patience diagnostics.")
    if not config.get("baseline_results_dir"):
        raise ValueError("Set baseline_results_dir to the completed 1024-epoch result directory.")


def without_internal(config):
    return {k: v for k, v in config.items() if not k.startswith("_")}


def compatible_reference_config(config, old):
    """Only results path, diagnostic metadata and the intended training budget differ."""
    expected = without_internal(old)
    actual = without_internal(config)
    for name in ["results_dir", "diagnostic_check", "baseline_results_dir", "diagnostic_protocol"]:
        actual.pop(name, None)
        expected.pop(name, None)
    actual["smoke_test"] = expected.get("smoke_test", False)
    for name in ["max_epochs", "patience"]:
        actual["training"] = dict(actual["training"])
        actual["training"][name] = expected["training"][name]
    if actual != expected:
        different = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise ValueError(f"Reference protocol differs beyond the intended budget: {different}")
    if old["training"]["max_epochs"] != 1024 or old["training"]["patience"] != 30:
        raise ValueError("The reference must be the completed 1024-epoch, patience-30 experiment.")


def verify_prefix(old, new, smoke=False):
    """An extension must reproduce every available shared training epoch."""
    old = old.sort_values("epoch")
    new = new.sort_values("epoch")
    if old.empty or new.empty or old.epoch.duplicated().any() or new.epoch.duplicated().any():
        raise ValueError("Missing or duplicate training history.")
    for frame in [old, new]:
        if frame.epoch.tolist() != list(range(1, int(frame.epoch.max()) + 1)):
            raise ValueError("Training history has missing epochs.")
    if not smoke and new.epoch.max() < old.epoch.max():
        raise ValueError("New training stopped before the reference: pairing is not reproducible.")
    shared = old.merge(new, on="epoch", suffixes=("_old", "_new"))
    differences = {}
    for name in ["train_loss", "validation_loss"]:
        a, b = shared[name + "_old"].to_numpy(), shared[name + "_new"].to_numpy()
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite loss in prefix comparison.")
        differences[name + "_max_abs_difference"] = float(np.max(np.abs(a - b)))
        if not np.allclose(a, b, rtol=1e-6, atol=1e-8):
            raise ValueError(f"Training prefix differs in {name}; check environment/determinism.")
    return dict(shared_epochs=len(shared), prefix_matches=True, **differences)


def replace_tasks(reference, updates):
    """Replace complete task rows by semantic keys; preserve each row's source identity."""
    keys = ["target", "model", "lookback_steps", "seed"]
    changed = set(updates[keys].itertuples(index=False, name=None))
    keep = ~reference[keys].apply(tuple, axis=1).isin(changed)
    return pd.concat([reference.loc[keep], updates], ignore_index=True)


class TrainingDiagnosticRunner(RandomWindowRunner):
    def __init__(self, config, **kwargs):
        validate_diagnostic_config(config)
        self._postprocessed = False
        self.check = config["diagnostic_check"]
        self.plan = task_plan(self.check)
        if config.get("smoke_test", False):
            self.plan = self.plan.iloc[[0]].copy()
        self.reference = resolve_path(config, config["baseline_results_dir"])
        self.reference_config = read_json(self.reference / "run_config.json")
        compatible_reference_config(config, self.reference_config)
        if read_json(self.reference / "run_status.json").get("run_status") != "complete":
            raise ValueError("The 1024-epoch reference is not complete.")
        self.old_diagnostics = pd.read_csv(self.reference / "model_diagnostics.csv")
        self.old_history = pd.read_csv(self.reference / "training_history.csv")
        self.old_validation = pd.read_csv(self.reference / "validation_metrics.csv")
        self.old_metrics = pd.read_csv(self.reference / "metrics.csv")
        for frame in [self.old_diagnostics, self.old_validation, self.old_metrics]:
            if not frame.model.isin(["lstm", "persistence", "seasonal_naive"]).all():
                raise ValueError("Unexpected reference models.")
        self._validate_reference_grid()
        self._validate_plan()
        super().__init__(config, **kwargs)
        try:
            # Pin both the historical reference and the new diagnostic implementation on resume.
            self._pin("diagnostic_reference_fingerprints.json", {n: sha256(self.reference / n) for n in REFERENCE_FILES})
            self._pin("diagnostic_source_fingerprint.json", {"window_training_diagnostics.py": sha256(__file__)})
            for name in ["data_fingerprints.json", "source_fingerprints.json"]:
                if read_json(self.output / name) != read_json(self.reference / name):
                    raise ValueError(f"{name} differs from the reference; pairing rejected.")
            self.plan.assign(lookback_days=self.plan.lookback_steps / 6).to_csv(
                self.output / "task_plan.csv", index=False, encoding="utf-8-sig")
        except BaseException:
            self._release_run_lock()
            raise

    def _pin(self, filename, payload):
        path = self.output / filename
        if path.exists() and read_json(path) != payload:
            raise ValueError(f"{filename} changed; use a new result directory.")
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _validate_reference_grid(self):
        expected = {(t, w, s, hour) for t in ["CODMN", "TP"] for w in GRIDS["main"]
                    for s in self.config_seeds for hour in range(4, 73, 4)}
        for frame in [self.old_metrics, self.old_validation]:
            modeled = frame[frame.model == "lstm"]
            actual = set(modeled[KEY + ["horizon_hours"]].itertuples(index=False, name=None))
            if actual != expected or len(modeled) != len(expected):
                raise ValueError("Reference LSTM metric grid is incomplete or duplicated.")
        diagnostic = self.old_diagnostics[self.old_diagnostics.model == "lstm"]
        expected_tasks = {row[:3] for row in expected}
        if (len(diagnostic) != len(expected_tasks)
                or set(diagnostic[KEY].itertuples(index=False, name=None)) != expected_tasks):
            raise ValueError("Reference model diagnostics are incomplete or duplicated.")
        for r in self.old_diagnostics[self.old_diagnostics.model == "lstm"].itertuples():
            if not r.reached_epoch_cap and r.epochs_run - r.best_epoch != 30:
                raise ValueError("Cannot reuse a reference task that did not finish patience-30 stopping.")

    @property
    def config_seeds(self):
        return [42, 2026, 3407, 7, 99]

    def _validate_plan(self):
        available = self.old_diagnostics[self.old_diagnostics.model == "lstm"].set_index(KEY)
        if available.index.duplicated().any():
            raise ValueError("Reference diagnostics contain duplicate tasks.")
        for r in self.plan.itertuples():
            if (r.target, r.lookback_steps, r.seed) not in available.index:
                raise ValueError("A planned task is missing from the reference.")
        if self.check == "cap" and len(self.plan) == 7:
            capped = self.old_diagnostics[(self.old_diagnostics.model == "lstm") & self.old_diagnostics.reached_epoch_cap]
            if set(capped[KEY].itertuples(index=False, name=None)) != set(self.plan[KEY].itertuples(index=False, name=None)):
                raise ValueError("Cap task list does not exactly match the seven reference capped tasks.")

    def _repair_partial_outputs(self):
        super()._repair_partial_outputs()
        # A completed diagnostic needs history and diagnostics too, not just scored rows.
        required = set(self.completed)
        for filename in ["model_diagnostics.csv", "training_history.csv"]:
            path = self.output / filename
            required &= set(pd.read_csv(path).task_id) if path.exists() else set()
        if required != self.completed:
            self.completed = required
            super()._repair_partial_outputs()

    def _task_rows(self, frame, task):
        return frame[(frame.target == task.target) & (frame.lookback_steps == task.lookback_steps)
                     & (frame.seed == task.seed)]

    def _check_support(self, target, manifest):
        old = pd.read_csv(self.reference / "sample_manifest.csv")
        old = old[old.target == target]
        cols = [c for c in manifest if c not in ["support_id", "protocol_hash"]]
        a, b = manifest[cols].reset_index(drop=True), old[cols].reset_index(drop=True)
        for name in cols:
            if name.endswith("_time"):
                a[name], b[name] = pd.to_datetime(a[name]), pd.to_datetime(b[name])
        pd.testing.assert_frame_equal(a, b, check_dtype=False)

    def _experiment_03(self):
        for target in self.config["targets"]:
            sequence, split = assign_pool(self.data, self.config, target)
            support_id = f"{self.protocol_hash}|main|C|{target}|84|18"
            common = dict(protocol_hash=self.protocol_hash, support_id=support_id,
                          experiment=self.experiment, subexperiment="training_diagnostic_" + self.check,
                          fold=0, split_scheme=SPLIT_PROTOCOL, target_station="C", target=target,
                          target_column=sequence.target_column, forecast_horizon_steps=18,
                          support_lookback_steps=84)
            audit, manifest = random_support_tables(self.data, sequence, split, dict(
                support_id=support_id, protocol_hash=self.protocol_hash, target_station="C", target=target,
                baseline_protocol=SPLIT_PROTOCOL, window_phase="main",
                window_horizon_protocol=self.config["window_horizon_protocol"]))
            self._check_support(target, manifest)
            _upsert_csv(audit, self.output / "sample_support_audit.csv", "support_id")
            _upsert_csv(manifest, self.output / "sample_manifest.csv", "support_id")
            print(f"[03/{self.check}] {target}: common_windows={len(sequence.x)}, " +
                  ", ".join(f"{n}={len(split[n])}" for n in SPLITS), flush=True)
            scale = training_mase_scale(sequence, split["train"])
            for task in self.plan[self.plan.target == target].itertuples():
                task_id = f"{self.protocol_hash}|03|{self.check}|{target}|{task.lookback_steps}|{task.seed}"
                if task_id in self.completed:
                    print(f"  skip: {target}/L{task.lookback_steps}/seed={task.seed}", flush=True)
                    continue
                if self.max_tasks is not None and self.tasks_run >= self.max_tasks:
                    continue
                metadata = dict(common, task_id=task_id, feature_set="all_stations", model="lstm",
                                seed=task.seed, lookback_steps=task.lookback_steps,
                                lookback_days=task.lookback_steps / 6, n_features=len(sequence.feature_columns),
                                n_train=len(split["train"]), n_validation=len(split["validation"]),
                                n_test=len(split["test"]))
                try:
                    x = np.ascontiguousarray(sequence.x[:, -task.lookback_steps:, :])
                    result = fit_with_validation("lstm", x, sequence, split, task.seed, self.config["training"])
                    history = pd.DataFrame(result.history)
                    prefix = verify_prefix(self._task_rows(self.old_history, task), history,
                                           self.config.get("smoke_test", False))
                    for name, prediction in [("test", result.prediction), ("validation", result.validation_prediction)]:
                        if prediction.shape != sequence.y[split[name]].shape or not np.isfinite(prediction).all():
                            raise ValueError(f"Invalid {name} predictions.")
                    metadata.update(best_epoch=result.best_epoch, parameter_count=result.parameter_count)
                    test = pd.DataFrame(calculate_metric_rows(sequence.y[split["test"]], result.prediction, metadata, scale))
                    val = pd.DataFrame(calculate_metric_rows(sequence.y[split["validation"]], result.validation_prediction, metadata, scale))
                    val = val.rename(columns={"n_test": "n_scored"})
                    val["n_test"], val["evaluation_split"] = len(split["test"]), "validation"
                    _append_csv(test, self.metrics_path)
                    _append_csv(val, self.output / "validation_metrics.csv")
                    _append_csv(prediction_frame(sequence.target_start_times[split["test"]], sequence.y[split["test"]],
                                                 result.prediction, metadata), self.predictions_path)
                    for key, value in metadata.items():
                        history[key] = value
                    _append_csv(history, self.history_path)
                    last = int(history.epoch.max())
                    _append_csv(pd.DataFrame([dict(metadata, epochs_run=last,
                        max_epochs=self.config["training"]["max_epochs"], patience=self.config["training"]["patience"],
                        reached_epoch_cap=last >= self.config["training"]["max_epochs"],
                        best_epoch_near_cap=result.best_epoch >= .95 * self.config["training"]["max_epochs"],
                        diagnostic_role=task.diagnostic_role, **prefix)]), self.output / "model_diagnostics.csv")
                    self._mark_completed(task_id)
                    self.tasks_run += 1
                    print(f"  done: {target}/L{task.lookback_steps}/seed={task.seed}; best={result.best_epoch}, epochs={last}", flush=True)
                except Exception as exc:
                    with self.error_path.open("a", encoding="utf-8") as handle:
                        handle.write(task_id + "\n" + traceback.format_exc() + "\n")
                    self.failed[task_id] = f"{type(exc).__name__}: {exc}"
                    self._write_failed()
                    raise

    def _comparisons(self, metrics, validation, diagnostics):
        rows, horizon_rows = [], []
        for task in self.plan.itertuples():
            new_d = self._task_rows(diagnostics, task)
            if new_d.empty:
                continue
            old_d = self._task_rows(self.old_diagnostics, task).iloc[0]
            new_d = new_d.iloc[0]
            base = dict(target=task.target, lookback_steps=task.lookback_steps,
                        lookback_days=task.lookback_steps / 6, seed=task.seed, diagnostic_role=task.diagnostic_role,
                        baseline_best_epoch=int(old_d.best_epoch), baseline_epochs_run=int(old_d.epochs_run),
                        new_best_epoch=int(new_d.best_epoch), new_epochs_run=int(new_d.epochs_run),
                        new_reached_epoch_cap=bool(new_d.reached_epoch_cap),
                        best_after_original_stop=bool(new_d.best_epoch > old_d.epochs_run),
                        prefix_matches=bool(new_d.prefix_matches), selection_source="validation_only")
            for hour in [0] + list(range(4, 73, 4)):
                row = dict(base, horizon_hours=hour)
                for label, old, new in [("validation", self.old_validation, validation), ("test", self.old_metrics, metrics)]:
                    a, b = self._task_rows(old, task), self._task_rows(new, task)
                    if hour:
                        a, b = a[a.horizon_hours == hour], b[b.horizon_hours == hour]
                    for metric in ["MAE", "RMSE", "R2"]:
                        row[f"baseline_{label}_{metric}"] = a[metric].mean()
                        row[f"new_{label}_{metric}"] = b[metric].mean()
                    den = row[f"baseline_{label}_MAE"]
                    row[f"{label}_MAE_reduction_percent"] = 100 * (1 - row[f"new_{label}_MAE"] / den) if den else np.nan
                (rows if hour == 0 else horizon_rows).append(row)
        pd.DataFrame(rows).to_csv(self.output / "paired_task_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(horizon_rows).to_csv(self.output / "paired_horizon_comparison.csv", index=False, encoding="utf-8-sig")

    def _merge_cap(self, metrics, validation, diagnostics):
        out = self.output / "cap_merged_full_grid"
        out.mkdir(exist_ok=True)
        for name, old, new in [("metrics.csv", self.old_metrics, metrics),
                               ("validation_metrics.csv", self.old_validation, validation),
                               ("model_diagnostics.csv", self.old_diagnostics, diagnostics)]:
            reference = old.copy()
            reference["lineage"] = "reused_natural_stop_1024_reference"
            update = new.copy()
            update["lineage"] = "rerun_cap_2048"
            merged = replace_tasks(reference, update)
            merged["source_subexperiment"] = merged.subexperiment
            merged["subexperiment"] = "window_horizon_main_effective_cap2048"
            merged["effective_max_epochs"], merged["effective_patience"] = 2048, 30
            merged.to_csv(out / name, index=False, encoding="utf-8-sig")
        merged_test = pd.read_csv(out / "metrics.csv")
        merged_val = pd.read_csv(out / "validation_metrics.csv")
        mean_sd_table(merged_test).to_csv(out / "metrics_mean_sd.csv", index=False, encoding="utf-8-sig")
        mean_sd_table(merged_val).to_csv(out / "validation_metrics_mean_sd.csv", index=False, encoding="utf-8-sig")
        summary, choices, progress = select_windows(merged_val, self.config)
        if not progress.status.eq("complete").all():
            raise ValueError("Merged full grid is incomplete; do not use selections.")
        summary.to_csv(out / "validation_window_summary.csv", index=False, encoding="utf-8-sig")
        choices.to_csv(out / "validation_window_selection.csv", index=False, encoding="utf-8-sig")
        describe_selected_test_windows(merged_test, choices).to_csv(
            out / "test_metrics_for_validation_selected_windows.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([dict(target=r.target, lookback_steps=r.lookback_steps, seed=r.seed,
                           treatment="rerun_cap_2048") for r in self.plan.itertuples()]).to_csv(
            out / "replacement_tasks.csv", index=False, encoding="utf-8-sig")
        (out / "merge_protocol.json").write_text(json.dumps(dict(
            effective_max_epochs=2048, patience=30, rerun_lstm_tasks=7, reused_naturally_stopped_lstm_tasks=143,
            reused_naive_tasks=20, selection_source="validation_only", full_predictions_and_histories="retained in source runs",
            protocol_hashes="Original source hashes retained per row; effective protocol is recorded separately.",
            reference=str(self.reference)), indent=2), encoding="utf-8")

    def _postprocess(self):
        self._repair_partial_outputs()
        if self.metrics_path.exists():
            metrics = pd.read_csv(self.metrics_path)
            validation = pd.read_csv(self.output / "validation_metrics.csv")
            diagnostics = pd.read_csv(self.output / "model_diagnostics.csv")
            mean_sd_table(metrics).to_csv(self.output / "metrics_mean_sd.csv", index=False, encoding="utf-8-sig")
            self._comparisons(metrics, validation, diagnostics)
            if self.check == "cap" and not self.config.get("smoke_test", False) and len(self.completed) == 7 and not self.failed:
                self._merge_cap(metrics, validation, diagnostics)
        self._postprocessed = True
        self._write_run_status()

    def _write_run_status(self):
        ExperimentRunner._write_run_status(self)
        path = self.output / "run_status.json"
        status = read_json(path)
        expected = len(self.plan)
        status.update(diagnostic_check=self.check, expected_tasks=expected, expected_lstm_tasks=expected,
                      expected_naive_tasks=0, completed_tasks=len(self.completed),
                      run_status="complete" if len(self.completed) == expected and not self.failed and self._postprocessed else (
                          "audit_only" if self.max_tasks == 0 and not self.completed else "partial"),
                      smoke_test=self.config.get("smoke_test", False),
                      window_selection_source="cap_full_grid_only" if self.check == "cap" else "diagnostic_only_no_window_ranking",
                      evaluation_scope="fixed_random_window_holdout_not_temporally_independent")
        path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
