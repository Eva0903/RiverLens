from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback
from typing import Iterable
import warnings

import numpy as np
import pandas as pd

from .config import resolve_path
from .baseline_within_split import (
    PROTOCOL as WITHIN_SPLIT_PROTOCOL,
    make_within_split_sequences,
    support_tables,
    training_mase_scale,
)
from .data import (
    PreparedDataset,
    SequenceData,
    feature_set_columns,
    make_sequences,
    official_split,
    prepare_dataset,
    rolling_origin_datasets,
)
from .evaluation import (
    calculate_metric_rows,
    distribution_shift_table,
    forecastability_envelope,
    mase_scale_from_training,
    mean_sd_table,
    paired_block_bootstrap_table,
    prediction_frame,
    skill_scores_table,
)
from .models import fit_ridge_predict, fit_torch_predict
from .plots import generate_experiment_figures


EXPERIMENTS = {
    "01_baseline",
    "02_station_ablation",
    "03_window_horizon",
    "04_meteorology",
    "05_robustness",
}


@dataclass(frozen=True)
class Method:
    feature_set: str
    model: str


def _append_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


def _read_csv_resilient(path: Path, **kwargs) -> pd.DataFrame:
    """Recover complete rows if a process was interrupted during a CSV append."""
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        return pd.read_csv(path, on_bad_lines="skip", engine="python", **kwargs)


def _upsert_csv(frame: pd.DataFrame, path: Path, key: str) -> None:
    if frame.empty:
        return
    if path.exists():
        existing = _read_csv_resilient(path)
        existing = existing[~existing[key].astype(str).isin(frame[key].astype(str))]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
                category=FutureWarning,
            )
            frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _baseline_methods() -> list["Method"]:
    return [
        Method("persistence", "persistence"),
        Method("seasonal_naive", "seasonal_naive"),
    ]


def _random_split(n: int, train_fraction: float, validation_fraction: float, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    return {
        "train": order[:train_end],
        "validation": order[train_end:validation_end],
        "test": order[validation_end:],
    }


class ExperimentRunner:
    def __init__(
        self,
        config: dict,
        experiment: str,
        overwrite: bool = False,
        model_override: str | None = None,
        max_tasks: int | None = None,
        continue_on_error: bool = False,
    ):
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment {experiment}; choose from {sorted(EXPERIMENTS)}")
        self.config = config
        self.clean_config = {
            key: value for key, value in self.config.items() if not key.startswith("_")
        }
        self.config_sha256 = hashlib.sha256(
            json.dumps(self.clean_config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        self.protocol_hash = self.config_sha256[:16]
        self.experiment = experiment
        self.within_split_baseline = (
            experiment == "01_baseline"
            and config.get("baseline_protocol") == WITHIN_SPLIT_PROTOCOL
        )
        self.model_override = model_override
        self.max_tasks = max_tasks
        self.continue_on_error = continue_on_error
        self.tasks_run = 0
        self.output = resolve_path(config, config["results_dir"]) / experiment
        self.output.mkdir(parents=True, exist_ok=True)
        self._lock_handle = None
        self.metrics_path = self.output / "metrics.csv"
        self.predictions_path = self.output / "predictions.csv"
        self.history_path = self.output / "training_history.csv"
        self.shap_path = self.output / "shap_importance.csv"
        self.distribution_path = self.output / "distribution_shift.csv"
        self.completed_path = self.output / "completed_tasks.txt"
        self.error_path = self.output / "errors.log"
        self.failed_path = self.output / "failed_tasks.json"
        run_config_path = self.output / "run_config.json"
        if run_config_path.exists() and not overwrite:
            previous_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            previous_clean = {
                key: value for key, value in previous_config.items() if not key.startswith("_")
            }
            previous_hash = hashlib.sha256(
                json.dumps(previous_clean, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            if previous_hash != self.config_sha256:
                raise RuntimeError(
                    "The existing result directory was created with a different configuration. "
                    "Use a different results_dir or rerun once with --overwrite."
                )
        self._acquire_run_lock()
        if overwrite:
            generated_files = [
                self.metrics_path, self.predictions_path, self.history_path, self.shap_path,
                self.distribution_path, self.completed_path, self.error_path, self.failed_path,
                self.output / "metrics_mean_sd.csv", self.output / "skill_scores.csv",
                self.output / "paired_block_bootstrap.csv", self.output / "model_ranking.csv",
                self.output / "split_optimism_summary.csv",
                self.output / "forecastability_full_grid.csv",
                self.output / "forecastability_best_windows.csv",
                self.output / "data_audit.csv", self.output / "run_config.json",
                self.output / "environment.json", self.output / "run_status.json",
                self.output / "sample_support_audit.csv", self.output / "sample_manifest.csv",
            ]
            for path in generated_files:
                if path.exists():
                    path.unlink()
            figure_dir = self.output / "figures"
            if figure_dir.exists():
                shutil.rmtree(figure_dir)
        self.completed = set()
        if self.completed_path.exists():
            self.completed = set(self.completed_path.read_text(encoding="utf-8").splitlines())
        self.failed: dict[str, str] = {}
        if self.failed_path.exists():
            self.failed = json.loads(self.failed_path.read_text(encoding="utf-8"))
        self._repair_partial_outputs()
        data_dir = resolve_path(config, config["data_dir"])
        self.data = prepare_dataset(data_dir, config)
        self.data.audit.to_csv(self.output / "data_audit.csv", index=False, encoding="utf-8-sig")
        (self.output / "run_config.json").write_text(
            json.dumps(self.clean_config, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        (self.output / "environment.json").write_text(
            json.dumps(self._environment_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _environment_manifest(self) -> dict[str, object]:
        packages = {}
        for name in ["numpy", "pandas", "scikit-learn", "scipy", "torch", "shap"]:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        manifest: dict[str, object] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "config_sha256": self.config_sha256,
            "protocol_hash": self.protocol_hash,
        }
        try:
            import torch

            manifest.update({
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            })
        except ImportError:
            manifest.update({"torch_cuda_version": None, "cuda_available": False, "gpu": None})
        return manifest

    def _acquire_run_lock(self) -> None:
        """Prevent two terminals from writing the same experiment concurrently."""
        lock_path = self.output / ".run.lock"
        handle = lock_path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise RuntimeError(
                f"Experiment {self.experiment} is already running in another terminal. "
                "Stop that process before starting a second copy."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n".encode("ascii"))
        handle.flush()
        self._lock_handle = handle

    def _release_run_lock(self) -> None:
        handle = self._lock_handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._lock_handle = None

    def _repair_partial_outputs(self) -> None:
        """Drop rows from interrupted tasks and re-open tasks missing essential outputs."""
        essential_sets = []
        for path in [self.metrics_path, self.predictions_path]:
            if not path.exists():
                continue
            frame = _read_csv_resilient(path)
            if "task_id" not in frame.columns:
                continue
            essential_sets.append(set(frame["task_id"].astype(str)))
        if self.completed and essential_sets:
            valid_completed = self.completed.intersection(*essential_sets)
            if valid_completed != self.completed:
                self.completed = valid_completed
        for path, duplicate_keys in [
            (self.metrics_path, ["task_id", "horizon_step"]),
            (self.predictions_path, ["task_id", "valid_time", "horizon_step"]),
            (self.history_path, ["task_id", "epoch"]),
            (self.shap_path, ["task_id", "feature"]),
        ]:
            if not path.exists():
                continue
            frame = _read_csv_resilient(path)
            if "task_id" not in frame.columns:
                continue
            frame = frame[frame["task_id"].astype(str).isin(self.completed)]
            keys = [key for key in duplicate_keys if key in frame.columns]
            if keys:
                frame = frame.drop_duplicates(keys, keep="last")
            if frame.empty:
                path.unlink()
            else:
                frame.to_csv(path, index=False, encoding="utf-8-sig")
        if self.completed_path.exists() or self.completed:
            self.completed_path.write_text(
                "".join(task_id + "\n" for task_id in sorted(self.completed)),
                encoding="utf-8",
            )

    def _write_failed(self) -> None:
        self.failed_path.write_text(
            json.dumps(self.failed, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _mark_completed(self, task_id: str) -> None:
        if task_id not in self.completed:
            with self.completed_path.open("a", encoding="utf-8") as handle:
                handle.write(task_id + "\n")
            self.completed.add(task_id)
        if task_id in self.failed:
            self.failed.pop(task_id, None)
            self._write_failed()

    def _fit_method(
        self,
        method: Method,
        model_x: np.ndarray,
        sequence: SequenceData,
        split: dict[str, np.ndarray],
        seed: int,
    ):
        if method.model == "persistence":
            return sequence.persistence[split["test"]], [], 0, 0, None
        if method.model == "seasonal_naive":
            return sequence.seasonal_naive[split["test"]], [], 0, 0, None
        if method.model == "ridge":
            result = fit_ridge_predict(model_x, sequence.y, split)
        else:
            training = dict(self.config["training"])
            interpretability = self.config.get("interpretability", {})
            compute_shap = (
                bool(interpretability.get("enabled", False))
                and self.experiment == interpretability.get("experiment")
                and method.feature_set == interpretability.get("feature_set")
                and method.model == interpretability.get("model")
                and seed == int(interpretability.get("seed", seed))
            )
            training["compute_shap"] = compute_shap
            training["shap_background_samples"] = int(interpretability.get("background_samples", 128))
            training["shap_explanation_samples"] = int(interpretability.get("explanation_samples", 256))
            result = fit_torch_predict(
                method.model,
                model_x,
                sequence.y,
                split,
                seed,
                training,
            )
        return result.prediction, result.history, result.best_epoch, result.parameter_count, result.feature_importance

    def _run_group(
        self,
        target_station: str,
        target: str,
        lookback: int,
        horizon: int,
        methods: list[Method],
        subexperiment: str,
        split_mode: str = "official",
        rolling_folds: int | None = None,
        data_override: PreparedDataset | None = None,
        fold_override: int | None = None,
        split_name_override: str | None = None,
        support_lookback: int | None = None,
    ) -> None:
        if split_mode == "rolling" and data_override is None:
            fold_datasets = rolling_origin_datasets(self.data, self.config)
            if rolling_folds is not None:
                fold_datasets = fold_datasets[:rolling_folds]
            for fold_index, fold_data in enumerate(fold_datasets):
                if self.max_tasks is not None and self.tasks_run >= self.max_tasks:
                    break
                self._run_group(
                    target_station,
                    target,
                    lookback,
                    horizon,
                    methods,
                    subexperiment,
                    split_mode="official",
                    data_override=fold_data,
                    fold_override=fold_index,
                    split_name_override="rolling_origin_fold_specific_preprocessing",
                    support_lookback=support_lookback,
                )
            return

        data = data_override or self.data
        target_column = data.target_columns[target_station][target]
        columns_by_feature: dict[str, list[str]] = {}
        for method in methods:
            if method.feature_set in {"persistence", "seasonal_naive"}:
                continue
            columns_by_feature[method.feature_set] = feature_set_columns(
                data, method.feature_set, target_station, target
            )
        union_columns = list(dict.fromkeys(
            column for columns in columns_by_feature.values() for column in columns
        ))
        if not union_columns:
            union_columns = [target_column]
        sequence_lookback = max(lookback, support_lookback or lookback)
        sequence_builder = make_within_split_sequences if self.within_split_baseline else make_sequences
        sequence = sequence_builder(
            data, union_columns, target_column, sequence_lookback, horizon
        )
        if self.within_split_baseline:
            if split_mode != "official":
                raise ValueError("Within-split baseline only permits chronological partitions.")
            support_id = f"{self.protocol_hash}|{target_station}|{target}|{sequence_lookback}|{horizon}"
            audit, manifest = support_tables(data, sequence, sequence_lookback, horizon, {
                "support_id": support_id, "protocol_hash": self.protocol_hash,
                "baseline_protocol": WITHIN_SPLIT_PROTOCOL,
                "target_station": target_station, "target": target,
                "lookback_steps": sequence_lookback, "forecast_horizon_steps": horizon,
            })
            _upsert_csv(audit, self.output / "sample_support_audit.csv", "support_id")
            _upsert_csv(manifest, self.output / "sample_manifest.csv", "support_id")
            print("  within-split valid windows: " + ", ".join(
                f"{row.data_split}={row.valid_windows}" for row in audit.itertuples()
            ))
        if split_mode == "official":
            splits = [official_split(sequence)]
            split_names = [split_name_override or (
                WITHIN_SPLIT_PROTOCOL if self.within_split_baseline else "chronological_70_15_15"
            )]
        elif split_mode == "random_diagnostic":
            splits = [_random_split(
                len(sequence.x),
                self.config["split"]["train_fraction"],
                self.config["split"]["validation_fraction"],
                seed=42,
            )]
            split_names = ["random_window_diagnostic"]
        else:
            raise ValueError(split_mode)

        print(
            f"[{self.experiment}] {subexperiment}: station={target_station}, target={target}, "
            f"lookback={lookback}, support_lookback={sequence_lookback}, horizon={horizon}, "
            f"common_sequences={len(sequence.x)}"
        )
        for local_fold, (split, split_name) in enumerate(zip(splits, split_names)):
            fold = fold_override if fold_override is not None else local_fold
            diagnostic_id = "|".join(map(str, [
                self.protocol_hash, self.experiment, subexperiment, fold, split_name,
                target_station, target, lookback, horizon,
            ]))
            distribution_metadata = {
                "diagnostic_id": diagnostic_id,
                "protocol_hash": self.protocol_hash,
                "experiment": self.experiment,
                "subexperiment": subexperiment,
                "fold": fold,
                "split_scheme": split_name,
                "target_station": target_station,
                "target": target,
                "target_column": target_column,
                "lookback_steps": lookback,
                "lookback_days": lookback * 4 / 24,
                "forecast_horizon_steps": horizon,
            }
            _upsert_csv(
                distribution_shift_table(sequence, split, distribution_metadata),
                self.distribution_path,
                "diagnostic_id",
            )
            scale_function = training_mase_scale if self.within_split_baseline else mase_scale_from_training
            mase_scale = scale_function(sequence, split["train"])
            for method in methods:
                if self.max_tasks is not None and self.tasks_run >= self.max_tasks:
                    return
                if method.feature_set in {"persistence", "seasonal_naive"}:
                    feature_columns = [target_column]
                    model_x = sequence.x
                else:
                    feature_columns = columns_by_feature[method.feature_set]
                    positions = [sequence.feature_columns.index(column) for column in feature_columns]
                    model_x = sequence.x[:, -lookback:, positions]
                for seed in self.config["training"]["seeds"]:
                    if self.max_tasks is not None and self.tasks_run >= self.max_tasks:
                        return
                    task_id = "|".join(map(str, [
                        self.protocol_hash, self.experiment, subexperiment, fold, split_name,
                        target_station, target,
                        method.feature_set, method.model, seed, lookback, horizon,
                    ]))
                    if task_id in self.completed:
                        print(f"  skip completed: {method.feature_set}/{method.model}/seed={seed}/fold={fold}")
                        continue
                    metadata = {
                        "task_id": task_id,
                        "protocol_hash": self.protocol_hash,
                        "experiment": self.experiment,
                        "subexperiment": subexperiment,
                        "fold": fold,
                        "split_scheme": split_name,
                        "target_station": target_station,
                        "target": target,
                        "target_column": target_column,
                        "feature_set": method.feature_set,
                        "model": method.model,
                        "seed": seed,
                        "lookback_steps": lookback,
                        "lookback_days": lookback * 4 / 24,
                        "forecast_horizon_steps": horizon,
                        "n_features": len(feature_columns),
                        "n_train": len(split["train"]),
                        "n_validation": len(split["validation"]),
                        "n_test": len(split["test"]),
                    }
                    try:
                        prediction, history, best_epoch, parameter_count, feature_importance = self._fit_method(
                            method, model_x, sequence, split, seed
                        )
                        metadata["best_epoch"] = best_epoch
                        metadata["parameter_count"] = parameter_count
                        y_true = sequence.y[split["test"]]
                        metrics = pd.DataFrame(
                            calculate_metric_rows(y_true, prediction, metadata, mase_scale=mase_scale)
                        )
                        predictions = prediction_frame(
                            sequence.target_start_times[split["test"]], y_true, prediction, metadata
                        )
                        _append_csv(metrics, self.metrics_path)
                        _append_csv(predictions, self.predictions_path)
                        if history:
                            history_frame = pd.DataFrame(history)
                            for key, value in metadata.items():
                                history_frame[key] = value
                            _append_csv(history_frame, self.history_path)
                        if feature_importance is not None:
                            shap_frame = pd.DataFrame({
                                **{key: [value] * len(feature_columns) for key, value in metadata.items()},
                                "feature": feature_columns,
                                "mean_absolute_shap": feature_importance,
                            })
                            _append_csv(shap_frame, self.shap_path)
                        self._mark_completed(task_id)
                        self.tasks_run += 1
                        print(
                            f"  done: {method.feature_set}/{method.model}/seed={seed}/fold={fold}, "
                            f"test={len(split['test'])}"
                        )
                    except Exception as exc:
                        message = task_id + f"\n{type(exc).__name__}: {exc}\n" + traceback.format_exc()
                        with self.error_path.open("a", encoding="utf-8") as handle:
                            handle.write(message + "\n\n")
                        self.failed[task_id] = f"{type(exc).__name__}: {exc}"
                        self._write_failed()
                        print(f"  ERROR: {type(exc).__name__}: {exc}")
                        if not self.continue_on_error:
                            raise

    def run(self) -> Path:
        try:
            if self.experiment == "01_baseline":
                self._experiment_01()
            elif self.experiment == "02_station_ablation":
                self._experiment_02()
            elif self.experiment == "03_window_horizon":
                self._experiment_03()
            elif self.experiment == "04_meteorology":
                self._experiment_04()
            elif self.experiment == "05_robustness":
                self._experiment_05()
            self._postprocess()
            return self.output
        finally:
            self._release_run_lock()

    def _experiment_01(self) -> None:
        if self.model_override:
            official_methods = _baseline_methods() + [Method("all_stations", self.model_override)]
        else:
            official_methods = _baseline_methods()
            official_methods += [
                Method("all_stations", model)
                for model in self.config["baseline_models"]
                if model not in {"persistence", "seasonal_naive"}
            ]
        for target in self.config["targets"]:
            lookback = int(self.config["baseline_lookback_steps"])
            horizon = int(self.config["forecast_horizon_steps"])
            self._run_group(
                self.config["target_station"], target, lookback, horizon,
                official_methods, "unified_model_benchmark", "official",
            )
            if self.within_split_baseline:
                continue  # The strict protocol deliberately does not run random-window diagnostics.
            diagnostic_model = self.model_override or self.config.get("diagnostic_model", "lstm")
            self._run_group(
                self.config["target_station"], target, lookback, horizon,
                [Method("all_stations", diagnostic_model)],
                "split_optimism_diagnostic", "random_diagnostic",
            )

    def _experiment_02(self) -> None:
        model = self.model_override or self.config.get("diagnostic_model", "lstm")
        methods = _baseline_methods()
        methods += [Method(feature_set, model) for feature_set in [
            "local_target", "local_multivariate", "station_A", "station_B", "upstream_AB", "all_stations"
        ]]
        if model != "ridge":
            methods.append(Method("all_stations", "ridge"))
        for target in self.config["targets"]:
            self._run_group(
                self.config["target_station"], target,
                int(self.config["target_default_windows"][target]),
                int(self.config["forecast_horizon_steps"]),
                methods, "station_information_value", "official",
            )

    def _experiment_03(self) -> None:
        model = self.model_override or self.config.get("diagnostic_model", "lstm")
        common_support_lookback = max(int(value) for value in self.config["window_grid"])
        for target in self.config["targets"]:
            for lookback in self.config["window_grid"]:
                self._run_group(
                    self.config["target_station"], target, int(lookback),
                    int(self.config["forecast_horizon_steps"]),
                    _baseline_methods() + [Method("all_stations", model)],
                    "forecastability_surface", "official",
                    support_lookback=common_support_lookback,
                )

    def _experiment_04(self) -> None:
        model = self.model_override or self.config.get("diagnostic_model", "lstm")
        feature_sets = [
            "all_stations",
            "all_stations_weather",
            "all_stations_rain_lags",
            "all_stations_weather_rain_lags",
        ]
        methods = _baseline_methods() + [Method(feature_set, model) for feature_set in feature_sets]
        for target in self.config["targets"]:
            self._run_group(
                self.config["target_station"], target,
                int(self.config["target_default_windows"][target]),
                int(self.config["forecast_horizon_steps"]),
                methods, "conditional_meteorological_value", "official",
            )

    def _experiment_05(self) -> None:
        model = self.model_override or self.config.get("diagnostic_model", "lstm")
        rolling_methods = _baseline_methods() + [Method(feature_set, model) for feature_set in [
            "local_target", "upstream_AB", "all_stations"
        ]]
        for target in self.config["targets"]:
            self._run_group(
                self.config["target_station"], target,
                int(self.config["target_default_windows"][target]),
                int(self.config["forecast_horizon_steps"]),
                rolling_methods, "rolling_origin", "rolling",
            )
        spatial_methods = _baseline_methods() + [
            Method("local_target", model),
            Method("other_stations", model),
            Method("all_stations", model),
        ]
        for station in "ABC":
            for target in self.config["targets"]:
                if target not in self.data.target_columns[station]:
                    continue
                self._run_group(
                    station, target,
                    int(self.config["target_default_windows"][target]),
                    int(self.config["forecast_horizon_steps"]),
                    spatial_methods, "target_station_rotation", "official",
                )

    def _postprocess(self) -> None:
        # A final integrity pass also makes accidental concurrent launches converge
        # to one row per task instead of silently double-counting results.
        self._repair_partial_outputs()
        if not self.metrics_path.exists():
            self._write_run_status()
            return
        metrics = _read_csv_resilient(self.metrics_path)
        mean_sd_table(metrics).to_csv(self.output / "metrics_mean_sd.csv", index=False, encoding="utf-8-sig")
        skills = skill_scores_table(metrics)
        skills.to_csv(self.output / "skill_scores.csv", index=False, encoding="utf-8-sig")

        diagnostic_model = self.model_override or self.config.get("diagnostic_model", "lstm")
        if self.experiment == "01_baseline":
            official = metrics[metrics["subexperiment"] == "unified_model_benchmark"]
            ranking = official.groupby(
                ["target_station", "target", "feature_set", "model", "horizon_step", "horizon_hours"],
                as_index=False,
            ).agg(
                R2_mean=("R2", "mean"),
                R2_sd=("R2", "std"),
                MAE_mean=("MAE", "mean"),
                MAE_sd=("MAE", "std"),
                MASE_mean=("MASE", "mean"),
            )
            ranking["MAE_rank"] = ranking.groupby(
                ["target_station", "target", "horizon_hours"]
            )["MAE_mean"].rank(method="min", ascending=True)
            ranking["R2_rank"] = ranking.groupby(
                ["target_station", "target", "horizon_hours"]
            )["R2_mean"].rank(method="min", ascending=False)
            ranking.to_csv(self.output / "model_ranking.csv", index=False, encoding="utf-8-sig")

            if not self.within_split_baseline:
                chronological = metrics[
                    (metrics["subexperiment"] == "unified_model_benchmark")
                    & (metrics["model"] == diagnostic_model)
                ].groupby(["target_station", "target", "horizon_step", "horizon_hours"], as_index=False).agg(
                    chronological_R2_mean=("R2", "mean"),
                    chronological_R2_sd=("R2", "std"),
                )
                random = metrics[
                    (metrics["subexperiment"] == "split_optimism_diagnostic")
                    & (metrics["model"] == diagnostic_model)
                ].groupby(["target_station", "target", "horizon_step", "horizon_hours"], as_index=False).agg(
                    random_R2_mean=("R2", "mean"),
                    random_R2_sd=("R2", "std"),
                )
                optimism = chronological.merge(
                    random,
                    on=["target_station", "target", "horizon_step", "horizon_hours"],
                    how="inner",
                )
                optimism["random_minus_chronological_R2"] = (
                    optimism["random_R2_mean"] - optimism["chronological_R2_mean"]
                )
                optimism.to_csv(self.output / "split_optimism_summary.csv", index=False, encoding="utf-8-sig")

        if self.experiment == "03_window_horizon":
            thresholds = self.config.get("forecastability", {})
            grid, best = forecastability_envelope(
                metrics,
                skills,
                diagnostic_model,
                float(thresholds.get("minimum_R2", 0.0)),
                float(thresholds.get("minimum_MAE_skill_vs_best_naive", 0.0)),
            )
            grid.to_csv(self.output / "forecastability_full_grid.csv", index=False, encoding="utf-8-sig")
            best.to_csv(self.output / "forecastability_best_windows.csv", index=False, encoding="utf-8-sig")
        if not self.predictions_path.exists():
            self._write_run_status()
            return
        predictions = _read_csv_resilient(
            self.predictions_path, parse_dates=["target_start_time", "valid_time"]
        )
        bootstrap = self.config["bootstrap"]
        references: list[tuple[str, str, str | None]] = []
        if self.experiment == "01_baseline":
            references = [
                ("persistence", "persistence", "unified_model_benchmark"),
                ("seasonal_naive", "seasonal_naive", "unified_model_benchmark"),
                ("all_stations", diagnostic_model, "unified_model_benchmark"),
            ]
        elif self.experiment == "02_station_ablation":
            references = [
                ("persistence", "persistence", None),
                ("seasonal_naive", "seasonal_naive", None),
                ("all_stations", diagnostic_model, None),
            ]
        elif self.experiment == "04_meteorology":
            references = [
                ("seasonal_naive", "seasonal_naive", None),
                ("all_stations", diagnostic_model, None),
            ]
        elif self.experiment == "05_robustness":
            references = [
                ("seasonal_naive", "seasonal_naive", "rolling_origin"),
                ("all_stations", diagnostic_model, "rolling_origin"),
                ("seasonal_naive", "seasonal_naive", "target_station_rotation"),
                ("other_stations", diagnostic_model, "target_station_rotation"),
            ]
        frames = []
        for reference_feature, reference_model, subexperiment in references:
            selected = predictions
            if subexperiment is not None:
                selected = selected[selected["subexperiment"] == subexperiment]
            table = paired_block_bootstrap_table(
                selected,
                reference_feature,
                reference_model,
                int(bootstrap["block_length_steps"]),
                int(bootstrap["replicates"]),
                int(bootstrap["seed"]),
            )
            if not table.empty:
                frames.append(table)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                self.output / "paired_block_bootstrap.csv", index=False, encoding="utf-8-sig"
            )
        generate_experiment_figures(self.metrics_path, self.experiment, self.output)
        self._write_run_status()

    def _write_run_status(self) -> None:
        status = {
            "experiment": self.experiment,
            "completed_tasks": len(self.completed),
            "unresolved_failed_tasks": len(self.failed),
            "failed_task_ids": sorted(self.failed),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        if self.within_split_baseline:
            learned = [self.model_override] if self.model_override else list(dict.fromkeys(
                model for model in self.config["baseline_models"]
                if model not in {"persistence", "seasonal_naive"}
            ))
            expected = len(self.config["targets"]) * (2 + len(learned)) * len(self.config["training"]["seeds"])
            status.update({
                "baseline_protocol": WITHIN_SPLIT_PROTOCOL,
                "expected_tasks": expected,
                "run_status": "complete" if len(self.completed) == expected and not self.failed else (
                    "audit_only" if self.max_tasks == 0 and not self.completed else "partial"
                ),
            })
        (self.output / "run_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )
