"""Common-support random-window experiments, validation-only window selection.

This module does not alter shared data/models/runner code or the 01/02 protocols.
The main and long-history experiments intentionally use separate support pools.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from .baseline_random import (
    PROTOCOL as SPLIT_PROTOCOL, SPLITS, legacy_random_split,
    prepare_random_pool, random_support_tables,
)
from .baseline_within_split import training_mase_scale
from .config import resolve_path
from .data import SequenceData, feature_set_columns
from .evaluation import calculate_metric_rows, distribution_shift_table, mean_sd_table, prediction_frame
from .models import fit_ridge_predict, fit_torch_predict
from .runner import ExperimentRunner, _append_csv, _read_csv_resilient, _upsert_csv


PROTOCOL = "window_horizon_random_common72h_v1"
GRIDS = {"main": [3] + list(range(6, 85, 6)), "long": [18, 42, 84, 126, 168]}
SUPPORT = {"main": 84, "long": 168}
BASELINES = ("persistence", "seasonal_naive")


def validate_config(config):
    if config.get("window_horizon_protocol") != PROTOCOL or config.get("baseline_protocol") != SPLIT_PROTOCOL:
        raise ValueError("Use a new window_horizon_random_* configuration, not paper.json.")
    phase = config.get("window_phase")
    if phase not in GRIDS:
        raise ValueError("window_phase must be main or long.")
    if config.get("frequency") != "4h" or config.get("target_station") != "C":
        raise ValueError("03 fixes 4h frequency and target C.")
    if config.get("forecast_horizon_steps") != 18 or config.get("support_lookback_steps") != SUPPORT[phase]:
        raise ValueError("03 fixes 18 outputs and phase-specific maximum history support.")
    grid = config.get("window_grid", [])
    if not grid or grid != sorted(set(grid)) or not set(grid).issubset(GRIDS[phase]):
        raise ValueError("Window steps must be unique, sorted, and part of the predefined phase grid.")
    if not config.get("smoke_test", False) and grid != GRIDS[phase]:
        raise ValueError("Formal configs must include the full predefined window grid.")
    if (config["split"]["train_fraction"], config["split"]["validation_fraction"], config.get("split_seed")) != (.8, .1, 42):
        raise ValueError("Use the 01/02 two-call 80/10/10 split algorithm with seed42.")
    if config["missing_data"]["interpolation_policy"] != "causal_ffill":
        raise ValueError("Use causal_ffill input preprocessing.")
    if config.get("window_model", "lstm") not in {"lstm", "ridge"}:
        raise ValueError("window_model must be lstm, or ridge for a labeled CPU check.")
    if not 0 <= float(config.get("near_optimal_tolerance", .02)) < 1:
        raise ValueError("near_optimal_tolerance must be between 0 and 1.")
    for key, values in (("targets", config["targets"]), ("training.seeds", config["training"]["seeds"])):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{key} must be nonempty and unique.")


def make_common_pool(data, target, support_steps, horizon=18):
    """Full maximum history is screened ONCE; labels must all be observed.

    A future seasonal-naive forecast repeats the last observed six-step day.
    Its source indices NEVER advance into the forecast period.
    """
    if support_steps < 6 or horizon < 1:
        raise ValueError("Common history must contain the last full 24h day.")
    columns = feature_set_columns(data, "all_stations", "C", target)
    target_column = data.target_columns["C"][target]
    x = data.feature_frame[columns].to_numpy(dtype=np.float32)
    observed = data.raw_frame[target_column].to_numpy(dtype=np.float32)
    historical = data.feature_frame[target_column].to_numpy(dtype=np.float32)
    index = pd.DatetimeIndex(data.raw_frame.index)
    starts = []
    for i in range(support_steps, len(index) - horizon + 1):
        if (np.isfinite(x[i - support_steps:i]).all()
                and np.isfinite(observed[i:i + horizon]).all()
                and np.isfinite(historical[i - 6:i]).all()):
            starts.append(i)
    if not starts:
        raise ValueError(f"No common windows for {target}, support={support_steps}, horizon={horizon}.")
    starts = np.asarray(starts, dtype=int)
    return SequenceData(
        x=np.stack([x[i - support_steps:i] for i in starts]),
        y=np.stack([observed[i:i + horizon] for i in starts]),
        persistence=np.stack([np.repeat(historical[i - 1], horizon) for i in starts]),
        seasonal_naive=np.stack([historical[i - 6 + np.arange(horizon) % 6] for i in starts]),
        split_labels=np.full(len(starts), "unassigned", dtype=object),
        issue_times=index[starts - 1], target_start_times=index[starts],
        target_end_times=index[starts + horizon - 1], feature_columns=columns, target_column=target_column,
    )


def assign_pool(data, config, target):
    sequence = make_common_pool(data, target, int(config["support_lookback_steps"]), 18)
    split = legacy_random_split(len(sequence.x), .8, .1, 42)
    labels = np.empty(len(sequence.x), dtype=object)
    for name in SPLITS:
        labels[split[name]] = name
    return replace(sequence, split_labels=labels), split


def fit_with_validation(model, x, sequence, split, seed, training):
    """Fit ONCE. Obtain validation and test inference from the SAME selected weights.

    Existing model fitting uses train for scaling/updates and validation for early
    stopping. Only the final inference index list is extended; test observations
    are never used by early stopping or by window selection.
    """
    prediction_indices = np.concatenate([split["validation"], split["test"]])
    inference_split = dict(split, test=prediction_indices)
    if model == "ridge":
        result = fit_ridge_predict(x, sequence.y, inference_split)
    else:
        options = dict(training, compute_shap=False)
        result = fit_torch_predict(model, x, sequence.y, inference_split, seed, options)
    n_validation = len(split["validation"])
    return replace(result, prediction=result.prediction[n_validation:],
                   validation_prediction=result.prediction[:n_validation])


def window_skills(metrics):
    """Naive tasks are evaluated once per seed, then matched to all windows."""
    if metrics.empty:
        return pd.DataFrame()
    keys = ["target", "seed", "horizon_hours"]
    result = metrics[~metrics.model.isin(BASELINES)].copy()
    for baseline in BASELINES:
        ref = metrics[metrics.model == baseline][keys + ["MAE", "MSE"]].rename(
            columns={"MAE": f"MAE_{baseline}", "MSE": f"MSE_{baseline}"})
        result = result.merge(ref, on=keys, how="left", validate="many_to_one")
        result[f"MAE_skill_vs_{baseline}"] = 1 - result.MAE / result[f"MAE_{baseline}"].replace(0, np.nan)
    result["MAE_best_naive"] = result[[f"MAE_{b}" for b in BASELINES]].min(axis=1)
    result["MAE_skill_vs_best_naive"] = 1 - result.MAE / result.MAE_best_naive.replace(0, np.nan)
    return result


def select_windows(validation_metrics, config):
    """No test-metric argument by design. Require ALL windows x seeds x 18 leads."""
    model = config.get("window_model", "lstm")
    grid, seeds = config["window_grid"], config["training"]["seeds"]
    selected = validation_metrics[validation_metrics.model == model]
    summary, selections, progress = [], [], []
    for target in config["targets"]:
        group = selected[selected.target == target]
        expected = {(window, seed, hour) for window in grid for seed in seeds for hour in range(4, 73, 4)}
        actual = set(group[["lookback_steps", "seed", "horizon_hours"]].itertuples(index=False, name=None))
        complete = actual == expected and len(group) == len(expected)
        progress.append(dict(target=target, status="complete" if complete else "pending_all_windows_and_seeds",
                             expected_validation_rows=len(expected), available_validation_rows=len(group)))
        if not complete:
            continue
        # Every seed contributes equally; each of the 18 horizons receives equal weight.
        for hours in [0] + list(range(4, 73, 4)):
            score_rows = group if hours == 0 else group[group.horizon_hours == hours]
            by_seed = score_rows.groupby(["lookback_steps", "seed"]).MAE.mean().reset_index()
            scores = by_seed.groupby("lookback_steps").MAE.agg(["mean", "std"]).reset_index().sort_values("lookback_steps")
            best = scores.sort_values(["mean", "lookback_steps"]).iloc[0]
            threshold = float(best["mean"]) * (1 + float(config["near_optimal_tolerance"]))
            near = scores[scores["mean"] <= threshold + 1e-12]
            shortest = int(near.lookback_steps.min())
            selections.append(dict(
                target=target, model=model, horizon_hours=hours,
                horizon_scope="equal_mean_of_18_horizons" if hours == 0 else "individual_horizon",
                best_lookback_steps=int(best.lookback_steps), best_lookback_days=float(best.lookback_steps) / 6,
                best_validation_MAE=float(best["mean"]), shortest_near_optimal_steps=shortest,
                shortest_near_optimal_days=shortest / 6,
                near_optimal_tolerance=float(config["near_optimal_tolerance"]),
                near_optimal_windows_steps="|".join(str(int(v)) for v in near.lookback_steps),
                selection_source="validation_only", n_seeds=len(seeds),
            ))
            for row in scores.itertuples(index=False):
                summary.append(dict(target=target, model=model, horizon_hours=hours,
                    lookback_steps=row.lookback_steps, lookback_days=row.lookback_steps / 6,
                    validation_MAE_mean=row.mean, validation_MAE_seed_sd=row.std,
                    validation_relative_excess_percent=100 * (row.mean / best["mean"] - 1) if best["mean"] > 0 else np.nan,
                    within_near_optimal_tolerance=bool(row.mean <= threshold + 1e-12),
                    selection_source="validation_only"))
    return pd.DataFrame(summary), pd.DataFrame(selections), pd.DataFrame(progress)


def describe_selected_test_windows(test_metrics, selections):
    """Describe only windows already selected on validation; never rank test scores."""
    rows = []
    if selections.empty:
        return pd.DataFrame()
    for selection in selections.to_dict("records"):
        for name, steps in [("minimum_validation_MAE", selection["best_lookback_steps"]),
                            ("shortest_within_validation_tolerance", selection["shortest_near_optimal_steps"])]:
            group = test_metrics[(test_metrics.target == selection["target"])
                                 & (test_metrics.model == selection["model"])
                                 & (test_metrics.lookback_steps == steps)]
            hours = selection["horizon_hours"]
            if hours:
                group = group[group.horizon_hours == hours]
            per_seed = group.groupby("seed")[["MAE", "RMSE", "R2"]].mean()
            rows.append(dict(target=selection["target"], horizon_hours=hours,
                             horizon_scope=selection["horizon_scope"], selection_rule=name,
                             selected_lookback_steps=steps, selected_lookback_days=steps / 6,
                             selection_source="validation_only",
                             **{f"test_{metric}_mean": per_seed[metric].mean() for metric in per_seed},
                             **{f"test_{metric}_seed_sd": per_seed[metric].std() for metric in per_seed}))
    return pd.DataFrame(rows)


def render_figures(metrics, skills, out, phase):
    if metrics.empty:
        return
    import matplotlib.pyplot as plt
    import seaborn as sns
    out.mkdir(exist_ok=True)
    modeled = metrics[~metrics.model.isin(BASELINES)]
    for target, group in modeled.groupby("target"):
        for name, table, value, center in [
            ("R2", group, "R2", None),
            ("MAE_skill", skills[skills.target == target], "MAE_skill_vs_best_naive", 0.),
        ]:
            if table.empty:
                continue
            matrix = table.pivot_table(index="lookback_days", columns="horizon_hours", values=value).sort_index()
            fig, ax = plt.subplots(figsize=(12, max(4, len(matrix) * .35)))
            sns.heatmap(matrix, cmap="viridis" if name == "R2" else "vlag", center=center,
                        annot=False, ax=ax)  # Autoscale: negative R2 is never clipped.
            ax.set_xlabel("Forecast horizon (h)");ax.set_ylabel("Input history (days)")
            ax.set_title(f"{target}: {phase} / random-window holdout / {name} (seed mean)")
            fig.tight_layout();fig.savefig(out / f"window_horizon_{target}_{name}.png", dpi=200);plt.close(fig)


class RandomWindowRunner(ExperimentRunner):
    def __init__(self, config, **kwargs):
        validate_config(config)
        if kwargs.get("overwrite") or kwargs.get("model_override"):
            raise ValueError("Change results_dir for a changed protocol; overwrite/model override is disabled.")
        super().__init__(config, "03_window_horizon", **kwargs)
        try:
            signatures = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in sorted(resolve_path(config, config["data_dir"]).glob("*.xlsx"))}
            sources = ["window_horizon_random.py", "baseline_random.py", "baseline_within_split.py",
                       "data.py", "models.py", "runner.py", "evaluation.py", "config.py"]
            code = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources}
            for name, payload in [("data_fingerprints.json", signatures), ("source_fingerprints.json", code)]:
                path = self.output / name
                if path.exists() and json.loads(path.read_text()) != payload:
                    raise RuntimeError(f"{name}: data/source changed; use a new results_dir.")
                path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            self.data = prepare_random_pool(self.data, config)
            audit = self.data.audit.drop(columns=["splitwise_missing_cells_after_imputation"], errors="ignore").copy()
            if "source" in audit:
                audit["random_pool_missing_cells_after_imputation"] = [
                    int(self.data.feature_frame[self.data.station_columns[source]].isna().sum().sum())
                    if source in self.data.station_columns else int(self.data.feature_frame[self.data.weather_columns].isna().sum().sum())
                    for source in audit.source]
            audit.to_csv(self.output / "data_audit.csv", index=False, encoding="utf-8-sig")
        except BaseException:
            self._release_run_lock();raise

    def _repair_partial_outputs(self):
        super()._repair_partial_outputs()
        path = self.output / "validation_metrics.csv"
        if self.completed:
            if path.exists():
                frame = _read_csv_resilient(path).drop_duplicates(["task_id", "horizon_step"], keep="last")
                valid = set(frame.groupby("task_id").horizon_step.apply(
                    lambda s: set(s) == set(range(1, 19))).loc[lambda s: s].index)
                self.completed.intersection_update(valid)
            else:
                self.completed.clear()
            super()._repair_partial_outputs()
        for name, keys in [("validation_metrics.csv", ["task_id", "horizon_step"]),
                           ("model_diagnostics.csv", ["task_id"])]:
            output = self.output / name
            if output.exists():
                frame = _read_csv_resilient(output)
                frame = frame[frame.task_id.isin(self.completed)].drop_duplicates(keys, keep="last")
                if frame.empty:
                    output.unlink()
                else:
                    frame.to_csv(output, index=False, encoding="utf-8-sig")

    def run(self):
        try:
            self._experiment_03();self._postprocess();return self.output
        except BaseException:
            self._write_run_status();raise
        finally:
            self._release_run_lock()

    def _experiment_03(self):
        config = self.config
        for target in config["targets"]:
            sequence, split = assign_pool(self.data, config, target)
            support_id = f"{self.protocol_hash}|{config['window_phase']}|C|{target}|{config['support_lookback_steps']}|18"
            common = dict(protocol_hash=self.protocol_hash, support_id=support_id,
                experiment=self.experiment, subexperiment=f"window_horizon_{config['window_phase']}",
                fold=0, split_scheme=SPLIT_PROTOCOL, target_station="C", target=target,
                target_column=sequence.target_column, forecast_horizon_steps=18,
                support_lookback_steps=config["support_lookback_steps"])
            audit, manifest = random_support_tables(self.data, sequence, split, {
                "support_id": support_id, "protocol_hash": self.protocol_hash,
                "target_station": "C", "target": target, "baseline_protocol": SPLIT_PROTOCOL,
                "window_phase": config["window_phase"], "window_horizon_protocol": PROTOCOL,
            })
            _upsert_csv(audit, self.output / "sample_support_audit.csv", "support_id")
            _upsert_csv(manifest, self.output / "sample_manifest.csv", "support_id")
            _upsert_csv(distribution_shift_table(sequence, split, {**common, "diagnostic_id": support_id}),
                        self.distribution_path, "diagnostic_id")
            grid_audit = []
            for lookback in config["window_grid"]:
                for name in SPLITS:
                    ids = split[name]
                    grid_audit.append(dict(support_id=support_id, target=target, data_split=name,
                        lookback_steps=lookback, lookback_days=lookback / 6, valid_windows=len(ids),
                        n_features=len(sequence.feature_columns),
                        ordered_sample_indices_sha256=hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                        actual_input_start_min=(sequence.target_start_times[ids] - pd.Timedelta(hours=4 * lookback)).min(),
                        actual_input_start_max=(sequence.target_start_times[ids] - pd.Timedelta(hours=4 * lookback)).max(),
                        common_target_start_times_sha256=audit.loc[audit.data_split == name, "target_start_times_sha256"].iloc[0],
                        seasonal_naive_policy="repeat_last_observed_24h_pattern_for_all_18_leads",
                        baseline_history_steps=6, all_baseline_source_times_before_issue_or_at_issue=True))
            _upsert_csv(pd.DataFrame(grid_audit), self.output / "window_input_audit.csv", "support_id")
            scale = training_mase_scale(sequence, split["train"])
            print(f"[03/{config['window_phase']}] {target}: common_windows={len(sequence.x)}, " +
                  ", ".join(f"{name}={len(split[name])}" for name in SPLITS), flush=True)
            tasks = [(baseline, 0) for baseline in BASELINES]
            tasks += [(config.get("window_model", "lstm"), steps) for steps in config["window_grid"]]
            for model, lookback in tasks:
                for seed in config["training"]["seeds"]:
                    if self.max_tasks is not None and self.tasks_run >= self.max_tasks:
                        break
                    task_id = "|".join(map(str, [self.protocol_hash, self.experiment, config["window_phase"], target, model, lookback, seed, 18]))
                    if task_id in self.completed:
                        print(f"  skip: {target}/{model}/L{lookback}/seed={seed}", flush=True);continue
                    metadata = dict(common, task_id=task_id, feature_set=model if model in BASELINES else "all_stations",
                        model=model, seed=seed, lookback_steps=lookback, lookback_days=lookback / 6,
                        n_features=1 if model in BASELINES else len(sequence.feature_columns),
                        n_train=len(split["train"]), n_validation=len(split["validation"]), n_test=len(split["test"]))
                    try:
                        if model in BASELINES:
                            all_predictions = sequence.persistence if model == "persistence" else sequence.seasonal_naive
                            prediction, validation_prediction = all_predictions[split["test"]], all_predictions[split["validation"]]
                            history, best_epoch, parameters = [], 0, 0
                        else:
                            x = np.ascontiguousarray(sequence.x[:, -lookback:, :])
                            result = fit_with_validation(model, x, sequence, split, seed, config["training"])
                            prediction, validation_prediction = result.prediction, result.validation_prediction
                            history, best_epoch, parameters = result.history, result.best_epoch, result.parameter_count
                        for name, values in [("test", prediction), ("validation", validation_prediction)]:
                            if values.shape != sequence.y[split[name]].shape or not np.isfinite(values).all():
                                raise ValueError(f"Invalid {name} prediction shape or nonfinite values.")
                        metadata.update(best_epoch=best_epoch, parameter_count=parameters)
                        _append_csv(pd.DataFrame(calculate_metric_rows(sequence.y[split["test"]], prediction, metadata, scale)), self.metrics_path)
                        validation_rows = pd.DataFrame(calculate_metric_rows(
                            sequence.y[split["validation"]], validation_prediction, metadata, scale))
                        # The shared evaluator calls its scored count n_test. Keep the
                        # actual test partition count separate in a validation table.
                        validation_rows = validation_rows.rename(columns={"n_test": "n_scored"})
                        validation_rows["n_test"] = len(split["test"])
                        validation_rows["evaluation_split"] = "validation"
                        _append_csv(validation_rows, self.output / "validation_metrics.csv")
                        _append_csv(prediction_frame(sequence.target_start_times[split["test"]], sequence.y[split["test"]], prediction, metadata), self.predictions_path)
                        if history:
                            frame = pd.DataFrame(history)
                            for key, value in metadata.items():
                                frame[key] = value
                            _append_csv(frame, self.history_path)
                        last_epoch = int(history[-1]["epoch"]) if history else 0
                        _append_csv(pd.DataFrame([dict(metadata, epochs_run=last_epoch,
                            max_epochs=config["training"]["max_epochs"],
                            reached_epoch_cap=bool(history and last_epoch >= config["training"]["max_epochs"]),
                            best_epoch_near_cap=bool(history and best_epoch >= .95 * config["training"]["max_epochs"]))]),
                            self.output / "model_diagnostics.csv")
                        self._mark_completed(task_id);self.tasks_run += 1
                        print(f"  done: {target}/{model}/L{lookback}/seed={seed}, best_epoch={best_epoch}, epochs={last_epoch}", flush=True)
                    except Exception as exc:
                        with self.error_path.open("a", encoding="utf-8") as handle:
                            handle.write(task_id + "\n" + traceback.format_exc() + "\n")
                        self.failed[task_id] = f"{type(exc).__name__}: {exc}";self._write_failed();raise

    def _postprocess(self):
        self._repair_partial_outputs()
        if self.metrics_path.exists():
            metrics = _read_csv_resilient(self.metrics_path)
            validation = _read_csv_resilient(self.output / "validation_metrics.csv")
            mean_sd_table(metrics).to_csv(self.output / "metrics_mean_sd.csv", index=False, encoding="utf-8-sig")
            mean_sd_table(validation).to_csv(self.output / "validation_metrics_mean_sd.csv", index=False, encoding="utf-8-sig")
            skills = window_skills(metrics)
            skills.to_csv(self.output / "skill_scores.csv", index=False, encoding="utf-8-sig")
            if not skills.empty:
                grid = metrics[~metrics.model.isin(BASELINES)].groupby(["target", "lookback_steps", "lookback_days", "horizon_hours"], as_index=False).agg(
                    R2_mean=("R2", "mean"), R2_seed_sd=("R2", "std"), MAE_mean=("MAE", "mean"), MAE_seed_sd=("MAE", "std"))
                gain = skills.groupby(["target", "lookback_steps", "horizon_hours"], as_index=False).MAE_skill_vs_best_naive.mean()
                grid.merge(gain, on=["target", "lookback_steps", "horizon_hours"]).to_csv(
                    self.output / "window_horizon_full_grid.csv", index=False, encoding="utf-8-sig")
            summary, selections, progress = select_windows(validation, self.config)
            summary.to_csv(self.output / "validation_window_summary.csv", index=False, encoding="utf-8-sig")
            selections.to_csv(self.output / "validation_window_selection.csv", index=False, encoding="utf-8-sig")
            progress.to_csv(self.output / "window_selection_status.csv", index=False, encoding="utf-8-sig")
            describe_selected_test_windows(metrics, selections).to_csv(self.output / "test_metrics_for_validation_selected_windows.csv", index=False, encoding="utf-8-sig")
            render_figures(metrics, skills, self.output / "figures", self.config["window_phase"])
        self._write_run_status()

    def _write_run_status(self):
        super()._write_run_status()
        path = self.output / "run_status.json"
        status = json.loads(path.read_text())
        n_models = len(self.config["targets"]) * len(self.config["training"]["seeds"]) * len(self.config["window_grid"])
        n_naive = len(self.config["targets"]) * len(self.config["training"]["seeds"]) * 2
        expected = n_models + n_naive
        status.update(window_horizon_protocol=PROTOCOL, window_phase=self.config["window_phase"],
            expected_tasks=expected, expected_lstm_tasks=n_models if self.config.get("window_model", "lstm") == "lstm" else 0,
            expected_naive_tasks=n_naive, forecast_horizon_steps=18, window_grid=self.config["window_grid"],
            support_lookback_steps=self.config["support_lookback_steps"],
            window_selection_source="validation_only", near_optimal_tolerance=self.config["near_optimal_tolerance"],
            run_status="complete" if len(self.completed) == expected and not self.failed else (
                "audit_only" if self.max_tasks == 0 and not self.completed else "partial"),
            smoke_test=self.config.get("smoke_test", False),
            evaluation_scope="fixed_random_window_holdout_not_temporally_independent")
        path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
