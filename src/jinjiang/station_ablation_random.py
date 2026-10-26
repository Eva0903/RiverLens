"""02: fixed-window station ablation, paired samples and explicit random-holdout scope."""
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
from .data import feature_set_columns, make_sequences
from .evaluation import (
    calculate_metric_rows, distribution_shift_table, mean_sd_table,
    prediction_frame, skill_scores_table,
)
from .plots import _line_figure
from .runner import ExperimentRunner, Method, _append_csv, _read_csv_resilient, _upsert_csv


PROTOCOL = "station_ablation_random_common18_v1"
ARMS = ("local_target", "local_multivariate", "station_A", "station_B", "upstream_AB", "all_stations")
CONTRASTS = (
    ("upstream_increment", "local_multivariate", "all_stations", "primary"),
    ("local_multivariate_increment", "local_target", "local_multivariate", "secondary"),
    ("AB_vs_A", "station_A", "upstream_AB", "secondary"),
    ("AB_vs_B", "station_B", "upstream_AB", "secondary"),
    ("local_increment_over_upstream", "upstream_AB", "all_stations", "secondary"),
    ("network_vs_local_target", "local_target", "all_stations", "secondary"),
)
SCOPE = "conditional_on_fixed_random_window_split_and_fitted_models"


def validate_config(config):
    if config.get("station_ablation_protocol") != PROTOCOL:
        raise ValueError("Use configs/station_ablation_random_gpu.json, not the old paper.json.")
    if config.get("baseline_protocol") != SPLIT_PROTOCOL:
        raise ValueError("02 must use the same random-window split protocol as 01.")
    if config.get("frequency") != "4h" or config.get("target_station") != "C":
        raise ValueError("This experiment fixes 4h frequency and target station C.")
    if config["missing_data"]["interpolation_policy"] != "causal_ffill":
        raise ValueError("Use causal_ffill input preprocessing.")
    if (config["split"]["train_fraction"], config["split"]["validation_fraction"]) != (.8, .1):
        raise ValueError("This protocol fixes 80/10/10 random-window holdout.")
    if config.get("split_seed") != 42:
        raise ValueError("Keep split_seed=42 to match the completed 01.")
    if config.get("baseline_lookback_steps") != 18 or config.get("forecast_horizon_steps") != 6:
        raise ValueError("02 fixes 18 input steps and 6 output steps for every indicator.")
    if config.get("station_ablation_model", "lstm") not in {"lstm", "ridge"}:
        raise ValueError("station_ablation_model must be lstm (formal) or ridge (CPU verification).")
    for key, values in (("targets", config["targets"]), ("training.seeds", config["training"]["seeds"])):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{key} must be nonempty and unique.")
    if int(config["bootstrap"]["block_length_steps"]) < 1 or int(config["bootstrap"]["replicates"]) < 1:
        raise ValueError("Bootstrap length and replicates must be positive.")


def common_station_pool(data, config, target):
    """Build ONE all-station sample pool. Smaller arms only select its columns."""
    columns = feature_set_columns(data, "all_stations", "C", target)
    sequence = make_sequences(data, columns, data.target_columns["C"][target], 18, 6)
    split = legacy_random_split(len(sequence.x), .8, .1, config["split_seed"])
    # The random memberships, rather than make_sequences' temporary all-train labels.
    labels = np.empty(len(sequence.x), dtype=object)
    for name in SPLITS:
        labels[split[name]] = name
    sequence = replace(sequence, split_labels=labels)
    arms = {}
    for name in ARMS:
        selected = feature_set_columns(data, name, "C", target)
        if not selected:
            raise ValueError(f"No available input columns for {name}.")
        positions = [columns.index(column) for column in selected]
        arms[name] = replace(sequence, x=sequence.x[:, :, positions], feature_columns=selected)
    return sequence, split, arms


def paired_seed_gains(metrics, model):
    """Positive MAE/RMSE reduction and positive R2 gain favor the augmented arm."""
    selected = metrics[metrics.model == model]
    keys = ["target", "seed", "horizon_hours"]
    rows = []
    for contrast, reference, augmented, role in CONTRASTS:
        a = selected[selected.feature_set == reference]
        b = selected[selected.feature_set == augmented]
        merged = a.merge(b, on=keys, suffixes=("_ref", "_aug"), validate="one_to_one")
        for row in merged.to_dict("records"):
            rows.append({
                **{key: row[key] for key in keys}, "contrast": contrast, "role": role,
                "reference_feature_set": reference, "augmented_feature_set": augmented,
                "model": model, "MAE_reference": row["MAE_ref"], "MAE_augmented": row["MAE_aug"],
                "MAE_reduction": row["MAE_ref"] - row["MAE_aug"],
                "MAE_reduction_percent": 100 * (1 - row["MAE_aug"] / row["MAE_ref"]) if row["MAE_ref"] > 0 else np.nan,
                "RMSE_reduction": row["RMSE_ref"] - row["RMSE_aug"],
                "R2_gain": row["R2_aug"] - row["R2_ref"],
            })
    return pd.DataFrame(rows)


def calendar_block_ci(delta, block_length, replicates, seed):
    """Resample calendar days, including days without randomly held-out windows."""
    delta = delta.sort_index()
    origin = delta.index.min().floor(f"{4 * block_length}h")
    block_ids = np.asarray((delta.index - origin) / pd.Timedelta(hours=4 * block_length), dtype=int)
    nblocks = int(block_ids.max()) + 1
    totals = np.bincount(block_ids, weights=delta.to_numpy(), minlength=nblocks)
    counts = np.bincount(block_ids, minlength=nblocks)
    rng = np.random.default_rng(seed)
    samples = []
    # Bound memory even if a user supplies a very long record or many replicates.
    for begin in range(0, replicates, 100):
        draws = rng.integers(0, nblocks, size=(min(100, replicates - begin), nblocks))
        denominators = counts[draws].sum(axis=1)
        numerators = totals[draws].sum(axis=1)
        samples.extend((numerators[denominators > 0] / denominators[denominators > 0]).tolist())
    low, high = np.quantile(samples, [.025, .975]) if len(delta) >= 10 and samples else (np.nan, np.nan)
    return dict(MAE_reduction=float(delta.mean()), ci95_lower=float(low), ci95_upper=float(high),
                n_test_origins=len(delta), n_calendar_blocks=nblocks,
                n_nonempty_calendar_blocks=int((counts > 0).sum()))


def paired_calendar_gains(predictions, model, expected_seeds, bootstrap):
    """Average seed losses first; report per-lead and equal-six-lead contrasts.

    Only seeds with BOTH complete six-lead arms enter any contrast. Intervals are
    conditional summaries, not split uncertainty or independent future validation.
    """
    selected = predictions[predictions.model == model]
    rows = []
    keys = ["seed", "target_start_time", "horizon_hours"]
    for target, group in selected.groupby("target"):
        for contrast, reference, augmented, role in CONTRASTS:
            ref = group[group.feature_set == reference]
            aug = group[group.feature_set == augmented]
            completed_seeds = []
            for seed in expected_seeds:
                a = ref[ref.seed == seed][keys]
                b = aug[aug.seed == seed][keys]
                if a.empty or b.empty or set(a.horizon_hours) != {4, 8, 12, 16, 20, 24}:
                    continue
                if not a.sort_values(keys).reset_index(drop=True).equals(b.sort_values(keys).reset_index(drop=True)):
                    continue
                completed_seeds.append(seed)
            if not completed_seeds:
                continue
            ref = ref[ref.seed.isin(completed_seeds)]
            aug = aug[aug.seed.isin(completed_seeds)]
            paired = ref.merge(aug, on=keys, suffixes=("_ref", "_aug"), validate="one_to_one")
            if not np.array_equal(paired.y_true_ref.to_numpy(), paired.y_true_aug.to_numpy()):
                raise AssertionError("Paired arms must use the same observed labels.")
            paired["reduction"] = paired.absolute_error_ref - paired.absolute_error_aug
            per_seed = paired.groupby(["seed", "horizon_hours"]).reduction.mean().unstack()
            per_seed_mean = per_seed.mean(axis=1)
            per_origin = paired.groupby(["target_start_time", "horizon_hours"]).reduction.mean().unstack()
            for hours in [0, 4, 8, 12, 16, 20, 24]:
                delta = per_origin.mean(axis=1) if hours == 0 else per_origin[hours]
                seed_delta = per_seed_mean if hours == 0 else per_seed[hours]
                result = calendar_block_ci(delta, int(bootstrap["block_length_steps"]),
                                           int(bootstrap["replicates"]), int(bootstrap["seed"]))
                rows.append(dict(
                    target=target, contrast=contrast, role=role, reference_feature_set=reference,
                    augmented_feature_set=augmented, model=model, horizon_hours=hours,
                    horizon_scope="equal_mean_of_six_horizons" if hours == 0 else "individual_horizon",
                    n_paired_seeds=len(completed_seeds), paired_seeds="|".join(map(str, completed_seeds)),
                    n_seeds_with_positive_reduction=int((seed_delta > 0).sum()),
                    seed_mean_MAE_reduction_sd=float(seed_delta.std(ddof=1)),
                    status="complete" if set(completed_seeds) == set(expected_seeds) else "partial_paired_seeds",
                    block_length_steps=int(bootstrap["block_length_steps"]),
                    bootstrap_replicates=int(bootstrap["replicates"]),
                    interval_scope=SCOPE,
                    resampling="nonoverlapping_calendar_blocks_of_forecast_origins_preserving_empty_intervals",
                    multiple_comparison_adjustment="none_secondary_contrasts_exploratory", **result,
                ))
    return pd.DataFrame(rows)


class RandomStationRunner(ExperimentRunner):
    def __init__(self, config, **kwargs):
        validate_config(config)
        if kwargs.get("overwrite") or kwargs.get("model_override"):
            raise ValueError("Use a new results_dir to change protocol; no overwrite/model override.")
        super().__init__(config, "02_station_ablation", **kwargs)
        try:
            files = sorted(resolve_path(config, config["data_dir"]).glob("*.xlsx"))
            signatures = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
            source_names = ["station_ablation_random.py", "baseline_random.py", "baseline_within_split.py",
                            "runner.py", "data.py", "models.py", "evaluation.py", "config.py"]
            source_signatures = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in source_names}
            for filename, payload in (("data_fingerprints.json", signatures), ("source_fingerprints.json", source_signatures)):
                path = self.output / filename
                if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
                    raise RuntimeError(f"{filename}: source/data changed; use a new results_dir.")
                path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            self.data = prepare_random_pool(self.data, config)
            audit = self.data.audit.drop(columns=["splitwise_missing_cells_after_imputation"], errors="ignore").copy()
            if "source" in audit:
                audit["random_pool_missing_cells_after_imputation"] = [
                    int(self.data.feature_frame[self.data.station_columns[source]].isna().sum().sum())
                    if source in self.data.station_columns else
                    int(self.data.feature_frame[self.data.weather_columns].isna().sum().sum())
                    for source in audit.source
                ]
            audit.to_csv(self.output / "data_audit.csv", index=False, encoding="utf-8-sig")
        except BaseException:
            self._release_run_lock()
            raise

    def run(self):
        try:
            self._experiment_02()
            self._postprocess()
            return self.output
        except BaseException:
            # A Ctrl+C retains completed tasks and records partial progress.
            self._write_run_status()
            raise
        finally:
            self._release_run_lock()

    @property
    def neural_model(self):
        return self.config.get("station_ablation_model", "lstm")

    def _methods(self):
        methods = [Method("persistence", "persistence"), Method("seasonal_naive", "seasonal_naive")]
        methods += [Method(arm, self.neural_model) for arm in ARMS]
        if self.neural_model != "ridge":
            methods.append(Method("all_stations", "ridge"))
        return methods

    def _experiment_02(self):
        for target in self.config["targets"]:
            sequence, split, arms = common_station_pool(self.data, self.config, target)
            support_id = f"{self.protocol_hash}|C|{target}|18|6"
            common = dict(protocol_hash=self.protocol_hash, experiment=self.experiment,
                          subexperiment="station_information_value", fold=0, split_scheme=SPLIT_PROTOCOL,
                          target_station="C", target=target, target_column=sequence.target_column,
                          lookback_steps=18, lookback_days=3., forecast_horizon_steps=6)
            audit, manifest = random_support_tables(self.data, sequence, split, {
                "support_id": support_id, "protocol_hash": self.protocol_hash,
                "target_station": "C", "target": target, "baseline_protocol": SPLIT_PROTOCOL,
                "station_ablation_protocol": PROTOCOL,
            })
            _upsert_csv(audit, self.output / "sample_support_audit.csv", "support_id")
            _upsert_csv(manifest, self.output / "sample_manifest.csv", "support_id")
            _upsert_csv(distribution_shift_table(sequence, split, {**common, "diagnostic_id": support_id}),
                        self.distribution_path, "diagnostic_id")
            group_rows = []
            for method in self._methods():
                seq = arms.get(method.feature_set, sequence)
                columns = [sequence.target_column] if method.model in {"persistence", "seasonal_naive"} else seq.feature_columns
                group_rows.append(dict(
                    support_id=support_id, target=target, feature_set=method.feature_set, model=method.model,
                    n_features=len(columns), feature_columns="|".join(columns),
                    common_samples=True, identical_split_indices=True,
                    **{f"n_{name}": len(split[name]) for name in SPLITS},
                    **{f"{name}_ordered_indices_sha256": hashlib.sha256(np.asarray(split[name], dtype="<i8").tobytes()).hexdigest()
                       for name in SPLITS},
                ))
            _upsert_csv(pd.DataFrame(group_rows), self.output / "input_groups.csv", "support_id")
            scale = training_mase_scale(sequence, split["train"])
            print(f"[02_station_ablation/random] {target}: common_windows={len(sequence.x)}, " +
                  ", ".join(f"{name}={len(split[name])}" for name in SPLITS), flush=True)
            for method in self._methods():
                selected = arms.get(method.feature_set, sequence)
                for seed in self.config["training"]["seeds"]:
                    if self.max_tasks is not None and self.tasks_run >= self.max_tasks:
                        break
                    task_id = "|".join(map(str, [self.protocol_hash, self.experiment, PROTOCOL,
                                               "C", target, method.feature_set, method.model, seed, 18, 6]))
                    if task_id in self.completed:
                        print(f"  skip: {target}/{method.feature_set}/{method.model}/seed={seed}", flush=True)
                        continue
                    metadata = dict(common, task_id=task_id, feature_set=method.feature_set, model=method.model,
                                    seed=seed, n_features=1 if method.model in {"persistence", "seasonal_naive"} else len(selected.feature_columns),
                                    n_train=len(split["train"]), n_validation=len(split["validation"]), n_test=len(split["test"]))
                    try:
                        prediction, history, best_epoch, parameters, _ = self._fit_method(method, selected.x, selected, split, seed)
                        if prediction.shape != sequence.y[split["test"]].shape or not np.isfinite(prediction).all():
                            raise ValueError("Predictions must be finite and match the common test target shape.")
                        metadata.update(best_epoch=best_epoch, parameter_count=parameters)
                        _append_csv(pd.DataFrame(calculate_metric_rows(sequence.y[split["test"]], prediction, metadata, scale)), self.metrics_path)
                        _append_csv(prediction_frame(sequence.target_start_times[split["test"]], sequence.y[split["test"]], prediction, metadata), self.predictions_path)
                        if history:
                            frame = pd.DataFrame(history)
                            for key, value in metadata.items():
                                frame[key] = value
                            _append_csv(frame, self.history_path)
                        self._mark_completed(task_id)
                        self.tasks_run += 1
                        print(f"  done: {target}/{method.feature_set}/{method.model}/seed={seed}, best_epoch={best_epoch}", flush=True)
                    except Exception as exc:
                        with self.error_path.open("a", encoding="utf-8") as handle:
                            handle.write(task_id + "\n" + traceback.format_exc() + "\n")
                        self.failed[task_id] = f"{type(exc).__name__}: {exc}"
                        self._write_failed()
                        if not self.continue_on_error:
                            raise

    def _postprocess(self):
        self._repair_partial_outputs()
        if self.metrics_path.exists():
            metrics = _read_csv_resilient(self.metrics_path)
            mean_sd_table(metrics).to_csv(self.output / "metrics_mean_sd.csv", index=False, encoding="utf-8-sig")
            skill_scores_table(metrics).to_csv(self.output / "skill_scores.csv", index=False, encoding="utf-8-sig")
            paired_seed_gains(metrics, self.neural_model).to_csv(self.output / "paired_seed_horizon.csv", index=False, encoding="utf-8-sig")
            predictions = _read_csv_resilient(self.predictions_path, parse_dates=["valid_time", "target_start_time"])
            paired = paired_calendar_gains(predictions, self.neural_model, self.config["training"]["seeds"], self.config["bootstrap"])
            paired.to_csv(self.output / "paired_block_bootstrap.csv", index=False, encoding="utf-8-sig")
            if not paired.empty:
                paired[(paired.role == "primary") & (paired.horizon_hours == 0)].to_csv(
                    self.output / "upstream_gain_summary.csv", index=False, encoding="utf-8-sig")
            figure_dir = self.output / "figures"
            figure_dir.mkdir(exist_ok=True)
            plot_data = metrics.copy()
            plot_data["method_label"] = plot_data.feature_set + "/" + plot_data.model
            _line_figure(plot_data, "method_label", figure_dir / "station_ablation_R2.png",
                         "Station ablation: random-window holdout (80/10/10; 18 steps)")
        self._write_run_status()

    def _write_run_status(self):
        super()._write_run_status()
        path = self.output / "run_status.json"
        status = json.loads(path.read_text(encoding="utf-8"))
        expected = len(self.config["targets"]) * len(self._methods()) * len(self.config["training"]["seeds"])
        status.update(station_ablation_protocol=PROTOCOL, split_scheme=SPLIT_PROTOCOL,
                      expected_tasks=expected, expected_lstm_tasks=len(self.config["targets"]) * len(ARMS) * len(self.config["training"]["seeds"]) if self.neural_model == "lstm" else 0,
                      station_ablation_model=self.neural_model, split_seed=self.config["split_seed"],
                      common_samples_across_input_groups=True, lookback_steps=18,
                      primary_contrast="local_multivariate -> all_stations",
                      run_status="complete" if len(self.completed) == expected and not self.failed else (
                          "audit_only" if self.max_tasks == 0 and not self.completed else "partial"),
                      evaluation_scope="fixed_random_window_holdout_not_temporally_independent")
        path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
