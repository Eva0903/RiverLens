"""Random-window holdout benchmark. Windows stay ordered internally; splits mix dates."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .baseline_within_split import training_mase_scale
from .config import resolve_path
from .data import apply_splitwise_imputation, feature_set_columns, make_sequences
from .evaluation import (
    calculate_metric_rows, distribution_shift_table, mean_sd_table,
    prediction_frame, skill_scores_table,
)
from .plots import _line_figure
from .runner import ExperimentRunner, Method, _append_csv, _read_csv_resilient, _upsert_csv


PROTOCOL = "random_window_holdout_80_10_10_v1"
SPLITS = ("train", "validation", "test")


def legacy_random_split(n, train_fraction=.8, validation_fraction=.1, seed=42):
    """The old TensorFlow script's TWO sklearn splits, not a new RNG permutation."""
    remainder = round(1. - float(train_fraction), 12)
    train, temporary = train_test_split(np.arange(n), test_size=remainder, random_state=seed)
    validation, test = train_test_split(
        temporary, test_size=1. - float(validation_fraction) / remainder, random_state=seed,
    )
    return dict(train=np.asarray(train), validation=np.asarray(validation), test=np.asarray(test))


def prepare_random_pool(data, config):
    """Keep the whole raw timeline; no chronological window purging is applied.

    Existing short-gap rules are reused over the continuous frame. Scaling is NOT
    performed here: the model functions fit their scalers using training windows only.
    """
    end = data.raw_frame.index[-1]
    boundaries = dict(train_end=end, validation_end=end, test_end=end)
    missing = config["missing_data"]
    features = apply_splitwise_imputation(
        data.raw_frame, data.station_columns, data.weather_columns, boundaries,
        missing["max_water_gap_steps"], missing["max_weather_gap_hours"],
        missing["interpolation_policy"],
    )
    return replace(data, feature_frame=features, split_boundaries=boundaries)


def random_support_tables(data, sequence, split, metadata):
    """Document unique window assignment AND cross-split raw-time overlap honestly."""
    index = pd.DatetimeIndex(data.raw_frame.index)
    starts = index.get_indexer(sequence.target_start_times)
    lookback, horizon = sequence.x.shape[1], sequence.y.shape[1]
    if any(len(split[name]) < 2 for name in SPLITS):
        raise ValueError("Every partition needs at least two windows.")
    combined = np.concatenate([split[name] for name in SPLITS])
    if len(combined) != len(starts) or not np.array_equal(np.sort(combined), np.arange(len(starts))):
        raise AssertionError("Every window must belong to exactly one partition.")
    labels = np.empty(len(starts), dtype=object)
    masks = {}
    for name in SPLITS:
        labels[split[name]] = name
        mask = np.zeros(len(index), dtype=bool)
        for offset in range(horizon):
            mask[starts[split[name]] + offset] = True
        masks[name] = mask
    manifest = pd.DataFrame({
        **metadata, "sample_index": np.arange(len(starts)), "data_split": labels,
        "input_start_time": index[starts - lookback], "issue_time": sequence.issue_times,
        "target_start_time": sequence.target_start_times, "target_end_time": sequence.target_end_times,
    })
    audit = []
    for name in SPLITS:
        ids = split[name]
        target_times = masks[name]
        training_overlap = (target_times & masks["train"]).sum() if name != "train" else 0
        signature = "\n".join(t.isoformat() for t in sequence.target_start_times[np.sort(ids)])
        audit.append({
            **metadata, "data_split": name, "valid_windows": len(ids),
            "window_ids_disjoint": True, "temporal_independence": False,
            "target_start_min": sequence.target_start_times[ids].min(),
            "target_start_max": sequence.target_start_times[ids].max(),
            "unique_target_timestamps": int(target_times.sum()),
            "target_timestamps_also_in_training_labels": int(training_overlap),
            "fraction_target_times_seen_in_training_labels": (
                training_overlap / target_times.sum() if name != "train" else np.nan
            ),
            "feature_count": len(sequence.feature_columns),
            "feature_columns": "|".join(sequence.feature_columns),
            "target_start_times_sha256": hashlib.sha256(signature.encode()).hexdigest(),
        })
    return pd.DataFrame(audit), manifest


def conditional_time_block_comparisons(predictions, block_length=6, replicates=2000, seed=2026):
    """Calendar-grid block CIs conditional on this fitted model and fixed random split.

    Calendar blocks include intervals without test samples, rather than concatenating distant dates.
    These intervals do not assess the bias of the random-window evaluation protocol.
    """
    rows = []
    rng = np.random.default_rng(seed)
    for (target, hours), group in predictions.groupby(["target", "horizon_hours"]):
        origin = pd.Timestamp(group.valid_time.min()).floor(f"{4 * block_length}h")
        nblocks = int((pd.Timestamp(group.valid_time.max()) - origin) / pd.Timedelta(hours=4 * block_length)) + 1
        # Share paired calendar-block draws across every comparison at this lead time.
        draws_index = rng.integers(0, nblocks, size=(replicates, nblocks))
        methods = list(group[["feature_set", "model"]].drop_duplicates().itertuples(index=False, name=None))
        references = [("persistence", "persistence"), ("seasonal_naive", "seasonal_naive"), ("all_stations", "lstm")]
        for ref_feature, ref_model in references:
            ref = group[(group.feature_set == ref_feature) & (group.model == ref_model)]
            if ref.empty:
                continue
            for feature, model in methods:
                if (feature, model) == (ref_feature, ref_model):
                    continue
                comp = group[(group.feature_set == feature) & (group.model == model)]
                paired = ref.merge(comp, on=["seed", "valid_time"], suffixes=("_ref", "_comp"), validate="one_to_one")
                for loss in ("absolute_error", "squared_error"):
                    delta = pd.Series(
                        paired[loss + "_comp"].to_numpy() - paired[loss + "_ref"].to_numpy(),
                        index=pd.DatetimeIndex(paired.valid_time),
                    ).groupby(level=0).mean().sort_index()
                    block_ids = np.asarray((delta.index - origin) / pd.Timedelta(hours=4 * block_length), dtype=int)
                    totals = np.bincount(block_ids, weights=delta.to_numpy(), minlength=nblocks)
                    counts = np.bincount(block_ids, minlength=nblocks)
                    sample_totals = totals[draws_index].sum(axis=1)
                    sample_counts = counts[draws_index].sum(axis=1)
                    draws = sample_totals[sample_counts > 0] / sample_counts[sample_counts > 0]
                    low, high = np.quantile(draws, [.025, .975]) if len(delta) >= 10 and len(draws) else (np.nan, np.nan)
                    rows.append(dict(
                        target=target, horizon_hours=hours, reference_feature_set=ref_feature,
                        reference_model=ref_model, competitor_feature_set=feature, competitor_model=model,
                        loss=loss, n_common_times=len(delta), block_length_steps=block_length,
                        bootstrap_replicates=replicates, competitor_minus_reference_loss=float(delta.mean()),
                        ci95_lower=low, ci95_upper=high,
                        scope="conditional_on_fixed_random_window_split_and_fitted_models",
                        resampling="nonoverlapping_calendar_blocks_preserving_empty_intervals",
                        n_calendar_blocks=nblocks, n_nonempty_calendar_blocks=int((counts > 0).sum()),
                    ))
    return pd.DataFrame(rows)


class RandomBaselineRunner(ExperimentRunner):
    """Reuse V4 model fitting, output repair and locking; do not replace the old runner."""

    def __init__(self, config, *args, **kwargs):
        if config.get("baseline_protocol") != PROTOCOL:
            raise ValueError("Use configs/baseline_random_gpu.json.")
        super().__init__(config, "01_baseline", *args, **kwargs)
        try:
            files = sorted(resolve_path(config, config["data_dir"]).glob("*.xlsx"))
            signatures = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
            signature_path = self.output / "data_fingerprints.json"
            if signature_path.exists() and json.loads(signature_path.read_text(encoding="utf-8")) != signatures:
                raise RuntimeError("Raw data changed since this run started; use a new results_dir.")
            signature_path.write_text(json.dumps(signatures, indent=2) + "\n", encoding="utf-8")
            self.data = prepare_random_pool(self.data, config)
            if "source" in self.data.audit:
                audit = self.data.audit.drop(columns=["splitwise_missing_cells_after_imputation"], errors="ignore").copy()
                audit["random_pool_missing_cells_after_imputation"] = [
                    int(self.data.feature_frame[self.data.station_columns[source]].isna().sum().sum())
                    if source in self.data.station_columns
                    else int(self.data.feature_frame[self.data.weather_columns].isna().sum().sum())
                    for source in audit["source"]
                ]
                audit.to_csv(self.output / "data_audit.csv", index=False, encoding="utf-8-sig")
        except Exception:
            self._release_run_lock()
            raise

    def _methods(self):
        models = [self.model_override] if self.model_override else list(dict.fromkeys(self.config["baseline_models"]))
        return [Method("persistence", "persistence"), Method("seasonal_naive", "seasonal_naive")] + [
            Method("all_stations", model) for model in models if model not in {"persistence", "seasonal_naive"}
        ]

    def _experiment_01(self):
        config = self.config
        methods = self._methods()
        for target in config["targets"]:
            station = config["target_station"]
            target_column = self.data.target_columns[station][target]
            columns = feature_set_columns(self.data, "all_stations", station, target)
            lookback = int(config["baseline_lookback_steps"])
            horizon = int(config["forecast_horizon_steps"])
            sequence = make_sequences(self.data, columns, target_column, lookback, horizon)
            split = legacy_random_split(len(sequence.x), config["split"]["train_fraction"],
                                        config["split"]["validation_fraction"], config["split_seed"])
            support_id = f"{self.protocol_hash}|{station}|{target}|{lookback}|{horizon}"
            common = dict(protocol_hash=self.protocol_hash, experiment="01_baseline",
                          subexperiment="unified_model_benchmark", fold=0, split_scheme=PROTOCOL,
                          target_station=station, target=target, target_column=target_column,
                          lookback_steps=lookback, lookback_days=lookback / 6,
                          forecast_horizon_steps=horizon)
            audit, manifest = random_support_tables(self.data, sequence, split, {
                "support_id": support_id, "protocol_hash": self.protocol_hash,
                "target_station": station, "target": target, "baseline_protocol": PROTOCOL,
            })
            _upsert_csv(audit, self.output / "sample_support_audit.csv", "support_id")
            _upsert_csv(manifest, self.output / "sample_manifest.csv", "support_id")
            _upsert_csv(distribution_shift_table(sequence, split, {**common, "diagnostic_id": support_id}),
                        self.distribution_path, "diagnostic_id")
            scale = training_mase_scale(sequence, split["train"])
            print(f"[01_baseline/random] {target}: common_windows={len(sequence.x)}, " +
                  ", ".join(f"{name}={len(split[name])}" for name in SPLITS), flush=True)
            for method in methods:
                for seed in config["training"]["seeds"]:
                    if self.max_tasks is not None and self.tasks_run >= self.max_tasks:
                        break
                    task_id = "|".join(map(str, [self.protocol_hash, "01_baseline", "random_window_benchmark",
                                               station, target, method.feature_set, method.model, seed, lookback, horizon]))
                    if task_id in self.completed:
                        print(f"  skip: {method.model}/seed={seed}", flush=True)
                        continue
                    metadata = dict(common, task_id=task_id, feature_set=method.feature_set, model=method.model,
                                    seed=seed, n_features=1 if method.model in {"persistence", "seasonal_naive"} else len(columns),
                                    n_train=len(split["train"]), n_validation=len(split["validation"]), n_test=len(split["test"]))
                    try:
                        prediction, history, best_epoch, parameters, _ = self._fit_method(method, sequence.x, sequence, split, seed)
                        metadata.update(best_epoch=best_epoch, parameter_count=parameters)
                        _append_csv(pd.DataFrame(calculate_metric_rows(sequence.y[split["test"]], prediction, metadata, scale)), self.metrics_path)
                        _append_csv(prediction_frame(sequence.target_start_times[split["test"]], sequence.y[split["test"]], prediction, metadata), self.predictions_path)
                        if history:
                            history_frame = pd.DataFrame(history)
                            for key, value in metadata.items():
                                history_frame[key] = value
                            _append_csv(history_frame, self.history_path)
                        self._mark_completed(task_id)
                        self.tasks_run += 1
                        print(f"  done: {method.model}/seed={seed}, best_epoch={best_epoch}", flush=True)
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
            ranking = metrics.groupby(["target", "feature_set", "model", "horizon_hours"], as_index=False).agg(
                R2_mean=("R2", "mean"), R2_sd=("R2", "std"), RMSE_mean=("RMSE", "mean"),
                RMSE_sd=("RMSE", "std"), MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
            )
            ranking["MAE_rank"] = ranking.groupby(["target", "horizon_hours"]).MAE_mean.rank(method="min")
            ranking["R2_rank"] = ranking.groupby(["target", "horizon_hours"]).R2_mean.rank(method="min", ascending=False)
            ranking.to_csv(self.output / "model_ranking.csv", index=False, encoding="utf-8-sig")
            if self.predictions_path.exists():
                predictions = _read_csv_resilient(self.predictions_path, parse_dates=["valid_time", "target_start_time"])
                cfg = self.config["bootstrap"]
                conditional_time_block_comparisons(predictions, int(cfg["block_length_steps"]), int(cfg["replicates"]), int(cfg["seed"])).to_csv(
                    self.output / "paired_block_bootstrap.csv", index=False, encoding="utf-8-sig")
            figure_dir = self.output / "figures"
            figure_dir.mkdir(exist_ok=True)
            _line_figure(metrics, "model", figure_dir / "model_benchmark_R2.png",
                         "Random-window holdout (80/10/10; split seed=42)")
        self._write_run_status()

    def _write_run_status(self):
        super()._write_run_status()
        path = self.output / "run_status.json"
        status = json.loads(path.read_text(encoding="utf-8"))
        expected = len(self.config["targets"]) * len(self._methods()) * len(self.config["training"]["seeds"])
        status.update(baseline_protocol=PROTOCOL, expected_tasks=expected,
                      selected_models=[method.model for method in self._methods()], split_seed=self.config["split_seed"],
                      run_status="complete" if len(self.completed) == expected and not self.failed else (
                          "audit_only" if self.max_tasks == 0 and not self.completed else "partial"),
                      evaluation_scope="fixed_random_window_holdout_not_temporally_independent")
        path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
