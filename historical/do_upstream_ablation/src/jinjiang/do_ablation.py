"""Fixed DO input ablation. No feature/window search; both arms share one cohort."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_path
from .data import feature_set_columns, make_sequences, official_split, prepare_dataset
from .evaluation import calculate_metric_rows, mean_sd_table, prediction_frame
from .lstm_upgrade import _candidate_x
from .models import fit_residual_lstm_predict

EXPERIMENT = "07_do_upstream_ablation"
ARMS = ("local_target", "cross_station_target")
LOCKED_NAIVES = ["persistence", "seasonal_naive", "seasonal_naive",
                 "seasonal_naive", "seasonal_naive", "persistence"]


def _json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temp.replace(path)


def _csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config: dict) -> None:
    if (config.get("target_station") != "C" or config.get("targets") != ["DO"]
            or config.get("frequency") != "4h"
            or config.get("forecast_horizon_steps") != 6):
        raise ValueError("This fixed experiment requires C/DO, 4h sampling, and six forecast steps")
    ablation = config["do_ablation"]
    if ablation.get("lookback_steps") != 6 or ablation.get("arms") != list(ARMS):
        raise ValueError("This experiment fixes lb6 and local_target/cross_station_target")
    if ablation.get("locked_baseline_names") != LOCKED_NAIVES:
        raise ValueError("Keep the predeclared DO baseline names identical in both arms")
    if config["missing_data"].get("interpolation_policy") != "causal_ffill":
        raise ValueError("The ablation requires causal_ffill")
    if config["training"].get("hidden_size") != 32 or config["training"].get("num_layers") != 1:
        raise ValueError("The ablation fixes one layer and hidden_size=32")
    seeds = config["training"]["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be a nonempty unique list")


def build_common_cohort(config: dict):
    """Filter union inputs once. Never build/filter arms independently."""
    validate_config(config)
    data = prepare_dataset(resolve_path(config, config["data_dir"]), config)
    columns = {arm: feature_set_columns(data, arm, "C", "DO") for arm in ARMS}
    if set(columns["cross_station_target"]) != {"A_DO", "B_DO", "C_DO"}:
        raise ValueError("All three station DO columns must be present")
    union = list(dict.fromkeys(column for arm in ARMS for column in columns[arm]))
    sequence = make_sequences(data, union, data.target_columns["C"]["DO"], 6, 6)
    split = official_split(sequence)
    inputs = {arm: _candidate_x(sequence, union, columns[arm], 6, True, True) for arm in ARMS}
    # These are the SAME row indices, labels, naive arrays and targets for both arms.
    for arm in ARMS:
        if inputs[arm].shape[:2] != (len(sequence.y), 6) or not np.isfinite(inputs[arm]).all():
            raise AssertionError(f"Invalid common-cohort input for {arm}")
    if not sequence.issue_times.is_unique:
        raise AssertionError("Duplicate forecast issue times")
    for earlier, later in (("train", "validation"), ("validation", "test")):
        if sequence.target_end_times[split[earlier]].max() >= sequence.target_start_times[split[later]].min():
            raise AssertionError("Forecast target blocks cross split boundaries")
    return data, sequence, split, inputs, columns


def _metadata(arm: str, model: str, seed: int, protocol: str) -> dict:
    return dict(experiment=EXPERIMENT, subexperiment="fixed_common_cohort", fold=0,
                split_scheme="chronological", target_station="C", target="DO",
                feature_set=arm, model=model, seed=seed, lookback_steps=6,
                lookback_days=1, forecast_horizon_steps=6, protocol_hash=protocol,
                task_id=f"DO/{arm}/{model}/seed={seed}/fold=0")


def _signature(config: dict) -> dict:
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    sources = Path(__file__).parent
    code_hashes = {name: _hash(sources / name) for name in
                   ("do_ablation.py", "data.py", "models.py", "evaluation.py", "lstm_upgrade.py")}
    data_dir = resolve_path(config, config["data_dir"])
    data_hashes = {p.name: _hash(p) for p in sorted(data_dir.glob("*.xlsx"))}
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return dict(config=clean, source_sha256=code_hashes, data_sha256=data_hashes,
                package_versions=versions)


def _paired_tables(metrics: pd.DataFrame, predictions: pd.DataFrame, config: dict):
    """Positive differences mean adding upstream inputs helped; no test-based selection."""
    pair_keys = ["seed", "horizon_step", "horizon_hours"]
    local = metrics[metrics.feature_set == ARMS[0]]
    cross = metrics[metrics.feature_set == ARMS[1]]
    pairs = local.merge(cross, on=pair_keys, suffixes=("_local", "_cross"), validate="one_to_one")
    baseline = metrics[metrics.model == "locked_naive"][pair_keys + ["MAE"]]
    pairs = pairs.merge(baseline.rename(columns={"MAE": "MAE_locked_naive"}), on=pair_keys,
                        validate="one_to_one")
    result = pairs[pair_keys].copy()
    for metric in ("MAE", "RMSE"):
        result[f"{metric}_reduction"] = pairs[f"{metric}_local"] - pairs[f"{metric}_cross"]
        result[f"{metric}_reduction_percent"] = 100 * result[f"{metric}_reduction"] / pairs[f"{metric}_local"]
    result["R2_gain"] = pairs.R2_cross - pairs.R2_local
    result["MAE_skill_gain_vs_locked_naive"] = result.MAE_reduction / pairs.MAE_locked_naive
    seed_summary = result.groupby("seed", as_index=False).agg(
        mean_MAE_skill_gain=("MAE_skill_gain_vs_locked_naive", "mean"),
        horizons_with_lower_MAE=("MAE_reduction", lambda s: int((s > 0).sum())))

    # Resample actual 4-hour grid blocks, retaining missing-time gaps as NaNs.
    # Average seed losses before resampling time; seeds are NOT independent time observations.
    p = predictions[predictions.feature_set.isin(ARMS)]
    keys = ["seed", "target_start_time", "horizon_step", "horizon_hours"]
    joined = p[p.feature_set == ARMS[0]].merge(p[p.feature_set == ARMS[1]], on=keys,
                 suffixes=("_local", "_cross"), validate="one_to_one")
    if not np.allclose(joined.y_true_local, joined.y_true_cross):
        raise AssertionError("Paired predictions have different observed targets")
    rows = []
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    for hour, group in joined.groupby("horizon_hours"):
        values = pd.DataFrame({"time": pd.to_datetime(group.target_start_time),
            "gain": group.absolute_error_local - group.absolute_error_cross}).groupby("time").gain.mean()
        estimate, lower, upper = _time_block_ci(values, config["bootstrap"], rng)
        rows.append(dict(horizon_hours=hour, MAE_reduction=estimate, ci95_lower=lower,
                         ci95_upper=upper, n_common_times=len(values)))
    # Primary endpoint: mean across the six horizons of skill gain against the LOCKED baseline.
    denom = baseline.groupby("horizon_step").MAE.mean()
    joined["skill_gain"] = (joined.absolute_error_local - joined.absolute_error_cross) / joined.horizon_step.map(denom)
    by_time = joined.groupby("target_start_time").skill_gain.mean()
    by_time.index = pd.to_datetime(by_time.index)
    estimate, lower, upper = _time_block_ci(by_time, config["bootstrap"], rng)
    summary = dict(primary_endpoint="mean six-horizon MAE skill gain: cross minus local",
                   estimate=estimate, ci95_lower=lower, ci95_upper=upper,
                   improved_seed_count=int((seed_summary.mean_MAE_skill_gain > 0).sum()),
                   seed_count=len(seed_summary), positive_means="upstream input helps",
                   note="Single historical split; bootstrap over time after averaging seed losses. No causal attribution or independent temporal replication.")
    return result, seed_summary, pd.DataFrame(rows), summary


def _time_block_ci(values: pd.Series, config: dict, rng):
    values = values.sort_index()
    estimate = float(values.mean())
    block = int(config["block_length_steps"])
    if len(values) < 2 * block:
        return estimate, None, None
    grid = values.reindex(pd.date_range(values.index.min(), values.index.max(), freq="4h")).to_numpy()
    if len(grid) < block:
        return estimate, None, None
    draws = []
    for _ in range(int(config["replicates"])):
        starts = rng.integers(0, len(grid) - block + 1, size=int(np.ceil(len(grid) / block)))
        sample = np.concatenate([grid[s:s + block] for s in starts])[:len(grid)]
        if np.isfinite(sample).any():
            draws.append(float(np.nanmean(sample)))
    if not draws:
        return estimate, None, None
    low, high = np.quantile(draws, [.025, .975])
    return estimate, float(low), float(high)


def run_do_ablation(config: dict, *, audit_only: bool = False,
                    seeds: list[int] | None = None) -> Path:
    validate_config(config)
    configured_seeds = [int(s) for s in config["training"]["seeds"]]
    requested = configured_seeds if seeds is None else [int(s) for s in seeds]
    if not requested or len(set(requested)) != len(requested) or not set(requested).issubset(configured_seeds):
        raise ValueError("--seeds must be a nonempty unique subset of the configured seeds")
    # Dedicated output only. No overwrite switch and no recursive deletion of old results.
    root = Path(config["_project_root"]).resolve()
    parent = resolve_path(config, config["results_dir"]).resolve()
    if parent.parent != root / "results" or not parent.name.startswith("do_upstream_ablation"):
        raise ValueError("results_dir must be results/do_upstream_ablation or a similarly prefixed sibling")
    output = parent / EXPERIMENT
    signature = _signature(config)
    protocol = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if output.exists() and any(output.iterdir()) and not (output / "protocol.json").exists():
        raise RuntimeError("Output is not a recognized ablation directory; use a new results_dir")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "protocol.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        # Audit-only is allowed without torch. Installing torch between audit and first training is safe.
        for key in ("config", "source_sha256", "data_sha256"):
            if previous[key] != signature[key]:
                raise RuntimeError(f"Existing output has changed {key}; use a new results_dir")
        if any((output / "tasks").glob("*/done.json")) and previous["package_versions"] != signature["package_versions"]:
            raise RuntimeError("Dependencies changed after training began; use a new results_dir")
    _json(manifest, {**signature, "protocol_hash": protocol})
    data, sequence, split, inputs, columns = build_common_cohort(config)
    cohort = pd.DataFrame(dict(sample_id=np.arange(len(sequence.y)),
        split=sequence.split_labels, issue_time=sequence.issue_times,
        target_start_time=sequence.target_start_times, target_end_time=sequence.target_end_times))
    for h in range(6):
        cohort[f"observed_target_h{h + 1}"] = sequence.y[:, h]
    support_hash = hashlib.sha256(cohort.to_csv(index=False).encode()).hexdigest()
    audit = dict(shared_cohort_sha256=support_hash, n_common_samples=len(cohort),
        per_arm_cohort_sha256={arm: support_hash for arm in ARMS},
        split_boundaries=data.split_boundaries, raw_columns=columns,
        augmented_feature_count={arm: inputs[arm].shape[2] for arm in ARMS},
        locked_baseline_names=LOCKED_NAIVES, first_differences=True, calendar_features=True,
        splits={name: dict(n=len(idx), first_issue=str(sequence.issue_times[idx].min()),
                  last_issue=str(sequence.issue_times[idx].max()),
                  first_target=str(sequence.target_start_times[idx].min()),
                  last_target=str(sequence.target_end_times[idx].max())) for name, idx in split.items()},
        assertion="Both arms slice the same union sequence. No per-arm row filtering.",
        scope="Downstream-history-conditioned upstream information ablation; not pure-upstream prediction.")
    old_audit = output / "sample_audit.json"
    if old_audit.exists() and json.loads(old_audit.read_text(encoding="utf-8"))["shared_cohort_sha256"] != support_hash:
        raise RuntimeError("Shared cohort changed; use a new results_dir")
    _csv(output / "common_samples.csv", cohort)
    _csv(output / "data_audit.csv", data.audit)
    _json(old_audit, audit)
    print(f"[{EXPERIMENT}] common samples: " + ", ".join(f"{k}={len(v)}" for k, v in split.items()), flush=True)
    if audit_only:
        print("Audit only: no model training or test prediction performed.", flush=True)
        return output
    locked = np.column_stack([(sequence.persistence if name == "persistence" else sequence.seasonal_naive)[:, h]
                             for h, name in enumerate(LOCKED_NAIVES)])
    task_root = output / "tasks"
    task_root.mkdir(exist_ok=True)
    test = split["test"]
    for seed in requested:
        for arm in ARMS:
            task = task_root / f"{arm}__seed{seed}"
            if (task / "done.json").exists():
                done = json.loads((task / "done.json").read_text(encoding="utf-8"))
                if done["protocol_hash"] != protocol or done["shared_cohort_sha256"] != support_hash:
                    raise RuntimeError(f"Task provenance mismatch: {task}")
                print(f"Skip completed: {arm}, seed={seed}", flush=True)
                continue
            print(f"Train: {arm}, seed={seed}, input_features={inputs[arm].shape[2]}", flush=True)
            result = fit_residual_lstm_predict(inputs[arm], sequence.y,
                sequence.persistence, sequence.seasonal_naive, split, seed, config["training"],
                refit_train_validation=True, locked_baseline_names=LOCKED_NAIVES)
            metadata = _metadata(arm, "residual_lstm", seed, protocol)
            staging = task.with_name(task.name + ".incomplete")
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir()
            _csv(staging / "metrics.csv", pd.DataFrame(calculate_metric_rows(sequence.y[test], result.prediction, metadata)))
            _csv(staging / "predictions.csv", prediction_frame(sequence.target_start_times[test], sequence.y[test], result.prediction, metadata))
            _csv(staging / "training_history.csv", pd.DataFrame(result.history))
            _json(staging / "diagnostics.json", dict(best_epoch=result.best_epoch,
                   parameter_count=result.parameter_count, **(result.diagnostics or {})))
            _json(staging / "done.json", dict(protocol_hash=protocol,
                  shared_cohort_sha256=support_hash, seed=seed, arm=arm))
            if task.exists():
                raise RuntimeError(f"Unexpected unfinished task directory: {task}")
            staging.rename(task)
    # Publish paired output only for seeds with BOTH completed arms.
    complete = [seed for seed in configured_seeds if all(
        (task_root / f"{arm}__seed{seed}" / "done.json").exists() for arm in ARMS)]
    metric_frames, prediction_frames = [], []
    for seed in complete:
        for arm in ARMS:
            task = task_root / f"{arm}__seed{seed}"
            metric_frames.append(pd.read_csv(task / "metrics.csv"))
            prediction_frames.append(pd.read_csv(task / "predictions.csv"))
        for name, prediction in (("persistence", sequence.persistence),
                                 ("seasonal_naive", sequence.seasonal_naive), ("locked_naive", locked)):
            meta = _metadata(name, name, seed, protocol)
            metric_frames.append(pd.DataFrame(calculate_metric_rows(sequence.y[test], prediction[test], meta)))
            prediction_frames.append(prediction_frame(sequence.target_start_times[test], sequence.y[test], prediction[test], meta))
    if complete:
        metrics = pd.concat(metric_frames, ignore_index=True)
        predictions = pd.concat(prediction_frames, ignore_index=True)
        _csv(output / "metrics.csv", metrics)
        _csv(output / "metrics_mean_sd.csv", mean_sd_table(metrics))
        _csv(output / "predictions.csv", predictions)
        pairs, seed_summary, intervals, primary = _paired_tables(metrics, predictions, config)
        _csv(output / "paired_seed_horizon.csv", pairs)
        _csv(output / "paired_seed_summary.csv", seed_summary)
        _csv(output / "paired_time_block_ci.csv", intervals)
        all_complete = len(complete) == len(configured_seeds)
        primary["decision"] = ("pending_all_seeds" if not all_complete else
            "supports_upstream_gain_on_this_split" if primary["improved_seed_count"] >= 4
            and primary["ci95_lower"] is not None and primary["ci95_lower"] > 0
            else "stable_upstream_gain_not_established")
        _json(output / "primary_endpoint.json", primary)
    _json(output / "run_status.json", dict(status="complete" if len(complete) == len(configured_seeds) else "partial",
        completed_paired_seeds=complete, configured_seeds=configured_seeds,
        planned_arm_seed_tasks=2 * len(configured_seeds), shared_cohort_sha256=support_hash,
        finished_utc=datetime.now(timezone.utc).isoformat(), test_set_used=bool(complete)))
    return output
