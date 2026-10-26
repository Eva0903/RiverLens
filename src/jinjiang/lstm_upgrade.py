from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_path
from .data import (
    SequenceData,
    feature_set_columns,
    make_sequences,
    official_split,
    prepare_dataset,
)
from .evaluation import (
    calculate_metric_rows,
    distribution_shift_table,
    mase_scale_from_training,
    mean_sd_table,
    paired_block_bootstrap_table,
    prediction_frame,
    skill_scores_table,
)
from .models import choose_validation_baseline, fit_residual_lstm_predict


def _clean_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _calendar_features(issue_times: pd.DatetimeIndex, lookback: int) -> np.ndarray:
    offsets = np.arange(lookback - 1, -1, -1)
    rows = []
    for issue_time in issue_times:
        timestamps = issue_time - pd.to_timedelta(offsets * 4, unit="h")
        hour_fraction = (timestamps.hour + timestamps.minute / 60.0) / 24.0
        year_fraction = (
            timestamps.dayofyear.to_numpy(dtype=float) - 1.0 + hour_fraction
        ) / 365.25
        rows.append(np.column_stack([
            np.sin(2 * np.pi * hour_fraction),
            np.cos(2 * np.pi * hour_fraction),
            np.sin(2 * np.pi * year_fraction),
            np.cos(2 * np.pi * year_fraction),
        ]))
    return np.asarray(rows, dtype=np.float32)


def _candidate_x(
    common_sequence: SequenceData,
    union_columns: list[str],
    feature_columns: list[str],
    lookback: int,
    use_differences: bool,
    use_calendar: bool,
) -> np.ndarray:
    column_positions = [union_columns.index(column) for column in feature_columns]
    values = common_sequence.x[:, -lookback:, column_positions].astype(np.float32)
    blocks = [values]
    if use_differences:
        differences = np.diff(values, axis=1, prepend=values[:, :1, :])
        blocks.append(differences.astype(np.float32))
    if use_calendar:
        blocks.append(_calendar_features(common_sequence.issue_times, lookback))
    return np.concatenate(blocks, axis=2).astype(np.float32)


def _metadata(
    protocol_hash: str,
    target_station: str,
    target: str,
    feature_set: str,
    model: str,
    seed: int,
    lookback: int,
    horizon: int,
) -> dict[str, object]:
    return {
        "experiment": "06_lstm_upgrade",
        "subexperiment": "validation_selected_residual_lstm",
        "fold": 0,
        "split_scheme": "chronological",
        "target_station": target_station,
        "target": target,
        "feature_set": feature_set,
        "model": model,
        "seed": seed,
        "lookback_steps": lookback,
        "lookback_days": lookback / 6,
        "forecast_horizon_steps": horizon,
        "protocol_hash": protocol_hash,
    }


def _candidate_specs(upgrade: dict[str, Any], target: str) -> list[dict[str, Any]]:
    specs = []
    windows = upgrade["candidate_windows_by_target"][target]
    for feature_set in upgrade["candidate_feature_sets"]:
        for lookback in windows:
            for model_index, model_config in enumerate(upgrade["candidate_model_configs"]):
                specs.append({
                    "feature_set": feature_set,
                    "lookback": int(lookback),
                    "model_index": model_index,
                    "model_config": dict(model_config),
                })
    return specs


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _append_completed(path: Path, identifier: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(identifier + "\n")
        handle.flush()


def _completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _selection_protocol_hash(config: dict[str, Any]) -> str:
    upgrade = config["lstm_upgrade"]
    selection_protocol = {
        "data_dir": config["data_dir"],
        "frequency": config["frequency"],
        "target_station": config["target_station"],
        "targets": config["targets"],
        "forecast_horizon_steps": config["forecast_horizon_steps"],
        "split": config["split"],
        "missing_data": config["missing_data"],
        "candidate_feature_sets": upgrade["candidate_feature_sets"],
        "candidate_windows_by_target": upgrade["candidate_windows_by_target"],
        "candidate_model_configs": upgrade["candidate_model_configs"],
        "append_first_differences": upgrade.get("append_first_differences", True),
        "append_calendar_features": upgrade.get("append_calendar_features", True),
        "selection_seed": upgrade.get("selection_seed", 42),
        "selection_training": {
            **config["training"],
            **upgrade.get("selection_training", {}),
        },
    }
    return hashlib.sha256(
        json.dumps(selection_protocol, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def run_lstm_upgrade(config: dict[str, Any], overwrite: bool = False) -> Path:
    """Run validation-only LSTM selection and, optionally, one locked test evaluation."""
    clean_config = _clean_config(config)
    config_hash = hashlib.sha256(
        json.dumps(clean_config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    protocol_hash = config_hash[:16]
    output = resolve_path(config, config["results_dir"]) / "06_lstm_upgrade"
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    previous_config = output / "run_config.json"
    if previous_config.exists() and not overwrite:
        previous = json.loads(previous_config.read_text(encoding="utf-8"))
        if _clean_config(previous) != clean_config:
            raise RuntimeError(
                "Existing 06_lstm_upgrade results use another configuration. "
                "Use another results_dir or rerun once with --overwrite."
            )
    _write_json(previous_config, clean_config)

    upgrade = config["lstm_upgrade"]
    selection_protocol_hash = _selection_protocol_hash(config)
    target_station = config["target_station"]
    horizon = int(config["forecast_horizon_steps"])
    use_differences = bool(upgrade.get("append_first_differences", True))
    use_calendar = bool(upgrade.get("append_calendar_features", True))
    selection_seed = int(upgrade.get("selection_seed", 42))
    selection_training = {**config["training"], **upgrade.get("selection_training", {})}
    final_training = {**config["training"], **upgrade.get("final_training", {})}
    candidate_metrics_path = output / "candidate_validation_metrics.csv"
    candidate_summary_path = output / "candidate_validation_summary.csv"
    completed_candidates_path = output / "completed_candidates.txt"
    selection_cache_dir = upgrade.get("selection_cache_dir")
    if selection_cache_dir and not candidate_metrics_path.exists():
        cache_dir = resolve_path(config, selection_cache_dir)
        cache_status_path = cache_dir / "run_status.json"
        if cache_status_path.exists():
            cache_status = json.loads(cache_status_path.read_text(encoding="utf-8"))
            if cache_status.get("selection_protocol_hash") != selection_protocol_hash:
                raise RuntimeError(
                    "The validation-selection cache does not match the formal selection protocol."
                )
            required_cache_files = [
                "candidate_validation_metrics.csv",
                "candidate_validation_summary.csv",
                "completed_candidates.txt",
            ]
            missing_cache_files = [name for name in required_cache_files if not (cache_dir / name).exists()]
            if missing_cache_files:
                raise RuntimeError(f"Incomplete validation-selection cache: {missing_cache_files}")
            for name in required_cache_files:
                shutil.copy2(cache_dir / name, output / name)
            print(f"[06_lstm_upgrade] reused validation-only selection: {cache_dir}")
        else:
            print(
                f"[06_lstm_upgrade] selection cache not found at {cache_dir}; "
                "running selection in the formal directory"
            )
    data = prepare_dataset(resolve_path(config, config["data_dir"]), config)
    data.audit.to_csv(output / "data_audit.csv", index=False, encoding="utf-8-sig")
    existing_candidate_metrics = (
        pd.read_csv(candidate_metrics_path) if candidate_metrics_path.exists() else pd.DataFrame()
    )
    existing_candidate_summary = (
        pd.read_csv(candidate_summary_path) if candidate_summary_path.exists() else pd.DataFrame()
    )
    candidate_metric_rows: list[dict[str, object]] = existing_candidate_metrics.to_dict("records")
    candidate_summary_rows: list[dict[str, object]] = existing_candidate_summary.to_dict("records")
    completed_candidates = _completed(completed_candidates_path)
    selected_configs: dict[str, dict[str, Any]] = {}

    for target in config["targets"]:
        specs = _candidate_specs(upgrade, target)
        max_lookback = max(spec["lookback"] for spec in specs)
        feature_sets = {
            name: feature_set_columns(data, name, target_station, target)
            for name in upgrade["candidate_feature_sets"]
        }
        union_columns = list(dict.fromkeys(
            column for columns in feature_sets.values() for column in columns
        ))
        target_column = data.target_columns[target_station][target]
        common_sequence = make_sequences(
            data,
            union_columns,
            target_column,
            max_lookback,
            horizon,
        )
        split = official_split(common_sequence)
        selected_baseline, baseline_names = choose_validation_baseline(
            common_sequence.y,
            common_sequence.persistence,
            common_sequence.seasonal_naive,
            split["validation"],
        )
        validation_baseline = selected_baseline[split["validation"]]
        validation_truth = common_sequence.y[split["validation"]]
        baseline_mae = np.mean(np.abs(validation_truth - validation_baseline), axis=0)
        print(
            f"[06_lstm_upgrade/select] station={target_station}, target={target}, "
            f"common_sequences={len(common_sequence.y)}, candidates={len(specs)}"
        )
        for candidate_number, spec in enumerate(specs, start=1):
            feature_set = spec["feature_set"]
            lookback = spec["lookback"]
            candidate_id = (
                f"{target}__{feature_set}__lb{lookback}__m{spec['model_index']}"
            )
            if candidate_id in completed_candidates:
                print(f"  skip completed candidate: {candidate_id}")
                continue
            x = _candidate_x(
                common_sequence,
                union_columns,
                feature_sets[feature_set],
                lookback,
                use_differences,
                use_calendar,
            )
            training = {
                **selection_training,
                **spec["model_config"],
            }
            result = fit_residual_lstm_predict(
                x,
                common_sequence.y,
                common_sequence.persistence,
                common_sequence.seasonal_naive,
                split,
                selection_seed,
                training,
                refit_train_validation=False,
            )
            metadata = _metadata(
                protocol_hash,
                target_station,
                target,
                feature_set,
                "residual_lstm",
                selection_seed,
                lookback,
                horizon,
            )
            rows = calculate_metric_rows(
                validation_truth,
                result.validation_prediction,
                metadata,
            )
            gates = result.diagnostics["residual_gates"]
            for row in rows:
                horizon_index = int(row["horizon_step"]) - 1
                row.update({
                    "candidate_id": candidate_id,
                    "model_index": spec["model_index"],
                    "hidden_size": int(training["hidden_size"]),
                    "num_layers": int(training["num_layers"]),
                    "dropout": float(training.get("dropout", 0.0)),
                    "bidirectional": bool(training.get("bidirectional", False)),
                    "input_feature_count": int(x.shape[2]),
                    "parameter_count": result.parameter_count,
                    "best_epoch": result.best_epoch,
                    "selected_naive": baseline_names[horizon_index],
                    "residual_gate": gates[horizon_index],
                    "MAE_selected_naive": float(baseline_mae[horizon_index]),
                    "MAE_skill_vs_selected_naive": float(
                        1 - row["MAE"] / max(float(baseline_mae[horizon_index]), 1e-12)
                    ),
                    "evaluation_split": "validation",
                })
            candidate_metric_rows.extend(rows)
            mean_mae = float(np.mean([row["MAE"] for row in rows]))
            mean_skill = float(np.mean([row["MAE_skill_vs_selected_naive"] for row in rows]))
            candidate_summary_rows.append({
                "target": target,
                "candidate_id": candidate_id,
                "feature_set": feature_set,
                "lookback_steps": lookback,
                "model_index": spec["model_index"],
                "hidden_size": int(training["hidden_size"]),
                "num_layers": int(training["num_layers"]),
                "dropout": float(training.get("dropout", 0.0)),
                "bidirectional": bool(training.get("bidirectional", False)),
                "input_feature_count": int(x.shape[2]),
                "parameter_count": result.parameter_count,
                "best_epoch": result.best_epoch,
                "validation_MAE_mean": mean_mae,
                "validation_MAE_skill_vs_selected_naive_mean": mean_skill,
            })
            candidate_metrics_frame = pd.DataFrame(candidate_metric_rows).drop_duplicates(
                ["candidate_id", "horizon_step"], keep="last"
            )
            candidate_summary_frame = pd.DataFrame(candidate_summary_rows).drop_duplicates(
                ["candidate_id"], keep="last"
            )
            _atomic_csv(candidate_metrics_frame, candidate_metrics_path)
            _atomic_csv(candidate_summary_frame, candidate_summary_path)
            _append_completed(completed_candidates_path, candidate_id)
            completed_candidates.add(candidate_id)
            print(
                f"  candidate {candidate_number:02d}/{len(specs)}: {candidate_id}, "
                f"validation_skill={mean_skill:+.3f}"
            )

        target_summary = pd.DataFrame(
            row for row in candidate_summary_rows if row["target"] == target
        ).drop_duplicates(["candidate_id"], keep="last").sort_values(
            [
                "validation_MAE_skill_vs_selected_naive_mean",
                "validation_MAE_mean",
                "parameter_count",
                "lookback_steps",
            ],
            ascending=[False, True, True, True],
        )
        winner = target_summary.iloc[0].to_dict()
        winner_spec = next(spec for spec in specs if spec["model_index"] == winner["model_index"]
                           and spec["feature_set"] == winner["feature_set"]
                           and spec["lookback"] == winner["lookback_steps"])
        selected_configs[target] = {
            **winner,
            "model_config": winner_spec["model_config"],
            "selected_naive_by_horizon": baseline_names,
            "selection_rule": "maximum mean validation MAE skill; then lower MAE, parameters, lookback",
        }
    candidate_metrics_frame = pd.DataFrame(candidate_metric_rows).drop_duplicates(
        ["candidate_id", "horizon_step"], keep="last"
    )
    candidate_summary_frame = pd.DataFrame(candidate_summary_rows).drop_duplicates(
        ["candidate_id"], keep="last"
    )
    _atomic_csv(candidate_metrics_frame, candidate_metrics_path)
    _atomic_csv(candidate_summary_frame, candidate_summary_path)
    _write_json(output / "selected_configs.json", selected_configs)

    if bool(upgrade.get("selection_only", False)):
        _write_json(output / "run_status.json", {
            "status": "selection_complete_test_not_evaluated",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_hash": protocol_hash,
            "selection_protocol_hash": selection_protocol_hash,
            "targets": list(config["targets"]),
            "test_set_used": False,
        })
        return output

    metrics_path = output / "metrics.csv"
    predictions_path = output / "predictions.csv"
    history_path = output / "training_history.csv"
    diagnostics_path = output / "model_diagnostics.csv"
    completed_final_path = output / "completed_final_tasks.txt"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
    history_frame = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    diagnostics_frame = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
    completed_final = _completed(completed_final_path)
    shift_frames: list[pd.DataFrame] = []
    final_seeds = [int(seed) for seed in upgrade.get("final_seeds", config["training"]["seeds"])]
    for target in config["targets"]:
        winner = selected_configs[target]
        lookback = int(winner["lookback_steps"])
        feature_set = str(winner["feature_set"])
        final_feature_columns = feature_set_columns(
            data, feature_set, target_station, target
        )
        sequence = make_sequences(
            data,
            final_feature_columns,
            data.target_columns[target_station][target],
            lookback,
            horizon,
        )
        split = official_split(sequence)
        x = _candidate_x(
            sequence,
            final_feature_columns,
            final_feature_columns,
            lookback,
            use_differences,
            use_calendar,
        )
        mase_scale = mase_scale_from_training(sequence, split["train"])
        shift_frames.append(distribution_shift_table(
            sequence,
            split,
            {
                "experiment": "06_lstm_upgrade",
                "target_station": target_station,
                "target": target,
                "lookback_steps": lookback,
            },
        ))
        for seed in final_seeds:
            final_group_id = f"{target}/seed={seed}/fold=0"
            if final_group_id in completed_final:
                print(f"[06_lstm_upgrade/final] skip completed: {final_group_id}")
                continue
            seed_metric_rows: list[dict[str, object]] = []
            seed_prediction_frames: list[pd.DataFrame] = []
            seed_history_rows: list[dict[str, object]] = []
            seed_diagnostic_rows: list[dict[str, object]] = []
            for feature_set_name, model_name, prediction in [
                ("persistence", "persistence", sequence.persistence[split["test"]]),
                ("seasonal_naive", "seasonal_naive", sequence.seasonal_naive[split["test"]]),
            ]:
                metadata = _metadata(
                    protocol_hash,
                    target_station,
                    target,
                    feature_set_name,
                    model_name,
                    seed,
                    lookback,
                    horizon,
                )
                metadata["task_id"] = f"{target}/{model_name}/seed={seed}/fold=0"
                seed_metric_rows.extend(calculate_metric_rows(
                    sequence.y[split["test"]], prediction, metadata, mase_scale
                ))
                seed_prediction_frames.append(prediction_frame(
                    sequence.target_start_times[split["test"]],
                    sequence.y[split["test"]],
                    prediction,
                    metadata,
                ))
            training = {**final_training, **winner["model_config"]}
            result = fit_residual_lstm_predict(
                x,
                sequence.y,
                sequence.persistence,
                sequence.seasonal_naive,
                split,
                seed,
                training,
                refit_train_validation=True,
            )
            metadata = _metadata(
                protocol_hash,
                target_station,
                target,
                feature_set,
                "residual_lstm",
                seed,
                lookback,
                horizon,
            )
            metadata["task_id"] = f"{target}/residual_lstm/seed={seed}/fold=0"
            seed_metric_rows.extend(calculate_metric_rows(
                sequence.y[split["test"]], result.prediction, metadata, mase_scale
            ))
            seed_prediction_frames.append(prediction_frame(
                sequence.target_start_times[split["test"]],
                sequence.y[split["test"]],
                result.prediction,
                metadata,
            ))
            for history_row in result.history:
                seed_history_rows.append({**metadata, **history_row})
            for horizon_index, baseline_name in enumerate(result.diagnostics["baseline_names"]):
                seed_diagnostic_rows.append({
                    **metadata,
                    "horizon_step": horizon_index + 1,
                    "horizon_hours": (horizon_index + 1) * 4,
                    "selected_naive": baseline_name,
                    "residual_gate": result.diagnostics["residual_gates"][horizon_index],
                    "validation_skill": result.diagnostics["validation_skill_by_horizon"][horizon_index],
                    "best_epoch": result.best_epoch,
                    "parameter_count": result.parameter_count,
                    "refit_samples": result.diagnostics["refit_samples"],
                })
            metrics = pd.concat([metrics, pd.DataFrame(seed_metric_rows)], ignore_index=True).drop_duplicates(
                ["task_id", "horizon_step"], keep="last"
            )
            predictions = pd.concat(
                [predictions, *seed_prediction_frames], ignore_index=True
            ).drop_duplicates(["task_id", "valid_time", "horizon_step"], keep="last")
            history_frame = pd.concat(
                [history_frame, pd.DataFrame(seed_history_rows)], ignore_index=True
            ).drop_duplicates(["task_id", "stage", "epoch"], keep="last")
            diagnostics_frame = pd.concat(
                [diagnostics_frame, pd.DataFrame(seed_diagnostic_rows)], ignore_index=True
            ).drop_duplicates(["task_id", "horizon_step"], keep="last")
            _atomic_csv(metrics, metrics_path)
            _atomic_csv(predictions, predictions_path)
            _atomic_csv(history_frame, history_path)
            _atomic_csv(diagnostics_frame, diagnostics_path)
            _append_completed(completed_final_path, final_group_id)
            completed_final.add(final_group_id)
            print(
                f"[06_lstm_upgrade/final] done: target={target}, seed={seed}, "
                f"lookback={lookback}, features={feature_set}"
            )

    _atomic_csv(metrics, metrics_path)
    _atomic_csv(predictions, predictions_path)
    _atomic_csv(history_frame, history_path)
    _atomic_csv(diagnostics_frame, diagnostics_path)
    pd.concat(shift_frames, ignore_index=True).to_csv(
        output / "distribution_shift.csv", index=False, encoding="utf-8-sig"
    )
    mean_sd_table(metrics).to_csv(
        output / "metrics_mean_sd.csv", index=False, encoding="utf-8-sig"
    )
    skill_scores_table(metrics).to_csv(
        output / "skill_scores.csv", index=False, encoding="utf-8-sig"
    )
    bootstrap = config["bootstrap"]
    bootstrap_frames = [
        paired_block_bootstrap_table(
            predictions,
            baseline,
            baseline,
            int(bootstrap["block_length_steps"]),
            int(bootstrap["replicates"]),
            int(bootstrap["seed"]),
        )
        for baseline in ("persistence", "seasonal_naive")
    ]
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(
        output / "paired_block_bootstrap.csv", index=False, encoding="utf-8-sig"
    )
    _write_json(output / "run_status.json", {
        "status": "complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": protocol_hash,
        "selection_protocol_hash": selection_protocol_hash,
        "targets": list(config["targets"]),
        "test_set_used": True,
        "candidate_count": len(candidate_summary_frame),
        "final_seed_count": len(final_seeds),
    })
    return output
