"""06: same-backbone residualization and validation-only gates; twenty new fits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from .baseline_random import PROTOCOL as SPLIT_PROTOCOL, prepare_random_pool
from .baseline_within_split import training_mase_scale
from .config import resolve_path
from .data import prepare_dataset
from .evaluation import calculate_metric_rows, mean_sd_table, prediction_frame
from .meteorology_random import FIXED_WINDOWS, calendar_block_ci
from .models import choose_validation_baseline, optimal_residual_gates, fit_torch_predict
from .station_rotation_random import RandomStationRotationRunner, SOURCES, SEEDS, digest, pool_tables

PROTOCOL = "same_backbone_residual_random_v1"
FEATURES = "all_stations_weather_rain_lags"
OUTPUTS = ("persistence", "seasonal_naive", "validation_selected_naive", "residual_lstm", "gated_residual_lstm")
CONTRASTS = (
    ("gated_vs_lstm", "ordinary_lstm", "gated_residual_lstm", "primary"),
    ("gated_vs_selected_naive", "validation_selected_naive", "gated_residual_lstm", "primary"),
    ("residual_vs_lstm", "ordinary_lstm", "residual_lstm", "secondary"),
    ("residual_vs_selected_naive", "validation_selected_naive", "residual_lstm", "secondary"),
    ("gating_increment", "residual_lstm", "gated_residual_lstm", "secondary"),
)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_csv(path, frame):
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def validate_config(c):
    if c.get("residual_protocol") != PROTOCOL or c.get("baseline_protocol") != SPLIT_PROTOCOL:
        raise ValueError("Use configs/residual_random_gpu.json; old 06 configs are incompatible.")
    if c.get("target_station") != "C" or c.get("feature_set") != FEATURES or c.get("frequency") != "4h":
        raise ValueError("06 fixes target station C, full meteorological inputs, and 4h frequency.")
    if c.get("forecast_horizon_steps") != 6 or c.get("target_default_windows") != FIXED_WINDOWS or c.get("split_seed") != 42:
        raise ValueError("Keep the 03 windows, six horizons, and split_seed=42.")
    if (c["split"]["train_fraction"], c["split"]["validation_fraction"]) != (.8, .1):
        raise ValueError("Keep the same two-stage 80/10/10 random-window split as 04/05.")
    if c["missing_data"]["interpolation_policy"] != "causal_ffill":
        raise ValueError("Use causal_ffill inputs.")
    if c.get("gate_grid_size") != 21 or c.get("refit_train_validation") is not False:
        raise ValueError("Use 21 fixed gate values, and no train+validation refit.")
    if c.get("append_first_differences") is not False or c.get("append_calendar_features") is not False:
        raise ValueError("Extra difference/calendar channels are disabled for the matched-backbone comparison.")
    if c.get("interpretability", {}).get("enabled", False):
        raise ValueError("SHAP is deferred in 06.")
    t = c["training"]
    expected = dict(hidden_size=64, num_layers=2, dropout=.1, batch_size=32,
                    learning_rate=.001, device="cuda", deterministic="strict")
    if any(t.get(k) != v for k, v in expected.items()):
        raise ValueError("Use the unchanged 04 ordinary LSTM backbone, Adam and CUDA settings.")
    # Reject silently ignored old-upgrade knobs such as AdamW/loss/scheduler settings.
    if set(t) != set(expected) | {"seeds", "max_epochs", "patience"}:
        raise ValueError("training must contain only the matched 04 training keys.")
    smoke = c.get("smoke_only", False)
    desired = (["DO"], [42], 2, 1) if smoke else (list(FIXED_WINDOWS), SEEDS, 512, 100)
    if (c["targets"], t["seeds"], t["max_epochs"], t["patience"]) != desired:
        raise ValueError("Formal 06 fixes four targets, five seeds, 512 epochs/patience100; use the separate two-epoch smoke config.")
    if bool("smoke" in c["results_dir"].lower()) != bool(smoke):
        raise ValueError("Smoke and formal results_dir must be separate, with 'smoke' only in the smoke path.")
    if c["bootstrap"] != dict(block_length_steps=6, replicates=2000, seed=2026):
        raise ValueError("Keep the 04 calendar-block settings.")


def fit_residual_once(sequence, split, seed, training):
    """One unchanged ordinary LSTM trained on residuals, returning two forecasts.

    Appending validation to the INFERENCE index list exposes validation forecasts
    without retraining or changing train/early-stop partitions. Test labels never
    enter baseline choice, scaling, early stopping or gate selection.
    """
    selected, names = choose_validation_baseline(sequence.y, sequence.persistence, sequence.seasonal_naive, split["validation"])
    residual_target = (sequence.y - selected).astype(np.float32)
    inference_split = dict(split)
    inference_split["test"] = np.concatenate([split["validation"], split["test"]])
    settings = dict(training, compute_shap=False)
    result = fit_torch_predict("lstm", sequence.x, residual_target, inference_split, seed, settings)
    nval = len(split["validation"])
    expected = (nval + len(split["test"]), sequence.y.shape[1])
    if result.prediction.shape != expected or not np.isfinite(result.prediction).all():
        raise ValueError("Residual inference must return finite validation and test predictions in order.")
    validation_residual, test_residual = result.prediction[:nval], result.prediction[nval:]
    gates = optimal_residual_gates(sequence.y[split["validation"]], selected[split["validation"]], validation_residual, 21)
    test_baseline = selected[split["test"]]
    predictions = dict(
        persistence=sequence.persistence[split["test"]],
        seasonal_naive=sequence.seasonal_naive[split["test"]],
        validation_selected_naive=test_baseline,
        residual_lstm=(test_baseline + test_residual).astype(np.float32),
        gated_residual_lstm=(test_baseline + gates * test_residual).astype(np.float32),
    )
    validation_raw = selected[split["validation"]] + validation_residual
    validation_gated = selected[split["validation"]] + gates * validation_residual
    gates_table = pd.DataFrame(dict(horizon_step=np.arange(1, 7), horizon_hours=np.arange(1, 7) * 4,
        selected_baseline=names, gate=gates,
        validation_naive_MAE=np.abs(sequence.y[split["validation"]] - selected[split["validation"]]).mean(axis=0),
        validation_residual_MAE=np.abs(sequence.y[split["validation"]] - validation_raw).mean(axis=0),
        validation_gated_MAE=np.abs(sequence.y[split["validation"]] - validation_gated).mean(axis=0)))
    if (gates_table.validation_gated_MAE > gates_table.validation_naive_MAE + 1e-6).any():
        raise AssertionError("Gate zero is always available; validation gating cannot exceed its naive MAE.")
    return predictions, result.history, result.best_epoch, result.parameter_count, gates_table


class RandomResidualRunner(RandomStationRotationRunner):
    """Reuse 05's audited C-reference reader, with separate 06 task transactions."""
    def _guard(self, filename, payload):
        if filename == "reference_04_provenance.json":
            payload = dict(payload)
            payload["verified_C_lstm_tasks"] = payload.pop("reused_lstm_tasks")
            payload["reused_ordinary_lstm_tasks"] = 20
            payload["selected_feature_set"] = FEATURES
        return super()._guard(filename, payload)

    def __init__(self, config, max_tasks=None, continue_on_error=False):
        validate_config(config)
        if max_tasks is not None and max_tasks < 0:
            raise ValueError("max_tasks must be nonnegative.")
        self.config = config
        self.clean_config = {k: v for k, v in config.items() if not k.startswith("_")}
        self.config_sha256 = hashlib.sha256(json.dumps(self.clean_config, sort_keys=True).encode()).hexdigest()
        self.protocol_hash = self.config_sha256[:16]
        self.experiment = "06_residual_random"
        self.output = resolve_path(config, config["results_dir"]) / self.experiment
        self.output.mkdir(parents=True, exist_ok=True)
        self._lock_handle = None
        self.max_tasks, self.continue_on_error, self.tasks_run = max_tasks, continue_on_error, 0
        self.completed, self.failed = set(), {}
        self.reference_verified = False
        self.reused_metrics, self.reused_predictions, self.reused_diagnostics = (pd.DataFrame() for _ in range(3))
        self._acquire_run_lock()
        try:
            self._guard("run_config.json", self.clean_config)
            atomic_json(self.output / "environment.json", self._environment_manifest())
            data_dir = resolve_path(config, config["data_dir"])
            self.data_hashes = {p.name: digest(p) for p in sorted(data_dir.glob("*.xlsx"))}
            self.source_hashes = {n: digest(Path(__file__).with_name(n)) for n in SOURCES + ["station_rotation_random.py", "residual_random.py"]}
            self._guard("data_fingerprints.json", self.data_hashes)
            self._guard("source_fingerprints.json", self.source_hashes)
            self.data = prepare_random_pool(prepare_dataset(data_dir, config), config)
            atomic_csv(self.output / "data_audit.csv", self.data.audit.drop(columns=["splitwise_missing_cells_after_imputation"], errors="ignore"))
            if not config["smoke_only"]:
                # Validates all 40 C tasks; only the twenty matching full-input LSTMs
                # enter this comparison. No 04/05 file is modified.
                self._load_reference()
                self.reused_metrics = self.reused_metrics[self.reused_metrics.feature_set == FEATURES].copy()
                self.reused_predictions = self.reused_predictions[self.reused_predictions.feature_set == FEATURES].copy()
                self.reused_diagnostics = self.reused_diagnostics[self.reused_diagnostics.feature_set == FEATURES].copy()
                for frame in (self.reused_metrics, self.reused_predictions, self.reused_diagnostics):
                    frame["model"] = "ordinary_lstm"
                if self.reused_metrics.task_id.nunique() != 20:
                    raise ValueError("06 requires exactly twenty verified full-input ordinary LSTM tasks.")
                atomic_csv(self.output / "reused_ordinary_metrics.csv", self.reused_metrics)
            failed_path = self.output / "failed_tasks.json"
            if failed_path.exists():
                self.failed = json.loads(failed_path.read_text(encoding="utf-8"))
            self._scan_completed()
            self._write_run_status()
        except BaseException:
            self._release_run_lock()
            raise

    def _fit_key(self, target, seed):
        return f"C_{target}_seed{seed}"

    def _fit_id(self, target, seed):
        return f"{self.protocol_hash}|{self.experiment}|C|{target}|{FEATURES}|{seed}"

    def _valid_commit(self, target, seed):
        directory = self.output / "tasks" / self._fit_key(target, seed)
        try:
            commit = json.loads((directory / "commit.json").read_text(encoding="utf-8"))
            if commit["fit_id"] != self._fit_id(target, seed) or commit["protocol_hash"] != self.protocol_hash:
                return False
            names = {"metrics.csv", "predictions.csv", "training_history.csv", "gate_diagnostics.csv"}
            if set(commit["files_sha256"]) != names or any(digest(directory / n) != h for n, h in commit["files_sha256"].items()):
                return False
            m, p, h, g = [pd.read_csv(directory / n) for n in ["metrics.csv", "predictions.csv", "training_history.csv", "gate_diagnostics.csv"]]
            if len(m) != 30 or set(m.model) != set(OUTPUTS) or m.duplicated(["model", "horizon_step"]).any():
                return False
            if h.empty or h.epoch.isna().any() or not np.isfinite(h[["train_loss", "validation_loss"]].to_numpy()).all():
                return False
            neural = m[m.model == "residual_lstm"]
            if set(h.epoch.astype(int)) != set(range(1, int(h.epoch.max()) + 1)) or int(neural.best_epoch.iloc[0]) not in set(h.epoch):
                return False
            if len(g) != 6 or set(g.horizon_step) != set(range(1, 7)) or not g.gate.between(0, 1).all():
                return False
            if not np.isfinite(m[["MAE", "RMSE"]].to_numpy()).all() or not np.isfinite(p[["y_true", "y_pred"]].to_numpy()).all():
                return False
            n = int(m.n_test.iloc[0])
            if len(p) != 5 * 6 * n or p.duplicated(["model", "target_start_time", "horizon_step"]).any():
                return False
            if set(p.model) != set(OUTPUTS) or not p.groupby(["model", "horizon_step"]).size().eq(n).all():
                return False
            return True
        except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError):
            return False

    def _scan_completed(self):
        self.completed = {self._fit_key(t, s) for t in self.config["targets"] for s in self.config["training"]["seeds"] if self._valid_commit(t, s)}
        for key in self.completed:
            self.failed.pop(key, None)

    def _save_fit(self, target, seed, seq, split, forecasts, history, epoch, parameters, gates):
        if set(forecasts) != set(OUTPUTS) or not history:
            raise ValueError("One fit must provide five forecast outputs and nonempty history.")
        if not self.reused_metrics.empty:
            ordinary = self.reused_metrics[(self.reused_metrics.target == target) & (self.reused_metrics.seed == seed)]
            if len(ordinary) != 6 or set(ordinary.parameter_count.astype(int)) != {int(parameters)}:
                raise ValueError("Residual/ordinary LSTM parameter counts must be identical.")
        fit_id = self._fit_id(target, seed)
        common = dict(fit_id=fit_id, protocol_hash=self.protocol_hash, experiment=self.experiment,
            subexperiment="matched_backbone_residualization", fold=0, split_scheme=SPLIT_PROTOCOL,
            target_station="C", target=target, target_column=seq.target_column, seed=seed,
            feature_set=FEATURES, lookback_steps=self.config["target_default_windows"][target],
            lookback_days=self.config["target_default_windows"][target] / 6, forecast_horizon_steps=6,
            n_train=len(split["train"]), n_validation=len(split["validation"]), n_test=len(split["test"]),
            best_epoch=epoch, parameter_count=parameters, result_origin="new_06",
            source_experiment=self.experiment, source_protocol_hash=self.protocol_hash)
        metric_rows, pred_rows = [], []
        scale = training_mase_scale(seq, split["train"])
        for method in OUTPUTS:
            prediction = forecasts[method]
            if prediction.shape != seq.y[split["test"]].shape or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid predictions for {method}.")
            metadata = dict(common, model=method, task_id=fit_id + "|" + method,
                            n_features=len(seq.feature_columns) if "lstm" in method else 1,
                            best_epoch=epoch if "lstm" in method else 0,
                            parameter_count=parameters if "lstm" in method else 0)
            metric_rows.extend(calculate_metric_rows(seq.y[split["test"]], prediction, metadata, scale))
            pred_rows.append(prediction_frame(seq.target_start_times[split["test"]], seq.y[split["test"]], prediction, metadata))
        h = pd.DataFrame(history)
        for key, value in dict(common, model="residual_lstm", task_id=fit_id).items():
            h[key] = value
        g = gates.copy()
        for key, value in dict(fit_id=fit_id, target=target, target_station="C", seed=seed).items():
            g[key] = value
        directory = self.output / "tasks" / self._fit_key(target, seed)
        directory.mkdir(parents=True, exist_ok=True)
        # Invalidate any previous incomplete/corrupt transaction before replacing it.
        (directory / "commit.json").unlink(missing_ok=True)
        frames = {"metrics.csv": pd.DataFrame(metric_rows), "predictions.csv": pd.concat(pred_rows, ignore_index=True),
                  "training_history.csv": h, "gate_diagnostics.csv": g}
        for name, frame in frames.items():
            atomic_csv(directory / name, frame)
        atomic_json(directory / "commit.json", dict(fit_id=fit_id, protocol_hash=self.protocol_hash,
                    files_sha256={n: digest(directory / n) for n in frames}))
        if not self._valid_commit(target, seed):
            (directory / "commit.json").unlink(missing_ok=True)
            raise ValueError("Fit transaction failed its completeness checks.")

    def run(self):
        try:
            for target in self.config["targets"]:
                seq, split, arms, audit, manifest, groups = pool_tables(self.data, self.config, "C", target, self.protocol_hash)
                seq = arms[FEATURES]
                groups = groups[groups.feature_set == FEATURES].copy()
                groups["model"] = "residual_lstm"
                groups["raw_and_gated_share_one_fit"] = True
                # The common pool helper is shared with 05; label this audit as 06.
                for frame in (audit, manifest, groups):
                    frame.drop(columns=["station_rotation_protocol"], inplace=True, errors="ignore")
                    frame["residual_protocol"] = PROTOCOL
                from .runner import _upsert_csv
                for filename, frame in [("sample_support_audit.csv", audit), ("sample_manifest.csv", manifest), ("input_groups.csv", groups)]:
                    _upsert_csv(frame, self.output / filename, "support_id")
                print(f"[06] C/{target}: L={seq.x.shape[1]}, F={seq.x.shape[2]}, train/val/test=" + "/".join(str(len(split[k])) for k in ["train", "validation", "test"]), flush=True)
                for seed in self.config["training"]["seeds"]:
                    key = self._fit_key(target, seed)
                    if key in self.completed or (self.max_tasks is not None and self.tasks_run >= self.max_tasks):
                        continue
                    try:
                        result = fit_residual_once(seq, split, seed, self.config["training"])
                        self._save_fit(target, seed, seq, split, *result)
                        self.completed.add(key)
                        self.failed.pop(key, None)
                        self.tasks_run += 1
                        self._write_run_status()
                        print(f"  done {key}; one LSTM fit, raw + gated predictions", flush=True)
                    except Exception as exc:
                        self.failed[key] = f"{type(exc).__name__}: {exc}"
                        with (self.output / "errors.log").open("a", encoding="utf-8") as handle:
                            handle.write(key + "\n" + traceback.format_exc() + "\n")
                        self._write_run_status()
                        if not self.continue_on_error:
                            raise
            self._postprocess()
            return self.output
        except BaseException:
            self._write_run_status()
            raise
        finally:
            self._release_run_lock()

    def _postprocess(self):
        self._scan_completed()
        tables = {n: [] for n in ["metrics.csv", "predictions.csv", "training_history.csv", "gate_diagnostics.csv"]}
        for key in sorted(self.completed):
            for name in tables:
                tables[name].append(pd.read_csv(self.output / "tasks" / key / name))
        merged = {}
        for name, frames in tables.items():
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            merged[name] = frame
            if not frame.empty:
                atomic_csv(self.output / name, frame)
            elif (self.output / name).exists():
                (self.output / name).unlink()
        metric_frames = [x for x in [merged["metrics.csv"], self.reused_metrics] if not x.empty]
        prediction_frames = [x for x in [merged["predictions.csv"], self.reused_predictions] if not x.empty]
        if not metric_frames:
            self._write_run_status()
            return
        m = pd.concat(metric_frames, ignore_index=True)
        p = pd.concat(prediction_frames, ignore_index=True)
        for column in ["target_start_time", "valid_time"]:
            p[column] = pd.to_datetime(p[column])
        atomic_csv(self.output / "combined_metrics.csv", m)
        atomic_csv(self.output / "combined_predictions.csv", p)
        report = m.copy()
        report["experiment"], report["subexperiment"] = self.experiment, "matched_backbone_residualization"
        atomic_csv(self.output / "metrics_mean_sd.csv", mean_sd_table(report))
        if not merged["training_history.csv"].empty:
            h = merged["training_history.csv"]
            d = h.groupby(["fit_id", "target", "seed", "best_epoch", "parameter_count"]).agg(epochs_run=("epoch", "max")).reset_index()
            d["max_epochs"], d["patience"] = self.config["training"]["max_epochs"], self.config["training"]["patience"]
            d["reached_epoch_cap"] = d.epochs_run >= d.max_epochs
            atomic_csv(self.output / "model_diagnostics.csv", d)
        paired, blocks = compare_models(m, p, self.config)
        atomic_csv(self.output / "paired_seed_horizon.csv", paired)
        atomic_csv(self.output / "paired_block_bootstrap.csv", blocks)
        atomic_csv(self.output / "residual_gain_summary.csv", summarize_gains(paired, blocks, len(self.config["training"]["seeds"])))
        self._write_run_status()

    def _write_run_status(self):
        expected = len(self.config["targets"]) * len(self.config["training"]["seeds"])
        smoke = self.config["smoke_only"]
        complete = len(self.completed) == expected and not self.failed and (smoke or self.reference_verified)
        atomic_json(self.output / "failed_tasks.json", self.failed)
        atomic_json(self.output / "run_status.json", dict(experiment=self.experiment, residual_protocol=PROTOCOL,
            expected_new_lstm_fits=expected, completed_new_lstm_fits=len(self.completed),
            reused_ordinary_lstm_fits=20 if self.reference_verified else 0,
            unresolved_failed_tasks=len(self.failed), failed_task_ids=sorted(self.failed),
            run_status=("smoke_complete" if smoke else "complete") if complete else (
                "audit_only" if self.max_tasks == 0 and not self.completed else "partial"),
            smoke_only=smoke, raw_and_gated_share_one_fit=True, additional_neural_refit=False,
            target_windows_steps=FIXED_WINDOWS, max_epochs=self.config["training"]["max_epochs"],
            patience=self.config["training"]["patience"], split_scheme=SPLIT_PROTOCOL,
            evaluation_scope="exploratory_fixed_random_window_split_and_fixed_budget_not_temporal_generalization"))


def compare_models(metrics, predictions, config):
    rows, intervals = [], []
    keys = ["target", "seed", "horizon_hours"]
    for contrast, reference, augmented, role in CONTRASTS:
        left, right = metrics[metrics.model == reference], metrics[metrics.model == augmented]
        paired = left.merge(right, on=keys, suffixes=("_ref", "_aug"), validate="one_to_one")
        for row in paired.to_dict("records"):
            rows.append(dict(**{k: row[k] for k in keys}, contrast=contrast, role=role,
                reference_model=reference, augmented_model=augmented,
                MAE_reference=row["MAE_ref"], MAE_augmented=row["MAE_aug"],
                MAE_reduction=row["MAE_ref"] - row["MAE_aug"],
                MAE_reduction_percent=100 * (1 - row["MAE_aug"] / row["MAE_ref"]) if row["MAE_ref"] > 0 else np.nan,
                RMSE_reduction=row["RMSE_ref"] - row["RMSE_aug"], R2_gain=row["R2_aug"] - row["R2_ref"]))
        for target, group in paired.groupby("target"):
            seeds = [s for s, g in group.groupby("seed") if len(g) == 6 and set(g.horizon_hours) == {4, 8, 12, 16, 20, 24}]
            if not seeds:
                continue
            a = predictions[(predictions.target == target) & (predictions.model == reference) & predictions.seed.isin(seeds)]
            b = predictions[(predictions.target == target) & (predictions.model == augmented) & predictions.seed.isin(seeds)]
            join = a.merge(b, on=["seed", "target_start_time", "horizon_hours"], suffixes=("_ref", "_aug"), validate="one_to_one")
            if len(join) != len(a) or len(join) != len(b) or not np.array_equal(join.y_true_ref.to_numpy(dtype=np.float32), join.y_true_aug.to_numpy(dtype=np.float32)):
                raise ValueError("Compared models must share all test origins and observed labels.")
            # Recompute losses from predictions, rather than trusting serialized errors.
            join["delta"] = abs(join.y_pred_ref - join.y_true_ref) - abs(join.y_pred_aug - join.y_true_aug)
            origin = join.groupby(["target_start_time", "horizon_hours"]).delta.mean().unstack()
            origin.index = pd.DatetimeIndex(origin.index)
            for lead in [0, 4, 8, 12, 16, 20, 24]:
                delta = origin.mean(axis=1) if lead == 0 else origin[lead]
                result = calendar_block_ci(delta, config["bootstrap"]["block_length_steps"], config["bootstrap"]["replicates"], config["bootstrap"]["seed"])
                intervals.append(dict(target=target, contrast=contrast, role=role, horizon_hours=lead,
                    reference_model=reference, augmented_model=augmented, n_paired_seeds=len(seeds),
                    status="complete" if set(seeds) == set(config["training"]["seeds"]) else "partial_paired_seeds",
                    interval_scope="conditional_on_fixed_split_and_fitted_models_exploratory_unadjusted", **result))
    columns = keys + ["contrast", "role", "reference_model", "augmented_model", "MAE_reference", "MAE_augmented", "MAE_reduction", "MAE_reduction_percent", "RMSE_reduction", "R2_gain"]
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(intervals, columns=[
        "target", "contrast", "role", "horizon_hours", "reference_model", "augmented_model", "n_paired_seeds", "status",
        "interval_scope", "MAE_reduction", "ci95_lower", "ci95_upper", "n_test_origins", "n_calendar_blocks", "n_nonempty_calendar_blocks"])


def summarize_gains(paired, intervals, expected):
    rows = []
    for (target, contrast), group in paired.groupby(["target", "contrast"]):
        complete = group.groupby("seed").filter(lambda g: len(g) == 6 and set(g.horizon_hours) == {4, 8, 12, 16, 20, 24})
        if complete.empty:
            continue
        means = complete.groupby("seed")[["MAE_reference", "MAE_augmented", "RMSE_reduction", "R2_gain"]].mean()
        gain = means.MAE_reference - means.MAE_augmented
        ref, aug = means.MAE_reference.mean(), means.MAE_augmented.mean()
        ci = intervals[(intervals.target == target) & (intervals.contrast == contrast) & (intervals.horizon_hours == 0)]
        rows.append(dict(target=target, contrast=contrast, role=group.role.iloc[0], reference_model=group.reference_model.iloc[0],
            augmented_model=group.augmented_model.iloc[0], n_paired_seeds=len(means), expected_seeds=expected,
            status="complete" if len(means) == expected else "partial_paired_seeds", MAE_reference=ref, MAE_augmented=aug,
            MAE_reduction=ref - aug, MAE_reduction_percent=100 * (1 - aug / ref) if ref > 0 else np.nan,
            n_seeds_with_positive_reduction=int((gain > 0).sum()), RMSE_reduction=means.RMSE_reduction.mean(),
            R2_gain=means.R2_gain.mean(), ci95_lower=ci.ci95_lower.iloc[0] if len(ci) else np.nan,
            ci95_upper=ci.ci95_upper.iloc[0] if len(ci) else np.nan))
    return pd.DataFrame(rows, columns=["target", "contrast", "role", "reference_model", "augmented_model", "n_paired_seeds",
        "expected_seeds", "status", "MAE_reference", "MAE_augmented", "MAE_reduction", "MAE_reduction_percent",
        "n_seeds_with_positive_reduction", "RMSE_reduction", "R2_gain", "ci95_lower", "ci95_upper"])
