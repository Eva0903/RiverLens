"""Grouped full-history Kernel SHAP of original eval predictions; no gradients."""
from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .shap_kernel_common import (
    ReadOnlyCaseSource, TARGETS, background_indices, cached_arrays, digest,
    feature_group, read_json, save_arrays, summarize_backgrounds, write_csv, write_json,
)

PROTOCOL = "kernel_shap_whole_history_31players_v1"


def player_groups(columns):
    groups = []
    for i, name in enumerate(columns):
        if not name.startswith(("W_RAIN", "W_WIND")):
            groups.append(dict(name=name, columns=[i], color_columns=[i]))
    rain = [i for i, c in enumerate(columns) if c.startswith("W_RAIN")]
    wind = [i for i, c in enumerate(columns) if c.startswith("W_WIND")]
    if rain:
        color = columns.index("W_RAIN_4H") if "W_RAIN_4H" in columns else rain[0]
        groups.append(dict(name="W_RAIN_HISTORY", columns=rain, color_columns=[color]))
    if wind:
        color = columns.index("W_WIND_SPEED_MAX") if "W_WIND_SPEED_MAX" in columns else wind[0]
        groups.append(dict(name="W_WIND_GROUP", columns=wind, color_columns=[color]))
    covered = [i for g in groups for i in g["columns"]]
    if sorted(covered) != list(range(len(columns))):
        raise ValueError("Every input column must occur in exactly one player.")
    return groups


class CoalitionGame:
    """v(S)=mean_b f(x_S,b_notS), each player is an intact historical sequence.

    Empty and full endpoints are cached actual original-model predictions.
    Masks select whole variable trajectories and keep rain/wind encodings joint.
    No gradients, native LSTM replicas, corrected outputs or target labels.
    """
    def __init__(self, predict, x, background, groups, full_forecast, empty_forecast, coalition_batch=4):
        self.predict, self.x, self.background, self.groups = predict, x, background, groups
        self.batch = coalition_batch
        self.feature_player = np.empty(x.shape[1], dtype=int)
        for j, group in enumerate(groups):
            self.feature_player[group["columns"]] = j
        self.cache = {
            bytes([0] * len(groups)): np.asarray(empty_forecast, dtype=np.float64),
            bytes([1] * len(groups)): np.asarray(full_forecast, dtype=np.float64),
        }
        self.evaluated_hybrid_windows = 0

    def __call__(self, masks):
        masks = np.asarray(masks)
        if masks.ndim != 2 or masks.shape[1] != len(self.groups) or not np.isin(masks, [0, 1]).all():
            raise ValueError("Kernel game requires binary full-history coalition masks.")
        masks = masks.astype(np.uint8)
        pending = {}
        for mask in masks:
            key = bytes(mask.tolist())
            if key not in self.cache:
                pending[key] = mask
        items = list(pending.items())
        for begin in range(0, len(items), self.batch):
            subset = items[begin:begin + self.batch]
            m = np.stack([item[1] for item in subset])[:, self.feature_player].astype(bool)
            hybrid = np.where(m[:, None, None, :], self.x[None, None, :, :], self.background[None, :, :, :])
            n, n_bg, steps, features = hybrid.shape
            output = np.asarray(self.predict(hybrid.reshape(n * n_bg, steps, features)), dtype=np.float64)
            if output.shape != (n * n_bg, len(next(iter(self.cache.values())))) or not np.isfinite(output).all():
                raise ValueError("Invalid original-model coalition predictions.")
            values = output.reshape(n, n_bg, -1).mean(axis=1)
            for (key, _), value in zip(subset, values):
                self.cache[key] = value
            self.evaluated_hybrid_windows += n * n_bg
        return np.stack([self.cache[bytes(mask.tolist())] for mask in masks])


def explain_origin(game, groups, nsamples, seed):
    import shap
    count = len(groups)
    previous_rng = np.random.get_state()
    try:
        np.random.seed(seed)
        explainer = shap.KernelExplainer(game, np.zeros((1, count)), link="identity")
        values = explainer.shap_values(np.ones((1, count)), nsamples=nsamples, l1_reg=0.0, silent=True)
    finally:
        np.random.set_state(previous_rng)
    if isinstance(values, list):
        phi = np.stack([np.asarray(v)[0] for v in values], axis=1)
    else:
        phi = np.asarray(values)[0]
        if phi.ndim == 1:
            phi = phi[:, None]
    base = np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)
    full = game(np.ones((1, count)))[0]
    if phi.shape != (count, len(full)) or not np.isfinite(phi).all():
        raise ValueError("Invalid Kernel SHAP output shape/values.")
    residual = phi.sum(axis=0) + base - full
    tolerance = 1e-7 + 1e-5 * np.abs(full)
    if not (np.abs(residual) <= tolerance).all():
        raise ValueError("Kernel SHAP reconstruction failed; preserve diagnostic arrays.")
    masks = np.asarray(explainer.maskMatrix[:explainer.nsamplesAdded], dtype=np.uint8)
    design_rank = int(np.linalg.matrix_rank(np.vstack([masks, np.ones((1, count))])))
    if design_rank != count:
        raise ValueError("Kernel coalition design is rank-deficient; do not report the explanation.")
    return dict(phi=np.asarray(phi, dtype=np.float64), background_mean=base, predictions=full,
                reconstruction_residual=residual, masks=masks,
                coalition_outputs=np.asarray(explainer.ey[:explainer.nsamplesAdded], dtype=np.float64),
                kernel_weights=np.asarray(explainer.kernelWeights[:explainer.nsamplesAdded]),
                design_rank=np.asarray(design_rank), evaluated_hybrid_windows=np.asarray(game.evaluated_hybrid_windows))


def local_table(phi, x, groups, ids, issue_times, target, bg_seed):
    # phi: origins x players x six outputs. No decomposition within a player.
    rows = []
    for i, sample in enumerate(ids):
        for j, group in enumerate(groups):
            color = float(x[i][:, group["color_columns"]].mean())
            for lead in range(6):
                rows.append(dict(target=target, model_seed=42, background_seed=bg_seed,
                                 sample_index=int(sample), issue_time=str(issue_times[i]),
                                 horizon_hours=4 * (lead + 1), feature=group["name"],
                                 group=feature_group(group["name"]), signed_SHAP=float(phi[i, j, lead]),
                                 feature_history_mean=color, n_input_columns=len(group["columns"]),
                                 player_scope="entire_history_joint_columns"))
    return pd.DataFrame(rows)


def make_summary(local):
    return local.groupby(["target", "horizon_hours", "feature", "group", "background_seed"], as_index=False).agg(
        mean_abs_net_SHAP=("signed_SHAP", lambda v: float(np.abs(v).mean())),
        mean_signed_SHAP=("signed_SHAP", "mean"), n_origins=("sample_index", "nunique"))


def figures(output, summary, locals_):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    for target in summary.target.unique():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, lead in zip(axes, [4, 24]):
            s = summary[(summary.target == target) & (summary.horizon_hours == lead)].nlargest(12, "mean_abs_net_SHAP").sort_values("mean_abs_net_SHAP")
            ax.barh(s.feature, s.mean_abs_net_SHAP, xerr=s.background_draw_sd.fillna(0),
                    color=["#327ea1" if f.startswith("W_") else "#788891" for f in s.feature])
            ax.set_title(f"C {target}, {lead} h")
            ax.set_xlabel("Mean |whole-history player SHAP| (mg/L)")
        fig.suptitle("Grouped Kernel SHAP; seed42; bars: background-draw SD only")
        fig.tight_layout(); fig.savefig(directory / f"C_{target}_importance.png", dpi=180); plt.close(fig)
        for lead in [4, 24]:
            s = locals_[(locals_.target == target) & (locals_.horizon_hours == lead) & (locals_.background_seed == 2026)]
            if s.empty:
                continue
            names = s.groupby("feature").signed_SHAP.apply(lambda v: np.abs(v).mean()).nlargest(12).index[::-1]
            fig, ax = plt.subplots(figsize=(9, 6)); rng = np.random.default_rng(2026)
            for row, name in enumerate(names):
                part = s[s.feature == name]; v = part.feature_history_mean.to_numpy()
                color = (v - v.min()) / max(float(np.ptp(v)), 1e-12)
                ax.scatter(part.signed_SHAP, row + rng.uniform(-.2, .2, len(part)), c=color, cmap="coolwarm", vmin=0, vmax=1, s=12)
            ax.set_yticks(range(len(names)), names); ax.axvline(0, color="gray", lw=.8)
            ax.set_xlabel("Signed whole-history player SHAP (mg/L)")
            ax.set_title(f"C {target}, {lead} h; background2026\nColor: variable history mean; rain=current-rain series, wind=wind speed")
            fig.tight_layout(); fig.savefig(directory / f"C_{target}_h{lead}_direction.png", dpi=180); plt.close(fig)


def validate_spec(c):
    expected = dict(protocol=PROTOCOL, targets=TARGETS, model_seed=42, background_seeds=[2026, 3407, 99],
                    background_samples=64, kernel_nsamples=512, kernel_seed=2026, coalition_batch=4)
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValueError("Keep the declared fixed-model, background and Kernel SHAP protocol.")
    if c.get("device") not in ["cpu", "cuda"]:
        raise ValueError("Use cpu or cuda explicitly; no automatic fallback.")


def run(spec, root, mode="audit", max_new_origins=None):
    validate_spec(spec)
    import torch
    if spec["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate the existing GPU environment.")
    smoke = mode == "smoke"
    settings = dict(spec)
    if smoke:
        settings.update(background_samples=4, kernel_nsamples=64)
    output = Path(root) / spec["results_dir"] / ("07_kernel_smoke" if smoke else "07_kernel_shap")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".run.lock"
    try:
        with lock.open("x") as f:
            f.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError(f"Output locked at {lock}; confirm the previous process ended before removing this 07 lock.") from exc
    try:
        source = ReadOnlyCaseSource(spec, root)
        versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__)
        if mode != "audit":
            import shap
            versions["shap"] = shap.__version__
        stamp = dict(settings=settings, sources=source.fingerprint, versions=versions, code_sha256=digest(__file__),
                     readonly_source_sha256=digest(Path(__file__).with_name("shap_kernel_common.py")),
                     original_loader_sha256=digest(Path(__file__).with_name("interpretation_precision_fix.py")))
        guard = output / "protocol_manifest.json"
        if guard.exists():
            old = read_json(guard); old["versions"].pop("shap", None)
            compare = json.loads(json.dumps(stamp)); compare["versions"].pop("shap", None)
            if old != compare:
                raise RuntimeError("Kernel SHAP protocol/code/data changed; use a new results_dir.")
        write_json(guard, stamp)
        audit, collected, planned, complete = [], [], [], []
        new_origins = 0
        for target in (TARGETS[:1] if smoke else TARGETS):
            seq, split, checkpoint, meta, expected = source.case(target)
            predict = source._predictor_loader(checkpoint, spec["device"])
            actual = predict(seq.x[split["test"]])
            report = source._prediction_diagnostics(actual, expected)
            report.update(backend="original_eval_predictor_no_gradient", gradient_calls=0, automatic_training=False)
            write_json(output / f"C_{target}_inference_check.json", report)
            if not report["passed"]:
                raise ValueError("Original cached predictions do not match; Kernel SHAP is blocked.")
            groups = player_groups(seq.feature_columns)
            if len(groups) != 31 or len(seq.feature_columns) != 39:
                raise ValueError("Expected 39 inputs covered by the declared 31 whole-history players.")
            write_json(output / f"C_{target}_players.json", [dict(**g, input_columns=[seq.feature_columns[i] for i in g["columns"]],
                            color_input_columns=[seq.feature_columns[i] for i in g["color_columns"]]) for g in groups])
            audit.append(dict(target=target, checkpoint_metadata_sha256=digest(checkpoint / "metadata.json"),
                              n_train=len(split["train"]), n_test=len(split["test"]), n_input_columns=39, n_players=31,
                              original_prediction_max_abs=report["max_abs_difference"], passed=True))
            write_csv(output / "case_audit.csv", pd.DataFrame(audit))
            print(f"[Kernel SHAP audit] C/{target}: original predictions passed, max_abs={report['max_abs_difference']:.3g}; no gradient.", flush=True)
            if mode == "audit":
                del predict
                continue
            test_ids = np.asarray(split["test"][:1] if smoke else split["test"], dtype=np.int64)
            for bg_seed in (spec["background_seeds"][:1] if smoke else spec["background_seeds"]):
                task_key = f"C_{target}_bg{bg_seed}"
                planned.append(task_key)
                directory = output / "tasks" / task_key
                directory.mkdir(parents=True, exist_ok=True)
                bg_ids = background_indices(split["train"], settings["background_samples"], bg_seed)
                background = seq.x[bg_ids]
                empty = predict(background).astype(np.float64).mean(axis=0)
                task = dict(target=target, station="C", model_seed=42, background_seed=bg_seed,
                            background_indices=bg_ids.tolist(), test_indices=test_ids.tolist(), groups=groups,
                            checkpoint_metadata_sha256=digest(checkpoint / "metadata.json"),
                            scope="whole_history_player_group_Kernel_SHAP_not_time_decomposed_or_causal")
                write_json(directory / "task.json", task)
                parts, ids = [], []
                for position, sample_id in enumerate(test_ids):
                    path = directory / f"origin_{position:04d}.npz"
                    record_stamp = dict(protocol=stamp, task=task, sample_index=int(sample_id), position=position)
                    arrays = cached_arrays(path, record_stamp)
                    if arrays is None:
                        if max_new_origins is not None and new_origins >= max_new_origins:
                            continue
                        print(f"[Kernel SHAP] {task_key}: origin {position + 1}/{len(test_ids)}, all six horizons", flush=True)
                        game = CoalitionGame(predict, seq.x[sample_id], background, groups, actual[position], empty, spec["coalition_batch"])
                        arrays = explain_origin(game, groups, settings["kernel_nsamples"], spec["kernel_seed"] + int(sample_id))
                        arrays["sample_index"] = np.asarray(sample_id)
                        save_arrays(path, record_stamp, arrays)
                        new_origins += 1
                    parts.append(arrays["phi"]); ids.append(sample_id)
                if len(parts) == len(test_ids):
                    local = local_table(np.stack(parts), seq.x[test_ids], groups, test_ids, seq.issue_times[test_ids], target, bg_seed)
                    write_csv(directory / "local_contributions.csv", local)
                    write_csv(directory / "feature_importance.csv", make_summary(local))
                    write_json(directory / "complete.json", dict(task=task, smoke_only=smoke,
                        files_sha256={f.name: digest(f) for f in directory.iterdir() if f.suffix in [".npz", ".csv"]}))
                    collected.append(local); complete.append(task_key)
                    print(f"[Kernel SHAP] {task_key}: complete", flush=True)
                else:
                    print(f"[Kernel SHAP] {task_key}: partial {len(parts)}/{len(test_ids)}", flush=True)
            del predict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if collected:
            local = pd.concat(collected, ignore_index=True)
            each = make_summary(local)
            summary = summarize_backgrounds(each, ["target", "horizon_hours", "feature", "group"])
            write_csv(output / "feature_by_background.csv", each)
            write_csv(output / "feature_summary.csv", summary)
            grouped = local.groupby(["target", "horizon_hours", "background_seed", "sample_index", "group"], as_index=False).signed_SHAP.sum()
            grouped = grouped.rename(columns={"group": "feature"}); grouped["group"] = grouped.feature
            write_csv(output / "group_summary.csv", summarize_backgrounds(make_summary(grouped), ["target", "horizon_hours", "group"]))
            figures(output, summary, local)
        write_json(output / "run_status.json", dict(protocol=PROTOCOL, mode=mode,
            status="audit_complete" if mode == "audit" else ("smoke_only" if smoke else ("complete" if len(complete) == len(planned) else "partial")),
            completed_background_tasks=complete, missing_background_tasks=[p for p in planned if p not in complete],
            expected_background_tasks=12, expected_horizon_views=72, newly_explained_origins=new_origins,
            automatic_training=False, gradient_calls=0,
            uncertainty_scope="background_choice_only_one_fitted_seed_finite_coalition_approximation"))
        return output
    except BaseException as exc:
        write_json(output / "run_status.json", dict(status="interrupted_or_failed", error=repr(exc),
                                                    gradient_calls=0, automatic_training=False))
        raise
    finally:
        lock.unlink(missing_ok=True)
