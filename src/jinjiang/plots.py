from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_MPL_CONFIG = Path(__file__).resolve().parents[2] / "results" / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(_DEFAULT_MPL_CONFIG))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


TARGET_ORDER = ["DO", "NH3N", "CODMN", "TP"]


def _line_figure(data: pd.DataFrame, hue: str, output: Path, title: str) -> None:
    if data.empty:
        return
    targets = [target for target in TARGET_ORDER if target in set(data["target"])]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, target in zip(axes.ravel(), targets):
        selected = data[data["target"] == target]
        sns.lineplot(data=selected, x="horizon_hours", y="R2", hue=hue, marker="o", errorbar="sd", ax=axis)
        axis.set_title(target)
        axis.set_xlabel("Forecast horizon (h)")
        axis.set_ylabel("Test R²")
        axis.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
        axis.legend(fontsize=7)
    for axis in axes.ravel()[len(targets):]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _forecastability_heatmaps(metrics: pd.DataFrame, figure_dir: Path) -> None:
    selected = metrics[metrics["subexperiment"] == "forecastability_surface"]
    if selected.empty:
        return
    grouped = selected.groupby(["target", "lookback_days", "horizon_hours"], as_index=False)["R2"].mean()
    for target in TARGET_ORDER:
        target_data = grouped[grouped["target"] == target]
        if target_data.empty:
            continue
        matrix = target_data.pivot(index="lookback_days", columns="horizon_hours", values="R2")
        figure, axis = plt.subplots(figsize=(7, 7))
        sns.heatmap(matrix, cmap="vlag", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", ax=axis)
        axis.set_title(f"{target}: forecastability surface")
        axis.set_xlabel("Forecast horizon (h)")
        axis.set_ylabel("Input history (days)")
        figure.tight_layout()
        figure.savefig(figure_dir / f"forecastability_{target}.png", dpi=300, bbox_inches="tight")
        plt.close(figure)


def generate_experiment_figures(metrics_path: Path, experiment: str, output_dir: Path) -> None:
    if not metrics_path.exists():
        return
    metrics = pd.read_csv(metrics_path)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if experiment == "01_baseline":
        selected = metrics[metrics["subexperiment"] == "unified_model_benchmark"]
        _line_figure(selected, "model", figure_dir / "model_benchmark_R2.png", "Unified model benchmark")
    elif experiment == "02_station_ablation":
        _line_figure(metrics, "feature_set", figure_dir / "station_ablation_R2.png", "Station information value")
    elif experiment == "03_window_horizon":
        _forecastability_heatmaps(metrics, figure_dir)
    elif experiment == "04_meteorology":
        _line_figure(metrics, "feature_set", figure_dir / "meteorology_R2.png", "Conditional meteorological value")
    elif experiment == "05_robustness":
        for subexperiment in ["rolling_origin", "target_station_rotation"]:
            selected = metrics[metrics["subexperiment"] == subexperiment]
            _line_figure(selected, "feature_set", figure_dir / f"{subexperiment}_R2.png", subexperiment.replace("_", " ").title())
