"""Generate artificial loader fixtures; never a substitute for research observations."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate(destination: str | Path, periods: int = 420, seed: int = 123) -> Path:
    if periods < 120:
        raise ValueError("At least 120 four-hour periods are required.")
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    names = {"A": ("Anshun", "监测时间_安顺廊桥"),
             "B": ("Chengren", "监测时间_成仁桥"),
             "C": ("Yongan", "监测时间_永安")}
    destinations = [folder / f"station_{s}_{label}.xlsx" for s, (label, _) in names.items()]
    destinations += [folder / "weather_hourly.xlsx", folder / "SYNTHETIC_ONLY.txt"]
    if any(p.exists() for p in destinations):
        raise FileExistsError("Refusing to overwrite existing input files; choose a new empty directory.")
    rng = np.random.default_rng(seed)
    times = pd.date_range("2000-01-01", periods=periods, freq="4h")
    t = np.arange(periods)
    wave = np.sin(2 * np.pi * t / 6)
    for i, (station, (label, time_column)) in enumerate(names.items()):
        columns = {time_column: times,
                   "水温": 18 + 2 * wave + i,
                   "pH": 7.2 + .1 * wave,
                   "溶解氧": 8 - .3 * wave + rng.normal(0, .03, periods),
                   "电导率": 300 + 10 * wave + i * 5,
                   "浊度": 10 + wave,
                   "CODMn": 3 + .2 * wave + rng.normal(0, .02, periods),
                   "氨氮": .3 + .03 * wave + rng.normal(0, .002, periods),
                   "总磷": .05 + .005 * wave + rng.normal(0, .0003, periods)}
        if station == "B":
            columns["流量"] = 15 + 2 * wave
        if station == "C":
            columns["总氮"] = 1.5 + .1 * wave
        frame = pd.DataFrame(columns)
        frame.loc[30:31, "溶解氧"] = np.nan
        frame.to_excel(folder / f"station_{station}_{label}.xlsx", index=False)
    hours = pd.date_range(times[0] - pd.Timedelta(hours=32), times[-1], freq="h")
    h = np.arange(len(hours))
    pd.DataFrame({"DATETIME": hours,
                  "PRE_1H": np.where(h % 59 < 3, .4, 0.),
                  "PRS": 1010 + np.sin(h / 30),
                  "RHU": 60 + 10 * np.sin(h / 24),
                  "TEM": 18 + 3 * np.sin(2 * np.pi * h / 24),
                  "WIN_D_INST_MAX": h * 7 % 360,
                  "WIN_S_INST_MAX": 2 + .5 * np.sin(h / 10)}).to_excel(
                      folder / "weather_hourly.xlsx", index=False)
    (folder / "SYNTHETIC_ONLY.txt").write_text(
        "ARTIFICIAL SOFTWARE TEST FIXTURE. Not observed river data. Not research evidence.\n",
        encoding="utf-8")
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--periods", type=int, default=420)
    args = parser.parse_args()
    print(generate(args.output, args.periods))
