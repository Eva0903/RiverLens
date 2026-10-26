"""Select one current 01--07 experiment without mixing historical entry points."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    "01": ("01_baseline_random.py", "baseline_random_gpu.json", "baseline_random_smoke_gpu.json"),
    "02": ("02_station_ablation.py", "station_ablation_random_gpu.json", "station_ablation_random_smoke_gpu.json"),
    "03": ("03_window_horizon.py", "window_horizon_random_main_gpu.json", "window_horizon_random_smoke_gpu.json"),
    "04": ("04_meteorology_random.py", "meteorology_random_gpu.json", "meteorology_random_smoke_gpu.json"),
    "05": ("05_station_rotation_random.py", "station_rotation_random_gpu.json", "station_rotation_random_smoke_gpu.json"),
    "06": ("06_interpretation_precision_fix.py", "interpretation_random.json", None),
    "07": ("07_shap_kernel.py", "shap_kernel.json", None),
}


def command(experiment, mode, config=None, max_replays=1):
    script, full, smoke = STAGES[experiment]
    if experiment == "06":
        if mode == "smoke":
            raise ValueError("06 has no smoke mode. Use audit, overview or run with existing checkpoints.")
        flags = ["--mode", {"run": "explain", "recover": "explain"}.get(mode, mode)]
        if mode == "recover":
            flags += ["--replay-missing", "--max-replays", str(max_replays)]
    elif experiment == "07":
        if mode not in {"audit", "smoke", "run"}:
            raise ValueError("07 supports audit, smoke and run only; it never trains models.")
        flags = ["--mode", mode]
    else:
        if mode not in {"audit", "smoke", "run"}:
            raise ValueError("01--05 support audit, smoke and run only.")
        flags = ["--audit-only"] if mode == "audit" else []
    selected = config or "configs/" + (smoke if mode == "smoke" and smoke else full)
    return [sys.executable, str(ROOT / "experiments" / script), "--config", str(selected), *flags]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=STAGES)
    parser.add_argument("--mode", default="audit", choices=["audit", "smoke", "run", "overview", "recover"])
    parser.add_argument("--config")
    parser.add_argument("--max-replays", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print the selected command without importing dependencies or accessing data.")
    args = parser.parse_args()
    if args.max_replays < 1:
        parser.error("--max-replays must be positive")
    try:
        cmd = command(args.experiment, args.mode, args.config, args.max_replays)
    except ValueError as exc:
        parser.error(str(exc))
    print(subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd), flush=True)
    if args.mode == "recover":
        print("Explicit checkpoint recovery: this mode MAY TRAIN models.", flush=True)
    if args.dry_run:
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    raise SystemExit(subprocess.run(cmd, cwd=ROOT, env=env).returncode)


if __name__ == "__main__":
    main()
