from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jinjiang.config import load_config
from jinjiang.do_ablation import run_do_ablation


def main():
    parser = argparse.ArgumentParser(description="Fixed DO C-only versus A/B/C common-cohort ablation")
    parser.add_argument("--config", default="configs/do_upstream_ablation.json")
    parser.add_argument("--audit-only", action="store_true", help="Write shared sample audit without training")
    parser.add_argument("--seeds", type=int, nargs="+", help="Run a subset, then resume with the same config")
    args = parser.parse_args()
    output = run_do_ablation(load_config(args.config), audit_only=args.audit_only, seeds=args.seeds)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
