"""Precision-correct inference; by default use existing checkpoints only."""
from __future__ import annotations
import argparse
from jinjiang.config import load_config
from jinjiang.interpretation_precision_fix import fixed_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/interpretation_random.json")
    parser.add_argument("--mode", choices=["audit", "overview", "explain"], default="explain")
    parser.add_argument("--replay-missing", action="store_true")
    parser.add_argument("--max-replays", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    if args.max_replays < 0 or (args.max_cases is not None and args.max_cases < 0):
        parser.error("Task limits must be nonnegative.")
    if args.replay_missing and args.mode != "explain":
        parser.error("Recovery requires --mode explain.")
    from jinjiang import interpretation_random as core
    spec = load_config(args.config)
    core.validate_spec(spec)
    with fixed_inference(core):
        runner = core.InterpretationRunner(spec)
        print(f"Finished: {runner.run(args.mode, args.replay_missing, args.max_replays, args.max_cases)}")


if __name__ == "__main__":
    main()
