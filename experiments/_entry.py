from __future__ import annotations

import argparse

from jinjiang.config import load_config
from jinjiang.runner import ExperimentRunner


def run(experiment: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run {experiment}")
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model-override", choices=["ridge", "ann", "cnn", "rnn", "lstm", "transformer"])
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    runner = ExperimentRunner(
        load_config(args.config),
        experiment,
        overwrite=args.overwrite,
        model_override=args.model_override,
        max_tasks=args.max_tasks,
        continue_on_error=args.continue_on_error,
    )
    print(f"Finished: {runner.run()}")
