from __future__ import annotations

import argparse

from .config import load_config
from .runner import EXPERIMENTS, ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the five Jinjiang supplementary experiments")
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS) + ["all"], required=True)
    parser.add_argument("--overwrite", action="store_true", help="Delete existing outputs for the selected experiment")
    parser.add_argument("--model-override", choices=["ridge", "ann", "cnn", "rnn", "lstm", "transformer"])
    parser.add_argument("--max-tasks", type=int, help="Stop after N newly completed model/seed tasks")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiments = sorted(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for experiment in experiments:
        runner = ExperimentRunner(
            config,
            experiment,
            overwrite=args.overwrite,
            model_override=args.model_override,
            max_tasks=args.max_tasks,
            continue_on_error=args.continue_on_error,
        )
        output = runner.run()
        print(f"Finished {experiment}: {output}")


if __name__ == "__main__":
    main()
