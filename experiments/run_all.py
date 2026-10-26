from __future__ import annotations

import argparse

from jinjiang.config import load_config
from jinjiang.runner import EXPERIMENTS, ExperimentRunner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all five experiments in order")
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model-override", choices=["ridge", "ann", "cnn", "rnn", "lstm", "transformer"])
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for experiment in sorted(EXPERIMENTS):
        runner = ExperimentRunner(
            config,
            experiment,
            overwrite=args.overwrite,
            model_override=args.model_override,
            continue_on_error=args.continue_on_error,
        )
        print(f"Finished: {runner.run()}")
