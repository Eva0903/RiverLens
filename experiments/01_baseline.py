"""01 baseline: split the raw timeline first, then window each set separately."""
from __future__ import annotations

import argparse

from jinjiang.baseline_within_split import PROTOCOL
from jinjiang.config import load_config
from jinjiang.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline_within_split_gpu.json")
    parser.add_argument("--audit-only", action="store_true", help="Validate all sample boundaries without training")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model-override", choices=["ridge", "ann", "cnn", "rnn", "lstm", "transformer"])
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("baseline_protocol") != PROTOCOL:
        parser.error("Use configs/baseline_within_split_gpu.json or its smoke configuration; old configs are not this protocol.")
    split = config["split"]
    if not (0 < split["train_fraction"] < 1 and 0 < split["validation_fraction"]
            and split["train_fraction"] + split["validation_fraction"] < 1):
        parser.error("Invalid train/validation fractions.")
    if config["frequency"] != "4h":
        parser.error("This protocol uses 4h sampling and a 6-step seasonal naive.")
    if config["missing_data"]["interpolation_policy"] != "causal_ffill":
        parser.error("Use causal_ffill for within-split features.")
    if args.max_tasks is not None and args.max_tasks < 0:
        parser.error("--max-tasks must be nonnegative.")
    runner = ExperimentRunner(
        config, "01_baseline", overwrite=args.overwrite,
        model_override=args.model_override,
        max_tasks=0 if args.audit_only else args.max_tasks,
        continue_on_error=args.continue_on_error,
    )
    output = runner.run()
    print(f"{'Audit only (no models trained)' if args.audit_only else 'Finished'}: {output}")


if __name__ == "__main__":
    main()

