"""Eight-model random-window holdout baseline; chronological 01 remains available."""
from __future__ import annotations

import argparse

from jinjiang.baseline_random import RandomBaselineRunner, PROTOCOL
from jinjiang.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline_random_gpu.json")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--model-override", choices=["ridge", "ann", "cnn", "rnn", "lstm", "transformer"])
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    c = load_config(args.config)
    if c.get("baseline_protocol") != PROTOCOL:
        parser.error("Use the random-window configuration.")
    if c["frequency"] != "4h" or c["missing_data"]["interpolation_policy"] != "causal_ffill":
        parser.error("Use 4h frequency and causal_ffill preprocessing.")
    if (c["split"]["train_fraction"], c["split"]["validation_fraction"]) != (.8, .1):
        parser.error("This protocol fixes the old script's 80/10/10 split.")
    if args.max_tasks is not None and args.max_tasks < 0:
        parser.error("--max-tasks must be nonnegative.")
    runner = RandomBaselineRunner(c, model_override=args.model_override,
                                  max_tasks=0 if args.audit_only else args.max_tasks,
                                  continue_on_error=args.continue_on_error)
    print(f"Finished: {runner.run()}")


if __name__ == "__main__":
    main()
