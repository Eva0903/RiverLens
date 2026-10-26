"""Run seven cap or seven patience diagnostic tasks against completed 1024 results."""
import argparse
from copy import deepcopy

from jinjiang.config import load_config
from jinjiang.window_training_diagnostics import TrainingDiagnosticRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One listed task, two epochs, separate _smoke output.")
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    if args.max_tasks is not None and args.max_tasks < 0:
        parser.error("--max-tasks must be nonnegative.")
    config = deepcopy(load_config(args.config))
    if args.smoke:
        config["smoke_test"] = True
        config["training"]["max_epochs"] = 2
        config["results_dir"] += "_smoke"
    runner = TrainingDiagnosticRunner(config, max_tasks=0 if args.audit_only else args.max_tasks)
    print(f"Finished: {runner.run()}")


if __name__ == "__main__":
    main()
