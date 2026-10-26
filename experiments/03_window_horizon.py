"""03 fixed-sample window x 4--72h horizon experiment."""
from __future__ import annotations

import argparse

from jinjiang.config import load_config
from jinjiang.window_horizon_random import RandomWindowRunner, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/window_horizon_random_main_gpu.json")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    if args.max_tasks is not None and args.max_tasks < 0:
        parser.error("--max-tasks must be nonnegative.")
    config = load_config(args.config)
    try:
        validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    runner = RandomWindowRunner(config, max_tasks=0 if args.audit_only else args.max_tasks)
    print(f"Finished: {runner.run()}")


if __name__ == "__main__":
    main()
