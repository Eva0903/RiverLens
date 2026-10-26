"""Target-station rotation: 80 new A/B LSTM fits, with verified C reuse."""
from __future__ import annotations

import argparse

from jinjiang.config import load_config
from jinjiang.station_rotation_random import RandomStationRotationRunner, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/station_rotation_random_gpu.json")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.max_tasks is not None and args.max_tasks < 0:
        parser.error("--max-tasks must be nonnegative.")
    config = load_config(args.config)
    try:
        validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    runner = RandomStationRotationRunner(
        config, max_tasks=0 if args.audit_only else args.max_tasks,
        continue_on_error=args.continue_on_error,
    )
    print(f"Finished: {runner.run()}")


if __name__ == "__main__":
    main()
