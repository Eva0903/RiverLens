from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from jinjiang.config import load_config
from jinjiang.lstm_upgrade import run_lstm_upgrade


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-selected gated residual LSTM experiment"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = run_lstm_upgrade(load_config(args.config), overwrite=args.overwrite)
    print(f"Finished: {output}")


if __name__ == "__main__":
    main()
