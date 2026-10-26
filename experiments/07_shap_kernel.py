"""Kernel SHAP of existing model values only; no input gradients or training."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["audit", "smoke", "run"], default="audit")
    p.add_argument("--config", default="configs/shap_kernel.json")
    p.add_argument("--max-new-origins", type=int)
    args = p.parse_args()
    if args.max_new_origins is not None and args.max_new_origins < 1:
        p.error("--max-new-origins must be positive.")
    root = Path(__file__).resolve().parents[1]
    config = Path(args.config)
    if not config.is_absolute():
        config = root / config
    spec = json.loads(config.read_text(encoding="utf-8-sig"))
    from jinjiang.shap_kernel_supplement import run
    print("Original eval predictor only. Kernel SHAP: no backward, no training, no recovery.", flush=True)
    print(f"Finished: {run(spec, root, args.mode, args.max_new_origins)}")


if __name__ == "__main__":
    main()
