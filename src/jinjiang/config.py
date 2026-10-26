from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = project_root() / path
    config = json.loads(path.read_text(encoding="utf-8"))
    config["_config_path"] = str(path)
    config["_project_root"] = str(project_root())
    return config


def resolve_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path
