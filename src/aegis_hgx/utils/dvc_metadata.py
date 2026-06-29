from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_dvc_file(path: str | Path) -> dict[str, Any]:
    dvc_path = Path(path)

    if not dvc_path.exists():
        raise FileNotFoundError(f"Cannot read missing DVC file: {dvc_path}")

    with dvc_path.open("r", encoding="utf-8") as file:
        metadata = yaml.safe_load(file)

    if not isinstance(metadata, dict):
        raise ValueError(f"DVC file must contain a mapping: {dvc_path}")

    return metadata


def get_first_dvc_output(path: str | Path) -> dict[str, Any]:
    metadata = read_dvc_file(path)
    outputs = metadata.get("outs")

    if not isinstance(outputs, list) or not outputs:
        raise ValueError(f"DVC file does not contain outputs: {path}")

    output = outputs[0]

    if not isinstance(output, dict):
        raise ValueError(f"DVC output must be a mapping: {path}")

    return output