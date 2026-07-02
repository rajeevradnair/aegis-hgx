from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse

import yaml

from aegis_hgx.models.baselines.config_schema import (
    TrainingConfig,
    validate_training_config,
)


CONFIG_PATH = "configs/xgboost_cic.yaml"


def load_training_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Training config must contain a YAML mapping.")

    return config


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to experiment config YAML file.",
    )
    return arg_parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_config = load_training_config(args.config)
    config = validate_training_config(raw_config)

    print("Config path:", args.config)
    print("Input path:", config.data.input_path)
    print("Target column:", config.data.target_column)
    print("Metrics path:", config.outputs.metrics_path)
    print("Model path:", config.outputs.model_path)
    print("Experiment:", config.experiment_tracking.experiment_name)


if __name__ == "__main__":
    main()