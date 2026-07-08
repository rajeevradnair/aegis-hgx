from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import pandas as pd
import yaml


CONFIG_PATH = "configs/lanl_ingest.yaml"
CANONICAL_COLUMNS = {
    "auth": [
        "timestamp",
        "source_user",
        "destination_user",
        "source_host",
        "destination_host",
        "auth_type",
        "logon_type",
        "auth_orientation",
        "result",
    ],
    "dns": [
        "timestamp",
        "source_host",
        "resolved_host",
    ],
    "flows": [
        "timestamp",
        "duration",
        "source_host",
        "source_port",
        "destination_host",
        "destination_port",
        "protocol",
        "packet_count",
        "byte_count",
    ],
    "proc": [
        "timestamp",
        "source_user",
        "host",
        "process_name",
        "process_event",
    ],
    "redteam": [
        "timestamp",
        "source_user",
        "source_host",
        "destination_host",
    ],
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")

    return config


def validate_config(config: dict[str, Any]) -> None:
    required_top_level_keys = ["input", "files", "output", "schema"]

    for key in required_top_level_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")

    input_directory = Path(config["input"]["directory"])

    if not input_directory.exists():
        raise FileNotFoundError(f"Input directory not found: {input_directory}")

    if not input_directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_directory}")

    if not isinstance(config["files"], dict):
        raise ValueError("Config section 'files' must be a mapping.")

    for file_key, file_config in config["files"].items():
        if "filename" not in file_config:
            raise ValueError(f"Missing filename for file config: {file_key}")

        if "output_filename" not in file_config:
            raise ValueError(f"Missing output_filename for file config: {file_key}")

        if "event_family" not in file_config:
            raise ValueError(f"Missing event_family for file config: {file_key}")


def build_ingest_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    input_directory = Path(config["input"]["directory"])
    output_directory = Path(config["output"]["directory"])

    ingest_plan = []

    for file_key, file_config in config["files"].items():
        input_path = input_directory / file_config["filename"]
        output_path = output_directory / file_config["output_filename"]

        ingest_plan.append(
            {
                "file_key": file_key,
                "event_family": file_config["event_family"],
                "input_path": input_path,
                "output_path": output_path,
            }
        )

    return ingest_plan


def validate_input_files(ingest_plan: list[dict[str, Any]]) -> None:
    for item in ingest_plan:
        input_path = item["input_path"]

        if not input_path.exists():
            raise FileNotFoundError(f"Configured LANL file not found: {input_path}")

        if not input_path.is_file():
            raise FileNotFoundError(f"Configured LANL path is not a file: {input_path}")


def get_canonical_columns(file_key: str) -> list[str]:
    if file_key not in CANONICAL_COLUMNS:
        raise ValueError(f"No canonical schema registered for file key: {file_key}")

    return CANONICAL_COLUMNS[file_key]


def read_raw_preview(
    input_path: Path,
    file_key: str,
    preview_rows: int = 10,
) -> pd.DataFrame:
    
    canonical_columns = get_canonical_columns(file_key)

    preview = pd.read_csv(
        input_path,
        compression="infer",
        header=None,
        nrows=preview_rows,
        low_memory=False,
    )

    if len(preview.columns) != len(canonical_columns):
        raise ValueError(
            "Unexpected column count for "
            f"{input_path}. "
            f"Expected {len(canonical_columns)}, "
            f"found {len(preview.columns)}."
        )

    preview.columns = canonical_columns

    return preview


def inspect_raw_files(ingest_plan: list[dict[str, Any]]) -> None:
    print("Raw LANL file inspection")

    for item in ingest_plan:
        preview = read_raw_preview(
            input_path=item["input_path"],
            file_key=item["file_key"],
        )

        print(
            {
                "file_key": item["file_key"],
                "event_family": item["event_family"],
                "input_path": str(item["input_path"]),
                "preview_rows": len(preview),
                "preview_columns": len(preview.columns),
                "columns": list(preview.columns),
            }
        )

        print(preview.head())


def print_ingest_plan(ingest_plan: list[dict[str, Any]]) -> None:
    print("LANL ingestion plan")

    for item in ingest_plan:
        print(
            {
                "file_key": item["file_key"],
                "event_family": item["event_family"],
                "input_path": str(item["input_path"]),
                "output_path": str(item["output_path"]),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to LANL ingestion config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    validate_config(config)

    ingest_plan = build_ingest_plan(config)
    validate_input_files(ingest_plan)

    output_directory = Path(config["output"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    print("Config path:", args.config)
    print("Input directory:", config["input"]["directory"])
    print("Output directory:", output_directory)
    print("Configured file count:", len(ingest_plan))

    print_ingest_plan(ingest_plan)
    
    inspect_raw_files(ingest_plan)


if __name__ == "__main__":
    main()