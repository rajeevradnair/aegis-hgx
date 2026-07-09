from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse

import yaml
from datetime import datetime, timezone
import json

import pandas as pd

CONFIG_PATH = "configs/lanl_event_tables.yaml"


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
    required_sections = ["input", "output", "schema"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    if "directory" not in config["input"]:
        raise ValueError("Missing input.directory in config.")

    if "files" not in config["input"]:
        raise ValueError("Missing input.files in config.")

    if "directory" not in config["output"]:
        raise ValueError("Missing output.directory in config.")

    if "files" not in config["output"]:
        raise ValueError("Missing output.files in config.")

    if "manifest_filename" not in config["output"]:
        raise ValueError("Missing output.manifest_filename in config.")

    if not isinstance(config["input"]["files"], dict):
        raise ValueError("input.files must be a mapping.")

    if not isinstance(config["output"]["files"], dict):
        raise ValueError("output.files must be a mapping.")


def build_input_paths(config: dict[str, Any]) -> dict[str, Path]:
    input_directory = Path(config["input"]["directory"])

    return {
        file_key: input_directory / filename
        for file_key, filename in config["input"]["files"].items()
    }


def build_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    output_directory = Path(config["output"]["directory"])

    return {
        file_key: output_directory / filename
        for file_key, filename in config["output"]["files"].items()
    }


def validate_input_paths(input_paths: dict[str, Path]) -> None:
    for file_key, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Input parquet file not found for {file_key}: {path}"
            )

        if not path.is_file():
            raise FileNotFoundError(
                f"Input parquet path is not a file for {file_key}: {path}"
            )


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{table_name} is missing required columns: {missing_columns}"
        )

def normalize_string_series(
    series: pd.Series,
    unknown_value: str,
) -> pd.Series:
    return (
        series.fillna(unknown_value)
        .astype(str)
        .str.strip()
        .replace("", unknown_value)
    )


def build_event_ids(
    prefix: str,
    event_family: str,
    row_numbers: pd.Series,
) -> pd.Series:
    return (
        prefix
        + "_"
        + event_family
        + "_"
        + row_numbers.astype(str)
    )

def normalize_auth_events(
    auth_dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    required_columns = [
        "row_number",
        "timestamp",
        "source_user",
        "destination_user",
        "source_host",
        "destination_host",
        "auth_type",
        "logon_type",
        "auth_orientation",
        "result",
        "source_file",
        "event_family",
    ]

    validate_columns(
        dataframe=auth_dataframe,
        required_columns=required_columns,
        table_name="auth",
    )

    unknown_value = config["schema"]["unknown_value"]
    default_label = config["schema"]["default_label"]
    event_id_prefix = config["schema"]["event_id_prefix"]

    clean = pd.DataFrame()

    clean["event_id"] = build_event_ids(
        prefix=event_id_prefix,
        event_family="authentication",
        row_numbers=auth_dataframe["row_number"],
    )
    clean["timestamp"] = pd.to_numeric(
        auth_dataframe["timestamp"],
        errors="coerce",
    ).fillna(-1).astype("int64")
    clean["event_family"] = "authentication"
    clean["event_type"] = (
        "auth_"
        + normalize_string_series(
            auth_dataframe["auth_orientation"],
            unknown_value,
        ).str.lower()
    )

    clean["source_entity"] = normalize_string_series(
        auth_dataframe["source_user"],
        unknown_value,
    )
    clean["destination_entity"] = normalize_string_series(
        auth_dataframe["destination_host"],
        unknown_value,
    )
    clean["source_entity_type"] = "user"
    clean["destination_entity_type"] = "host"

    clean["event_result"] = normalize_string_series(
        auth_dataframe["result"],
        unknown_value,
    ).str.lower()
    clean["label"] = default_label

    clean["source_user"] = normalize_string_series(
        auth_dataframe["source_user"],
        unknown_value,
    )
    clean["destination_user"] = normalize_string_series(
        auth_dataframe["destination_user"],
        unknown_value,
    )
    clean["source_host"] = normalize_string_series(
        auth_dataframe["source_host"],
        unknown_value,
    )
    clean["destination_host"] = normalize_string_series(
        auth_dataframe["destination_host"],
        unknown_value,
    )

    clean["source_file"] = normalize_string_series(
        auth_dataframe["source_file"],
        unknown_value,
    )
    clean["row_number"] = auth_dataframe["row_number"].astype("int64")

    return clean


def normalize_dns_events(
    dns_dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    required_columns = [
        "row_number",
        "timestamp",
        "source_host",
        "resolved_host",
        "source_file",
        "event_family",
    ]

    validate_columns(
        dataframe=dns_dataframe,
        required_columns=required_columns,
        table_name="dns",
    )

    unknown_value = config["schema"]["unknown_value"]
    default_label = config["schema"]["default_label"]
    event_id_prefix = config["schema"]["event_id_prefix"]

    clean = pd.DataFrame()

    clean["event_id"] = build_event_ids(
        prefix=event_id_prefix,
        event_family="dns",
        row_numbers=dns_dataframe["row_number"],
    )
    clean["timestamp"] = pd.to_numeric(
        dns_dataframe["timestamp"],
        errors="coerce",
    ).fillna(-1).astype("int64")
    clean["event_family"] = "dns"
    clean["event_type"] = "dns_resolution"

    clean["source_entity"] = normalize_string_series(
        dns_dataframe["source_host"],
        unknown_value,
    )
    clean["destination_entity"] = normalize_string_series(
        dns_dataframe["resolved_host"],
        unknown_value,
    )
    clean["source_entity_type"] = "host"
    clean["destination_entity_type"] = "host_or_domain"

    clean["event_result"] = unknown_value
    clean["label"] = default_label

    clean["source_host"] = normalize_string_series(
        dns_dataframe["source_host"],
        unknown_value,
    )
    clean["resolved_host"] = normalize_string_series(
        dns_dataframe["resolved_host"],
        unknown_value,
    )

    clean["source_file"] = normalize_string_series(
        dns_dataframe["source_file"],
        unknown_value,
    )
    clean["row_number"] = dns_dataframe["row_number"].astype("int64")

    return clean


def normalize_flow_events(
    flows_dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    required_columns = [
        "row_number",
        "timestamp",
        "duration",
        "source_host",
        "source_port",
        "destination_host",
        "destination_port",
        "protocol",
        "packet_count",
        "byte_count",
        "source_file",
        "event_family",
    ]

    validate_columns(
        dataframe=flows_dataframe,
        required_columns=required_columns,
        table_name="flows",
    )

    unknown_value = config["schema"]["unknown_value"]
    default_label = config["schema"]["default_label"]
    event_id_prefix = config["schema"]["event_id_prefix"]

    clean = pd.DataFrame()

    clean["event_id"] = build_event_ids(
        prefix=event_id_prefix,
        event_family="network_flow",
        row_numbers=flows_dataframe["row_number"],
    )
    clean["timestamp"] = pd.to_numeric(
        flows_dataframe["timestamp"],
        errors="coerce",
    ).fillna(-1).astype("int64")
    clean["event_family"] = "network_flow"
    clean["event_type"] = "network_flow"

    clean["source_entity"] = normalize_string_series(
        flows_dataframe["source_host"],
        unknown_value,
    )
    clean["destination_entity"] = normalize_string_series(
        flows_dataframe["destination_host"],
        unknown_value,
    )
    clean["source_entity_type"] = "host"
    clean["destination_entity_type"] = "host"

    clean["event_result"] = "observed"
    clean["label"] = default_label

    clean["source_host"] = normalize_string_series(
        flows_dataframe["source_host"],
        unknown_value,
    )
    clean["destination_host"] = normalize_string_series(
        flows_dataframe["destination_host"],
        unknown_value,
    )
    clean["source_port"] = normalize_string_series(
        flows_dataframe["source_port"],
        unknown_value,
    )
    clean["destination_port"] = normalize_string_series(
        flows_dataframe["destination_port"],
        unknown_value,
    )
    clean["protocol"] = normalize_string_series(
        flows_dataframe["protocol"],
        unknown_value,
    )

    clean["duration"] = pd.to_numeric(
        flows_dataframe["duration"],
        errors="coerce",
    ).fillna(0)
    clean["packet_count"] = pd.to_numeric(
        flows_dataframe["packet_count"],
        errors="coerce",
    ).fillna(0)
    clean["byte_count"] = pd.to_numeric(
        flows_dataframe["byte_count"],
        errors="coerce",
    ).fillna(0)

    clean["source_file"] = normalize_string_series(
        flows_dataframe["source_file"],
        unknown_value,
    )
    clean["row_number"] = flows_dataframe["row_number"].astype("int64")

    return clean


def normalize_process_events(
    proc_dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    required_columns = [
        "row_number",
        "timestamp",
        "source_user",
        "host",
        "process_name",
        "process_event",
        "source_file",
        "event_family",
    ]

    validate_columns(
        dataframe=proc_dataframe,
        required_columns=required_columns,
        table_name="proc",
    )

    unknown_value = config["schema"]["unknown_value"]
    default_label = config["schema"]["default_label"]
    event_id_prefix = config["schema"]["event_id_prefix"]

    clean = pd.DataFrame()

    clean["event_id"] = build_event_ids(
        prefix=event_id_prefix,
        event_family="process",
        row_numbers=proc_dataframe["row_number"],
    )
    clean["timestamp"] = pd.to_numeric(
        proc_dataframe["timestamp"],
        errors="coerce",
    ).fillna(-1).astype("int64")
    clean["event_family"] = "process"
    clean["event_type"] = (
        "process_"
        + normalize_string_series(
            proc_dataframe["process_event"],
            unknown_value,
        ).str.lower()
    )

    clean["source_entity"] = normalize_string_series(
        proc_dataframe["source_user"],
        unknown_value,
    )
    clean["destination_entity"] = normalize_string_series(
        proc_dataframe["process_name"],
        unknown_value,
    )
    clean["source_entity_type"] = "user"
    clean["destination_entity_type"] = "process"

    clean["event_result"] = normalize_string_series(
        proc_dataframe["process_event"],
        unknown_value,
    ).str.lower()
    clean["label"] = default_label

    clean["source_user"] = normalize_string_series(
        proc_dataframe["source_user"],
        unknown_value,
    )
    clean["host"] = normalize_string_series(
        proc_dataframe["host"],
        unknown_value,
    )
    clean["process_name"] = normalize_string_series(
        proc_dataframe["process_name"],
        unknown_value,
    )
    clean["process_event"] = normalize_string_series(
        proc_dataframe["process_event"],
        unknown_value,
    )

    clean["source_file"] = normalize_string_series(
        proc_dataframe["source_file"],
        unknown_value,
    )
    clean["row_number"] = proc_dataframe["row_number"].astype("int64")

    return clean


def normalize_redteam_events(
    redteam_dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    required_columns = [
        "row_number",
        "timestamp",
        "source_user",
        "source_host",
        "destination_host",
        "source_file",
        "event_family",
    ]

    validate_columns(
        dataframe=redteam_dataframe,
        required_columns=required_columns,
        table_name="redteam",
    )

    unknown_value = config["schema"]["unknown_value"]
    event_id_prefix = config["schema"]["event_id_prefix"]

    clean = pd.DataFrame()

    clean["event_id"] = build_event_ids(
        prefix=event_id_prefix,
        event_family="redteam",
        row_numbers=redteam_dataframe["row_number"],
    )
    clean["timestamp"] = pd.to_numeric(
        redteam_dataframe["timestamp"],
        errors="coerce",
    ).fillna(-1).astype("int64")
    clean["event_family"] = "redteam_ground_truth"
    clean["event_type"] = "redteam_activity"

    clean["source_entity"] = normalize_string_series(
        redteam_dataframe["source_user"],
        unknown_value,
    )
    clean["destination_entity"] = normalize_string_series(
        redteam_dataframe["destination_host"],
        unknown_value,
    )
    clean["source_entity_type"] = "user"
    clean["destination_entity_type"] = "host"

    clean["event_result"] = "confirmed_redteam"
    clean["label"] = 1

    clean["source_user"] = normalize_string_series(
        redteam_dataframe["source_user"],
        unknown_value,
    )
    clean["source_host"] = normalize_string_series(
        redteam_dataframe["source_host"],
        unknown_value,
    )
    clean["destination_host"] = normalize_string_series(
        redteam_dataframe["destination_host"],
        unknown_value,
    )

    clean["source_file"] = normalize_string_series(
        redteam_dataframe["source_file"],
        unknown_value,
    )
    clean["row_number"] = redteam_dataframe["row_number"].astype("int64")

    return clean


def build_dns_event_table(
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    dns_dataframe = read_parquet_file(input_paths["dns"])

    clean_dns = normalize_dns_events(
        dns_dataframe=dns_dataframe,
        config=config,
    )

    write_parquet_file(
        dataframe=clean_dns,
        output_path=output_paths["clean_dns"],
    )

    return {
        "table": "clean_dns",
        "input_path": str(input_paths["dns"]),
        "output_path": str(output_paths["clean_dns"]),
        "rows": int(len(clean_dns)),
        "columns": list(clean_dns.columns),
    }


def build_flow_event_table(
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    flows_dataframe = read_parquet_file(input_paths["flows"])

    clean_flows = normalize_flow_events(
        flows_dataframe=flows_dataframe,
        config=config,
    )

    write_parquet_file(
        dataframe=clean_flows,
        output_path=output_paths["clean_flows"],
    )

    return {
        "table": "clean_flows",
        "input_path": str(input_paths["flows"]),
        "output_path": str(output_paths["clean_flows"]),
        "rows": int(len(clean_flows)),
        "columns": list(clean_flows.columns),
    }


def build_process_event_table(
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    proc_dataframe = read_parquet_file(input_paths["proc"])

    clean_process = normalize_process_events(
        proc_dataframe=proc_dataframe,
        config=config,
    )

    write_parquet_file(
        dataframe=clean_process,
        output_path=output_paths["clean_process"],
    )

    return {
        "table": "clean_process",
        "input_path": str(input_paths["proc"]),
        "output_path": str(output_paths["clean_process"]),
        "rows": int(len(clean_process)),
        "columns": list(clean_process.columns),
    }


def build_redteam_event_table(
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    redteam_dataframe = read_parquet_file(input_paths["redteam"])

    clean_redteam = normalize_redteam_events(
        redteam_dataframe=redteam_dataframe,
        config=config,
    )

    write_parquet_file(
        dataframe=clean_redteam,
        output_path=output_paths["clean_redteam"],
    )

    return {
        "table": "clean_redteam",
        "input_path": str(input_paths["redteam"]),
        "output_path": str(output_paths["clean_redteam"]),
        "rows": int(len(clean_redteam)),
        "columns": list(clean_redteam.columns),
    }


def read_parquet_file(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet_file(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)


def build_auth_event_table(
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    auth_dataframe = read_parquet_file(input_paths["auth"])

    clean_auth = normalize_auth_events(
        auth_dataframe=auth_dataframe,
        config=config,
    )

    write_parquet_file(
        dataframe=clean_auth,
        output_path=output_paths["clean_auth"],
    )

    return {
        "table": "clean_auth",
        "input_path": str(input_paths["auth"]),
        "output_path": str(output_paths["clean_auth"]),
        "rows": int(len(clean_auth)),
        "columns": list(clean_auth.columns),
    }


def print_event_table_plan(
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
) -> None:
    print("LANL event table build plan")

    print("Input files:")
    for file_key, path in input_paths.items():
        print(
            {
                "file_key": file_key,
                "input_path": str(path),
            }
        )

    print("Output files:")
    for file_key, path in output_paths.items():
        print(
            {
                "file_key": file_key,
                "output_path": str(path),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to LANL event table config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    validate_config(config)

    input_paths = build_input_paths(config)
    output_paths = build_output_paths(config)

    validate_input_paths(input_paths)

    output_directory = Path(config["output"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    print("Config path:", args.config)
    print("Input directory:", config["input"]["directory"])
    print("Output directory:", output_directory)
    print("Input file count:", len(input_paths))
    print("Output file count:", len(output_paths))

    print_event_table_plan(
        input_paths=input_paths,
        output_paths=output_paths,
    )

    table_results = [
        build_auth_event_table(
            input_paths=input_paths,
            output_paths=output_paths,
            config=config,
        ),
        build_dns_event_table(
            input_paths=input_paths,
            output_paths=output_paths,
            config=config,
        ),
        build_flow_event_table(
            input_paths=input_paths,
            output_paths=output_paths,
            config=config,
        ),
        build_process_event_table(
            input_paths=input_paths,
            output_paths=output_paths,
            config=config,
        ),
        build_redteam_event_table(
            input_paths=input_paths,
            output_paths=output_paths,
            config=config,
        ),
    ]

    for result in table_results:
        print("Built event table:", result)


if __name__ == "__main__":
    main()