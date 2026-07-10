from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse

import pandas as pd
import yaml


CONFIG_PATH = "configs/lanl_pyg_data.yaml"


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
    required_sections = [
        "input",
        "output",
        "features",
        "validation",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    required_input_keys = [
        "directory",
        "graph_nodes_filename",
        "graph_edges_filename",
    ]

    for key in required_input_keys:
        if key not in config["input"]:
            raise ValueError(f"Missing input.{key} in config.")

    required_output_keys = [
        "directory",
        "graph_filename",
        "metadata_filename",
    ]

    for key in required_output_keys:
        if key not in config["output"]:
            raise ValueError(f"Missing output.{key} in config.")
    
    required_validation_keys = [
        "invalid_edge_policy",
        "require_compact_indices",
    ]

    for key in required_validation_keys:
        if key not in config["validation"]:
            raise ValueError(f"Missing validation.{key} in config.")


    valid_invalid_edge_policies = {"warn", "fail"}
    invalid_edge_policy = config["validation"]["invalid_edge_policy"]

    if invalid_edge_policy not in valid_invalid_edge_policies:
        raise ValueError(
            "validation.invalid_edge_policy must be one of: "
            f"{sorted(valid_invalid_edge_policies)}"
        )


def build_paths(config: dict[str, Any]) -> dict[str, Path]:
    input_directory = Path(config["input"]["directory"])
    output_directory = Path(config["output"]["directory"])

    return {
        "graph_nodes": input_directory / config["input"]["graph_nodes_filename"],
        "graph_edges": input_directory / config["input"]["graph_edges_filename"],
        "pyg_graph": output_directory / config["output"]["graph_filename"],
        "metadata": output_directory / config["output"]["metadata_filename"],
    }


def validate_input_paths(paths: dict[str, Path]) -> None:
    for key in ["graph_nodes", "graph_edges"]:
        path = paths[key]

        if not path.exists():
            raise FileNotFoundError(f"Required input not found for {key}: {path}")

        if not path.is_file():
            raise FileNotFoundError(f"Required input is not a file for {key}: {path}")


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


def read_graph_tables(
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_parquet(paths["graph_nodes"])
    edges = pd.read_parquet(paths["graph_edges"])

    required_node_columns = [
        "node_id",
        "entity_type",
        "entity_name",
        "node_key",
        "first_seen_timestamp",
        "last_seen_timestamp",
        "event_count",
        "label",
    ]

    required_edge_columns = [
        "edge_id",
        "source_node_id",
        "destination_node_id",
        "source_entity",
        "destination_entity",
        "source_entity_type",
        "destination_entity_type",
        "edge_type",
        "event_family",
        "event_type",
        "timestamp",
        "event_result",
        "label",
        "source_file",
        "row_number",
    ]

    validate_columns(
        dataframe=nodes,
        required_columns=required_node_columns,
        table_name="graph_nodes",
    )

    validate_columns(
        dataframe=edges,
        required_columns=required_edge_columns,
        table_name="graph_edges",
    )

    return nodes, edges


def print_graph_table_summary(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    print()
    print("Graph table summary")
    print(
        {
            "node_rows": len(nodes),
            "edge_rows": len(edges),
            "unique_node_ids": nodes["node_id"].nunique(),
            "unique_edge_ids": edges["edge_id"].nunique(),
            "node_columns": list(nodes.columns),
            "edge_columns": list(edges.columns),
        }
    )


def print_conversion_plan(
    config: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    print("LANL PyG data conversion plan")
    print("Inputs:")
    print(
        {
            "graph_nodes": str(paths["graph_nodes"]),
            "graph_edges": str(paths["graph_edges"]),
        }
    )

    print("Outputs:")
    print(
        {
            "pyg_graph": str(paths["pyg_graph"]),
            "metadata": str(paths["metadata"]),
        }
    )

    print("Features:")
    print(config["features"])

    print("Validation:")
    print(config["validation"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to LANL PyG data config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    validate_config(config)

    paths = build_paths(config)
    validate_input_paths(paths)

    paths["pyg_graph"].parent.mkdir(parents=True, exist_ok=True)

    print("Config path:", args.config)
    print_conversion_plan(
        config=config,
        paths=paths,
    )

    nodes, edges = read_graph_tables(paths)

    print_graph_table_summary(
        nodes=nodes,
        edges=edges,
    )


if __name__ == "__main__":
    main()