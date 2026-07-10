from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse

import pandas as pd
import torch
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

def prepare_nodes_for_pyg(
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    prepared_nodes = nodes.copy()

    prepared_nodes["node_id"] = prepared_nodes["node_id"].astype("int64")

    if not prepared_nodes["node_id"].is_unique:
        raise ValueError("node_id values must be unique before PyG conversion.")

    prepared_nodes = prepared_nodes.sort_values(
        by=["node_id"],
        ascending=True,
        kind="stable",
    ).reset_index(drop=True)

    prepared_nodes["pyg_node_index"] = range(len(prepared_nodes))

    return prepared_nodes


def node_id_equals_pyg_index(
    prepared_nodes: pd.DataFrame,
) -> bool:
    return bool(
        (
            prepared_nodes["node_id"].astype("int64")
            == prepared_nodes["pyg_node_index"].astype("int64")
        ).all()
    )


def validate_compact_pyg_indices(
    prepared_nodes: pd.DataFrame,
) -> None:
    expected_indices = list(range(len(prepared_nodes)))
    actual_indices = prepared_nodes["pyg_node_index"].astype("int64").tolist()

    if actual_indices != expected_indices:
        raise ValueError("pyg_node_index values must be compact and ordered.")


def build_node_id_to_pyg_index(
    prepared_nodes: pd.DataFrame,
) -> dict[int, int]:
    return {
        int(row["node_id"]): int(row["pyg_node_index"])
        for _, row in prepared_nodes.iterrows()
    }


def print_pyg_node_mapping_summary(
    prepared_nodes: pd.DataFrame,
    node_mapping: dict[int, int],
) -> None:
    print()
    identity_mapping = node_id_equals_pyg_index(prepared_nodes)

    if prepared_nodes.empty:
        print("PyG node mapping summary")
        print(
            {
                "node_count": 0,
                "mapping_entries": 0,
                "node_id_equals_pyg_index": identity_mapping,
            }
        )
        return

    print("PyG node mapping summary")
    print(
        {
            "node_count": len(prepared_nodes),
            "mapping_entries": len(node_mapping),
            "min_node_id": int(prepared_nodes["node_id"].min()),
            "max_node_id": int(prepared_nodes["node_id"].max()),
            "min_pyg_node_index": int(prepared_nodes["pyg_node_index"].min()),
            "max_pyg_node_index": int(prepared_nodes["pyg_node_index"].max()),
            "node_id_equals_pyg_index": identity_mapping,
        }
    )

def split_valid_and_invalid_edges(
    edges: pd.DataFrame,
    node_id_to_pyg_index: dict[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_node_ids = set(node_id_to_pyg_index.keys())

    source_is_valid = edges["source_node_id"].astype("int64").isin(valid_node_ids)
    destination_is_valid = edges["destination_node_id"].astype("int64").isin(
        valid_node_ids
    )

    valid_mask = source_is_valid & destination_is_valid

    valid_edges = edges[valid_mask].copy()
    invalid_edges = edges[~valid_mask].copy()

    return valid_edges, invalid_edges


def apply_invalid_edge_policy(
    invalid_edges: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    invalid_edge_count = len(invalid_edges)
    policy = config["validation"]["invalid_edge_policy"]

    if invalid_edge_count == 0:
        return

    if policy == "fail":
        raise ValueError(
            f"Found {invalid_edge_count} edges with missing node endpoints."
        )

    print(
        "Warning:",
        {
            "invalid_edge_count": invalid_edge_count,
            "policy": policy,
            "action": "invalid edges will be excluded from PyG conversion",
        },
    )


def build_edge_index(
    valid_edges: pd.DataFrame,
    node_id_to_pyg_index: dict[int, int],
) -> torch.Tensor:
    source_indices = valid_edges["source_node_id"].astype("int64").map(
        node_id_to_pyg_index
    )
    destination_indices = valid_edges["destination_node_id"].astype("int64").map(
        node_id_to_pyg_index
    )

    if source_indices.isna().any() or destination_indices.isna().any():
        raise ValueError("Valid edges contain endpoints missing from node mapping.")

    edge_index = torch.tensor(
        [
            source_indices.astype("int64").tolist(),
            destination_indices.astype("int64").tolist(),
        ],
        dtype=torch.long,
    )

    return edge_index


def validate_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> None:
    if edge_index.ndim != 2:
        raise ValueError("edge_index must be a rank-2 tensor.")

    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    if edge_index.numel() == 0:
        return

    min_index = int(edge_index.min().item())
    max_index = int(edge_index.max().item())

    if min_index < 0:
        raise ValueError("edge_index contains negative node indices.")

    if max_index >= num_nodes:
        raise ValueError(
            "edge_index contains node indices outside the valid compact range."
        )


def print_edge_index_summary(
    edge_index: torch.Tensor,
    valid_edges: pd.DataFrame,
    invalid_edges: pd.DataFrame,
) -> None:
    print("PyG edge_index summary")
    print(
        {
            "shape": list(edge_index.shape),
            "valid_edge_rows": len(valid_edges),
            "invalid_edge_rows": len(invalid_edges),
            "edge_index_edges": int(edge_index.shape[1]),
        }
    )


def print_conversion_plan(
    config: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    print()
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

    prepared_nodes = prepare_nodes_for_pyg(nodes)
    assert (prepared_nodes["node_id"]==prepared_nodes["pyg_node_index"]).all()

    validate_compact_pyg_indices(prepared_nodes)

    node_id_to_pyg_index: dict[int, int] = build_node_id_to_pyg_index(prepared_nodes)

    print_pyg_node_mapping_summary(
        prepared_nodes=prepared_nodes,
        node_mapping=node_id_to_pyg_index,
    )

    valid_edges, invalid_edges = split_valid_and_invalid_edges(
        edges=edges,
        node_id_to_pyg_index=node_id_to_pyg_index,
    )

    apply_invalid_edge_policy(
        invalid_edges=invalid_edges,
        config=config,
    )

    edge_index = build_edge_index(
        valid_edges=valid_edges,
        node_id_to_pyg_index=node_id_to_pyg_index,
    )

    validate_edge_index(
        edge_index=edge_index,
        num_nodes=len(prepared_nodes),
    )

    print_edge_index_summary(
        edge_index=edge_index,
        valid_edges=valid_edges,
        invalid_edges=invalid_edges,
    )


if __name__ == "__main__":
    main()