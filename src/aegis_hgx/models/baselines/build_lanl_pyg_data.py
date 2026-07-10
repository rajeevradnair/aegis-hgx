from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json

import pandas as pd
import torch
import yaml
from torch_geometric.data import Data


CONFIG_PATH = "configs/lanl_pyg_data.yaml"


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            make_json_safe(item)
            for item in value
        ]

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        return value.item()

    return value


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
    print()
    print("PyG edge_index summary")
    print(
        {
            "shape": list(edge_index.shape),
            "valid_edge_rows": len(valid_edges),
            "invalid_edge_rows": len(invalid_edges),
            "edge_index_edges": int(edge_index.shape[1]),
        }
    )


def add_node_derived_features(
    prepared_nodes: pd.DataFrame,
) -> pd.DataFrame:
    nodes_with_features = prepared_nodes.copy()

    nodes_with_features["active_span"] = (
        nodes_with_features["last_seen_timestamp"].astype("float64")
        - nodes_with_features["first_seen_timestamp"].astype("float64")
    )

    nodes_with_features["active_span"] = nodes_with_features[
        "active_span"
    ].clip(lower=0.0)

    return nodes_with_features


def min_max_scale_series(
    series: pd.Series,
) -> pd.Series:
    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0).astype("float64")

    minimum = numeric_series.min()
    maximum = numeric_series.max()

    if maximum == minimum:
        return pd.Series(
            0.0,
            index=numeric_series.index,
        )

    return (numeric_series - minimum) / (maximum - minimum)


def build_node_feature_tensor(
    prepared_nodes: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    nodes_with_features = add_node_derived_features(prepared_nodes)

    numeric_feature_names = list(config["features"]["node_numeric"])
    categorical_feature_names = list(config["features"]["node_categorical"])

    numeric_features: list[pd.Series] = []

    for feature_name in numeric_feature_names:
        if feature_name not in nodes_with_features.columns:
            raise ValueError(f"Missing node numeric feature: {feature_name}")

        scaled_feature = min_max_scale_series(nodes_with_features[feature_name])
        numeric_features.append(scaled_feature.rename(feature_name))

    numeric_frame = pd.concat(
        numeric_features,
        axis=1,
    )

    categorical_frames: list[pd.DataFrame] = []
    categorical_metadata: dict[str, list[str]] = {}

    for feature_name in categorical_feature_names:
        if feature_name not in nodes_with_features.columns:
            raise ValueError(f"Missing node categorical feature: {feature_name}")

        one_hot = pd.get_dummies(
            nodes_with_features[feature_name].astype("string").fillna("unknown"),
            prefix=feature_name,
            dtype="float32",
        )

        categorical_metadata[feature_name] = list(one_hot.columns)
        categorical_frames.append(one_hot)

    #Create the full dataframe for x
    feature_frame = pd.concat(
        [numeric_frame, *categorical_frames],
        axis=1,
    ).astype("float32")

    x = torch.tensor(
        feature_frame.to_numpy(),
        dtype=torch.float,
    )

    metadata = {
        "node_feature_columns": list(feature_frame.columns),
        "node_numeric_features": numeric_feature_names,
        "node_categorical_features": categorical_feature_names,
        "node_categorical_columns": categorical_metadata,
    }

    return x, metadata


def build_node_label_tensor(
    prepared_nodes: pd.DataFrame,
) -> torch.Tensor:
    labels = pd.to_numeric(
        prepared_nodes["label"],
        errors="coerce",
    ).fillna(0).astype("int64")

    y = torch.tensor(
        labels.to_numpy(),
        dtype=torch.long,
    )

    return y


def print_node_tensor_summary(
    x: torch.Tensor,
    y: torch.Tensor,
    node_feature_metadata: dict[str, Any],
) -> None:
    print()
    print("Node tensor summary")
    print(
        {
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
            "node_feature_count": int(x.shape[1]) if x.ndim == 2 else 0,
            "label_values": sorted(y.unique().tolist()),
            "node_feature_columns": node_feature_metadata["node_feature_columns"],
        }
    )


def sanitize_edge_categorical_feature(
    valid_edges: pd.DataFrame,
    feature_name: str,
) -> pd.Series:
    values = valid_edges[feature_name].astype("string").fillna("unknown")

    if feature_name != "edge_type":
        return values

    ground_truth_mask = (
        valid_edges["event_family"].astype("string").fillna("unknown")
        == "redteam_ground_truth"
    )

    sanitized_values = values.copy()
    sanitized_values.loc[ground_truth_mask] = "ground_truth_edge_type_withheld"

    return sanitized_values


def build_edge_feature_tensor(
    valid_edges: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    numeric_feature_names = list(config["features"]["edge_numeric"])
    categorical_feature_names = list(config["features"]["edge_categorical"])

    numeric_features = []

    for feature_name in numeric_feature_names:
        if feature_name not in valid_edges.columns:
            raise ValueError(f"Missing edge numeric feature: {feature_name}")

        scaled_feature = min_max_scale_series(valid_edges[feature_name])
        numeric_features.append(scaled_feature.rename(feature_name))

    numeric_frame = pd.concat(
        numeric_features,
        axis=1,
    )

    categorical_frames = []
    categorical_metadata = {}

    for feature_name in categorical_feature_names:
        if feature_name not in valid_edges.columns:
            raise ValueError(f"Missing edge categorical feature: {feature_name}")

        sanitized_values = sanitize_edge_categorical_feature(
            valid_edges=valid_edges,
            feature_name=feature_name,
        )

        one_hot = pd.get_dummies(
            sanitized_values,
            prefix=feature_name,
            dtype="float32",
        )

        categorical_metadata[feature_name] = list(one_hot.columns)
        categorical_frames.append(one_hot)

    feature_frame = pd.concat(
        [numeric_frame, *categorical_frames],
        axis=1,
    ).astype("float32")

    # Required to avoid leakage into the edge attributes
    leakage_columns = [
        "edge_type_ground_truth_edge_type_withheld",
    ]

    feature_frame = feature_frame.drop(
        columns=[
            column
            for column in leakage_columns
            if column in feature_frame.columns
        ]
    )


    edge_attr = torch.tensor(
        feature_frame.to_numpy(),
        dtype=torch.float,
    )

    metadata = {
        "edge_feature_columns": list(feature_frame.columns),
        "edge_numeric_features": numeric_feature_names,
        "edge_categorical_features": categorical_feature_names,
        "edge_categorical_columns": categorical_metadata,
    }

    return edge_attr, metadata


def build_edge_label_tensor(
    valid_edges: pd.DataFrame,
) -> torch.Tensor:
    labels = pd.to_numeric(
        valid_edges["label"],
        errors="coerce",
    ).fillna(0).astype("int64")

    return torch.tensor(
        labels.to_numpy(),
        dtype=torch.long,
    )


def build_categorical_id_tensor(
    values: pd.Series,
) -> tuple[torch.Tensor, dict[str, int]]:
    unique_values = sorted(
        values.astype("string").fillna("unknown").unique().tolist()
    )

    value_to_id = {
        str(value): index
        for index, value in enumerate(unique_values)
    }

    ids = values.astype("string").fillna("unknown").map(value_to_id).astype("int64")

    return (
        torch.tensor(
            ids.to_numpy(),
            dtype=torch.long,
        ),
        value_to_id,
    )


def build_edge_metadata_tensors(
    valid_edges: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    edge_label = build_edge_label_tensor(valid_edges)

    edge_type_id, edge_type_mapping = build_categorical_id_tensor(
        valid_edges["edge_type"]
    )

    event_family_id, event_family_mapping = build_categorical_id_tensor(
        valid_edges["event_family"]
    )

    edge_metadata_tensors = {
        "edge_label": edge_label,
        "edge_type_id": edge_type_id,
        "event_family_id": event_family_id,
        "edge_id": valid_edges["edge_id"].astype("string").fillna("").tolist(),
    }

    edge_metadata = {
        "edge_type_mapping": edge_type_mapping,
        "event_family_mapping": event_family_mapping,
    }

    return edge_metadata_tensors, edge_metadata


def print_edge_tensor_summary(
    edge_attr: torch.Tensor,
    edge_metadata_tensors: dict[str, Any],
    edge_feature_metadata: dict[str, Any],
) -> None:
    edge_label = edge_metadata_tensors["edge_label"]
    edge_type_id = edge_metadata_tensors["edge_type_id"]

    print()
    print("Edge tensor summary")
    print(
        {
            "edge_attr_shape": list(edge_attr.shape),
            "edge_label_shape": list(edge_label.shape),
            "edge_type_id_shape": list(edge_type_id.shape),
            "edge_label_values": sorted(edge_label.unique().tolist()),
            "edge_feature_columns": edge_feature_metadata["edge_feature_columns"],
        }
    )

def build_node_metadata_tensors(
    prepared_nodes: pd.DataFrame,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    node_id = torch.tensor(
        prepared_nodes["node_id"].astype("int64").to_numpy(),
        dtype=torch.long,
    )

    node_type_id, node_type_mapping = build_categorical_id_tensor(
        prepared_nodes["entity_type"]
    )

    node_metadata_tensors = {
        "node_id": node_id,
        "node_type_id": node_type_id,
    }

    node_metadata = {
        "node_type_mapping": node_type_mapping,
    }

    return node_metadata_tensors, node_metadata


def build_pyg_data_object(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    y: torch.Tensor,
    node_metadata_tensors: dict[str, torch.Tensor],
    edge_metadata_tensors: dict[str, Any],
) -> Data:
    
    print("******************************")
    print(x.shape)
    print(edge_index.shape)
    print(edge_attr.shape)
    print(y.shape)
    print("******************************")

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )

    data.num_nodes = x.shape[0]

    data.node_id = node_metadata_tensors["node_id"]
    data.node_type_id = node_metadata_tensors["node_type_id"]

    data.edge_label = edge_metadata_tensors["edge_label"]
    data.edge_type_id = edge_metadata_tensors["edge_type_id"]
    data.event_family_id = edge_metadata_tensors["event_family_id"]
    data.edge_id = edge_metadata_tensors["edge_id"]

    return data


def validate_pyg_data_object(
    data: Data,
) -> None:
    data.validate(raise_on_error=True)

    if data.x.ndim != 2:
        raise ValueError("data.x must have shape [num_nodes, num_node_features].")

    if data.y.ndim != 1:
        raise ValueError("data.y must have shape [num_nodes].")

    if data.x.shape[0] != data.num_nodes:
        raise ValueError("data.x row count must match data.num_nodes.")

    if data.y.shape[0] != data.num_nodes:
        raise ValueError("data.y length must match data.num_nodes.")

    if data.edge_index.shape[1] != data.edge_attr.shape[0]:
        raise ValueError("edge_attr rows must match edge_index columns.")

    if data.edge_label.shape[0] != data.edge_index.shape[1]:
        raise ValueError("edge_label length must match number of edges.")

    if data.edge_type_id.shape[0] != data.edge_index.shape[1]:
        raise ValueError("edge_type_id length must match number of edges.")

    if data.event_family_id.shape[0] != data.edge_index.shape[1]:
        raise ValueError("event_family_id length must match number of edges.")

    if len(data.edge_id) != data.edge_index.shape[1]:
        raise ValueError("edge_id count must match number of edges.")


def print_pyg_data_summary(
    data: Data,
) -> None:
    print()
    print("PyG Data object summary")
    print(
        {
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.shape[1]),
            "x_shape": list(data.x.shape),
            "edge_index_shape": list(data.edge_index.shape),
            "edge_attr_shape": list(data.edge_attr.shape),
            "y_shape": list(data.y.shape),
            "edge_label_shape": list(data.edge_label.shape),
            "node_id_shape": list(data.node_id.shape),
            "node_type_id_shape": list(data.node_type_id.shape),
            "edge_type_id_shape": list(data.edge_type_id.shape),
            "event_family_id_shape": list(data.event_family_id.shape),
        }
    )


def build_graph_metadata(
    config: dict[str, Any],
    paths: dict[str, Path],
    prepared_nodes: pd.DataFrame,
    edges: pd.DataFrame,
    valid_edges: pd.DataFrame,
    invalid_edges: pd.DataFrame,
    data: Data,
    node_feature_metadata: dict[str, Any],
    edge_feature_metadata: dict[str, Any],
    node_metadata: dict[str, Any],
    edge_metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "pyg_data_preparation_before_model_training",
        "purpose": "Build homogeneous LANL PyG Data object from graph tables.",
        "inputs": {
            "graph_nodes": str(paths["graph_nodes"]),
            "graph_edges": str(paths["graph_edges"]),
        },
        "outputs": {
            "pyg_graph": str(paths["pyg_graph"]),
            "metadata": str(paths["metadata"]),
        },
        "counts": {
            "node_rows": int(len(prepared_nodes)),
            "edge_rows": int(len(edges)),
            "valid_edge_rows": int(len(valid_edges)),
            "invalid_edge_rows": int(len(invalid_edges)),
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.shape[1]),
        },
        "tensor_shapes": {
            "x": list(data.x.shape),
            "edge_index": list(data.edge_index.shape),
            "edge_attr": list(data.edge_attr.shape),
            "y": list(data.y.shape),
            "edge_label": list(data.edge_label.shape),
            "node_id": list(data.node_id.shape),
            "node_type_id": list(data.node_type_id.shape),
            "edge_type_id": list(data.edge_type_id.shape),
            "event_family_id": list(data.event_family_id.shape),
        },
        "labels": {
            "node_label_values": sorted(data.y.unique().tolist()),
            "edge_label_values": sorted(data.edge_label.unique().tolist()),
        },
        "indexing": {
            "node_id_equals_pyg_index": node_id_equals_pyg_index(prepared_nodes),
            "min_pyg_node_index": int(prepared_nodes["pyg_node_index"].min())
            if not prepared_nodes.empty
            else None,
            "max_pyg_node_index": int(prepared_nodes["pyg_node_index"].max())
            if not prepared_nodes.empty
            else None,
        },
        "features": {
            **node_feature_metadata,
            **edge_feature_metadata,
        },
        "mappings": {
            **node_metadata,
            **edge_metadata,
        },
        "validation": config["validation"],
    }

    return make_json_safe(metadata)


def save_pyg_outputs(
    data: Data,
    metadata: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Path]:
    paths["pyg_graph"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        data,
        paths["pyg_graph"],
    )

    paths["metadata"].write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "pyg_graph": paths["pyg_graph"],
        "metadata": paths["metadata"],
    }


def print_output_paths(
    output_paths: dict[str, Path],
) -> None:
    print()
    print("PyG conversion outputs")
    print(
        {
            key: str(path)
            for key, path in output_paths.items()
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

    x, node_feature_metadata = build_node_feature_tensor(
        prepared_nodes=prepared_nodes,
        config=config,
    )

    y = build_node_label_tensor(prepared_nodes)

    print_node_tensor_summary(
        x=x,
        y=y,
        node_feature_metadata=node_feature_metadata,
    )

    edge_attr, edge_feature_metadata = build_edge_feature_tensor(
        valid_edges=valid_edges,
        config=config,
    )

    edge_metadata_tensors, edge_metadata = build_edge_metadata_tensors(
        valid_edges=valid_edges,
    )

    print_edge_tensor_summary(
        edge_attr=edge_attr,
        edge_metadata_tensors=edge_metadata_tensors,
        edge_feature_metadata=edge_feature_metadata,
    )

    node_metadata_tensors, node_metadata = build_node_metadata_tensors(
        prepared_nodes=prepared_nodes,
    )

    data = build_pyg_data_object(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        node_metadata_tensors=node_metadata_tensors,
        edge_metadata_tensors=edge_metadata_tensors,
    )

    validate_pyg_data_object(data)

    print_pyg_data_summary(data)

    metadata = build_graph_metadata(
        config=config,
        paths=paths,
        prepared_nodes=prepared_nodes,
        edges=edges,
        valid_edges=valid_edges,
        invalid_edges=invalid_edges,
        data=data,
        node_feature_metadata=node_feature_metadata,
        edge_feature_metadata=edge_feature_metadata,
        node_metadata=node_metadata,
        edge_metadata=edge_metadata,
    )

    output_paths = save_pyg_outputs(
        data=data,
        metadata=metadata,
        paths=paths,
    )

    print_output_paths(output_paths)


if __name__ == "__main__":
    main()