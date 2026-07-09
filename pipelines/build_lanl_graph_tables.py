from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import pandas as pd
import yaml
from datetime import datetime, timezone
import json

CONFIG_PATH = "configs/lanl_graph_tables.yaml"


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

    if "clean_events_filename" not in config["input"]:
        raise ValueError("Missing input.clean_events_filename in config.")

    if "directory" not in config["output"]:
        raise ValueError("Missing output.directory in config.")

    required_output_keys = [
        "graph_nodes_filename",
        "graph_edges_filename",
        "manifest_filename",
    ]

    for key in required_output_keys:
        if key not in config["output"]:
            raise ValueError(f"Missing output.{key} in config.")

    required_schema_keys = [
        "node_key_separator",
        "unknown_value",
        "default_node_label",
    ]

    for key in required_schema_keys:
        if key not in config["schema"]:
            raise ValueError(f"Missing schema.{key} in config.")


def build_paths(config: dict[str, Any]) -> dict[str, Path]:
    input_directory = Path(config["input"]["directory"])
    output_directory = Path(config["output"]["directory"])

    return {
        "clean_events": input_directory / config["input"]["clean_events_filename"],
        "graph_nodes": output_directory / config["output"]["graph_nodes_filename"],
        "graph_edges": output_directory / config["output"]["graph_edges_filename"],
        "manifest": output_directory / config["output"]["manifest_filename"],
    }


def validate_input_paths(paths: dict[str, Path]) -> None:
    clean_events_path = paths["clean_events"]

    if not clean_events_path.exists():
        raise FileNotFoundError(
            f"Clean events parquet file not found: {clean_events_path}"
        )

    if not clean_events_path.is_file():
        raise FileNotFoundError(
            f"Clean events path is not a file: {clean_events_path}"
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


def read_clean_events(
    clean_events_path: Path,
) -> pd.DataFrame:
    events = pd.read_parquet(clean_events_path)

    required_columns = [
        "event_id",
        "timestamp",
        "event_family",
        "event_type",
        "source_entity",
        "destination_entity",
        "source_entity_type",
        "destination_entity_type",
        "event_result",
        "label",
        "source_file",
        "row_number",
    ]

    validate_columns(
        dataframe=events,
        required_columns=required_columns,
        table_name="clean_events",
    )

    return events


def build_node_keys(
    entity_types: pd.Series,
    entity_names: pd.Series,
    separator: str,
    unknown_value: str,
) -> pd.Series:
    clean_entity_types = normalize_string_series(
        entity_types,
        unknown_value,
    )
    clean_entity_names = normalize_string_series(
        entity_names,
        unknown_value,
    )

    return clean_entity_types + separator + clean_entity_names


def build_graph_nodes(
    events: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    separator = config["schema"]["node_key_separator"]
    unknown_value = config["schema"]["unknown_value"]

    source_nodes = pd.DataFrame(
        {
            "entity_type": normalize_string_series(
                events["source_entity_type"],
                unknown_value,
            ),
            "entity_name": normalize_string_series(
                events["source_entity"],
                unknown_value,
            ),
        }
    )

    destination_nodes = pd.DataFrame(
        {
            "entity_type": normalize_string_series(
                events["destination_entity_type"],
                unknown_value,
            ),
            "entity_name": normalize_string_series(
                events["destination_entity"],
                unknown_value,
            ),
        }
    )

    nodes = pd.concat(
        [source_nodes, destination_nodes],
        axis=0,
        ignore_index=True,
    )

    nodes["node_key"] = build_node_keys(
        entity_types=nodes["entity_type"],
        entity_names=nodes["entity_name"],
        separator=separator,
        unknown_value=unknown_value,
    )

    nodes = (
        nodes.drop_duplicates(subset=["node_key"])
        .sort_values(by=["entity_type", "entity_name"], kind="stable")
        .reset_index(drop=True)
    )

    nodes.insert(0, "node_id", range(len(nodes)))

    return nodes


def write_parquet_file(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)


def read_parquet_file(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_node_id_mapping(nodes: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "node_id",
        "entity_type",
        "entity_name",
        "node_key",
    ]

    validate_columns(
        dataframe=nodes,
        required_columns=required_columns,
        table_name="graph_nodes",
    )

    return nodes[
        [
            "node_id",
            "entity_type",
            "entity_name",
            "node_key",
        ]
    ].copy()


def build_edge_type(
    event_family: pd.Series,
    event_type: pd.Series,
    unknown_value: str,
) -> pd.Series:
    clean_event_family = normalize_string_series(
        event_family,
        unknown_value,
    )
    clean_event_type = normalize_string_series(
        event_type,
        unknown_value,
    )

    return clean_event_family + ":" + clean_event_type


def build_graph_edges(
    events: pd.DataFrame,
    nodes: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    separator = config["schema"]["node_key_separator"]
    unknown_value = config["schema"]["unknown_value"]

    edge_base = events.copy()

    edge_base["source_node_key"] = build_node_keys(
        entity_types=edge_base["source_entity_type"],
        entity_names=edge_base["source_entity"],
        separator=separator,
        unknown_value=unknown_value,
    )
    edge_base["destination_node_key"] = build_node_keys(
        entity_types=edge_base["destination_entity_type"],
        entity_names=edge_base["destination_entity"],
        separator=separator,
        unknown_value=unknown_value,
    )

    node_mapping = build_node_id_mapping(nodes)

    source_mapping = node_mapping.rename(
        columns={
            "node_id": "source_node_id",
            "node_key": "source_node_key",
        }
    )[["source_node_id", "source_node_key"]]

    destination_mapping = node_mapping.rename(
        columns={
            "node_id": "destination_node_id",
            "node_key": "destination_node_key",
        }
    )[["destination_node_id", "destination_node_key"]]

    edge_base = edge_base.merge(
        source_mapping,
        on="source_node_key",
        how="left",
        validate="many_to_one",
    )

    edge_base = edge_base.merge(
        destination_mapping,
        on="destination_node_key",
        how="left",
        validate="many_to_one",
    )

    if edge_base["source_node_id"].isna().any():
        raise ValueError("Some edges are missing source_node_id values.")

    if edge_base["destination_node_id"].isna().any():
        raise ValueError("Some edges are missing destination_node_id values.")

    edges = pd.DataFrame()

    edges["edge_id"] = edge_base["event_id"]
    edges["source_node_id"] = edge_base["source_node_id"].astype("int64")
    edges["destination_node_id"] = edge_base["destination_node_id"].astype("int64")

    edges["source_entity"] = normalize_string_series(
        edge_base["source_entity"],
        unknown_value,
    )
    edges["destination_entity"] = normalize_string_series(
        edge_base["destination_entity"],
        unknown_value,
    )
    edges["source_entity_type"] = normalize_string_series(
        edge_base["source_entity_type"],
        unknown_value,
    )
    edges["destination_entity_type"] = normalize_string_series(
        edge_base["destination_entity_type"],
        unknown_value,
    )

    edges["edge_type"] = build_edge_type(
        event_family=edge_base["event_family"],
        event_type=edge_base["event_type"],
        unknown_value=unknown_value,
    )
    edges["event_family"] = normalize_string_series(
        edge_base["event_family"],
        unknown_value,
    )
    edges["event_type"] = normalize_string_series(
        edge_base["event_type"],
        unknown_value,
    )

    edges["timestamp"] = pd.to_numeric(
        edge_base["timestamp"],
        errors="coerce",
    ).fillna(-1).astype("int64")
    edges["event_result"] = normalize_string_series(
        edge_base["event_result"],
        unknown_value,
    )
    edges["label"] = pd.to_numeric(
        edge_base["label"],
        errors="coerce",
    ).fillna(0).astype("int64")

    edges["source_file"] = normalize_string_series(
        edge_base["source_file"],
        unknown_value,
    )
    edges["row_number"] = pd.to_numeric(
        edge_base["row_number"],
        errors="coerce",
    ).fillna(-1).astype("int64")

    return edges


def build_node_statistics(
    edges: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = [
        "source_node_id",
        "destination_node_id",
        "timestamp",
        "label",
    ]

    validate_columns(
        dataframe=edges,
        required_columns=required_columns,
        table_name="graph_edges",
    )

    source_participation = edges[
        [
            "source_node_id",
            "timestamp",
            "label",
        ]
    ].rename(
        columns={
            "source_node_id": "node_id",
        }
    )

    destination_participation = edges[
        [
            "destination_node_id",
            "timestamp",
            "label",
        ]
    ].rename(
        columns={
            "destination_node_id": "node_id",
        }
    )

    participation = pd.concat(
        [
            source_participation,
            destination_participation,
        ],
        axis=0,
        ignore_index=True,
    )

    statistics = (
        participation.groupby("node_id")
        .agg(
            first_seen_timestamp=("timestamp", "min"),
            last_seen_timestamp=("timestamp", "max"),
            event_count=("timestamp", "count"),
            label=("label", "max"),
        )
        .reset_index()
    )

    statistics["node_id"] = statistics["node_id"].astype("int64")
    statistics["first_seen_timestamp"] = statistics[
        "first_seen_timestamp"
    ].astype("int64")
    statistics["last_seen_timestamp"] = statistics[
        "last_seen_timestamp"
    ].astype("int64")
    statistics["event_count"] = statistics["event_count"].astype("int64")
    statistics["label"] = statistics["label"].astype("int64")

    return statistics


def add_node_statistics_and_labels(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    required_node_columns = [
        "node_id",
        "entity_type",
        "entity_name",
        "node_key",
    ]

    validate_columns(
        dataframe=nodes,
        required_columns=required_node_columns,
        table_name="graph_nodes",
    )

    default_node_label = config["schema"]["default_node_label"]

    statistics = build_node_statistics(edges)

    enriched_nodes = nodes.merge(
        statistics,
        on="node_id",
        how="left",
        validate="one_to_one",
    )

    enriched_nodes["first_seen_timestamp"] = enriched_nodes[
        "first_seen_timestamp"
    ].fillna(-1).astype("int64")
    enriched_nodes["last_seen_timestamp"] = enriched_nodes[
        "last_seen_timestamp"
    ].fillna(-1).astype("int64")
    enriched_nodes["event_count"] = enriched_nodes[
        "event_count"
    ].fillna(0).astype("int64")
    enriched_nodes["label"] = enriched_nodes[
        "label"
    ].fillna(default_node_label).astype("int64")

    return enriched_nodes


def update_node_table_with_statistics(
    paths: dict[str, Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    nodes = read_parquet_file(paths["graph_nodes"])
    edges = read_parquet_file(paths["graph_edges"])

    enriched_nodes = add_node_statistics_and_labels(
        nodes=nodes,
        edges=edges,
        config=config,
    )

    write_parquet_file(
        dataframe=enriched_nodes,
        output_path=paths["graph_nodes"],
    )

    return {
        "table": "graph_nodes",
        "output_path": str(paths["graph_nodes"]),
        "rows": int(len(enriched_nodes)),
        "columns": list(enriched_nodes.columns),
        "entity_type_counts": {
            str(key): int(value)
            for key, value in enriched_nodes[
                "entity_type"
            ].value_counts().sort_index().items()
        },
        "label_counts": {
            str(key): int(value)
            for key, value in enriched_nodes[
                "label"
            ].value_counts().sort_index().items()
        },
    }


def build_edge_table(
    paths: dict[str, Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    events = read_clean_events(paths["clean_events"])
    nodes = read_parquet_file(paths["graph_nodes"])

    edges = build_graph_edges(
        events=events,
        nodes=nodes,
        config=config,
    )

    write_parquet_file(
        dataframe=edges,
        output_path=paths["graph_edges"],
    )

    return {
        "table": "graph_edges",
        "output_path": str(paths["graph_edges"]),
        "rows": int(len(edges)),
        "columns": list(edges.columns),
        "edge_type_counts": {
            str(key): int(value)
            for key, value in edges["edge_type"].value_counts().sort_index().items()
        },
        "label_counts": {
            str(key): int(value)
            for key, value in edges["label"].value_counts().sort_index().items()
        },
    }

def write_json_file(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def build_node_table(
    paths: dict[str, Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    events = read_clean_events(paths["clean_events"])

    nodes = build_graph_nodes(
        events=events,
        config=config,
    )

    write_parquet_file(
        dataframe=nodes,
        output_path=paths["graph_nodes"],
    )

    return {
        "table": "graph_nodes",
        "output_path": str(paths["graph_nodes"]),
        "rows": int(len(nodes)),
        "columns": list(nodes.columns),
        "entity_type_counts": {
            str(key): int(value)
            for key, value in nodes["entity_type"].value_counts().sort_index().items()
        },
    }


def print_graph_table_plan(paths: dict[str, Path]) -> None:
    print("LANL graph table build plan")
    print("Input:")
    print(
        {
            "clean_events": str(paths["clean_events"]),
        }
    )

    print("Outputs:")
    print(
        {
            "graph_nodes": str(paths["graph_nodes"]),
            "graph_edges": str(paths["graph_edges"]),
            "manifest": str(paths["manifest"]),
        }
    )


def build_graph_table_manifest(
    config: dict[str, Any],
    paths: dict[str, Path],
    node_result: dict[str, Any],
    edge_result: dict[str, Any],
    enriched_node_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "directory": config["input"]["directory"],
            "clean_events_path": str(paths["clean_events"]),
        },
        "output": {
            "directory": config["output"]["directory"],
            "graph_nodes_path": str(paths["graph_nodes"]),
            "graph_edges_path": str(paths["graph_edges"]),
            "manifest_path": str(paths["manifest"]),
        },
        "schema": {
            "node_key_separator": config["schema"]["node_key_separator"],
            "unknown_value": config["schema"]["unknown_value"],
            "default_node_label": config["schema"]["default_node_label"],
        },
        "tables": {
            "initial_nodes": node_result,
            "edges": edge_result,
            "enriched_nodes": enriched_node_result,
        },
    }


def write_graph_table_manifest(
    config: dict[str, Any],
    paths: dict[str, Path],
    node_result: dict[str, Any],
    edge_result: dict[str, Any],
    enriched_node_result: dict[str, Any],
) -> Path:
    manifest = build_graph_table_manifest(
        config=config,
        paths=paths,
        node_result=node_result,
        edge_result=edge_result,
        enriched_node_result=enriched_node_result,
    )

    write_json_file(
        payload=manifest,
        output_path=paths["manifest"],
    )

    return paths["manifest"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to LANL graph table config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    validate_config(config)

    paths = build_paths(config)
    validate_input_paths(paths)

    output_directory = Path(config["output"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    print("Config path:", args.config)
    print("Input directory:", config["input"]["directory"])
    print("Output directory:", output_directory)
    print("Node key separator:", config["schema"]["node_key_separator"])

    print_graph_table_plan(paths)

    node_result = build_node_table(
        paths=paths,
        config=config,
    )

    edge_result = build_edge_table(
        paths=paths,
        config=config,
    )

    enriched_node_result = update_node_table_with_statistics(
        paths=paths,
        config=config,
    )

    print("Built graph node table:", node_result)
    print("Built graph edge table:", edge_result)
    print("Updated graph node table:", enriched_node_result)

    manifest_path = write_graph_table_manifest(
        config=config,
        paths=paths,
        node_result=node_result,
        edge_result=edge_result,
        enriched_node_result=enriched_node_result,
    )

    print("Graph table manifest path:", manifest_path)


if __name__ == "__main__":
    main()