from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import networkx as nx
import pandas as pd
import yaml


CONFIG_PATH = "configs/lanl_graph_inspection.yaml"


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
        "inspection",
        "graph",
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
        "markdown_report_filename",
        "json_report_filename",
        "top_nodes_filename",
    ]

    for key in required_output_keys:
        if key not in config["output"]:
            raise ValueError(f"Missing output.{key} in config.")

    required_inspection_keys = [
        "top_k_nodes",
        "component_sample_size",
        "redteam_sample_size",
        "max_edges_for_visual_sample",
        "invalid_edge_policy",
        "invalid_edge_sample_size",
    ]

    for key in required_inspection_keys:
        if key not in config["inspection"]:
            raise ValueError(f"Missing inspection.{key} in config.")

    if "type" not in config["graph"]:
        raise ValueError("Missing graph.type in config.")

    valid_invalid_edge_policies = {"warn", "fail"}

    if config["inspection"]["invalid_edge_policy"] not in valid_invalid_edge_policies:
        raise ValueError(
            "inspection.invalid_edge_policy must be one of: "
            f"{sorted(valid_invalid_edge_policies)}"
        )

    if config["graph"]["type"] != "multidigraph":
        raise ValueError("Only graph.type=multidigraph is supported.")


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


def split_valid_and_invalid_edges(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_node_ids = set(nodes["node_id"].astype("int64"))

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
    policy = config["inspection"]["invalid_edge_policy"]

    if invalid_edge_count == 0:
        print("No invalid edges found.")
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
            "action": "invalid edges will be excluded from graph construction",
        },
    )


def print_graph_table_summary(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    valid_edges: pd.DataFrame,
    invalid_edges: pd.DataFrame,
) -> None:
    print()
    print("Graph table summary - graph_nodes.parquet, graph_edges.parquet")
    print(
        {
            "node_rows": len(nodes),
            "edge_rows": len(edges),
            "valid_edge_rows": len(valid_edges),
            "invalid_edge_rows": len(invalid_edges),
            "unique_node_ids": nodes["node_id"].nunique(),
            "unique_edge_ids": edges["edge_id"].nunique(),
        }
    )

def to_python_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def build_node_attributes(row: pd.Series) -> dict[str, Any]:
    return {
        "entity_type": to_python_scalar(row["entity_type"]),
        "entity_name": to_python_scalar(row["entity_name"]),
        "node_key": to_python_scalar(row["node_key"]),
        "first_seen_timestamp": to_python_scalar(row["first_seen_timestamp"]),
        "last_seen_timestamp": to_python_scalar(row["last_seen_timestamp"]),
        "event_count": to_python_scalar(row["event_count"]),
        "label": to_python_scalar(row["label"]),
    }

def build_edge_attributes(row: pd.Series) -> dict[str, Any]:
    return {
        "edge_id": to_python_scalar(row["edge_id"]),
        "edge_type": to_python_scalar(row["edge_type"]),
        "event_family": to_python_scalar(row["event_family"]),
        "event_type": to_python_scalar(row["event_type"]),
        "timestamp": to_python_scalar(row["timestamp"]),
        "event_result": to_python_scalar(row["event_result"]),
        "label": to_python_scalar(row["label"]),
        "source_entity": to_python_scalar(row["source_entity"]),
        "destination_entity": to_python_scalar(row["destination_entity"]),
        "source_entity_type": to_python_scalar(row["source_entity_type"]),
        "destination_entity_type": to_python_scalar(row["destination_entity_type"]),
        "source_file": to_python_scalar(row["source_file"]),
        "row_number": to_python_scalar(row["row_number"]),
    }


def build_networkx_graph(
    nodes: pd.DataFrame,
    valid_edges: pd.DataFrame,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    for _, row in nodes.iterrows():
        node_id = int(row["node_id"])
        graph.add_node(
            node_id,
            **build_node_attributes(row),
        )

    for _, row in valid_edges.iterrows():
        source_node_id = int(row["source_node_id"])
        destination_node_id = int(row["destination_node_id"])
        edge_key = str(row["edge_id"])

        graph.add_edge(
            source_node_id,
            destination_node_id,
            key=edge_key,
            **build_edge_attributes(row),
        )

    return graph

def print_networkx_graph_summary(
    graph: nx.MultiDiGraph,
) -> None:
    print()
    print("NetworkX graph summary")
    print(
        {
            "graph_type": type(graph).__name__,
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "is_directed": graph.is_directed(),
            "is_multigraph": graph.is_multigraph(),
        }
    )


def value_counts_to_dict(
    series: pd.Series,
) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()

    return {
        str(key): int(value)
        for key, value in counts.items()
    }


def build_graph_type_summary(
    graph: nx.MultiDiGraph,
) -> dict[str, Any]:
    return {
        "graph_type": type(graph).__name__,
        "is_directed": bool(graph.is_directed()),
        "is_multigraph": bool(graph.is_multigraph()),
        "node_count": int(graph.number_of_nodes()),
        "edge_count": int(graph.number_of_edges()),
    }


def build_basic_graph_profile(
    graph: nx.MultiDiGraph,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    valid_edges: pd.DataFrame,
    invalid_edges: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "graph": build_graph_type_summary(graph),
        "table_counts": {
            "node_rows": int(len(nodes)),
            "edge_rows": int(len(edges)),
            "valid_edge_rows": int(len(valid_edges)),
            "invalid_edge_rows": int(len(invalid_edges)),
            "unique_node_ids": int(nodes["node_id"].nunique()),
            "unique_edge_ids": int(edges["edge_id"].nunique()),
        },
        "node_type_counts": value_counts_to_dict(nodes["entity_type"]),
        "node_label_counts": value_counts_to_dict(nodes["label"]),
        "edge_type_counts": value_counts_to_dict(valid_edges["edge_type"]),
        "event_family_counts": value_counts_to_dict(valid_edges["event_family"]),
        "edge_label_counts": value_counts_to_dict(valid_edges["label"]),
    }


def print_basic_graph_profile(
    profile: dict[str, Any],
) -> None:
    print()
    print("Basic graph profile")
    print("Graph:")
    print(profile["graph"])

    print("Table counts:")
    print(profile["table_counts"])

    print("Node type counts:")
    print(profile["node_type_counts"])

    print("Node label counts:")
    print(profile["node_label_counts"])

    print("Edge type counts:")
    print(profile["edge_type_counts"])

    print("Event family counts:")
    print(profile["event_family_counts"])

    print("Edge label counts:")
    print(profile["edge_label_counts"])


def summarize_numeric_series(
    series: pd.Series,
) -> dict[str, float | int]:
    if series.empty:
        return {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
        }

    return {
        "min": int(series.min()),
        "max": int(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
    }


def build_degree_table(
    graph: nx.MultiDiGraph,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    degree_rows = []

    for node_id in graph.nodes:
        degree_rows.append(
            {
                "node_id": int(node_id),
                "total_degree": int(graph.degree(node_id)),
                "in_degree": int(graph.in_degree(node_id)),
                "out_degree": int(graph.out_degree(node_id)),
            }
        )

    degrees = pd.DataFrame(degree_rows)

    node_context_columns = [
        "node_id",
        "entity_type",
        "entity_name",
        "node_key",
        "label",
    ]

    validate_columns(
        dataframe=nodes,
        required_columns=node_context_columns,
        table_name="graph_nodes",
    )

    degrees = degrees.merge(
        nodes[node_context_columns],
        on="node_id",
        how="left",
        validate="one_to_one",
    )

    return degrees[
        [
            "node_id",
            "node_key",
            "entity_type",
            "entity_name",
            "label",
            "total_degree",
            "in_degree",
            "out_degree",
        ]
    ]


def select_top_nodes(
    degree_table: pd.DataFrame,
    degree_column: str,
    top_k: int,
) -> list[dict[str, Any]]:
    top_nodes = (
        degree_table.sort_values(
            by=[degree_column, "node_id"],
            ascending=[False, True],
            kind="stable",
        )
        .head(top_k)
        .copy()
    )

    return [
        {
            "node_id": int(row["node_id"]),
            "node_key": str(row["node_key"]),
            "entity_type": str(row["entity_type"]),
            "entity_name": str(row["entity_name"]),
            "label": int(row["label"]),
            "total_degree": int(row["total_degree"]),
            "in_degree": int(row["in_degree"]),
            "out_degree": int(row["out_degree"]),
        }
        for _, row in top_nodes.iterrows()
    ]


def build_degree_profile(
    graph: nx.MultiDiGraph,
    nodes: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    top_k = int(config["inspection"]["top_k_nodes"])

    degree_table = build_degree_table(
        graph=graph,
        nodes=nodes,
    )

    degree_profile = {
        "degree_summary": {
            "total_degree": summarize_numeric_series(
                degree_table["total_degree"]
            ),
            "in_degree": summarize_numeric_series(
                degree_table["in_degree"]
            ),
            "out_degree": summarize_numeric_series(
                degree_table["out_degree"]
            ),
        },
        "top_total_degree_nodes": select_top_nodes(
            degree_table=degree_table,
            degree_column="total_degree",
            top_k=top_k,
        ),
        "top_in_degree_nodes": select_top_nodes(
            degree_table=degree_table,
            degree_column="in_degree",
            top_k=top_k,
        ),
        "top_out_degree_nodes": select_top_nodes(
            degree_table=degree_table,
            degree_column="out_degree",
            top_k=top_k,
        ),
    }

    return degree_profile, degree_table


def print_degree_profile(
    degree_profile: dict[str, Any],
) -> None:
    print()
    print("Degree profile")
    print("Degree summary:")
    print(degree_profile["degree_summary"])

    print("Top total-degree nodes:")
    print(degree_profile["top_total_degree_nodes"])

    print("Top in-degree nodes:")
    print(degree_profile["top_in_degree_nodes"])

    print("Top out-degree nodes:")
    print(degree_profile["top_out_degree_nodes"])


def summarize_component_sizes(
    components: list[set[int]],
    sample_size: int,
) -> dict[str, Any]:
    component_sizes = sorted(
        [len(component) for component in components],
        reverse=True,
    )

    if not component_sizes:
        return {
            "component_count": 0,
            "largest_component_size": 0,
            "smallest_component_size": 0,
            "component_size_sample": [],
        }

    return {
        "component_count": int(len(component_sizes)),
        "largest_component_size": int(component_sizes[0]),
        "smallest_component_size": int(component_sizes[-1]),
        "component_size_sample": [
            int(size)
            for size in component_sizes[:sample_size]
        ],
    }


def build_connected_component_profile(
    graph: nx.MultiDiGraph,
    config: dict[str, Any],
) -> dict[str, Any]:
    component_sample_size = int(config["inspection"]["component_sample_size"])

    weak_components = [
        set(component)
        for component in nx.weakly_connected_components(graph)
    ]
    strong_components = [
        set(component)
        for component in nx.strongly_connected_components(graph)
    ]

    weak_summary = summarize_component_sizes(
        components=weak_components,
        sample_size=component_sample_size,
    )
    strong_summary = summarize_component_sizes(
        components=strong_components,
        sample_size=component_sample_size,
    )

    node_count = graph.number_of_nodes()

    if node_count == 0:
        largest_weak_component_ratio = 0.0
        largest_strong_component_ratio = 0.0
    else:
        largest_weak_component_ratio = (
            weak_summary["largest_component_size"] / node_count
        )
        largest_strong_component_ratio = (
            strong_summary["largest_component_size"] / node_count
        )

    return {
        "weak_components": {
            **weak_summary,
            "largest_component_ratio": float(largest_weak_component_ratio),
        },
        "strong_components": {
            **strong_summary,
            "largest_component_ratio": float(largest_strong_component_ratio),
        },
    }


def build_isolated_node_profile(
    graph: nx.MultiDiGraph,
    nodes: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    sample_size = int(config["inspection"]["component_sample_size"])

    isolated_node_ids = [
        int(node_id)
        for node_id in nx.isolates(graph)
    ]

    isolated_nodes = nodes[
        nodes["node_id"].astype("int64").isin(isolated_node_ids)
    ].copy()

    sample_columns = [
        "node_id",
        "node_key",
        "entity_type",
        "entity_name",
        "label",
    ]

    isolated_sample = isolated_nodes[
        sample_columns
    ].head(sample_size).to_dict(orient="records")

    return {
        "isolated_node_count": int(len(isolated_node_ids)),
        "isolated_node_sample": isolated_sample,
    }


def build_self_loop_profile(
    graph: nx.MultiDiGraph,
    config: dict[str, Any],
) -> dict[str, Any]:
    sample_size = int(config["inspection"]["component_sample_size"])

    self_loop_edges = list(
        nx.selfloop_edges(
            graph,
            keys=True,
            data=True,
        )
    )

    sample = []

    for source_node_id, destination_node_id, edge_key, edge_data in self_loop_edges[
        :sample_size
    ]:
        sample.append(
            {
                "source_node_id": int(source_node_id),
                "destination_node_id": int(destination_node_id),
                "edge_key": str(edge_key),
                "edge_type": str(edge_data.get("edge_type")),
                "event_family": str(edge_data.get("event_family")),
                "label": int(edge_data.get("label", 0)),
            }
        )

    return {
        "self_loop_count": int(len(self_loop_edges)),
        "self_loop_sample": sample,
    }


def build_structural_graph_profile(
    graph: nx.MultiDiGraph,
    nodes: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "connected_components": build_connected_component_profile(
            graph=graph,
            config=config,
        ),
        "isolated_nodes": build_isolated_node_profile(
            graph=graph,
            nodes=nodes,
            config=config,
        ),
        "self_loops": build_self_loop_profile(
            graph=graph,
            config=config,
        ),
    }


def print_structural_graph_profile(
    structural_profile: dict[str, Any],
) -> None:
    print()
    print("Structural graph profile")
    print("Connected components:")
    print(structural_profile["connected_components"])

    print("Isolated nodes:")
    print(structural_profile["isolated_nodes"])

    print("Self-loops:")
    print(structural_profile["self_loops"])


def build_node_context_lookup(
    nodes: pd.DataFrame,
) -> dict[int, dict[str, Any]]:
    required_columns = [
        "node_id",
        "node_key",
        "entity_type",
        "entity_name",
        "label",
    ]

    validate_columns(
        dataframe=nodes,
        required_columns=required_columns,
        table_name="graph_nodes",
    )

    lookup = {}

    for _, row in nodes.iterrows():
        node_id = int(row["node_id"])
        lookup[node_id] = {
            "node_key": str(row["node_key"]),
            "entity_type": str(row["entity_type"]),
            "entity_name": str(row["entity_name"]),
            "label": int(row["label"]),
        }

    return lookup


def filter_redteam_edges(
    valid_edges: pd.DataFrame,
) -> pd.DataFrame:
    edge_labels = pd.to_numeric(
        valid_edges["label"],
        errors="coerce",
    ).fillna(0).astype("int64")

    return valid_edges[edge_labels == 1].copy()


def build_redteam_edge_sample(
    redteam_edges: pd.DataFrame,
    nodes: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    sample_size = int(config["inspection"]["redteam_sample_size"])
    node_lookup = build_node_context_lookup(nodes)

    sample = []

    ordered_edges = redteam_edges.sort_values(
        by=["timestamp", "edge_id"],
        kind="stable",
    ).head(sample_size)

    for _, row in ordered_edges.iterrows():
        source_node_id = int(row["source_node_id"])
        destination_node_id = int(row["destination_node_id"])

        source_context = node_lookup.get(source_node_id, {})
        destination_context = node_lookup.get(destination_node_id, {})

        sample.append(
            {
                "edge_id": str(row["edge_id"]),
                "timestamp": int(row["timestamp"]),
                "edge_type": str(row["edge_type"]),
                "event_family": str(row["event_family"]),
                "source_node_id": source_node_id,
                "destination_node_id": destination_node_id,
                "source_node_key": str(source_context.get("node_key")),
                "destination_node_key": str(destination_context.get("node_key")),
                "source_entity_type": str(row["source_entity_type"]),
                "destination_entity_type": str(row["destination_entity_type"]),
                "label": int(row["label"]),
            }
        )

    return sample


def build_redteam_graph_profile(
    valid_edges: pd.DataFrame,
    nodes: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    redteam_edges = filter_redteam_edges(valid_edges)

    if redteam_edges.empty:
        participating_node_ids: set[int] = set()
    else:
        source_node_ids = set(redteam_edges["source_node_id"].astype("int64"))
        destination_node_ids = set(
            redteam_edges["destination_node_id"].astype("int64")
        )
        participating_node_ids = source_node_ids | destination_node_ids

    labeled_nodes = nodes[
        pd.to_numeric(
            nodes["label"],
            errors="coerce",
        ).fillna(0).astype("int64")
        == 1
    ].copy()

    participating_nodes = nodes[
        nodes["node_id"].astype("int64").isin(participating_node_ids)
    ].copy()

    return {
        "redteam_edge_count": int(len(redteam_edges)),
        "redteam_participating_node_count": int(len(participating_nodes)),
        "redteam_labeled_node_count": int(len(labeled_nodes)),
        "redteam_edge_type_counts": value_counts_to_dict(
            redteam_edges["edge_type"]
        )
        if not redteam_edges.empty
        else {},
        "redteam_event_family_counts": value_counts_to_dict(
            redteam_edges["event_family"]
        )
        if not redteam_edges.empty
        else {},
        "redteam_node_type_counts": value_counts_to_dict(
            participating_nodes["entity_type"]
        )
        if not participating_nodes.empty
        else {},
        "redteam_edge_sample": build_redteam_edge_sample(
            redteam_edges=redteam_edges,
            nodes=nodes,
            config=config,
        ),
    }


def print_redteam_graph_profile(
    redteam_profile: dict[str, Any],
) -> None:
    print()
    print("Red-team graph profile")
    print(
        {
            "redteam_edge_count": redteam_profile["redteam_edge_count"],
            "redteam_participating_node_count": redteam_profile[
                "redteam_participating_node_count"
            ],
            "redteam_labeled_node_count": redteam_profile[
                "redteam_labeled_node_count"
            ],
        }
    )

    print("Red-team edge type counts:")
    print(redteam_profile["redteam_edge_type_counts"])

    print("Red-team event family counts:")
    print(redteam_profile["redteam_event_family_counts"])

    print("Red-team node type counts:")
    print(redteam_profile["redteam_node_type_counts"])

    print("Red-team edge sample:")
    print(redteam_profile["redteam_edge_sample"])


def build_paths(config: dict[str, Any]) -> dict[str, Path]:
    input_directory = Path(config["input"]["directory"])
    output_directory = Path(config["output"]["directory"])

    return {
        "graph_nodes": input_directory / config["input"]["graph_nodes_filename"],
        "graph_edges": input_directory / config["input"]["graph_edges_filename"],
        "markdown_report": output_directory
        / config["output"]["markdown_report_filename"],
        "json_report": output_directory
        / config["output"]["json_report_filename"],
        "top_nodes": output_directory / config["output"]["top_nodes_filename"],
    }


def validate_input_paths(paths: dict[str, Path]) -> None:
    required_inputs = [
        "graph_nodes",
        "graph_edges",
    ]

    for key in required_inputs:
        path = paths[key]

        if not path.exists():
            raise FileNotFoundError(f"Required input not found for {key}: {path}")

        if not path.is_file():
            raise FileNotFoundError(f"Required input is not a file for {key}: {path}")


def print_inspection_plan(
    config: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    print("LANL graph inspection plan")
    print("Graph type:", config["graph"]["type"])

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
            "markdown_report": str(paths["markdown_report"]),
            "json_report": str(paths["json_report"]),
            "top_nodes": str(paths["top_nodes"]),
        }
    )

    print("Inspection limits:")
    print(
        {
            "top_k_nodes": config["inspection"]["top_k_nodes"],
            "component_sample_size": config["inspection"]["component_sample_size"],
            "redteam_sample_size": config["inspection"]["redteam_sample_size"],
            "max_edges_for_visual_sample": config["inspection"][
                "max_edges_for_visual_sample"
            ],
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to LANL graph inspection config YAML file.",
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

    print_inspection_plan(
        config=config,
        paths=paths,
    )

    nodes, edges = read_graph_tables(paths)

    valid_edges, invalid_edges = split_valid_and_invalid_edges(
        nodes=nodes,
        edges=edges,
    )

    apply_invalid_edge_policy(
        invalid_edges=invalid_edges,
        config=config,
    )

    print_graph_table_summary(
        nodes=nodes,
        edges=edges,
        valid_edges=valid_edges,
        invalid_edges=invalid_edges,
    )

    graph = build_networkx_graph(
        nodes=nodes,
        valid_edges=valid_edges,
    )

    print_networkx_graph_summary(graph)

    basic_profile = build_basic_graph_profile(
        graph=graph,
        nodes=nodes,
        edges=edges,
        valid_edges=valid_edges,
        invalid_edges=invalid_edges,
    )

    print_basic_graph_profile(basic_profile)

    degree_profile, degree_table = build_degree_profile(
        graph=graph,
        nodes=nodes,
        config=config,
    )

    print_degree_profile(degree_profile)

    structural_profile = build_structural_graph_profile(
        graph=graph,
        nodes=nodes,
        config=config,
    )

    print_structural_graph_profile(structural_profile)

    redteam_profile = build_redteam_graph_profile(
        valid_edges=valid_edges,
        nodes=nodes,
        config=config,
    )

    print_redteam_graph_profile(redteam_profile)



if __name__ == "__main__":
    main()