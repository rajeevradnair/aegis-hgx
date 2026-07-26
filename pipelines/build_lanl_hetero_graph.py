from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
import yaml
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData


NODE_TYPES = (
    "user",
    "host",
    "process",
    "host_or_domain",
)

SUPERVISED_NODE_TYPES = (
    "user",
    "host",
)

FEATURE_NAMES = (
    "log_outgoing_event_count",
    "log_incoming_event_count",
    "log_unique_neighbor_count",
    "log_total_event_count",
    "log_activity_duration",
    "normalized_first_seen",
    "normalized_last_seen",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    return config


def remove_label_leakage_edges(
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """Remove explicit red-team ground-truth records from graph input."""

    required_columns = {
        "source_node_id",
        "destination_node_id",
        "source_entity_type",
        "destination_entity_type",
        "event_type",
        "timestamp",
        "label",
        "event_family",
        "edge_type",
        "event_result",
        "source_file",
    }

    missing = required_columns.difference(edges.columns)

    if missing:
        raise ValueError(
            f"Edge table is missing columns: {sorted(missing)}"
        )

    leakage_mask = (
        edges["label"].eq(1)
        | edges["event_family"].eq("redteam_ground_truth")
        | edges["event_type"].eq("redteam_activity")
        | edges["edge_type"]
        .astype(str)
        .str.startswith("redteam_ground_truth", na=False)
        | edges["event_result"].eq("confirmed_redteam")
        | edges["source_file"]
        .astype(str)
        .str.contains("redteam", case=False, na=False)
    )

    safe_edges = edges.loc[~leakage_mask].copy()

    remaining_leakage = (
        safe_edges["label"].eq(1)
        | safe_edges["event_family"].eq("redteam_ground_truth")
        | safe_edges["event_type"].eq("redteam_activity")
        | safe_edges["event_result"].eq("confirmed_redteam")
    )

    if remaining_leakage.any():
        raise ValueError(
            "Red-team ground-truth information remains after filtering."
        )

    return safe_edges


def create_split_masks(
    labels: np.ndarray,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create mutually exclusive node masks."""

    if not np.isclose(
        train_ratio + validation_ratio + test_ratio,
        1.0,
    ):
        raise ValueError(
            "Train, validation, and test ratios must sum to 1."
        )

    indices = np.arange(labels.shape[0])

    class_counts = np.bincount(
        labels.astype(np.int64),
        minlength=2,
    )

    use_stratification = (
        np.unique(labels).size == 2
        and class_counts.min() >= 3
    )

    stratify_labels = labels if use_stratification else None

    try:
        train_indices, temporary_indices = train_test_split(
            indices,
            test_size=validation_ratio + test_ratio,
            random_state=seed,
            stratify=stratify_labels,
        )

        temporary_labels = labels[temporary_indices]

        second_stratification = (
            temporary_labels
            if use_stratification
            and np.unique(temporary_labels).size == 2
            else None
        )

        relative_test_ratio = (
            test_ratio
            / (validation_ratio + test_ratio)
        )

        validation_indices, test_indices = train_test_split(
            temporary_indices,
            test_size=relative_test_ratio,
            random_state=seed,
            stratify=second_stratification,
        )

    except ValueError:
        generator = np.random.default_rng(seed)
        shuffled = generator.permutation(indices)

        train_end = int(train_ratio * len(shuffled))
        validation_end = train_end + int(
            validation_ratio * len(shuffled)
        )

        train_indices = shuffled[:train_end]
        validation_indices = shuffled[
            train_end:validation_end
        ]
        test_indices = shuffled[validation_end:]

    train_mask = torch.zeros(
        labels.shape[0],
        dtype=torch.bool,
    )
    validation_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)

    train_mask[torch.as_tensor(train_indices)] = True
    validation_mask[
        torch.as_tensor(validation_indices)
    ] = True
    test_mask[torch.as_tensor(test_indices)] = True

    if bool(
        (train_mask & validation_mask).any()
        or (train_mask & test_mask).any()
        or (validation_mask & test_mask).any()
    ):
        raise ValueError("Node split masks overlap.")

    if not torch.all(
        train_mask | validation_mask | test_mask
    ):
        raise ValueError(
            "Some nodes were not assigned to a split."
        )

    return train_mask, validation_mask, test_mask


def build_node_statistics(
    nodes: pd.DataFrame,
    safe_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Build features only from ordinary telemetry edges."""

    outgoing_count = safe_edges.groupby(
        "source_node_id"
    ).size()

    incoming_count = safe_edges.groupby(
        "destination_node_id"
    ).size()

    outgoing_neighbors = safe_edges.groupby(
        "source_node_id"
    )["destination_node_id"].nunique()

    incoming_neighbors = safe_edges.groupby(
        "destination_node_id"
    )["source_node_id"].nunique()

    outgoing_first = safe_edges.groupby(
        "source_node_id"
    )["timestamp"].min()

    incoming_first = safe_edges.groupby(
        "destination_node_id"
    )["timestamp"].min()

    outgoing_last = safe_edges.groupby(
        "source_node_id"
    )["timestamp"].max()

    incoming_last = safe_edges.groupby(
        "destination_node_id"
    )["timestamp"].max()

    first_seen = pd.concat(
        [
            outgoing_first.rename("outgoing"),
            incoming_first.rename("incoming"),
        ],
        axis=1,
    ).min(axis=1)

    last_seen = pd.concat(
        [
            outgoing_last.rename("outgoing"),
            incoming_last.rename("incoming"),
        ],
        axis=1,
    ).max(axis=1)

    statistics = nodes[
        [
            "node_id",
            "entity_type",
            "entity_name",
            "label",
        ]
    ].copy()

    statistics["outgoing_count"] = (
        statistics["node_id"]
        .map(outgoing_count)
        .fillna(0)
        .astype(np.float32)
    )

    statistics["incoming_count"] = (
        statistics["node_id"]
        .map(incoming_count)
        .fillna(0)
        .astype(np.float32)
    )

    statistics["unique_neighbor_count"] = (
        statistics["node_id"]
        .map(outgoing_neighbors)
        .fillna(0)
        .astype(np.float32)
        +
        statistics["node_id"]
        .map(incoming_neighbors)
        .fillna(0)
        .astype(np.float32)
    )

    statistics["total_event_count"] = (
        statistics["outgoing_count"]
        + statistics["incoming_count"]
    )

    statistics["first_seen"] = (
        statistics["node_id"]
        .map(first_seen)
        .fillna(0)
        .astype(np.float32)
    )

    statistics["last_seen"] = (
        statistics["node_id"]
        .map(last_seen)
        .fillna(0)
        .astype(np.float32)
    )

    statistics["activity_duration"] = np.maximum(
        statistics["last_seen"]
        - statistics["first_seen"],
        0,
    )

    return statistics


def build_feature_tensor(
    typed_nodes: pd.DataFrame,
    maximum_timestamp: float,
) -> torch.Tensor:
    """Create seven leakage-safe numeric node features."""

    maximum_timestamp = max(maximum_timestamp, 1.0)

    feature_matrix = np.column_stack(
        [
            np.log1p(
                typed_nodes["outgoing_count"].to_numpy()
            ),
            np.log1p(
                typed_nodes["incoming_count"].to_numpy()
            ),
            np.log1p(
                typed_nodes[
                    "unique_neighbor_count"
                ].to_numpy()
            ),
            np.log1p(
                typed_nodes["total_event_count"].to_numpy()
            ),
            np.log1p(
                typed_nodes["activity_duration"].to_numpy()
            ),
            (
                typed_nodes["first_seen"].to_numpy()
                / maximum_timestamp
            ),
            (
                typed_nodes["last_seen"].to_numpy()
                / maximum_timestamp
            ),
        ]
    ).astype(np.float32)

    if not np.isfinite(feature_matrix).all():
        raise ValueError(
            "Node feature matrix contains invalid values."
        )

    # Shape: [number_of_nodes_of_this_type, 7]
    return torch.from_numpy(feature_matrix)


def build_hetero_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> HeteroData:
    """Build the static leakage-safe LANL heterogeneous graph."""

    required_node_columns = {
        "node_id",
        "entity_type",
        "entity_name",
        "label",
    }

    missing_node_columns = required_node_columns.difference(
        nodes.columns
    )

    if missing_node_columns:
        raise ValueError(
            "Node table is missing columns: "
            f"{sorted(missing_node_columns)}"
        )

    if nodes["node_id"].duplicated().any():
        raise ValueError("Node IDs must be unique.")

    unknown_node_types = set(
        nodes["entity_type"].unique()
    ).difference(NODE_TYPES)

    if unknown_node_types:
        raise ValueError(
            f"Unknown node types: {sorted(unknown_node_types)}"
        )

    safe_edges = remove_label_leakage_edges(edges)

    node_type_by_id = nodes.set_index(
        "node_id"
    )["entity_type"]

    actual_source_types = safe_edges[
        "source_node_id"
    ].map(node_type_by_id)

    actual_destination_types = safe_edges[
        "destination_node_id"
    ].map(node_type_by_id)

    if (
        actual_source_types.isna().any()
        or actual_destination_types.isna().any()
    ):
        raise ValueError(
            "Some edges reference node IDs absent from graph_nodes."
        )

    if not actual_source_types.equals(
        safe_edges["source_entity_type"]
    ):
        raise ValueError(
            "Edge source types disagree with the node table."
        )

    if not actual_destination_types.equals(
        safe_edges["destination_entity_type"]
    ):
        raise ValueError(
            "Edge destination types disagree with the node table."
        )

    node_statistics = build_node_statistics(
        nodes=nodes,
        safe_edges=safe_edges,
    )

    maximum_timestamp = float(
        safe_edges["timestamp"].max()
    )

    data = HeteroData()

    global_to_local: dict[int, int] = {}

    for node_type in NODE_TYPES:
        typed_nodes = (
            node_statistics.loc[
                node_statistics["entity_type"].eq(node_type)
            ]
            .sort_values("node_id")
            .reset_index(drop=True)
        )

        local_indices = np.arange(
            len(typed_nodes),
            dtype=np.int64,
        )

        global_to_local.update(
            {
                int(global_id): int(local_id)
                for global_id, local_id
                in zip(
                    typed_nodes["node_id"],
                    local_indices,
                    strict=True,
                )
            }
        )

        data[node_type].x = build_feature_tensor(
            typed_nodes=typed_nodes,
            maximum_timestamp=maximum_timestamp,
        )

        data[node_type].y = torch.as_tensor(
            typed_nodes["label"].to_numpy(),
            dtype=torch.float32,
        )

        data[node_type].node_id = torch.as_tensor(
            typed_nodes["node_id"].to_numpy(),
            dtype=torch.long,
        )

        data[node_type].entity_name = (
            typed_nodes["entity_name"]
            .astype(str)
            .tolist()
        )

        data[node_type].feature_names = list(
            FEATURE_NAMES
        )

        if node_type in SUPERVISED_NODE_TYPES:
            (
                train_mask,
                validation_mask,
                test_mask,
            ) = create_split_masks(
                labels=typed_nodes[
                    "label"
                ].to_numpy(dtype=np.int64),
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )

            data[node_type].train_mask = train_mask
            data[node_type].val_mask = validation_mask
            data[node_type].test_mask = test_mask

    safe_edges["source_local"] = (
        safe_edges["source_node_id"]
        .map(global_to_local)
    )

    safe_edges["destination_local"] = (
        safe_edges["destination_node_id"]
        .map(global_to_local)
    )

    if (
        safe_edges["source_local"].isna().any()
        or safe_edges["destination_local"].isna().any()
    ):
        raise ValueError(
            "Failed to map global node IDs to typed local IDs."
        )

    safe_edges["source_local"] = (
        safe_edges["source_local"].astype(np.int64)
    )

    safe_edges["destination_local"] = (
        safe_edges["destination_local"].astype(np.int64)
    )

    relation_columns = [
        "source_entity_type",
        "event_type",
        "destination_entity_type",
    ]

    for (
        source_type,
        relation_name,
        destination_type,
    ), relation_edges in safe_edges.groupby(
        relation_columns,
        sort=True,
    ):
        relation_name = str(relation_name)

        if "redteam" in relation_name.lower():
            raise ValueError(
                "A red-team relation reached graph construction."
            )

        # Repeated events become one structural relation.
        unique_pairs = relation_edges[
            [
                "source_local",
                "destination_local",
            ]
        ].drop_duplicates()

        edge_index = torch.as_tensor(
            unique_pairs.to_numpy().T,
            dtype=torch.long,
        )

        # Shape: [2, number_of_unique_relation_edges]
        data[
            str(source_type),
            relation_name,
            str(destination_type),
        ].edge_index = edge_index

    # Keep reverse relations separate, such as rev_auth_logon.
    data = T.ToUndirected(
        merge=False,
    )(data)

    for edge_type in data.edge_types:
        if "redteam" in edge_type[1].lower():
            raise ValueError(
                f"Leaking relation found: {edge_type}"
            )

        edge_index = data[edge_type].edge_index

        if edge_index.dtype != torch.long:
            raise ValueError(
                f"{edge_type} edge_index must be torch.long."
            )

        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"{edge_type} edge_index must have shape [2, E]."
            )

    return data


def print_summary(
    data: HeteroData,
    original_edge_count: int,
    safe_edge_count: int,
) -> None:
    print("\nLANL heterogeneous graph")
    print("------------------------")
    print(f"Original event rows: {original_edge_count:,}")
    print(f"Leakage-safe event rows: {safe_edge_count:,}")
    print(
        "Removed red-team rows:",
        f"{original_edge_count - safe_edge_count:,}",
    )

    for node_type in data.node_types:
        store = data[node_type]

        print(
            f"{node_type}: "
            f"nodes={store.num_nodes:,}, "
            f"features={store.x.size(1)}, "
            f"positives={int(store.y.sum().item()):,}"
        )

        if node_type in SUPERVISED_NODE_TYPES:
            print(
                "  split: "
                f"train={int(store.train_mask.sum()):,}, "
                f"validation={int(store.val_mask.sum()):,}, "
                f"test={int(store.test_mask.sum()):,}"
            )

    print(f"Relations: {len(data.edge_types)}")

    for edge_type in data.edge_types:
        print(
            f"  {edge_type}: "
            f"{data[edge_type].edge_index.size(1):,}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lanl_hetero_gnn.yaml"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    nodes_path = Path(
        config["input"]["nodes_path"]
    )
    edges_path = Path(
        config["input"]["edges_path"]
    )
    graph_path = Path(
        config["input"]["graph_path"]
    )

    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)

    safe_edge_count = len(
        remove_label_leakage_edges(edges)
    )

    data = build_hetero_graph(
        nodes=nodes,
        edges=edges,
        train_ratio=float(
            config["split"]["train_ratio"]
        ),
        validation_ratio=float(
            config["split"]["validation_ratio"]
        ),
        test_ratio=float(
            config["split"]["test_ratio"]
        ),
        seed=int(config["split"]["seed"]),
    )

    graph_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(data, graph_path)

    print_summary(
        data=data,
        original_edge_count=len(edges),
        safe_edge_count=safe_edge_count,
    )

    print(f"\nSaved graph: {graph_path}")


if __name__ == "__main__":
    main()