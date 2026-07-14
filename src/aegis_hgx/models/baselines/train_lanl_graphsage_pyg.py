from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from torch_geometric.data import Data


CONFIG_PATH = "configs/lanl_graphsage_pyg.yaml"


def load_config(config_path: str) -> dict[str, Any]:
    # Read the YAML config from disk.
    #
    # The config tells this trainer:
    #   where the graph artifact lives
    #   where outputs should be saved
    #   what split/model/training settings to use
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    return yaml.safe_load(path.read_text())


def validate_config(config: dict[str, Any]) -> None:
    # Keep validation simple and readable.
    # We only check the sections needed for this skeleton.
    required_sections = [
        "input",
        "output",
        "split",
        "model",
        "training",
        "evaluation",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing config section: {section}")

    if "graph_path" not in config["input"]:
        raise ValueError("Missing config value: input.graph_path")


def load_pyg_graph(graph_path: str) -> Data:
    # Load the PyTorch Geometric Data object created earlier.
    #
    # weights_only=False is needed because this file stores a full PyG Data object,
    # not just plain tensor weights.
    path = Path(graph_path)

    if not path.exists():
        raise FileNotFoundError(f"PyG graph artifact not found: {path}")

    data = torch.load(
        path,
        weights_only=False,
    )

    if not isinstance(data, Data):
        raise TypeError(f"Expected PyG Data object, got: {type(data)}")

    return data


def validate_graph_for_graphsage(data: Data) -> None:
    # GraphSAGE needs node features.
    if not hasattr(data, "x") or data.x is None:
        raise ValueError("PyG graph is missing node features: data.x")

    # GraphSAGE needs graph connectivity.
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("PyG graph is missing graph edges: data.edge_index")

    # Supervised node classification needs node labels.
    if not hasattr(data, "y") or data.y is None:
        raise ValueError("PyG graph is missing node labels: data.y")

    # data.x should be a matrix:
    #   [num_nodes, num_node_features]
    if data.x.dim() != 2:
        raise ValueError(f"Expected data.x to be 2D, got shape: {data.x.shape}")

    # data.edge_index should be:
    #   [2, num_edges]
    if data.edge_index.dim() != 2 or data.edge_index.shape[0] != 2:
        raise ValueError(
            f"Expected data.edge_index shape [2, num_edges], got: {data.edge_index.shape}"
        )

    # data.y should have one label per node:
    #   [num_nodes]
    if data.y.dim() != 1:
        raise ValueError(f"Expected data.y to be 1D, got shape: {data.y.shape}")

    num_nodes = int(data.num_nodes)

    if data.x.shape[0] != num_nodes:
        raise ValueError(
            f"data.x row count {data.x.shape[0]} does not match num_nodes {num_nodes}"
        )

    if data.y.shape[0] != num_nodes:
        raise ValueError(
            f"data.y length {data.y.shape[0]} does not match num_nodes {num_nodes}"
        )


def print_graph_summary(data: Data) -> None:
    # Print the important tensors we need before modeling.
    print()
    print("LANL PyG graph summary for GraphSAGE training")
    print(
        {
            "num_nodes": int(data.num_nodes),
            "x_shape": list(data.x.shape),
            "edge_index_shape": list(data.edge_index.shape),
            "y_shape": list(data.y.shape),
            "num_node_features": int(data.x.shape[1]),
            "label_values": sorted(data.y.unique().tolist()),
            "positive_labels": int((data.y == 1).sum().item()),
            "negative_labels": int((data.y == 0).sum().item()),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GraphSAGE baseline on the LANL PyG graph."
    )

    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help="Path to GraphSAGE training config.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    validate_config(config)

    data = load_pyg_graph(config["input"]["graph_path"])
    validate_graph_for_graphsage(data)

    print_graph_summary(data)


if __name__ == "__main__":
    main()