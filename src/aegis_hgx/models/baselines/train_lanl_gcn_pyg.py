from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse

import torch
import yaml
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


CONFIG_PATH = "configs/lanl_gcn_pyg.yaml"


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
        "split",
        "model",
        "training",
        "evaluation",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    if "graph_path" not in config["input"]:
        raise ValueError("Missing input.graph_path in config.")

    required_output_keys = [
        "model_path",
        "metrics_path",
    ]

    for key in required_output_keys:
        if key not in config["output"]:
            raise ValueError(f"Missing output.{key} in config.")

    required_split_keys = [
        "train_ratio",
        "val_ratio",
        "test_ratio",
        "seed",
    ]

    for key in required_split_keys:
        if key not in config["split"]:
            raise ValueError(f"Missing split.{key} in config.")

    train_ratio = float(config["split"]["train_ratio"])
    val_ratio = float(config["split"]["val_ratio"])
    test_ratio = float(config["split"]["test_ratio"])

    ratio_sum = train_ratio + val_ratio + test_ratio

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            "split.train_ratio + split.val_ratio + split.test_ratio must equal 1.0."
        )

    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("All split ratios must be positive.")

    # These values control how we split nodes into train/validation/test groups.
    required_split_keys = [
        "train_ratio",
        "val_ratio",
        "test_ratio",
        "seed",
    ]

    for key in required_split_keys:
        if key not in config["split"]:
            raise ValueError(f"Missing split.{key} in config.")

    train_ratio = float(config["split"]["train_ratio"])
    val_ratio = float(config["split"]["val_ratio"])
    test_ratio = float(config["split"]["test_ratio"])

    # The three split ratios should cover 100% of the nodes.
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must add up to 1.0.")

def create_node_masks(
    data: Data,
    config: dict[str, Any],
) -> Data:
    # Total number of nodes in the graph.
    num_nodes = int(data.num_nodes)

    # Read split ratios from config.
    train_ratio = float(config["split"]["train_ratio"])
    val_ratio = float(config["split"]["val_ratio"])

    # Seed makes the split reproducible.
    seed = int(config["split"]["seed"])
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Start with all nodes excluded from all splits.
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Split each label class separately.
    # This keeps positive/negative label proportions more balanced across splits.
    label_values = sorted(data.y.unique().tolist())

    for label_value in label_values:
        # Find all nodes with this label.
        label_indices = torch.where(data.y == int(label_value))[0]

        # Shuffle only nodes from this label group.
        shuffled_label_indices = label_indices[
            torch.randperm(
                label_indices.numel(),
                generator=generator,
            )
        ]

        # Count how many nodes from this label go into train and validation.
        label_count = shuffled_label_indices.numel()
        train_count = int(label_count * train_ratio)
        val_count = int(label_count * val_ratio)

        # Slice this label group into train, validation, and test.
        train_indices = shuffled_label_indices[:train_count]
        val_indices = shuffled_label_indices[train_count : train_count + val_count]
        test_indices = shuffled_label_indices[train_count + val_count :]

        # Mark these node positions in the global masks.
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

    # Attach masks to the PyG Data object.
    # These masks align with rows of data.x and entries of data.y.
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

def print_node_mask_summary(data: Data) -> None:
    train_labels = data.y[data.train_mask]
    val_labels = data.y[data.val_mask]
    test_labels = data.y[data.test_mask]

    print()
    print("Node mask summary")
    print(
        {
            "train_nodes": int(data.train_mask.sum().item()),
            "val_nodes": int(data.val_mask.sum().item()),
            "test_nodes": int(data.test_mask.sum().item()),
            "train_positive_labels": int((train_labels == 1).sum().item()),
            "val_positive_labels": int((val_labels == 1).sum().item()),
            "test_positive_labels": int((test_labels == 1).sum().item()),
            "train_negative_labels": int((train_labels == 0).sum().item()),
            "val_negative_labels": int((val_labels == 0).sum().item()),
            "test_negative_labels": int((test_labels == 0).sum().item()),
        }
    )


def build_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        "graph": Path(config["input"]["graph_path"]),
        "model": Path(config["output"]["model_path"]),
        "metrics": Path(config["output"]["metrics_path"]),
    }


def validate_input_paths(paths: dict[str, Path]) -> None:
    graph_path = paths["graph"]

    if not graph_path.exists():
        raise FileNotFoundError(f"PyG graph artifact not found: {graph_path}")

    if not graph_path.is_file():
        raise FileNotFoundError(f"PyG graph artifact is not a file: {graph_path}")
    

def load_pyg_graph(graph_path: Path) -> Data:
    data = torch.load(
        graph_path,
        weights_only=False,
    )

    if not isinstance(data, Data):
        raise TypeError(f"Expected torch_geometric.data.Data, got {type(data)}")

    return data

def validate_graph_for_gcn(data: Data) -> None:
    data.validate(raise_on_error=True)

    required_attributes = [
        "x",
        "edge_index",
        "y",
        "num_nodes",
    ]

    for attribute_name in required_attributes:
        if not hasattr(data, attribute_name):
            raise ValueError(f"PyG data object is missing: {attribute_name}")

    if data.x.ndim != 2:
        raise ValueError("data.x must have shape [num_nodes, num_node_features].")

    if data.edge_index.ndim != 2:
        raise ValueError("data.edge_index must have shape [2, num_edges].")

    if data.edge_index.shape[0] != 2:
        raise ValueError("data.edge_index first dimension must be 2.")

    if data.y.ndim != 1:
        raise ValueError("data.y must have shape [num_nodes].")

    if data.x.shape[0] != data.num_nodes:
        raise ValueError("data.x rows must match data.num_nodes.")

    if data.y.shape[0] != data.num_nodes:
        raise ValueError("data.y length must match data.num_nodes.")

    if data.x.shape[1] == 0:
        raise ValueError("data.x must contain at least one node feature.")

    unique_labels = sorted(data.y.unique().tolist())

    if len(unique_labels) < 2:
        print(
            "Warning:",
            {
                "message": "Only one node label value found. Training may not be meaningful.",
                "label_values": unique_labels,
            },
        )

def print_graph_summary(data: Data) -> None:
    print()
    print("LANL PyG graph summary for GCN training")
    print(
        {
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.shape[1]),
            "x_shape": list(data.x.shape),
            "edge_index_shape": list(data.edge_index.shape),
            "y_shape": list(data.y.shape),
            "node_label_values": sorted(data.y.unique().tolist()),
            "x_dtype": str(data.x.dtype),
            "edge_index_dtype": str(data.edge_index.dtype),
            "y_dtype": str(data.y.dtype),
        }
    )


class LanlGCN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # First graph convolution layer.
        # It reads the original node features from data.x.
        # Shape: [num_nodes, input_channels] -> [num_nodes, hidden_channels]
        self.conv1 = GCNConv(
            input_channels,
            hidden_channels,
        )

        # Second graph convolution layer.
        # It converts hidden node embeddings into class logits.
        # Shape: [num_nodes, hidden_channels] -> [num_nodes, output_channels]
        self.conv2 = GCNConv(
            hidden_channels,
            output_channels,
        )

        # Dropout randomly zeroes part of the hidden representation during training.
        # This helps reduce overfitting.
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # x contains node features.
        # Shape: [num_nodes, input_channels]

        # edge_index contains graph connectivity.
        # Shape: [2, num_edges]

        # First message-passing step.
        # Each node receives information from its neighbors.
        hidden = self.conv1(
            x,
            edge_index,
        )

        # Nonlinear activation.
        # This lets the model learn more than a simple linear transformation.
        hidden = F.relu(hidden)

        # Apply dropout only during training.
        # During evaluation/inference, self.training is False, so dropout is disabled.
        hidden = F.dropout(
            hidden,
            p=self.dropout,
            training=self.training,
        )

        # Second message-passing step.
        # Produces raw class scores for each node.
        logits = self.conv2(
            hidden,
            edge_index,
        )

        # logits shape: [num_nodes, output_channels]
        return logits

def build_model(
    data: Data,
    config: dict[str, Any],
) -> LanlGCN:
    # Number of input features per node.
    # Example: if data.x shape is [24102, 8], input_channels = 8.
    input_channels = int(data.x.shape[1])

    # Number of output classes.
    # For labels 0 and 1, this should be 2.
    output_channels = int(data.y.max().item()) + 1

    # Hidden dimension comes from config.
    hidden_channels = int(config["model"]["hidden_channels"])

    # Dropout probability comes from config.
    dropout = float(config["model"]["dropout"])

    model = LanlGCN(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        output_channels=output_channels,
        dropout=dropout,
    )

    return model


def print_model_summary(
    model: LanlGCN,
    data: Data,
) -> None:
    # Count trainable model parameters.
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print("GCN model summary")
    print(
        {
            "model_class": model.__class__.__name__,
            "input_channels": int(data.x.shape[1]),
            "output_classes": int(data.y.max().item()) + 1,
            "trainable_parameters": trainable_parameters,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to LANL GCN config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    validate_config(config)

    paths = build_paths(config)
    validate_input_paths(paths)

    print()
    print("Config path:", args.config)
    print("Graph path:", paths["graph"])

    data = load_pyg_graph(paths["graph"])

    validate_graph_for_gcn(data)

    print_graph_summary(data)

    data = create_node_masks(
        data=data,
        config=config,
    )

    print_node_mask_summary(data)

    model = build_model(
        data=data,
        config=config,
    )

    print_model_summary(
        model=model,
        data=data,
    )


if __name__ == "__main__":
    main()



