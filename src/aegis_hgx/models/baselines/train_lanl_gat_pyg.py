from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import mlflow
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import yaml


CONFIG_PATH = "configs/lanl_gat_pyg.yaml"


def make_json_safe(value: Any) -> Any:
    """Convert common experiment values into JSON-safe Python objects."""
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

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return make_json_safe(value.item())

        return value.detach().cpu().tolist()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML experiment configuration."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Config path is not a file: {path}")

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")

    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate all configuration fields required by the GAT baseline."""
    required_sections = [
        "input",
        "output",
        "split",
        "model",
        "training",
        "evaluation",
        "experiment_tracking",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

        if not isinstance(config[section], dict):
            raise ValueError(f"Config section {section} must be a mapping.")

    required_keys_by_section = {
        "input": [
            "graph_path",
        ],
        "output": [
            "model_path",
            "metrics_path",
        ],
        "split": [
            "train_ratio",
            "val_ratio",
            "test_ratio",
            "seed",
        ],
        "model": [
            "hidden_channels",
            "heads",
            "dropout",
            "attention_dropout",
            "negative_slope",
            "add_self_loops",
        ],
        "training": [
            "epochs",
            "learning_rate",
            "weight_decay",
        ],
        "evaluation": [
            "positive_label",
        ],
        "experiment_tracking": [
            "experiment_name",
            "uri",
            "artifact_root",
        ],
    }

    for section, required_keys in required_keys_by_section.items():
        for key in required_keys:
            if key not in config[section]:
                raise ValueError(f"Missing {section}.{key} in config.")

    train_ratio = float(config["split"]["train_ratio"])
    val_ratio = float(config["split"]["val_ratio"])
    test_ratio = float(config["split"]["test_ratio"])

    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("All split ratios must be positive.")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError(
            "split.train_ratio + split.val_ratio + split.test_ratio must equal 1.0."
        )

    hidden_channels = int(config["model"]["hidden_channels"])
    heads = int(config["model"]["heads"])

    if hidden_channels <= 0:
        raise ValueError("model.hidden_channels must be positive.")

    if heads <= 0:
        raise ValueError("model.heads must be positive.")

    dropout = float(config["model"]["dropout"])
    attention_dropout = float(config["model"]["attention_dropout"])

    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout must be in the range [0.0, 1.0).")

    if not 0.0 <= attention_dropout < 1.0:
        raise ValueError(
            "model.attention_dropout must be in the range [0.0, 1.0)."
        )

    negative_slope = float(config["model"]["negative_slope"])

    if negative_slope <= 0.0:
        raise ValueError("model.negative_slope must be positive.")

    epochs = int(config["training"]["epochs"])
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])

    if epochs <= 0:
        raise ValueError("training.epochs must be positive.")

    if learning_rate <= 0.0:
        raise ValueError("training.learning_rate must be positive.")

    if weight_decay < 0.0:
        raise ValueError("training.weight_decay cannot be negative.")

    positive_label = int(config["evaluation"]["positive_label"])

    if positive_label < 0:
        raise ValueError("evaluation.positive_label cannot be negative.")


def build_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Build filesystem paths used by the experiment."""
    return {
        "graph": Path(config["input"]["graph_path"]),
        "model": Path(config["output"]["model_path"]),
        "metrics": Path(config["output"]["metrics_path"]),
    }


def validate_input_paths(paths: dict[str, Path]) -> None:
    """Ensure the serialized PyG graph exists before training begins."""
    graph_path = paths["graph"]

    if not graph_path.exists():
        raise FileNotFoundError(
            f"PyG graph artifact not found: {graph_path}"
        )

    if not graph_path.is_file():
        raise FileNotFoundError(
            f"PyG graph artifact is not a file: {graph_path}"
        )


def load_pyg_graph(graph_path: Path) -> Data:
    """Load the homogeneous LANL PyG graph artifact."""
    data = torch.load(
        graph_path,
        weights_only=False,
    )

    if not isinstance(data, Data):
        raise TypeError(
            "Expected torch_geometric.data.Data, "
            f"got {type(data)}"
        )

    return data


def validate_graph_for_gat(data: Data) -> None:
    """Validate graph tensors required by the GAT node classifier."""
    data.validate(raise_on_error=True)

    required_attributes = [
        "x",
        "edge_index",
        "y",
        "num_nodes",
    ]

    for attribute_name in required_attributes:
        if not hasattr(data, attribute_name):
            raise ValueError(
                f"PyG data object is missing: {attribute_name}"
            )

    if data.x is None:
        raise ValueError("data.x cannot be None.")

    if data.edge_index is None:
        raise ValueError("data.edge_index cannot be None.")

    if data.y is None:
        raise ValueError("data.y cannot be None.")

    if data.x.ndim != 2:
        raise ValueError(
            "data.x must have shape "
            "[num_nodes, num_node_features]."
        )

    if data.edge_index.ndim != 2:
        raise ValueError(
            "data.edge_index must have shape [2, num_edges]."
        )

    if data.edge_index.shape[0] != 2:
        raise ValueError(
            "data.edge_index first dimension must be 2."
        )

    if data.y.ndim != 1:
        raise ValueError(
            "data.y must have shape [num_nodes]."
        )

    if data.x.shape[0] != data.num_nodes:
        raise ValueError(
            "data.x rows must match data.num_nodes."
        )

    if data.y.shape[0] != data.num_nodes:
        raise ValueError(
            "data.y length must match data.num_nodes."
        )

    if data.x.shape[1] == 0:
        raise ValueError(
            "data.x must contain at least one node feature."
        )

    if data.edge_index.dtype != torch.long:
        raise TypeError(
            "data.edge_index must use torch.long node indices."
        )

    if not data.x.dtype.is_floating_point:
        raise TypeError(
            "data.x must use a floating-point dtype."
        )

    if data.y.dtype != torch.long:
        raise TypeError(
            "data.y must use torch.long class labels."
        )

    if not bool(torch.isfinite(data.x).all().item()):
        raise ValueError(
            "data.x contains NaN or infinite values."
        )

    if int(data.edge_index.shape[1]) == 0:
        raise ValueError(
            "data.edge_index must contain at least one edge."
        )

    if int(data.edge_index.min().item()) < 0:
        raise ValueError(
            "data.edge_index contains a negative node index."
        )

    if int(data.edge_index.max().item()) >= int(data.num_nodes):
        raise ValueError(
            "data.edge_index contains a node index outside data.num_nodes."
        )

    unique_labels = sorted(
        int(label)
        for label in data.y.unique().tolist()
    )

    if any(label < 0 for label in unique_labels):
        raise ValueError(
            "data.y labels must be non-negative integers."
        )

    if len(unique_labels) < 2:
        print(
            "Warning:",
            {
                "message": (
                    "Only one node-label value was found. "
                    "Supervised classification will not be meaningful."
                ),
                "label_values": unique_labels,
            },
        )


def print_graph_summary(data: Data) -> None:
    """Print the graph contract consumed by the GAT model."""
    print()
    print("LANL PyG graph summary for GAT training")
    print(
        {
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.shape[1]),
            "x_shape": list(data.x.shape),
            "edge_index_shape": list(data.edge_index.shape),
            "y_shape": list(data.y.shape),
            "node_label_values": sorted(
                int(label)
                for label in data.y.unique().tolist()
            ),
            "x_dtype": str(data.x.dtype),
            "edge_index_dtype": str(data.edge_index.dtype),
            "y_dtype": str(data.y.dtype),
        }
    )


def create_node_masks(
    data: Data,
    config: dict[str, Any],
) -> Data:
    """Create reproducible stratified train, validation, and test masks."""
    num_nodes = int(data.num_nodes)

    train_ratio = float(config["split"]["train_ratio"])
    val_ratio = float(config["split"]["val_ratio"])

    seed = int(config["split"]["seed"])
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )
    val_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )
    test_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )

    label_values = sorted(
        int(label)
        for label in data.y.unique().tolist()
    )

    for label_value in label_values:
        label_indices = torch.where(
            data.y == label_value
        )[0]

        shuffled_label_indices = label_indices[
            torch.randperm(
                label_indices.numel(),
                generator=generator,
            )
        ]

        label_count = int(
            shuffled_label_indices.numel()
        )

        train_count = int(
            label_count * train_ratio
        )
        val_count = int(
            label_count * val_ratio
        )

        train_indices = shuffled_label_indices[
            :train_count
        ]
        val_indices = shuffled_label_indices[
            train_count : train_count + val_count
        ]
        test_indices = shuffled_label_indices[
            train_count + val_count :
        ]

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def validate_node_masks(data: Data) -> None:
    """Check mask shapes, coverage, overlap, and split non-emptiness."""
    required_masks = [
        "train_mask",
        "val_mask",
        "test_mask",
    ]

    for mask_name in required_masks:
        if not hasattr(data, mask_name):
            raise ValueError(
                f"PyG data object is missing {mask_name}."
            )

        mask = getattr(data, mask_name)

        if mask.ndim != 1:
            raise ValueError(
                f"{mask_name} must have shape [num_nodes]."
            )

        if mask.shape[0] != data.num_nodes:
            raise ValueError(
                f"{mask_name} length must match data.num_nodes."
            )

        if mask.dtype != torch.bool:
            raise TypeError(
                f"{mask_name} must use torch.bool."
            )

        if int(mask.sum().item()) == 0:
            raise ValueError(
                f"{mask_name} does not contain any nodes."
            )

    overlap = (
        data.train_mask.to(torch.int8)
        + data.val_mask.to(torch.int8)
        + data.test_mask.to(torch.int8)
    )

    if bool((overlap > 1).any().item()):
        raise ValueError(
            "Train, validation, and test masks overlap."
        )

    if bool((overlap == 0).any().item()):
        raise ValueError(
            "At least one node is not assigned to any split."
        )

    all_labels = sorted(
        int(label)
        for label in data.y.unique().tolist()
    )

    for mask_name in required_masks:
        split_labels = sorted(
            int(label)
            for label in data.y[
                getattr(data, mask_name)
            ].unique().tolist()
        )

        if split_labels != all_labels:
            print(
                "Warning:",
                {
                    "message": (
                        f"{mask_name} does not contain every graph label."
                    ),
                    "all_labels": all_labels,
                    "split_labels": split_labels,
                },
            )


def print_node_mask_summary(data: Data) -> None:
    """Print class counts for every node split."""
    train_labels = data.y[data.train_mask]
    val_labels = data.y[data.val_mask]
    test_labels = data.y[data.test_mask]

    print()
    print("Node mask summary")
    print(
        {
            "train_nodes": int(
                data.train_mask.sum().item()
            ),
            "val_nodes": int(
                data.val_mask.sum().item()
            ),
            "test_nodes": int(
                data.test_mask.sum().item()
            ),
            "train_positive_labels": int(
                (train_labels == 1).sum().item()
            ),
            "val_positive_labels": int(
                (val_labels == 1).sum().item()
            ),
            "test_positive_labels": int(
                (test_labels == 1).sum().item()
            ),
            "train_negative_labels": int(
                (train_labels == 0).sum().item()
            ),
            "val_negative_labels": int(
                (val_labels == 0).sum().item()
            ),
            "test_negative_labels": int(
                (test_labels == 0).sum().item()
            ),
        }
    )


class LanlGAT(nn.Module):
    """Two-layer multi-head GAT for LANL node classification."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        heads: int,
        dropout: float,
        attention_dropout: float,
        negative_slope: float,
        add_self_loops: bool,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.heads = heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops

        self.conv1 = GATConv(
            in_channels=input_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            negative_slope=negative_slope,
            dropout=attention_dropout,
            add_self_loops=add_self_loops,
        )

        concatenated_hidden_channels = hidden_channels * heads

        self.conv2 = GATConv(
            in_channels=concatenated_hidden_channels,
            out_channels=output_channels,
            heads=1,
            concat=False,
            negative_slope=negative_slope,
            dropout=attention_dropout,
            add_self_loops=add_self_loops,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # Shape: [num_nodes, input_channels]
        hidden = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # Shape:
        # [num_nodes, input_channels]
        # ->
        # [num_nodes, hidden_channels * heads]
        hidden = self.conv1(
            hidden,
            edge_index,
        )

        hidden = F.elu(hidden)

        hidden = F.dropout(
            hidden,
            p=self.dropout,
            training=self.training,
        )

        # Shape:
        # [num_nodes, hidden_channels * heads]
        # ->
        # [num_nodes, output_channels]
        logits = self.conv2(
            hidden,
            edge_index,
        )

        return logits


def build_model(
    data: Data,
    config: dict[str, Any],
) -> LanlGAT:
    """Create the GAT model from graph and configuration metadata."""
    input_channels = int(data.x.shape[1])
    output_channels = int(data.y.max().item()) + 1

    return LanlGAT(
        input_channels=input_channels,
        hidden_channels=int(config["model"]["hidden_channels"]),
        output_channels=output_channels,
        heads=int(config["model"]["heads"]),
        dropout=float(config["model"]["dropout"]),
        attention_dropout=float(config["model"]["attention_dropout"]),
        negative_slope=float(config["model"]["negative_slope"]),
        add_self_loops=bool(config["model"]["add_self_loops"]),
    )


def print_model_summary(
    model: LanlGAT,
    data: Data,
    device: torch.device,
) -> None:
    """Print architecture widths and trainable parameter count."""
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    concatenated_hidden_width = (
        model.hidden_channels * model.heads
    )

    print()
    print("GAT model summary")
    print(
        {
            "model_class": model.__class__.__name__,
            "device": str(device),
            "input_channels": int(data.x.shape[1]),
            "hidden_channels_per_head": model.hidden_channels,
            "attention_heads": model.heads,
            "concatenated_hidden_width": concatenated_hidden_width,
            "output_classes": int(data.y.max().item()) + 1,
            "feature_dropout": model.dropout,
            "attention_dropout": model.attention_dropout,
            "negative_slope": model.negative_slope,
            "add_self_loops": model.add_self_loops,
            "trainable_parameters": trainable_parameters,
        }
    )


def run_forward_pass_smoke_check(
    model: LanlGAT,
    data: Data,
) -> torch.Tensor:
    """Run one evaluation forward pass and validate its output contract."""
    model.eval()

    with torch.no_grad():
        logits = model(
            data.x,
            data.edge_index,
        )

    expected_node_count = int(data.num_nodes)
    expected_class_count = int(data.y.max().item()) + 1

    if logits.ndim != 2:
        raise ValueError(
            "GAT logits must have shape [num_nodes, num_classes]."
        )

    if logits.shape[0] != expected_node_count:
        raise ValueError(
            "GAT logits row count must match the number of graph nodes."
        )

    if logits.shape[1] != expected_class_count:
        raise ValueError(
            "GAT logits column count must match the number of node classes."
        )

    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError(
            "GAT logits contain NaN or infinite values."
        )

    return logits


def print_forward_pass_summary(
    logits: torch.Tensor,
    data: Data,
    positive_label: int,
) -> None:
    """Print probability statistics from the smoke-check logits."""
    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    if positive_label >= probabilities.shape[1]:
        raise ValueError(
            "evaluation.positive_label is outside the model output range."
        )

    positive_class_probabilities = probabilities[:, positive_label]

    print()
    print("Forward-pass smoke check summary")
    print(
        {
            "logits_shape": list(logits.shape),
            "expected_nodes": int(data.num_nodes),
            "expected_classes": int(data.y.max().item()) + 1,
            "logits_are_finite": bool(
                torch.isfinite(logits).all().item()
            ),
            "positive_probability_min": float(
                positive_class_probabilities.min().item()
            ),
            "positive_probability_max": float(
                positive_class_probabilities.max().item()
            ),
            "positive_probability_mean": float(
                positive_class_probabilities.mean().item()
            ),
        }
    )


def build_class_weights(data: Data) -> torch.Tensor:
    """Build inverse-frequency class weights from training labels only."""
    train_labels = data.y[data.train_mask]
    num_classes = int(data.y.max().item()) + 1

    class_counts = torch.bincount(
        train_labels,
        minlength=num_classes,
    ).float()

    if bool((class_counts == 0).any().item()):
        missing_classes = torch.where(
            class_counts == 0
        )[0].tolist()

        raise ValueError(
            "At least one class is absent from the training split. "
            f"Missing class indices: {missing_classes}"
        )

    return train_labels.numel() / (
        num_classes * class_counts
    )


def build_optimizer(
    model: LanlGAT,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    """Create the Adam optimizer for the GAT baseline."""
    return torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )


def train_model(
    model: LanlGAT,
    data: Data,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    config: dict[str, Any],
) -> tuple[
    list[dict[str, float]],
    dict[str, torch.Tensor],
]:
    """Train the model and retain the lowest-validation-loss checkpoint."""
    epochs = int(config["training"]["epochs"])

    training_history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_model_state: dict[str, torch.Tensor] = {}

    for epoch in range(1, epochs + 1):
        # ------------------------------------------------------------
        # TRAINING PHASE
        # ------------------------------------------------------------
        model.train()
        optimizer.zero_grad()

        logits = model(
            data.x,
            data.edge_index,
        )

        train_logits = logits[data.train_mask]
        train_labels = data.y[data.train_mask]

        train_loss = F.cross_entropy(
            train_logits,
            train_labels,
            weight=class_weights,
        )

        train_loss.backward()
        optimizer.step()

        train_predictions = train_logits.argmax(dim=1)
        train_accuracy = (
            train_predictions == train_labels
        ).float().mean().item()

        # ------------------------------------------------------------
        # VALIDATION PHASE
        # ------------------------------------------------------------
        model.eval()

        with torch.no_grad():
            val_logits_all_nodes = model(
                data.x,
                data.edge_index,
            )

            val_logits = val_logits_all_nodes[data.val_mask]
            val_labels = data.y[data.val_mask]

            val_loss = F.cross_entropy(
                val_logits,
                val_labels,
                weight=class_weights,
            )

            val_predictions = val_logits.argmax(dim=1)
            val_accuracy = (
                val_predictions == val_labels
            ).float().mean().item()

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(train_loss.item()),
            "train_accuracy": float(train_accuracy),
            "val_loss": float(val_loss.item()),
            "val_accuracy": float(val_accuracy),
        }

        training_history.append(epoch_record)

        current_val_loss = float(val_loss.item())

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss

            best_model_state = {
                parameter_name: parameter_tensor
                .detach()
                .cpu()
                .clone()
                for (
                    parameter_name,
                    parameter_tensor,
                ) in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch == epochs
            or epoch % 10 == 0
        ):
            print(epoch_record)

    if not best_model_state:
        raise RuntimeError(
            "Training did not produce a best model checkpoint."
        )

    return training_history, best_model_state


def print_training_summary(
    training_history: list[dict[str, float]],
) -> None:
    """Print the first, final, and best-validation epochs."""
    first_epoch = training_history[0]
    final_epoch = training_history[-1]

    best_epoch = min(
        training_history,
        key=lambda record: record["val_loss"],
    )

    print()
    print("GAT training summary")
    print(
        {
            "first_epoch": first_epoch,
            "final_epoch": final_epoch,
            "best_val_loss_epoch": best_epoch,
        }
    )


def evaluate_test_set(
    model: LanlGAT,
    data: Data,
    best_model_state: dict[str, torch.Tensor],
    positive_label: int,
) -> dict[str, float | int]:
    """Evaluate the best validation checkpoint on test nodes only."""
    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        logits = model(
            data.x,
            data.edge_index,
        )

        test_logits = logits[data.test_mask]
        test_labels = data.y[data.test_mask]

        probabilities = torch.softmax(
            test_logits,
            dim=1,
        )

        if positive_label >= probabilities.shape[1]:
            raise ValueError(
                "evaluation.positive_label is outside the model output range."
            )

        positive_probabilities = probabilities[:, positive_label]
        predicted_labels = test_logits.argmax(dim=1)

    y_true = test_labels.detach().cpu().numpy()
    y_pred = predicted_labels.detach().cpu().numpy()
    y_score = positive_probabilities.detach().cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=positive_label,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=positive_label,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=positive_label,
        zero_division=0,
    )

    unique_test_labels = sorted(set(y_true.tolist()))

    if len(unique_test_labels) == 2:
        roc_auc = roc_auc_score(
            y_true,
            y_score,
        )

        pr_auc = average_precision_score(
            y_true,
            y_score,
            pos_label=positive_label,
        )
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "test_accuracy": float(accuracy),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_positive_nodes": int(
            (test_labels == positive_label).sum().item()
        ),
        "test_negative_nodes": int(
            (test_labels != positive_label).sum().item()
        ),
        "test_total_nodes": int(test_labels.numel()),
    }


def print_test_metrics(
    test_metrics: dict[str, float | int],
) -> None:
    """Print final held-out test metrics."""
    print()
    print("GAT test-set metrics")
    print(test_metrics)


def save_training_outputs(
    model: LanlGAT,
    data: Data,
    best_model_state: dict[str, torch.Tensor],
    training_history: list[dict[str, float]],
    test_metrics: dict[str, float | int],
    class_weights: torch.Tensor,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    """Save the best model checkpoint and experiment metrics JSON."""
    paths["model"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths["metrics"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation_epoch = min(
        training_history,
        key=lambda record: record["val_loss"],
    )

    checkpoint = {
        "model_state_dict": best_model_state,
        "model_class": model.__class__.__name__,
        "input_channels": int(data.x.shape[1]),
        "hidden_channels": model.hidden_channels,
        "output_channels": model.output_channels,
        "heads": model.heads,
        "dropout": model.dropout,
        "attention_dropout": model.attention_dropout,
        "negative_slope": model.negative_slope,
        "add_self_loops": model.add_self_loops,
        "class_weights": class_weights.detach().cpu(),
        "best_validation_epoch": best_validation_epoch,
        "config": config,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    torch.save(
        checkpoint,
        paths["model"],
    )

    metrics_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "gat_training_and_evaluation",
        "graph_path": str(paths["graph"]),
        "model_path": str(paths["model"]),
        "metrics_path": str(paths["metrics"]),
        "graph": {
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.shape[1]),
            "node_feature_count": int(data.x.shape[1]),
            "label_values": sorted(
                int(label)
                for label in data.y.unique().tolist()
            ),
        },
        "model": {
            "class": model.__class__.__name__,
            "input_channels": model.input_channels,
            "hidden_channels_per_head": model.hidden_channels,
            "heads": model.heads,
            "concatenated_hidden_width": (
                model.hidden_channels * model.heads
            ),
            "output_channels": model.output_channels,
            "dropout": model.dropout,
            "attention_dropout": model.attention_dropout,
            "negative_slope": model.negative_slope,
            "add_self_loops": model.add_self_loops,
        },
        "training": {
            "epochs": int(config["training"]["epochs"]),
            "learning_rate": float(
                config["training"]["learning_rate"]
            ),
            "weight_decay": float(
                config["training"]["weight_decay"]
            ),
            "class_weights": (
                class_weights.detach().cpu().tolist()
            ),
        },
        "split": {
            "train_ratio": float(config["split"]["train_ratio"]),
            "val_ratio": float(config["split"]["val_ratio"]),
            "test_ratio": float(config["split"]["test_ratio"]),
            "seed": int(config["split"]["seed"]),
        },
        "split_counts": {
            "train_nodes": int(data.train_mask.sum().item()),
            "val_nodes": int(data.val_mask.sum().item()),
            "test_nodes": int(data.test_mask.sum().item()),
        },
        "test_metrics": test_metrics,
        "best_validation_epoch": best_validation_epoch,
        "training_history": training_history,
    }

    paths["metrics"].write_text(
        json.dumps(
            make_json_safe(metrics_payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def print_saved_output_paths(
    paths: dict[str, Path],
) -> None:
    """Print paths to the saved model and metrics evidence."""
    print()
    print("Saved GAT training outputs")
    print(
        {
            "model_checkpoint": str(paths["model"]),
            "metrics_json": str(paths["metrics"]),
        }
    )


def prepare_mlflow_storage(
    config: dict[str, Any],
) -> None:
    """Create local directories required by SQLite and artifact storage."""
    tracking_uri = str(
        config["experiment_tracking"]["uri"]
    )

    sqlite_prefix = "sqlite:///"

    if tracking_uri.startswith(sqlite_prefix):
        database_path = Path(
            tracking_uri.removeprefix(sqlite_prefix)
        )

        database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    artifact_root = Path(
        config["experiment_tracking"]["artifact_root"]
    ).resolve()

    artifact_root.mkdir(
        parents=True,
        exist_ok=True,
    )


def configure_experiment(
    config: dict[str, Any],
) -> str:
    """Configure MLflow and create or reuse the GAT experiment."""
    prepare_mlflow_storage(config)

    tracking_config = config["experiment_tracking"]

    mlflow.set_tracking_uri(
        tracking_config["uri"]
    )

    experiment_name = str(
        tracking_config["experiment_name"]
    )

    artifact_root = Path(
        tracking_config["artifact_root"]
    ).resolve()

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_root.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(
        experiment_name=experiment_name,
    )

    return str(experiment_id)


def build_run_parameters(
    data: Data,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, object]:
    """Build flat MLflow parameters for filtering and comparison."""
    hidden_channels = int(
        config["model"]["hidden_channels"]
    )
    heads = int(
        config["model"]["heads"]
    )

    return {
        "model_type": "pyg_gat_node_classifier",
        "dataset_type": "lanl_homogeneous_graph",
        "graph_path": config["input"]["graph_path"],
        "device": str(device),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "node_feature_count": int(data.x.shape[1]),
        "train_ratio": float(config["split"]["train_ratio"]),
        "val_ratio": float(config["split"]["val_ratio"]),
        "test_ratio": float(config["split"]["test_ratio"]),
        "split_seed": int(config["split"]["seed"]),
        "hidden_channels_per_head": hidden_channels,
        "attention_heads": heads,
        "concatenated_hidden_width": hidden_channels * heads,
        "feature_dropout": float(config["model"]["dropout"]),
        "attention_dropout": float(
            config["model"]["attention_dropout"]
        ),
        "negative_slope": float(
            config["model"]["negative_slope"]
        ),
        "add_self_loops": bool(
            config["model"]["add_self_loops"]
        ),
        "epochs": int(config["training"]["epochs"]),
        "learning_rate": float(
            config["training"]["learning_rate"]
        ),
        "weight_decay": float(
            config["training"]["weight_decay"]
        ),
        "positive_label": int(
            config["evaluation"]["positive_label"]
        ),
    }


def build_mlflow_metrics(
    training_history: list[dict[str, float]],
    test_metrics: dict[str, float | int],
) -> dict[str, float]:
    """Build finite scalar summary metrics for MLflow."""
    best_validation_epoch = min(
        training_history,
        key=lambda record: record["val_loss"],
    )

    candidate_metrics = {
        "best_val_loss": float(
            best_validation_epoch["val_loss"]
        ),
        "best_val_accuracy": float(
            best_validation_epoch["val_accuracy"]
        ),
        "final_train_loss": float(
            training_history[-1]["train_loss"]
        ),
        "final_train_accuracy": float(
            training_history[-1]["train_accuracy"]
        ),
        "test_accuracy": float(
            test_metrics["test_accuracy"]
        ),
        "test_precision": float(
            test_metrics["test_precision"]
        ),
        "test_recall": float(
            test_metrics["test_recall"]
        ),
        "test_f1": float(
            test_metrics["test_f1"]
        ),
        "test_roc_auc": float(
            test_metrics["test_roc_auc"]
        ),
        "test_pr_auc": float(
            test_metrics["test_pr_auc"]
        ),
    }

    return {
        metric_name: metric_value
        for (
            metric_name,
            metric_value,
        ) in candidate_metrics.items()
        if math.isfinite(metric_value)
    }


def log_epoch_history_to_mlflow(
    training_history: list[dict[str, float]],
) -> None:
    """Log train and validation curves with epoch as the MLflow step."""
    for epoch_record in training_history:
        step = int(epoch_record["epoch"])

        mlflow.log_metrics(
            {
                "epoch_train_loss": float(
                    epoch_record["train_loss"]
                ),
                "epoch_train_accuracy": float(
                    epoch_record["train_accuracy"]
                ),
                "epoch_val_loss": float(
                    epoch_record["val_loss"]
                ),
                "epoch_val_accuracy": float(
                    epoch_record["val_accuracy"]
                ),
            },
            step=step,
        )


def log_run_artifacts(
    config_path: Path,
    paths: dict[str, Path],
) -> None:
    """Log config, metrics, and model checkpoint as MLflow evidence."""
    mlflow.log_artifact(
        str(config_path),
        artifact_path="run_evidence/config",
    )

    mlflow.log_artifact(
        str(paths["metrics"]),
        artifact_path="run_evidence/metrics",
    )

    mlflow.log_artifact(
        str(paths["model"]),
        artifact_path="run_evidence/model",
    )


def parse_args() -> argparse.Namespace:
    """Parse the config path used by the training command."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the LANL PyG GAT baseline."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(CONFIG_PATH),
        help="Path to LANL GAT config YAML file.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete LANL GAT training and evaluation workflow."""
    args = parse_args()

    torch.manual_seed(42)

    config = load_config(args.config)
    validate_config(config)

    experiment_id = configure_experiment(config)

    paths = build_paths(config)
    validate_input_paths(paths)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    with mlflow.start_run(
        run_name="lanl-gat-pyg-baseline"
    ) as run:
        print("Config path:", args.config)
        print("Graph path:", paths["graph"])
        print("Device:", device)

        data = load_pyg_graph(paths["graph"])

        validate_graph_for_gat(data)
        print_graph_summary(data)

        data = create_node_masks(
            data=data,
            config=config,
        )
        validate_node_masks(data)
        print_node_mask_summary(data)

        data = data.to(device)

        model = build_model(
            data=data,
            config=config,
        ).to(device)

        print_model_summary(
            model=model,
            data=data,
            device=device,
        )

        positive_label = int(
            config["evaluation"]["positive_label"]
        )

        logits = run_forward_pass_smoke_check(
            model=model,
            data=data,
        )

        print_forward_pass_summary(
            logits=logits,
            data=data,
            positive_label=positive_label,
        )

        class_weights = build_class_weights(data).to(device)

        print()
        print(
            "Training class weights:",
            class_weights.detach().cpu().tolist(),
        )

        optimizer = build_optimizer(
            model=model,
            config=config,
        )

        (
            training_history,
            best_model_state,
        ) = train_model(
            model=model,
            data=data,
            optimizer=optimizer,
            class_weights=class_weights,
            config=config,
        )

        print_training_summary(training_history)

        test_metrics = evaluate_test_set(
            model=model,
            data=data,
            best_model_state=best_model_state,
            positive_label=positive_label,
        )

        print_test_metrics(test_metrics)

        save_training_outputs(
            model=model,
            data=data,
            best_model_state=best_model_state,
            training_history=training_history,
            test_metrics=test_metrics,
            class_weights=class_weights,
            config=config,
            paths=paths,
        )

        print_saved_output_paths(paths)

        run_parameters = build_run_parameters(
            data=data,
            config=config,
            device=device,
        )

        summary_metrics = build_mlflow_metrics(
            training_history=training_history,
            test_metrics=test_metrics,
        )

        mlflow.set_tags(
            {
                "project": "aegis-hgx",
                "model_family": "gat",
                "dataset_type": "lanl",
                "graph_type": "homogeneous",
                "pipeline_stage": "graph_training",
            }
        )

        mlflow.log_params(run_parameters)
        log_epoch_history_to_mlflow(training_history)
        mlflow.log_metrics(summary_metrics)

        log_run_artifacts(
            config_path=args.config,
            paths=paths,
        )

        print()
        print("MLflow tracking")
        print(
            {
                "experiment_id": experiment_id,
                "run_id": run.info.run_id,
                "tracking_uri": mlflow.get_tracking_uri(),
                "artifact_uri": run.info.artifact_uri,
            }
        )


if __name__ == "__main__":
    main()