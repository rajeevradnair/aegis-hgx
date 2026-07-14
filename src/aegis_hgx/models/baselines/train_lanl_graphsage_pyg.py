from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import yaml
from torch_geometric.data import Data
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


CONFIG_PATH = "configs/lanl_graphsage_pyg.yaml"


def make_json_safe(value: Any) -> Any:
    # JSON cannot safely represent NaN or infinity in strict tools.
    # This helper converts those values to None.
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(value, dict):
        return {
            key: make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    return value


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

    if "model_path" not in config["output"]:
        raise ValueError("Missing config value: output.model_path")

    if "metrics_path" not in config["output"]:
        raise ValueError("Missing config value: output.metrics_path")
    

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


def create_node_masks(
    data: Data,
    config: dict[str, Any],
) -> Data:
    # Total number of nodes.
    #
    # Each node needs exactly one split assignment:
    #   train
    #   validation
    #   test
    num_nodes = int(data.num_nodes)

    # Read split ratios from config.
    train_ratio = float(config["split"]["train_ratio"])
    val_ratio = float(config["split"]["val_ratio"])

    # Use a fixed seed so the split is reproducible.
    seed = int(config["split"]["seed"])

    generator = torch.Generator()
    generator.manual_seed(seed)

    # Start with all nodes excluded from every split.
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Split each label class separately.
    #
    # This is called stratification.
    # It helps avoid a bad split where, for example,
    # all suspicious nodes accidentally land in only one split.
    label_values = sorted(data.y.unique().tolist())

    for label_value in label_values:
        # Get node indices for this label.
        #
        # Example:
        #   all benign node indices
        #   or all suspicious node indices
        label_indices = torch.where(data.y == int(label_value))[0]

        # Shuffle nodes within this label group.
        shuffled_label_indices = label_indices[
            torch.randperm(
                label_indices.numel(),
                generator=generator,
            )
        ]

        # Decide how many nodes from this label go into train and validation.
        label_count = shuffled_label_indices.numel()
        train_count = int(label_count * train_ratio)
        val_count = int(label_count * val_ratio)

        # Slice this label group into train, validation, and test.
        train_indices = shuffled_label_indices[:train_count]

        val_indices = shuffled_label_indices[
            train_count : train_count + val_count
        ]

        test_indices = shuffled_label_indices[
            train_count + val_count :
        ]

        # Mark the global masks.
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

    # Attach masks to the PyG Data object.
    #
    # These masks align with:
    #   data.x rows
    #   data.y labels
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def print_node_mask_summary(data: Data) -> None:
    # Select labels for each split.
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


class LanlGraphSAGE(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        dropout: float,
        aggregation: str,
    ) -> None:
        super().__init__()

        # First GraphSAGE layer.
        #
        # Input shape:
        #   x = [num_nodes, input_channels]
        #
        # Output shape:
        #   hidden = [num_nodes, hidden_channels]
        #
        # This layer learns how to combine:
        #   each node's own features
        #   mean-aggregated neighbor features
        self.conv1 = SAGEConv(
            in_channels=input_channels,
            out_channels=hidden_channels,
            aggr=aggregation,
        )

        # Second GraphSAGE layer.
        #
        # Input shape:
        #   hidden = [num_nodes, hidden_channels]
        #
        # Output shape:
        #   logits = [num_nodes, output_channels]
        #
        # Each row in logits contains raw class scores for one node.
        self.conv2 = SAGEConv(
            in_channels=hidden_channels,
            out_channels=output_channels,
            aggr=aggregation,
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
        #
        # Shape:
        #   [num_nodes, input_channels]

        # edge_index contains graph connectivity.
        #
        # Shape:
        #   [2, num_edges]

        # First GraphSAGE message-passing layer.
        #
        # For each node, SAGEConv aggregates neighbor features
        # and combines them with the node's own features.
        hidden = self.conv1(
            x,
            edge_index,
        )

        # ReLU adds nonlinearity.
        hidden = F.relu(hidden)

        # Dropout is active only when model.train() is set.
        # It is automatically disabled when model.eval() is set.
        hidden = F.dropout(
            hidden,
            p=self.dropout,
            training=self.training,
        )

        # Final GraphSAGE layer.
        #
        # Produces raw class scores for every node.
        logits = self.conv2(
            hidden,
            edge_index,
        )

        # Shape:
        #   [num_nodes, output_channels]
        return logits


def build_model(
    data: Data,
    config: dict[str, Any],
) -> LanlGraphSAGE:
    # Number of input features per node.
    #
    # If data.x shape is [24102, 8],
    # then input_channels = 8.
    input_channels = int(data.x.shape[1])

    # Hidden dimension from config.
    hidden_channels = int(config["model"]["hidden_channels"])

    # Number of output classes.
    #
    # If labels are 0 and 1,
    # then output_channels = 2.
    output_channels = int(data.y.max().item()) + 1

    # Dropout probability from config.
    dropout = float(config["model"]["dropout"])

    # Aggregation method from config.
    #
    # For today:
    #   "mean"
    aggregation = str(config["model"]["aggregation"])

    model = LanlGraphSAGE(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        output_channels=output_channels,
        dropout=dropout,
        aggregation=aggregation,
    )

    return model


def print_model_summary(
    model: LanlGraphSAGE,
    data: Data,
) -> None:
    print()
    print("GraphSAGE model summary")
    print(
        {
            "model_class": model.__class__.__name__,
            "num_node_features": int(data.x.shape[1]),
            "num_output_classes": int(data.y.max().item()) + 1,
            "conv1": str(model.conv1),
            "conv2": str(model.conv2),
            "dropout": float(model.dropout),
        }
    )


def run_forward_pass_smoke_check(
    model: LanlGraphSAGE,
    data: Data,
) -> torch.Tensor:
    # Evaluation mode turns dropout OFF.
    #
    # This makes the smoke check deterministic.
    # We are not training yet.
    model.eval()

    # We do not need gradients for a smoke check.
    with torch.no_grad():
        # Forward pass through the GraphSAGE model.
        #
        # Inputs:
        #   data.x          -> [num_nodes, num_node_features]
        #   data.edge_index -> [2, num_edges]
        #
        # Output:
        #   logits -> [num_nodes, num_output_classes]
        logits = model(
            data.x,
            data.edge_index,
        )

    # Expected number of nodes.
    num_nodes = int(data.num_nodes)

    # Expected number of classes.
    #
    # For binary labels 0 and 1:
    #   output_classes = 2
    output_classes = int(data.y.max().item()) + 1

    # Validate output shape.
    expected_shape = (
        num_nodes,
        output_classes,
    )

    if tuple(logits.shape) != expected_shape:
        raise ValueError(
            f"Expected logits shape {expected_shape}, got {tuple(logits.shape)}"
        )

    # Validate numeric health.
    if not torch.isfinite(logits).all():
        raise ValueError("GraphSAGE logits contain NaN or infinite values.")

    return logits


def print_forward_pass_summary(
    logits: torch.Tensor,
    data: Data,
) -> None:
    # Convert raw logits into probabilities for inspection.
    #
    # This is only for debugging.
    # During training, cross-entropy should receive raw logits.
    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    # Probability for suspicious class.
    positive_probabilities = probabilities[:, 1]

    print()
    print("GraphSAGE forward-pass smoke check summary")
    print(
        {
            "logits_shape": list(logits.shape),
            "expected_num_nodes": int(data.num_nodes),
            "expected_num_classes": int(data.y.max().item()) + 1,
            "min_positive_probability": float(positive_probabilities.min().item()),
            "mean_positive_probability": float(positive_probabilities.mean().item()),
            "max_positive_probability": float(positive_probabilities.max().item()),
        }
    )


def train_graphsage_model(
    model: LanlGraphSAGE,
    data: Data,
    config: dict[str, Any],
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor]]:
    # Number of training epochs.
    epochs = int(config["training"]["epochs"])

    # Optimizer settings from config.
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])

    # Adam updates the learnable GraphSAGE weights.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # ------------------------------------------------------------
    # CLASS WEIGHTS
    # ------------------------------------------------------------
    # Cyber anomaly labels are usually imbalanced.
    # There are usually many benign nodes and few suspicious nodes.
    #
    # Cross-entropy can use class weights so rare classes matter more.
    train_labels = data.y[data.train_mask]

    class_counts = torch.bincount(
        train_labels,
        minlength=int(data.y.max().item()) + 1,
    ).float()

    class_weights = train_labels.numel() / (
        class_counts.clamp(min=1.0) * class_counts.numel()
    )

    # Store metrics from each epoch.
    training_history: list[dict[str, float]] = []

    # Track the best validation loss.
    best_val_loss = float("inf")

    # Store the best model weights.
    best_model_state: dict[str, torch.Tensor] = {}

    print()
    print("Training GraphSAGE model")

    for epoch in range(1, epochs + 1):
        # ------------------------------------------------------------
        # TRAINING PHASE
        # ------------------------------------------------------------
        # model.train() turns dropout ON.
        model.train()

        # Clear gradients from the previous epoch.
        optimizer.zero_grad()

        # Forward pass over the full graph.
        #
        # Inputs:
        #   data.x          -> [num_nodes, num_node_features]
        #   data.edge_index -> [2, num_edges]
        #
        # Output:
        #   logits -> [num_nodes, num_classes]
        logits = model(
            data.x,
            data.edge_index,
        )

        # Use only training nodes for training loss.
        #
        # train_logits shape:
        #   [num_train_nodes, num_classes]
        #
        # train_labels shape:
        #   [num_train_nodes]
        train_logits = logits[data.train_mask]
        train_labels = data.y[data.train_mask]

        # Cross-entropy compares raw logits against true class labels.
        #
        # Important:
        #   Do not apply softmax before F.cross_entropy.
        train_loss = F.cross_entropy(
            train_logits,
            train_labels,
            weight=class_weights,
        )

        # Backward pass computes gradients.
        train_loss.backward()

        # Optimizer updates GraphSAGE weights.
        optimizer.step()

        # Training accuracy.
        train_predictions = train_logits.argmax(dim=1)

        train_accuracy = (
            train_predictions == train_labels
        ).float().mean().item()

        # ------------------------------------------------------------
        # VALIDATION PHASE
        # ------------------------------------------------------------
        # model.eval() turns dropout OFF.
        model.eval()

        # We do not need gradients for validation.
        with torch.no_grad():
            # Forward pass again with dropout disabled.
            val_logits_all_nodes = model(
                data.x,
                data.edge_index,
            )

            # Use only validation nodes for validation metrics.
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

        # ------------------------------------------------------------
        # RECORD METRICS
        # ------------------------------------------------------------
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(train_loss.item()),
            "train_accuracy": float(train_accuracy),
            "val_loss": float(val_loss.item()),
            "val_accuracy": float(val_accuracy),
        }

        training_history.append(epoch_record)

        # ------------------------------------------------------------
        # SAVE BEST MODEL STATE
        # ------------------------------------------------------------
        # Validation loss tells us which epoch generalized best
        # to held-out validation nodes.
        if float(val_loss.item()) < best_val_loss:
            best_val_loss = float(val_loss.item())

            best_model_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

        # Print progress without flooding the terminal.
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(epoch_record)

    print()
    print("GraphSAGE training summary")
    print(
        {
            "epochs": epochs,
            "best_val_loss": float(best_val_loss),
            "final_train_loss": training_history[-1]["train_loss"],
            "final_val_loss": training_history[-1]["val_loss"],
            "final_train_accuracy": training_history[-1]["train_accuracy"],
            "final_val_accuracy": training_history[-1]["val_accuracy"],
        }
    )

    return training_history, best_model_state


def evaluate_test_set(
    model: LanlGraphSAGE,
    data: Data,
    best_model_state: dict[str, torch.Tensor],
    positive_label: int,
) -> dict[str, float]:
    # The test set must use the best validation checkpoint,
    # not necessarily the final epoch.
    if not best_model_state:
        raise ValueError("Best model state is empty. Training did not produce a checkpoint.")

    # Load the best validation weights back into the model.
    model.load_state_dict(best_model_state)

    # Evaluation mode turns dropout OFF.
    model.eval()

    with torch.no_grad():
        # Forward pass over the full graph.
        #
        # Inputs:
        #   data.x          -> [num_nodes, num_node_features]
        #   data.edge_index -> [2, num_edges]
        #
        # Output:
        #   logits -> [num_nodes, num_classes]
        logits = model(
            data.x,
            data.edge_index,
        )

        # Select only test nodes.
        #
        # test_logits shape:
        #   [num_test_nodes, num_classes]
        #
        # test_labels shape:
        #   [num_test_nodes]
        test_logits = logits[data.test_mask]
        test_labels = data.y[data.test_mask]

        # Convert logits to probabilities for ranking metrics.
        probabilities = torch.softmax(
            test_logits,
            dim=1,
        )

        if positive_label >= probabilities.shape[1]:
            raise ValueError(
                f"positive_label={positive_label} is outside probability shape {probabilities.shape}"
            )

        # Probability assigned to the suspicious class.
        positive_probabilities = probabilities[:, positive_label]

        # Predicted class is the class with largest logit.
        predictions = test_logits.argmax(dim=1)

    # Move tensors to NumPy for sklearn metrics.
    y_true = test_labels.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    y_score = positive_probabilities.cpu().numpy()

    # Classification metrics.
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

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

    # ROC-AUC and PR-AUC require both classes to be present.
    unique_labels = set(y_true.tolist())

    if len(unique_labels) == 2:
        roc_auc = roc_auc_score(
            y_true,
            y_score,
        )

        pr_auc = average_precision_score(
            y_true,
            y_score,
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
        "test_positive_labels": int((test_labels == positive_label).sum().item()),
        "test_total_nodes": int(test_labels.numel()),
    }


def print_test_metrics(test_metrics: dict[str, float]) -> None:
    print()
    print("GraphSAGE test-set metrics")
    print(test_metrics)


def save_training_outputs(
    model: LanlGraphSAGE,
    data: Data,
    best_model_state: dict[str, torch.Tensor],
    training_history: list[dict[str, float]],
    test_metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    # Read output paths from config.
    model_path = Path(config["output"]["model_path"])
    metrics_path = Path(config["output"]["metrics_path"])

    # Create parent directories if they do not exist.
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Restore the best validation checkpoint before saving.
    model.load_state_dict(best_model_state)

    # Save model checkpoint.
    #
    # This is enough to reconstruct the model later because we store:
    #   model weights
    #   model dimensions
    #   aggregation type
    #   dropout
    #   config
    checkpoint = {
        "model_state_dict": best_model_state,
        "model_class": model.__class__.__name__,
        "input_channels": int(data.x.shape[1]),
        "hidden_channels": int(config["model"]["hidden_channels"]),
        "output_channels": int(data.y.max().item()) + 1,
        "dropout": float(config["model"]["dropout"]),
        "aggregation": str(config["model"]["aggregation"]),
        "config": config,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    torch.save(
        checkpoint,
        model_path,
    )

    # Identify the epoch with the best validation loss.
    best_epoch_record = min(
        training_history,
        key=lambda record: record["val_loss"],
    )

    # Build metrics payload.
    metrics_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "graphsage_training_and_evaluation",
        "graph_path": config["input"]["graph_path"],
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "model": {
            "class": model.__class__.__name__,
            "input_channels": int(data.x.shape[1]),
            "hidden_channels": int(config["model"]["hidden_channels"]),
            "output_channels": int(data.y.max().item()) + 1,
            "dropout": float(config["model"]["dropout"]),
            "aggregation": str(config["model"]["aggregation"]),
        },
        "training": {
            "epochs": int(config["training"]["epochs"]),
            "learning_rate": float(config["training"]["learning_rate"]),
            "weight_decay": float(config["training"]["weight_decay"]),
        },
        "split_counts": {
            "train_nodes": int(data.train_mask.sum().item()),
            "val_nodes": int(data.val_mask.sum().item()),
            "test_nodes": int(data.test_mask.sum().item()),
            "train_positive_labels": int((data.y[data.train_mask] == 1).sum().item()),
            "val_positive_labels": int((data.y[data.val_mask] == 1).sum().item()),
            "test_positive_labels": int((data.y[data.test_mask] == 1).sum().item()),
        },
        "best_validation_epoch": best_epoch_record,
        "test_metrics": test_metrics,
        "training_history": training_history,
    }

    # Write metrics as readable JSON.
    metrics_path.write_text(
        json.dumps(
            make_json_safe(metrics_payload),
            indent=2,
        )
    )

    print()
    print("Saved GraphSAGE training outputs")
    print(
        {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
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

    logits = run_forward_pass_smoke_check(
        model=model,
        data=data,
    )

    print_forward_pass_summary(
        logits=logits,
        data=data,
    )

    training_history, best_model_state = train_graphsage_model(
        model=model,
        data=data,
        config=config,
    )

    test_metrics = evaluate_test_set(
        model=model,
        data=data,
        best_model_state=best_model_state,
        positive_label=int(config["evaluation"]["positive_label"]),
    )

    print_test_metrics(test_metrics)

    save_training_outputs(
        model=model,
        data=data,
        best_model_state=best_model_state,
        training_history=training_history,
        test_metrics=test_metrics,
        config=config,
    )

if __name__ == "__main__":
    main()