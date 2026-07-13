from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
from datetime import datetime, timezone
import json
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import mlflow



CONFIG_PATH = "configs/lanl_gcn_pyg.yaml"


def make_json_safe(value: Any) -> Any:
    # Convert dictionaries recursively.
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    # Convert lists recursively.
    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    # Convert tuples to JSON-compatible lists.
    if isinstance(value, tuple):
        return [
            make_json_safe(item)
            for item in value
        ]

    # Convert PyTorch scalar tensors to normal Python values.
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()

        return value.detach().cpu().tolist()

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
        "split",
        "model",
        "training",
        "evaluation",
        "experiment_tracking",
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

    required_training_keys = [
        "epochs",
        "learning_rate",
        "weight_decay",
    ]

    for key in required_training_keys:
        if key not in config["training"]:
            raise ValueError(f"Missing training.{key} in config.")

    # MLflow settings control where experiment evidence is logged.
    required_tracking_keys = [
        "uri",
        "experiment_name",
        "artifact_root",
    ]

    for key in required_tracking_keys:
        if key not in config["experiment_tracking"]:
            raise ValueError(f"Missing experiment_tracking.{key} in config.")


def build_class_weights(data: Data) -> torch.Tensor:
    # We calculate class weights from the training nodes only.
    # This avoids using validation/test label distribution to shape training behavior.
    train_labels = data.y[data.train_mask]

    # Number of classes.
    # For binary labels 0 and 1, num_classes = 2.
    num_classes = int(data.y.max().item()) + 1

    # Count how many training examples exist for each class.
    # Example: class_counts might be tensor([16800, 71]).
    class_counts = torch.bincount(
        train_labels,
        minlength=num_classes,
    ).float()

    # Avoid division by zero if a class is missing from the training split.
    class_counts = class_counts.clamp(min=1.0)

    # Inverse frequency weighting:
    # rare class gets larger weight, common class gets smaller weight.
    class_weights = train_labels.numel() / (
        num_classes * class_counts
    )

    return class_weights


def build_optimizer(
    model: LanlGCN,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    # Adam is a standard optimizer for neural network baselines.
    # It updates model weights using gradients from backpropagation.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    return optimizer


def run_one_training_step(
    model: LanlGCN,
    data: Data,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
) -> dict[str, float]:
    # Turn on training behavior.
    # This matters because dropout is active during training.
    model.train()

    # Clear old gradients.
    # PyTorch accumulates gradients by default, so we reset them every step.
    optimizer.zero_grad()

    # Forward pass:
    # The GCN reads node features and graph structure.
    # Output shape: [num_nodes, num_classes]
    logits = model(
        data.x,
        data.edge_index,
    )

    # Select only training nodes.
    # We do not train on validation or test nodes.
    train_logits = logits[data.train_mask]
    train_labels = data.y[data.train_mask]

    # Calculate supervised classification loss.
    # Cross entropy compares raw logits against class labels.
    # Penalize mistakes on class 1 more heavily. Mistakes is nothing but loss.
    loss = F.cross_entropy(
        train_logits,
        train_labels,
        weight=class_weights,
    )

    # Backward pass:
    # PyTorch computes gradients for every trainable model parameter.
    loss.backward()

    # Optimizer step:
    # Adam updates the model weights using the gradients.
    optimizer.step()

    # Simple training accuracy for this one step.
    # This is only a sanity check, not the final metric.
    train_predictions = train_logits.argmax(dim=1)
    train_accuracy = (
        train_predictions == train_labels
    ).float().mean().item()

    return {
        "one_step_train_loss": float(loss.item()),
        "one_step_train_accuracy": float(train_accuracy),
    }


def print_one_training_step_summary(
    step_metrics: dict[str, float],
    class_weights: torch.Tensor,
) -> None:
    print()
    print("One training step summary")
    print(
        {
            "one_step_train_loss": step_metrics["one_step_train_loss"],
            "one_step_train_accuracy": step_metrics["one_step_train_accuracy"],
            "class_weights": class_weights.tolist(),
        }
    )


def train_model(
    model: LanlGCN,
    data: Data,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    config: dict[str, Any],
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor]]:
    # Number of times we train over the graph.
    epochs = int(config["training"]["epochs"])

    # We store metrics from every epoch here.
    training_history: list[dict[str, float]] = []

    # We keep track of the best validation loss.
    # Lower validation loss means the model is doing better on held-out nodes.
    best_val_loss = float("inf")

    # This will store the best model weights seen during training.
    best_model_state: dict[str, torch.Tensor] = {}

    for epoch in range(1, epochs + 1):
        # ------------------------------------------------------------
        # TRAINING PHASE
        # ------------------------------------------------------------

        # Training mode turns dropout ON.
        model.train()

        # Clear gradients from the previous epoch.
        optimizer.zero_grad()

        # Forward pass over the full graph.
        # The model uses:
        #   data.x          -> node features
        #   data.edge_index -> graph structure
        #
        # Output:
        #   logits -> raw class scores for every node
        #
        # Shape:
        #   logits = [num_nodes, num_classes]
        logits = model(
            data.x,
            data.edge_index,
        )

        # Select only the training nodes.
        # We do not train on validation or test nodes.
        train_logits = logits[data.train_mask]
        train_labels = data.y[data.train_mask]

        # Calculate training loss.
        # Cross entropy compares raw logits against true class labels.
        train_loss = F.cross_entropy(
            train_logits,
            train_labels,
            weight=class_weights,
        )

        # Backpropagation:
        # PyTorch computes gradients for all trainable parameters.
        train_loss.backward()

        # Optimizer step:
        # Adam updates the GCN weights using those gradients.
        optimizer.step()

        # Calculate simple training accuracy.
        train_predictions = train_logits.argmax(dim=1)
        train_accuracy = (
            train_predictions == train_labels
        ).float().mean().item()

        # ------------------------------------------------------------
        # VALIDATION PHASE
        # ------------------------------------------------------------

        # Evaluation mode turns dropout OFF.
        model.eval()

        # We do not need gradients for validation.
        with torch.no_grad():
            # Forward pass again, now with dropout disabled.
            val_logits_all_nodes = model(
                data.x,
                data.edge_index,
            )

            # Select only validation nodes.
            val_logits = val_logits_all_nodes[data.val_mask]
            val_labels = data.y[data.val_mask]

            # Calculate validation loss.
            # This tells us how well the model performs on held-out nodes.
            val_loss = F.cross_entropy(
                val_logits,
                val_labels,
                weight=class_weights,
            )

            # Calculate validation accuracy.
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
        # TRACK BEST MODEL
        # ------------------------------------------------------------

        # If this epoch has the best validation loss so far,
        # save a copy of the model weights.
        if float(val_loss.item()) < best_val_loss:
            best_val_loss = float(val_loss.item())

            best_model_state = {
                param_name: param_weight_tensor.detach().cpu().clone() 
                for param_name, param_weight_tensor in model.state_dict().items()
            }

        # Print progress occasionally.
        # This avoids flooding the terminal.
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            print(epoch_record)

    return training_history, best_model_state


def print_training_summary(
    training_history: list[dict[str, float]],
) -> None:
    # First epoch tells us where training started.
    first_epoch = training_history[0]

    # Last epoch tells us where training ended.
    final_epoch = training_history[-1]

    # Best validation-loss epoch tells us which checkpoint performed best.
    best_epoch = min(
        training_history,
        key=lambda record: record["val_loss"],
    )

    print()
    print("GCN training summary")
    print(
        {
            "first_epoch": first_epoch,
            "final_epoch": final_epoch,
            "best_val_loss_epoch": best_epoch,
        }
    )


def evaluate_test_set(
    model: LanlGCN,
    data: Data,
    best_model_state: dict[str, torch.Tensor],
    positive_label: int,
) -> dict[str, float]:
    # Load the best model weights found during validation.
    # We do not want to evaluate a worse final epoch if validation loss was better earlier.
    model.load_state_dict(best_model_state)

    # Evaluation mode turns dropout OFF.
    model.eval()

    # We do not calculate gradients during test evaluation.
    # This is evaluation only, not training.
    with torch.no_grad():
        # Forward pass over the full graph.
        # Shape: [num_nodes, num_classes]
        logits = model(
            data.x,
            data.edge_index,
        )

        # Select only test nodes.
        test_logits = logits[data.test_mask]
        test_labels = data.y[data.test_mask]

        # Convert logits to probabilities.
        # For binary classification, column 1 is suspicious probability.
        probabilities = torch.softmax(
            test_logits,
            dim=1,
        )

        positive_probabilities = probabilities[:, positive_label]

        # Convert logits to hard class predictions.
        predicted_labels = test_logits.argmax(dim=1)

    # Move tensors to CPU NumPy arrays so sklearn can compute metrics.
    y_true = test_labels.cpu().numpy()
    y_pred = predicted_labels.cpu().numpy()
    y_score = positive_probabilities.cpu().numpy()

    # Accuracy = fraction of correct predictions.
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    # Precision = of predicted alerts, how many were truly positive?
    precision = precision_score(
        y_true,
        y_pred,
        pos_label=positive_label,
        zero_division=0,
    )

    # Recall = of true positives, how many did we catch?
    recall = recall_score(
        y_true,
        y_pred,
        pos_label=positive_label,
        zero_division=0,
    )

    # F1 balances precision and recall.
    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=positive_label,
        zero_division=0,
    )

    # ROC-AUC and PR-AUC need both classes to exist in the test set.
    # If the test set has only one class, these metrics are undefined.
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
        "test_positive_nodes": int((test_labels == positive_label).sum().item()),
        "test_total_nodes": int(test_labels.numel()),
    }


def print_test_metrics(
    test_metrics: dict[str, float],
) -> None:
    print()
    print("GCN test-set metrics")
    print(test_metrics)


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

        print(f"Label value: {label_value}, {label_indices[:10].tolist()}")

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

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        # Dropout randomly zeroes part of the hidden representation during training.
        # This helps reduce overfitting.
        self.dropout = dropout

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
            "hidden_channels": model.hidden_channels,
            "output_classes": int(data.y.max().item()) + 1,
            "trainable_parameters": trainable_parameters,
        }
    )

def run_forward_pass_smoke_check(
    model: LanlGCN,
    data: Data,
) -> torch.Tensor:
    # Put the model in evaluation mode for this smoke check.
    # This disables dropout, so the output is stable and easier to inspect.
    model.eval()

    # We are not training yet.
    # no_grad tells PyTorch not to build a gradient graph.
    # This saves memory and makes the check faster.
    with torch.no_grad():
        # Run one forward pass through the GCN.
        # Inputs:
        #   data.x          -> node feature matrix
        #   data.edge_index -> graph connectivity
        #
        # Output:
        #   logits -> raw class scores for every node
        logits = model(
            data.x,
            data.edge_index,
        )

    # The model should return one row per node.
    expected_node_count = int(data.num_nodes)

    # The model should return one column per class.
    expected_class_count = int(data.y.max().item()) + 1

    # Check rank.
    # logits must be a 2D tensor: [num_nodes, num_classes]
    if logits.ndim != 2:
        raise ValueError("GCN logits must have shape [num_nodes, num_classes].")

    # Check number of rows.
    if logits.shape[0] != expected_node_count:
        raise ValueError(
            "GCN logits row count must match the number of graph nodes."
        )

    # Check number of columns.
    if logits.shape[1] != expected_class_count:
        raise ValueError(
            "GCN logits column count must match the number of node classes."
        )

    # Check for NaN or infinite values.
    # These usually indicate numerical instability or corrupted inputs.
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("GCN logits contain NaN or infinite values.")

    return logits


def save_training_outputs(
    model: LanlGCN,
    data: Data,
    best_model_state: dict[str, torch.Tensor],
    training_history: list[dict[str, float]],
    test_metrics: dict[str, float],
    config: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    # Make sure output folders exist before writing files.
    paths["model"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths["metrics"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load the best validation-loss weights back into the model.
    # This ensures the saved checkpoint is the best checkpoint, not merely the final epoch.
    model.load_state_dict(best_model_state)

    # Save model checkpoint.
    # We save enough information to rebuild and reload the model later.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "input_channels": int(data.x.shape[1]),
        "hidden_channels": int(config["model"]["hidden_channels"]),
        "output_channels": int(data.y.max().item()) + 1,
        "dropout": float(config["model"]["dropout"]),
        "config": config,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    torch.save(
        checkpoint,
        paths["model"],
    )

    # Save metrics and training history as JSON.
    # This is human-readable and easy to compare across experiments.
    metrics_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "gcn_training_and_evaluation",
        "graph_path": str(paths["graph"]),
        "model_path": str(paths["model"]),
        "metrics_path": str(paths["metrics"]),
        "model": {
            "class": model.__class__.__name__,
            "input_channels": int(data.x.shape[1]),
            "hidden_channels": int(config["model"]["hidden_channels"]),
            "output_channels": int(data.y.max().item()) + 1,
            "dropout": float(config["model"]["dropout"]),
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
        },
        "test_metrics": test_metrics,
        "best_validation_epoch": min(
            training_history,
            key=lambda record: record["val_loss"],
        ),
        "training_history": training_history,
    }

    paths["metrics"].write_text(
        json.dumps(
            make_json_safe(metrics_payload),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def print_saved_output_paths(
    paths: dict[str, Path],
) -> None:
    print()
    print("Saved GCN training outputs")
    print(
        {
            "model_checkpoint": str(paths["model"]),
            "metrics_json": str(paths["metrics"]),
        }
    )


def print_forward_pass_summary(
    logits: torch.Tensor,
    data: Data,
) -> None:
    # Convert logits to probabilities for easier inspection.
    # We do not use these for training yet.
    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    # Probability assigned to the positive class.
    # For binary classification, class 1 means suspicious.
    positive_class_probabilities = probabilities[:, 1]

    print()
    print("Forward-pass smoke check summary")
    print(
        {
            "logits_shape": list(logits.shape),
            "expected_nodes": int(data.num_nodes),
            "expected_classes": int(data.y.max().item()) + 1,
            "logits_are_finite": bool(torch.isfinite(logits).all().item()),
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


def configure_experiment(config: dict[str, Any]) -> str:
    # Read MLflow tracking settings from config.
    tracking_config = config["experiment_tracking"]

    # Local artifact root for MLflow run artifacts.
    artifact_root = Path(tracking_config["artifact_root"]).resolve()
    artifact_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Example: file:./mlruns
    mlflow.set_tracking_uri(tracking_config["uri"])

    # Reuse the experiment if it already exists.
    experiment = mlflow.get_experiment_by_name(
        tracking_config["experiment_name"]
    )

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=tracking_config["experiment_name"],
            artifact_location=artifact_root.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(
        experiment_id=experiment_id,
    )

    return str(experiment_id)


def build_run_parameters(
    data: Data,
    config: dict[str, Any],
) -> dict[str, object]:
    # These are the important experiment settings we want MLflow to remember.
    return {
        "model_type": "pyg_gcn_node_classifier",
        "dataset_type": "lanl_homogeneous_graph",
        "graph_path": config["input"]["graph_path"],
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "node_feature_count": int(data.x.shape[1]),
        "train_ratio": float(config["split"]["train_ratio"]),
        "val_ratio": float(config["split"]["val_ratio"]),
        "test_ratio": float(config["split"]["test_ratio"]),
        "split_seed": int(config["split"]["seed"]),
        "hidden_channels": int(config["model"]["hidden_channels"]),
        "dropout": float(config["model"]["dropout"]),
        "epochs": int(config["training"]["epochs"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "weight_decay": float(config["training"]["weight_decay"]),
    }


def build_mlflow_metrics(
    training_history: list[dict[str, float]],
    test_metrics: dict[str, float],
) -> dict[str, float]:
    
    # Find the epoch with the lowest validation loss.
    best_validation_epoch = min(
        training_history,
        key=lambda record: record["val_loss"],
    )

    # MLflow metrics should be flat scalar values.
    return {
        "best_val_loss": float(best_validation_epoch["val_loss"]),
        "best_val_accuracy": float(best_validation_epoch["val_accuracy"]),
        "final_train_loss": float(training_history[-1]["train_loss"]),
        "final_train_accuracy": float(training_history[-1]["train_accuracy"]),
        "test_accuracy": float(test_metrics["test_accuracy"]),
        "test_precision": float(test_metrics["test_precision"]),
        "test_recall": float(test_metrics["test_recall"]),
        "test_f1": float(test_metrics["test_f1"]),
        "test_roc_auc": float(test_metrics["test_roc_auc"]),
        "test_pr_auc": float(test_metrics["test_pr_auc"]),
    }



def log_mlflow_run(
    config_path: Path,
    config: dict[str, Any],
    data: Data,
    training_history: list[dict[str, float]],
    test_metrics: dict[str, float],
    paths: dict[str, Path],
) -> str:
    # Build flat dictionaries for MLflow.
    parameters = build_run_parameters(
        data=data,
        config=config,
    )

    scalar_metrics = build_mlflow_metrics(
        training_history=training_history,
        test_metrics=test_metrics,
    )

    # Start one MLflow run for this GCN experiment.
    with mlflow.start_run(run_name="lanl-gcn-pyg-baseline") as run:
        # Tags make the run easier to search/filter later.
        mlflow.set_tags(
            {
                "project": "aegis-hgx",
                "model_family": "gcn",
                "dataset_type": "lanl",
                "pipeline_stage": "graph_training",
            }
        )

        # Log config/model/data parameters.
        mlflow.log_params(parameters)

        # Log scalar metrics.
        mlflow.log_metrics(scalar_metrics)

        # Log useful evidence artifacts.
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

        return str(run.info.run_id)


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

    experiment_id = configure_experiment(config)

    paths = build_paths(config)
    validate_input_paths(paths)

    with mlflow.start_run(run_name="lanl-gcn-pyg-baseline") as run:
        print("Config path:", args.config)
        print("Graph path:", paths["graph"])

        data = load_pyg_graph(paths["graph"])

        validate_graph_for_gcn(data)

        print_graph_summary(data)

        data = create_node_masks(
            data=data,
            config=config,
        )

        # validate_node_masks(data)

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

        class_weights = build_class_weights(data)

        optimizer = build_optimizer(
            model=model,
            config=config,
        )

        training_history, best_model_state = train_model(
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
            paths=paths,
        )

        print_saved_output_paths(paths)

        parameters = build_run_parameters(
            data=data,
            config=config,
        )

        scalar_metrics = build_mlflow_metrics(
            training_history=training_history,
            test_metrics=test_metrics,
        )

        mlflow.set_tags(
            {
                "project": "aegis-hgx",
                "model_family": "gcn",
                "dataset_type": "lanl",
                "pipeline_stage": "graph_training",
            }
        )

        mlflow.log_params(parameters)

        mlflow.log_metrics(scalar_metrics)

        # Log the training config.
        mlflow.log_artifact(
            str(args.config),
            artifact_path="run_evidence/config",
        )

        # Log the metrics JSON.
        mlflow.log_artifact(
            str(paths["metrics"]),
            artifact_path="run_evidence/metrics",
        )

        # Log the actual saved model checkpoint.
        # This is the key model artifact.
        mlflow.log_artifact(
            str(paths["model"]),
            artifact_path="run_evidence/model",
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



