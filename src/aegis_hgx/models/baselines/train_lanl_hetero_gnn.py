from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


SUPERVISED_NODE_TYPES = (
    "user",
    "host",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graph(path: Path) -> HeteroData:
    try:
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        data = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(data, HeteroData):
        raise TypeError(
            "Expected a PyG HeteroData graph artifact."
        )

    for node_type in SUPERVISED_NODE_TYPES:
        required_attributes = {
            "x",
            "y",
            "train_mask",
            "val_mask",
            "test_mask",
        }

        missing = {
            name
            for name in required_attributes
            if not hasattr(data[node_type], name)
        }

        if missing:
            raise ValueError(
                f"{node_type} is missing: {sorted(missing)}"
            )

    for edge_type in data.edge_types:
        if "redteam" in edge_type[1].lower():
            raise ValueError(
                f"Label-leaking relation found: {edge_type}"
            )

    return data


class HeterogeneousGraphSAGE(nn.Module):
    def __init__(
        self,
        metadata: tuple[
            list[str],
            list[tuple[str, str, str]],
        ],
        hidden_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()

        _, edge_types = metadata

        self.dropout = dropout

        self.conv1 = HeteroConv(
            {
                edge_type: SAGEConv(
                    (-1, -1),
                    hidden_channels,
                )
                for edge_type in edge_types
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
                edge_type: SAGEConv(
                    (-1, -1),
                    hidden_channels,
                )
                for edge_type in edge_types
            },
            aggr="sum",
        )

        self.user_head = nn.Linear(
            hidden_channels,
            1,
        )

        self.host_head = nn.Linear(
            hidden_channels,
            1,
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[
            tuple[str, str, str],
            torch.Tensor,
        ],
    ) -> dict[str, torch.Tensor]:
        hidden_dict = self.conv1(
            x_dict,
            edge_index_dict,
        )

        hidden_dict = {
            node_type: F.dropout(
                F.relu(hidden),
                p=self.dropout,
                training=self.training,
            )
            for node_type, hidden in hidden_dict.items()
        }

        embedding_dict = self.conv2(
            hidden_dict,
            edge_index_dict,
        )

        # Each output has shape [number_of_nodes_of_type].
        return {
            "user": self.user_head(
                embedding_dict["user"]
            ).squeeze(-1),
            "host": self.host_head(
                embedding_dict["host"]
            ).squeeze(-1),
        }


def build_positive_weights(
    data: HeteroData,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    positive_weights: dict[str, torch.Tensor] = {}

    for node_type in SUPERVISED_NODE_TYPES:
        labels = data[node_type].y[
            data[node_type].train_mask
        ]

        positive_count = labels.sum()
        negative_count = labels.numel() - positive_count

        if positive_count.item() == 0:
            raise ValueError(
                f"{node_type} training split has no positives."
            )

        positive_weights[node_type] = (
            negative_count / positive_count
        ).to(device)

    return positive_weights


def calculate_training_loss(
    logits: dict[str, torch.Tensor],
    data: HeteroData,
    positive_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []

    for node_type in SUPERVISED_NODE_TYPES:
        mask = data[node_type].train_mask

        node_loss = F.binary_cross_entropy_with_logits(
            logits[node_type][mask],
            data[node_type].y[mask],
            pos_weight=positive_weights[node_type],
        )

        losses.append(node_loss)

    return torch.stack(losses).mean()


def predict_split(
    model: HeterogeneousGraphSAGE,
    data: HeteroData,
    mask_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    labels: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []

    with torch.no_grad():
        logits = model(
            data.x_dict,
            data.edge_index_dict,
        )

        for node_type in SUPERVISED_NODE_TYPES:
            mask = getattr(
                data[node_type],
                mask_name,
            )

            labels.append(
                data[node_type].y[mask]
            )

            probabilities.append(
                torch.sigmoid(
                    logits[node_type][mask]
                )
            )

    combined_labels = torch.cat(
        labels
    ).detach().cpu().numpy().astype(np.int64)

    combined_probabilities = torch.cat(
        probabilities
    ).detach().cpu().numpy()

    return combined_labels, combined_probabilities


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    if np.unique(labels).size != 2:
        raise ValueError(
            "Metrics require both positive and negative labels."
        )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    return {
        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "threshold": float(threshold),
        "examples": int(labels.size),
        "positives": int(labels.sum()),
        "alerts": int(predictions.sum()),
    }


def select_validation_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.linspace(
        0.05,
        0.95,
        91,
    ):
        predictions = (
            probabilities >= threshold
        ).astype(np.int64)

        score = f1_score(
            labels,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold


def train_model(
    model: HeterogeneousGraphSAGE,
    data: HeteroData,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[
    int,
    dict[str, float | int],
]:
    epochs = int(
        config["training"]["epochs"]
    )

    log_every = int(
        config["training"][
            "log_every_n_epochs"
        ]
    )

    # Materialize the lazy SAGEConv input dimensions.
    model.eval()

    with torch.no_grad():
        model(
            data.x_dict,
            data.edge_index_dict,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            config["training"]["learning_rate"]
        ),
        weight_decay=float(
            config["training"]["weight_decay"]
        ),
    )

    positive_weights = build_positive_weights(
        data=data,
        device=device,
    )

    best_epoch = 0
    best_validation_ap = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None

    print("\nTraining")
    print("--------")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(
            data.x_dict,
            data.edge_index_dict,
        )

        loss = calculate_training_loss(
            logits=logits,
            data=data,
            positive_weights=positive_weights,
        )

        if not torch.isfinite(loss):
            raise ValueError(
                f"Non-finite loss at epoch {epoch}."
            )

        loss.backward()
        optimizer.step()

        validation_labels, validation_probabilities = (
            predict_split(
                model=model,
                data=data,
                mask_name="val_mask",
            )
        )

        validation_ap = average_precision_score(
            validation_labels,
            validation_probabilities,
        )

        if validation_ap > best_validation_ap:
            best_validation_ap = float(
                validation_ap
            )
            best_epoch = epoch

            # Checkpoint selection uses validation data only.
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor
                in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch == epochs
            or epoch % log_every == 0
        ):
            validation_roc_auc = roc_auc_score(
                validation_labels,
                validation_probabilities,
            )

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"loss={loss.item():.4f} | "
                f"val AP={validation_ap:.4f} | "
                f"val ROC-AUC={validation_roc_auc:.4f}"
            )

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a checkpoint."
        )

    model.load_state_dict(best_state)

    validation_labels, validation_probabilities = (
        predict_split(
            model=model,
            data=data,
            mask_name="val_mask",
        )
    )

    threshold = select_validation_threshold(
        labels=validation_labels,
        probabilities=validation_probabilities,
    )

    validation_metrics = calculate_metrics(
        labels=validation_labels,
        probabilities=validation_probabilities,
        threshold=threshold,
    )

    print("\nBest validation checkpoint")
    print("--------------------------")
    print(f"Epoch: {best_epoch}")
    print(
        "Validation average precision:",
        f"{validation_metrics['average_precision']:.6f}",
    )
    print(
        "Validation threshold:",
        f"{threshold:.2f}",
    )

    return best_epoch, validation_metrics


def save_results(
    model: HeterogeneousGraphSAGE,
    data: HeteroData,
    config: dict[str, Any],
    best_epoch: int,
    validation_metrics: dict[str, float | int],
    test_metrics: dict[str, float | int],
) -> None:
    model_path = Path(
        config["output"]["model_path"]
    )

    metrics_path = Path(
        config["output"]["metrics_path"]
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cpu_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }

    torch.save(
        {
            "model_state_dict": cpu_state,
            "metadata": data.metadata(),
            "hidden_channels": int(
                config["model"]["hidden_channels"]
            ),
            "dropout": float(
                config["model"]["dropout"]
            ),
            "best_epoch": best_epoch,
            "threshold": float(
                validation_metrics["threshold"]
            ),
            "feature_names": list(
                data["user"].feature_names
            ),
            "supervised_node_types": list(
                SUPERVISED_NODE_TYPES
            ),
        },
        model_path,
    )

    report = {
        "model": {
            "name": "heterogeneous_graphsage",
            "operator": "relation_specific_sageconv",
            "aggregation": "sum",
            "objective": (
                "weighted_user_and_host_node_classification"
            ),
            "hidden_channels": int(
                config["model"]["hidden_channels"]
            ),
            "dropout": float(
                config["model"]["dropout"]
            ),
        },
        "graph": {
            "node_types": list(data.node_types),
            "edge_types": [
                list(edge_type)
                for edge_type in data.edge_types
            ],
            "node_counts": {
                node_type: int(
                    data[node_type].num_nodes
                )
                for node_type in data.node_types
            },
            "relation_edge_counts": {
                "|".join(edge_type): int(
                    data[edge_type].edge_index.size(1)
                )
                for edge_type in data.edge_types
            },
            "redteam_ground_truth_edges_used": False,
        },
        "training": {
            "epochs_requested": int(
                config["training"]["epochs"]
            ),
            "best_epoch": best_epoch,
            "checkpoint_selection_metric": (
                "validation_average_precision"
            ),
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "limitations": [
            (
                "This is a static transductive node-classification "
                "baseline, not a chronological evaluation."
            ),
            (
                "Process and host_or_domain nodes provide context "
                "but are not supervised targets."
            ),
            (
                "Red-team ground-truth edges are excluded from "
                "features and message passing."
            ),
            (
                "Threshold selection uses validation F1 and may "
                "not represent a production SOC operating point."
            ),
        ],
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("\nSaved artifacts")
    print("---------------")
    print(f"Model: {model_path}")
    print(f"Metrics: {metrics_path}")


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

    seed = int(config["split"]["seed"])
    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    data = load_graph(
        Path(config["input"]["graph_path"])
    )

    data = data.to(device)

    model = HeterogeneousGraphSAGE(
        metadata=data.metadata(),
        hidden_channels=int(
            config["model"]["hidden_channels"]
        ),
        dropout=float(
            config["model"]["dropout"]
        ),
    ).to(device)

    best_epoch, validation_metrics = train_model(
        model=model,
        data=data,
        config=config,
        device=device,
    )

    # The test masks are touched only after checkpoint selection.
    test_labels, test_probabilities = predict_split(
        model=model,
        data=data,
        mask_name="test_mask",
    )

    test_metrics = calculate_metrics(
        labels=test_labels,
        probabilities=test_probabilities,
        threshold=float(
            validation_metrics["threshold"]
        ),
    )

    print("\nHeld-out test results")
    print("---------------------")
    print(
        "Test average precision:",
        f"{test_metrics['average_precision']:.6f}",
    )
    print(
        "Test ROC-AUC:",
        f"{test_metrics['roc_auc']:.6f}",
    )
    print(
        "Test precision:",
        f"{test_metrics['precision']:.6f}",
    )
    print(
        "Test recall:",
        f"{test_metrics['recall']:.6f}",
    )
    print(
        "Test F1:",
        f"{test_metrics['f1']:.6f}",
    )
    print(
        "Test alerts:",
        test_metrics["alerts"],
    )

    save_results(
        model=model,
        data=data,
        config=config,
        best_epoch=best_epoch,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )


if __name__ == "__main__":
    main()