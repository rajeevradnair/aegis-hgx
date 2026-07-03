from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import joblib

import pandas as pd
import numpy as np
import torch
import yaml
import json

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from torch import nn

import matplotlib.pyplot as plt

import mlflow

from aegis_hgx.models.baselines.lineage import (
    LineageManifestInput,
    build_lineage_manifest,
    write_lineage_manifest,
)


CONFIG_PATH = "configs/torch_mlp_cic.yaml"
DEFAULT_TRAINING_DATA_DVC_PATH = Path(
    "data/processed/cicids2017/cic_tabular_features.csv.dvc"
)
DEFAULT_LINEAGE_MANIFEST_PATH = Path(
    "artifacts/lineage/torch_mlp_cic_manifest.json"
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")

    return config


def configure_experiment(config: dict[str, Any]) -> str:
    tracking = config["experiment_tracking"]

    artifact_root = Path(tracking["artifact_root"]).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking["uri"])

    experiment = mlflow.get_experiment_by_name(
        tracking["experiment_name"]
    )

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=tracking["experiment_name"],
            artifact_location=artifact_root.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)

    return experiment_id


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    dataset_path = Path(config["data"]["input_path"])

    if not dataset_path.exists():
        raise FileNotFoundError(f"CIC feature dataset not found: {dataset_path}")

    dataset = pd.read_csv(dataset_path)

    if dataset.empty:
        raise ValueError("CIC feature dataset contains no rows.")

    return dataset


def validate_target_column(
    dataset: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    target_column = config["data"]["target_column"]

    if target_column not in dataset.columns:
        raise ValueError(
            f"CIC feature dataset is missing target column: {target_column}"
        )

    target_classes = sorted(dataset[target_column].dropna().unique())

    if len(target_classes) < 2:
        raise ValueError(
            "PyTorch MLP training requires at least two target classes. "
            f"Found classes: {target_classes}"
        )

def split_features_and_target(
    dataset: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    target_column = config["data"]["target_column"]

    validate_target_column(dataset, config)

    features = dataset.drop(columns=[target_column]).copy()
    target = dataset[target_column].copy()

    if features.empty:
        raise ValueError("CIC feature dataset contains no feature columns.")

    return features, target


def split_train_validation_test(
    features: pd.DataFrame,
    target: pd.Series,
    config: dict[str, Any],
) -> tuple[
    tuple[pd.DataFrame, pd.Series],
    tuple[pd.DataFrame, pd.Series],
    tuple[pd.DataFrame, pd.Series],
]:
    split_config = config["split"]

    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        features,
        target,
        test_size=split_config["test_size"],
        random_state=split_config["random_seed"],
        stratify=target,
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=split_config["validation_size"],
        random_state=split_config["random_seed"],
        stratify=y_train_validation,
    )

    return (
        (X_train, y_train),
        (X_validation, y_validation),
        (X_test, y_test),
    )


def build_scaled_tensors(
    train_data: tuple[pd.DataFrame, pd.Series],
    validation_data: tuple[pd.DataFrame, pd.Series],
    test_data: tuple[pd.DataFrame, pd.Series],
) -> tuple[
    StandardScaler,
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    X_train, y_train = train_data
    X_validation, y_validation = validation_data
    X_test, y_test = test_data

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    train_tensors = (
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1),
    )
    validation_tensors = (
        torch.tensor(X_validation_scaled, dtype=torch.float32),
        torch.tensor(y_validation.to_numpy(), dtype=torch.float32).view(-1, 1),
    )
    test_tensors = (
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1),
    )

    return scaler, train_tensors, validation_tensors, test_tensors


def build_train_loader(
    train_tensors: tuple[torch.Tensor, torch.Tensor],
    config: dict[str, Any],
) -> DataLoader:
    X_train_tensor, y_train_tensor = train_tensors

    dataset = TensorDataset(X_train_tensor, y_train_tensor)

    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

def build_model(
    input_dim: int,
    config: dict[str, Any],
) -> TabularMLP:
    model_config = config["model"]

    return TabularMLP(
        input_dim=input_dim,
        hidden_dim_1=model_config["hidden_dim_1"],
        hidden_dim_2=model_config["hidden_dim_2"],
        dropout=model_config["dropout"],
    )

def calculate_pos_weight(target_tensor: torch.Tensor) -> torch.Tensor:
    positive_count = torch.sum(target_tensor == 1)
    negative_count = torch.sum(target_tensor == 0)

    if positive_count.item() == 0:
        raise ValueError("Cannot calculate positive class weight with no positive rows.")

    return negative_count / positive_count


def build_loss_function(train_tensors: tuple[torch.Tensor, torch.Tensor]) -> nn.BCEWithLogitsLoss:
    _, y_train_tensor = train_tensors
    pos_weight = calculate_pos_weight(y_train_tensor)

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def build_optimizer(
    model: nn.Module,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    training_config = config["training"]

    return torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

def train_one_epoch(
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()

    total_loss = 0.0
    total_rows = 0

    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()

        logits = model(batch_features)
        loss = loss_function(logits, batch_labels)

        loss.backward()
        optimizer.step()

        batch_size = batch_features.shape[0]
        total_loss += loss.item() * batch_size
        total_rows += batch_size

    return total_loss / total_rows

# Evaluate for each epoch
def evaluate_epoch(
    model: nn.Module,
    tensors: tuple[torch.Tensor, torch.Tensor],
    loss_function: nn.Module,
    epoch: int,
) -> dict[str, object]:
    model.eval()

    features_tensor, labels_tensor = tensors

    with torch.no_grad():
        logits = model(features_tensor)
        loss = loss_function(logits, labels_tensor)
        probabilities = torch.sigmoid(logits)

    y_true = labels_tensor.detach().cpu().numpy().reshape(-1)
    y_score = probabilities.detach().cpu().numpy().reshape(-1)

    return {
        "loss": float(loss.item()),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "y_true": y_true,
        "y_score": y_score,
    }

# Logic for each probability threshold within an epoch
def build_threshold_list(config: dict[str, Any]) -> list[float]:
    threshold_config = config["thresholds"]

    start = threshold_config["start"]
    stop = threshold_config["stop"]
    step = threshold_config["step"]

    values = np.arange(start, stop + step / 2, step)

    return [round(float(value), 4) for value in values]

# Calculate TP, FP, NP, FN, Accuracy, Precision, Recall, F1 at a given threshold level
def calculate_threshold_metrics(
    epoch: int,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    y_pred = (y_score >= threshold).astype(int)

    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = true_positive + false_positive + true_negative + false_negative

    accuracy = (
        (true_positive + true_negative) / total
        if total > 0
        else 0.0
    )

    precision_denominator = true_positive + false_positive
    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall_denominator = true_positive + false_negative
    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1_denominator = precision + recall
    f1 = (
        2 * precision * recall / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    return {
        "epoch": epoch,
        "threshold": threshold,
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

def build_epoch_threshold_rows(
    epoch: int,
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, object]]:
    rows = []

    for threshold in thresholds:
        print(f"Calculating metrics for Epoch: {epoch} / threshold: {threshold}")
        row = calculate_threshold_metrics(
            epoch=epoch,
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
        )
        row["epoch"] = epoch
        rows.append(row)

    return rows

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_tensors: tuple[torch.Tensor, torch.Tensor],
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> tuple[nn.Module, list[dict[str, object]], list[dict[str, object]]]:
    epochs = config["training"]["epochs"]
    thresholds = build_threshold_list(config)

    epoch_history = []
    threshold_history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
        )

        validation_result = evaluate_epoch(
            epoch=epoch,
            model=model,
            tensors=validation_tensors,
            loss_function=loss_function,
        )

        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": float(validation_result["loss"]),
                "validation_pr_auc": float(validation_result["pr_auc"]),
                "validation_roc_auc": float(validation_result["roc_auc"]),
            }
        )

        threshold_rows = build_epoch_threshold_rows(
            epoch=epoch,
            y_true=validation_result["y_true"],
            y_score=validation_result["y_score"],
            thresholds=thresholds,
        )
        threshold_history.extend(threshold_rows)

        print(
            "Epoch",
            epoch,
            "train_loss=",
            round(float(train_loss), 6),
            "validation_loss=",
            round(float(validation_result["loss"]), 6),
            "validation_pr_auc=",
            round(float(validation_result["pr_auc"]), 6),
            "validation_roc_auc=",
            round(float(validation_result["roc_auc"]), 6),
        )

    return model, epochs, epoch_history, len(thresholds), threshold_history


def save_epoch_history(
    epoch_history: list[dict[str, object]],
    config: dict[str, Any],
) -> Path:
    history_path = Path(config["outputs"]["epoch_history_path"])
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with history_path.open("w", encoding="utf-8") as history_file:
        json.dump(epoch_history, history_file, indent=2)

    return history_path


def save_threshold_history(
    threshold_history: list[dict[str, object]],
    config: dict[str, Any],
) -> Path:
    threshold_history_path = Path(config["outputs"]["threshold_history_path"])
    threshold_history_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "epoch",
        "threshold",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    threshold_history_frame = pd.DataFrame(threshold_history)
    threshold_history_frame = threshold_history_frame[columns]
    threshold_history_frame.to_csv(threshold_history_path, index=False)

    return threshold_history_path


# Identify the best validation epoch for reporting purposes
def find_best_validation_epoch(
    epoch_history: list[dict[str, object]],
) -> dict[str, object]:
    if not epoch_history:
        raise ValueError("Epoch history is empty.")

    return max(
        epoch_history,
        key=lambda row: float(row["validation_pr_auc"]),
    )


def build_final_metrics_from_test_data(
    trained_model: nn.Module,
    train_data: tuple[pd.DataFrame, pd.Series],
    validation_data: tuple[pd.DataFrame, pd.Series],
    test_data: tuple[pd.DataFrame, pd.Series],
    test_tensors: tuple[torch.Tensor, torch.Tensor],
    loss_function: nn.Module,
    epoch_history: list[dict[str, object]],
    config: dict[str, Any],
    epoch_history_path: Path,
    threshold_history_path: Path,
) -> dict[str, object]:
    _, y_train = train_data
    _, y_validation = validation_data
    _, y_test = test_data

    best_validation_epoch = find_best_validation_epoch(epoch_history)

    test_result = evaluate_epoch(
        epoch=0,
        model=trained_model,
        tensors=test_tensors,
        loss_function=loss_function,
    )

    default_threshold = 0.5
    default_threshold_metrics = calculate_threshold_metrics(
        epoch=0,
        y_true=test_result["y_true"],
        y_score=test_result["y_score"],
        threshold=default_threshold,
    )

    confusion_matrix = [
        [
            default_threshold_metrics["tn"],
            default_threshold_metrics["fp"],
        ],
        [
            default_threshold_metrics["fn"],
            default_threshold_metrics["tp"],
        ],
    ]

    return {
        "model_name": "torch_mlp_cic_baseline",
        "dataset": "cicids2017",
        "split_type": "train_validation_test",
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_validation)),
        "test_rows": int(len(y_test)),
        "train_positive_rows": int(y_train.sum()),
        "validation_positive_rows": int(y_validation.sum()),
        "test_positive_rows": int(y_test.sum()),
        "test_negative_rows": int(len(y_test) - y_test.sum()),
        "threshold_independent_metrics": {
            "test_loss": float(test_result["loss"]),
            "test_pr_auc": float(test_result["pr_auc"]),
            "test_roc_auc": float(test_result["roc_auc"]),
        },
        "default_threshold": default_threshold,
        "default_threshold_metrics": {
            "test_accuracy": default_threshold_metrics["accuracy"],
            "test_precision": default_threshold_metrics["precision"],
            "test_recall": default_threshold_metrics["recall"],
            "test_f1": default_threshold_metrics["f1"],
            "test_true_positive": default_threshold_metrics["tp"],
            "test_false_positive": default_threshold_metrics["fp"],
            "test_true_negative": default_threshold_metrics["tn"],
            "test_false_negative": default_threshold_metrics["fn"],
            "test_confusion_matrix": confusion_matrix,
        },
        "best_validation_epoch": int(best_validation_epoch["epoch"]),
        "best_validation_metric": "validation_pr_auc",
        "best_validation_pr_auc": float(
            best_validation_epoch["validation_pr_auc"]
        ),
        "best_validation_roc_auc": float(
            best_validation_epoch["validation_roc_auc"]
        ),
        "selected_threshold": default_threshold,
        "selected_threshold_reason": (
            "Default threshold used for baseline comparability. "
            "Threshold tuning is handled in a later evaluation step."
        ),
        "artifact_paths": {
            "model_path": config["outputs"]["model_path"],
            "scaler_path": config["outputs"]["scaler_path"],
            "epoch_history_path": str(epoch_history_path),
            "threshold_history_path": str(threshold_history_path),
            "loss_by_epoch_path": config["outputs"]["loss_by_epoch_path"],
            "pr_auc_by_epoch_path": config["outputs"]["pr_auc_by_epoch_path"],
            "roc_auc_by_epoch_path": config["outputs"]["roc_auc_by_epoch_path"],
            "threshold_f1_by_threshold_path": config["outputs"][
                "threshold_f1_by_threshold_path"
            ],
            "final_test_precision_recall_curve_path": config["outputs"][
                "final_test_precision_recall_curve_path"
            ],
            "final_test_roc_curve_path": config["outputs"]["final_test_roc_curve_path"],
        },
    }

def save_final_metrics(
    metrics: dict[str, object],
    config: dict[str, Any],
) -> Path:
    metrics_path = Path(config["outputs"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    return metrics_path


def save_model_artifact(
    trained_model: nn.Module,
    feature_columns: list[str],
    config: dict[str, Any],
) -> Path:
    model_path = Path(config["outputs"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_config = config["model"]

    checkpoint = {
        "model_name": "torch_mlp_cic_baseline",
        "model_state_dict": trained_model.state_dict(),
        "input_dim": len(feature_columns),
        "hidden_dim_1": model_config["hidden_dim_1"],
        "hidden_dim_2": model_config["hidden_dim_2"],
        "dropout": model_config["dropout"],
        "feature_columns": feature_columns,
        "target_column": config["data"]["target_column"],
    }

    torch.save(checkpoint, model_path)

    return model_path


def save_scaler_artifact(
    scaler: StandardScaler,
    config: dict[str, Any],
) -> Path:
    scaler_path = Path(config["outputs"]["scaler_path"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, scaler_path)

    return scaler_path


def save_training_plots(
    epoch_history: list[dict[str, object]],
    threshold_history: list[dict[str, object]],
    test_result: dict[str, object],
    config: dict[str, Any],
) -> dict[str, str]:
    figure_dir = Path(config["outputs"]["figure_dir"])
    figure_dir.mkdir(parents=True, exist_ok=True)

    epoch_frame = pd.DataFrame(epoch_history)
    threshold_frame = pd.DataFrame(threshold_history)

    loss_by_epoch_path = Path(config["outputs"]["loss_by_epoch_path"])
    pr_auc_by_epoch_path = Path(config["outputs"]["pr_auc_by_epoch_path"])
    roc_auc_by_epoch_path = Path(config["outputs"]["roc_auc_by_epoch_path"])
    threshold_f1_by_threshold_path = Path(
        config["outputs"]["threshold_f1_by_threshold_path"]
    )
    final_pr_curve_path = Path(
        config["outputs"]["final_test_precision_recall_curve_path"]
    )
    final_roc_curve_path = Path(
        config["outputs"]["final_test_roc_curve_path"]
    )

    #MLP Loss by Epoch
    plt.figure()
    plt.plot(epoch_frame["epoch"], epoch_frame["train_loss"], label="train_loss")
    plt.plot(
        epoch_frame["epoch"],
        epoch_frame["validation_loss"],
        label="validation_loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PyTorch MLP Loss by Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_by_epoch_path)
    plt.close()

    # Validation PR-AUC by Epoch
    plt.figure()
    plt.plot(epoch_frame["epoch"], epoch_frame["validation_pr_auc"])
    plt.xlabel("Epoch")
    plt.ylabel("Validation PR-AUC")
    plt.title("PyTorch MLP Validation PR-AUC by Epoch")
    plt.tight_layout()
    plt.savefig(pr_auc_by_epoch_path)
    plt.close()

    # Validation ROC-AUC by Epoch
    plt.figure()
    plt.plot(epoch_frame["epoch"], epoch_frame["validation_roc_auc"])
    plt.xlabel("Epoch")
    plt.ylabel("Validation ROC-AUC")
    plt.title("PyTorch MLP Validation ROC-AUC by Epoch")
    plt.tight_layout()
    plt.savefig(roc_auc_by_epoch_path)
    plt.close()

    # Final-Epoch F1 by Threshold
    final_epoch = int(threshold_frame["epoch"].max())
    final_epoch_thresholds = threshold_frame[
        threshold_frame["epoch"] == final_epoch
    ].copy()

    plt.figure()
    plt.plot(
        final_epoch_thresholds["threshold"],
        final_epoch_thresholds["f1"],
    )
    plt.xlabel("Probability Threshold")
    plt.ylabel("Validation F1")
    plt.title("PyTorch MLP Final-Epoch F1 by Threshold")
    plt.tight_layout()
    plt.savefig(threshold_f1_by_threshold_path)
    plt.close()

    # Prediction of Test dataset
    y_true = test_result["y_true"]
    y_score = test_result["y_score"]

    # Final Test Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PyTorch MLP Final Test Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(final_pr_curve_path)
    plt.close()

    # Final Test ROC Curve
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("PyTorch MLP Final Test ROC Curve")
    plt.tight_layout()
    plt.savefig(final_roc_curve_path)
    plt.close()

    return {
        "loss_by_epoch_path": str(loss_by_epoch_path),
        "pr_auc_by_epoch_path": str(pr_auc_by_epoch_path),
        "roc_auc_by_epoch_path": str(roc_auc_by_epoch_path),
        "threshold_f1_by_threshold_path": str(threshold_f1_by_threshold_path),
        "final_test_precision_recall_curve_path": str(final_pr_curve_path),
        "final_test_roc_curve_path": str(final_roc_curve_path),
    }


def build_run_parameters(
    config: dict[str, Any],
    feature_count: int,
) -> dict[str, object]:
    return {
        "model_type": "torch_mlp_classifier",
        "dataset_type": "cicids2017",
        "split_type": "train_validation_test",
        "random_seed": config["split"]["random_seed"],
        "validation_size": config["split"]["validation_size"],
        "test_size": config["split"]["test_size"],
        "feature_count": feature_count,
        "hidden_dim_1": config["model"]["hidden_dim_1"],
        "hidden_dim_2": config["model"]["hidden_dim_2"],
        "dropout": config["model"]["dropout"],
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "threshold_start": config["thresholds"]["start"],
        "threshold_stop": config["thresholds"]["stop"],
        "threshold_step": config["thresholds"]["step"],
    }


def extract_final_scalar_metrics(
    final_metrics: dict[str, object],
) -> dict[str, float]:
    threshold_independent = final_metrics["threshold_independent_metrics"]
    default_threshold_metrics = final_metrics["default_threshold_metrics"]

    return {
        "test_loss": float(threshold_independent["test_loss"]),
        "test_pr_auc": float(threshold_independent["test_pr_auc"]),
        "test_roc_auc": float(threshold_independent["test_roc_auc"]),
        "test_accuracy": float(default_threshold_metrics["test_accuracy"]),
        "test_precision": float(default_threshold_metrics["test_precision"]),
        "test_recall": float(default_threshold_metrics["test_recall"]),
        "test_f1": float(default_threshold_metrics["test_f1"]),
        "best_validation_pr_auc": float(
            final_metrics["best_validation_pr_auc"]
        ),
        "best_validation_roc_auc": float(
            final_metrics["best_validation_roc_auc"]
        ),
    }


def write_torch_mlp_lineage(
    config: dict[str, Any],
    config_path: Path,
    metrics_path: Path,
    model_path: Path,
    experiment_id: str,
    run_id: str,
) -> Path:
    lineage_input = LineageManifestInput(
        model_name="aegis_hgx_torch_mlp_cic_baseline",
        model_version="torch_mlp_cic_baseline_v1",
        model_type="torch_mlp_classifier",
        model_artifact_path=str(model_path),
        training_data_path=config["data"]["input_path"],
        training_data_dvc_path=str(DEFAULT_TRAINING_DATA_DVC_PATH),
        training_config_path=str(config_path),
        data_generation_config_path=None,
        metrics_path=str(metrics_path),
        mlflow_experiment_name=config["experiment_tracking"]["experiment_name"],
        mlflow_experiment_id=str(experiment_id),
        mlflow_run_id=run_id,
        mlflow_tracking_uri=mlflow.get_tracking_uri(),
        training_entrypoint=(
            "aegis_hgx.models.baselines.train_torch_mlp_cic"
        ),
        training_command=(
            "python -m "
            "aegis_hgx.models.baselines.train_torch_mlp_cic "
            f"--config {config_path}"
        ),
        feature_store_provider="local_snapshot",
        feature_snapshot_id="cicids2017_tabular_features_v1",
        offline_store_path=config["data"]["input_path"],
    )

    lineage_manifest = build_lineage_manifest(lineage_input)

    return write_lineage_manifest(
        lineage_manifest,
        DEFAULT_LINEAGE_MANIFEST_PATH,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to PyTorch MLP training config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment_id = configure_experiment(config)

    with mlflow.start_run(run_name="torch-mlp-cic-baseline") as run:

        dataset = load_dataset(config)
        validate_target_column(dataset, config)

        target_column = config["data"]["target_column"]
        target_counts = dataset[target_column].value_counts().sort_index().to_dict()

        features, target = split_features_and_target(dataset, config)
        train_data, validation_data, test_data = split_train_validation_test(
            features,
            target,
            config,
        )

        X_train, y_train = train_data
        X_validation, y_validation = validation_data
        X_test, y_test = test_data

        scaler, train_tensors, validation_tensors, test_tensors = build_scaled_tensors(
            train_data,
            validation_data,
            test_data,
        )
        train_loader = build_train_loader(train_tensors, config)

        X_train_tensor, y_train_tensor = train_tensors
        X_validation_tensor, y_validation_tensor = validation_tensors
        X_test_tensor, y_test_tensor = test_tensors

        input_dim = X_train_tensor.shape[1]
        model = build_model(input_dim=input_dim, config=config)

        loss_function = build_loss_function(train_tensors)
        optimizer = build_optimizer(model, config)

        """
        # Testing logic for one epoch
        first_epoch_loss = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
        )

        validation_result = evaluate_split(
            model=model,
            tensors=validation_tensors,
            loss_function=loss_function,
        )

        thresholds = build_threshold_list(config)
        threshold_rows = build_epoch_threshold_rows(
            epoch=1,
            y_true=validation_result["y_true"],
            y_score=validation_result["y_score"],
            thresholds=thresholds,
        )
        """

        trained_model, num_of_epochs, epoch_history, num_of_thresholds, threshold_history = train_model(
            model=model,
            train_loader=train_loader,
            validation_tensors=validation_tensors,
            loss_function=loss_function,
            optimizer=optimizer,
            config=config,
        )

        epoch_history_path = save_epoch_history(epoch_history, config)
        threshold_history_path = save_threshold_history(threshold_history, config)

        test_result = evaluate_epoch(
            epoch=0,
            model=trained_model,
            tensors=test_tensors,
            loss_function=loss_function,
        )

        final_metrics = build_final_metrics_from_test_data(
            trained_model=trained_model,
            train_data=train_data,
            validation_data=validation_data,
            test_data=test_data,
            test_tensors=test_tensors,
            loss_function=loss_function,
            epoch_history=epoch_history,
            config=config,
            epoch_history_path=epoch_history_path,
            threshold_history_path=threshold_history_path,
        )
        metrics_path = save_final_metrics(final_metrics, config)


        model_path = save_model_artifact(
            trained_model=trained_model,
            feature_columns=list(features.columns),
            config=config,
        )
        scaler_path = save_scaler_artifact(scaler, config)

        plot_paths = save_training_plots(
            epoch_history=epoch_history,
            threshold_history=threshold_history,
            test_result=test_result,
            config=config,
        )

        parameters = build_run_parameters(
            config=config,
            feature_count=len(features.columns),
        )
        scalar_metrics = extract_final_scalar_metrics(final_metrics)

        mlflow.set_tags(
            {
                "project": "aegis-hgx",
                "model_family": "torch_mlp",
                "dataset_type": "cicids2017",
                "pipeline_stage": "training_diagnostics",
            }
        )

        mlflow.log_params(parameters)
        mlflow.log_metrics(scalar_metrics)

        mlflow.log_artifact(
            str(args.config),
            artifact_path="run_evidence/parameters",
        )
        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="run_evidence/final_metrics",
        )
        mlflow.log_artifact(
            str(epoch_history_path),
            artifact_path="run_evidence/epoch_history",
        )
        mlflow.log_artifact(
            str(threshold_history_path),
            artifact_path="run_evidence/threshold_history",
        )
        mlflow.log_artifact(
            str(model_path),
            artifact_path="run_evidence/model",
        )
        mlflow.log_artifact(
            str(scaler_path),
            artifact_path="run_evidence/scaler",
        )
        mlflow.log_artifacts(
            str(Path(config["outputs"]["figure_dir"])),
            artifact_path="run_evidence/figures",
        )

        lineage_manifest_path = write_torch_mlp_lineage(
            config=config,
            config_path=args.config,
            metrics_path=metrics_path,
            model_path=model_path,
            experiment_id=experiment_id,
            run_id=run.info.run_id,
        )

        mlflow.log_artifact(
            str(lineage_manifest_path),
            artifact_path="run_evidence/lineage",
        )

        print()
        print("Config path:", args.config)
        print("Input path:", config["data"]["input_path"])
        print("Target column:", config["data"]["target_column"])
        print("Metrics path:", config["outputs"]["metrics_path"])
        print()
        print("Epoch history path:", config["outputs"]["epoch_history_path"])
        print("Epoch Threshold history path:", config["outputs"]["threshold_history_path"])
        print("Model path:", config["outputs"]["model_path"])
        print("Scaler path:", config["outputs"]["scaler_path"])
        print("Experiment:", config["experiment_tracking"]["experiment_name"])
        print()
        print("Columns:", len(dataset.columns))
        print("Feature columns:", len(features.columns))
        print("Target counts:", target_counts)
        print()
        #print("First features:", list(features.columns[:10]))
        print("Rows:", len(dataset))
        print("Train rows:", len(X_train))
        print("Validation rows:", len(X_validation))
        print("Test rows:", len(X_test))
        print("Train target counts:", y_train.value_counts().sort_index().to_dict())
        print(
            "Validation target counts:",
            y_validation.value_counts().sort_index().to_dict(),
        )
        print("Test target counts:", y_test.value_counts().sort_index().to_dict())
        print()
        print("Train tensor shape:", tuple(X_train_tensor.shape))
        print("Validation tensor shape:", tuple(X_validation_tensor.shape))
        print("Test tensor shape:", tuple(X_test_tensor.shape))
        print("Train label tensor shape:", tuple(y_train_tensor.shape))
        print("Train batches:", len(train_loader))
        print()
        print("Input dimension:", input_dim)
        print()
        print("Training begins ...")
        print("Number of epochs:", num_of_epochs)
        print("Number of thresholds:", num_of_thresholds)
        print("Epoch history rows:", len(epoch_history))
        print("Threshold history rows:", len(threshold_history), "(Number of epochs x thresholds)")
        for i, epoch_history_row in enumerate(epoch_history):
            #print(f"Epoch {i}: {epoch_history_row}")
            pass
        print()
        print("Final metrics manifest path:", metrics_path)
        print("Model path:", model_path)
        print("Scaler path:", scaler_path)
        print()
        print("Experiment ID:", experiment_id)
        print("Run ID:", run.info.run_id)
        print("Artifact URI:", run.info.artifact_uri)
        print("Lineage manifest:", lineage_manifest_path)


if __name__ == "__main__":
    main()