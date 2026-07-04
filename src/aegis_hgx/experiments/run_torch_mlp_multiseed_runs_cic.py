from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import pandas as pd
import yaml
import random
import json

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score

CONFIG_PATH = "configs/torch_mlp_multiseed_runs_cic.yaml"


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Seed-run config not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Seed-run config must contain a YAML mapping.")

    return config


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
            "Seed-run experiments require at least two target classes. "
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


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_train_validation_test(
    features: pd.DataFrame,
    target: pd.Series,
    seed: int,
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
        random_state=seed,
        stratify=target,
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=split_config["validation_size"],
        random_state=seed,
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
    seed: int,
    config: dict[str, Any],
) -> DataLoader:
    X_train_tensor, y_train_tensor = train_tensors

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
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


def build_model(input_dim: int, config: dict[str, Any]) -> TabularMLP:
    model_config = config["model"]

    return TabularMLP(
        input_dim=input_dim,
        hidden_dim_1=int(model_config["hidden_dim_1"]),
        hidden_dim_2=int(model_config["hidden_dim_2"]),
        dropout=float(model_config["dropout"]),
    )

def calculate_pos_weight(target_tensor: torch.Tensor) -> torch.Tensor:
    positive_count = target_tensor.sum()
    total_count = torch.tensor(float(target_tensor.numel()))
    negative_count = total_count - positive_count

    if positive_count.item() == 0:
        raise ValueError("Training target contains no positive examples.")

    return negative_count / positive_count


def build_loss_function(
    train_tensors: tuple[torch.Tensor, torch.Tensor],
) -> nn.BCEWithLogitsLoss:
    _, y_train_tensor = train_tensors
    pos_weight = calculate_pos_weight(y_train_tensor)

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def build_optimizer(
    model: TabularMLP,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    training_config = config["training"]

    return torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )


def train_one_epoch(
    model: TabularMLP,
    train_loader: DataLoader,
    loss_function: nn.BCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()

    total_loss = 0.0
    total_rows = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        logits = model(X_batch)
        loss = loss_function(logits, y_batch)

        loss.backward()
        optimizer.step()

        batch_rows = X_batch.shape[0]
        total_loss += float(loss.item()) * batch_rows
        total_rows += batch_rows

    return float(total_loss / total_rows)


def train_model_for_seed(
    model: TabularMLP,
    train_loader: DataLoader,
    loss_function: nn.BCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> tuple[TabularMLP, float]:
    epochs = int(config["training"]["epochs"])
    final_train_loss = 0.0

    for _ in range(epochs):
        final_train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
        )

    return model, final_train_loss


def calculate_threshold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (y_score >= threshold).astype(int)

    true_positive = int(((y_pred == 1) & (y_true == 1)).sum())
    false_positive = int(((y_pred == 1) & (y_true == 0)).sum())
    true_negative = int(((y_pred == 0) & (y_true == 0)).sum())
    false_negative = int(((y_pred == 0) & (y_true == 1)).sum())

    total = true_positive + false_positive + true_negative + false_negative

    accuracy = (
        (true_positive + true_negative) / total
        if total > 0
        else 0.0
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def evaluate_model(
    model: TabularMLP,
    test_tensors: tuple[torch.Tensor, torch.Tensor],
    loss_function: nn.BCEWithLogitsLoss,
    threshold: float,
) -> dict[str, float | int]:
    X_test_tensor, y_test_tensor = test_tensors

    model.eval()

    with torch.no_grad():
        logits = model(X_test_tensor)
        loss = loss_function(logits, y_test_tensor)
        probabilities = torch.sigmoid(logits)

    y_true = y_test_tensor.detach().cpu().numpy().reshape(-1).astype(int)
    y_score = probabilities.detach().cpu().numpy().reshape(-1)

    threshold_metrics = calculate_threshold_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=threshold,
    )

    return {
        "test_loss": float(loss.item()),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        **threshold_metrics,
    }


def run_single_seed(
    seed: int,
    features: pd.DataFrame,
    target: pd.Series,
    config: dict[str, Any],
) -> dict[str, float | int]:
    set_all_seeds(seed)

    train_data, validation_data, test_data = split_train_validation_test(
        features=features,
        target=target,
        seed=seed,
        config=config,
    )

    X_train, y_train = train_data
    X_validation, y_validation = validation_data
    X_test, y_test = test_data

    _, train_tensors, validation_tensors, test_tensors = build_scaled_tensors(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
    )

    train_loader = build_train_loader(
        train_tensors=train_tensors,
        seed=seed,
        config=config,
    )

    X_train_tensor, _ = train_tensors
    input_dim = X_train_tensor.shape[1]

    model = build_model(input_dim=input_dim, config=config)
    loss_function = build_loss_function(train_tensors)
    optimizer = build_optimizer(model=model, config=config)

    trained_model, final_train_loss = train_model_for_seed(
        model=model,
        train_loader=train_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        config=config,
    )

    threshold = float(config["evaluation"]["threshold"])
    test_metrics = evaluate_model(
        model=trained_model,
        test_tensors=test_tensors,
        loss_function=loss_function,
        threshold=threshold,
    )

    return {
        "seed": int(seed),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),
        "train_positive_rows": int(y_train.sum()),
        "validation_positive_rows": int(y_validation.sum()),
        "test_positive_rows": int(y_test.sum()),
        "final_train_loss": float(final_train_loss),
        **test_metrics,
    }

def run_all_seeds(
    features: pd.DataFrame,
    target: pd.Series,
    config: dict[str, Any],
) -> list[dict[str, float | int]]:
    seed_rows = []

    for seed in config["experiment"]["seeds"]:
        seed_value = int(seed)
        print(f"Running seed: {seed_value}")

        seed_row = run_single_seed(
            seed=seed_value,
            features=features,
            target=target,
            config=config,
        )
        seed_rows.append(seed_row)

        print(
            "Seed result:",
            {
                "seed": seed_row["seed"],
                "pr_auc": seed_row["pr_auc"],
                "roc_auc": seed_row["roc_auc"],
                "f1": seed_row["f1"],
            },
        )

    return seed_rows


def save_per_seed_metrics(
    seed_rows: list[dict[str, float | int]],
    config: dict[str, Any],
) -> Path:
    metrics_path = Path(config["outputs"]["per_seed_metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "seed",
        "train_rows",
        "validation_rows",
        "test_rows",
        "train_positive_rows",
        "validation_positive_rows",
        "test_positive_rows",
        "final_train_loss",
        "test_loss",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]

    frame = pd.DataFrame(seed_rows)
    frame = frame[columns]
    frame.to_csv(metrics_path, index=False)

    return metrics_path


def summarize_metric(values: pd.Series) -> dict[str, float | int]:
    metric_values = values.astype(float)
    count = int(metric_values.count())

    if count == 0:
        raise ValueError("Cannot summarize an empty metric series.")

    mean_value = float(metric_values.mean())
    std_value = float(metric_values.std(ddof=1)) if count > 1 else 0.0
    min_value = float(metric_values.min())
    max_value = float(metric_values.max())
    standard_error = float(std_value / np.sqrt(count)) if count > 1 else 0.0

    ci95_margin = float(1.96 * standard_error)

    return {
        "count": count,
        "mean": mean_value,
        "std": std_value,
        "min": min_value,
        "max": max_value,
        "standard_error": standard_error,
        "ci95_lower": mean_value - ci95_margin,
        "ci95_upper": mean_value + ci95_margin,
    }


def build_seed_summary(
    seed_rows: list[dict[str, float | int]],
    config: dict[str, Any],
) -> dict[str, object]:
    frame = pd.DataFrame(seed_rows)

    metrics_to_summarize = [
        "final_train_loss",
        "test_loss",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]

    metric_summaries = {
        metric_name: summarize_metric(frame[metric_name])
        for metric_name in metrics_to_summarize
    }

    return {
        "model_name": config["experiment"]["model_name"],
        "dataset": config["experiment"]["dataset"],
        "experiment_type": "multi_seed_stability",
        "evaluated_split": "test",
        "seed_count": int(len(seed_rows)),
        "seeds": [int(seed) for seed in config["experiment"]["seeds"]],
        "split": {
            "validation_size": float(config["split"]["validation_size"]),
            "test_size": float(config["split"]["test_size"]),
        },
        "training": {
            "epochs": int(config["training"]["epochs"]),
            "batch_size": int(config["training"]["batch_size"]),
            "learning_rate": float(config["training"]["learning_rate"]),
            "weight_decay": float(config["training"]["weight_decay"]),
        },
        "evaluation": {
            "threshold": float(config["evaluation"]["threshold"]),
        },
        "metrics": metric_summaries,
        "artifact_paths": {
            "per_seed_metrics_path": config["outputs"]["per_seed_metrics_path"],
            "summary_metrics_path": config["outputs"]["summary_metrics_path"],
        },
    }

def save_seed_summary(
    summary: dict[str, object],
    config: dict[str, Any],
) -> Path:
    summary_path = Path(config["outputs"]["summary_metrics_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to PyTorch MLP seed-run config YAML file.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset = load_dataset(config)
    validate_target_column(dataset, config)
    features, target = split_features_and_target(dataset, config)

    print("Config path:", args.config)
    print("Input path:", config["data"]["input_path"])
    print("Target column:", config["data"]["target_column"])
    print("Model name:", config["experiment"]["model_name"])
    print("Dataset:", config["experiment"]["dataset"])
    print("Seeds:", config["experiment"]["seeds"])
    print("Rows:", len(dataset))
    print("Feature columns:", len(features.columns))
    print("Target counts:", target.value_counts().sort_index().to_dict())

    seed_rows = run_all_seeds(
        features=features,
        target=target,
        config=config,
    )
    per_seed_metrics_path = save_per_seed_metrics(
        seed_rows=seed_rows,
        config=config,
    )

    seed_summary = build_seed_summary(
        seed_rows=seed_rows,
        config=config,
    )
    summary_metrics_path = save_seed_summary(
        summary=seed_summary,
        config=config,
    )

    print("Per-seed metrics path:", per_seed_metrics_path)
    print("Seed run count:", len(seed_rows))
    print("Per-seed metrics path:", per_seed_metrics_path)
    print("Summary metrics path:", summary_metrics_path)
    print("Seed run count:", len(seed_rows))
    print(
        "Mean PR-AUC:",
        seed_summary["metrics"]["pr_auc"]["mean"],
    )
    print(
        "PR-AUC std:",
        seed_summary["metrics"]["pr_auc"]["std"],
    )
    print(
        "Mean F1:",
        seed_summary["metrics"]["f1"]["mean"],
    )

if __name__ == "__main__":
    main()