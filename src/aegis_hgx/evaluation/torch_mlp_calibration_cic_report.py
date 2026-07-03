from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import argparse
import joblib
import numpy as np

import yaml
import pandas as pd

import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

import matplotlib.pyplot as plt


CONFIG_PATH = "configs/calibration_cic.yaml"


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Calibration config not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Calibration config must contain a YAML mapping.")

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
            "Calibration analysis requires at least two target classes. "
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to calibration config YAML file.",
    )
    return parser.parse_args()

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


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    required_keys = {
        "model_state_dict",
        "model_name",
        "input_dim",
        "hidden_dim_1",
        "hidden_dim_2",
        "dropout",
        "feature_columns",
        "target_column",
    }

    missing_keys = required_keys.difference(checkpoint.keys())

    if missing_keys:
        raise ValueError(
            "Model checkpoint is missing required keys: "
            f"{sorted(missing_keys)}"
        )
    

def load_model_and_scaler(
    config: dict[str, Any],
) -> tuple[TabularMLP, StandardScaler, dict[str, Any]]:
    model_path = Path(config["model"]["model_path"])
    scaler_path = Path(config["model"]["scaler_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    validate_checkpoint(checkpoint)

    model = TabularMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim_1=int(checkpoint["hidden_dim_1"]),
        hidden_dim_2=int(checkpoint["hidden_dim_2"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = joblib.load(scaler_path)

    if not isinstance(scaler, StandardScaler):
        raise TypeError(
            "Scaler artifact must be a sklearn StandardScaler. "
            f"Found: {type(scaler)}"
        )

    return model, scaler, checkpoint


def validate_feature_columns(
    features: pd.DataFrame,
    checkpoint: dict[str, Any],
) -> list[str]:
    expected_columns = list(checkpoint["feature_columns"])
    actual_columns = list(features.columns)

    missing_columns = sorted(set(expected_columns).difference(actual_columns))
    extra_columns = sorted(set(actual_columns).difference(expected_columns))

    if missing_columns or extra_columns:
        raise ValueError(
            "Feature columns do not match checkpoint metadata. "
            f"Missing columns: {missing_columns}. "
            f"Extra columns: {extra_columns}."
        )

    return expected_columns


def generate_test_probabilities(
    model: TabularMLP,
    scaler: StandardScaler,
    checkpoint: dict[str, Any],
    test_data: tuple[pd.DataFrame, pd.Series],
) -> tuple[np.ndarray, np.ndarray]:
    X_test, y_test = test_data

    feature_columns = validate_feature_columns(X_test, checkpoint)
    X_test_ordered = X_test[feature_columns]

    X_test_scaled = scaler.transform(X_test_ordered)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probabilities = torch.sigmoid(logits)

    y_true = y_test.to_numpy().astype(int)
    y_probability = probabilities.detach().cpu().numpy().reshape(-1)

    return y_true, y_probability


def compute_brier_score(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> float:
    return float(brier_score_loss(y_true, y_probability))

def build_calibration_bins(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    config: dict[str, Any],
) -> list[dict[str, object]]:
    n_bins = int(config["calibration"]["n_bins"])
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    total_rows = len(y_true)

    for bin_index in range(n_bins):
        lower_bound = float(bin_edges[bin_index])
        upper_bound = float(bin_edges[bin_index + 1])

        if bin_index == n_bins - 1:
            in_bin = (
                (y_probability >= lower_bound)
                & (y_probability <= upper_bound)
            )
        else:
            in_bin = (
                (y_probability >= lower_bound)
                & (y_probability < upper_bound)
            )

        bin_true = y_true[in_bin]
        bin_probability = y_probability[in_bin]
        bin_count = int(len(bin_true))

        if bin_count == 0:
            mean_predicted_probability = None
            observed_positive_rate = None
            absolute_calibration_gap = None
        else:
            mean_predicted_probability = float(bin_probability.mean())
            observed_positive_rate = float(bin_true.mean())
            absolute_calibration_gap = abs(
                mean_predicted_probability - observed_positive_rate
            )

        rows.append(
            {
                "bin_index": bin_index,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "sample_count": bin_count,
                "sample_fraction": (
                    float(bin_count / total_rows)
                    if total_rows > 0
                    else 0.0
                ),
                "mean_predicted_probability": mean_predicted_probability,
                "observed_positive_rate": observed_positive_rate,
                "absolute_calibration_gap": absolute_calibration_gap,
            }
        )

    return rows


def compute_expected_calibration_error(
    calibration_bins: list[dict[str, object]],
) -> float:
    error = 0.0

    for row in calibration_bins:
        gap = row["absolute_calibration_gap"]

        if gap is None:
            continue

        error += float(row["sample_fraction"]) * float(gap)

    return float(error)


def build_calibration_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    brier_score: float,
    expected_calibration_error: float,
    config: dict[str, Any],
) -> dict[str, object]:
    return {
        "model_name": "torch_mlp_cic_baseline",
        "model_family": config["model"]["model_family"],
        "dataset": "cicids2017",
        "split_type": "train_validation_test",
        "evaluated_split": "test",
        "test_rows": int(len(y_true)),
        "test_positive_rows": int(y_true.sum()),
        "test_negative_rows": int(len(y_true) - y_true.sum()),
        "calibration": {
            "brier_score": float(brier_score),
            "expected_calibration_error": float(
                expected_calibration_error
            ),
            "n_bins": int(config["calibration"]["n_bins"]),
            "strategy": config["calibration"]["strategy"],
        },
        "probability_summary": {
            "min_probability": float(y_probability.min()),
            "max_probability": float(y_probability.max()),
            "mean_probability": float(y_probability.mean()),
            "median_probability": float(np.median(y_probability)),
        },
        "artifact_paths": {
            "metrics_path": config["outputs"]["metrics_path"],
            "bins_path": config["outputs"]["bins_path"],
            "reliability_diagram_path": config["outputs"][
                "reliability_diagram_path"
            ],
            "probability_histogram_path": config["outputs"][
                "probability_histogram_path"
            ],
            "model_path": config["model"]["model_path"],
            "scaler_path": config["model"]["scaler_path"],
        },
    }


def save_calibration_metrics(
    metrics: dict[str, object],
    config: dict[str, Any],
) -> Path:
    metrics_path = Path(config["outputs"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    return metrics_path


def save_calibration_bins(
    calibration_bins: list[dict[str, object]],
    config: dict[str, Any],
) -> Path:
    bins_path = Path(config["outputs"]["bins_path"])
    bins_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "bin_index",
        "lower_bound",
        "upper_bound",
        "sample_count",
        "sample_fraction",
        "mean_predicted_probability",
        "observed_positive_rate",
        "absolute_calibration_gap",
    ]

    bins_frame = pd.DataFrame(calibration_bins)
    bins_frame = bins_frame[columns]
    bins_frame.to_csv(bins_path, index=False)

    return bins_path


def save_calibration_plots(
    calibration_bins: list[dict[str, object]],
    y_probability: np.ndarray,
    config: dict[str, Any],
) -> dict[str, str]:
    reliability_diagram_path = Path(
        config["outputs"]["reliability_diagram_path"]
    )
    probability_histogram_path = Path(
        config["outputs"]["probability_histogram_path"]
    )

    reliability_diagram_path.parent.mkdir(parents=True, exist_ok=True)
    probability_histogram_path.parent.mkdir(parents=True, exist_ok=True)

    bins_frame = pd.DataFrame(calibration_bins)
    print(bins_frame)
    non_empty_bins = bins_frame[
        bins_frame["sample_count"] > 0
    ].copy()

    plt.figure()
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="perfect_calibration")
    plt.plot(
        non_empty_bins["mean_predicted_probability"],
        non_empty_bins["observed_positive_rate"],
        marker="o",
        label="model_calibration",
    )
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Positive Rate")
    plt.title("PyTorch MLP Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reliability_diagram_path)
    plt.close()

    plt.figure()
    plt.hist(y_probability, bins=20)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Sample Count")
    plt.title("PyTorch MLP Probability Histogram")
    plt.tight_layout()
    plt.savefig(probability_histogram_path)
    plt.close()

    return {
        "reliability_diagram_path": str(reliability_diagram_path),
        "probability_histogram_path": str(probability_histogram_path),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset = load_dataset(config)
    validate_target_column(dataset, config)

    features, target = split_features_and_target(dataset, config)
    train_data, validation_data, test_data = split_train_validation_test(
        features,
        target,
        config,
    )

    X_train, y_train = train_data
    X_validation, y_validation = validation_data
    X_test, y_test = test_data

    model, scaler, checkpoint = load_model_and_scaler(config)

    y_true, y_probability = generate_test_probabilities(
        model=model,
        scaler=scaler,
        checkpoint=checkpoint,
        test_data=test_data,
    )

    brier_score = compute_brier_score(
        y_true=y_true,
        y_probability=y_probability,
    )
    calibration_bins = build_calibration_bins(
        y_true=y_true,
        y_probability=y_probability,
        config=config,
    )
    expected_calibration_error = compute_expected_calibration_error(
        calibration_bins
    )

    calibration_metrics = build_calibration_metrics(
        y_true=y_true,
        y_probability=y_probability,
        brier_score=brier_score,
        expected_calibration_error=expected_calibration_error,
        config=config,
    )

    metrics_path = save_calibration_metrics(
        metrics=calibration_metrics,
        config=config,
    )
    bins_path = save_calibration_bins(
        calibration_bins=calibration_bins,
        config=config,
    )

    plot_paths = save_calibration_plots(
        calibration_bins=calibration_bins,
        y_probability=y_probability,
        config=config,
    )

    print("Config path:", args.config)
    print("Input path:", config["data"]["input_path"])
    print("Target column:", config["data"]["target_column"])
    print("Model path:", config["model"]["model_path"])
    print("Scaler path:", config["model"]["scaler_path"])
    print("Model family:", config["model"]["model_family"])
    print("Calibration bins:", config["calibration"]["n_bins"])
    print("Calibration strategy:", config["calibration"]["strategy"])
    print("Metrics path:", config["outputs"]["metrics_path"])
    print("Bins path:", config["outputs"]["bins_path"])
    print(
        "Reliability diagram path:",
        config["outputs"]["reliability_diagram_path"],
    )
    print(
        "Probability histogram path:",
        config["outputs"]["probability_histogram_path"],
    )
    print("Rows:", len(dataset))
    print("Columns:", len(dataset.columns))
    print("Feature columns:", len(features.columns))
    print("Train rows:", len(X_train))
    print("Validation rows:", len(X_validation))
    print("Test rows:", len(X_test))
    print("Train target counts:", y_train.value_counts().sort_index().to_dict())
    print(
        "Validation target counts:",
        y_validation.value_counts().sort_index().to_dict(),
    )
    print("Test target counts:", y_test.value_counts().sort_index().to_dict())
    print("Loaded model:", checkpoint["model_name"])
    print("Checkpoint input dim:", checkpoint["input_dim"])
    print("Checkpoint feature count:", len(checkpoint["feature_columns"]))
    print("Scaler feature count:", len(scaler.mean_))
    print("Test probability rows:", len(y_probability))
    print("Test positive rows:", int(y_true.sum()))
    #print("Predicted probability:", y_probability)
    #print("Ground truth:", y_true)
    print("Min probability:", float(y_probability.min()))
    print("Max probability:", float(y_probability.max()))
    print("Mean probability:", float(y_probability.mean()))
    print("Brier score:", brier_score)
    print("Expected calibration error:", expected_calibration_error)
    print("Calibration bin count:", len(calibration_bins))
    #print("Calibration bins:", calibration_bins)
    print("First calibration bin:", calibration_bins[0])
    print("Last calibration bin:", calibration_bins[-1])
    print("Calibration metrics path:", metrics_path)
    print("Calibration bins path:", bins_path)
    print(
        "Reliability diagram path:",
        plot_paths["reliability_diagram_path"],
    )
    print(
        "Probability histogram path:",
        plot_paths["probability_histogram_path"],
    )


if __name__ == "__main__":
    main()