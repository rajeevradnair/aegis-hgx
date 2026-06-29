from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
import json

from aegis_hgx.models.baselines.config_schema import (
    TrainingConfig,
    validate_training_config,
)
from aegis_hgx.models.baselines.train_logistic_baseline import (
    load_dataset,
    load_training_config,
    split_features_and_target,
    split_train_test,
)


DEFAULT_CONFIG_PATH = Path("configs/baseline_logistic.yaml")
DEFAULT_OUTPUT_REPORT_PATH = Path(
    "reports/evaluation/threshold_tuning_report.json"
)

def load_threshold_tuning_inputs(
    config_path: Path,
) -> tuple[TrainingConfig, Pipeline, pd.DataFrame, pd.Series]:
    raw_config = load_training_config(str(config_path))
    config = validate_training_config(raw_config)

    dataset = load_dataset(config)
    features, target = split_features_and_target(dataset, config)
    _, test_data = split_train_test(features, target, config)

    model_path = Path(config.outputs.model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. "
            "Run baseline training before threshold tuning."
        )

    model = joblib.load(model_path)

    if not isinstance(model, Pipeline):
        raise TypeError(
            f"Expected sklearn Pipeline, got {type(model).__name__}."
        )

    X_test, y_test = test_data

    return config, model, X_test, y_test

def evaluate_thresholds(
    y_true: pd.Series,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict[str, Any]]:
    threshold_rows = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        true_positives = int(
            ((y_true == 1) & (predictions == 1)).sum()
        )
        false_positives = int(
            ((y_true == 0) & (predictions == 1)).sum()
        )
        true_negatives = int(
            ((y_true == 0) & (predictions == 0)).sum()
        )
        false_negatives = int(
            ((y_true == 1) & (predictions == 0)).sum()
        )

        row = {
            "threshold": float(round(threshold, 4)),
            "precision": float(
                precision_score(y_true, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, predictions, zero_division=0)
            ),
            "f1": float(
                f1_score(y_true, predictions, zero_division=0)
            ),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "alert_count": int(predictions.sum()),
        }

        threshold_rows.append(row)

    recommended_row = max(
        threshold_rows,
        key=lambda row: (row["f1"], row["recall"], row["precision"])
    )

    return {
        "selection_strategy": "max_f1_then_recall_then_precision",
        "recommended_threshold": recommended_row["threshold"],
        "recommended_row": recommended_row,
        "thresholds": threshold_rows,
    }

def write_threshold_report(
    report: dict[str, Any],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)

    return output_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune alert thresholds for a trained classifier."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the training config YAML file.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT_PATH,
        help="Path where the threshold tuning report should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config, model, X_test, y_test = load_threshold_tuning_inputs(
        args.config
    )

    probabilities = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.05, 1.0, 0.05)
    threshold_report = evaluate_thresholds(
        y_true=y_test,
        probabilities=probabilities,
        thresholds=thresholds,
    )

    print("Config:", args.config)
    print("Model path:", config.outputs.model_path)
    print("Test rows:", len(y_test))
    print("Test feature columns:", list(X_test.columns))
    print("Output report:", args.output_report)
    print(
        "Thresholds evaluated:",
        len(threshold_report["thresholds"]),
    )
    print(
        "Recommended threshold:",
        threshold_report["recommended_threshold"],
    )
    print(
        "Recommended row:",
        threshold_report["recommended_row"],
    )

    output_report_path = write_threshold_report(
        report=threshold_report,
        output_path=args.output_report,
    )
    print("Threshold report saved:", output_report_path)

if __name__ == "__main__":
    main()