import json
from pathlib import Path

import pandas as pd

from src.aegis_hgx.models.baselines.train_logistic_baseline import (
    build_model,
    evaluate_model,
    save_metrics,
    split_features_and_target,
    split_train_test,
    train_model,
)


def create_dataset() -> pd.DataFrame:
    rows = []

    for index in range(50):
        suspicious = index >= 40

        rows.append(
            {
                "user_id": f"user_{index % 5}",
                "host_id": f"host_{index % 4}",
                "process_name": (
                    "mimikatz" if suspicious else "chrome"
                ),
                "event_type": (
                    "privilege_change"
                    if suspicious
                    else "login_success"
                ),
                "source_ip": f"10.0.0.{index % 10 + 1}",
                "destination_ip": (
                    f"203.0.113.{index}"
                    if suspicious
                    else f"10.0.1.{index}"
                ),
                "bytes_in": 500,
                "bytes_out": 50_000 if suspicious else 800,
                "event_hour": 2 if suspicious else 10,
                "is_business_hour": not suspicious,
                "label": int(suspicious),
            }
        )

    return pd.DataFrame(rows)


def create_config(tmp_path: Path) -> dict:
    return {
        "data": {
            "target_column": "label",
            "numeric_features": [
                "bytes_in",
                "bytes_out",
                "event_hour",
                "is_business_hour",
            ],
            "categorical_features": [
                "user_id",
                "host_id",
                "process_name",
                "event_type",
                "source_ip",
                "destination_ip",
            ],
        },
        "split": {
            "test_size": 0.2,
            "random_seed": 42,
        },
        "model": {
            "max_iter": 1000,
            "class_weight": "balanced",
        },
        "outputs": {
            "metrics_path": str(tmp_path / "metrics.json"),
            "model_path": str(tmp_path / "model.joblib"),
        },
    }


def test_baseline_training_pipeline(tmp_path: Path) -> None:
    config = create_config(tmp_path)
    dataset = create_dataset()

    features, target = split_features_and_target(dataset, config)
    train_data, test_data = split_train_test(features, target, config)

    model = build_model(config)
    trained_model = train_model(model, train_data)
    metrics = evaluate_model(trained_model, test_data)

    assert metrics["test_rows"] == 10
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2


def test_metrics_are_saved(tmp_path: Path) -> None:
    config = create_config(tmp_path)
    metrics = {"f1": 0.95}

    saved_path = save_metrics(metrics, config)

    assert saved_path.exists()

    saved_metrics = json.loads(saved_path.read_text())
    assert saved_metrics == metrics