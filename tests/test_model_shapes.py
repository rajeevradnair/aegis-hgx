from pathlib import Path

import pandas as pd

from src.aegis_hgx.models.baselines.config_schema import (
    TrainingConfig,
    validate_training_config,
)
from src.aegis_hgx.models.baselines.train_logistic_baseline import (
    split_features_and_target,
    split_train_test,
    build_model,
    train_model,
)

def create_shape_test_dataset() -> pd.DataFrame:
    rows = []


    for index in range(50):
        suspicious = index >= 40

        rows.append(
            {
                "user_id": f"user_{index % 5}",
                "host_id": f"host_{index % 4}",
                "process_name": "mimikatz" if suspicious else "chrome",
                "event_type": (
                    "privilege_change" if suspicious else "login_success"
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


def create_shape_test_config(tmp_path: Path) -> TrainingConfig:
    raw_config = {
        "data": {
            "input_path": str(tmp_path / "synthetic_events.csv"),
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
        "experiment_tracking": {
            "experiment_name": "test-model-shapes",
            "uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "artifact_root": str(tmp_path / "mlruns"),
        },
    }

    return validate_training_config(raw_config)

def test_feature_target_shapes_match_config(tmp_path: Path) -> None:
    dataset = create_shape_test_dataset()
    config = create_shape_test_config(tmp_path)

    features, target = split_features_and_target(dataset, config)

    expected_feature_columns = (
        config.data.numeric_features
        + config.data.categorical_features
    )

    assert features.shape[0] == dataset.shape[0]
    assert target.shape[0] == dataset.shape[0]
    assert list(features.columns) == expected_feature_columns
    assert target.name == config.data.target_column
    assert config.data.target_column not in features.columns

def test_train_test_split_shapes_are_consistent(tmp_path: Path) -> None:
    dataset = create_shape_test_dataset()
    config = create_shape_test_config(tmp_path)

    assert not dataset.empty
    assert dataset.shape[0] == 50

    features, target = split_features_and_target(dataset, config)
    train_data, test_data = split_train_test(features, target, config)

    X_train, y_train = train_data
    X_test, y_test = test_data

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[0] + X_test.shape[0] == dataset.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == target.shape[0]
    assert X_test.shape[0] == int(dataset.shape[0] * config.split.test_size)

def test_trained_model_prediction_shape_matches_test_rows(
    tmp_path: Path,
) -> None:
    dataset = create_shape_test_dataset()
    config = create_shape_test_config(tmp_path)

    features, target = split_features_and_target(dataset, config)
    train_data, test_data = split_train_test(features, target, config)

    model = build_model(config)
    trained_model = train_model(model, train_data)

    X_test, y_test = test_data
    predictions = trained_model.predict(X_test)

    assert len(predictions) == X_test.shape[0]
    assert len(predictions) == y_test.shape[0]
    assert set(predictions).issubset({0, 1})