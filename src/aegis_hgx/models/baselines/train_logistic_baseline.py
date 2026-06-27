from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse

import pandas as pd
import numpy as np
import yaml
import json
import shutil

import mlflow
import mlflow.sklearn

import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt


CONFIG_PATH = "configs/baseline_logistic.yaml"

def load_training_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Training config must contain a YAML mapping.")

    return config


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    dataset_path = Path(config["data"]["input_path"])

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = pd.read_csv(dataset_path)

    if dataset.empty:
        raise ValueError("Dataset contains no rows.")

    return dataset


def split_features_and_target(
    dataset: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    data_config = config["data"]

    target_column = data_config["target_column"]
    feature_columns = (
        data_config["numeric_features"]
        + data_config["categorical_features"]
    )

    required_columns = feature_columns + [target_column]
    missing_columns = [
        column for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    features = dataset[feature_columns].copy()
    target = dataset[target_column].copy()

    return features, target


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
    config: dict[str, Any],
) -> tuple[
    tuple[pd.DataFrame, pd.Series],
    tuple[pd.DataFrame, pd.Series],
]:
    split_config = config["split"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=split_config["test_size"],
        random_state=split_config["random_seed"],
        stratify=target,
    )

    return (X_train, y_train), (X_test, y_test)


def build_model(config: dict[str, Any]) -> Pipeline:
    data_config = config["data"]
    model_config = config["model"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                data_config["numeric_features"],
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                data_config["categorical_features"],
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=model_config["max_iter"],
        class_weight=model_config["class_weight"],
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

def train_model(
    model: Pipeline,
    train_data: tuple[pd.DataFrame, pd.Series],
) -> Pipeline:
    X_train, y_train = train_data

    model.fit(X_train, y_train)

    return model

def evaluate_model(
    trained_model: Pipeline,
    test_data: tuple[pd.DataFrame, pd.Series],
) -> dict[str, Any]:
    X_test, y_test = test_data

    predictions = trained_model.predict(X_test)
    probabilities = trained_model.predict_proba(X_test)[:, 1]

    return {
        "test_rows": len(y_test),
        "positive_rows": int(y_test.sum()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
    }


def save_metrics(
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    metrics_path = Path(config["outputs"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    return metrics_path


def save_model(
    trained_model: Pipeline,
    config: dict[str, Any],
) -> Path:
    model_path = Path(config["outputs"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(trained_model, model_path)

    return model_path

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

def build_run_parameters(
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_type": "logistic_regression",
        "random_seed": config["split"]["random_seed"],
        "test_size": config["split"]["test_size"],
        "max_iter": config["model"]["max_iter"],
        "class_weight": config["model"]["class_weight"],
        "numeric_feature_count": len(
            config["data"]["numeric_features"]
        ),
        "categorical_feature_count": len(
            config["data"]["categorical_features"]
        ),
    }

def extract_scalar_metrics(
    metrics: dict[str, Any],
) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in metrics.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
    }

def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to experiment config YAML file",
    )
    return arg_parser.parse_args()

def main() -> None:

    args = parse_args()
    config_path = args.config
    config = load_training_config(config_path)

    experiment_id = configure_experiment(config)
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run(run_name="logistic-baseline") as run:

        mlflow.set_tags(
            {
                "project": "aegis-hgx",
                "model_family": "logistic_regression",
                "dataset_type": "synthetic",
                "pipeline_stage": "baseline",
            }
        )
        dataset = load_dataset(config=config)

        features, target = split_features_and_target( dataset,config,)

        train_data, test_data = split_train_test(features, target, config,)

        model = build_model(config)

        trained_model = train_model(model, train_data)
        
        metrics = evaluate_model(trained_model, test_data)

        parameters = build_run_parameters(config)
        scalar_metrics = extract_scalar_metrics(metrics)

        # save model & metrics in local fs
        metrics_path = save_metrics(metrics, config)
        model_path = save_model(trained_model, config)

        print()
        print(f"Saving in local filesystem")
        print(f"Local metrics path: {metrics_path}")
        print(f"Local model path: {model_path}")

        # save input parameters, metrics in mlflow.db
        print()
        print("Experiment ID:", experiment_id)
        print("Run ID initiated:", run.info.run_id)
        print("Artifact URI:", run.info.artifact_uri)

        print(f"Saving in mlflow - mlflow.db and {run.info.artifact_uri}")
        #save params and metrics
        mlflow.log_params(parameters)
        mlflow.log_metrics(scalar_metrics)
        print(f"Input parameters and output metrics logged in mlflow.db")

        # save config_yaml as evidence in mlflow/mlruns
        mlflow.log_artifact(
            str(config_path), 
            artifact_path="run_evidence/parameters"
        )
        print(f"Config file logged in {run.info.artifact_uri}/run_evidence/parameters")

        # save metrics as evidence in mlflow/mlruns
        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="run_evidence/metrics_report",
        )
        print(f"Metrics file logged in {run.info.artifact_uri}/run_evidence/metrics_report")

        # save joblib model as evidence in mlflow/mlruns/model
        mlflow.log_artifact(
            str(model_path),
            artifact_path="run_evidence/model",
        )
        print(f"Joblib model logged in {run.info.artifact_uri}/run_evidence/model")

        """
        # save mlfow-formatted model as evidence in mlflow/mlruns
        model_info = mlflow.sklearn.log_model(
            sk_model=trained_model,
            name="logistic-baseline"
        )
        """

        # save mlflow-formatted model as evidence in mlflow/mlruns/mlflow_model
        mlflow_model_path = Path("artifacts/models/logistic_baseline_mlflow")
        if mlflow_model_path.exists():
            shutil.rmtree(mlflow_model_path)

        mlflow.sklearn.save_model(
            sk_model=trained_model,
            path=str(mlflow_model_path),
        )

        mlflow.log_artifacts(
            str(mlflow_model_path),
            artifact_path="run_evidence/mlflow_model",
        )
        print(f"Mlflow-formatted model logged in {run.info.artifact_uri}/run_evidence/mlflow_model")

        confusion_matrix = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["Class 0", "Class 1"]).plot(ax=ax)

        mlflow.log_figure(
            fig,
            "run_evidence/metrics_report/confusion_matrix.png",
        )

# Deprecated
def main_legacy() -> None:

    # Read configuration file
    config = load_training_config("configs/baseline_logistic.yaml")

    # Read the dataset
    dataset = load_dataset(config)

    # Split data into features & target
    features, target = split_features_and_target(dataset, config)

    # Split dataset into train & test
    train_data, test_data = split_train_test(features, target, config)

    # Build the sklearn.pipeline training pipeline
    model = build_model(config)

    # Train the model (pipeline)
    trained_model = train_model(model, train_data)

    # Evaluate the model using Test split
    metrics = evaluate_model(trained_model, test_data)

    # Save metrics
    save_metrics(metrics, config)

    # Save model
    save_model(trained_model, config)


    metrics_path = save_metrics(metrics, config)
    model_path = save_model(trained_model, config)

    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to: {metrics_path}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()