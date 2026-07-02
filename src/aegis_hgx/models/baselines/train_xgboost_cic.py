from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse

import yaml
import pandas as pd
import json
import joblib

from aegis_hgx.models.baselines.config_schema import (
    TrainingConfig,
    validate_training_config,
)

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import shutil

import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay

from aegis_hgx.models.baselines.lineage import (
    LineageManifestInput,
    build_lineage_manifest,
    write_lineage_manifest,
)

CONFIG_PATH = "configs/xgboost_cic.yaml"
DEFAULT_TRAINING_DATA_DVC_PATH = Path(
    "data/processed/cicids2017/cic_tabular_features.csv.dvc"
)
DEFAULT_LINEAGE_MANIFEST_PATH = Path(
    "artifacts/lineage/xgboost_cic_manifest.json"
)
DEFAULT_MLFLOW_MODEL_PATH = Path(
    "artifacts/models/xgboost_cic_mlflow"
)

def load_training_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Training config must contain a YAML mapping.")

    return config


def load_dataset(config: TrainingConfig) -> pd.DataFrame:
    dataset_path = Path(config.data.input_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"CIC feature dataset not found: {dataset_path}")

    dataset = pd.read_csv(dataset_path)

    if dataset.empty:
        raise ValueError("CIC feature dataset contains no rows.")

    return dataset


def validate_target_column(
    dataset: pd.DataFrame,
    config: TrainingConfig,
) -> None:
    target_column = config.data.target_column

    if target_column not in dataset.columns:
        raise ValueError(
            f"CIC feature dataset is missing target column: {target_column}"
        )

    target_classes = sorted(dataset[target_column].dropna().unique())

    if len(target_classes) < 2:
        raise ValueError(
            "XGBoost CIC training requires at least two target classes. "
            f"Found classes: {target_classes}"
        )

def split_features_and_target(
    dataset: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    target_column = config.data.target_column

    validate_target_column(dataset, config)

    features = dataset.drop(columns=[target_column]).copy()
    target = dataset[target_column].copy()

    if features.empty:
        raise ValueError("CIC feature dataset contains no feature columns.")

    return features, target

def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
    config: TrainingConfig,
) -> tuple[
    tuple[pd.DataFrame, pd.Series],
    tuple[pd.DataFrame, pd.Series],
]:
    split_config = config.split

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=split_config.test_size,
        random_state=split_config.random_seed,
        stratify=target,
    )

    return (X_train, y_train), (X_test, y_test)


def calculate_scale_pos_weight(target: pd.Series) -> float:

    print("Class 0 count:", (target==0).astype(int).sum())
    print("Class 1 count:", (target==1).astype(int).sum())

    negative_count = int((target == 0).sum())
    positive_count = int((target == 1).sum())

    if positive_count == 0:
        raise ValueError("Cannot calculate scale_pos_weight with no positive rows.")

    return negative_count / positive_count

def build_model(
    config: TrainingConfig,
    target: pd.Series,
) -> XGBClassifier:
    scale_pos_weight = calculate_scale_pos_weight(target)

    return XGBClassifier(
        n_estimators=config.model.max_iter,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=config.split.random_seed,
        n_jobs=-1,
    )

def train_model(
    model: XGBClassifier,
    train_data: tuple[pd.DataFrame, pd.Series],
) -> XGBClassifier:
    X_train, y_train = train_data
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    trained_model: XGBClassifier,
    test_data: tuple[pd.DataFrame, pd.Series],
) -> dict[str, object]:
    X_test, y_test = test_data

    predictions = trained_model.predict(X_test)
    probabilities = trained_model.predict_proba(X_test)[:, 1]

    return {
        "test_rows": int(len(y_test)),
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
    metrics: dict[str, object],
    config: TrainingConfig,
) -> Path:
    metrics_path = Path(config.outputs.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    return metrics_path


def save_model(
    trained_model: XGBClassifier,
    config: TrainingConfig,
) -> Path:
    model_path = Path(config.outputs.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(trained_model, model_path)

    return model_path

def configure_experiment(config: TrainingConfig) -> str:
    tracking = config.experiment_tracking

    artifact_root = Path(tracking.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking.uri)

    experiment = mlflow.get_experiment_by_name(
        tracking.experiment_name
    )

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=tracking.experiment_name,
            artifact_location=artifact_root.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)

    return experiment_id


def build_run_parameters(
    config: TrainingConfig,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, object]:
    return {
        "model_type": "xgboost_classifier",
        "dataset_type": "cicids2017",
        "random_seed": config.split.random_seed,
        "test_size": config.split.test_size,
        "n_estimators": config.model.max_iter,
        "feature_count": len(features.columns),
        "scale_pos_weight": calculate_scale_pos_weight(target),
    }


def extract_scalar_metrics(
    metrics: dict[str, object],
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
        help="Path to experiment config YAML file.",
    )
    return arg_parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_config = load_training_config(args.config)
    config = validate_training_config(raw_config)

    experiment_id = configure_experiment(config)

    with mlflow.start_run(run_name="xgboost-cic-baseline") as run:

        print("Experiment started.")
        print(f"Experiment name: {config.experiment_tracking.experiment_name}" )
        print(f"Experiment ID: {experiment_id}")
        print(f"Run ID: {run.info.run_id}")
        print("Artifact URI:", run.info.artifact_uri)

        dataset = load_dataset(config)
        validate_target_column(dataset, config)

        features, target = split_features_and_target(dataset, config)
        train_data, test_data = split_train_test(features, target, config)

        X_train, y_train = train_data
        X_test, y_test = test_data

        target_counts = (
            dataset[config.data.target_column]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        model = build_model(config, target)

        trained_model = train_model(model, train_data)
        metrics = evaluate_model(trained_model, test_data)
        metrics_path = save_metrics(metrics, config)
        model_path = save_model(trained_model, config)

        print("Experiment:", config.experiment_tracking.experiment_name)
        print("Config path:", args.config)
        print("Input path:", config.data.input_path)
        print("Rows:", len(dataset))
        print("Columns:", len(dataset.columns))
        print("Target column:", config.data.target_column)
        print("Target counts:", target_counts)
        print("Metrics path:", config.outputs.metrics_path)
        print("Model path:", config.outputs.model_path)
        print("Feature columns:", len(features.columns))
        print("First features:", list(features.columns[:10]))
        print("Train rows:", len(X_train))
        print("Test rows:", len(X_test))
        print("Train target counts:", y_train.value_counts().sort_index().to_dict())
        print("Test target counts:", y_test.value_counts().sort_index().to_dict())
        print("Raw Model:", model)
        print("Trained Model:", trained_model)
        print("Metrics:", metrics)
        print("Metrics path:", metrics_path)
        print("Model path:", model_path)

        mlflow.set_tags(
            {
                "project": "aegis-hgx",
                "model_family": "xgboost",
                "dataset_type": "cicids2017",
                "pipeline_stage": "baseline",
            }
        )

        parameters = build_run_parameters(config, features, target)
        scalar_metrics = extract_scalar_metrics(metrics)

        mlflow.log_params(parameters)
        mlflow.log_metrics(scalar_metrics)

        mlflow.log_artifact(
            str(args.config),
            artifact_path="run_evidence/parameters",
        )
        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="run_evidence/metrics_report",
        )
        mlflow.log_artifact(
            str(model_path),
            artifact_path="run_evidence/model",
        )

        if DEFAULT_MLFLOW_MODEL_PATH.exists():
            shutil.rmtree(DEFAULT_MLFLOW_MODEL_PATH)

        mlflow.xgboost.save_model(
            xgb_model=trained_model.get_booster(),
            path=str(DEFAULT_MLFLOW_MODEL_PATH),
        )

        mlflow.log_artifacts(
            str(DEFAULT_MLFLOW_MODEL_PATH),
            artifact_path="run_evidence/mlflow_model",
        )

        confusion = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix=confusion,
            display_labels=["benign", "attack"],
        ).plot(ax=ax)

        mlflow.log_figure(
            fig,
            "run_evidence/metrics_report/confusion_matrix.png",
        )
        plt.close(fig)

        lineage_input = LineageManifestInput(
            model_name="aegis_hgx_xgboost_cic_baseline",
            model_version="xgboost_cic_baseline_v1",
            model_type="xgboost_classifier",
            model_artifact_path=str(model_path),
            training_data_path=config.data.input_path,
            training_data_dvc_path=str(DEFAULT_TRAINING_DATA_DVC_PATH),
            training_config_path=str(args.config),
            data_generation_config_path=None,
            metrics_path=str(metrics_path),
            mlflow_experiment_name=config.experiment_tracking.experiment_name,
            mlflow_experiment_id=str(experiment_id),
            mlflow_run_id=run.info.run_id,
            mlflow_tracking_uri=mlflow.get_tracking_uri(),
            training_entrypoint=(
                "aegis_hgx.models.baselines.train_xgboost_cic"
            ),
            training_command=(
                "python -m "
                "aegis_hgx.models.baselines.train_xgboost_cic "
                f"--config {args.config}"
            ),
            feature_store_provider="local_snapshot",
            feature_snapshot_id="cicids2017_tabular_features_v1",
            offline_store_path=config.data.input_path,
        )

        lineage_manifest = build_lineage_manifest(lineage_input)
        lineage_manifest_path = write_lineage_manifest(
            lineage_manifest,
            DEFAULT_LINEAGE_MANIFEST_PATH,
        )

        mlflow.log_artifact(
            str(lineage_manifest_path),
            artifact_path="run_evidence/lineage",
        )

        print("Lineage manifest:", lineage_manifest_path)



if __name__ == "__main__":
    main()