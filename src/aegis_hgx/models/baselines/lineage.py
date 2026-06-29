from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from aegis_hgx.utils.dvc_metadata import get_first_dvc_output
from aegis_hgx.utils.git_metadata import (
    get_git_branch,
    get_git_commit,
    is_git_dirty,
)
from aegis_hgx.utils.hashing import sha256_file


@dataclass(frozen=True)
class LineageManifestInput:
    model_name: str
    model_version: str
    model_type: str
    model_artifact_path: str
    training_data_path: str
    training_data_dvc_path: str
    training_config_path: str
    data_generation_config_path: str
    metrics_path: str
    mlflow_experiment_name: str
    mlflow_experiment_id: str
    mlflow_run_id: str
    mlflow_tracking_uri: str
    training_entrypoint: str
    training_command: str
    feature_store_provider: str = "local_snapshot"
    feature_view_name: str | None = None
    feature_view_version: str | None = None
    feature_snapshot_id: str | None = None
    offline_store_path: str | None = None


def load_metrics_values(metrics_path: str | Path) -> dict[str, Any]:
    path = Path(metrics_path)

    if not path.exists():
        raise FileNotFoundError(f"Cannot read missing metrics file: {path}")

    with path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics file must contain a JSON object: {path}")

    return metrics


def build_lineage_manifest(
    lineage_input: LineageManifestInput,
) -> dict[str, Any]:
    training_data = pd.read_csv(lineage_input.training_data_path)
    dvc_output = get_first_dvc_output(lineage_input.training_data_dvc_path)
    metrics_values = load_metrics_values(lineage_input.metrics_path)

    feature_snapshot_id = (
        lineage_input.feature_snapshot_id
        or Path(lineage_input.training_data_path).stem
    )

    offline_store_path = (
        lineage_input.offline_store_path
        or lineage_input.training_data_path
    )

    return {
        "lineage_manifest_version": "0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_name": lineage_input.model_name,
            "model_version": lineage_input.model_version,
            "model_type": lineage_input.model_type,
            "model_artifact_path": lineage_input.model_artifact_path,
            "model_artifact_sha256": sha256_file(
                lineage_input.model_artifact_path
            ),
        },
        "mlflow": {
            "experiment_name": lineage_input.mlflow_experiment_name,
            "experiment_id": lineage_input.mlflow_experiment_id,
            "run_id": lineage_input.mlflow_run_id,
            "tracking_uri": lineage_input.mlflow_tracking_uri,
        },
        "git": {
            "commit": get_git_commit(),
            "branch": get_git_branch(),
            "is_dirty": is_git_dirty(),
        },
        "training_data": {
            "dataset_name": "aegis_hgx_synthetic_events",
            "dataset_role": "training_reference",
            "data_path": lineage_input.training_data_path,
            "data_sha256": sha256_file(lineage_input.training_data_path),
            "row_count": int(training_data.shape[0]),
            "column_count": int(training_data.shape[1]),
            "columns": list(training_data.columns),
            "dvc": {
                "tracked": True,
                "dvc_file": lineage_input.training_data_dvc_path,
                "dvc_out_path": dvc_output.get("path"),
                "dvc_out_hash": dvc_output.get("md5"),
                "dvc_out_size": dvc_output.get("size"),
                "dvc_remote": None,
            },
        },
        "configs": {
            "training_config": {
                "path": lineage_input.training_config_path,
                "sha256": sha256_file(lineage_input.training_config_path),
            },
            "data_generation_config": {
                "path": lineage_input.data_generation_config_path,
                "sha256": sha256_file(
                    lineage_input.data_generation_config_path
                ),
            },
        },
        "metrics": {
            "metrics_path": lineage_input.metrics_path,
            "metrics_sha256": sha256_file(lineage_input.metrics_path),
            "values": metrics_values,
        },
        "feature_store": {
            "provider": lineage_input.feature_store_provider,
            "feature_view_name": lineage_input.feature_view_name,
            "feature_view_version": lineage_input.feature_view_version,
            "feature_snapshot_id": feature_snapshot_id,
            "offline_store_path": offline_store_path,
        },
        "reproducibility": {
            "python_version": platform.python_version(),
            "training_entrypoint": lineage_input.training_entrypoint,
            "training_command": lineage_input.training_command,
        },
    }


def write_lineage_manifest(
    manifest: dict[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)

    return path