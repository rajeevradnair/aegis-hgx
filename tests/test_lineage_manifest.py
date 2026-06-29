import json
from pathlib import Path


def test_logistic_baseline_lineage_manifest_has_required_sections() -> None:
    manifest_path = Path(
        "artifacts/lineage/logistic_baseline_manifest.json"
    )

    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    required_sections = {
        "lineage_manifest_version",
        "created_at",
        "model",
        "mlflow",
        "git",
        "training_data",
        "configs",
        "metrics",
        "feature_store",
        "reproducibility",
    }

    assert required_sections.issubset(manifest.keys())

def test_lineage_manifest_model_section_links_model_artifact() -> None:
    manifest_path = Path(
        "artifacts/lineage/logistic_baseline_manifest.json"
    )

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    model = manifest["model"]

    assert model["model_name"] == "aegis_hgx_logistic_baseline"
    assert model["model_version"] == "logistic_baseline_v1"
    assert model["model_type"] == "logistic_regression"
    assert model["model_artifact_path"].endswith(
        "logistic_baseline.joblib"
    )
    assert len(model["model_artifact_sha256"]) == 64

def test_lineage_manifest_training_data_section_links_dvc_snapshot() -> None:
    manifest_path = Path(
        "artifacts/lineage/logistic_baseline_manifest.json"
    )

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    training_data = manifest["training_data"]

    assert training_data["dataset_name"] == "aegis_hgx_synthetic_events"
    assert training_data["dataset_role"] == "training_reference"
    assert training_data["data_path"].endswith("synthetic_events.csv")
    assert len(training_data["data_sha256"]) == 64
    assert training_data["row_count"] > 0
    assert training_data["column_count"] > 0
    assert training_data["dvc"]["tracked"] is True
    assert training_data["dvc"]["dvc_file"].endswith(
        "synthetic_events.csv.dvc"
    )