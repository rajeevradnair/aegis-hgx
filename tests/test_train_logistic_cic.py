from pathlib import Path
import json
import subprocess
import sys


def test_train_logistic_cic_writes_model_and_metrics() -> None:
    input_csv = Path(
        "data/processed/cicids2017/cic_tabular_features.csv"
    )
    model_path = Path(
        "artifacts/models/logistic_cic_baseline.joblib"
    )
    metrics_path = Path(
        "reports/logistic_cic_metrics.json"
    )

    if not input_csv.exists():
        return

    if model_path.exists():
        model_path.unlink()

    if metrics_path.exists():
        metrics_path.unlink()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aegis_hgx.models.baselines.train_logistic_cic",
            "--config",
            "configs/logistic_cic.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Metrics path:" in result.stdout
    assert "Model path:" in result.stdout
    assert model_path.exists()
    assert metrics_path.exists()

    with metrics_path.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    required_metrics = {
        "test_rows",
        "positive_rows",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "confusion_matrix",
    }

    assert required_metrics.issubset(metrics.keys())
    assert metrics["test_rows"] > 0