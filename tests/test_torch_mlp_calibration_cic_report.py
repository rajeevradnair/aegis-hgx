from pathlib import Path
import json
import subprocess
import sys

import pandas as pd


def test_calibration_report_writes_artifacts() -> None:
    input_csv = Path(
        "data/processed/cicids2017/cic_tabular_features.csv"
    )
    model_path = Path("artifacts/models/torch_mlp_cic_baseline.pt")
    scaler_path = Path("artifacts/models/torch_mlp_cic_scaler.joblib")

    metrics_path = Path(
        "reports/calibration/torch_mlp_cic_calibration_metrics.json"
    )
    bins_path = Path(
        "reports/calibration/torch_mlp_cic_calibration_bins.csv"
    )
    reliability_diagram_path = Path(
        "reports/figures/torch_mlp_cic_reliability_diagram.png"
    )
    probability_histogram_path = Path(
        "reports/figures/torch_mlp_cic_probability_histogram.png"
    )

    if not input_csv.exists() or not model_path.exists() or not scaler_path.exists():
        return

    for path in [
        metrics_path,
        bins_path,
        reliability_diagram_path,
        probability_histogram_path,
    ]:
        if path.exists():
            path.unlink()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aegis_hgx.evaluation.torch_mlp_calibration_cic_report",
            "--config",
            "configs/torch_mlp_calibration_cic.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Calibration metrics path:" in result.stdout
    assert "Calibration bins path:" in result.stdout
    assert "Reliability diagram path:" in result.stdout
    assert "Probability histogram path:" in result.stdout

    assert metrics_path.exists()
    assert bins_path.exists()
    assert reliability_diagram_path.exists()
    assert probability_histogram_path.exists()

    with metrics_path.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    assert metrics["model_name"] == "torch_mlp_cic_baseline"
    assert metrics["evaluated_split"] == "test"
    assert "calibration" in metrics
    assert "brier_score" in metrics["calibration"]
    assert "expected_calibration_error" in metrics["calibration"]
    assert "probability_summary" in metrics

    bins = pd.read_csv(bins_path)

    required_columns = {
        "bin_index",
        "lower_bound",
        "upper_bound",
        "sample_count",
        "sample_fraction",
        "mean_predicted_probability",
        "observed_positive_rate",
        "absolute_calibration_gap",
    }

    assert required_columns.issubset(bins.columns)
    assert len(bins) > 0