from pathlib import Path
import subprocess
import sys


def test_baseline_comparison_report_is_generated() -> None:
    required_paths = [
        Path("reports/logistic_cic_metrics.json"),
        Path("reports/xgboost_cic_metrics.json"),
        Path("reports/mlp_cic_metrics.json"),
        Path("reports/torch_mlp_cic_train_test_validation_metrics.json"),
        Path("reports/calibration/torch_mlp_cic_calibration_metrics.json"),
        Path("reports/experiments/torch_mlp_cic_seed_summary.json"),
    ]

    if any(not path.exists() for path in required_paths):
        return

    report_path = Path("reports/baseline_comparison_report.md")

    if report_path.exists():
        report_path.unlink()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aegis_hgx.reports.build_baseline_comparison_report",
            "--config",
            "configs/baseline_comparison_report.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Report path:" in result.stdout
    assert report_path.exists()

    report_text = report_path.read_text(encoding="utf-8")

    required_sections = [
        "# AEGIS-HGX Baseline Comparison Report",
        "## Purpose",
        "## Phase Classification",
        "## Artifact Inventory",
        "## Model Comparison",
        "## Initial Interpretation",
        "## Calibration Summary",
        "## Seed-Stability Summary",
        "## Graph-Model Target Bar",
        "## Limitations",
        "## Next Steps",
    ]

    for section in required_sections:
        assert section in report_text

    required_phrases = [
        "Training/evaluation/test",
        "PR-AUC",
        "ROC-AUC",
        "Brier score",
        "Expected calibration error",
        "Future graph models must justify their additional complexity",
        "A GNN should not be treated as better simply because it is more advanced",
    ]

    for phrase in required_phrases:
        assert phrase in report_text