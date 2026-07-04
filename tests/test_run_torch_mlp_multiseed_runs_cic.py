from pathlib import Path
import json
import subprocess
import sys

import pandas as pd


def test_torch_mlp_seed_runs_write_artifacts() -> None:
    input_csv = Path(
        "data/processed/cicids2017/cic_tabular_features.csv"
    )

    per_seed_metrics_path = Path(
        "reports/experiments/torch_mlp_cic_multiseed_runs.csv"
    )
    summary_metrics_path = Path(
        "reports/experiments/torch_mlp_cic_multiseed_summary.json"
    )

    if not input_csv.exists():
        return

    for path in [per_seed_metrics_path, summary_metrics_path]:
        if path.exists():
            path.unlink()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aegis_hgx.experiments.run_torch_mlp_multiseed_runs_cic",
            "--config",
            "configs/torch_mlp_multiseed_runs_cic.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Per-seed metrics path:" in result.stdout
    assert "Summary metrics path:" in result.stdout
    assert "Seed run count:" in result.stdout
    assert "Mean PR-AUC:" in result.stdout
    assert "PR-AUC std:" in result.stdout
    assert "Mean F1:" in result.stdout

    assert per_seed_metrics_path.exists()
    assert summary_metrics_path.exists()

    per_seed_metrics = pd.read_csv(per_seed_metrics_path)

    required_columns = {
        "seed",
        "train_rows",
        "validation_rows",
        "test_rows",
        "train_positive_rows",
        "validation_positive_rows",
        "test_positive_rows",
        "final_train_loss",
        "test_loss",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    }

    assert required_columns.issubset(per_seed_metrics.columns)
    assert len(per_seed_metrics) > 0

    with summary_metrics_path.open(encoding="utf-8") as summary_file:
        summary = json.load(summary_file)

    assert summary["model_name"] == "torch_mlp_cic_baseline"
    assert summary["dataset"] == "cicids2017"
    assert summary["experiment_type"] == "multi_seed_stability"
    assert summary["evaluated_split"] == "test"
    assert summary["seed_count"] == len(per_seed_metrics)

    assert "metrics" in summary
    assert "pr_auc" in summary["metrics"]
    assert "roc_auc" in summary["metrics"]
    assert "f1" in summary["metrics"]

    for metric_name in ["pr_auc", "roc_auc", "f1"]:
        metric_summary = summary["metrics"][metric_name]

        assert "mean" in metric_summary
        assert "std" in metric_summary
        assert "min" in metric_summary
        assert "max" in metric_summary
        assert "standard_error" in metric_summary
        assert "ci95_lower" in metric_summary
        assert "ci95_upper" in metric_summary