from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import torch


def test_train_torch_mlp_cic_writes_diagnostic_artifacts() -> None:
    input_csv = Path(
        "data/processed/cicids2017/cic_tabular_features.csv"
    )

    metrics_path = Path(
        "reports/torch_mlp_cic_train_test_validation_metrics.json"
    )
    epoch_history_path = Path(
        "reports/torch_mlp_cic_epoch_history.json"
    )
    threshold_history_path = Path(
        "reports/torch_mlp_cic_epoch_probability_threshold_history.csv"
    )
    model_path = Path(
        "artifacts/models/torch_mlp_cic_baseline.pt"
    )
    scaler_path = Path(
        "artifacts/models/torch_mlp_cic_scaler.joblib"
    )
    lineage_path = Path(
        "artifacts/lineage/torch_mlp_cic_manifest.json"
    )

    figure_paths = [
        Path("reports/figures/torch_mlp_loss_by_epoch.png"),
        Path("reports/figures/torch_mlp_pr_auc_by_epoch.png"),
        Path("reports/figures/torch_mlp_roc_auc_by_epoch.png"),
        Path(
            "reports/figures/"
            "torch_mlp_threshold_f1_by_probability_threshold.png"
        ),
        Path(
            "reports/figures/"
            "torch_mlp_final_test_precision_recall_curve.png"
        ),
        Path("reports/figures/torch_mlp_final_test_roc_curve.png"),
    ]

    if not input_csv.exists():
        return

    for path in [
        metrics_path,
        epoch_history_path,
        threshold_history_path,
        model_path,
        scaler_path,
        lineage_path,
        *figure_paths,
    ]:
        if path.exists():
            path.unlink()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aegis_hgx.models.baselines.train_torch_mlp_cic",
            "--config",
            "configs/torch_mlp_cic.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert metrics_path.exists()
    assert epoch_history_path.exists()
    assert threshold_history_path.exists()
    assert model_path.exists()
    assert scaler_path.exists()
    assert lineage_path.exists()

    for figure_path in figure_paths:
        assert figure_path.exists()

    with metrics_path.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    assert metrics["model_name"] == "torch_mlp_cic_baseline"
    assert metrics["split_type"] == "train_validation_test"
    assert "threshold_independent_metrics" in metrics
    assert "default_threshold_metrics" in metrics
    assert "best_validation_epoch" in metrics

    with epoch_history_path.open(encoding="utf-8") as history_file:
        epoch_history = json.load(history_file)

    assert len(epoch_history) > 0
    assert {
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_pr_auc",
        "validation_roc_auc",
    }.issubset(epoch_history[0].keys())

    threshold_history = pd.read_csv(threshold_history_path)

    assert len(threshold_history) > 0
    assert {
        "epoch",
        "threshold",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
    }.issubset(threshold_history.columns)

    checkpoint = torch.load(model_path, map_location="cpu")

    assert checkpoint["model_name"] == "torch_mlp_cic_baseline"
    assert "model_state_dict" in checkpoint
    assert "feature_columns" in checkpoint
    assert checkpoint["input_dim"] == len(checkpoint["feature_columns"])