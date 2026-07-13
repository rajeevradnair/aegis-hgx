from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data


def test_train_lanl_gcn_pyg_smoke(tmp_path: Path) -> None:
    # Create temporary folders for test inputs and outputs.
    artifact_directory = tmp_path / "artifacts"
    report_directory = tmp_path / "reports"

    artifact_directory.mkdir()
    report_directory.mkdir()

    graph_path = artifact_directory / "tiny_graph.pt"
    model_path = artifact_directory / "tiny_gcn_model.pt"
    metrics_path = report_directory / "tiny_gcn_metrics.json"
    config_path = tmp_path / "lanl_gcn_pyg.yaml"

    # Build a tiny graph with 20 nodes.
    # Each node has 4 numeric features.
    x = torch.randn(
        20,
        4,
        dtype=torch.float,
    )

    # Build simple directed edges.
    # edge_index shape must be [2, num_edges].
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.long,
    )

    # Create balanced labels.
    # First 10 nodes are class 0.
    # Last 10 nodes are class 1.
    y = torch.tensor(
        [0] * 10 + [1] * 10,
        dtype=torch.long,
    )

    # Create the PyG Data object expected by the trainer.
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )

    # Explicitly store number of nodes.
    data.num_nodes = x.shape[0]

    # Validate before saving so the test input is known-good.
    data.validate(raise_on_error=True)

    torch.save(
        data,
        graph_path,
    )

    # Use small training settings so the test stays fast.
    config = {
        "input": {
            "graph_path": str(graph_path),
        },
        "output": {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
        },
        "split": {
            "train_ratio": 0.60,
            "val_ratio": 0.20,
            "test_ratio": 0.20,
            "seed": 42,
        },
        "model": {
            "hidden_channels": 8,
            "dropout": 0.10,
        },
        "training": {
            "epochs": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0005,
        },
        "evaluation": {
            "positive_label": 1,
        },
    }

    config_path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    # Run the trainer exactly like a user would run it from the command line.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aegis_hgx.models.baselines.train_lanl_gcn_pyg",
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # Confirm the major training stages printed successfully.
    assert "LANL PyG graph summary for GCN training" in result.stdout
    assert "Node mask summary" in result.stdout
    assert "GCN model summary" in result.stdout
    assert "Forward-pass smoke check summary" in result.stdout
    assert "GCN training summary" in result.stdout
    assert "GCN test-set metrics" in result.stdout
    assert "Saved GCN training outputs" in result.stdout

    # Confirm the trainer wrote the expected artifacts.
    assert model_path.exists()
    assert metrics_path.exists()

    # Confirm the checkpoint has the fields needed to reload the model later.
    checkpoint = torch.load(
        model_path,
        weights_only=False,
    )

    assert checkpoint["model_class"] == "LanlGCN"
    assert checkpoint["input_channels"] == 4
    assert checkpoint["hidden_channels"] == 8
    assert checkpoint["output_channels"] == 2
    assert "model_state_dict" in checkpoint

    # Confirm the metrics JSON has the expected structure.
    metrics = json.loads(
        metrics_path.read_text(encoding="utf-8")
    )

    assert metrics["phase"] == "gcn_training_and_evaluation"
    assert metrics["model"]["class"] == "LanlGCN"
    assert metrics["training"]["epochs"] == 2

    assert "test_metrics" in metrics
    assert "training_history" in metrics
    assert "best_validation_epoch" in metrics

    assert len(metrics["training_history"]) == 2

    # Confirm the split created train/validation/test nodes.
    assert metrics["split_counts"]["train_nodes"] > 0
    assert metrics["split_counts"]["val_nodes"] > 0
    assert metrics["split_counts"]["test_nodes"] > 0