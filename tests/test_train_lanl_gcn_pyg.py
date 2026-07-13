from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data


def test_train_lanl_gcn_pyg_smoke(tmp_path: Path) -> None:
    # Temporary folders for this test only.
    artifact_directory = tmp_path / "artifacts"
    report_directory = tmp_path / "reports"

    artifact_directory.mkdir()
    report_directory.mkdir()

    graph_path = artifact_directory / "tiny_graph.pt"
    model_path = artifact_directory / "tiny_gcn_model.pt"
    metrics_path = report_directory / "tiny_gcn_metrics.json"
    config_path = tmp_path / "lanl_gcn_pyg.yaml"

    # Build a tiny graph with 20 nodes and 4 features per node.
    x = torch.randn(
        20,
        4,
        dtype=torch.float,
    )

    # Simple directed graph.
    # edge_index shape: [2, num_edges]
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.long,
    )

    # Balanced labels so train/val/test splits have both classes.
    y = torch.tensor(
        [0] * 10 + [1] * 10,
        dtype=torch.long,
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )
    data.num_nodes = x.shape[0]
    data.validate(raise_on_error=True)

    torch.save(
        data,
        graph_path,
    )

    # Start from the real project config so the test automatically includes
    # every section that train_lanl_gcn_pyg.py validates.
    base_config_path = Path("configs/lanl_gcn_pyg.yaml")

    if not base_config_path.exists():
        raise FileNotFoundError(
            "Expected configs/lanl_gcn_pyg.yaml to exist for this smoke test."
        )

    base_config = yaml.safe_load(
        base_config_path.read_text(encoding="utf-8")
    )

    if not isinstance(base_config, dict):
        raise ValueError("configs/lanl_gcn_pyg.yaml must contain a YAML mapping.")

    # Copy the real config, then override only test-specific values.
    config = dict(base_config)

    config["input"] = dict(config["input"])
    config["input"]["graph_path"] = str(graph_path)

    config["output"] = dict(config["output"])
    config["output"]["model_path"] = str(model_path)
    config["output"]["metrics_path"] = str(metrics_path)

    config["split"] = dict(config["split"])
    config["split"]["train_ratio"] = 0.60
    config["split"]["val_ratio"] = 0.20
    config["split"]["test_ratio"] = 0.20
    config["split"]["seed"] = 42

    config["model"] = dict(config["model"])
    config["model"]["hidden_channels"] = 8
    config["model"]["dropout"] = 0.10

    config["training"] = dict(config["training"])
    config["training"]["epochs"] = 2
    config["training"]["learning_rate"] = 0.01
    config["training"]["weight_decay"] = 0.0005

    config["evaluation"] = dict(config["evaluation"])
    config["evaluation"]["positive_label"] = 1

    # Keep experiment tracking disabled for the smoke test,
    # but preserve every key the real trainer expects.
    if "experiment_tracking" in config:
        config["experiment_tracking"] = dict(config["experiment_tracking"])
        config["experiment_tracking"]["enabled"] = False

    config_path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

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

    assert "LANL PyG graph summary for GCN training" in result.stdout
    assert "Node mask summary" in result.stdout
    assert "GCN model summary" in result.stdout
    assert "Forward-pass smoke check summary" in result.stdout
    assert "GCN training summary" in result.stdout
    assert "GCN test-set metrics" in result.stdout
    assert "Saved GCN training outputs" in result.stdout

    assert model_path.exists()
    assert metrics_path.exists()

    checkpoint = torch.load(
        model_path,
        weights_only=False,
    )

    assert checkpoint["model_class"] == "LanlGCN"
    assert checkpoint["input_channels"] == 4
    assert checkpoint["hidden_channels"] == 8
    assert checkpoint["output_channels"] == 2
    assert "model_state_dict" in checkpoint

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

    assert metrics["split_counts"]["train_nodes"] > 0
    assert metrics["split_counts"]["val_nodes"] > 0
    assert metrics["split_counts"]["test_nodes"] > 0