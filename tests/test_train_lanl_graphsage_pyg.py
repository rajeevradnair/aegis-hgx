from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data


def test_train_lanl_graphsage_pyg_end_to_end(tmp_path: Path) -> None:
    # ------------------------------------------------------------
    # CREATE A TINY PYTORCH GEOMETRIC GRAPH
    # ------------------------------------------------------------
    # This graph is intentionally tiny so the test runs quickly.
    #
    # Shape:
    #   x = [20 nodes, 4 node features]
    x = torch.randn(
        20,
        4,
    )

    # edge_index stores graph connectivity.
    #
    # Shape:
    #   [2, num_edges]
    #
    # Each column is one directed edge:
    #   source node -> destination node
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.long,
    )

    # Node labels.
    #
    # Shape:
    #   [20]
    #
    # 0 = benign
    # 1 = suspicious
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

    graph_path = tmp_path / "tiny_graph.pt"
    model_path = tmp_path / "tiny_graphsage_model.pt"
    metrics_path = tmp_path / "tiny_graphsage_metrics.json"
    config_path = tmp_path / "tiny_graphsage_config.yaml"

    torch.save(
        data,
        graph_path,
    )

    # ------------------------------------------------------------
    # CREATE A TINY CONFIG
    # ------------------------------------------------------------
    # Keep epochs low so the smoke test remains fast.
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
            "aggregation": "mean",
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
        yaml.safe_dump(config)
    )

    # ------------------------------------------------------------
    # RUN THE TRAINER AS A MODULE
    # ------------------------------------------------------------
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "aegis_hgx.models.baselines.train_lanl_graphsage_pyg",
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # ------------------------------------------------------------
    # VERIFY EXPECTED LOG OUTPUT
    # ------------------------------------------------------------
    stdout = completed.stdout

    assert "LANL PyG graph summary for GraphSAGE training" in stdout
    assert "Node mask summary" in stdout
    assert "GraphSAGE model summary" in stdout
    assert "GraphSAGE forward-pass smoke check summary" in stdout
    assert "Training GraphSAGE model" in stdout
    assert "GraphSAGE training summary" in stdout
    assert "GraphSAGE test-set metrics" in stdout
    assert "Saved GraphSAGE training outputs" in stdout

    # ------------------------------------------------------------
    # VERIFY OUTPUT FILES EXIST
    # ------------------------------------------------------------
    assert model_path.exists()
    assert metrics_path.exists()

    # ------------------------------------------------------------
    # VERIFY CHECKPOINT CONTENT
    # ------------------------------------------------------------
    checkpoint = torch.load(
        model_path,
        weights_only=False,
    )

    assert checkpoint["model_class"] == "LanlGraphSAGE"
    assert checkpoint["input_channels"] == 4
    assert checkpoint["hidden_channels"] == 8
    assert checkpoint["output_channels"] == 2
    assert checkpoint["aggregation"] == "mean"
    assert "model_state_dict" in checkpoint

    # ------------------------------------------------------------
    # VERIFY METRICS CONTENT
    # ------------------------------------------------------------
    metrics = json.loads(
        metrics_path.read_text()
    )

    assert metrics["phase"] == "graphsage_training_and_evaluation"
    assert metrics["model"]["class"] == "LanlGraphSAGE"
    assert metrics["model"]["input_channels"] == 4
    assert metrics["model"]["hidden_channels"] == 8
    assert metrics["model"]["output_channels"] == 2
    assert metrics["model"]["aggregation"] == "mean"
    assert metrics["training"]["epochs"] == 2

    assert "best_validation_epoch" in metrics
    assert "test_metrics" in metrics
    assert "training_history" in metrics

    assert len(metrics["training_history"]) == 2

    assert metrics["split_counts"]["train_nodes"] > 0
    assert metrics["split_counts"]["val_nodes"] > 0
    assert metrics["split_counts"]["test_nodes"] > 0