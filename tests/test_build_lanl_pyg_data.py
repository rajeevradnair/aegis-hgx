from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml


def test_build_lanl_pyg_data_writes_valid_outputs(tmp_path: Path) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"

    input_directory.mkdir()
    output_directory.mkdir()

    nodes = pd.DataFrame(
        [
            {
                "node_id": 0,
                "entity_type": "user",
                "entity_name": "U001",
                "node_key": "user::U001",
                "first_seen_timestamp": 1,
                "last_seen_timestamp": 20,
                "event_count": 2,
                "label": 0,
            },
            {
                "node_id": 1,
                "entity_type": "host",
                "entity_name": "C001",
                "node_key": "host::C001",
                "first_seen_timestamp": 1,
                "last_seen_timestamp": 40,
                "event_count": 4,
                "label": 1,
            },
            {
                "node_id": 2,
                "entity_type": "host_or_domain",
                "entity_name": "corp.example",
                "node_key": "host_or_domain::corp.example",
                "first_seen_timestamp": 30,
                "last_seen_timestamp": 30,
                "event_count": 1,
                "label": 0,
            },
            {
                "node_id": 3,
                "entity_type": "user",
                "entity_name": "U999",
                "node_key": "user::U999",
                "first_seen_timestamp": 40,
                "last_seen_timestamp": 40,
                "event_count": 1,
                "label": 1,
            },
        ]
    )

    edges = pd.DataFrame(
        [
            {
                "edge_id": "e1",
                "source_node_id": 0,
                "destination_node_id": 1,
                "source_entity": "U001",
                "destination_entity": "C001",
                "source_entity_type": "user",
                "destination_entity_type": "host",
                "edge_type": "authentication:auth_logon",
                "event_family": "authentication",
                "event_type": "auth_logon",
                "timestamp": 1,
                "event_result": "success",
                "label": 0,
                "source_file": "auth.parquet",
                "row_number": 0,
            },
            {
                "edge_id": "e2",
                "source_node_id": 0,
                "destination_node_id": 1,
                "source_entity": "U001",
                "destination_entity": "C001",
                "source_entity_type": "user",
                "destination_entity_type": "host",
                "edge_type": "authentication:auth_logon",
                "event_family": "authentication",
                "event_type": "auth_logon",
                "timestamp": 20,
                "event_result": "success",
                "label": 0,
                "source_file": "auth.parquet",
                "row_number": 1,
            },
            {
                "edge_id": "e3",
                "source_node_id": 1,
                "destination_node_id": 2,
                "source_entity": "C001",
                "destination_entity": "corp.example",
                "source_entity_type": "host",
                "destination_entity_type": "host_or_domain",
                "edge_type": "dns:dns_resolution",
                "event_family": "dns",
                "event_type": "dns_resolution",
                "timestamp": 30,
                "event_result": "resolved",
                "label": 0,
                "source_file": "dns.parquet",
                "row_number": 2,
            },
            {
                "edge_id": "e4",
                "source_node_id": 3,
                "destination_node_id": 1,
                "source_entity": "U999",
                "destination_entity": "C001",
                "source_entity_type": "user",
                "destination_entity_type": "host",
                "edge_type": "redteam_ground_truth:redteam_activity",
                "event_family": "redteam_ground_truth",
                "event_type": "redteam_activity",
                "timestamp": 40,
                "event_result": "known_redteam",
                "label": 1,
                "source_file": "redteam.parquet",
                "row_number": 3,
            },
            {
                "edge_id": "bad_edge",
                "source_node_id": 999,
                "destination_node_id": 1,
                "source_entity": "missing_user",
                "destination_entity": "C001",
                "source_entity_type": "user",
                "destination_entity_type": "host",
                "edge_type": "authentication:auth_logon",
                "event_family": "authentication",
                "event_type": "auth_logon",
                "timestamp": 50,
                "event_result": "success",
                "label": 0,
                "source_file": "auth.parquet",
                "row_number": 4,
            },
        ]
    )

    nodes.to_parquet(
        input_directory / "graph_nodes.parquet",
        index=False,
    )
    edges.to_parquet(
        input_directory / "graph_edges.parquet",
        index=False,
    )

    config = {
        "input": {
            "directory": str(input_directory),
            "graph_nodes_filename": "graph_nodes.parquet",
            "graph_edges_filename": "graph_edges.parquet",
        },
        "output": {
            "directory": str(output_directory),
            "graph_filename": "lanl_homogeneous_graph.pt",
            "metadata_filename": "lanl_homogeneous_graph_metadata.json",
        },
        "features": {
            "node_numeric": [
                "event_count",
                "first_seen_timestamp",
                "last_seen_timestamp",
                "active_span",
            ],
            "node_categorical": [
                "entity_type",
            ],
            "edge_numeric": [
                "timestamp",
            ],
            "edge_categorical": [
                "edge_type",
            ],
        },
        "validation": {
            "invalid_edge_policy": "warn",
            "require_compact_indices": True,
        },
    }

    config_path = tmp_path / "lanl_pyg_data.yaml"
    config_path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aegis_hgx.models.baselines.build_lanl_pyg_data",
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "LANL PyG data conversion plan" in result.stdout
    assert "Graph table summary" in result.stdout
    assert "PyG node mapping summary" in result.stdout
    assert "PyG edge_index summary" in result.stdout
    assert "Node tensor summary" in result.stdout
    assert "Edge tensor summary" in result.stdout
    assert "PyG Data object summary" in result.stdout
    assert "PyG conversion outputs" in result.stdout

    graph_path = output_directory / "lanl_homogeneous_graph.pt"
    metadata_path = output_directory / "lanl_homogeneous_graph_metadata.json"

    assert graph_path.exists()
    assert metadata_path.exists()

    data = torch.load(
        graph_path,
        weights_only=False,
    )

    data.validate(raise_on_error=True)

    assert data.num_nodes == 4
    assert data.edge_index.shape == (2, 4)
    assert data.edge_attr.shape[0] == 4
    assert data.x.shape[0] == 4
    assert data.y.shape == (4,)
    assert data.edge_label.shape == (4,)

    assert data.y.tolist() == [0, 1, 0, 1]
    assert data.edge_label.tolist() == [0, 0, 0, 1]
    assert data.node_id.tolist() == [0, 1, 2, 3]
    assert data.edge_id == ["e1", "e2", "e3", "e4"]

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert metadata["phase"] == "pyg_data_preparation_before_model_training"
    assert metadata["counts"]["node_rows"] == 4
    assert metadata["counts"]["edge_rows"] == 5
    assert metadata["counts"]["valid_edge_rows"] == 4
    assert metadata["counts"]["invalid_edge_rows"] == 1
    assert metadata["counts"]["num_nodes"] == 4
    assert metadata["counts"]["num_edges"] == 4
    assert metadata["indexing"]["node_id_equals_pyg_index"] is True

    edge_feature_columns = metadata["features"]["edge_feature_columns"]

    assert "edge_type_ground_truth_edge_type_withheld" not in edge_feature_columns