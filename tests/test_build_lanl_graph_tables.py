from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import yaml


def test_build_lanl_graph_tables_from_temporary_clean_events(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    config_path = tmp_path / "lanl_graph_tables.yaml"

    input_directory.mkdir(parents=True)
    output_directory.mkdir(parents=True)

    clean_events = pd.DataFrame(
        {
            "event_id": [
                "lanl_authentication_0",
                "lanl_dns_0",
                "lanl_network_flow_0",
                "lanl_process_0",
                "lanl_redteam_0",
            ],
            "timestamp": [
                1,
                2,
                3,
                4,
                5,
            ],
            "event_family": [
                "authentication",
                "dns",
                "network_flow",
                "process",
                "redteam_ground_truth",
            ],
            "event_type": [
                "auth_logon",
                "dns_resolution",
                "network_flow",
                "process_start",
                "redteam_activity",
            ],
            "source_entity": [
                "U001",
                "C001",
                "C001",
                "U001",
                "U999",
            ],
            "destination_entity": [
                "C001",
                "corp.example",
                "C002",
                "powershell.exe",
                "C002",
            ],
            "source_entity_type": [
                "user",
                "host",
                "host",
                "user",
                "user",
            ],
            "destination_entity_type": [
                "host",
                "host_or_domain",
                "host",
                "process",
                "host",
            ],
            "event_result": [
                "success",
                "unknown",
                "observed",
                "start",
                "confirmed_redteam",
            ],
            "label": [
                0,
                0,
                0,
                0,
                1,
            ],
            "source_file": [
                "auth.txt.gz",
                "dns.txt.gz",
                "flows.txt.gz",
                "proc.txt.gz",
                "redteam.txt.gz",
            ],
            "row_number": [
                0,
                0,
                0,
                0,
                0,
            ],
        }
    )

    clean_events.to_parquet(
        input_directory / "clean_all_events.parquet",
        index=False,
    )

    config = {
        "input": {
            "directory": str(input_directory),
            "clean_events_filename": "clean_all_events.parquet",
        },
        "output": {
            "directory": str(output_directory),
            "graph_nodes_filename": "graph_nodes.parquet",
            "graph_edges_filename": "graph_edges.parquet",
            "manifest_filename": "graph_table_manifest.json",
        },
        "schema": {
            "node_key_separator": "::",
            "unknown_value": "unknown",
            "default_node_label": 0,
        },
    }

    config_path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "pipelines/build_lanl_graph_tables.py",
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Built graph node table:" in result.stdout
    assert "Built graph edge table:" in result.stdout
    assert "Updated graph node table:" in result.stdout
    assert "Graph table manifest path:" in result.stdout

    nodes_path = output_directory / "graph_nodes.parquet"
    edges_path = output_directory / "graph_edges.parquet"
    manifest_path = output_directory / "graph_table_manifest.json"

    assert nodes_path.exists()
    assert edges_path.exists()
    assert manifest_path.exists()

    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)

    expected_node_columns = {
        "node_id",
        "entity_type",
        "entity_name",
        "node_key",
        "first_seen_timestamp",
        "last_seen_timestamp",
        "event_count",
        "label",
    }

    expected_edge_columns = {
        "edge_id",
        "source_node_id",
        "destination_node_id",
        "source_entity",
        "destination_entity",
        "source_entity_type",
        "destination_entity_type",
        "edge_type",
        "event_family",
        "event_type",
        "timestamp",
        "event_result",
        "label",
        "source_file",
        "row_number",
    }

    assert expected_node_columns.issubset(set(nodes.columns))
    assert expected_edge_columns.issubset(set(edges.columns))

    assert nodes["node_id"].is_unique
    assert nodes["node_key"].is_unique
    assert len(edges) == len(clean_events)

    assert edges["source_node_id"].isna().sum() == 0
    assert edges["destination_node_id"].isna().sum() == 0

    expected_node_keys = {
        "user::U001",
        "host::C001",
        "host_or_domain::corp.example",
        "host::C002",
        "process::powershell.exe",
        "user::U999",
    }

    assert expected_node_keys.issubset(set(nodes["node_key"]))

    redteam_edges = edges[
        edges["event_family"] == "redteam_ground_truth"
    ]

    assert len(redteam_edges) == 1
    assert set(redteam_edges["label"]) == {1}
    assert set(redteam_edges["edge_type"]) == {
        "redteam_ground_truth:redteam_activity"
    }

    labeled_nodes = nodes[nodes["label"] == 1]

    assert "user::U999" in set(labeled_nodes["node_key"])
    assert "host::C002" in set(labeled_nodes["node_key"])

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["input"]["clean_events_path"] == str(
        input_directory / "clean_all_events.parquet"
    )
    assert manifest["output"]["graph_nodes_path"] == str(nodes_path)
    assert manifest["output"]["graph_edges_path"] == str(edges_path)
    assert manifest["tables"]["enriched_nodes"]["rows"] == len(nodes)
    assert manifest["tables"]["edges"]["rows"] == len(edges)