from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import yaml


def write_parquet(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)


def test_build_lanl_event_tables_from_temporary_parquet_inputs(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "processed_input"
    output_directory = tmp_path / "processed_output"
    config_path = tmp_path / "lanl_event_tables.yaml"

    input_directory.mkdir(parents=True)
    output_directory.mkdir(parents=True)

    write_parquet(
        input_directory / "auth.parquet",
        pd.DataFrame(
            {
                "row_number": [0, 1],
                "timestamp": [1, 2],
                "source_user": ["U001", "U002"],
                "destination_user": ["U001", "U002"],
                "source_host": ["C001", "C003"],
                "destination_host": ["C002", "C004"],
                "auth_type": ["Kerberos", "NTLM"],
                "logon_type": ["Network", "RemoteInteractive"],
                "auth_orientation": ["LogOn", "LogOn"],
                "result": ["Success", "Failure"],
                "source_file": ["auth.txt.gz", "auth.txt.gz"],
                "event_family": ["authentication", "authentication"],
            }
        ),
    )

    write_parquet(
        input_directory / "dns.parquet",
        pd.DataFrame(
            {
                "row_number": [0, 1],
                "timestamp": [1, 2],
                "source_host": ["C001", "C002"],
                "resolved_host": ["corp.example", "login.example"],
                "source_file": ["dns.txt.gz", "dns.txt.gz"],
                "event_family": ["dns", "dns"],
            }
        ),
    )

    write_parquet(
        input_directory / "flows.parquet",
        pd.DataFrame(
            {
                "row_number": [0, 1],
                "timestamp": [1, 2],
                "duration": [10, 5],
                "source_host": ["C001", "C003"],
                "source_port": ["12345", "23456"],
                "destination_host": ["C002", "C004"],
                "destination_port": ["443", "80"],
                "protocol": ["TCP", "TCP"],
                "packet_count": [12, 7],
                "byte_count": [4096, 2048],
                "source_file": ["flows.txt.gz", "flows.txt.gz"],
                "event_family": ["network_flow", "network_flow"],
            }
        ),
    )

    write_parquet(
        input_directory / "proc.parquet",
        pd.DataFrame(
            {
                "row_number": [0, 1],
                "timestamp": [1, 2],
                "source_user": ["U001", "U002"],
                "host": ["C001", "C003"],
                "process_name": ["powershell.exe", "cmd.exe"],
                "process_event": ["Start", "Start"],
                "source_file": ["proc.txt.gz", "proc.txt.gz"],
                "event_family": ["process", "process"],
            }
        ),
    )

    write_parquet(
        input_directory / "redteam.parquet",
        pd.DataFrame(
            {
                "row_number": [0, 1],
                "timestamp": [2, 5],
                "source_user": ["U002", "U009"],
                "source_host": ["C003", "C010"],
                "destination_host": ["C004", "C011"],
                "source_file": ["redteam.txt.gz", "redteam.txt.gz"],
                "event_family": [
                    "redteam_ground_truth",
                    "redteam_ground_truth",
                ],
            }
        ),
    )

    config = {
        "input": {
            "directory": str(input_directory),
            "files": {
                "auth": "auth.parquet",
                "dns": "dns.parquet",
                "flows": "flows.parquet",
                "proc": "proc.parquet",
                "redteam": "redteam.parquet",
            },
        },
        "output": {
            "directory": str(output_directory),
            "files": {
                "clean_auth": "clean_auth_events.parquet",
                "clean_dns": "clean_dns_events.parquet",
                "clean_flows": "clean_flow_events.parquet",
                "clean_process": "clean_process_events.parquet",
                "clean_redteam": "clean_redteam_events.parquet",
                "clean_all": "clean_all_events.parquet",
            },
            "manifest_filename": "event_table_manifest.json",
        },
        "schema": {
            "event_id_prefix": "lanl",
            "unknown_value": "unknown",
            "default_label": 0,
        },
    }

    config_path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "pipelines/build_lanl_event_tables.py",
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Built combined event table:" in result.stdout
    assert "Event table manifest path:" in result.stdout

    expected_outputs = [
        "clean_auth_events.parquet",
        "clean_dns_events.parquet",
        "clean_flow_events.parquet",
        "clean_process_events.parquet",
        "clean_redteam_events.parquet",
        "clean_all_events.parquet",
        "event_table_manifest.json",
    ]

    for filename in expected_outputs:
        assert (output_directory / filename).exists()

    clean_all = pd.read_parquet(output_directory / "clean_all_events.parquet")

    assert len(clean_all) == 10

    expected_common_columns = {
        "event_id",
        "timestamp",
        "event_family",
        "event_type",
        "source_entity",
        "destination_entity",
        "source_entity_type",
        "destination_entity_type",
        "event_result",
        "label",
        "source_file",
        "row_number",
    }

    assert expected_common_columns.issubset(set(clean_all.columns))

    event_family_counts = clean_all["event_family"].value_counts().to_dict()

    assert event_family_counts["authentication"] == 2
    assert event_family_counts["dns"] == 2
    assert event_family_counts["network_flow"] == 2
    assert event_family_counts["process"] == 2
    assert event_family_counts["redteam_ground_truth"] == 2

    label_counts = clean_all["label"].value_counts().to_dict()

    assert label_counts[0] == 8
    assert label_counts[1] == 2

    redteam_events = clean_all[
        clean_all["event_family"] == "redteam_ground_truth"
    ]

    assert set(redteam_events["label"]) == {1}
    assert set(redteam_events["event_result"]) == {"confirmed_redteam"}

    manifest_path = output_directory / "event_table_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["table_count"] == 6
    assert manifest["combined_table"]["rows"] == 10
    assert manifest["combined_table"]["event_family_counts"][
        "authentication"
    ] == 2
    assert manifest["combined_table"]["event_family_counts"][
        "redteam_ground_truth"
    ] == 2