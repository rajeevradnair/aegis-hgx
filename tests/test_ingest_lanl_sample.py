from pathlib import Path
import gzip
import json
import subprocess
import sys

import pandas as pd
import yaml


def write_gzip_text(path: Path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as file:
        file.write(text)


def test_ingest_lanl_sample_uses_temporary_files(tmp_path: Path) -> None:
    input_directory = tmp_path / "external"
    output_directory = tmp_path / "processed"
    config_path = tmp_path / "lanl_ingest.yaml"

    input_directory.mkdir(parents=True)
    output_directory.mkdir(parents=True)

    write_gzip_text(
        input_directory / "auth.txt.gz",
        "\n".join(
            [
                "1,U001,U001,C001,C002,Kerberos,Network,LogOn,Success",
                "2,U002,U002,C003,C004,NTLM,RemoteInteractive,LogOn,Failure",
            ]
        ),
    )
    write_gzip_text(
        input_directory / "dns.txt.gz",
        "\n".join(
            [
                "1,C001,corp.example",
                "2,C002,login.example",
            ]
        ),
    )
    write_gzip_text(
        input_directory / "flows.txt.gz",
        "\n".join(
            [
                "1,10,C001,12345,C002,443,TCP,12,4096",
                "2,5,C003,23456,C004,80,TCP,7,2048",
            ]
        ),
    )
    write_gzip_text(
        input_directory / "proc.txt.gz",
        "\n".join(
            [
                "1,U001,C001,powershell.exe,Start",
                "2,U002,C003,cmd.exe,Start",
            ]
        ),
    )
    write_gzip_text(
        input_directory / "redteam.txt.gz",
        "\n".join(
            [
                "2,U002,C003,C004",
                "5,U009,C010,C011",
            ]
        ),
    )

    config = {
        "input": {
            "directory": str(input_directory),
            "compression": "gzip",
            "max_rows_per_file": None,
        },
        "files": {
            "auth": {
                "filename": "auth.txt.gz",
                "output_filename": "auth.parquet",
                "event_family": "authentication",
            },
            "dns": {
                "filename": "dns.txt.gz",
                "output_filename": "dns.parquet",
                "event_family": "dns",
            },
            "flows": {
                "filename": "flows.txt.gz",
                "output_filename": "flows.parquet",
                "event_family": "network_flow",
            },
            "proc": {
                "filename": "proc.txt.gz",
                "output_filename": "proc.parquet",
                "event_family": "process",
            },
            "redteam": {
                "filename": "redteam.txt.gz",
                "output_filename": "redteam.parquet",
                "event_family": "redteam_ground_truth",
            },
        },
        "output": {
            "directory": str(output_directory),
            "manifest_filename": "ingest_manifest.json",
        },
        "schema": {
            "add_source_file_column": True,
            "add_event_family_column": True,
            "add_row_number_column": True,
            "normalize_column_names": True,
        },
    }

    config_path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "pipelines/ingest_lanl_sample.py",
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Configured file count: 5" in result.stdout
    assert "Manifest path:" in result.stdout

    expected_outputs = [
        "auth.parquet",
        "dns.parquet",
        "flows.parquet",
        "proc.parquet",
        "redteam.parquet",
    ]

    for filename in expected_outputs:
        path = output_directory / filename
        assert path.exists()

        dataframe = pd.read_parquet(path)

        assert len(dataframe) == 2
        assert "row_number" in dataframe.columns
        assert "source_file" in dataframe.columns
        assert "event_family" in dataframe.columns

    auth_dataframe = pd.read_parquet(output_directory / "auth.parquet")

    assert list(auth_dataframe["row_number"]) == [0, 1]
    assert set(auth_dataframe["source_file"]) == {"auth.txt.gz"}
    assert set(auth_dataframe["event_family"]) == {"authentication"}

    manifest_path = output_directory / "ingest_manifest.json"

    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["file_count"] == 5
    assert manifest["input_directory"] == str(input_directory)
    assert manifest["output_directory"] == str(output_directory)
    assert len(manifest["files"]) == 5

    for item in manifest["files"]:
        assert item["rows"] == 2
        assert Path(item["output_path"]).exists()