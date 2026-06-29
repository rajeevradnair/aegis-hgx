from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_build_tabular_features_script_writes_model_ready_dataset() -> None:
    input_csv = Path(
        "data/processed/cicids2017/cic_sample.csv"
    )
    output_csv = Path(
        "data/processed/cicids2017/cic_tabular_features.csv"
    )

    if not input_csv.exists():
        return

    if output_csv.exists():
        output_csv.unlink()

    result = subprocess.run(
        [
            sys.executable,
            "pipelines/build_tabular_features.py",
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Output CSV:" in result.stdout
    assert "Feature summary:" in result.stdout
    assert "Target counts:" in result.stdout
    assert output_csv.exists()

    dataset = pd.read_csv(output_csv)

    assert not dataset.empty
    assert "target" in dataset.columns
    assert set(dataset["target"].unique()).issubset({0, 1})

    feature_columns = [
        column
        for column in dataset.columns
        if column != "target"
    ]

    assert feature_columns

    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(dataset[column])
    ]

    assert non_numeric_features == []