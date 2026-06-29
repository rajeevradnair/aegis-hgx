from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_ingest_cic_sample_script_writes_processed_sample() -> None:
    input_csv = Path(
        "data/external/cicids2017/Monday-WorkingHours.pcap_ISCX.csv"
    )
    output_csv = Path(
        "data/processed/cicids2017/cic_sample.csv"
    )

    if not input_csv.exists():
        return

    if output_csv.exists():
        output_csv.unlink()

    result = subprocess.run(
        [
            sys.executable,
            "pipelines/ingest_cic_sample.py",
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--max-rows",
            "1000",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Output CSV:" in result.stdout
    assert "Sample label counts:" in result.stdout
    assert output_csv.exists()

    dataset = pd.read_csv(output_csv)

    assert not dataset.empty
    assert "Label" in dataset.columns
    assert len(dataset) <= 1000