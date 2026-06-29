from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aegis_hgx.utils.hashing import sha256_file

import pandas as pd

DEFAULT_LINEAGE_MANIFEST_PATH = Path(
    "artifacts/lineage/logistic_baseline_manifest.json"
)
DEFAULT_OUTPUT_REPORT_PATH = Path(
    "reports/monitoring/basic_drift_report.html"
)

def load_lineage_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Lineage manifest not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict):
        raise ValueError("Lineage manifest must contain a JSON object.")

    return manifest


def get_reference_data_path(manifest: dict[str, Any]) -> Path:
    training_data = manifest.get("training_data")

    if not isinstance(training_data, dict):
        raise ValueError("Lineage manifest is missing training_data section.")

    data_path = training_data.get("data_path")

    if not isinstance(data_path, str):
        raise ValueError("Lineage manifest is missing training_data.data_path.")

    return Path(data_path)


def get_reference_data_hash(manifest: dict[str, Any]) -> str:
    training_data = manifest.get("training_data")

    if not isinstance(training_data, dict):
        raise ValueError("Lineage manifest is missing training_data section.")

    data_hash = training_data.get("data_sha256")

    if not isinstance(data_hash, str):
        raise ValueError(
            "Lineage manifest is missing training_data.data_sha256."
        )

    return data_hash


def verify_reference_data_hash(
    reference_data_path: Path,
    expected_hash: str,
) -> None:
    actual_hash = sha256_file(reference_data_path)

    if actual_hash != expected_hash:
        raise ValueError(
            "Reference data hash mismatch. "
            f"Expected {expected_hash}, got {actual_hash}. "
            "The local dataset no longer matches the lineage manifest."
        )

def load_monitoring_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Monitoring dataset not found: {path}")

    dataset = pd.read_csv(path)

    if dataset.empty:
        raise ValueError("Monitoring dataset contains no rows.")

    return dataset


def select_monitoring_features(dataset: pd.DataFrame) -> pd.DataFrame:
    excluded_columns = {"timestamp", "label"}
    selected_columns = [
        column
        for column in dataset.columns
        if column not in excluded_columns
    ]

    if not selected_columns:
        raise ValueError("No monitoring feature columns were selected.")

    return dataset[selected_columns].copy()


def split_reference_current(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(dataset) < 2:
        raise ValueError(
            "Monitoring dataset must contain at least two rows."
        )

    split_index = len(dataset) // 2

    reference_data = dataset.iloc[:split_index].copy()
    current_data = dataset.iloc[split_index:].copy()

    if reference_data.empty or current_data.empty:
        raise ValueError(
            "Reference and current data splits must both be non-empty."
        )

    return reference_data, current_data


def create_data_drift_report() -> Any:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        return Report([DataDriftPreset()])
    except ImportError:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        return Report(metrics=[DataDriftPreset()])

def write_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = create_data_drift_report()
    result = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    # print("****", type(report))
    # print("****", type(result))

    if hasattr(result, "save_html"):
        result.save_html(str(output_path))
    else:
        report.save_html(str(output_path))

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a basic data drift report."
    )
    parser.add_argument(
        "--lineage-manifest",
        type=Path,
        default=DEFAULT_LINEAGE_MANIFEST_PATH,
        help="Path to the model lineage manifest JSON file.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT_PATH,
        help="Path where the drift report HTML should be written.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    manifest = load_lineage_manifest(args.lineage_manifest)
    reference_data_path = get_reference_data_path(manifest)
    expected_reference_hash = get_reference_data_hash(manifest)

    verify_reference_data_hash(
        reference_data_path=reference_data_path,
        expected_hash=expected_reference_hash,
    )

    dataset = load_monitoring_dataset(reference_data_path)
    monitoring_features = select_monitoring_features(dataset)
    reference_data, current_data = split_reference_current(
        monitoring_features
    )

    output_report_path = write_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        output_path=args.output_report,
    )

    print("Lineage manifest:", args.lineage_manifest)
    print("Reference data:", reference_data_path)
    print("Reference data hash verified:", expected_reference_hash)
    print("Monitoring columns:", list(monitoring_features.columns))
    print("Reference rows:", len(reference_data))
    print("Current rows:", len(current_data))
    print("Output report:", args.output_report)


if __name__ == "__main__":
    main()