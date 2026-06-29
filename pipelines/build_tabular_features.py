from __future__ import annotations

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np

DEFAULT_INPUT_CSV = Path(
    "data/processed/cicids2017/cic_sample.csv"
)
DEFAULT_OUTPUT_CSV = Path(
    "data/processed/cicids2017/cic_tabular_features.csv"
)

CATEGORICAL_NUMERIC_FEATURES = {
    "destination_port",
    "protocol",
}

EXCLUDED_COLUMNS = {
    "label",
}

def validate_label_column(dataset: pd.DataFrame) -> None:
    if "label" not in dataset.columns:
        raise ValueError("CIC feature dataset must contain a label column.")


def build_binary_target(dataset: pd.DataFrame) -> pd.Series:
    validate_label_column(dataset)

    labels = dataset["label"].astype(str).str.strip().str.upper()

    return (labels != "BENIGN").astype(int)

def build_numeric_features(dataset: pd.DataFrame) -> pd.DataFrame:
    feature_candidates = dataset.drop(
        columns=[
            column
            for column in EXCLUDED_COLUMNS
            if column in dataset.columns
        ]
    )

    numeric_features = feature_candidates.select_dtypes(
        include=["number", "bool"]
    ).copy()

    numeric_features = numeric_features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    numeric_features = numeric_features.dropna(
        axis=1,
        how="all",
    )

    numeric_features = numeric_features.fillna(0)

    return numeric_features

def summarize_feature_groups(features: pd.DataFrame) -> dict[str, int]:
    categorical_numeric_count = len(
        [
            column
            for column in features.columns
            if column in CATEGORICAL_NUMERIC_FEATURES
        ]
    )

    binary_indicator_count = len(
        [
            column
            for column in features.columns
            if set(features[column].dropna().unique()).issubset({0, 1})
        ]
    )

    continuous_numeric_count = (
        len(features.columns)
        - categorical_numeric_count
        - binary_indicator_count
    )

    return {
        "total_features": int(len(features.columns)),
        "continuous_numeric_features": int(continuous_numeric_count),
        "binary_indicator_features": int(binary_indicator_count),
        "categorical_numeric_features": int(categorical_numeric_count),
    }

def normalize_column_name(column: str) -> str:
    normalized = column.strip().lower()
    normalized = normalized.replace("/", "_per_")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")

def load_cic_sample(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CIC sample CSV not found: {path}")

    dataset = pd.read_csv(path)

    if dataset.empty:
        raise ValueError(f"CIC sample CSV contains no rows: {path}")

    dataset = dataset.rename(
        columns={
            column: normalize_column_name(column)
            for column in dataset.columns
        }
    )

    return dataset

def build_tabular_feature_dataset(
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    output = features.copy()
    output["target"] = target.to_numpy()
    return output

def write_tabular_features(
    dataset: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return output_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clean tabular features from a CIC sample."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to the processed CIC sample CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path where the tabular feature CSV should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_cic_sample(args.input_csv)
    target = build_binary_target(dataset)
    features = build_numeric_features(dataset)
    feature_summary = summarize_feature_groups(features)

    output_dataset = build_tabular_feature_dataset(
        features=features,
        target=target,
    )
    output_path = write_tabular_features(
        dataset=output_dataset,
        output_path=args.output_csv,
    )

    print("Input CSV:", args.input_csv)
    print("Output CSV:", output_path)
    print("Rows:", len(output_dataset))
    print("Input columns:", len(dataset.columns))
    print("Output columns:", len(output_dataset.columns))
    print("Feature summary:", feature_summary)
    print("Target counts:", target.value_counts().sort_index().to_dict())
    print("First features:", list(features.columns[:10]))

if __name__ == "__main__":
    main()