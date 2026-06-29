from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_CSV = Path(
    "data/external/cicids2017/Monday-WorkingHours.pcap_ISCX.csv"
)
DEFAULT_OUTPUT_CSV = Path(
    "data/processed/cicids2017/cic_sample.csv"
)
DEFAULT_MAX_ROWS = 10000


def load_cic_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CIC input CSV not found: {path}")

    dataset = pd.read_csv(path)

    if dataset.empty:
        raise ValueError(f"CIC input CSV contains no rows: {path}")

    dataset.columns = [
        column.strip()
        for column in dataset.columns
    ]

    return dataset


def validate_label_column(dataset: pd.DataFrame) -> None:
    if "Label" not in dataset.columns:
        raise ValueError("CIC dataset must contain a Label column.")
    

def get_label_counts(dataset: pd.DataFrame) -> dict[str, int]:
    validate_label_column(dataset)

    counts = dataset["Label"].value_counts(dropna=False)

    return {
        str(label): int(count)
        for label, count in counts.items()
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a local CICIDS2017 CSV sample."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to a local CICIDS2017 CSV file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path where the processed CIC sample should be written.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Maximum number of rows to keep in the sample.",
    )
    return parser.parse_args()

def sample_dataset(
    dataset: pd.DataFrame,
    max_rows: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    if max_rows <= 0:
        raise ValueError("max_rows must be greater than zero.")

    if len(dataset) <= max_rows:
        return dataset.copy()
    
    dataset.sample(n=1000, frac=0.5, )

    return dataset.sample(
        n=max_rows,
        random_state=random_seed,
    ).reset_index(drop=True)


def sample_dataset(
    dataset: pd.DataFrame,
    max_rows: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    if max_rows <= 0:
        raise ValueError("max_rows must be greater than zero.")

    if len(dataset) <= max_rows:
        return dataset.copy()

    return dataset.sample(
        n=max_rows,
        random_state=random_seed, 
    ).reset_index(drop=True)


def write_cic_sample(
    dataset: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return output_path


def main() -> None:

    args = parse_args()

    dataset = load_cic_csv(args.input_csv)
    original_label_counts = get_label_counts(dataset)

    sampled_dataset = sample_dataset(
        dataset=dataset,
        max_rows=args.max_rows,
    )
    sampled_label_counts = get_label_counts(sampled_dataset)

    output_path = write_cic_sample(
        dataset=sampled_dataset,
        output_path=args.output_csv,
    )

    print("Input CSV:", args.input_csv)
    print("Output CSV:", output_path)
    print("Max rows:", args.max_rows)
    print("Loaded rows:", len(dataset))
    print("Loaded columns:", len(dataset.columns))
    print("Sample rows:", len(sampled_dataset))
    print("First columns:", list(sampled_dataset.columns[:10]))
    print("Original label counts:", original_label_counts)
    print("Sample label counts:", sampled_label_counts)


if __name__ == "__main__":
    main()