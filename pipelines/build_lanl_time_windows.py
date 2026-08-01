from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

def load_inputs(
    temporal_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    events = pd.read_parquet(
        temporal_dir / "ordinary_temporal_events.parquet"
    )

    with (temporal_dir / "split_manifest.json").open() as file:
        manifest = json.load(file)

    return events, manifest

def build_window_metadata(
    events: pd.DataFrame,
    manifest: dict,
) -> pd.DataFrame:
    number_of_windows = manifest["number_of_windows"]
    dataset_start = manifest["dataset_start_timestamp"]
    window_seconds = manifest["window_seconds"]

    windows = pd.DataFrame({
        "window_id": np.arange(number_of_windows, dtype=np.int64)
    })

    windows["window_start_timestamp"] = (
        dataset_start
        + windows["window_id"] * window_seconds
    )

    windows["window_end_timestamp"] = (
        windows["window_start_timestamp"]
        + window_seconds
    )

    train_end = manifest["train_end_window_exclusive"]
    validation_end = manifest[
        "validation_end_window_exclusive"
    ]

    windows["split"] = np.select(
        [
            windows["window_id"] < train_end,
            windows["window_id"] < validation_end,
        ],
        [
            "train",
            "validation",
        ],
        default="test",
    )

    event_counts = (
        events.groupby("window_id")
        .size()
        .rename("ordinary_event_count")
        .reset_index()
    )

    windows = windows.merge(
        event_counts,
        on="window_id",
        how="left",
    )

    windows["ordinary_event_count"] = (
        windows["ordinary_event_count"]
        .fillna(0)
        .astype("int64")
    )

    windows["is_empty"] = (
        windows["ordinary_event_count"] == 0
    )

    return windows


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--temporal-dir",
        type=Path,
        default=Path("data/processed/lanl/temporal"),
    )

    args = parser.parse_args()

    events, manifest = load_inputs(args.temporal_dir)

    windows = build_window_metadata(
        events,
        manifest,
    )

    output_path = args.temporal_dir / "window_metadata.parquet"
    windows.to_parquet(output_path, index=False)

    print(f"Saved: {output_path}")
    print(windows.groupby("split").size())
    print(f"Empty windows: {windows['is_empty'].sum()}")


if __name__ == "__main__":
    main()