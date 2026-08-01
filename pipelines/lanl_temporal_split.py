from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_SECONDS = 3600

REDTEAM_MARKERS = (
    "redteam_ground_truth",
    "confirmed_redteam",
    "redteam.txt.gz",
    "redteam_activity",
)

def build_redteam_mask(events: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=events.index)

    searchable_columns = [
        "event_family",
        "event_result",
        "source_file",
        "edge_type",
    ]

    pattern = "|".join(re.escape(value) for value in REDTEAM_MARKERS)

    for column in searchable_columns:
        if column in events.columns:
            mask |= (
                events[column]
                .astype("string")
                .str.contains(pattern, case=False, na=False)
            )

    return mask


def add_global_window_ids(
    events: pd.DataFrame,
    window_seconds: int,
) -> tuple[pd.DataFrame, int]:
    events = events.sort_values(
        [
            "timestamp",
            "source_node_id",
            "edge_type",
            "destination_node_id",
        ]
    ).reset_index(drop=True)

    dataset_start = int(events["timestamp"].min())

    events["window_id"] = (
        (events["timestamp"] - dataset_start) // window_seconds
    ).astype("int64")

    return events, dataset_start


def assign_chronological_splits(
    events: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, int, int]:
    number_of_windows = int(events["window_id"].max()) + 1

    train_end_window = int(number_of_windows * train_fraction)
    validation_end_window = int(
        number_of_windows * (train_fraction + validation_fraction)
    )

    if not 0 < train_end_window < validation_end_window < number_of_windows:
        raise ValueError(
            "The dataset must contain enough windows for train, validation, and test."
        )

    events["split"] = np.select(
        [
            events["window_id"] < train_end_window,
            events["window_id"] < validation_end_window,
        ],
        [
            "train",
            "validation",
        ],
        default="test",
    )

    return events, train_end_window, validation_end_window


def build_temporal_split(
    edge_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    events = pd.read_parquet(edge_path)

    required_columns = {
        "timestamp",
        "source_node_id",
        "destination_node_id",
        "edge_type",
    }

    missing_columns = required_columns - set(events.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Create dataset-wide contiguous window IDs.
    events, dataset_start = add_global_window_ids(
        events,
        window_seconds=WINDOW_SECONDS,
    )

    # Assign complete windows to chronological splits.
    events, train_end_window, validation_end_window = (
        assign_chronological_splits(events)
    )

    number_of_windows = int(events["window_id"].max()) + 1

    # Red-team rows are labels only, never model inputs.
    redteam_mask = build_redteam_mask(events)

    ordinary_events = events.loc[~redteam_mask].copy()
    redteam_events = events.loc[redteam_mask].copy()

    manifest = {
        "dataset_start_timestamp": dataset_start,
        "window_seconds": WINDOW_SECONDS,
        "number_of_windows": number_of_windows,

        "train_start_window": 0,
        "train_end_window_exclusive": train_end_window,

        "validation_start_window": train_end_window,
        "validation_end_window_exclusive": validation_end_window,

        "test_start_window": validation_end_window,
        "test_end_window_exclusive": number_of_windows,

        "maximum_window_id": number_of_windows - 1,

        "ordinary_event_count": len(ordinary_events),
        "redteam_event_count": len(redteam_events),
    }

    return ordinary_events, redteam_events, manifest


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/lanl/temporal"),
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ordinary_events, redteam_events, manifest = build_temporal_split(
        args.edges
    )

    ordinary_events.to_parquet(
        args.output_dir / "ordinary_temporal_events.parquet",
        index=False,
    )

    redteam_events.to_parquet(
        args.output_dir / "redteam_label_events.parquet",
        index=False,
    )

    with (args.output_dir / "split_manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()