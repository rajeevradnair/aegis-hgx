from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def load_events(temporal_dir: Path) -> pd.DataFrame:
    path = temporal_dir / "ordinary_temporal_events.parquet"
    return pd.read_parquet(path)

RELATION_KEY = [
    "split",
    "source_node_id",
    "edge_type",
    "destination_node_id",
]

def add_temporal_features(events: pd.DataFrame) -> pd.DataFrame:
    events = events.sort_values(
        [
            "timestamp",
            "source_node_id",
            "edge_type",
            "destination_node_id",
        ]
    ).reset_index(drop=True)

    grouped = events.groupby(
        RELATION_KEY,
        sort=False,
        observed=True,
    )

    events["event_count_so_far"] = (
        grouped.cumcount() + 1
    )

    events["last_seen_timestamp"] = (
        grouped["timestamp"].shift(1)
    )

    events["time_since_previous"] = (
        events["timestamp"]
        - events["last_seen_timestamp"]
    )

    events["first_seen_timestamp"] = (
        grouped["timestamp"].cummin()
    )

    gap = events["time_since_previous"]

    gap_sum_through_current = (
        gap.fillna(0)
        .groupby(
            [events[column] for column in RELATION_KEY],
            sort=False,
        )
        .cumsum()
    )

    previous_gap_sum = (
        gap_sum_through_current - gap.fillna(0)
    )

    valid_gap = gap.notna().astype("int64")

    gap_count_through_current = (
        valid_gap.groupby(
            [events[column] for column in RELATION_KEY],
            sort=False,
        )
        .cumsum()
    )

    previous_gap_count = (
        gap_count_through_current - valid_gap
    )

    events["mean_previous_gap"] = (
        previous_gap_sum
        / previous_gap_count.replace(0, np.nan)
    )

    window_counts = (
        events.groupby(
            RELATION_KEY + ["window_id"],
            sort=False,
            observed=True,
        )
        .size()
        .rename("events_in_window")
        .reset_index()
    )

    previous_window_counts = window_counts.copy()

    previous_window_counts["window_id"] += 1

    previous_window_counts = previous_window_counts.rename(
        columns={
            "events_in_window": "recent_event_count",
        }
    )

    events = events.merge(
        previous_window_counts,
        on=RELATION_KEY + ["window_id"],
        how="left",
    )

    events["recent_event_count"] = (
        events["recent_event_count"]
        .fillna(0)
        .astype("int64")
    )

    return events


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--temporal-dir",
        type=Path,
        default=Path("data/processed/lanl/temporal"),
    )

    args = parser.parse_args()

    events = load_events(args.temporal_dir)
    events = add_temporal_features(events)

    output_path = (
        args.temporal_dir
        / "temporal_edge_features.parquet"
    )

    events.to_parquet(output_path, index=False)

    feature_columns = [
        "event_count_so_far",
        "time_since_previous",
        "recent_event_count",
        "mean_previous_gap",
        "first_seen_timestamp",
        "last_seen_timestamp",
    ]

    print(f"Saved: {output_path}")
    print(events[feature_columns].head(10))


if __name__ == "__main__":
    main()