"""Tests for synthetic cyber event data generation."""

from pathlib import Path

import pandas as pd

from pipelines.generate_synthetic_events import (
    REQUIRED_COLUMNS,
    generate_dataset,
    load_generation_config,
    save_dataset,
)


def test_generated_dataset_has_expected_shape() -> None:
    config = load_generation_config("configs/data_generation.yaml")

    dataset = generate_dataset(config)

    assert dataset.shape[0] == config["generation"]["num_events"]
    assert list(dataset.columns) == REQUIRED_COLUMNS


def test_generated_dataset_has_binary_labels() -> None:
    config = load_generation_config("configs/data_generation.yaml")

    dataset = generate_dataset(config)

    labels = set(dataset["label"].unique())

    assert labels.issubset({0, 1})
    assert 0 in labels
    assert 1 in labels


def test_generated_dataset_has_valid_event_hours() -> None:
    config = load_generation_config("configs/data_generation.yaml")

    dataset = generate_dataset(config)

    assert dataset["event_hour"].between(0, 23).all()


def test_generated_dataset_has_boolean_business_hour_flag() -> None:
    config = load_generation_config("configs/data_generation.yaml")

    dataset = generate_dataset(config)

    assert dataset["is_business_hour"].isin([True, False]).all()


def test_save_dataset_writes_csv(tmp_path: Path) -> None:
    config = load_generation_config("configs/data_generation.yaml")
    dataset = generate_dataset(config)

    output_path = tmp_path / "synthetic_events.csv"
    saved_path = save_dataset(dataset, output_path)

    assert saved_path.exists()

    reloaded = pd.read_csv(saved_path)

    assert reloaded.shape[0] == config["generation"]["num_events"]
    assert list(reloaded.columns) == REQUIRED_COLUMNS