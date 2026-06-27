"""Tests for synthetic cyber event data generation."""

from pathlib import Path

import pandas as pd

from pipelines.generate_synthetic_events import (
    REQUIRED_COLUMNS,
    generate_dataset,
    load_generation_config,
    save_dataset,
    validate_generation_config
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

def test_generated_dataset_has_non_negative_bytes() -> None:
    config = load_generation_config("configs/data_generation.yaml")

    dataset = generate_dataset(config)

    assert (dataset["bytes_in"] >= 0).all()
    assert (dataset["bytes_out"] >= 0).all()

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

def test_generation_is_reproducible_with_same_seed() -> None:
    config = load_generation_config("configs/data_generation.yaml")

    first_dataset = generate_dataset(config)
    second_dataset = generate_dataset(config)

    pd.testing.assert_frame_equal(first_dataset, second_dataset)


def test_generation_config_rejects_invalid_num_events() -> None:
    config = load_generation_config("configs/data_generation.yaml")
    config["generation"]["num_events"] = 0

    try:
        validate_generation_config(config)
    except ValueError as error:
        assert "num_events" in str(error)
    else:
        raise AssertionError("Expected invalid num_events to raise ValueError")
    
def test_generated_dataset_respects_anomaly_rate_sanity_range() -> None:
    config = load_generation_config("configs/data_generation.yaml")

    dataset = generate_dataset(config)

    configured_rate = config["generation"]["anomaly_rate"]
    observed_rate = dataset["label"].mean()

    assert abs(observed_rate - configured_rate) <= 0.05