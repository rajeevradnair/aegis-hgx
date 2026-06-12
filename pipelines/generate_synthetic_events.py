from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from aegis_hgx.utils.logging import configure_logging, get_logger
from aegis_hgx.utils.seeds import set_global_seed

LOGGER = get_logger(__name__)

CONFIG_PATH = Path("configs/data_generation.yaml")

REQUIRED_COLUMNS = []

"""
REQUIRED_COLUMNS = [
    "timestamp",
    "user_id",
    "host_id",
    "process_name",
    "event_type",
    "source_ip",
    "destination_ip",
    "bytes_in",
    "bytes_out",
    "event_hour",
    "is_business_hour",
    "label",
]
"""

def load_generation_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Generation config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Generation config is empty: {config_path}")

    if not isinstance(config, dict):
        raise TypeError(f"Generation config must be a YAML mapping: {config_path}")

    return config


def create_entity_pool(prefix: str, count: int) -> list[str]:
    if count <= 0:
        raise ValueError(f"Entity count must be positive for prefix: {prefix}")
    return [f"{prefix}_{i:03d}" for i in range(count)]


def create_internal_ips(count: int) -> list[str]:
    if count <= 0:
        raise ValueError("Internal IP count must be positive")
    return [f"10.0.0.{i + 1}" for i in range(count)]


def create_external_ips(count):
    if count <= 0:
        raise ValueError("External IP count must be positive.")
    return [f"203.0.113.{index + 1}" for index in range(count)]


def choose_event_timestamp(start_timestamp: str, allowed_hours: list[int]) -> datetime:
    if not allowed_hours:
        raise ValueError("Allowed hours cannot be empty.")

    invalid_hours = [hour for hour in allowed_hours if hour < 0 or hour > 23]
    if invalid_hours:
        raise ValueError(f"Invalid event hours: {invalid_hours}")

    start = datetime.fromisoformat(start_timestamp)
    day_offset = random.randint(0, 6)
    hour = random.choice(allowed_hours)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)

    return start + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)


def is_business_hour_1(
    timestamp: datetime, business_hour_start: int = 8, business_hour_end: int = 18
) -> bool:
    if business_hour_start < 0 or business_hour_start > 23:
        raise ValueError("Business hour start must be between 0 and 23.")

    if business_hour_end < 0 or business_hour_end > 23:
        raise ValueError("Business hour end must be between 0 and 23.")

    if business_hour_start >= business_hour_end:
        raise ValueError("Business hour start must be earlier than business hour end.")

    return business_hour_start <= timestamp.hour <= business_hour_end


def is_business_hour(event_hour: int, business_hour_start: int, business_hour_end: int) -> bool:
    if business_hour_start < 0 or business_hour_start > 23:
        raise ValueError("Business hour start must be between 0 and 23.")

    if business_hour_end < 0 or business_hour_end > 23:
        raise ValueError("Business hour end must be between 0 and 23.")

    if business_hour_start >= business_hour_end:
        raise ValueError("Business hour start must be earlier than business hour end.")

    return business_hour_start <= event_hour < business_hour_end


def generate_normal_event(
    users: list[str],
    hosts: list[str],
    internal_ips: list[str],
    external_ips: list[str],
    common_processes: list[str],
    common_event_types: list[str],
    start_timestamp: str,
    business_hour_start: int,
    business_hour_end: int,
) -> dict[str, object]:
    allowed_hours = list(range(business_hour_start, business_hour_end))
    timestamp = choose_event_timestamp(start_timestamp, allowed_hours)
    event_hour = timestamp.hour
    destination_pool = internal_ips if random.random() < 0.9 else external_ips

    return {
        "timestamp": timestamp.isoformat(),
        "user_id": random.choice(users),
        "host_id": random.choice(hosts),
        "process_name": random.choice(common_processes),
        "event_type": random.choice(common_event_types),
        "source_ip": random.choice(internal_ips),
        "destination_ip": random.choice(destination_pool),
        "bytes_in": random.randint(100, 20_000),
        "bytes_out": random.randint(50, 10_000),
        "event_hour": event_hour,
        "is_business_hour": is_business_hour(
            event_hour=event_hour,
            business_hour_start=business_hour_start,
            business_hour_end=business_hour_end,
        ),
        "label": 0,
    }


def generate_suspicious_event(
    users: list[str],
    hosts: list[str],
    internal_ips: list[str],
    external_ips: list[str],
    suspicious_processes: list[str],
    suspicious_event_types: list[str],
    suspicious_hours: list[int],
    start_timestamp: str,
    business_hour_start: int,
    business_hour_end: int,
) -> dict[str, object]:
    timestamp = choose_event_timestamp(start_timestamp, suspicious_hours)
    event_hour = timestamp.hour
    destination_pool = external_ips if random.random() < 0.8 else internal_ips

    return {
        "timestamp": timestamp.isoformat(),
        "user_id": random.choice(users),
        "host_id": random.choice(hosts),
        "process_name": random.choice(suspicious_processes),
        "event_type": random.choice(suspicious_event_types),
        "source_ip": random.choice(internal_ips),
        "destination_ip": random.choice(destination_pool),
        "bytes_in": random.randint(100, 30_000),
        "bytes_out": random.randint(20_000, 250_000),
        "event_hour": event_hour,
        "is_business_hour": is_business_hour(
            event_hour=event_hour,
            business_hour_start=business_hour_start,
            business_hour_end=business_hour_end,
        ),
        "label": 1,
    }


def validate_generation_config(config: dict[str, Any]) -> None:
    required_top_level_keys = [
        "generation",
        "entities",
        "normal_behavior",
        "suspicious_behavior",
    ]

    missing_keys = [key for key in required_top_level_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config sections: {missing_keys}")

    num_events = config["generation"]["num_events"]
    anomaly_rate = config["generation"]["anomaly_rate"]

    if num_events <= 0:
        raise ValueError("num_events must be positive.")

    if anomaly_rate < 0 or anomaly_rate > 1:
        raise ValueError("anomaly_rate must be between 0 and 1.")


def generate_dataset(config: dict[str, Any]) -> pd.DataFrame:
    validate_generation_config(config)

    generation_config = config["generation"]
    entity_config = config["entities"]
    normal_config = config["normal_behavior"]
    suspicious_config = config["suspicious_behavior"]

    num_events = int(generation_config["num_events"])
    anomaly_rate = float(generation_config["anomaly_rate"])
    num_suspicious = int(num_events * anomaly_rate)
    num_normal = num_events - num_suspicious

    users = create_entity_pool("user", int(entity_config["num_users"]))
    hosts = create_entity_pool("host", int(entity_config["num_hosts"]))
    internal_ips = create_internal_ips(int(entity_config["num_internal_ips"]))
    external_ips = create_external_ips(int(entity_config["num_external_ips"]))

    start_timestamp = str(generation_config["start_timestamp"])
    business_hour_start = int(normal_config["business_hour_start"])
    business_hour_end = int(normal_config["business_hour_end"])

    normal_events = [
        generate_normal_event(
            users=users,
            hosts=hosts,
            internal_ips=internal_ips,
            external_ips=external_ips,
            common_processes=list(normal_config["common_processes"]),
            common_event_types=list(normal_config["common_event_types"]),
            start_timestamp=start_timestamp,
            business_hour_start=business_hour_start,
            business_hour_end=business_hour_end,
        )
        for _ in range(num_normal)
    ]

    suspicious_events = [
        generate_suspicious_event(
            users=users,
            hosts=hosts,
            internal_ips=internal_ips,
            external_ips=external_ips,
            suspicious_processes=list(suspicious_config["suspicious_processes"]),
            suspicious_event_types=list(suspicious_config["suspicious_event_types"]),
            suspicious_hours=list(suspicious_config["suspicious_hours"]),
            start_timestamp=start_timestamp,
            business_hour_start=business_hour_start,
            business_hour_end=business_hour_end,
        )
        for _ in range(num_suspicious)
    ]

    events = normal_events + suspicious_events
    random.shuffle(events)

    dataset = pd.DataFrame(events, columns=REQUIRED_COLUMNS)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Generated dataset is missing columns: {missing_columns}")

    return dataset


def save_dataset(dataset: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic tabular cyber event dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/data_generation.yaml",
        help="Path to the synthetic data generation config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_generation_config(args.config)

    global REQUIRED_COLUMNS 
    REQUIRED_COLUMNS = config["generation"]["required_columns"] 
    random_seed = int(config["generation"]["random_seed"])
    output_path = Path(config["generation"]["output_path"])

    configure_logging(level="INFO")
    set_global_seed(random_seed)

    LOGGER.info("Generating synthetic cyber events.")
    dataset = generate_dataset(config)

    saved_path = save_dataset(dataset, output_path)

    label_counts = dataset["label"].value_counts().sort_index().to_dict()
    LOGGER.info("Synthetic dataset saved to %s", saved_path)
    LOGGER.info("Generated rows: %s", len(dataset))
    LOGGER.info("Label distribution: %s", label_counts)


if __name__ == "__main__":
    main()
