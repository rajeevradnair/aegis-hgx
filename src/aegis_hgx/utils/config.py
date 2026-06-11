"""Configuration loading utilities for Aegis-HGX."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProjectInfo(BaseModel):
    name: str
    description: str
    default_seed: int = 42


class ProjectPaths(BaseModel):
    data_raw: Path
    data_interim: Path
    data_processed: Path
    data_external: Path
    reports: Path
    artifacts: Path
    logs: Path


class LoggingConfig(BaseModel):
    level: str = "INFO"


class ProjectConfig(BaseModel):
    project: ProjectInfo
    paths: ProjectPaths
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a YAML mapping: {config_path}")

    return data


def load_project_config(path: str | Path = "configs/project.yaml") -> ProjectConfig:
    data = load_yaml(path)
    return ProjectConfig.model_validate(data)
