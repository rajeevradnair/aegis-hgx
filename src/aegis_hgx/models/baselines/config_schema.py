from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    input_path: str
    target_column: str
    numeric_features: list[str]
    categorical_features: list[str]


class SplitConfig(BaseModel):
    test_size: float = Field(gt=0.0, lt=1.0)
    random_seed: int


class ModelConfig(BaseModel):
    max_iter: int = Field(gt=0)
    class_weight: str | None = None


class OutputsConfig(BaseModel):
    metrics_path: str
    model_path: str


class ExperimentTrackingConfig(BaseModel):
    experiment_name: str
    uri: str
    artifact_root: str


class TrainingConfig(BaseModel):
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    outputs: OutputsConfig
    experiment_tracking: ExperimentTrackingConfig


def validate_training_config(config: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig.model_validate(config)