"""Logging utilities for Aegis-HGX."""

import logging
from pathlib import Path


def configure_logging(level: str = "INFO", log_dir: str | Path | None = None) -> None:
    numeric_level = getattr(logging, level.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level}")

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path / "aegis_hgx.log", encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
