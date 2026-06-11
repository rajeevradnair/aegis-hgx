"""Random seed utilities for reproducible experiments."""

import os
import random

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def set_global_seed(seed: int) -> None:
    if seed < 0:
        raise ValueError("Seed must be non-negative.")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if np is not None:
        np.random.seed(seed)
