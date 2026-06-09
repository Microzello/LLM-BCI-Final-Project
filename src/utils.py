"""Small cross-cutting helpers: seeding, logging, and JSON IO."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches for reproducible runs.

    TensorFlow is seeded lazily so that importing this module does not force a
    TensorFlow import on classical-only runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except Exception:  # noqa: BLE001 - TF is optional for classical-only runs
        pass


def get_logger(name: str = "mi_eeg") -> logging.Logger:
    """Return a module logger configured once with a concise format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands NumPy scalars and arrays."""

    def default(self, obj: Any) -> Any:  # noqa: ANN401 - encoder contract
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(obj: Any, path: str | Path) -> None:
    """Write ``obj`` to ``path`` as pretty-printed, NumPy-aware JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, cls=_NumpyEncoder)
