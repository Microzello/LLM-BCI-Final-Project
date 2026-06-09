"""Central experiment configuration.

A single source of truth for every tunable parameter in the pipeline.
The dataclasses below mirror the structure of ``configs/default.yaml`` so a
config file can be loaded, partially overridden, and passed around as a typed
object instead of a loose dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    """Shape and statistics of the synthetic BCI-IV 2a dataset.

    Defaults reproduce the dimensions stated in the project report:
    9 subjects, 22 EEG channels, 250 Hz, 4-second trials, 4 classes.
    """

    n_subjects: int = 9
    trials_per_subject: int = 88  # 9 * 88 = 792 total, matching the report
    n_channels: int = 22
    n_eog_channels: int = 3
    sfreq: float = 250.0
    trial_seconds: float = 4.0
    n_classes: int = 4
    class_names: tuple[str, ...] = ("left_hand", "right_hand", "feet", "tongue")

    # Synthetic signal parameters. ``class_separability`` scales the amount of
    # class-dependent band-power modulation; higher means easier separation.
    noise_std: float = 1.0
    class_separability: float = 1.0
    # Per-subject modulation-strength multipliers create the "BCI illiteracy"
    # effect (some subjects are far harder than others). Length must equal
    # ``n_subjects``; if None, a deterministic spread is generated.
    subject_strength: tuple[float, ...] | None = None

    @property
    def n_samples(self) -> int:
        return int(round(self.sfreq * self.trial_seconds))

    @property
    def total_trials(self) -> int:
        return self.n_subjects * self.trials_per_subject


@dataclass
class PreprocessingConfig:
    """Bandpass filtering and epoching settings."""

    l_freq: float = 7.0
    h_freq: float = 30.0
    filter_order: int = 5  # Butterworth order per direction (filtfilt doubles it)


@dataclass
class FBCSPConfig:
    """Filter Bank CSP feature-extraction settings."""

    # Sub-band edges in Hz. Four bands, as recommended in the report.
    bands: tuple[tuple[float, float], ...] = (
        (8.0, 12.0),
        (12.0, 16.0),
        (16.0, 22.0),
        (22.0, 30.0),
    )
    n_csp_components: int = 4  # spatial filters retained per band (m pairs)
    filter_order: int = 4


@dataclass
class SVMConfig:
    C: float = 1.0
    gamma: str = "scale"
    kernel: str = "rbf"


@dataclass
class EEGNetConfig:
    """Hyperparameters shared by EEGNet and varEEGNet.

    The defaults give the report's parameter counts:
    EEGNet ~3,444 params, varEEGNet ~1,524 params.
    """

    F1: int = 8         # temporal filters
    D: int = 2          # depth multiplier
    F2: int = 16        # pointwise filters (EEGNet only; = F1 * D)
    kern_length: int = 64
    dropout: float = 0.5
    epochs: int = 75
    batch_size: int = 32
    learning_rate: float = 3e-3


@dataclass
class FusionCNNConfig:
    branch_filters: tuple[int, ...] = (32, 32, 32)
    branch_kernels: tuple[int, ...] = (32, 64, 128)
    dense_units: int = 128
    dropout: float = 0.5
    epochs: int = 75
    batch_size: int = 32
    learning_rate: float = 3e-3


@dataclass
class CVConfig:
    intra_session_folds: int = 5   # 5-fold CV across the pooled dataset
    subject_specific_folds: int = 3  # 3-fold CV per subject


@dataclass
class Config:
    seed: int = 42
    output_dir: str = "outputs"
    run_deep_models: bool = True   # set False for a fast classical-only run

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    fbcsp: FBCSPConfig = field(default_factory=FBCSPConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)
    eegnet: EEGNetConfig = field(default_factory=EEGNetConfig)
    fusion: FusionCNNConfig = field(default_factory=FusionCNNConfig)
    cv: CVConfig = field(default_factory=CVConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dc_instance: Any, overrides: dict[str, Any]) -> Any:
    """Recursively apply a dict of overrides onto a dataclass instance.

    Only keys already present on the dataclass are applied; unknown keys raise,
    so a typo in the YAML fails loudly instead of being silently ignored.
    """
    from dataclasses import fields, is_dataclass

    field_map = {f.name: f for f in fields(dc_instance)}
    for key, value in overrides.items():
        if key not in field_map:
            raise KeyError(
                f"Unknown config key '{key}' for {type(dc_instance).__name__}"
            )
        current = getattr(dc_instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            # Preserve tuple typing for sequence fields loaded from YAML lists.
            setattr(dc_instance, key, _coerce_tuple(current, value))
        else:
            setattr(dc_instance, key, value)
    return dc_instance


def _coerce_tuple(current: tuple, value: list) -> tuple:
    """Convert a YAML list back into the nested-tuple shape of ``current``."""
    if current and isinstance(current[0], tuple):
        return tuple(tuple(item) for item in value)
    return tuple(value)


def load_config(path: str | Path | None) -> Config:
    """Load a :class:`Config`, optionally overlaying values from a YAML file."""
    cfg = Config()
    if path is None:
        return cfg
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _merge(cfg, raw)
