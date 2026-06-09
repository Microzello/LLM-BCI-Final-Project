"""Synthetic BCI Competition IV 2a dataset generator.

The real BCI-IV 2a recordings are not redistributed with this project. This
module instead synthesises EEG-like trials with the *same tensor shape* and a
physiologically motivated structure so the full classification pipeline can run
end to end.

Design goals
------------
1. Match the report's dimensions: 9 subjects, 22 EEG (+ 3 EOG) channels,
   250 Hz, 4-second trials, 4 classes, 792 total trials.
2. Encode class information the way real MI signals do: as *band-power
   modulation* (ERD/ERS) in the mu (8-13 Hz) and beta (14-30 Hz) bands over
   motor-cortex channels, lateralised by class.
3. Reproduce inter-subject variability ("BCI illiteracy"): some subjects have
   weak modulation and are intrinsically harder to classify.

The generator is fully deterministic given a seed, so synthetic data is
identical across runs and machines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DatasetConfig


# Channel-index groups over a 22-channel 10-20-style motor montage. The real
# 2a montage is denser, but for the synthetic generator we only need a
# left / central / right partition to lateralise the ERD/ERS pattern.
_LEFT_MOTOR = (7, 8, 9)        # ~C3 neighbourhood
_CENTRAL_MOTOR = (10, 11, 12)  # ~Cz neighbourhood
_RIGHT_MOTOR = (13, 14, 15)    # ~C4 neighbourhood


@dataclass
class EEGDataset:
    """Container for a generated dataset.

    Attributes
    ----------
    X : np.ndarray
        EEG trials, shape ``(n_trials, n_channels, n_samples)``.
    y : np.ndarray
        Integer class labels in ``[0, n_classes)``, shape ``(n_trials,)``.
    subjects : np.ndarray
        Subject id per trial in ``[0, n_subjects)``, shape ``(n_trials,)``.
    sfreq : float
        Sampling frequency in Hz.
    class_names : tuple[str, ...]
        Human-readable class names ordered by label index.
    """

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    sfreq: float
    class_names: tuple[str, ...]

    @property
    def n_trials(self) -> int:
        return self.X.shape[0]

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    @property
    def n_samples(self) -> int:
        return self.X.shape[2]


def _subject_strengths(cfg: DatasetConfig, rng: np.random.Generator) -> np.ndarray:
    """Return a per-subject modulation-strength multiplier.

    If the config pins explicit values we use them; otherwise we lay out a
    deterministic spread between 0.45 and 1.0 so that exactly one subject is a
    clear "BCI illiteracy" case and the rest range from moderate to strong.
    """
    if cfg.subject_strength is not None:
        strengths = np.asarray(cfg.subject_strength, dtype=float)
        if strengths.shape[0] != cfg.n_subjects:
            raise ValueError(
                "subject_strength length must equal n_subjects "
                f"({strengths.shape[0]} != {cfg.n_subjects})"
            )
        return strengths
    # Evenly spaced strengths, then a tiny jitter so subjects are not perfectly
    # ordered. The lowest value reproduces the weak-modulation subject.
    base = np.linspace(0.5, 1.0, cfg.n_subjects)
    jitter = rng.uniform(-0.03, 0.03, size=cfg.n_subjects)
    return np.clip(base + jitter, 0.4, 1.0)


# Per-class mu/beta power gains over the (left, central, right) motor regions.
# Values below 1.0 model ERD (power suppression on the active contralateral
# side); values above 1.0 model ERS (power increase / idling). The four classes
# use distinct lateralisation signatures so they are separable in band power.
_CLASS_GAIN_TABLE: dict[int, dict[str, list[float]]] = {
    0: {"mu": [1.15, 1.00, 0.55], "beta": [1.10, 1.00, 0.65]},  # left hand
    1: {"mu": [0.55, 1.00, 1.15], "beta": [0.65, 1.00, 1.10]},  # right hand
    2: {"mu": [1.10, 0.55, 1.10], "beta": [1.05, 0.65, 1.05]},  # feet
    3: {"mu": [0.80, 0.70, 0.80], "beta": [1.15, 1.20, 1.15]},  # tongue
}


def _class_band_gains(n_classes: int) -> dict[int, dict[str, np.ndarray]]:
    """Return the mu/beta regional power gains for each of ``n_classes`` classes.

    Each class maps to two 3-vectors (``mu`` and ``beta``) of multiplicative
    power gains applied to the left, central, and right motor regions. Only the
    first ``n_classes`` entries of the gain table are returned.
    """
    if not 1 <= n_classes <= len(_CLASS_GAIN_TABLE):
        raise ValueError(
            f"n_classes must be in [1, {len(_CLASS_GAIN_TABLE)}], got {n_classes}"
        )
    return {
        c: {band: np.array(gain) for band, gain in _CLASS_GAIN_TABLE[c].items()}
        for c in range(n_classes)
    }


def _bandlimited_noise(
    n_channels: int,
    n_samples: int,
    sfreq: float,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate band-limited Gaussian noise via an FFT mask.

    Returns an array of shape ``(n_channels, n_samples)`` whose power is
    concentrated in ``[low, high]`` Hz. Used as the oscillatory building block
    for each frequency band so we can scale band power per region.
    """
    white = rng.standard_normal((n_channels, n_samples))
    spectrum = np.fft.rfft(white, axis=1)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    mask = (freqs >= low) & (freqs <= high)
    spectrum *= mask[np.newaxis, :]
    band = np.fft.irfft(spectrum, n=n_samples, axis=1)
    # Normalise to unit per-channel std so downstream gains are interpretable.
    std = band.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return band / std


def _apply_region_gain(
    signal: np.ndarray,
    region_gain: np.ndarray,
) -> np.ndarray:
    """Scale channels by region according to a (left, central, right) gain.

    Channels outside the three motor regions receive a gain of 1.0 (unchanged).
    """
    out = signal.copy()
    for region_idx, channels in enumerate(
        (_LEFT_MOTOR, _CENTRAL_MOTOR, _RIGHT_MOTOR)
    ):
        for ch in channels:
            if ch < out.shape[0]:
                out[ch] *= region_gain[region_idx]
    return out


def _generate_trial(
    cfg: DatasetConfig,
    label: int,
    strength: float,
    band_gains: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesise a single EEG trial of shape ``(n_channels, n_samples)``.

    The trial is a sum of mu-band and beta-band oscillatory components whose
    regional power is modulated by the class, plus broadband background noise
    and a slow drift. ``strength`` interpolates the class modulation toward the
    neutral (no-information) case, which is how subject difficulty is encoded.
    """
    n_ch = cfg.n_channels
    n_s = cfg.n_samples
    sep = cfg.class_separability

    # Oscillatory carriers in the two physiological bands.
    mu = _bandlimited_noise(n_ch, n_s, cfg.sfreq, 8.0, 13.0, rng)
    beta = _bandlimited_noise(n_ch, n_s, cfg.sfreq, 14.0, 30.0, rng)

    # Interpolate each region gain between 1.0 (neutral) and the class gain,
    # scaled by both global separability and per-subject strength. Raising the
    # exponent sharpens the contrast for stronger subjects.
    eff = np.clip(sep * strength, 0.0, None)
    mu_gain = 1.0 + (band_gains["mu"] - 1.0) * eff
    beta_gain = 1.0 + (band_gains["beta"] - 1.0) * eff

    mu = _apply_region_gain(mu, mu_gain)
    beta = _apply_region_gain(beta, beta_gain)

    # Broadband background and a slow drift common to scalp EEG.
    background = cfg.noise_std * rng.standard_normal((n_ch, n_s))
    t = np.arange(n_s) / cfg.sfreq
    drift = (0.3 * rng.standard_normal((n_ch, 1))) * np.sin(
        2 * np.pi * rng.uniform(0.1, 0.5) * t
    )[np.newaxis, :]

    trial = mu + beta + background + drift
    return trial.astype(np.float32)


def generate_dataset(cfg: DatasetConfig, seed: int) -> EEGDataset:
    """Generate the full synthetic dataset described by ``cfg``.

    Trials are balanced across the four classes within each subject as evenly
    as the per-subject trial budget allows, then concatenated across subjects.
    """
    rng = np.random.default_rng(seed)
    strengths = _subject_strengths(cfg, rng)
    gains = _class_band_gains(cfg.n_classes)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    subj_list: list[int] = []

    for subj in range(cfg.n_subjects):
        # Balanced label vector for this subject, shuffled deterministically.
        per_class = cfg.trials_per_subject // cfg.n_classes
        remainder = cfg.trials_per_subject - per_class * cfg.n_classes
        labels = np.repeat(np.arange(cfg.n_classes), per_class)
        if remainder:
            labels = np.concatenate([labels, np.arange(remainder)])
        rng.shuffle(labels)

        for label in labels:
            trial = _generate_trial(
                cfg, int(label), float(strengths[subj]), gains[int(label)], rng
            )
            X_list.append(trial)
            y_list.append(int(label))
            subj_list.append(subj)

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    subjects = np.asarray(subj_list, dtype=np.int64)

    return EEGDataset(
        X=X,
        y=y,
        subjects=subjects,
        sfreq=cfg.sfreq,
        class_names=cfg.class_names,
    )
