"""EEG preprocessing: zero-phase bandpass filtering and normalisation.

The synthetic generator already produces epoched, EOG-free trials, so the
preprocessing here focuses on the bandpass step that isolates the mu and beta
sensorimotor rhythms (the report's 7-30 Hz band) plus optional per-trial
standardisation for the deep-learning models.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from .config import PreprocessingConfig


def _butter_bandpass(
    low: float, high: float, sfreq: float, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return Butterworth bandpass coefficients for a given band.

    Frequencies are normalised to the Nyquist rate and clipped just inside
    ``(0, 1)`` so edge bands (e.g. an upper edge at exactly Nyquist) do not
    produce an invalid filter.
    """
    nyquist = 0.5 * sfreq
    low_n = max(low / nyquist, 1e-4)
    high_n = min(high / nyquist, 0.9999)
    if low_n >= high_n:
        raise ValueError(f"Invalid band after normalisation: ({low_n}, {high_n})")
    b, a = butter(order, [low_n, high_n], btype="band")
    return b, a


def bandpass_filter(
    X: np.ndarray, cfg: PreprocessingConfig, sfreq: float
) -> np.ndarray:
    """Apply a zero-phase bandpass filter to every trial and channel.

    Parameters
    ----------
    X : np.ndarray
        Trials of shape ``(n_trials, n_channels, n_samples)``.
    cfg : PreprocessingConfig
        Filter band and order.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Filtered trials, same shape and dtype as the input.
    """
    b, a = _butter_bandpass(cfg.l_freq, cfg.h_freq, sfreq, cfg.filter_order)
    # filtfilt operates along the last axis; our sample axis is already last.
    filtered = filtfilt(b, a, X, axis=-1)
    return filtered.astype(X.dtype, copy=False)


def standardize_per_trial(X: np.ndarray) -> np.ndarray:
    """Z-score each trial channel-wise (zero mean, unit variance over time).

    This stabilises deep-network training without leaking statistics across
    trials, since each trial is normalised using only its own samples.
    """
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return ((X - mean) / std).astype(X.dtype, copy=False)
