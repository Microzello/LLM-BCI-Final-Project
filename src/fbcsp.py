"""Filter Bank Common Spatial Patterns (FBCSP) feature extraction.

FBCSP splits the EEG spectrum into several sub-bands, learns Common Spatial
Pattern (CSP) spatial filters in each band, and stacks the resulting
log-variance features. For the 4-class problem we use a one-vs-rest CSP scheme
in every band (one CSP model per class), which is the standard multiclass
extension and matches the report's description.

The extractor follows the scikit-learn ``fit`` / ``transform`` convention so it
slots directly into cross-validation loops without leaking test data into the
spatial-filter estimation.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt

from .config import FBCSPConfig


def _bandpass_coeffs(
    low: float, high: float, sfreq: float, order: int
) -> tuple[np.ndarray, np.ndarray]:
    nyquist = 0.5 * sfreq
    low_n = max(low / nyquist, 1e-4)
    high_n = min(high / nyquist, 0.9999)
    return butter(order, [low_n, high_n], btype="band")


def _covariance(trial: np.ndarray) -> np.ndarray:
    """Normalised spatial covariance of one trial.

    ``trial`` has shape ``(n_channels, n_samples)``. The covariance is
    trace-normalised so amplitude differences between trials do not dominate the
    CSP solution.
    """
    cov = trial @ trial.T
    trace = np.trace(cov)
    if trace <= 0:
        return cov
    return cov / trace


def _csp_filters(
    class_trials: np.ndarray, other_trials: np.ndarray, n_components: int
) -> np.ndarray:
    """Solve the two-class CSP generalised eigenproblem.

    Returns the ``n_components`` spatial filters that best separate the two
    classes, taken from both ends of the eigenvalue spectrum (the largest and
    smallest eigenvalues correspond to maximal variance for one class and the
    other respectively).
    """
    cov_a = np.mean([_covariance(t) for t in class_trials], axis=0)
    cov_b = np.mean([_covariance(t) for t in other_trials], axis=0)

    # Generalised eigendecomposition: cov_a w = lambda (cov_a + cov_b) w.
    eigvals, eigvecs = eigh(cov_a, cov_a + cov_b)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    m = n_components // 2
    # Take m filters from each extreme of the spectrum.
    selected = np.concatenate([eigvecs[:, :m], eigvecs[:, -m:]], axis=1)
    return selected.T  # shape (n_components, n_channels)


class FBCSP:
    """Filter Bank CSP feature extractor with one-vs-rest multiclass CSP."""

    def __init__(self, cfg: FBCSPConfig, sfreq: float, n_classes: int):
        self.cfg = cfg
        self.sfreq = sfreq
        self.n_classes = n_classes
        # Per-band Butterworth coefficients, computed once.
        self._coeffs = [
            _bandpass_coeffs(low, high, sfreq, cfg.filter_order)
            for (low, high) in cfg.bands
        ]
        # Learned spatial filters: filters_[band_idx][class_idx] -> array.
        self.filters_: list[list[np.ndarray]] = []

    def _filter_band(self, X: np.ndarray, band_idx: int) -> np.ndarray:
        b, a = self._coeffs[band_idx]
        return filtfilt(b, a, X, axis=-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FBCSP":
        """Learn one-vs-rest CSP filters for every sub-band.

        Parameters
        ----------
        X : np.ndarray
            Trials of shape ``(n_trials, n_channels, n_samples)``.
        y : np.ndarray
            Integer class labels of shape ``(n_trials,)``.
        """
        self.filters_ = []
        for band_idx in range(len(self.cfg.bands)):
            band_X = self._filter_band(X, band_idx)
            band_filters: list[np.ndarray] = []
            for cls in range(self.n_classes):
                class_mask = y == cls
                class_trials = band_X[class_mask]
                other_trials = band_X[~class_mask]
                band_filters.append(
                    _csp_filters(class_trials, other_trials, self.cfg.n_csp_components)
                )
            self.filters_.append(band_filters)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project trials through the learned filters into log-variance features.

        The feature dimension is
        ``n_bands * n_classes * n_csp_components``.
        """
        if not self.filters_:
            raise RuntimeError("FBCSP.transform called before fit.")

        features: list[np.ndarray] = []
        for band_idx in range(len(self.cfg.bands)):
            band_X = self._filter_band(X, band_idx)
            for cls in range(self.n_classes):
                W = self.filters_[band_idx][cls]  # (n_components, n_channels)
                # Apply spatial filters: projected has shape
                # (n_trials, n_components, n_samples).
                projected = np.einsum("kc,ncs->nks", W, band_X)
                var = projected.var(axis=-1)
                # Log-variance with per-trial normalisation (standard CSP feature).
                var_sum = var.sum(axis=1, keepdims=True)
                var_sum = np.where(var_sum == 0, 1.0, var_sum)
                log_var = np.log(var / var_sum + 1e-8)
                features.append(log_var)

        return np.concatenate(features, axis=1).astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
