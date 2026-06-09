"""Evaluation metrics and cross-validation helpers.

Provides Cohen's kappa, accuracy, and confusion-matrix computation, plus
stratified cross-validation loops for both the classical (feature-based) and
deep-learning (raw-trial) classifiers. Keeping the CV plumbing here means the
spatial-filter / network fitting always happens *inside* each training fold,
which prevents data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


@dataclass
class FoldResult:
    """Metrics for one cross-validation fold."""

    accuracy: float
    kappa: float


@dataclass
class ModelResult:
    """Aggregated cross-validation metrics for one model."""

    name: str
    fold_accuracies: list[float] = field(default_factory=list)
    fold_kappas: list[float] = field(default_factory=list)
    n_params: int | None = None
    notes: str = ""
    # Optional curves captured from the last fit (deep models only).
    history: dict[str, list[float]] | None = None

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean(self.fold_accuracies)) if self.fold_accuracies else 0.0

    @property
    def std_accuracy(self) -> float:
        return float(np.std(self.fold_accuracies)) if self.fold_accuracies else 0.0

    @property
    def mean_kappa(self) -> float:
        return float(np.mean(self.fold_kappas)) if self.fold_kappas else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "mean_kappa": self.mean_kappa,
            "fold_accuracies": self.fold_accuracies,
            "fold_kappas": self.fold_kappas,
            "n_params": self.n_params,
            "notes": self.notes,
        }


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> FoldResult:
    """Compute accuracy and Cohen's kappa for one set of predictions."""
    return FoldResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        kappa=float(cohen_kappa_score(y_true, y_pred)),
    )


def make_folds(
    y: np.ndarray, n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return stratified train/test index pairs for ``n_folds`` CV.

    Stratification keeps the four MI classes balanced across folds, which
    matters for kappa to be meaningful.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros_like(y), y))


def cross_validate_features(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_fit_transform: Callable[[np.ndarray, np.ndarray], np.ndarray],
    feature_transform: Callable[[np.ndarray], np.ndarray],
    classifier_factory: Callable[[], object],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> ModelResult:
    """Run feature-based CV where features are re-fit on each training split.

    ``feature_fit_transform`` learns the feature extractor on the training fold
    and returns its features; ``feature_transform`` applies the *already fit*
    extractor to the test fold. ``classifier_factory`` returns a fresh
    scikit-learn-style estimator per fold.
    """
    result = ModelResult(name=name)
    for train_idx, test_idx in folds:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        feats_tr = feature_fit_transform(X_tr, y_tr)
        feats_te = feature_transform(X_te)

        clf = classifier_factory()
        clf.fit(feats_tr, y_tr)
        preds = clf.predict(feats_te)

        fold = score_predictions(y_te, preds)
        result.fold_accuracies.append(fold.accuracy)
        result.fold_kappas.append(fold.kappa)
    return result


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Confusion matrix with a fixed label set so shape is always square."""
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
