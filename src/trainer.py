"""Cross-validation training loop for the Keras deep-learning models.

Separated from :mod:`src.evaluation` because it pulls in TensorFlow, which we
only want imported on deep-model runs. Each fold builds a *fresh* model so no
weights leak between folds, fits with early stopping on a validation split
carved from the training fold, and records the held-out accuracy and kappa.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .evaluation import ModelResult, score_predictions
from .models import compile_model
from .utils import get_logger

# Test folds have different row counts, so predict() legitimately retraces per
# fold. The warning is benign here; silence it to keep logs readable.
tf.get_logger().setLevel("ERROR")

logger = get_logger()


def _to_nchw(X: np.ndarray) -> np.ndarray:
    """Add the trailing channel dimension Keras Conv2D expects.

    ``X`` of shape ``(n_trials, n_channels, n_samples)`` becomes
    ``(n_trials, n_channels, n_samples, 1)``.
    """
    return X[..., np.newaxis].astype(np.float32)


def cross_validate_deep(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], tf.keras.Model],
    folds: list[tuple[np.ndarray, np.ndarray]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    val_fraction: float = 0.2,
) -> ModelResult:
    """Run k-fold CV for a deep model, returning aggregated metrics.

    The training history of the final fold is retained on the result so the
    caller can plot representative loss/accuracy curves.
    """
    result = ModelResult(name=name)
    X4 = _to_nchw(X)

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_tr_full, X_te = X4[train_idx], X4[test_idx]
        y_tr_full, y_te = y[train_idx], y[test_idx]

        # Carve a stratified validation split from the training fold for early
        # stopping; this never touches the held-out test indices.
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_full,
            y_tr_full,
            test_size=val_fraction,
            stratify=y_tr_full,
            random_state=seed + fold_idx,
        )

        tf.keras.backend.clear_session()
        model = compile_model(model_factory(), learning_rate)
        if result.n_params is None:
            result.n_params = int(model.count_params())

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
        )
        # Halve the learning rate when validation loss plateaus. This rescues
        # folds where a high initial rate would otherwise stall near chance.
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-5,
        )

        history = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stop, reduce_lr],
        )

        probs = model.predict(X_te, verbose=0)
        preds = probs.argmax(axis=1)
        fold = score_predictions(y_te, preds)
        result.fold_accuracies.append(fold.accuracy)
        result.fold_kappas.append(fold.kappa)
        logger.info(
            "  %s fold %d/%d  acc=%.3f  kappa=%.3f",
            name,
            fold_idx + 1,
            len(folds),
            fold.accuracy,
            fold.kappa,
        )

        # Keep the last fold's curves for plotting.
        result.history = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    return result
