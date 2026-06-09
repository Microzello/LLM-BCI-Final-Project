"""Main entry point: run the full Motor Imagery classification benchmark.

Pipeline
--------
1. Generate the synthetic BCI-IV 2a dataset.
2. Bandpass filter all trials (7-30 Hz).
3. Evaluate classical models (FBCSP+SVM, FBCSP+KNN, FBCSP+AdaBoost) with
   5-fold stratified CV, re-fitting FBCSP inside each fold.
4. Evaluate deep models (EEGNet, varEEGNet, Fusion 3CNNs) with the same folds.
5. Run per-subject FBCSP+SVM analysis (3-fold CV per subject).
6. Write metrics JSON, a text summary, and all figures to ``outputs/``.

Usage
-----
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --no-deep   # classical only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.config import Config, load_config
from src.dataset import EEGDataset, generate_dataset
from src.evaluation import (
    ModelResult,
    cross_validate_features,
    make_folds,
    score_predictions,
)
from src.fbcsp import FBCSP
from src.preprocessing import bandpass_filter, standardize_per_trial
from src.utils import ensure_dir, get_logger, save_json, set_global_seed

logger = get_logger()


def _build_fbcsp_callables(cfg: Config):
    """Return fit_transform / transform closures bound to a fresh FBCSP.

    A new FBCSP instance is created per call to ``fit_transform`` so each CV
    fold learns its own spatial filters. The same instance is then reused by
    ``transform`` via a mutable holder.
    """
    holder: dict[str, FBCSP] = {}

    def fit_transform(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        extractor = FBCSP(cfg.fbcsp, cfg.dataset.sfreq, cfg.dataset.n_classes)
        feats = extractor.fit_transform(X, y)
        holder["extractor"] = extractor
        return feats

    def transform(X: np.ndarray) -> np.ndarray:
        return holder["extractor"].transform(X)

    return fit_transform, transform


def run_classical_models(
    cfg: Config, X: np.ndarray, y: np.ndarray, folds
) -> list[ModelResult]:
    """Evaluate FBCSP feature models: SVM, KNN, AdaBoost."""
    results: list[ModelResult] = []

    classical_specs = [
        (
            "FBCSP+SVM",
            lambda: SVC(
                C=cfg.svm.C, gamma=cfg.svm.gamma, kernel=cfg.svm.kernel
            ),
            "RBF-kernel SVM on FBCSP log-variance features",
        ),
        (
            "FBCSP+KNN",
            lambda: KNeighborsClassifier(n_neighbors=7),
            "k-NN baseline on FBCSP features",
        ),
        (
            "FBCSP+AdaBoost",
            lambda: AdaBoostClassifier(n_estimators=50, random_state=cfg.seed),
            "AdaBoost (weak learners) on FBCSP features",
        ),
    ]

    for name, factory, notes in classical_specs:
        logger.info("Evaluating %s ...", name)
        fit_transform, transform = _build_fbcsp_callables(cfg)
        result = cross_validate_features(
            name=name,
            X=X,
            y=y,
            feature_fit_transform=fit_transform,
            feature_transform=transform,
            classifier_factory=factory,
            folds=folds,
        )
        result.notes = notes
        logger.info(
            "  %s  mean_acc=%.3f  mean_kappa=%.3f",
            name,
            result.mean_accuracy,
            result.mean_kappa,
        )
        results.append(result)

    return results


def run_deep_models(
    cfg: Config, X: np.ndarray, y: np.ndarray, folds
) -> list[ModelResult]:
    """Evaluate EEGNet, varEEGNet, and Fusion 3CNNs.

    Deep imports are local so a ``--no-deep`` run never imports TensorFlow.
    """
    from src.models import build_eegnet, build_fusion_3cnn, build_vareegnet
    from src.trainer import cross_validate_deep

    # Deep nets train on per-trial standardised signals.
    X_std = standardize_per_trial(X)
    n_ch, n_s = cfg.dataset.n_channels, cfg.dataset.n_samples

    results: list[ModelResult] = []

    deep_specs = [
        (
            "EEGNet",
            lambda: build_eegnet(n_ch, n_s, cfg.dataset.n_classes, cfg.eegnet),
            cfg.eegnet.epochs,
            cfg.eegnet.batch_size,
            cfg.eegnet.learning_rate,
            "Standard EEGNet-8,2 baseline",
        ),
        (
            "varEEGNet",
            lambda: build_vareegnet(n_ch, n_s, cfg.dataset.n_classes, cfg.eegnet),
            cfg.eegnet.epochs,
            cfg.eegnet.batch_size,
            cfg.eegnet.learning_rate,
            "EEGNet with log-variance compression (compact)",
        ),
        (
            "Fusion3CNNs",
            lambda: build_fusion_3cnn(n_ch, n_s, cfg.dataset.n_classes, cfg.fusion),
            cfg.fusion.epochs,
            cfg.fusion.batch_size,
            cfg.fusion.learning_rate,
            "Three parallel multi-scale CNN branches",
        ),
    ]

    for name, factory, epochs, batch, lr, notes in deep_specs:
        logger.info("Evaluating %s ...", name)
        result = cross_validate_deep(
            name=name,
            X=X_std,
            y=y,
            model_factory=factory,
            folds=folds,
            epochs=epochs,
            batch_size=batch,
            learning_rate=lr,
            seed=cfg.seed,
        )
        result.notes = notes
        logger.info(
            "  %s  mean_acc=%.3f  mean_kappa=%.3f  params=%s",
            name,
            result.mean_accuracy,
            result.mean_kappa,
            result.n_params,
        )
        results.append(result)

    return results


def run_per_subject_fbcsp_svm(
    cfg: Config, data: EEGDataset, X: np.ndarray
) -> dict[int, float]:
    """Subject-specific FBCSP+SVM accuracy via per-subject k-fold CV.

    For each subject we run stratified CV using only that subject's trials,
    re-fitting FBCSP within each fold, and report the mean held-out accuracy.
    """
    subject_acc: dict[int, float] = {}
    n_folds = cfg.cv.subject_specific_folds

    for subj in range(cfg.dataset.n_subjects):
        mask = data.subjects == subj
        Xs, ys = X[mask], data.y[mask]
        folds = make_folds(ys, n_folds, cfg.seed)

        accs: list[float] = []
        for train_idx, test_idx in folds:
            extractor = FBCSP(cfg.fbcsp, cfg.dataset.sfreq, cfg.dataset.n_classes)
            feats_tr = extractor.fit_transform(Xs[train_idx], ys[train_idx])
            feats_te = extractor.transform(Xs[test_idx])
            clf = SVC(C=cfg.svm.C, gamma=cfg.svm.gamma, kernel=cfg.svm.kernel)
            clf.fit(feats_tr, ys[train_idx])
            preds = clf.predict(feats_te)
            accs.append(score_predictions(ys[test_idx], preds).accuracy)

        subject_acc[subj] = float(np.mean(accs))
        logger.info("  Subject S%d  acc=%.3f", subj + 1, subject_acc[subj])

    return subject_acc


def write_summary(
    cfg: Config,
    results: list[ModelResult],
    subject_acc: dict[int, float],
    out_dir: Path,
) -> Path:
    """Write a human-readable text summary of the benchmark."""
    lines: list[str] = []
    lines.append("Motor Imagery EEG Classification — Benchmark Summary (Group OLAF)")
    lines.append("=" * 68)
    lines.append("")
    lines.append(
        f"Dataset: synthetic BCI-IV 2a | {cfg.dataset.total_trials} trials | "
        f"{cfg.dataset.n_channels} ch | {cfg.dataset.n_classes} classes | "
        f"{cfg.dataset.n_subjects} subjects"
    )
    lines.append(
        f"Cross-validation: {cfg.cv.intra_session_folds}-fold (pooled) + "
        f"{cfg.cv.subject_specific_folds}-fold per-subject"
    )
    lines.append("")
    lines.append("Model comparison (mean accuracy / kappa over folds):")
    lines.append("-" * 68)
    lines.append(f"{'Model':<18}{'Accuracy':>12}{'Std':>10}{'Kappa':>10}{'Params':>12}")
    for r in sorted(results, key=lambda x: x.mean_accuracy, reverse=True):
        params = "N/A" if r.n_params is None else str(r.n_params)
        lines.append(
            f"{r.name:<18}{r.mean_accuracy * 100:>11.1f}%"
            f"{r.std_accuracy * 100:>9.1f}%{r.mean_kappa:>10.3f}{params:>12}"
        )
    lines.append("")
    lines.append("Per-subject FBCSP+SVM accuracy (subject-specific CV):")
    lines.append("-" * 68)
    for subj in sorted(subject_acc):
        lines.append(f"  S{subj + 1}: {subject_acc[subj] * 100:5.1f}%")
    weakest = min(subject_acc, key=subject_acc.get)
    lines.append(
        f"\nWeakest subject (BCI-illiteracy candidate): "
        f"S{weakest + 1} at {subject_acc[weakest] * 100:.1f}%"
    )
    lines.append("")
    lines.append("Note: high in-distribution accuracy reflects the synthetic, well-")
    lines.append("separated dataset and does not represent real BCI-IV 2a performance.")

    text = "\n".join(lines)
    path = out_dir / "summary.txt"
    path.write_text(text, encoding="utf-8")
    print("\n" + text + "\n")
    return path


def generate_all_figures(
    cfg: Config,
    results: list[ModelResult],
    subject_acc: dict[int, float],
    out_dir: Path,
) -> None:
    """Render every figure for the report."""
    from src import plots

    fig_dir = plots.figures_dir(out_dir)

    plots.plot_model_accuracy(results, fig_dir)
    plots.plot_per_fold_accuracy(results, fig_dir)
    plots.plot_per_subject_accuracy(subject_acc, fig_dir)
    plots.plot_kappa(results, fig_dir)
    plots.plot_param_efficiency(results, fig_dir)

    for r in results:
        if r.history:
            plots.plot_training_curves(r.history, r.name, fig_dir)

    logger.info("Figures written to %s", fig_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="MI-EEG classification benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--no-deep",
        action="store_true",
        help="Skip the deep-learning models (classical only, no TensorFlow).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_deep:
        cfg.run_deep_models = False

    set_global_seed(cfg.seed)
    out_dir = ensure_dir(cfg.output_dir)

    logger.info("Generating synthetic BCI-IV 2a dataset ...")
    data = generate_dataset(cfg.dataset, cfg.seed)
    logger.info(
        "  X shape=%s  y shape=%s  subjects=%d",
        data.X.shape,
        data.y.shape,
        cfg.dataset.n_subjects,
    )

    logger.info("Bandpass filtering (%.0f-%.0f Hz) ...", cfg.preprocessing.l_freq, cfg.preprocessing.h_freq)
    X = bandpass_filter(data.X, cfg.preprocessing, cfg.dataset.sfreq)

    folds = make_folds(data.y, cfg.cv.intra_session_folds, cfg.seed)

    results: list[ModelResult] = []
    results.extend(run_classical_models(cfg, X, data.y, folds))
    if cfg.run_deep_models:
        results.extend(run_deep_models(cfg, X, data.y, folds))
    else:
        logger.info("Skipping deep models (--no-deep).")

    logger.info("Per-subject FBCSP+SVM analysis ...")
    subject_acc = run_per_subject_fbcsp_svm(cfg, data, X)

    # Persist metrics.
    metrics = {
        "config": cfg.to_dict(),
        "models": [r.to_dict() for r in results],
        "per_subject_fbcsp_svm": {f"S{s + 1}": acc for s, acc in subject_acc.items()},
    }
    save_json(metrics, out_dir / "metrics.json")
    logger.info("Metrics written to %s", out_dir / "metrics.json")

    generate_all_figures(cfg, results, subject_acc, out_dir)
    write_summary(cfg, results, subject_acc, out_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
