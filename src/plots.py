"""Figure generation: model comparison, per-fold, per-subject, training curves.

All figures are written to ``<output_dir>/figures`` as PNGs. Matplotlib is used
with the non-interactive Agg backend so plotting works in headless environments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .evaluation import ModelResult  # noqa: E402
from .utils import ensure_dir  # noqa: E402


def plot_model_accuracy(
    results: list[ModelResult], out_dir: Path
) -> Path:
    """Bar chart of mean accuracy +/- std across models."""
    names = [r.name for r in results]
    means = [r.mean_accuracy * 100 for r in results]
    stds = [r.std_accuracy * 100 for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, means, yerr=stds, capsize=4, color="#3b7dd8")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison (CV mean +/- std)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    path = out_dir / "model_accuracy_comparison.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_per_fold_accuracy(
    results: list[ModelResult], out_dir: Path
) -> Path:
    """Grouped bar chart of per-fold accuracy for every model."""
    fig, ax = plt.subplots(figsize=(10, 5))
    n_models = len(results)
    max_folds = max((len(r.fold_accuracies) for r in results), default=0)
    width = 0.8 / max(n_models, 1)

    for i, r in enumerate(results):
        accs = [a * 100 for a in r.fold_accuracies]
        positions = np.arange(len(accs)) + i * width
        ax.bar(positions, accs, width=width, label=r.name)

    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Fold Accuracy")
    ax.set_ylim(0, 105)
    ax.set_xticks(np.arange(max_folds) + 0.4 - width / 2)
    ax.set_xticklabels([f"Fold {i + 1}" for i in range(max_folds)])
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "per_fold_accuracy.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_per_subject_accuracy(
    subject_acc: dict[int, float], out_dir: Path
) -> Path:
    """Horizontal bar chart of per-subject accuracy (FBCSP+SVM)."""
    subjects = sorted(subject_acc)
    accs = [subject_acc[s] * 100 for s in subjects]
    labels = [f"S{s + 1}" for s in subjects]

    # Colour the weakest subject differently to highlight "BCI illiteracy".
    min_idx = int(np.argmin(accs))
    colors = ["#3b7dd8"] * len(accs)
    colors[min_idx] = "#d8543b"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels, accs, color=colors)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Subject Accuracy — FBCSP+SVM (subject-specific CV)")
    ax.set_xlim(min(accs) - 5 if accs else 0, 100)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    path = out_dir / "per_subject_accuracy.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_kappa(results: list[ModelResult], out_dir: Path) -> Path:
    """Bar chart of mean Cohen's kappa across models."""
    names = [r.name for r in results]
    kappas = [r.mean_kappa for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, kappas, color="#3b9d6e")
    ax.set_ylabel("Cohen's kappa")
    ax.set_title("Kappa Coefficient by Model")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    path = out_dir / "kappa_by_model.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_training_curves(
    history: dict[str, list[float]], model_name: str, out_dir: Path
) -> list[Path]:
    """Plot loss and accuracy curves for a deep model's training history."""
    paths: list[Path] = []

    if "loss" in history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history["loss"], label="train loss", color="#d88a3b")
        if "val_loss" in history:
            ax.plot(history["val_loss"], label="val loss", color="#3b7dd8")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{model_name} Training Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"{model_name.lower()}_loss.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    if "accuracy" in history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            [a * 100 for a in history["accuracy"]],
            label="train acc",
            color="#3b9d6e",
        )
        if "val_accuracy" in history:
            ax.plot(
                [a * 100 for a in history["val_accuracy"]],
                label="val acc",
                color="#7a5bd8",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{model_name} Training Accuracy")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"{model_name.lower()}_accuracy.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    return paths


def plot_param_efficiency(
    results: list[ModelResult], out_dir: Path
) -> Path:
    """Scatter of accuracy vs parameter count for models that have a count."""
    pts = [(r.n_params, r.mean_accuracy * 100, r.name) for r in results if r.n_params]
    fig, ax = plt.subplots(figsize=(8, 5))
    for params, acc, name in pts:
        ax.scatter(params, acc, s=90)
        ax.annotate(
            name, (params, acc), textcoords="offset points", xytext=(6, 4), fontsize=8
        )
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Parameter Efficiency vs Accuracy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "parameter_efficiency.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figures_dir(output_dir: str | Path) -> Path:
    """Return (and create) the figures sub-directory of ``output_dir``."""
    return ensure_dir(Path(output_dir) / "figures")
