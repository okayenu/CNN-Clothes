"""
evaluate.py — Evaluation, metrics, error analysis, and visualization for
Fashion-MNIST CNN classification.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve

from config import CLASS_NAMES, RESULTS_DIR, NUM_CLASSES


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path: str = None, normalize: bool = True):
    """Plot and optionally save a confusion matrix.

    Args:
        y_true: 1-D array of true integer class labels.
        y_pred: 1-D array of predicted integer class labels.
        save_path: File path to save the figure; shown interactively if None.
        normalize: If True, display percentages instead of raw counts.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(history, save_path: str = None):
    """Plot accuracy and loss curves from Keras History.

    Args:
        history: Keras History object or dict with keys
            'accuracy', 'val_accuracy', 'loss', 'val_loss'.
        save_path: Path to save the figure.
    """
    h = history.history if hasattr(history, "history") else history
    epochs = range(1, len(h["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, h["accuracy"], label="Train acc")
    ax1.plot(epochs, h["val_accuracy"], label="Val acc")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, h["loss"], label="Train loss")
    ax2.plot(epochs, h["val_loss"], label="Val loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


# ── ROC-AUC per class (Task 33) ───────────────────────────────────────────────

def compute_roc_auc(y_true_oh, y_prob, save_path: str = None):
    """Compute one-vs-rest ROC-AUC for each class and plot curves.

    Args:
        y_true_oh: One-hot encoded true labels [N, num_classes].
        y_prob: Softmax probabilities [N, num_classes].
        save_path: Path to save the figure.

    Returns:
        Dict mapping class name to AUC score.
    """
    from sklearn.metrics import roc_curve, auc

    aucs = {}
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
        auc_val = auc(fpr, tpr)
        aucs[name] = auc_val
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

    return aucs


# ── Calibration analysis (Task 34) ───────────────────────────────────────────

def plot_calibration(y_true_oh, y_prob, save_path: str = None):
    """Reliability diagram for each class.

    Args:
        y_true_oh: One-hot encoded true labels [N, num_classes].
        y_prob: Predicted probabilities [N, num_classes].
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(CLASS_NAMES):
        prob_true, prob_pred = calibration_curve(
            y_true_oh[:, i], y_prob[:, i], n_bins=10
        )
        ax.plot(prob_pred, prob_true, marker=".", label=name)

    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


# ── Error analysis (Task 32) ──────────────────────────────────────────────────

def error_analysis(x_test, y_true, y_pred, y_prob, n: int = 50, save_path: str = None):
    """Find and display the n most confidently wrong predictions.

    Args:
        x_test: Test images [N, H, W, 1].
        y_true: True integer labels [N].
        y_pred: Predicted integer labels [N].
        y_prob: Softmax probabilities [N, num_classes].
        n: Number of errors to inspect.
        save_path: Path to save a grid figure.

    Returns:
        List of (index, true_label, pred_label, confidence) tuples.
    """
    wrong_mask = y_true != y_pred
    wrong_idx = np.where(wrong_mask)[0]
    confidences = y_prob[wrong_idx, y_pred[wrong_idx]]
    top_n_idx = wrong_idx[np.argsort(confidences)[::-1][:n]]

    errors = [
        (int(i), CLASS_NAMES[y_true[i]], CLASS_NAMES[y_pred[i]], float(y_prob[i, y_pred[i]]))
        for i in top_n_idx
    ]

    if save_path:
        cols = 10
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 2))
        for ax_i, (idx, true_name, pred_name, conf) in enumerate(errors):
            r, c = divmod(ax_i, cols)
            ax = axes[r, c] if rows > 1 else axes[c]
            ax.imshow(x_test[idx].squeeze(), cmap="gray")
            ax.set_title(f"T:{true_name[:4]}\nP:{pred_name[:4]}\n{conf:.2f}", fontsize=6)
            ax.axis("off")
        for ax_i in range(len(errors), rows * cols):
            r, c = divmod(ax_i, cols)
            (axes[r, c] if rows > 1 else axes[c]).axis("off")
        plt.suptitle("Top Confident Errors", fontsize=12)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
        plt.close()

    return errors


# ── Inference latency benchmark (Task 35) ────────────────────────────────────

def benchmark_latency(model, x_sample, n_runs: int = 100):
    """Measure average per-image inference latency.

    Args:
        model: Compiled Keras model.
        x_sample: Array of images to run inference on.
        n_runs: Number of repeat passes for averaging.

    Returns:
        Dict with 'mean_ms', 'std_ms', 'total_images'.
    """
    _ = model.predict(x_sample[:1], verbose=0)  # warm up

    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(x_sample, verbose=0)
    elapsed = (time.perf_counter() - start) / n_runs / len(x_sample) * 1000

    return {
        "mean_ms": round(elapsed, 4),
        "n_runs": n_runs,
        "total_images": len(x_sample),
    }


# ── Classification report wrapper ────────────────────────────────────────────

def full_report(y_true, y_pred) -> str:
    """Return sklearn classification report as a string."""
    return classification_report(y_true, y_pred, target_names=CLASS_NAMES)


import math  # noqa: E402 (needed by error_analysis)
