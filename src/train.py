"""
train.py — Training loop, callbacks, and learning-rate scheduling for the
Fashion-MNIST CNN classification project.
"""

import os
import math
import numpy as np
import tensorflow as tf

from config import (
    SEED,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    BEST_MODEL_PATH,
    RESULTS_DIR,
)

tf.random.set_seed(SEED)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(model_path: str = BEST_MODEL_PATH, patience: int = EARLY_STOPPING_PATIENCE):
    """Build and return standard training callbacks.

    Includes:
        - EarlyStopping: stops when val_loss fails to improve for `patience` epochs,
          restoring best weights automatically.
        - ModelCheckpoint: saves the best model (by val_loss) during training.
        - ReduceLROnPlateau: halves LR when val_loss stalls for 5 epochs.

    Args:
        model_path: Where to save the best model weights.
        patience: EarlyStopping patience (number of epochs without improvement).

    Returns:
        List of Keras callbacks.
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )
    return [early_stop, checkpoint, reduce_lr]


# ── Learning-rate schedules ───────────────────────────────────────────────────

def cosine_annealing_schedule(epoch: int, lr: float) -> float:
    """Cosine annealing LR schedule callback function.

    Args:
        epoch: Current epoch number (0-indexed).
        lr: Current learning rate (unused; computed from LEARNING_RATE).

    Returns:
        New learning rate as float.
    """
    min_lr = 1e-6
    return min_lr + 0.5 * (LEARNING_RATE - min_lr) * (
        1 + math.cos(math.pi * epoch / EPOCHS)
    )


def get_lr_scheduler():
    """Return a LearningRateScheduler callback using cosine annealing."""
    return tf.keras.callbacks.LearningRateScheduler(
        cosine_annealing_schedule, verbose=0
    )


# ── Mixed-precision helper (Task 29) ─────────────────────────────────────────

def enable_mixed_precision():
    """Enable float16 mixed-precision training for compatible GPUs.

    Must be called before model construction.
    """
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


# ── Main training function ────────────────────────────────────────────────────

def train(
    model,
    train_ds,
    val_ds,
    epochs: int = EPOCHS,
    callbacks=None,
    use_lr_scheduler: bool = False,
    log_dir: str = None,
):
    """Train a compiled Keras model.

    Args:
        model: A compiled Keras model.
        train_ds: Training tf.data.Dataset.
        val_ds: Validation tf.data.Dataset.
        epochs: Maximum number of epochs.
        callbacks: List of Keras callbacks (get_callbacks() used if None).
        use_lr_scheduler: If True, replaces ReduceLROnPlateau with cosine
            annealing LR scheduler.
        log_dir: TensorBoard log directory (optional).

    Returns:
        Keras History object.
    """
    if callbacks is None:
        callbacks = get_callbacks()

    if use_lr_scheduler:
        callbacks = [
            c for c in callbacks
            if not isinstance(c, tf.keras.callbacks.ReduceLROnPlateau)
        ]
        callbacks.append(get_lr_scheduler())

    if log_dir:
        tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tb_cb)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ── Training report helper ────────────────────────────────────────────────────

def log_training_report(history, report_path: str = None):
    """Save key training metrics to a text file.

    Args:
        history: Keras History object returned by model.fit().
        report_path: File path to write the report. Defaults to
            RESULTS_DIR/training_report.txt.
    """
    if report_path is None:
        report_path = os.path.join(RESULTS_DIR, "training_report.txt")

    h = history.history
    best_epoch = int(np.argmin(h["val_loss"]))
    stopped_epoch = len(h["val_loss"])

    lines = [
        "=== Training Report ===",
        f"Stopped at epoch   : {stopped_epoch}",
        f"Best epoch         : {best_epoch + 1}",
        f"Best val_loss      : {h['val_loss'][best_epoch]:.4f}",
        f"Best val_accuracy  : {h['val_accuracy'][best_epoch]:.4f}",
        f"Final train_loss   : {h['loss'][-1]:.4f}",
        f"Final train_acc    : {h['accuracy'][-1]:.4f}",
    ]
    report_text = "\n".join(lines)
    print(report_text)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    return report_text

# [2025-10-20 9:14 AM] Task 43: Add mlflow.start_run() context in mlflow_train.py

# [2025-10-20 11:13 AM] Task 43: Log params: lr, batch_size, epochs, dropout, seed
