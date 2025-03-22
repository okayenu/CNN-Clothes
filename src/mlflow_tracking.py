"""
mlflow_tracking.py — MLflow experiment tracking integration (Task 43 — RISKY).

Wraps training with automatic logging of hyperparameters, metrics, and model
artifacts to a local MLflow tracking server.
"""

import os
import numpy as np

from config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    SEED,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    DROPOUT_RATE,
    L2_RATE,
    EARLY_STOPPING_PATIENCE,
)


def start_run(run_name: str = None, extra_params: dict = None):
    """Initialize an MLflow run and log standard hyperparameters.

    Args:
        run_name: Optional human-readable run name.
        extra_params: Additional params dict to log.

    Returns:
        The active MLflow run object.
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError("Install mlflow: pip install mlflow")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    run = mlflow.start_run(run_name=run_name)

    params = {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dropout_rate": DROPOUT_RATE,
        "l2_rate": L2_RATE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    }
    if extra_params:
        params.update(extra_params)
    mlflow.log_params(params)
    return run


def log_history(history):
    """Log all epoch-level metrics from a Keras History object to MLflow.

    Args:
        history: Keras History object returned by model.fit().
    """
    try:
        import mlflow
    except ImportError:
        return

    h = history.history if hasattr(history, "history") else history
    for epoch_idx in range(len(h["loss"])):
        mlflow.log_metrics(
            {k: float(v[epoch_idx]) for k, v in h.items()},
            step=epoch_idx,
        )


def log_model(model, artifact_path: str = "model"):
    """Save and log a Keras model as an MLflow artifact.

    Args:
        model: Trained Keras model.
        artifact_path: Sub-directory name within the run's artifact store.
    """
    try:
        import mlflow.keras
    except ImportError:
        return
    mlflow.keras.log_model(model, artifact_path)


def end_run():
    """End the current MLflow run."""
    try:
        import mlflow
        mlflow.end_run()
    except ImportError:
        pass
