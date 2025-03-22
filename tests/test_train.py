"""
tests/test_train.py — Unit tests for training callbacks and helpers.
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tensorflow as tf
from train import get_callbacks, cosine_annealing_schedule, log_training_report
from config import LEARNING_RATE, EPOCHS


# ── get_callbacks ─────────────────────────────────────────────────────────────

def test_get_callbacks_returns_list():
    cbs = get_callbacks(model_path="models/test_model.keras")
    assert isinstance(cbs, list)
    assert len(cbs) >= 2


def test_callbacks_contain_early_stopping():
    cbs = get_callbacks(model_path="models/test_model.keras")
    types = [type(c) for c in cbs]
    assert tf.keras.callbacks.EarlyStopping in types


def test_callbacks_contain_checkpoint():
    cbs = get_callbacks(model_path="models/test_model.keras")
    types = [type(c) for c in cbs]
    assert tf.keras.callbacks.ModelCheckpoint in types


# ── cosine annealing ──────────────────────────────────────────────────────────

def test_cosine_schedule_decreases():
    lrs = [cosine_annealing_schedule(e, LEARNING_RATE) for e in range(EPOCHS)]
    assert lrs[0] > lrs[-1], "LR should decrease over epochs with cosine annealing"


def test_cosine_schedule_starts_near_max():
    lr0 = cosine_annealing_schedule(0, LEARNING_RATE)
    assert abs(lr0 - LEARNING_RATE) < 1e-5, "First epoch LR should equal LEARNING_RATE"


@pytest.mark.parametrize("epoch", [0, 10, 25, 49])
def test_cosine_schedule_in_valid_range(epoch):
    lr = cosine_annealing_schedule(epoch, LEARNING_RATE)
    assert 1e-6 <= lr <= LEARNING_RATE + 1e-7, f"LR out of valid range at epoch {epoch}"


# ── log_training_report ───────────────────────────────────────────────────────

def test_log_training_report_writes_file(tmp_path):
    mock_history = type("H", (), {"history": {
        "loss": [0.5, 0.4, 0.3],
        "val_loss": [0.6, 0.45, 0.35],
        "accuracy": [0.7, 0.8, 0.85],
        "val_accuracy": [0.65, 0.75, 0.80],
    }})()
    report_path = str(tmp_path / "report.txt")
    text = log_training_report(mock_history, report_path=report_path)
    assert os.path.exists(report_path)
    assert "Best val_loss" in text


# ── Early stopping integration test ──────────────────────────────────────────

def test_early_stopping_triggers():
    """Training on random noise should trigger early stopping well before max epochs."""
    from model import build_baseline, compile_model
    from data import build_dataset, to_one_hot
    from train import train

    x = np.random.rand(64, 28, 28, 1).astype(np.float32)
    y = to_one_hot(np.random.randint(0, 10, size=64))
    ds = build_dataset(x, y, batch_size=32, shuffle=False)

    model = compile_model(build_baseline())
    cbs = get_callbacks(model_path="models/test_early_stop.keras", patience=2)
    history = train(model, ds, ds, epochs=50, callbacks=cbs)

    assert len(history.history["loss"]) < 50, "Early stopping should have triggered"
