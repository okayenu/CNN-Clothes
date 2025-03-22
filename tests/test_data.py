"""
tests/test_data.py — Unit tests for data loading and preprocessing pipeline.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import (
    normalize,
    standardize,
    to_one_hot,
    split_train_val,
    build_dataset,
)
from config import NUM_CLASSES, INPUT_SHAPE, SEED


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_images():
    rng = np.random.default_rng(SEED)
    return rng.integers(0, 256, size=(100, 28, 28, 1), dtype=np.float32)


@pytest.fixture
def dummy_labels():
    rng = np.random.default_rng(SEED)
    return rng.integers(0, NUM_CLASSES, size=(100,), dtype=np.int32)


# ── normalize ─────────────────────────────────────────────────────────────────

def test_normalize_range(dummy_images):
    normed = normalize(dummy_images)
    assert normed.max() <= 1.0, "Normalized images should not exceed 1.0"
    assert normed.min() >= 0.0, "Normalized images should not be below 0.0"


def test_normalize_shape(dummy_images):
    assert normalize(dummy_images).shape == dummy_images.shape


# ── standardize ───────────────────────────────────────────────────────────────

def test_standardize_mean_near_zero(dummy_images):
    imgs = normalize(dummy_images)
    standardized, mean, std = standardize(imgs)
    assert abs(standardized.mean()) < 0.1, "Standardized mean should be near 0"


def test_standardize_returns_same_stats_on_reuse(dummy_images):
    imgs = normalize(dummy_images)
    _, mean, std = standardize(imgs)
    result, _, _ = standardize(imgs, mean=mean, std=std)
    assert result.shape == imgs.shape


# ── to_one_hot ────────────────────────────────────────────────────────────────

def test_one_hot_shape(dummy_labels):
    oh = to_one_hot(dummy_labels)
    assert oh.shape == (100, NUM_CLASSES)


def test_one_hot_sum_to_one(dummy_labels):
    oh = to_one_hot(dummy_labels)
    sums = oh.sum(axis=1)
    assert np.allclose(sums, 1.0), "Each one-hot row should sum to 1"


# ── split_train_val ───────────────────────────────────────────────────────────

def test_split_sizes(dummy_images, dummy_labels):
    x_tr, x_val, y_tr, y_val = split_train_val(dummy_images, dummy_labels, val_size=0.2)
    assert len(x_tr) == 80
    assert len(x_val) == 20


def test_split_no_overlap(dummy_images, dummy_labels):
    x_tr, x_val, _, _ = split_train_val(dummy_images, dummy_labels)
    assert len(x_tr) + len(x_val) == len(dummy_images)


# ── build_dataset ─────────────────────────────────────────────────────────────

def test_build_dataset_batches(dummy_images, dummy_labels):
    import tensorflow as tf
    oh = to_one_hot(dummy_labels)
    ds = build_dataset(normalize(dummy_images), oh, batch_size=32, shuffle=False)
    batch = next(iter(ds))
    assert batch[0].shape[0] <= 32
    assert batch[1].shape[-1] == NUM_CLASSES


@pytest.mark.parametrize("val_size", [0.1, 0.2, 0.3])
def test_split_parametrized(dummy_images, dummy_labels, val_size):
    x_tr, x_val, _, _ = split_train_val(dummy_images, dummy_labels, val_size=val_size)
    expected_val = int(100 * val_size)
    assert abs(len(x_val) - expected_val) <= 2, f"Val size mismatch for {val_size}"
