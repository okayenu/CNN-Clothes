"""
data.py — Data loading, preprocessing, augmentation, and tf.data pipeline
for Fashion-MNIST classification.
"""

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import (
    SEED,
    BATCH_SIZE,
    INPUT_SHAPE,
    NUM_CLASSES,
    TRAIN_CSV,
    TEST_CSV,
)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── Raw loading ───────────────────────────────────────────────────────────────

def load_csv(path: str):
    """Load a Fashion-MNIST CSV and return (images, labels) as numpy arrays.

    Args:
        path: Path to CSV file with label in the first column.

    Returns:
        Tuple of (images float32 [N,28,28,1], labels int32 [N]).
    """
    df = pd.read_csv(path)
    labels = df.iloc[:, 0].values.astype(np.int32)
    pixels = df.iloc[:, 1:].values.astype(np.float32)
    images = pixels.reshape(-1, 28, 28, 1)
    return images, labels


# ── Normalization & standardization ──────────────────────────────────────────

def normalize(images: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0, 1]."""
    return images / 255.0


def standardize(images: np.ndarray, mean=None, std=None):
    """Per-channel standardization: subtract mean, divide by std.

    Args:
        images: Float array of shape [N, H, W, C].
        mean: Pre-computed channel mean (computed from images if None).
        std: Pre-computed channel std (computed from images if None).

    Returns:
        Tuple of (standardized_images, mean, std).
    """
    if mean is None:
        mean = images.mean(axis=(0, 1, 2), keepdims=True)
    if std is None:
        std = images.std(axis=(0, 1, 2), keepdims=True) + 1e-7
    return (images - mean) / std, mean, std


# ── One-hot encoding ──────────────────────────────────────────────────────────

def to_one_hot(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Convert integer labels to one-hot vectors."""
    return tf.keras.utils.to_categorical(labels, num_classes=num_classes)


# ── Train / validation split ──────────────────────────────────────────────────

def split_train_val(images, labels, val_size: float = 0.1):
    """Split arrays into train and validation subsets.

    Args:
        images: Input images array.
        labels: Corresponding labels.
        val_size: Fraction of data to reserve for validation.

    Returns:
        Tuple of (x_train, x_val, y_train, y_val).
    """
    return train_test_split(
        images, labels, test_size=val_size, random_state=SEED, stratify=labels
    )


# ── tf.data augmentation pipeline ────────────────────────────────────────────

def _augment(image, label):
    """Apply random flip and small rotation to a single image tensor."""
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.keras.layers.RandomRotation(0.05, seed=SEED)(
        tf.expand_dims(image, 0), training=True
    )
    image = tf.squeeze(image, 0)
    return image, label


def build_dataset(
    images,
    labels,
    augment: bool = False,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from numpy arrays.

    Args:
        images: Float32 array [N, H, W, C].
        labels: One-hot float32 array [N, num_classes] or int array [N].
        augment: Whether to apply random augmentation.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle before batching.

    Returns:
        A batched, prefetched tf.data.Dataset.
    """
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=SEED)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Convenience loader ────────────────────────────────────────────────────────

def get_datasets(
    train_path: str = TRAIN_CSV,
    test_path: str = TEST_CSV,
    val_size: float = 0.1,
    augment: bool = True,
    use_standardize: bool = False,
):
    """End-to-end data preparation.

    Returns:
        Dict with keys: train_ds, val_ds, test_ds, class_names,
        train_mean (if standardize), train_std (if standardize).
    """
    x_train_raw, y_train_raw = load_csv(train_path)
    x_test, y_test = load_csv(test_path)

    x_train_raw = normalize(x_train_raw)
    x_test = normalize(x_test)

    if use_standardize:
        x_train_raw, mean, std = standardize(x_train_raw)
        x_test, _, _ = standardize(x_test, mean, std)
    else:
        mean, std = None, None

    x_train, x_val, y_train, y_val = split_train_val(x_train_raw, y_train_raw, val_size)

    y_train_oh = to_one_hot(y_train)
    y_val_oh = to_one_hot(y_val)
    y_test_oh = to_one_hot(y_test)

    train_ds = build_dataset(x_train, y_train_oh, augment=augment, shuffle=True)
    val_ds = build_dataset(x_val, y_val_oh, augment=False, shuffle=False)
    test_ds = build_dataset(x_test, y_test_oh, augment=False, shuffle=False)

    result = {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "x_test": x_test,
        "y_test": y_test,
        "y_test_oh": y_test_oh,
    }
    if mean is not None:
        result["train_mean"] = mean
        result["train_std"] = std
    return result

# [2025-04-03 2:03 PM] Task 37: Set np.random.seed(SEED) and tf.random.set_seed(SEED)

# [2025-04-12 6:07 PM] Task 40: Import and use config constants in data.py

# [2025-04-30 11:37 AM] Task 1: Add /255.0 normalization in preprocess_data()

# [2025-04-30 2:03 PM] Task 1: Confirm dtype float32 after normalization

# [2025-05-10 10:19 AM] Task 4: Add split_data(X,y, val=0.1, test=0.1) with stratify

# [2025-05-10 10:44 AM] Task 4: Save hold-out indices to data/holdout_idx.npy

# [2025-05-10 12:18 PM] Task 4: Verify splits don't overlap via set intersection test

# [2025-05-15 12:07 PM] Task 5: Add fallback: load via tf.keras.datasets if CSV missing

# [2025-05-15 3:07 PM] Task 5: Write load_data(csv_path=None) function

# [2025-05-21 2:33 PM] Task 2: Add tf.keras.layers.RandomFlip to augmentation pipeline

# [2025-05-21 2:44 PM] Task 2: Add RandomRotation(factor=0.1) layer

# [2025-05-21 3:07 PM] Task 2: Add RandomZoom(height_factor=0.1) layer

# [2025-05-21 4:02 PM] Task 2: Wrap augmentation in a tf.keras.Sequential augment_layer

# [2025-05-22 3:22 PM] Task 2: Apply augment_layer only during training, skip for val/test

# [2025-05-26 10:19 AM] Task 6: Convert numpy arrays to tf.data.Dataset.from_tensor_slices

# [2025-05-26 2:17 PM] Task 6: Add .cache() after load to avoid re-read

# [2025-05-26 5:53 PM] Task 6: Add .shuffle(buffer_size=5000, seed=SEED)

# [2025-05-26 6:07 PM] Task 6: Add .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# [2025-05-29 3:22 PM] Task 6: Wire augment_layer into training Dataset only

# [2025-05-29 4:28 PM] Task 6: Add map() for one-hot encoding labels

# [2025-06-08 10:19 AM] Task 8: Compute mean/std on training set with np.mean/std

# [2025-06-08 12:07 PM] Task 8: Save stats to data/pixel_stats.npy

# [2025-06-08 5:53 PM] Task 8: Apply (x - mean) / std in preprocess_data()

# [2025-06-12 10:07 AM] Task 38: Move all data loading/preprocessing to data.py

# [2025-06-26 9:53 AM] Task 42: Add Google-style docstrings to all public functions in data.py
