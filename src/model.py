"""
model.py — CNN architectures for Fashion-MNIST classification.

Includes baseline, improved (with dropout), BatchNorm, deeper, ResNet-style,
GAP, VGG-style, MobileNetV2 transfer learning, EfficientNetB0 transfer
learning, ensemble, and lightweight edge models.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from config import (
    INPUT_SHAPE,
    NUM_CLASSES,
    DROPOUT_RATE,
    L2_RATE,
    LEARNING_RATE,
)


# ── Softmax output helper ─────────────────────────────────────────────────────

def _output_block(x, num_classes: int = NUM_CLASSES, label_smoothing: float = 0.0):
    """Dense softmax output layer."""
    return layers.Dense(num_classes, activation="softmax")(x)


# ── 1. Baseline CNN ───────────────────────────────────────────────────────────

def build_baseline(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """Two-block Conv+Pool baseline (Task 13: softmax output)."""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    out = _output_block(x, num_classes)
    return models.Model(inp, out, name="baseline_cnn")


# ── 2. CNN with Dropout ───────────────────────────────────────────────────────

def build_with_dropout(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                        dropout_rate=DROPOUT_RATE):
    """Baseline + dropout after dense layer."""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    out = _output_block(x, num_classes)
    return models.Model(inp, out, name="cnn_dropout")


# ── 3. BatchNorm CNN (Task 10) ────────────────────────────────────────────────

def build_batchnorm(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                     dropout_rate=DROPOUT_RATE):
    """Conv blocks with BatchNormalization after each Conv2D."""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    out = _output_block(x, num_classes)
    return models.Model(inp, out, name="cnn_batchnorm")


# ── 4. Deeper CNN — 4 blocks (Task 11) ───────────────────────────────────────

def build_deeper(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                  dropout_rate=DROPOUT_RATE):
    """Four Conv+Pool blocks for higher capacity."""
    inp = layers.Input(shape=input_shape)
    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(
            inp if filters == 32 else x
        )
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    out = _output_block(x, num_classes)
    return models.Model(inp, out, name="deeper_cnn")


# ── 5. ResNet-style skip connections (Task 12 — RISKY) ───────────────────────

def _residual_block(x, filters: int):
    """A simple residual block: two Conv layers with a skip connection."""
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding="same")(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def build_resnet_style(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                        dropout_rate=DROPOUT_RATE):
    """ResNet-inspired architecture with skip connections."""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = _residual_block(x, 32)
    x = layers.MaxPooling2D()(x)
    x = _residual_block(x, 64)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    out = _output_block(x, num_classes)
    return models.Model(inp, out, name="resnet_style")


# ── 6. Global Average Pooling model (Task 15) ─────────────────────────────────

def build_gap(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """Replace Flatten+Dense(128) with GlobalAveragePooling."""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(num_classes, (1, 1), padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Softmax()(x)
    return models.Model(inp, out, name="gap_cnn")


# ── 7. VGG-style (Task 16) ───────────────────────────────────────────────────

def build_vgg_style(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                     dropout_rate=DROPOUT_RATE):
    """Stacked 3×3 Conv pairs before each pooling step (VGG-inspired)."""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    out = _output_block(x, num_classes)
    return models.Model(inp, out, name="vgg_style")


# ── 8. MobileNetV2 transfer learning (Task 17 — RISKY) ───────────────────────

def build_mobilenetv2(input_shape=(96, 96, 3), num_classes=NUM_CLASSES,
                       dropout_rate=DROPOUT_RATE):
    """Fine-tune MobileNetV2 pretrained on ImageNet."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False
    inp = layers.Input(shape=input_shape)
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    out = _output_block(x, num_classes)
    return models.Model(inp, out, name="mobilenetv2_transfer")


# ── 9. EfficientNetB0 transfer learning (Task 18 — RISKY) ────────────────────

def build_efficientnetb0(input_shape=(96, 96, 3), num_classes=NUM_CLASSES,
                          dropout_rate=DROPOUT_RATE):
    """Fine-tune EfficientNetB0 pretrained on ImageNet."""
    base = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False
    inp = layers.Input(shape=input_shape)
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    out = _output_block(x, num_classes)
    return models.Model(inp, out, name="efficientnetb0_transfer")


# ── 10. Lightweight edge model (Task 20 — RISKY) ─────────────────────────────

def build_lightweight(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """<100K parameter model targeting ~90% accuracy for mobile/edge."""
    inp = layers.Input(shape=input_shape)
    x = layers.DepthwiseConv2D((3, 3), padding="same", activation="relu")(inp)
    x = layers.Conv2D(16, (1, 1), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.DepthwiseConv2D((3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(32, (1, 1), padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = _output_block(x, num_classes)
    return models.Model(inp, out, name="lightweight_cnn")


# ── 11. Ensemble predictions (Task 19 — RISKY) ───────────────────────────────

def ensemble_predict(models_list, x):
    """Average softmax probabilities from a list of models.

    Args:
        models_list: List of compiled Keras models.
        x: Input array.

    Returns:
        Averaged probability array of shape [N, num_classes].
    """
    preds = np.stack([m.predict(x, verbose=0) for m in models_list], axis=0)
    return preds.mean(axis=0)


# ── L2-regularized dense variant ─────────────────────────────────────────────

def build_l2_regularized(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                           l2_rate=L2_RATE, dropout_rate=DROPOUT_RATE):
    """Baseline with L2 weight regularization on Dense layers."""
    reg = regularizers.l2(l2_rate)
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    out = _output_block(x, num_classes)
    return models.Model(inp, out, name="l2_cnn")


# ── Compiler helper ───────────────────────────────────────────────────────────

def compile_model(model, learning_rate=LEARNING_RATE, label_smoothing=0.0,
                   optimizer_name="adam"):
    """Compile a model with categorical crossentropy and accuracy metric.

    Args:
        model: Uncompiled Keras model.
        learning_rate: Optimizer learning rate.
        label_smoothing: Float in [0, 1) for label smoothing regularization.
        optimizer_name: One of 'adam', 'sgd', 'rmsprop', 'adamw'.

    Returns:
        Compiled Keras model.
    """
    opt = get_optimizer(optimizer_name, learning_rate)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        ),
        metrics=["accuracy"],
    )
    return model


def get_optimizer(name: str, lr: float = LEARNING_RATE):
    """Return a Keras optimizer by name.

    Args:
        name: One of 'adam', 'sgd', 'rmsprop', 'adamw'.
        lr: Learning rate.

    Raises:
        ValueError: If name is not recognized.
    """
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name!r}")


# numpy needed for ensemble_predict — imported lazily to avoid circular import
import numpy as np  # noqa: E402
