"""
gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM) for
Fashion-MNIST CNN models.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import cv2
import tensorflow as tf

from config import CLASS_NAMES, RESULTS_DIR
import os


def get_gradcam_heatmap(model, img_array: np.ndarray, last_conv_layer_name: str,
                         pred_index: int = None) -> np.ndarray:
    """Compute Grad-CAM heatmap for a single image.

    Args:
        model: Keras model with at least one Conv2D layer.
        img_array: Input image array of shape [1, H, W, C].
        last_conv_layer_name: Name of the last convolutional layer.
        pred_index: Class index for which to compute gradients.
            Uses the top predicted class if None.

    Returns:
        Float32 heatmap array of shape [H, W] with values in [0, 1].
    """
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Superimpose Grad-CAM heatmap on a grayscale image.

    Args:
        img: Original image array [H, W] or [H, W, 1], values in [0, 1].
        heatmap: Grad-CAM heatmap [H_cam, W_cam] in [0, 1].
        alpha: Heatmap overlay transparency.

    Returns:
        RGB overlay image as uint8 array [H, W, 3].
    """
    if img.ndim == 3:
        img = img[:, :, 0]
    img_uint8 = (img * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)


def visualize_gradcam(model, x_samples: np.ndarray, y_true: np.ndarray,
                       last_conv_layer_name: str, save_path: str = None, n: int = 10):
    """Plot Grad-CAM overlays for a batch of samples.

    Args:
        model: Keras CNN model.
        x_samples: Array [N, H, W, C] of images.
        y_true: True integer labels [N].
        last_conv_layer_name: Name of the final Conv2D layer.
        save_path: Path to save the figure.
        n: Number of samples to visualize.
    """
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 5))
    for i in range(n):
        img = x_samples[i : i + 1]
        heatmap = get_gradcam_heatmap(model, img, last_conv_layer_name)
        overlay = overlay_gradcam(x_samples[i], heatmap)
        pred_idx = int(np.argmax(model.predict(img, verbose=0)[0]))

        axes[0, i].imshow(x_samples[i].squeeze(), cmap="gray")
        axes[0, i].set_title(f"True: {CLASS_NAMES[y_true[i]][:5]}", fontsize=7)
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"Pred: {CLASS_NAMES[pred_idx][:5]}", fontsize=7)
        axes[1, i].axis("off")

    plt.suptitle("Grad-CAM Visualizations", fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
