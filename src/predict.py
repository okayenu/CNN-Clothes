"""
predict.py — CLI prediction script for Fashion-MNIST CNN.

Usage:
    python predict.py --image path/to/image.png [--model models/best_model.keras]
"""

import argparse
import sys
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from config import BEST_MODEL_PATH, CLASS_NAMES, INPUT_SHAPE


def load_image(image_path: str) -> np.ndarray:
    """Load, resize, normalize, and reshape an image file for inference.

    Args:
        image_path: Path to a PNG/JPEG image.

    Returns:
        Float32 array of shape [1, H, W, 1].
    """
    img = tf.keras.utils.load_img(
        image_path,
        color_mode="grayscale",
        target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
    )
    arr = tf.keras.utils.img_to_array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_single(image_path: str, model_path: str = BEST_MODEL_PATH):
    """Run inference on a single image and return label and confidence.

    Args:
        image_path: Path to input image.
        model_path: Path to a saved Keras model.

    Returns:
        Tuple of (predicted_class_name: str, confidence: float,
                  all_probs: np.ndarray).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    x = load_image(image_path)
    probs = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    return CLASS_NAMES[top_idx], float(probs[top_idx]), probs


def main():
    parser = argparse.ArgumentParser(
        description="Fashion-MNIST CNN inference script."
    )
    parser.add_argument(
        "--image", required=True, help="Path to input image (grayscale PNG/JPEG)."
    )
    parser.add_argument(
        "--model", default=BEST_MODEL_PATH, help="Path to saved Keras model."
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="Number of top predictions to display."
    )
    args = parser.parse_args()

    label, confidence, probs = predict_single(args.image, args.model)

    print(f"\nPrediction : {label}")
    print(f"Confidence : {confidence * 100:.1f}%\n")
    print(f"Top-{args.top_k} predictions:")
    top_k_idx = np.argsort(probs)[::-1][: args.top_k]
    for rank, idx in enumerate(top_k_idx, 1):
        print(f"  {rank}. {CLASS_NAMES[idx]:<15} {probs[idx] * 100:.1f}%")


if __name__ == "__main__":
    main()

# [2025-12-12 11:52 AM] Task 46: Write CLI parser: argparse with --model, --image, --top_k args

# [2025-12-12 12:18 PM] Task 46: Load SavedModel or TFLite based on file extension

# [2025-12-12 1:07 PM] Task 46: Preprocess image: load, resize, normalize/standardize

# [2025-12-12 3:47 PM] Task 46: Print top_k predictions with class name and confidence %
