"""
export.py — Model export utilities: SavedModel and TFLite conversion (Tasks 44, 45).
"""

import os
import numpy as np
import tensorflow as tf

from config import BEST_MODEL_PATH, TFLITE_PATH, MODEL_DIR


def export_saved_model(model, export_dir: str = None):
    """Save the model in TensorFlow SavedModel format.

    Args:
        model: Trained Keras model.
        export_dir: Directory to export to. Defaults to MODEL_DIR/saved_model.
    """
    if export_dir is None:
        export_dir = os.path.join(MODEL_DIR, "saved_model")
    os.makedirs(export_dir, exist_ok=True)
    model.save(export_dir)
    print(f"SavedModel exported to: {export_dir}")


def convert_to_tflite(model_path: str = None, tflite_path: str = TFLITE_PATH,
                       quantize: bool = False) -> bytes:
    """Convert a SavedModel or .keras model to TFLite format.

    Args:
        model_path: Path to SavedModel directory or .keras file.
            Defaults to BEST_MODEL_PATH.
        tflite_path: Output path for the .tflite file.
        quantize: If True, apply dynamic-range quantization to reduce model size.

    Returns:
        TFLite flatbuffer as bytes.
    """
    if model_path is None:
        model_path = BEST_MODEL_PATH

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path) \
        if os.path.isdir(model_path) \
        else tf.lite.TFLiteConverter.from_keras_model(
            tf.keras.models.load_model(model_path)
        )

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {tflite_path} ({len(tflite_model) / 1024:.1f} KB)")
    return tflite_model


def benchmark_tflite(tflite_path: str = TFLITE_PATH, x_sample: np.ndarray = None,
                      n_runs: int = 50) -> dict:
    """Benchmark TFLite model inference latency.

    Args:
        tflite_path: Path to .tflite model file.
        x_sample: Array of images [N, H, W, C] for testing.
        n_runs: Number of inference passes.

    Returns:
        Dict with mean_ms and accuracy_drop placeholder.
    """
    import time

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if x_sample is None:
        x_sample = np.random.rand(10, 28, 28, 1).astype(np.float32)

    start = time.perf_counter()
    for _ in range(n_runs):
        for img in x_sample:
            interpreter.set_tensor(input_details[0]["index"], img[np.newaxis])
            interpreter.invoke()
    elapsed_ms = (time.perf_counter() - start) / n_runs / len(x_sample) * 1000

    return {"mean_ms_per_image": round(elapsed_ms, 4), "n_runs": n_runs}

# [2025-12-05 11:37 AM] Task 45: Write convert_to_tflite(model_path, quant='float16') function

# [2025-12-05 11:52 AM] Task 45: Add int8 quantization with representative_dataset_gen()
