"""
demo.py — Gradio interactive demo for Fashion-MNIST CNN (Task 50 — RISKY).

Usage:
    python demo.py [--model models/best_model.keras] [--share]
"""

import argparse
import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import BEST_MODEL_PATH, CLASS_NAMES, INPUT_SHAPE
from gradcam import get_gradcam_heatmap, overlay_gradcam

_model = None


def _load_model(model_path: str):
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(model_path)
    return _model


def _preprocess(pil_image):
    """Convert a PIL image to a normalized float32 array."""
    import PIL.Image as PILImage
    img = pil_image.convert("L").resize(
        (INPUT_SHAPE[1], INPUT_SHAPE[0]), PILImage.LANCZOS
    )
    arr = np.array(img).astype(np.float32) / 255.0
    return arr.reshape(1, INPUT_SHAPE[0], INPUT_SHAPE[1], 1)


def predict_and_explain(pil_image, model_path: str = BEST_MODEL_PATH):
    """Run inference and produce a confidence bar chart + Grad-CAM overlay.

    Args:
        pil_image: PIL Image from Gradio upload.
        model_path: Path to the saved Keras model.

    Returns:
        Tuple of (label_str, confidence_dict, overlay_image_array).
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    model = _load_model(model_path)
    x = _preprocess(pil_image)
    probs = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(probs))

    # Confidence bar chart
    fig, ax = plt.subplots(figsize=(5, 3))
    sorted_idx = np.argsort(probs)[::-1]
    ax.barh(
        [CLASS_NAMES[i] for i in sorted_idx[:5]][::-1],
        [probs[i] * 100 for i in sorted_idx[:5]][::-1],
    )
    ax.set_xlabel("Confidence (%)")
    ax.set_title(f"Prediction: {CLASS_NAMES[top_idx]}")
    plt.tight_layout()

    # Grad-CAM overlay
    try:
        last_conv = next(
            l.name for l in reversed(model.layers)
            if isinstance(l, tf.keras.layers.Conv2D)
        )
        heatmap = get_gradcam_heatmap(model, x, last_conv, top_idx)
        overlay = overlay_gradcam(x[0], heatmap)
    except Exception:
        overlay = np.zeros((28, 28, 3), dtype=np.uint8)

    label = f"{CLASS_NAMES[top_idx]} ({probs[top_idx] * 100:.1f}%)"
    conf_dict = {CLASS_NAMES[i]: float(round(probs[i], 4)) for i in range(len(CLASS_NAMES))}
    return label, conf_dict, fig, overlay


def build_interface(model_path: str = BEST_MODEL_PATH):
    """Build and return the Gradio interface.

    Args:
        model_path: Path to the saved Keras model.

    Returns:
        gr.Blocks or gr.Interface object.
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Install gradio: pip install gradio")

    with gr.Blocks(title="Fashion-MNIST CNN Demo") as demo:
        gr.Markdown("# Fashion-MNIST CNN — Interactive Demo")
        gr.Markdown(
            "Upload or draw a 28×28 grayscale fashion image to get a prediction, "
            "confidence scores, and a Grad-CAM attention overlay."
        )

        with gr.Row():
            image_input = gr.Image(type="pil", label="Input Image", image_mode="L")

        with gr.Row():
            predict_btn = gr.Button("Predict", variant="primary")

        with gr.Row():
            label_out = gr.Textbox(label="Top Prediction")
            conf_out = gr.Label(label="Class Confidences", num_top_classes=5)

        with gr.Row():
            chart_out = gr.Plot(label="Confidence Bar Chart")
            cam_out = gr.Image(label="Grad-CAM Overlay")

        predict_btn.click(
            fn=lambda img: predict_and_explain(img, model_path),
            inputs=image_input,
            outputs=[label_out, conf_out, chart_out, cam_out],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST Gradio Demo")
    parser.add_argument("--model", default=BEST_MODEL_PATH)
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_interface(model_path=args.model)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()

# [2026-01-02 2:03 PM] Task 50: Write Gradio interface: image upload + prediction output

# [2026-01-02 2:17 PM] Task 50: Load SavedModel at startup in demo.py
