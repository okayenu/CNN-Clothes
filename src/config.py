"""
config.py — Centralized hyperparameters and path constants for the
Fashion-MNIST CNN classification project.
"""

import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Training hyperparameters ─────────────────────────────────────────────────
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
DROPOUT_RATE = 0.5
L2_RATE = 1e-4
LABEL_SMOOTHING = 0.1

# ── Architecture ─────────────────────────────────────────────────────────────
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "input")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TRAIN_CSV = os.path.join(DATA_DIR, "fashion-mnist_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "fashion-mnist_test.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
TFLITE_PATH = os.path.join(MODEL_DIR, "model.tflite")

# ── MLflow / experiment tracking ─────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "fashion-mnist-cnn"
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")

# ── Deterministic ops (set before importing TF) ───────────────────────────────
# os.environ["TF_DETERMINISTIC_OPS"] = "1"

for _dir in (MODEL_DIR, RESULTS_DIR):
    os.makedirs(_dir, exist_ok=True)

# [2025-04-03 1:28 PM] Task 37: Add SEED=42 constant

# [2025-04-11 12:07 PM] Task 40: Create config.py with SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE

# [2025-04-11 12:18 PM] Task 40: Add NUM_CLASSES=10, INPUT_SHAPE=(28,28,1), CLASS_NAMES list

# [2025-04-11 1:07 PM] Task 40: Add path constants: DATA_DIR, MODEL_DIR, RESULTS_DIR

# [2025-04-11 1:44 PM] Task 40: Add EARLY_STOPPING_PATIENCE, DROPOUT_RATE, L2_RATE
