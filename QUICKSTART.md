# Quickstart Guide

## Prerequisites

- Python 3.9 or later
- 4 GB RAM minimum (8 GB recommended for transfer learning)

## 1. Clone and Set Up Environment

```bash
git clone https://github.com/okayenu/git-tutorial.git
cd git-tutorial

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## 2. Add Training Data

The training CSV (`input/fashion-mnist_train.csv`) is excluded from version control due to
size. Download it from the [Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist)
or Kaggle, then place it at:

```
input/fashion-mnist_train.csv
```

The test CSV (`input/fashion-mnist_test.csv`) is already included.

## 3. Train a Model

```bash
# Train the dropout CNN (recommended baseline)
python src/main.py --model dropout

# Train with cosine LR annealing
python src/main.py --model batchnorm --lr_schedule

# Train a ResNet-style model
python src/main.py --model resnet --epochs 30
```

Available model names: `baseline`, `dropout`, `batchnorm`, `deeper`, `resnet`, `gap`, `vgg`, `l2`

## 4. Predict on a Single Image

```bash
python src/predict.py --image path/to/your/image.png --top_k 3
```

Output example:
```
Prediction : Sneaker
Confidence : 97.3%

Top-3 predictions:
  1. Sneaker         97.3%
  2. Ankle boot      2.1%
  3. Sandal          0.4%
```

## 5. Launch the Interactive Demo

```bash
python demo.py
```

Open your browser at `http://localhost:7860`. Upload a grayscale fashion image to see
the prediction, confidence bar chart, and Grad-CAM attention overlay.

## 6. Run Tests

```bash
pytest tests/ -v
```

## 7. Export to TFLite

```python
from src.export import convert_to_tflite
convert_to_tflite(model_path="models/best_model.keras", quantize=True)
```

## 8. Track Experiments with MLflow

```bash
pip install mlflow
mlflow ui  # open http://localhost:5000 in browser
```

Then in your training code:
```python
from src.mlflow_tracking import start_run, log_history, end_run
run = start_run(run_name="my_experiment")
history = model.fit(...)
log_history(history)
end_run()
```

## Project Structure

```
├── src/
│   ├── config.py          # Hyperparameters and paths
│   ├── data.py            # Data loading and preprocessing
│   ├── model.py           # CNN architectures
│   ├── train.py           # Training loop and callbacks
│   ├── evaluate.py        # Metrics and visualizations
│   ├── predict.py         # CLI inference script
│   ├── gradcam.py         # Grad-CAM visualization
│   ├── embed.py           # t-SNE / UMAP embeddings
│   ├── export.py          # SavedModel and TFLite export
│   ├── mlflow_tracking.py # MLflow integration
│   └── main.py            # End-to-end training pipeline
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_train.py
├── input/                 # CSV data files
├── models/                # Saved model weights
├── results/               # Plots and evaluation outputs
├── demo.py                # Gradio interactive demo
├── requirements.txt
├── CHANGELOG.md
├── REPORT.md
├── RETROSPECTIVE.md
├── SUMMARY.md
└── QUICKSTART.md
```
