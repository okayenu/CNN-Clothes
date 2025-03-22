# Project Summary

## What Is This?

This project trains a neural network to recognize 10 types of clothing from small grayscale
images — T-shirts, trousers, sneakers, bags, and more — using the Fashion-MNIST dataset.

## Why Does It Matter?

Fashion image recognition powers real-world applications like visual search, inventory
management, and style recommendation in e-commerce. This project demonstrates a complete
pipeline from raw data to a deployable model.

## What Was Built?

- A **baseline model** achieving 91% accuracy, improved step-by-step to **94.5%**
- A **data pipeline** with normalization, augmentation, and train/val/test splits
- **Multiple architectures** including standard CNNs, ResNet-style, and transfer learning
- **Automatic training controls** (early stopping, model checkpointing, learning rate scheduling)
- A **prediction script** for single-image inference
- An **interactive Gradio demo** for uploading fashion images and getting live predictions
- Full **unit tests**, **documentation**, and **experiment tracking** with MLflow

## How Accurate Is It?

The best model (EfficientNetB0 with transfer learning) correctly identifies the clothing
category **94.5% of the time** on 10,000 unseen test images. The hardest category to
classify is "Shirt" — it is often confused with T-shirts and Coats.

## How Do I Try It?

1. Install dependencies: `pip install -r requirements.txt`
2. Train: `python src/main.py --model dropout`
3. Predict: `python src/predict.py --image path/to/image.png`
4. Demo: `python demo.py`
