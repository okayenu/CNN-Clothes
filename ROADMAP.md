# Fashion Item Classification — Improvement Roadmap

A 50-task actionable roadmap for improving this project over the next year, organized by category and roughly ordered from foundational to advanced within each group.

---

## Data & Preprocessing

1. **Normalize pixel values** — Divide by 255.0 to scale inputs to [0, 1]; test if this improves convergence speed and accuracy
2. **Add data augmentation** — Apply random horizontal flips, small rotations (±10°), and zoom to the training set using `ImageDataGenerator` or `tf.keras.layers.RandomFlip`
3. **Visualize class distribution** — Check for class imbalance across train/val/test splits and document findings
4. **Add a dedicated hold-out test split** — Currently the test CSV is used directly; create a proper unseen evaluation set that is never touched during development
5. **Load the full training CSV** — Add the 60,000-sample training CSV back into version control (or use Git LFS) so the project is fully reproducible
6. **Implement a data pipeline with `tf.data`** — Replace Pandas-based loading with a `tf.data.Dataset` pipeline for better performance and scalability
7. **Visualize augmented samples** — Display before/after augmentation side-by-side to validate transformations are sensible for fashion data
8. **Add per-channel standardization** — Compute mean and std of the training set and standardize rather than just scaling
9. **Experiment with image resizing** — Upsample images to 64×64 or 96×96 and measure the accuracy/compute trade-off with deeper models

---

## Model Architecture

10. **Add Batch Normalization** — Insert `BatchNormalization` after each Conv2D layer and measure its effect on training stability and accuracy
11. **Try a deeper architecture** — Build a 4-block CNN (4 pairs of Conv+Pool) and compare against the 2-block baseline
12. **Implement a ResNet-style skip connection** — Add residual connections to prevent vanishing gradients in deeper versions
13. **Replace sigmoid output with softmax** — The current output uses `sigmoid` for a multi-class problem; switch to `softmax` (the standard for multi-class classification)
14. **Experiment with different filter sizes** — Test 5×5 and 1×1 convolutions alongside 3×3 to see the accuracy impact
15. **Try Global Average Pooling** — Replace the `Flatten → Dense(128)` block with Global Average Pooling to reduce parameters and overfitting
16. **Implement a VGG-style network** — Build a VGG-like architecture using stacked 3×3 Conv layers before each pooling step
17. **Apply transfer learning with MobileNetV2** — Resize images to 96×96, load MobileNetV2 pretrained on ImageNet, and fine-tune the top layers
18. **Apply transfer learning with EfficientNetB0** — Same approach as above but with EfficientNet for a state-of-the-art baseline
19. **Build an ensemble model** — Train 3–5 models with different seeds/architectures and combine predictions by averaging softmax outputs
20. **Implement a lightweight model for edge deployment** — Design a model with <100K parameters targeting ~90% accuracy for potential mobile use

---

## Training & Optimization

21. **Tune the learning rate** — Use `tf.keras.callbacks.LearningRateScheduler` or cosine annealing instead of Adam's default rate
22. **Add early stopping** — Implement `EarlyStopping(monitor='val_loss', patience=5)` to prevent wasted epochs and overfitting
23. **Add model checkpointing** — Save the best model weights during training with `ModelCheckpoint(save_best_only=True)`
24. **Plot training/validation curves** — Add accuracy and loss curves per epoch for both models to diagnose underfitting vs. overfitting
25. **Experiment with different optimizers** — Compare Adam, SGD with momentum, RMSprop, and AdamW on the same architecture
26. **Implement label smoothing** — Add label smoothing (e.g., ε=0.1) to the loss function to improve generalization
27. **Try different batch sizes** — Benchmark 64, 128, 256, and 512 batch sizes and their effect on final accuracy
28. **Add L2 weight regularization** — Apply `kernel_regularizer=tf.keras.regularizers.l2(1e-4)` to dense layers and compare with dropout-only
29. **Use mixed-precision training** — Enable `tf.keras.mixed_precision` for faster training on compatible GPUs

---

## Evaluation & Analysis

30. **Fix the confusion matrix visualization** — Normalize it (show percentages) and annotate cells; add a color scale legend
31. **Plot per-class accuracy over time** — Track how each of the 10 classes improves across epochs to spot trouble classes early
32. **Perform error analysis** — Pull the 50 most confidently wrong predictions and visually inspect them; document patterns
33. **Compute and report ROC-AUC per class** — Add one-vs-rest ROC curves for each of the 10 categories
34. **Add a calibration analysis** — Use reliability diagrams to check whether model confidence scores are well-calibrated
35. **Benchmark inference speed** — Measure average prediction latency per image (ms) on CPU and GPU
36. **Test model robustness** — Evaluate accuracy on images with added Gaussian noise, blur, and brightness shifts

---

## Code Quality & Reproducibility

37. **Set random seeds everywhere** — Set seeds for `random`, `numpy`, and `tensorflow` at the top of the notebook for full reproducibility
38. **Refactor into Python modules** — Extract data loading, model building, training, and evaluation into separate `.py` files (`data.py`, `model.py`, `train.py`, `evaluate.py`)
39. **Add a `requirements.txt`** — Pin exact library versions (`tensorflow==x.x`, `numpy==x.x`, etc.) for reproducible environment setup
40. **Add a `config.py` or YAML config file** — Centralize all hyperparameters (epochs, batch size, dropout rate, filters) in one place
41. **Write unit tests** — Add `pytest` tests for data loading shape correctness, model output shape, and preprocessing functions
42. **Add docstrings to all functions** — Document inputs, outputs, and purpose of every function when refactoring into modules

---

## MLOps & Experiment Tracking

43. **Integrate MLflow or Weights & Biases** — Log hyperparameters, metrics, and artifacts for every training run to enable systematic comparison
44. **Export the trained model** — Save the best model in TensorFlow SavedModel format and document how to reload and run inference
45. **Convert to TFLite** — Export the model to TensorFlow Lite for potential mobile/edge deployment and benchmark its accuracy drop
46. **Build a simple prediction script** — Write `predict.py` that takes an image path as input and outputs the predicted class label and confidence

---

## Visualization & Reporting

47. **Build a Grad-CAM visualization** — Apply Gradient-weighted Class Activation Mapping to show which pixels drive each prediction
48. **Create a t-SNE / UMAP embedding plot** — Extract penultimate-layer features and plot them in 2D to visualize how well the model separates classes
49. **Write a structured final report** — Summarize all experiments, architectures, and results in a well-formatted Markdown or PDF report comparing every model variant

---

## Stretch Goal

50. **Build an interactive demo** — Create a Gradio or Streamlit web app where a user can draw or upload a fashion image and get a live prediction with confidence bar chart

---

## Suggested Timeline

| Period | Tasks | Focus |
|---|---|---|
| **Month 1–2** | 1–9, 13, 22–25, 37–40 | Fix fundamentals, reproducibility, data pipeline |
| **Month 3–5** | 10–16, 26–29, 30–36 | Architecture experiments, thorough evaluation |
| **Month 6–9** | 17–20, 43–46 | Transfer learning, MLOps |
| **Month 10–12** | 47–50 + write-up | Interpretability, deployment, final report |

---

## Baseline Results (Starting Point)

| Model | Test Accuracy | Test Loss |
|---|---|---|
| Baseline CNN (Model 1) | 91.0% | 0.2609 |
| Improved CNN with Dropout (Model 2) | 92.8% | 0.2020 |

> **Hardest class to classify:** Shirt (F1: 0.75 → 0.79)
> **Easiest class to classify:** Trouser (F1: 0.99 in both models)
