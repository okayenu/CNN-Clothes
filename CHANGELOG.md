# Changelog

All notable changes to the Fashion-MNIST CNN project are recorded here.

---

## [Unreleased]

---

## Task 13 — Sigmoid → Softmax Output
- Changed final Dense activation from `sigmoid` to `softmax`
- Switched loss from `binary_crossentropy` to `categorical_crossentropy`
- Added one-hot encoding for labels
- Verified output probabilities sum to 1 on sample batches

## Task 37 — Random Seed Reproducibility
- Added `SEED = 42` constant to `config.py`
- Set `np.random.seed(SEED)` and `tf.random.set_seed(SEED)` in `data.py`
- Passed `seed=SEED` to shuffle/split calls in `train.py`
- Documented `TF_DETERMINISTIC_OPS=1` option for GPU reproducibility

## Task 39 — requirements.txt
- Created `requirements.txt` with pinned library versions
- Added Python version constraint comment
- Verified clean install in virtual environment

## Task 40 — config.py
- Created `src/config.py` centralizing all hyperparameters and path constants
- Added `SEED`, `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `DROPOUT_RATE`, `L2_RATE`
- Added `NUM_CLASSES`, `INPUT_SHAPE`, `CLASS_NAMES`
- Added path constants: `DATA_DIR`, `MODEL_DIR`, `RESULTS_DIR`
- Added `EARLY_STOPPING_PATIENCE`, `LABEL_SMOOTHING`

## Task 22 — Early Stopping
- Added `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)` callback
- Integrated into `get_callbacks()` in `train.py`
- Logs `stopped_epoch` and best `val_loss` in training report

## Task 23 — Model Checkpointing
- Added `ModelCheckpoint(save_best_only=True)` callback
- Best model saved to `models/best_model.keras`

## Task 24 — Training Curves
- Added `plot_training_curves()` in `evaluate.py`
- Generates accuracy and loss plots per epoch for both train and val

## Task 1 — Pixel Normalization
- Added `normalize()` in `data.py` to scale inputs to [0, 1]

## Task 8 — Per-channel Standardization
- Added `standardize()` in `data.py` for mean/std normalization

## Task 2 — Data Augmentation
- Added `_augment()` function with random flip and rotation
- Integrated into `build_dataset()` pipeline

## Task 6 — tf.data Pipeline
- Replaced Pandas-based iteration with `tf.data.Dataset` pipeline
- Added batching, shuffling, and prefetching with `AUTOTUNE`

## Task 38 — Python Modules Refactor
- Extracted data loading into `src/data.py`
- Extracted model building into `src/model.py`
- Extracted training into `src/train.py`
- Extracted evaluation into `src/evaluate.py`

## Task 41 — Unit Tests
- Added `tests/test_data.py` — shape, normalization, split correctness
- Added `tests/test_model.py` — output shape, softmax sum, optimizer validation
- Added `tests/test_train.py` — callbacks, LR schedule, early stopping integration

## Task 42 — Docstrings
- Added docstrings to all functions across all modules

## Task 10 — Batch Normalization
- Added `build_batchnorm()` in `model.py`

## Task 11 — Deeper Architecture
- Added `build_deeper()` with 4 Conv+Pool blocks in `model.py`

## Task 15 — Global Average Pooling
- Added `build_gap()` replacing Flatten+Dense with GAP in `model.py`

## Task 16 — VGG-style Architecture
- Added `build_vgg_style()` with stacked 3×3 Conv pairs in `model.py`

## Task 13 (L2) — L2 Regularization
- Added `build_l2_regularized()` in `model.py`

## Task 12 — ResNet Skip Connections (risky)
- Added `_residual_block()` and `build_resnet_style()` in `model.py`

## Task 21 — Learning Rate Scheduling
- Added `cosine_annealing_schedule()` and `get_lr_scheduler()` in `train.py`

## Task 26 — Label Smoothing
- Added `label_smoothing` parameter to `compile_model()` in `model.py`

## Task 29 — Mixed Precision Training
- Added `enable_mixed_precision()` in `train.py`

## Task 25 — Optimizer Comparison
- Added `get_optimizer()` supporting adam, sgd, rmsprop, adamw in `model.py`

## Task 30 — Confusion Matrix
- Added `plot_confusion_matrix()` with normalization and cell annotations in `evaluate.py`

## Task 33 — ROC-AUC per Class
- Added `compute_roc_auc()` with one-vs-rest ROC curves in `evaluate.py`

## Task 34 — Calibration Analysis
- Added `plot_calibration()` reliability diagrams in `evaluate.py`

## Task 32 — Error Analysis
- Added `error_analysis()` for top-N confident wrong predictions in `evaluate.py`

## Task 35 — Inference Latency Benchmark
- Added `benchmark_latency()` in `evaluate.py`

## Task 17 — MobileNetV2 Transfer Learning (risky)
- Added `build_mobilenetv2()` in `model.py`

## Task 18 — EfficientNetB0 Transfer Learning (risky)
- Added `build_efficientnetb0()` in `model.py`

## Task 19 — Ensemble Model (risky)
- Added `ensemble_predict()` in `model.py`

## Task 20 — Lightweight Edge Model (risky)
- Added `build_lightweight()` using depthwise convolutions in `model.py`

## Task 43 — MLflow Integration (risky)
- Created `src/mlflow_tracking.py` with run management and metric logging

## Task 44 — SavedModel Export
- Added `export_saved_model()` in `src/export.py`

## Task 45 — TFLite Conversion (risky)
- Added `convert_to_tflite()` and `benchmark_tflite()` in `src/export.py`

## Task 46 — Prediction Script
- Created `src/predict.py` CLI tool for single-image inference

## Task 47 — Grad-CAM (risky)
- Created `src/gradcam.py` with heatmap computation and overlay visualization

## Task 48 — t-SNE / UMAP Embeddings (risky)
- Created `src/embed.py` with `plot_tsne()` and `plot_umap()`

## Task 49 — Final Report & Documentation
- Written `REPORT.md`, `RETROSPECTIVE.md`, `SUMMARY.md`, `QUICKSTART.md`

## Task 50 — Gradio Interactive Demo (risky)
- Created `demo.py` with image upload, confidence bar chart, and Grad-CAM overlay tab
