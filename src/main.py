"""
main.py — End-to-end training pipeline for Fashion-MNIST CNN classification.

Usage:
    python main.py [--model baseline|dropout|batchnorm|deeper|resnet|gap|vgg|l2]
                   [--epochs N] [--batch_size N] [--lr FLOAT]
                   [--augment] [--lr_schedule] [--mixed_precision]
"""

import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    SEED,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    DROPOUT_RATE,
    L2_RATE,
    LABEL_SMOOTHING,
    RESULTS_DIR,
    TRAIN_CSV,
    TEST_CSV,
)
from data import get_datasets
from model import (
    build_baseline,
    build_with_dropout,
    build_batchnorm,
    build_deeper,
    build_resnet_style,
    build_gap,
    build_vgg_style,
    build_l2_regularized,
    compile_model,
)
from train import train, get_callbacks, log_training_report
from evaluate import (
    plot_confusion_matrix,
    plot_training_curves,
    compute_roc_auc,
    full_report,
)

import random
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_BUILDERS = {
    "baseline": build_baseline,
    "dropout": build_with_dropout,
    "batchnorm": build_batchnorm,
    "deeper": build_deeper,
    "resnet": build_resnet_style,
    "gap": build_gap,
    "vgg": build_vgg_style,
    "l2": build_l2_regularized,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fashion-MNIST CNN training pipeline.")
    parser.add_argument("--model", default="dropout", choices=list(MODEL_BUILDERS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--lr_schedule", action="store_true", default=False)
    parser.add_argument("--mixed_precision", action="store_true", default=False)
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mixed_precision:
        from train import enable_mixed_precision
        enable_mixed_precision()

    print(f"\n{'='*50}")
    print(f"  Fashion-MNIST CNN — model: {args.model}")
    print(f"{'='*50}\n")

    data = get_datasets(
        train_path=TRAIN_CSV,
        test_path=TEST_CSV,
        augment=args.augment,
    )

    build_fn = MODEL_BUILDERS[args.model]
    model = build_fn()
    model = compile_model(model, learning_rate=args.lr,
                           label_smoothing=args.label_smoothing)
    model.summary()

    history = train(
        model,
        data["train_ds"],
        data["val_ds"],
        epochs=args.epochs,
        use_lr_scheduler=args.lr_schedule,
    )

    log_training_report(history)
    plot_training_curves(
        history,
        save_path=os.path.join(RESULTS_DIR, f"{args.model}_curves.png"),
    )

    print("\nEvaluating on test set…")
    test_loss, test_acc = model.evaluate(data["test_ds"], verbose=1)
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    y_prob = model.predict(data["x_test"], verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = data["y_test"]

    print("\n" + full_report(y_true, y_pred))

    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(RESULTS_DIR, f"{args.model}_confusion.png"),
    )
    compute_roc_auc(
        data["y_test_oh"], y_prob,
        save_path=os.path.join(RESULTS_DIR, f"{args.model}_roc.png"),
    )


if __name__ == "__main__":
    main()

# [2025-06-14 4:28 PM] Task 38: Wire data → model → train → evaluate in main.py pipeline
