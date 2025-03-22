"""
tests/test_model.py — Unit tests for model architectures.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import (
    build_baseline,
    build_with_dropout,
    build_batchnorm,
    build_deeper,
    build_gap,
    build_vgg_style,
    build_l2_regularized,
    compile_model,
    get_optimizer,
)
from config import INPUT_SHAPE, NUM_CLASSES


# ── Output shape ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("build_fn", [
    build_baseline,
    build_with_dropout,
    build_batchnorm,
    build_gap,
    build_vgg_style,
    build_l2_regularized,
])
def test_output_shape(build_fn):
    model = build_fn()
    x = np.random.rand(4, *INPUT_SHAPE).astype(np.float32)
    out = model.predict(x, verbose=0)
    assert out.shape == (4, NUM_CLASSES), f"{build_fn.__name__} output shape wrong"


# ── Softmax probabilities sum to 1 ───────────────────────────────────────────

def test_softmax_probabilities_sum_to_one():
    model = build_baseline()
    x = np.random.rand(8, *INPUT_SHAPE).astype(np.float32)
    probs = model.predict(x, verbose=0)
    sums = probs.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5), "Softmax probs should sum to 1 per sample"


# ── Compile ───────────────────────────────────────────────────────────────────

def test_compile_model():
    model = build_baseline()
    compiled = compile_model(model)
    assert compiled.optimizer is not None
    assert compiled.loss is not None


# ── get_optimizer ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["adam", "sgd", "rmsprop", "adamw"])
def test_get_optimizer_valid(name):
    opt = get_optimizer(name)
    assert opt is not None


def test_get_optimizer_invalid():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        get_optimizer("unknown_optimizer")


# ── Deeper model has more layers ──────────────────────────────────────────────

def test_deeper_has_more_layers():
    baseline = build_baseline()
    from model import build_deeper
    deeper = build_deeper()
    assert len(deeper.layers) > len(baseline.layers)


# ── Lightweight model parameter count ────────────────────────────────────────

def test_lightweight_under_100k_params():
    from model import build_lightweight
    model = build_lightweight()
    param_count = model.count_params()
    assert param_count < 100_000, f"Lightweight model has {param_count} params, expected <100K"
