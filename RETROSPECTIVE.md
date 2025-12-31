# Project Retrospective

## What Went Well

1. **Modular refactoring** (Task 38) paid off immediately — isolating data, model, train,
   and evaluate logic made every subsequent task faster to implement and easier to test.

2. **Early stopping** (Task 22) saved significant compute time; many runs converged 30–40%
   earlier than the fixed epoch budget.

3. **Config centralization** (Task 40) eliminated hardcoded constants scattered across
   the notebook, making hyperparameter sweeps trivial.

## Challenges & Resolutions

1. **Non-determinism on GPU** — `TF_DETERMINISTIC_OPS=1` introduced a significant training
   slowdown (~3×) on the test GPU. Resolution: documented the flag but left it opt-in rather
   than mandatory for everyday runs.

2. **Label shape mismatch** after switching to softmax — switching from `sigmoid` +
   `binary_crossentropy` to `softmax` + `categorical_crossentropy` required one-hot encoding
   all labels. Fixed by adding `to_one_hot()` in `data.py` and updating all callers.

3. **Dependency conflicts** — `scipy 1.12` and `numpy 1.26` had a transient conflict on
   Python 3.11. Resolution: pinned exact versions in `requirements.txt` and tested in a
   clean virtual environment.

## Lessons Learned

- Start with reproducibility and config infrastructure before experimenting — it avoids
  having to retrofit seeds and constants later.
- Transfer learning (MobileNetV2, EfficientNetB0) requires resizing Fashion-MNIST to
  96×96 and converting to 3-channel; preprocessing must match the ImageNet statistics.
- Grad-CAM is most informative on the final Conv2D layer; earlier layers capture textures
  rather than semantic regions.

## 2025-12-31 4:13 PM — Task 49 (Final Report & Documentation)
**[RETROSPECTIVE.md]** Write 3 challenges and how they were resolved
