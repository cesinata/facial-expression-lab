# Evaluation protocol

Applies to every study in this repo.

## Rules

1. Split by subject (or by character, actor or user) wherever IDs exist. Use grouped k-fold, or leave-one-subject-out for small sets.
2. Keep three roles apart: fitting (training rows), tuning (a validation split taken from the training subjects), and reporting (the test fold, used once). Feature selection, tuning and early stopping happen without the test fold.
3. Report accuracy, macro F1 and per-class recall together, with the classes that are never predicted listed. Add a bootstrap interval that resamples subjects, not rows.
4. Datasets without subject IDs (FER2013) cannot be split by person. Label those results optimistic.
5. Fix seeds, log library versions, and save metrics as JSON in `results/` with a provenance label (see `fel/results.py`).

## Why

A random split of frames puts near-identical frames from one video in both train and test. `tests/test_splits.py` demonstrates this on simulated data. Label: simulated.

## The original thesis protocol (what is being corrected)

Facts read from the thesis notebooks on 2026-11-08 (CK+, 8 classes, 348 test rows):

- Split: `train_test_split(..., test_size=0.3, random_state=42)` then `test_size=0.5`, so 70/15/15. No stratification, no groups. The thesis text says 60/20/20.
- The reported 91.95% is the test accuracy (320 of 348 rows) of the last-epoch weights. The best-validation checkpoint was saved but not reloaded.
- The same test set was used to rank 32 feature sets and to choose hyperparameters.
- Mutual Information was computed on all rows before the split.
- Each sequence contributes its first frame and last three frames.

These are reported as "reported (thesis)" and are not reused as results.

## Where old and corrected numbers appear
The README table lists the thesis result with the label "reported (thesis)". The corrected result for the same pipeline comes from `scripts/run_study1.py`, which also reports the random-row and legacy protocols on the same features. For CK+ and KDEF the corrected numbers come from a run on the owner's machine and appear in `results/private/` as numbers only.
