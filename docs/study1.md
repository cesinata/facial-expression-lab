# Study 1: landmark features, subject-wise evaluation

## Pipeline
1. MediaPipe Face Landmarker returns 478 3D landmarks and 52 blendshape scores per image (`fel/landmarks.py`).
2. Landmarks are converted to pixel scale (x and y by image size, z on the x scale).
3. Features: pairwise Euclidean distances between 40 selected landmarks (780 pairs), divided by the distance between the outer eye corners, so face size and image size drop out. Optional extras: the same points centred on the nose tip, and the 52 blendshape scores.
4. Mutual Information selection keeps the top fraction of features. It is fitted on the fitting rows of each training fold only (`fel/features.py`, `fel/pipeline.py`).
5. Classifiers: majority-class baseline, logistic regression, a small scikit-learn MLP, and optionally a 1D CNN in PyTorch (`fel/models.py`).

## What differs from the thesis
- The thesis region and pair lists were not available when this was written. The 40 points here are a documented set of mesh indices (`REGION_POINTS` in `fel/landmarks.py`). Plot them on a face (notebook 03) before trusting any region-level reading. Region results here are not comparable to the thesis region results.
- Selection happens inside each training fold. The thesis computed Mutual Information on all rows before splitting.
- The 1D CNN reloads its best-validation checkpoint before predicting. The thesis saved that checkpoint but evaluated the last-epoch weights.
- Splits are subject-wise. The thesis used a random split of frames, and each CK+ sequence contributed its first frame and last three frames.
- Head rotation is not corrected. Distances use the image plane and the mesh depth estimate as given.

## Protocols reported side by side
For every model and feature set the study reports the subject-wise result, a random-row 5-fold result and the legacy thesis protocol (selection on all rows, random 70/15/15 split), on the same features. The gap between them is the point of the study.

## CK+ and KDEF (private)
Run on your own machine with `scripts/extract_features.py` and `scripts/run_study1.py`. Only sequences with a provider emotion label are used for CK+. Any hand-labelled sequences must be reported as a separate run. KDEF is split by actor, with the same actor's two sessions kept together. Only numbers are committed.

## JAFFE (you supply it)
JAFFE has 213 images of 10 people, so a 10-fold subject-wise split is leave-one-subject-out. Run `scripts/extract_features.py --source jaffe` and `scripts/run_study1.py --dataset jaffe --splits 10`. Only numbers are published. Results from 10 people are noisy, so read the bootstrap interval, not just the mean.

## Known limits
- UIBVFEDPlus-Light is rendered avatars, not photos of people. Results do not transfer to real faces by default.
- Class sets differ between datasets. Compare results only on shared classes.
