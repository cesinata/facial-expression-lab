# Computer vision refresher map

A map of what this repo taught me, topic by topic, with links to the code and results that show it. Every number is copied from a file in `results/` and carries its label. Items marked [confirm] are things I still need to check myself.

## Evaluation and leakage

How data is split decides the headline number more than the model does. On simulated sequences, 1-nearest-neighbour scores 0.873 accuracy with a random 5-fold split of rows and 0.368 with a subject-wise 5-fold split (simulated, [`results/simulated_leakage_demo.json`](../results/simulated_leakage_demo.json)). On real JAFFE data the same gap appears: logistic regression on 2D distances scores 0.864 with random 10-fold rows and 0.573 with leave-one-subject-out (measured (local machine), [`results/study1_jaffe.json`](../results/study1_jaffe.json)). The feature-selection leak demo did not show a leak on its pure-noise data (0.233 when selected on all rows, 0.237 when selected inside folds, chance 0.25, simulated, [`results/simulated_selection_leak.json`](../results/simulated_selection_leak.json)), so that demo needs a different design before it can teach anything. Class imbalance is a second trap: with one class near 92%, every baseline model reaches about 0.92 accuracy while macro F1 stays between 0.137 and 0.180 (simulated, [`results/simulated_imbalance_study.json`](../results/simulated_imbalance_study.json)).

Code: [`src/fel/splits.py`](../src/fel/splits.py), [`src/fel/pipeline.py`](../src/fel/pipeline.py), [`src/fel/metrics.py`](../src/fel/metrics.py), [`src/fel/imbalance.py`](../src/fel/imbalance.py), [`tests/test_splits.py`](../tests/test_splits.py). Protocol: [`docs/evaluation.md`](evaluation.md).

## Classical landmark features

MediaPipe Face Landmarker gives 478 3D landmarks and 52 blendshape scores per face. Study 1 turns 40 of the landmarks into 780 pairwise distances divided by the distance between the outer eye corners, then keeps the top 75% by Mutual Information, fitted inside each training fold only. On JAFFE with leave-one-subject-out, logistic regression scores 0.573 on 2D distances and 0.528 on 3D distances (measured (local machine), [`results/study1_jaffe.json`](../results/study1_jaffe.json)), but the spread across subjects (about ±0.2) is larger than that gap, so 10 people cannot settle whether 2D beats 3D. The 40 points are not the thesis point lists, so region-level readings need the visual check in notebook 03 first [confirm].

Code: [`src/fel/landmarks.py`](../src/fel/landmarks.py), [`src/fel/features.py`](../src/fel/features.py), [`scripts/extract_features.py`](../scripts/extract_features.py), [`scripts/run_study1.py`](../scripts/run_study1.py). Write-up: [`docs/study1.md`](study1.md).

## A small CNN and transfer learning

There are two neural models here. The first is a 1D CNN for landmark features that re-implements the thesis block structure and fixes its checkpoint bug by reloading the best-validation weights before predicting ([`src/fel/models.py`](../src/fel/models.py)); it has tests but has not been run on any real dataset yet (not measured). The second is transfer learning: an ImageNet-pretrained MobileNetV3-small from timm, fine-tuned on FER2013 with FER+ labels, with all layers trained, class-weighted loss, label smoothing 0.1, flips and small shifts ([`src/fel/train_fer.py`](../src/fel/train_fer.py)). The 48x48 grey crop is normalised, upsampled to 96x96 and repeated to three channels inside the model, so the exported file needs no separate preprocessing. It scores 0.705 accuracy and 0.571 macro F1 on PrivateTest (measured (host CPU), [`results/study3_ferplus.json`](../results/study3_ferplus.json)); FER2013 has no subject IDs, so this is optimistic. The published model is the 40-epoch run, while notebook 04 still shows `--epochs 15` [confirm the exact training command].

Write-up: [`docs/study3.md`](study3.md), [`docs/model_card.md`](model_card.md).

## Calibration

A softmax score is not a probability unless it is calibrated. Expected calibration error on PrivateTest was 0.288 before temperature scaling and 0.232 after, with the temperature (0.759) fitted on PublicTest only (measured (host CPU), [`results/study3_ferplus.json`](../results/study3_ferplus.json)). A temperature below 1 makes the probabilities sharper, so on PublicTest the model was less confident than its accuracy justified. Label smoothing in training may be one cause, but that was not tested [confirm]. Even after scaling, the error is large, so the demo's percentages should not be read as probabilities.

Code: [`src/fel/calibration.py`](../src/fel/calibration.py), `expected_calibration_error` in [`src/fel/metrics.py`](../src/fel/metrics.py), [`tests/test_calibration.py`](../tests/test_calibration.py).

## ONNX export and quantisation

The PyTorch model is exported to ONNX with a dynamic batch axis, and the largest difference between PyTorch and ONNX FP32 logits on PrivateTest is 2.1e-05, with identical accuracy (0.705) and macro F1 (0.571) (measured (host CPU), [`results/study3_ferplus.json`](../results/study3_ferplus.json)). Dynamic INT8 quantisation is the measured lesson: it cut the file from 6.12 MB to 1.71 MB but dropped accuracy to 0.047 and macro F1 to 0.037, and median single-sample latency rose from 1.27 ms to 11.42 ms on the same CPU (measured (host CPU)). A smaller file is not automatically a faster or usable one, so every exported variant has to be evaluated, not assumed. Why INT8 broke this model was not investigated [confirm]. Only the FP32 file is published.

Code: `export_onnx` and `quantize_int8` in [`src/fel/train_fer.py`](../src/fel/train_fer.py), [`src/fel/latency.py`](../src/fel/latency.py), [`scripts/study3.py`](../scripts/study3.py).

## Browser deployment

The demo is a static page: webcam, MediaPipe Face Landmarker for the face box (10% margin, kept equal to the Python code by a test), a 48x48 grey crop drawn with the browser canvas, then the ONNX model in ONNX Runtime Web on a single wasm thread ([`demo/app.js`](../demo/app.js)). The emoji mirror shows the class with the highest smoothed probability; the smoothing default was not tuned. Nothing leaves the browser. The canvas resize differs from the Pillow resize used in Python, and the size of that difference has not been measured. The page is live on GitHub Pages; one browser reading gave a median model time of 3.2 ms over 30 frames in Chrome on a MacBook (measured (browser), [`results/study3_browser_latency.json`](../results/study3_browser_latency.json)), against 1.27 ms on the Colab CPU (measured (host CPU)). The two numbers come from different machines and runtimes, so the gap says little on its own.

Tests: [`tests/test_demo.py`](../tests/test_demo.py). Notes: [`demo/README.md`](../demo/README.md).

## Responsible use

The labels describe posed or crowd-annotated expressions, not verified feelings, so the model predicts a label for a pattern of pixels. Results do not transfer by default: on JAFFE, not seen in training, accuracy was 0.380 against 0.705 in distribution, and recall was 0.000 for anger, disgust and fear (measured (host CPU), [`results/study3_ood_jaffe.json`](../results/study3_ood_jaffe.json)). The model has not been tested across skin tones, ages, head poses, lighting or cameras. Dataset terms are part of the work too: JAFFE must never be redistributed, and publishing a model trained on FER2013 is an owner decision because the Kaggle rules do not address it ([`docs/decisions.md`](decisions.md)).

More: [`docs/responsible_use.md`](responsible_use.md), [`docs/model_card.md`](model_card.md), [`docs/data.md`](data.md).
