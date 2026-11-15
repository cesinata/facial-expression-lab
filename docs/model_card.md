# Model card

Last updated: 2026-11-15

Every number below comes from `results/study3_ferplus.json` and `results/study3_ood_jaffe.json`, and `tests/test_model_card.py` fails if the card and those files disagree. All were run once, on the CPU of a Colab virtual machine (label: measured (host CPU)), except where a row says otherwise.

## Model

| Field | Value |
|---|---|
| Model | MobileNetV3-small (timm `mobilenetv3_small_100.lamb_in1k`), 8 FER+ classes, input 48x48 grey crop |
| Training data | FER2013 Training split with FER+ majority-vote labels |
| Training run | 40 epochs, best epoch 34 chosen by macro F1 on PublicTest (validation macro F1 0.590). An earlier 15-epoch run reached 0.515; the 40-epoch model was chosen on PublicTest, before PrivateTest was read. Both numbers are from the training notebook output, not from a result file. |
| Pretrained weights | ImageNet weights from timm, Apache-2.0 on the timm model card. The terms of the ImageNet-1k data have not been reviewed. |
| Published file | `model.onnx` (FP32), 6.12 MB. Not in git. It is published only as a GitHub release attachment, and that release has not been created yet. |

## Data and attribution

- Data: Kaggle, "Challenges in Representation Learning: Facial Expression Recognition Challenge", https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data. Dataset prepared by Pierre-Luc Carrier and Aaron Courville.
- Paper: Goodfellow et al. (2013), "Challenges in Representation Learning: A report on three machine learning contests", https://arxiv.org/abs/1307.0414, DOI https://doi.org/10.48550/arXiv.1307.0414.
- Labels: FER+ (https://github.com/microsoft/FERPlus), MIT per the repository's LICENSE.md.

Terms: the Kaggle Data tab lists the licence as "Subject to Competition Rules" (read by the repo owner on 2026-11-15). The rules do not address whether models trained on the data may be published. Publishing this FP32 model is the repo owner's decision (see `docs/decisions.md`), not a permission stated in the rules. The dataset itself is never committed or released; the repo only links to it.

## Results

PrivateTest, 3,395 images, read once. FER2013 has no subject IDs, so the same person can be in several splits and these numbers are optimistic.

| Model file | Accuracy | Macro F1 |
|---|---|---|
| PyTorch FP32 | 0.705 | 0.571 |
| ONNX FP32 | 0.705 | 0.571 |
| ONNX INT8 (not published) | 0.047 | 0.037 |

The INT8 row is a measured failure. Dynamic quantisation damaged the model, so only the FP32 file is published. The largest difference between PyTorch and ONNX FP32 logits is 2.1e-05.

Per-class recall (all eight classes were predicted at least once):

| Class | Recall |
|---|---|
| anger | 0.756 |
| contempt | 0.481 |
| disgust | 0.609 |
| fear | 0.516 |
| happiness | 0.820 |
| neutral | 0.599 |
| sadness | 0.622 |
| surprise | 0.854 |

## Calibration

Expected calibration error on PrivateTest: 0.288 before temperature scaling, 0.232 after (temperature 0.759, fitted on PublicTest only). The model is still poorly calibrated after scaling. Its confidence scores should not be read as probabilities.

## Out of distribution

| Data | Accuracy | Macro F1 | Chance |
|---|---|---|---|
| JAFFE (private, numbers only), restricted to the 7 classes it shares with FER+ | 0.380 | 0.314 | about 0.143 |
| UIBVFEDPlus-Light | not measured | not measured | not measured |

JAFFE has 10 subjects, so this figure is noisy. It is well above chance and far below the in-distribution score. Recall on JAFFE was 0.000 for anger, disgust and fear (`results/study3_ood_jaffe.json`), so the overall figure comes from happiness, sadness, surprise and neutral.

## Speed

Single sample, one call at a time, on the Colab virtual machine's CPU (measured (host CPU)). Browser latency is a different machine and runtime and is not measured yet.

| File | Size | Median | 95th percentile |
|---|---|---|---|
| ONNX FP32 | 6.12 MB | 1.27 ms | 1.61 ms |
| ONNX INT8 (broken) | 1.71 MB | 11.42 ms | 15.15 ms |

| Field | Value |
|---|---|
| Browser latency | not measured |

## Intended use
Learning about evaluation of expression models. Demonstrating a small model running in a browser.

## Not intended for
Judging how a person feels, assessing people (hiring, school, workplace, security), or any decision about a person.

## Known limits
FER2013 has no subject IDs and its labels come from crowd annotation of web images. Contempt, fear and neutral are the weakest classes here. The model has not been tested across skin tones, ages, head poses, lighting or camera quality, and results should not be assumed to transfer. The FER2013 images keep their Kaggle terms ("Subject to Competition Rules"). Those rules do not address publishing trained weights; publishing this model is an owner decision, recorded in `docs/decisions.md`.
