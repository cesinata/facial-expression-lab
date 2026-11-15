# Release notes

Text for the GitHub release that carries `model.onnx`. Paste everything below the line into the release description.

---

## model.onnx: FP32 facial expression model for the browser demo

### What the file is

- `model.onnx`, FP32, 6.12 MB (6,117,108 bytes). This is the only model file in the release.
- MobileNetV3-small (timm `mobilenetv3_small_100.lamb_in1k`, ImageNet-pretrained), fine-tuned on the FER2013 Training split with FER+ majority-vote labels. 40-epoch run, best epoch 34, chosen by macro F1 on PublicTest before PrivateTest was read.
- Input `input`: float32, shape (batch, 1, 48, 48), a grey face crop scaled to 0..1. Normalisation and resizing to 96x96 happen inside the model.
- Output `logits`: 8 classes in this order: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt. Apply softmax yourself.
- Used by the browser demo: https://cesinata.github.io/facial-expression-lab/. The Pages workflow downloads this file from the latest release.

### PrivateTest results

FER2013 PrivateTest, 3,395 images, read once. Label: measured (host CPU), from `results/study3_ferplus.json`. FER2013 has no subject IDs, so the same person can appear in several splits and these numbers are optimistic.

| Model file | Accuracy | Macro F1 |
|---|---|---|
| ONNX FP32 (this file) | 0.705 | 0.571 |
| PyTorch FP32 (source model) | 0.705 | 0.571 |

Largest difference between PyTorch and ONNX FP32 logits: 2.1e-05.

Per-class recall: anger 0.756, contempt 0.481, disgust 0.609, fear 0.516, happiness 0.820, neutral 0.599, sadness 0.622, surprise 0.854. Contempt, fear and neutral are the weakest.

Calibration: expected calibration error 0.288 without temperature scaling, 0.232 with temperature 0.759 (fitted on PublicTest). The demo applies no temperature, so its percentages are not probabilities.

Single-sample latency on the Colab virtual machine's CPU: median 1.27 ms, 95th percentile 1.61 ms (measured (host CPU)). Browser latency: median 3.2 ms over 30 frames, Chrome on a MacBook, 2026-11-15 (measured (browser), one reading, `results/study3_browser_latency.json`).

Out of distribution: on JAFFE (not trained on, 7 shared classes), accuracy 0.380 and macro F1 0.314 (measured (host CPU), `results/study3_ood_jaffe.json`).

### INT8 caveat

An INT8 file was made by dynamic quantisation with onnxruntime and is **not** published. It was 1.71 MB but scored 0.047 accuracy and 0.037 macro F1 on PrivateTest, and its median latency on the same CPU was 11.42 ms against 1.27 ms for FP32 (measured (host CPU)). Why quantisation broke this model was not investigated.

### Not intended for

Judging how a person feels, assessing people (hiring, school, workplace, security), or any decision about a person. Facial expression is not inner emotion. The model has not been tested across skin tones, ages, head poses, lighting or camera quality. See `docs/model_card.md` and `docs/responsible_use.md` in the repository.

### Licence and data terms

- Code in this repository: MIT.
- Pretrained weights: ImageNet weights from timm (`mobilenetv3_small_100.lamb_in1k`), Apache-2.0 on the timm model card. The terms of the ImageNet-1k data have not been reviewed.
- Labels: FER+ (https://github.com/microsoft/FERPlus), MIT per the repository's LICENSE.md.
- Training images: FER2013 from Kaggle. The Data tab lists the licence as "Subject to Competition Rules" (read by the repo owner on 2026-11-15). The rules do not address whether models trained on the data may be published. Publishing this model is the repo owner's decision, recorded in `docs/decisions.md`. It is not a permission stated in the rules.
- The FER2013 dataset is not included in this release or in the repository. The repository only links to it.

### Attribution

- Data: Kaggle, "Challenges in Representation Learning: Facial Expression Recognition Challenge", https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data. Dataset prepared by Pierre-Luc Carrier and Aaron Courville.
- Paper: Goodfellow et al. (2013), "Challenges in Representation Learning: A report on three machine learning contests", https://arxiv.org/abs/1307.0414, DOI https://doi.org/10.48550/arXiv.1307.0414.
