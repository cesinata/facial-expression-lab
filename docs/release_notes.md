# Release notes

Text for the GitHub release that carries `model.onnx`.

## Licence and data terms

- Code in this repository: MIT.
- Pretrained weights: ImageNet weights from timm (`mobilenetv3_small_100.lamb_in1k`), Apache-2.0 on the timm model card.
- Labels: FER+ (https://github.com/microsoft/FERPlus), MIT per the repository's LICENSE.md.
- Training images: FER2013 from Kaggle. The Data tab lists the licence as "Subject to Competition Rules" (read by the repo owner on 2026-11-15). The rules do not address whether models trained on the data may be published. Publishing this model is the repo owner's decision, recorded in `docs/decisions.md`. It is not a permission stated in the rules.
- The FER2013 dataset is not included in this release or in the repository. The repository only links to it.

## Attribution

- Data: Kaggle, "Challenges in Representation Learning: Facial Expression Recognition Challenge", https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data. Dataset prepared by Pierre-Luc Carrier and Aaron Courville.
- Paper: Goodfellow et al. (2013), "Challenges in Representation Learning: A report on three machine learning contests", https://arxiv.org/abs/1307.0414, DOI https://doi.org/10.48550/arXiv.1307.0414.
