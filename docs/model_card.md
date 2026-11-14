# Model card (draft)

No model has been published yet. Fill the "not measured" fields from `results/` after running Study 3.

| Field | Value |
|---|---|
| Model | MobileNetV3-small (timm), 8 FER+ classes, input 48x48 grey crop |
| Training data | FER2013 Training split with FER+ majority-vote labels |
| Pretrained weights | ImageNet weights from timm. Licence not yet checked. |
| Accuracy and macro F1 (PrivateTest) | not measured |
| Calibration (ECE before and after temperature scaling) | not measured |
| Out of distribution (UIBVFEDPlus-Light, JAFFE) | not measured |
| Browser latency | not measured |

## Intended use
Learning about evaluation of expression models. Demonstrating a small model running in a browser.

## Not intended for
Judging how a person feels, assessing people (hiring, school, workplace, security), or any decision about a person.

## Known limits
FER2013 has no subject IDs and its labels come from crowd annotation of web images. The model has not been tested across skin tones, ages, head poses, lighting or camera quality, and results should not be assumed to transfer.
