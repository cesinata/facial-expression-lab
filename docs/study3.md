# Study 3: a small image model, evaluated and exported

## Data and labels
FER2013 images (48x48 grey) with FER+ majority-vote labels. Rows are dropped when "unknown" plus "not a face" votes outnumber the top emotion, when the top emotion is tied, or when it has no votes (`fel/fer.py`). The FER2013 `Usage` column gives Training, PublicTest and PrivateTest.

## Protocol
- Train on Training. Choose the checkpoint by macro F1 on PublicTest.
- Fit the calibration temperature on PublicTest only.
- Read PrivateTest once (`scripts/study3.py evaluate` refuses a second run without `--allow-repeat`).
- FER2013 has no subject IDs, so the same person can appear in several splits. Treat its results as optimistic.

## Model
A pretrained MobileNetV3-small from `timm`, wrapped so the ONNX file takes a 48x48 grey crop in 0..1 and does normalisation and upsampling inside. Check the pretrained weights' licence on the timm model card before publishing anything derived from them.

## Out of distribution
Face crops from UIBVFEDPlus-Light (and JAFFE, if you supply it) are classified without training on them. The prediction is the highest score among the classes the dataset shares with FER+. Class sets differ, so contempt is never scored on UIBVFEDPlus-Light.

## Export and speed
PyTorch FP32, ONNX FP32 and, if onnxruntime can quantise it, ONNX INT8 are each evaluated on PrivateTest. The script also records the largest difference between PyTorch and ONNX logits, file sizes, and single-sample latency on the machine that ran it. Latency from one machine says little about another. Evaluation and latency run on the CPU of whatever machine runs `evaluate` (in Colab, its virtual machine), so the label is "measured (host CPU)". Training is the only GPU step. INT8 may not be faster or smaller for convolution-heavy models, and the browser runtime may not support every INT8 operator, so the browser demo should use the FP32 file unless INT8 has been run there.
