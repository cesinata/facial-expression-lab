# All results

Every row comes from a file in `results/` or is an old number with its own label. Nothing is typed by hand: `scripts/make_results_table.py` writes this table, and CI fails if it is out of date. The README shows a short headline table drawn from the same files.

The first two rows are old numbers kept for comparison. [`evaluation.md`](evaluation.md) explains why the thesis number should not be read as a held-out result.

<!-- results:start -->
| Result | Data | Accuracy | Macro F1 | Label |
|---|---|---|---|---|
| Thesis, CK+, 595 features, tuned. Random frame split, test set also used for selection | thesis / report | 91.95% | 0.90 | reported (thesis) |
| Research report, student expression data, k-NN / logistic regression / linear SVM | thesis / report | 0.914 / 0.922 / 0.921 | 0.250 / 0.272 / 0.243 | reported (paper) |
| leakage demo: logreg | legacy (random 70/15/15 rows, selection on all rows) | synthetic | 0.636 ± 0.067 | 0.624 ± 0.069 | simulated |
| leakage demo: logreg | random 5-fold (rows) | synthetic | 0.577 ± 0.040 | 0.573 ± 0.039 | simulated |
| leakage demo: logreg | subject-wise 5-fold | synthetic | 0.489 ± 0.033 | 0.485 ± 0.030 | simulated |
| leakage demo: knn1 | legacy (random 70/15/15 rows, selection on all rows) | synthetic | 0.900 ± 0.034 | 0.895 ± 0.038 | simulated |
| leakage demo: knn1 | random 5-fold (rows) | synthetic | 0.873 ± 0.005 | 0.872 ± 0.006 | simulated |
| leakage demo: knn1 | subject-wise 5-fold | synthetic | 0.368 ± 0.066 | 0.369 ± 0.065 | simulated |
| leakage demo: mlp | legacy (random 70/15/15 rows, selection on all rows) | synthetic | 0.564 ± 0.134 | 0.553 ± 0.129 | simulated |
| leakage demo: mlp | random 5-fold (rows) | synthetic | 0.508 ± 0.057 | 0.498 ± 0.058 | simulated |
| leakage demo: mlp | subject-wise 5-fold | synthetic | 0.455 ± 0.066 | 0.448 ± 0.068 | simulated |
| selection leak demo: selected on all rows, then random 70/15/15 split | synthetic | 0.233 ± 0.073 | 0.220 ± 0.068 | simulated |
| selection leak demo: selected inside each training fold, random 4-fold | synthetic | 0.237 ± 0.015 | 0.228 ± 0.019 | simulated |
| class imbalance: knn9 / baseline | synthetic | 0.919 ± 0.001 | 0.138 ± 0.002 | simulated |
| class imbalance: knn9 / oversample | synthetic | 0.705 ± 0.015 | 0.165 ± 0.014 | simulated |
| class imbalance: logreg / baseline | synthetic | 0.919 ± 0.004 | 0.180 ± 0.012 | simulated |
| class imbalance: logreg / class_weight | synthetic | 0.394 ± 0.042 | 0.156 ± 0.012 | simulated |
| class imbalance: logreg / oversample | synthetic | 0.735 ± 0.039 | 0.229 ± 0.015 | simulated |
| class imbalance: linear_svc / baseline | synthetic | 0.920 ± 0.001 | 0.137 ± 0.000 | simulated |
| class imbalance: linear_svc / class_weight | synthetic | 0.843 ± 0.028 | 0.234 ± 0.013 | simulated |
| class imbalance: linear_svc / oversample | synthetic | 0.790 ± 0.042 | 0.231 ± 0.014 | simulated |
| Study 1: UIBVFEDPlus-Light | n/a | not measured | not measured | not measured |
| study3: PyTorch FP32 | ferplus | 0.705 | 0.571 | measured (host CPU) |
| study3: ONNX FP32 | ferplus | 0.705 | 0.571 | measured (host CPU) |
| study3: ONNX INT8 | ferplus | 0.047 | 0.037 | measured (host CPU) |
| Study 3: out-of-distribution, UIBVFEDPlus-Light | n/a | not measured | not measured | not measured |
| study3 out-of-distribution: restricted to 7 shared classes | jaffe | 0.380 | 0.314 | measured (host CPU) |
| study3_browser_latency: ONNX FP32 in the browser, median 3.2 ms over 30 runs, Chrome (version not recorded), MacBook (macOS), 2026-11-15 | ferplus | n/a | n/a | measured (browser) |
| Study 1: CK+ (private data, numbers only) | n/a | not measured | not measured | not measured |
| Study 1: KDEF (private data, numbers only) | n/a | not measured | not measured | not measured |
| study1_jaffe: distances 2D | majority | subject-wise 10-fold | jaffe | 0.145 ± 0.011 | 0.036 ± 0.002 | measured (local machine) |
| study1_jaffe: distances 2D | logreg | subject-wise 10-fold | jaffe | 0.573 ± 0.220 | 0.523 ± 0.253 | measured (local machine) |
| study1_jaffe: distances 2D | logreg | random 10-fold (rows) | jaffe | 0.864 ± 0.069 | 0.852 ± 0.087 | measured (local machine) |
| study1_jaffe: distances 2D | logreg | legacy thesis protocol | jaffe | 0.844 | 0.853 | measured (local machine) |
| study1_jaffe: distances 2D | mlp | subject-wise 10-fold | jaffe | 0.532 ± 0.184 | 0.471 ± 0.191 | measured (local machine) |
| study1_jaffe: distances 2D | mlp | random 10-fold (rows) | jaffe | 0.727 ± 0.117 | 0.715 ± 0.123 | measured (local machine) |
| study1_jaffe: distances 2D | mlp | legacy thesis protocol | jaffe | 0.562 | 0.556 | measured (local machine) |
| study1_jaffe: distances 3D | majority | subject-wise 10-fold | jaffe | 0.145 ± 0.011 | 0.036 ± 0.002 | measured (local machine) |
| study1_jaffe: distances 3D | logreg | subject-wise 10-fold | jaffe | 0.528 ± 0.171 | 0.459 ± 0.201 | measured (local machine) |
| study1_jaffe: distances 3D | logreg | random 10-fold (rows) | jaffe | 0.864 ± 0.065 | 0.860 ± 0.067 | measured (local machine) |
| study1_jaffe: distances 3D | logreg | legacy thesis protocol | jaffe | 0.906 | 0.910 | measured (local machine) |
| study1_jaffe: distances 3D | mlp | subject-wise 10-fold | jaffe | 0.536 ± 0.181 | 0.465 ± 0.204 | measured (local machine) |
| study1_jaffe: distances 3D | mlp | random 10-fold (rows) | jaffe | 0.718 ± 0.152 | 0.705 ± 0.171 | measured (local machine) |
| study1_jaffe: distances 3D | mlp | legacy thesis protocol | jaffe | 0.594 | 0.584 | measured (local machine) |
| study1_jaffe: distances 3D + blendshapes | majority | subject-wise 10-fold | jaffe | 0.145 ± 0.011 | 0.036 ± 0.002 | measured (local machine) |
| study1_jaffe: distances 3D + blendshapes | logreg | subject-wise 10-fold | jaffe | 0.569 ± 0.205 | 0.509 ± 0.244 | measured (local machine) |
| study1_jaffe: distances 3D + blendshapes | logreg | random 10-fold (rows) | jaffe | 0.850 ± 0.070 | 0.843 ± 0.074 | measured (local machine) |
| study1_jaffe: distances 3D + blendshapes | logreg | legacy thesis protocol | jaffe | 0.875 | 0.880 | measured (local machine) |
| study1_jaffe: distances 3D + blendshapes | mlp | subject-wise 10-fold | jaffe | 0.554 ± 0.168 | 0.500 ± 0.176 | measured (local machine) |
| study1_jaffe: distances 3D + blendshapes | mlp | random 10-fold (rows) | jaffe | 0.653 ± 0.170 | 0.642 ± 0.177 | measured (local machine) |
| study1_jaffe: distances 3D + blendshapes | mlp | legacy thesis protocol | jaffe | 0.688 | 0.715 | measured (local machine) |
| study1_jaffe: distances 3D + points + blendshapes | majority | subject-wise 10-fold | jaffe | 0.145 ± 0.011 | 0.036 ± 0.002 | measured (local machine) |
| study1_jaffe: distances 3D + points + blendshapes | logreg | subject-wise 10-fold | jaffe | 0.555 ± 0.189 | 0.502 ± 0.219 | measured (local machine) |
| study1_jaffe: distances 3D + points + blendshapes | logreg | random 10-fold (rows) | jaffe | 0.868 ± 0.063 | 0.864 ± 0.066 | measured (local machine) |
| study1_jaffe: distances 3D + points + blendshapes | logreg | legacy thesis protocol | jaffe | 0.844 | 0.853 | measured (local machine) |
| study1_jaffe: distances 3D + points + blendshapes | mlp | subject-wise 10-fold | jaffe | 0.519 ± 0.113 | 0.458 ± 0.124 | measured (local machine) |
| study1_jaffe: distances 3D + points + blendshapes | mlp | random 10-fold (rows) | jaffe | 0.741 ± 0.162 | 0.715 ± 0.185 | measured (local machine) |
| study1_jaffe: distances 3D + points + blendshapes | mlp | legacy thesis protocol | jaffe | 0.812 | 0.806 | measured (local machine) |

Note: Study 3 uses FER2013, which has no subject IDs, so the same person can be in several splits and its numbers are optimistic.
Note: The ONNX INT8 row is a measured failure: dynamic quantisation damaged the model, so it is not published and the demo uses the FP32 file.
Note: The browser latency row has no accuracy: it is the model time in the demo page, median 3.2 ms over 30 runs, Chrome (version not recorded), MacBook (macOS), 2026-11-15 (measured (browser)).
<!-- results:end -->

How each study was run: [`study1.md`](study1.md), [`study2.md`](study2.md), [`study3.md`](study3.md). Model details: [`model_card.md`](model_card.md).
