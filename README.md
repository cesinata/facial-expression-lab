# facial-expression-lab

Three small facial expression studies, a browser emoji demo (emoji mirror), and a corrected evaluation of my bachelor thesis results.

**Live demo:** https://cesinata.github.io/facial-expression-lab/ · **Learning notes:** [computer vision refresher map](docs/learning.md)

[Demo](#demo) · [Results](#results) · [Studies](#studies) · [Status](#status) · [Data](#data) · [Run it](#run-it) · [More](#more)

## Why this repo exists

It is 2026 and I am refreshing my computer vision skills. I combined my bachelor thesis project with a modern small-model study and a browser demo (emoji mirror) to practise evaluation, deployment and honest reporting end to end.

## What my bachelor thesis did

The thesis classified facial expressions on CK+ using MediaPipe landmark-distance features, mutual-information feature selection and a custom PyTorch CNN. It compared about 30 feature sets and found 2D distance features more informative than 3D ones, especially around the mouth, cheeks and chin. Reported accuracy rose from 86.5% to 91.95% after tuning (reported (thesis)).

The split was random at frame level, so frames from the same person could be in both training and test data. The 91.95% figure is therefore not a subject-independent result. The corrected protocol is in [`docs/evaluation.md`](docs/evaluation.md), and corrected numbers on the same kind of features are in the results below.

## Demo

<img src="https://github.com/cesinata/facial-expression-lab/releases/download/v1.0.0/demo-screenshot.png" alt="Demo page in Chrome: blurred webcam frame, the emoji mirror showing a smiling face, and probability bars for the eight classes" width="360">

Chrome on a MacBook, 2026-11-15, webcam frame blurred. The bars for this frame are close (disgust 27%, happiness 26%, contempt 25%), so the model is unsure; the percentages are not calibrated probabilities.

The page runs the FP32 `model.onnx` (6.12 MB) from release v1.0.0 in your browser; nothing is uploaded. The weakest classes are contempt, fear and neutral (recall 0.481, 0.516 and 0.599 on FER2013 PrivateTest, measured (host CPU)). More: [`demo/README.md`](demo/README.md), [`docs/release_notes.md`](docs/release_notes.md), [`docs/model_card.md`](docs/model_card.md).

## Results

Headline numbers, generated from `results/` with their labels:

<!-- results:start -->
| Result | Accuracy | Macro F1 | Label |
|---|---|---|---|
| Thesis, CK+, random frame split, test set also used for selection | 91.95% | 0.90 | reported (thesis) |
| Study 1, JAFFE, logistic regression on 2D distances: random split of rows | 0.864 ± 0.069 | 0.852 ± 0.087 | measured (local machine) |
| Study 1, same features: legacy thesis protocol | 0.844 | 0.853 | measured (local machine) |
| Study 1, same features: split by subject (leave-one-subject-out) | 0.573 ± 0.220 | 0.523 ± 0.253 | measured (local machine) |
| Study 2, one class near 92%: logistic regression, no fix | 0.919 ± 0.004 | 0.180 ± 0.012 | simulated |
| Study 2: linear SVM with class weights | 0.843 ± 0.028 | 0.234 ± 0.013 | simulated |
| Study 3, FER2013 PrivateTest: published ONNX FP32 model (optimistic: no subject IDs) | 0.705 | 0.571 | measured (host CPU) |
| Study 3: ONNX INT8 (measured failure, not published) | 0.047 | 0.037 | measured (host CPU) |
| Study 3 model on JAFFE, not seen in training (7 shared classes) | 0.380 | 0.314 | measured (host CPU) |

Browser latency of the published model: median 3.2 ms over 30 runs, Chrome (version not recorded), MacBook (macOS), 2026-11-15 (measured (browser)).
Not measured yet: Study 1: UIBVFEDPlus-Light; Study 3: out-of-distribution, UIBVFEDPlus-Light; Study 1: CK+ (private data, numbers only); Study 1: KDEF (private data, numbers only).
<!-- results:end -->

**More details:** every model, feature set and protocol in [`docs/results.md`](docs/results.md).

## Studies

| Study | Data | Status |
|---|---|---|
| 1. Landmark features from my 2023 thesis, evaluated by subject ([write-up](docs/study1.md)) | JAFFE. UIBVFEDPlus-Light, CK+ and KDEF planned. | Partly done |
| 2. Class imbalance from my 2022 research placement ([write-up](docs/study2.md)) | Simulated (the real data is not public) | Done |
| 3. Small pretrained image model, ONNX export and the emoji demo ([write-up](docs/study3.md)) | FER2013 with FER+ labels. JAFFE and UIBVFEDPlus-Light out of distribution. | Partly done |

Notebooks are in [`notebooks/`](notebooks/). `01_evaluation_leakage_demo` shows, on simulated data, how a random split of frames and selecting features on all rows change results.

## Status

| State | Items |
|---|---|
| Done | Study 2 and the leakage demos (simulated). Study 1 on JAFFE. Study 3 on FER2013 with FER+ labels, ONNX export and JAFFE test. Model card. CI. Demo live on GitHub Pages with release v1.0.0. One browser latency reading. Library and Actions versions checked (2026-11-15). Kaggle FER2013 terms: terms read; they do not address publishing trained weights (owner decision recorded in [`docs/decisions.md`](docs/decisions.md)). |
| In progress | Reviewing the AI log entries in [`docs/ai-log.md`](docs/ai-log.md) marked DRAFT. |
| Blocked | UIBVFEDPlus-Light: 33.7 GB download and my local storage is low, so I will run it when I have space (its rows stay "not measured"). CK+: no provider emotion labels in my copy. KDEF: the actor folders in my copy are empty. |
| To do | Try the demo in other browsers and on a phone. Record a 200-run browser benchmark with the "Measure model time" button. |

## Data

No dataset is redistributed or released here; the repo only links to the providers. Results on restricted datasets appear as numbers only.

<!-- datasets:start -->
| Dataset | Licence status | In this repo |
|---|---|---|
| [FER2013](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data) | partly confirmed | Used here (downloaded at run time) |
| [FER+ labels](https://github.com/microsoft/FERPlus) | confirmed | Used here (downloaded at run time) |
| [JAFFE](https://zenodo.org/records/14974867) | confirmed | Used here (you download it yourself) |

Planned, private or not used (7): UIBVFEDPlus-Light, CK+ (Extended Cohn-Kanade), KDEF, HSS Learning, AffectNet, RAF-DB, AFEW. Terms and status for all of them: [`docs/data.md`](docs/data.md).
<!-- datasets:end -->

FER2013 attribution: Kaggle, "Challenges in Representation Learning: Facial Expression Recognition Challenge" ([data](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data)), prepared by Pierre-Luc Carrier and Aaron Courville; paper: Goodfellow et al. (2013), "Challenges in Representation Learning: A report on three machine learning contests", [arXiv:1307.0414](https://arxiv.org/abs/1307.0414), DOI [10.48550/arXiv.1307.0414](https://doi.org/10.48550/arXiv.1307.0414). FER+ labels: [microsoft/FERPlus](https://github.com/microsoft/FERPlus) (MIT). The Kaggle terms do not say whether trained models may be published; publishing this one is my own decision.

## Run it

No data or network needed for the simulated studies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/run_synthetic_studies.py     # about a minute; writes results/simulated_*.json
python scripts/make_results_table.py        # fills the results tables from results/
```

For the real-data studies, follow [`docs/reproduce.md`](docs/reproduce.md).

## More

- **Evaluation rules:** split by subject where IDs exist, report macro F1 and per-class recall next to accuracy, read the test fold once, label every number. Details: [`docs/evaluation.md`](docs/evaluation.md).
- **How AI was used:** I used Claude (chat, and Claude Code as a coding agent) for drafting code, tests and docs. I wrote the goal and the acceptance check first, read every change, ran the tests and made every commit myself. Where the tool was wrong: [`docs/ai-log.md`](docs/ai-log.md).
- **Responsible use:** facial expression is not inner emotion, and the model is not for decisions about people. See [`docs/responsible_use.md`](docs/responsible_use.md).
- **Decisions and defaults:** [`docs/decisions.md`](docs/decisions.md).
- **Licence:** code under MIT (see `LICENSE`). Datasets and third-party models keep their own terms.
