# facial-expression-lab

Three small studies on facial expression recognition, built around one question: how much of a headline accuracy survives a proper evaluation?

**Status.** Code, tests, notebooks and a browser demo are written. A number appears in the results table only after I run the notebooks and commit the JSON files in `results/`. Rows that say "not measured" have not been run yet.

## What this repo contains

| Study | Question | Data | Notebook |
|---|---|---|---|
| 1. Landmark pipeline (from my 2023 thesis) | Do MediaPipe distance features, Mutual Information selection and a small classifier hold up under a subject-wise split? | UIBVFEDPlus-Light (public). CK+ and KDEF give numbers only. | `notebooks/03_study1_landmarks` |
| 2. Class imbalance (from my 2022 research placement) | How far does accuracy mislead when one class is about 92% of the rows, and what changes it? | Simulated data. The real data is not public. | `notebooks/02_study2_class_imbalance` |
| 3. A deployable recogniser | A small pretrained image model on FER2013 with FER+ labels, tested out of distribution, exported to ONNX, running in the browser. | FER2013 and FER+ labels. JAFFE if you supply it. | `notebooks/04_*`, `05_*` |

Also: `notebooks/01_evaluation_leakage_demo` shows, on simulated data, how a random split of frames and selecting features on all rows change results.

## Results

Every row comes from a file in `results/` or is an old number with its own label. Nothing is typed by hand (`scripts/make_results_table.py`).

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
| Study 3: FER2013 with FER+ labels | n/a | not measured | not measured | not measured |
| Study 3: out-of-distribution, UIBVFEDPlus-Light | n/a | not measured | not measured | not measured |
| Study 3: out-of-distribution, JAFFE (you supply the data) | n/a | not measured | not measured | not measured |
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
<!-- results:end -->

The first two rows are old numbers kept for comparison. `docs/evaluation.md` explains why the thesis number should not be read as a held-out result.

## Datasets

Several well-known facial-expression datasets cannot be shared or used freely. They need a request to the authors, limit use to non-commercial research, or restrict publishing the faces. This repository therefore **does not redistribute any dataset**. Every notebook anyone can run uses only the open datasets marked "Used here" or "Planned". Results on restricted datasets appear as numbers and charts only, and the code expects you to obtain that data yourself.

<!-- datasets:start -->
| Dataset | What it is | Access and terms | Licence status | In this repo |
|---|---|---|---|---|
| FER2013 | 35,887 grayscale 48x48 web images, 7 emotions, no subject IDs. | Kaggle competition data. A Kaggle account is needed and the competition rules must be accepted before download. [Provider page](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge). | unconfirmed | Planned (downloaded at run time) |
| FER+ labels | New labels for FER2013 from 10 annotators per image (label file only, no images). | Public GitHub repository (archived 13 June 2024) with a LICENSE.md file. [Provider page](https://github.com/microsoft/FERPlus). | confirmed | Planned (downloaded at run time) |
| JAFFE | 213 images, 10 posers, 7 expressions. | Zenodo record 14974867. Non-commercial scientific research only. Redistribution is not permitted (including via GitHub, Kaggle and Colab). Images must not be posted on the web. [Provider page](https://zenodo.org/records/14974867). | confirmed | Planned (you download it yourself) |
| UIBVFEDPlus-Light | 3,300 PNG images (1080x1920) of 100 virtual characters under 4 lighting setups, six emotion labels. | Download from the University of the Balearic Islands project page (33.7 GB). [Provider page](https://ugivia.uib.es/?p=1545). | partly confirmed | Planned (downloaded at run time) |
| CK+ (Extended Cohn-Kanade) | 593 video sequences, 123 subjects, lab-posed expressions. | Request from the authors. Non-commercial. A consent form limits publishing faces. [Provider page](https://www.jeffcohn.net/resources/). | partly confirmed | Private results only (planned) |
| KDEF | 4,900 pictures, 70 actors, 7 expressions, 5 angles (figures from the project brief; not checked). | Non-commercial research use. Request through the provider. [add provider link] | unconfirmed | Private results only (planned) |
| HSS Learning | Student expression logs from an e-learning platform, collected by a university lab. | Not public (student data). | partly confirmed | Private results only (planned) |
| AffectNet | Large in-the-wild expression dataset. | Request to the authors, research use. [add provider link] | unconfirmed | Not used |
| RAF-DB | Large in-the-wild expression dataset. | Request to the authors, research use. [add provider link] | unconfirmed | Not used |
| AFEW | Film clips with expression labels. | Academic, non-commercial licence; no redistribution. [add provider link] | unconfirmed | Not used |

Licence information last checked: 2026-11-14. Entries marked unconfirmed have not been read on the provider's page.
<!-- datasets:end -->

The table is generated from `datasets.yaml` by `scripts/make_dataset_table.py`. Terms can change, so read each provider's page before using a dataset. To reproduce the private-data results, request CK+ or KDEF from the providers and follow `docs/data.md` for the expected folder layout.

## Run it

No data or network needed for the simulated studies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/run_synthetic_studies.py     # about a minute; writes results/simulated_*.json
python scripts/make_results_table.py        # fills the table above from results/
```

For the real-data studies, follow `docs/reproduce.md`.

## Demo

Not published yet. When it is, the link goes here, with the measured browser latency and the model file's name. See `demo/README.md`.

## Evaluation rules used here

- Split by subject wherever subject IDs exist. FER2013 has none, so its results are labelled optimistic.
- Report macro F1 and per-class recall next to accuracy.
- Tune on a validation set made from training subjects. Read the test fold once.
- Label every number: measured (Colab T4), measured (host CPU), measured (local machine), measured (browser), estimated, simulated, reported, or not measured.

## How AI was used and verified

I used Claude (chat) for drafting code, suggesting tests and writing docs. I wrote the goal and the acceptance check first, read every change, ran the tests and made every commit myself. Every number in this repo comes from a run I did, and each one is labelled as measured, estimated or simulated.

What I checked: tests and CI, linting, dataset licences against the provider pages, and that no images, archives or private data are in the history. Where the tool was wrong, and what I changed, is in [`docs/ai-log.md`](docs/ai-log.md).

## Responsible use

Facial expression is not the same as inner emotion, and datasets differ in who appears in them. See `docs/responsible_use.md` and `docs/model_card.md`.

## Licence

Code: MIT (see `LICENSE`). Datasets and third-party models keep their own terms.
