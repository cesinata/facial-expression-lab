# facial-expression-lab

What I built as an undergraduate, what was wrong with my evaluation, and how I am rebuilding it properly.

**Status: Phase 1 of 5 (evaluation harness).** The registry, split code, metrics and tests exist. No model has been trained in this repo yet, so every results cell below says "not measured". Results appear only after I run the notebooks and commit the JSON files in `results/`.

## What this repo contains

| Study | Question | Data | Status |
|---|---|---|---|
| 1. Landmark pipeline (from my 2023 thesis) | Do MediaPipe distance features, Mutual Information selection and a small 1D CNN hold up under a subject-wise split? | UIBVFEDPlus-Light (public). Private results on CK+ and KDEF as numbers only. | Planned |
| 2. Class imbalance (from my 2022 research placement) | How far does accuracy mislead when one class is 92% of the data, and what fixes it? | Synthetic data with the same imbalance (label: simulated). The real data stays private. | Planned |
| 3. A deployable recogniser | A small pretrained model on FER2013 with FER+ labels, tested out of distribution, exported to ONNX INT8, running in the browser. | FER2013 and FER+ labels (public). JAFFE if you supply it. | Planned |

## Results

| Result | Value | Label |
|---|---|---|
| Thesis, CK+, 595 features, tuned (random frame split, test set also used for selection) | accuracy 91.95%, macro F1 0.90 (as printed in the thesis notebook) | reported (thesis) |
| Same pipeline, subject-wise split | not measured | not measured |
| Class-imbalance study, k-NN / logistic regression / linear SVM | accuracy 0.914 / 0.922 / 0.921, macro F1 0.250 / 0.272 / 0.243 (research report and conference paper) | reported (paper) |
| Study 3 macro F1, latency | not measured | not measured |

The first row is an old number kept for comparison. The code and notebooks in this repo show why it should not be read as a held-out result. See `docs/evaluation.md`.

## Datasets

Several well-known facial-expression datasets cannot be shared or used freely. They need a request to the authors, limit use to non-commercial research, or restrict publishing the faces. This repository therefore **does not redistribute any dataset**. Every notebook that anyone can run uses only the open datasets marked "Used here" or "Planned". Results on restricted datasets appear as numbers and charts only, and the code expects you to obtain that data yourself.

<!-- datasets:start -->
| Dataset | What it is | Access and terms | Licence status | In this repo |
|---|---|---|---|---|
| FER2013 | 35,887 grayscale 48x48 web images, 7 emotions, no subject IDs. | Kaggle competition data. A Kaggle account is needed and the competition rules must be accepted before download. [Provider page](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge). | unconfirmed | Planned (downloaded at run time) |
| FER+ labels | New labels for FER2013 from 10 annotators per image (label file only, no images). | Public GitHub repository (archived 13 June 2024) with a LICENSE.md file. [Provider page](https://github.com/microsoft/FERPlus). | unconfirmed | Planned (downloaded at run time) |
| JAFFE | 213 images, 10 posers, 7 expressions. | Zenodo record 14974867. Non-commercial scientific research only. Redistribution is not permitted (including via GitHub, Kaggle and Colab). Images must not be posted on the web. [Provider page](https://zenodo.org/records/14974867). | confirmed | Planned (you download it yourself) |
| UIBVFEDPlus-Light | 3,300 PNG images (1080x1920) of 100 virtual characters under 4 lighting setups, six emotion labels. | Download from the University of the Balearic Islands project page (33.7 GB). [Provider page](https://ugivia.uib.es/?p=1545). | partly confirmed | Planned (downloaded at run time) |
| CK+ (Extended Cohn-Kanade) | 593 video sequences, 123 subjects, lab-posed expressions. | Request from the authors. Non-commercial. A consent form limits publishing faces. [Provider page](https://www.jeffcohn.net/resources/). | partly confirmed | Private results only (planned) |
| KDEF | 4,900 pictures, 70 actors, 7 expressions, 5 angles (figures from the project brief; not checked). | Non-commercial research use. Request through the provider. [add provider link] | unconfirmed | Private results only (planned) |
| HSS Learning | Student expression logs from an e-learning platform, collected by a university lab. | Not public (student data). | partly confirmed | Private results only (planned) |
| AffectNet | Large in-the-wild expression dataset. | Request to the authors, research use. [add provider link] | unconfirmed | Not used |
| RAF-DB | Large in-the-wild expression dataset. | Request to the authors, research use. [add provider link] | unconfirmed | Not used |
| AFEW | Film clips with expression labels. | Academic, non-commercial licence; no redistribution. [add provider link] | unconfirmed | Not used |

Licence information last checked: 2026-11-08. Entries marked unconfirmed have not been read on the provider's page.
<!-- datasets:end -->

The table above is generated from `datasets.yaml` by `scripts/make_dataset_table.py`. Terms can change, so read each provider's page before using a dataset. To reproduce the private-data results, request CK+ or KDEF from the providers and follow `docs/data.md` for the expected folder layout.

## Evaluation rules used here

- Split by subject wherever subject IDs exist. FER2013 has none, so its results are labelled optimistic.
- Report macro F1 and per-class recall next to accuracy.
- Tune on a validation set made from training subjects. Touch the test fold once.
- Label every number: measured (Colab T4), measured (host CPU), measured (local machine), measured (browser), estimated, simulated, reported, or not measured.

## Run the checks

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/make_dataset_table.py --check
python scripts/check_notebooks_registry.py
```

## How AI was used and verified

I used Claude (chat) for drafting code, suggesting tests and writing docs. I wrote the goal and the acceptance check first, read every change, ran the tests and made every commit myself. Every number in this repo comes from a run I did, and each one is labelled as measured, estimated or simulated.

What I checked: tests and CI, linting, dataset licences against the provider pages, and that no images, archives or private data are in the history. Where the tool was wrong, and what I changed, is in [`docs/ai-log.md`](docs/ai-log.md).

## Responsible use

Facial expression is not the same as inner emotion, and datasets carry demographic bias. See `docs/responsible_use.md` (to be completed in Phase 5).

## Licence

Code: MIT (see `LICENSE`). Datasets and third-party models keep their own terms.
